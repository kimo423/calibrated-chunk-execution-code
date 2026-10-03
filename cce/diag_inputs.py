#!/usr/bin/env python
"""T2.3 input-consistency diagnostic: is the frame/proprio fed at evaluation time
the same thing the training pipeline fed?

Three independent comparisons, all numeric:

1. *Preprocessing path*. One demonstration frame is pushed through
   (a) the training transform inside `InfiniteDataReader` (Resize 224 bicubic ->
   ToTensor -> ImageNet Normalize) and (b) the inference path
   `XVLAProcessor.encode_image` (CLIPImageProcessor). Reports max/mean abs
   difference of the resulting tensors. A large difference here would mean the
   policy sees a different image statistic at test time than during training.

2. *Rendered frame*. A fresh evaluation env is reset at a **training** seed and
   its first `base_camera` frame is compared pixel-wise against frame 0 of the
   demonstration recorded at that same seed. Same seed, same scene, so a large
   difference means the evaluation env renders differently (camera, resolution,
   channel order, backend).

3. *Proprio*. The evaluation-time `tcp_root_pose` at reset is compared against
   the demonstration's `tcp_root_pose[0]` and against `abs_trajectory[0]`
   (== the training `proprio`) produced by the patched reader, element by element.

Side-by-side PNGs are written to `data/cce/diag/`.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

import xvla_bridge as XB

PATCH = Path(__file__).resolve().parent / "xvla_patch"


def to_u8(t: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation for visual inspection."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (t.detach().cpu().float() * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def side_by_side(imgs, labels, path: Path, gap: int = 8) -> None:
    from PIL import Image, ImageDraw

    h = max(i.shape[0] for i in imgs)
    w = sum(i.shape[1] for i in imgs) + gap * (len(imgs) - 1)
    canvas = np.full((h + 16, w, 3), 255, np.uint8)
    x = 0
    for im in imgs:
        canvas[16:16 + im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    out = Image.fromarray(canvas)
    d = ImageDraw.Draw(out)
    x = 0
    for im, lab in zip(imgs, labels):
        d.text((x + 2, 2), lab, fill=(0, 0, 0))
        x += im.shape[1] + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def main() -> None:
    ap = argparse.ArgumentParser("CCE input-consistency diagnostic")
    ap.add_argument("--uid", default="panda", choices=["panda", "xarm6_robotiq"])
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--seed", type=int, default=20260101, help="a TRAINING seed with a demo on disk")
    ap.add_argument("--demo-dir", default=None)
    ap.add_argument("--metas", default=None)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--sim-backend", default="physx_cpu")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    demo_dir = Path(args.demo_dir or (_bootstrap.DATA / "demos"))
    out_dir = Path(args.out_dir or (_bootstrap.DATA / "diag"))
    out_dir.mkdir(parents=True, exist_ok=True)
    h5p = demo_dir / f"{args.uid}_{args.task}_seed{args.seed}.h5"
    if not h5p.exists():
        raise SystemExit(f"no demo at {h5p}")

    rep: dict = {"uid": args.uid, "task": args.task, "seed": args.seed, "demo": str(h5p)}

    with h5py.File(h5p, "r") as f:
        demo_rgb = np.asarray(f["rgb"][args.frame], np.uint8)
        demo_pose = np.asarray(f["tcp_root_pose"][()], np.float64)
        demo_closed = np.asarray(f["grip_closed"][()], np.float64).reshape(-1)
        demo_T = int(f["rgb"].shape[0])
        ctrl_freq = float(f.attrs.get("control_freq", 20.0))

    rep["demo_meta"] = {"T": demo_T, "control_freq": ctrl_freq,
                        "rgb_shape": list(demo_rgb.shape), "rgb_dtype": "uint8",
                        "rgb_channel_means": [round(float(demo_rgb[..., c].mean()), 2) for c in range(3)]}

    # ---------------------------------------------------------------- 1. transforms
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    train_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: x),            # ColorJitter is off when training=False
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
    ])
    pil = Image.fromarray(demo_rgb)
    a = train_tf(pil)                                          # [3,224,224]

    processor = _load_processor()
    b = processor.encode_image([pil])["image_input"][0, 0]     # [3,224,224]

    d = (a - b).abs()
    rep["preprocess_train_vs_infer"] = {
        "train_shape": list(a.shape), "infer_shape": list(b.shape),
        "max_abs_diff": round(float(d.max()), 6),
        "mean_abs_diff": round(float(d.mean()), 6),
        "train_mean": round(float(a.mean()), 5), "infer_mean": round(float(b.mean()), 5),
        "train_std": round(float(a.std()), 5), "infer_std": round(float(b.std()), 5),
        "train_per_channel_mean": [round(float(a[c].mean()), 5) for c in range(3)],
        "infer_per_channel_mean": [round(float(b[c].mean()), 5) for c in range(3)],
        "note": "both should be Resize(224,bicubic)+ImageNet-normalise; a large gap means "
                "the policy sees different statistics at test time",
    }
    side_by_side([to_u8(a), to_u8(b), np.abs(to_u8(a).astype(int) - to_u8(b).astype(int)).astype(np.uint8) * 8],
                 ["train-tf", "processor", "|diff| x8"],
                 out_dir / f"{args.uid}_seed{args.seed}_preprocess.png")

    # ---------------------------------------------------------------- 2. rendered frame
    from eval_xvla import make_eval_env, obs_rgb

    env = make_eval_env(args.task, args.uid, args.img, 200, args.sim_backend)
    try:
        obs, _ = env.reset(seed=args.seed)
        eval_rgb = obs_rgb(obs, "base_camera")
        eval_pose = XB.tcp_root_pose(env.unwrapped)
    finally:
        env.close()

    diff = np.abs(eval_rgb.astype(np.int32) - demo_rgb.astype(np.int32))
    rep["rendered_frame_eval_vs_demo"] = {
        "eval_shape": list(eval_rgb.shape),
        "demo_shape": list(demo_rgb.shape),
        "max_abs_diff_u8": int(diff.max()),
        "mean_abs_diff_u8": round(float(diff.mean()), 4),
        "frac_pixels_diff_gt_8": round(float((diff.max(axis=-1) > 8).mean()), 5),
        "eval_channel_means": [round(float(eval_rgb[..., c].mean()), 2) for c in range(3)],
        "demo_channel_means": [round(float(demo_rgb[..., c].mean()), 2) for c in range(3)],
        "bgr_swap_test_mean_abs_diff_u8": round(
            float(np.abs(eval_rgb[..., ::-1].astype(np.int32) - demo_rgb.astype(np.int32)).mean()), 4),
        "note": "same seed, same scene: a near-zero diff means camera/resolution/channel "
                "order/backend all match. The BGR-swap figure being *smaller* would indicate "
                "a channel-order bug.",
    }
    side_by_side([demo_rgb, eval_rgb, np.clip(diff * 8, 0, 255).astype(np.uint8)],
                 ["demo frame0", "eval reset", "|diff| x8"],
                 out_dir / f"{args.uid}_seed{args.seed}_frame.png")

    # ---------------------------------------------------------------- 3. proprio
    XB.add_xvla_to_path()
    if str(PATCH) not in sys.path:
        sys.path.insert(0, str(PATCH))
    from datasets.dataset import InfiniteDataReader

    metas = Path(args.metas or (_bootstrap.DATA / "metas" / f"{XB.DATASET_NAMES[args.uid]}.json"))
    reader = InfiniteDataReader(str(metas), num_actions=30, training=False, action_mode="ee6d")
    s = next(iter(reader))
    train_proprio = s["proprio"].numpy().astype(np.float64)
    train_action0 = s["action"][0].numpy().astype(np.float64)
    train_action29 = s["action"][29].numpy().astype(np.float64)

    demo_p0 = XB.pose7_to_ee6d20(demo_pose[0], demo_closed[0]).astype(np.float64)
    eval_p0 = XB.pose7_to_ee6d20(eval_pose, 0.0).astype(np.float64)

    rep["proprio"] = {
        "eval_reset_tcp_root_pose_xyzwxyz": [round(float(x), 6) for x in eval_pose],
        "demo_frame0_tcp_root_pose_xyzwxyz": [round(float(x), 6) for x in demo_pose[0]],
        "eval_vs_demo_xyz_abs_diff_m": [round(float(abs(eval_pose[i] - demo_pose[0][i])), 6) for i in range(3)],
        "eval_vs_demo_quat_abs_diff": [round(float(abs(eval_pose[3 + i] - demo_pose[0][3 + i])), 6)
                                       for i in range(4)],
        "quat_norm_eval": round(float(np.linalg.norm(eval_pose[3:])), 8),
        "quat_convention": "wxyz (SAPIEN); w component listed first",
        "eval_ee6d20_first10": [round(float(x), 6) for x in eval_p0[:10]],
        "demo_ee6d20_first10": [round(float(x), 6) for x in demo_p0[:10]],
        "training_proprio_first10 (reader, episode 0 first sample)":
            [round(float(x), 6) for x in train_proprio[:10]],
        "training_action_grid1_first10": [round(float(x), 6) for x in train_action0[:10]],
        "training_action_grid30_first10": [round(float(x), 6) for x in train_action29[:10]],
        "training_second_arm_all_zero": bool(np.abs(train_proprio[10:]).max() < 1e-9),
        "chunk_xyz_span_m": round(float(np.linalg.norm(train_action29[:3] - train_proprio[:3])), 5),
        "grid1_xyz_step_m": round(float(np.linalg.norm(train_action0[:3] - train_proprio[:3])), 5),
    }

    # scale reference: how far does the plant actually move in one control tick?
    steps = np.linalg.norm(np.diff(demo_pose[:, :3], axis=0), axis=1)
    rep["demo_motion_scale"] = {
        "per_step_xyz_mean_m": round(float(steps.mean()), 5),
        "per_step_xyz_p95_m": round(float(np.percentile(steps, 95)), 5),
        "per_step_xyz_max_m": round(float(steps.max()), 5),
        "total_path_len_m": round(float(steps.sum()), 4),
        "note": "an action-chunk prediction error much larger than the per-step motion "
                "means the commanded pose is dominated by prediction error, not by intent",
    }

    print(json.dumps(rep, indent=2, ensure_ascii=False), flush=True)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
        print(f"[diag] wrote {p}", flush=True)
    print(f"[diag] PNGs in {out_dir}", flush=True)


def _load_processor():
    XB.add_xvla_to_path()
    from models.processing_xvla import XVLAProcessor
    return XVLAProcessor.from_pretrained(str(XB.XVLA_PT))


if __name__ == "__main__":
    main()
