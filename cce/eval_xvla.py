#!/usr/bin/env python
"""Closed-loop ManiSkill evaluation of an X-VLA CCE domain row, plus the G0' check.

Inference protocol (aligned with the official `processing_xvla` / `libero_client`):
  * 3 image views, unused ones zero-padded with image_mask=False
  * language tokenized with padding="max_length", max_length=50
  * 10 denoising steps
  * one 30-step action chunk, drained one grid step per 20 Hz control step
    (qdur = 30 / 20 = 1.5 s, i.e. the grid step and the control step coincide)
  * gripper thresholded hard at 0.5

Deviations from `libero_client`, both deliberate:
  * proprio is re-read from the plant every chunk (the LIBERO client keeps an
    open-loop proprio copied from its own last prediction). Training pairs the
    image with the *measured* pose, so the measured pose is the matching input.
  * the predicted absolute EE pose is tracked through the shared normalized
    `pd_ee_delta_pose` bridge, since ManiSkill has no absolute-pose controller.

G0' : teacher-forced per-dimension MAE against the trivial "hold current proprio"
predictor, on held-out demonstrations, using the exact training data pipeline.

Diagnostic additions (2026-09-05, all opt-in or purely additive reporting):
  * `--replan-every K` re-queries the policy after K executed grid steps instead
    of draining all 30. K = 0 keeps the pre-registered drain-the-chunk protocol;
    any K > 0 is a *protocol deviation* and is recorded as such in the JSON.
  * per-episode failure-stage fields (approach / gripper timing / lift).
  * G0' additionally reports the MAE restricted to grid step 1 and to grid steps
    1-10, so a chunk-horizon error growth is visible separately from the mean.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import xvla_bridge as XB

PATCH = Path(__file__).resolve().parent / "xvla_patch"


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #
def load_cce_state(model, ckpt: str | None) -> dict:
    raw = XB.unwrap_xvla(model)
    if not ckpt:
        return {"ckpt": None, "loaded_keys": 0}
    p = Path(ckpt)
    f = p / "cce_domain_state.safetensors"
    if f.exists():
        from safetensors.torch import load_file
        sd = load_file(str(f))
    else:
        f = p / "cce_domain_state.pt"
        sd = torch.load(str(f), map_location="cpu")
    cur = raw.state_dict()
    missing = [k for k in sd if k not in cur]
    if missing:
        raise SystemExit(f"checkpoint keys absent from the model: {missing[:5]}")
    deltas = {k: float((sd[k].float() - cur[k].detach().cpu().float()).abs().max()) for k in sd}
    raw.load_state_dict(sd, strict=False)
    step = None
    sj = p / "state.json"
    if sj.exists():
        step = json.loads(sj.read_text()).get("global_step")
    return {"ckpt": str(p), "global_step": step, "loaded_keys": len(sd),
            "max_abs_delta_vs_base": {k: round(v, 6) for k, v in deltas.items()}}


def load_lora(model, lora_dir: str | None, device: str):
    """Attach + merge a CCE LoRA adapter (peft) trained by `train_cce.py --lora`."""
    if not lora_dir:
        return model, {"lora": None}
    from peft import PeftModel
    before = XB._soft_prompt_weight(model).clone()
    peft_model = PeftModel.from_pretrained(model, str(lora_dir), torch_dtype=torch.float32).to(device)
    try:
        model = peft_model.merge_and_unload()
        merged = True
    except Exception as exc:  # pragma: no cover
        print(f"[warn] merge_and_unload failed ({exc}); keeping the wrapper", flush=True)
        model = peft_model
        merged = False
    after = XB._soft_prompt_weight(model)
    delta = float((after - before).abs().max().item())
    print(f"[lora] merged={merged} soft_prompt_hub max|delta|={delta:.6f}", flush=True)
    return model.to(device).eval(), {"lora": str(lora_dir), "merged": merged,
                                     "soft_prompt_max_abs_delta": round(delta, 6)}


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
class ChunkPolicy:
    def __init__(self, model, processor, domain_id: int, instruction: str,
                 device: str = "cuda:0", steps: int = 10, replan_every: int = 0):
        self.model = model
        self.processor = processor
        self.domain_id = int(domain_id)
        self.instruction = instruction
        self.device = device
        self.steps = steps
        self.replan_every = int(replan_every)
        self.plan: collections.deque = collections.deque()
        self.since_query = 0
        self.n_queries = 0
        self.query_s = 0.0

    def reset(self):
        self.plan.clear()
        self.since_query = 0

    @torch.no_grad()
    def _query(self, rgb: np.ndarray, proprio20: np.ndarray) -> np.ndarray:
        from PIL import Image

        t0 = time.time()
        inputs = self.processor([Image.fromarray(rgb)], self.instruction)
        dev = self.device
        dtype = next(self.model.parameters()).dtype
        batch = {}
        for k, v in inputs.items():
            batch[k] = v.to(device=dev, dtype=dtype) if v.is_floating_point() else v.to(dev)
        batch["proprio"] = torch.as_tensor(proprio20, dtype=dtype, device=dev).unsqueeze(0)
        batch["domain_id"] = torch.tensor([self.domain_id], dtype=torch.long, device=dev)
        act = self.model.generate_actions(**batch, steps=self.steps)
        self.n_queries += 1
        self.query_s += time.time() - t0
        return act.squeeze(0).float().cpu().numpy()

    def act(self, rgb: np.ndarray, pose7: np.ndarray, closed: float) -> np.ndarray:
        if self.replan_every and self.since_query >= self.replan_every:
            self.plan.clear()
        if not self.plan:
            proprio = XB.pose7_to_ee6d20(pose7, closed)
            for row in self._query(rgb, proprio):
                self.plan.append(row)
            self.since_query = 0
        self.since_query += 1
        return self.plan.popleft()


# --------------------------------------------------------------------------- #
# closed loop
# --------------------------------------------------------------------------- #
def make_eval_env(task: str, uid: str, img: int, budget: int, sim_backend: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        task,
        robot_uids=uid,
        num_envs=1,
        obs_mode="rgb",
        reward_mode="none",
        control_mode="pd_ee_delta_pose",
        sim_backend=sim_backend,
        render_backend="sapien_cuda",
        render_mode="rgb_array",
        max_episode_steps=budget,
        sensor_configs=dict(width=img, height=img),
    )


def obs_rgb(obs, cam: str) -> np.ndarray:
    rgb = XB.np_of(obs["sensor_data"][cam]["rgb"])
    if rgb.ndim == 4:
        rgb = rgb[0]
    return np.ascontiguousarray(rgb.astype(np.uint8))


def rollout(env, policy: ChunkPolicy, uid: str, seed: int, budget: int, cam: str) -> dict:
    base = env.unwrapped
    obs, _ = env.reset(seed=seed)
    policy.reset()
    pose = XB.tcp_root_pose(base)
    closed = 0.0
    success = False
    track_err = []
    grip_hist = []
    tcp_cube = []
    cube_z = []
    cube_goal = []
    placed_steps = 0
    grasped_steps = 0

    def _p(name):
        o = getattr(base, name, None)
        return None if o is None else XB.np_of(o.pose.p).reshape(-1)[:3]

    def _flag(info, key):
        if key not in info:
            return None
        return bool(np.asarray(XB.np_of(info[key])).reshape(-1)[0])

    cube0 = _p("cube")
    z0 = float(cube0[2]) if cube0 is not None else None
    goal_p = _p("goal_site")

    for step in range(budget):
        a20 = policy.act(obs_rgb(obs, cam), pose, closed)
        tgt_pose, tgt_closed = XB.ee6d20_to_pose7_closed(a20)
        hard_closed = 1.0 if tgt_closed > 0.5 else 0.0
        a7 = XB.shared_ee_delta_action(pose, tgt_pose, XB.closed_to_shared_gripper(hard_closed))
        obs, _, term, trunc, info = env.step(XB.to_plant_action(uid, a7))
        pose = XB.tcp_root_pose(base)
        closed = hard_closed
        track_err.append(float(np.linalg.norm(pose[:3] - tgt_pose[:3])))
        grip_hist.append(hard_closed)
        cube = _p("cube")
        if cube is not None:
            tcp_world = XB.np_of(base.agent.tcp.pose.p).reshape(-1)[:3]
            tcp_cube.append(float(np.linalg.norm(tcp_world - cube)))
            cube_z.append(float(cube[2]))
            if goal_p is not None:
                cube_goal.append(float(np.linalg.norm(cube - goal_p)))
        placed_steps += int(_flag(info, "is_obj_placed") or 0)
        grasped_steps += int(_flag(info, "is_grasped") or 0)
        success = bool(np.asarray(XB.np_of(info["success"])).reshape(-1)[0])
        if success or bool(np.asarray(XB.np_of(term)).reshape(-1)[0]):
            break
    cube, goal = _p("cube"), _p("goal_site")

    # --- failure-stage decomposition ---------------------------------------
    stage = {}
    if tcp_cube:
        j = int(np.argmin(tcp_cube))
        stage["min_tcp_cube_step"] = j
        stage["min_tcp_cube_dist_m"] = round(float(tcp_cube[j]), 4)
    g = np.asarray(grip_hist)
    close_idx = np.flatnonzero(g > 0.5)
    if close_idx.size:
        k = int(close_idx[0])
        stage["first_close_step"] = k
        stage["dist_at_first_close_m"] = round(float(tcp_cube[k]), 4) if k < len(tcp_cube) else None
        # was the gripper ever closed while actually near the cube?
        near = [tcp_cube[i] for i in close_idx if i < len(tcp_cube)]
        stage["min_dist_while_closed_m"] = round(float(np.min(near)), 4) if near else None
    else:
        stage["first_close_step"] = None
        stage["dist_at_first_close_m"] = None
        stage["min_dist_while_closed_m"] = None
    stage["n_grip_switches"] = int(np.sum(np.abs(np.diff(g)) > 0.5)) if g.size > 1 else 0
    stage["placed_steps"] = placed_steps
    stage["grasped_steps"] = grasped_steps
    stage["min_cube_goal_dist_m"] = round(float(np.min(cube_goal)), 4) if cube_goal else None
    # cube inside the goal ball at some point but the episode still failed:
    # PickCube also requires `is_robot_static`, so this separates "never delivered"
    # from "delivered but never settled".
    stage["placed_but_failed"] = int(placed_steps > 0 and not success)
    if cube_z:
        stage["cube_z0_m"] = round(z0, 4) if z0 is not None else None
        stage["max_cube_z_m"] = round(float(np.max(cube_z)), 4)
        stage["lifted"] = int(z0 is not None and (max(cube_z) - z0) > 0.01)

    return {
        "seed": seed,
        "success": int(success),
        "steps": step + 1,
        "mean_tcp_tracking_error_m": round(float(np.mean(track_err)), 5) if track_err else None,
        "final_tcp_tracking_error_m": round(track_err[-1], 5) if track_err else None,
        "grip_closed_frac": round(float(np.mean(grip_hist)), 3) if grip_hist else None,
        "min_tcp_cube_dist_m": round(float(np.min(tcp_cube)), 4) if tcp_cube else None,
        "final_cube_goal_dist_m": round(float(np.linalg.norm(cube - goal)), 4)
        if cube is not None and goal is not None else None,
        "cube_lifted_m": round(float(cube[2]), 4) if cube is not None else None,
        **stage,
    }


# --------------------------------------------------------------------------- #
# G0' : teacher-forced MAE vs the trivial predictor
# --------------------------------------------------------------------------- #
POS_IDX = [0, 1, 2]
ROT_IDX = [3, 4, 5, 6, 7, 8]
GRIP_IDX = [9]


def g0_prime(model, processor, metas_path: str, domain_id_override, n_samples: int,
             device: str, steps: int, seed: int = 0, stride: int = 17) -> dict:
    """Run the training data pipeline read-only and compare against `action == proprio`.

    `training=False` keeps the color jitter off and the ordering deterministic, so a
    stride is used to spread the sampled indices across episodes instead of taking
    the first `n_samples` frames of episode 0.

    Three horizons are reported: grid step 1 only, grid steps 1-10, and the full
    1-30 chunk, so "the first command is fine but the chunk drifts" is separable
    from "every grid step is wrong".
    """
    XB.add_xvla_to_path()
    if str(PATCH) not in sys.path:
        sys.path.insert(0, str(PATCH))
    from datasets.dataset import InfiniteDataReader  # patched copy

    reader = InfiniteDataReader(metas_path, num_actions=30, training=False, action_mode="ee6d")
    raw = XB.unwrap_xvla(model)
    torch.manual_seed(seed)

    HOR = {"g1": slice(0, 1), "g1_10": slice(0, 10), "g1_30": slice(0, 30)}
    agg = {}
    n = 0
    seen = 0
    it = iter(reader)
    while n < n_samples:
        try:
            s = next(it)
        except StopIteration:
            break
        seen += 1
        if (seen - 1) % max(1, stride) != 0:
            continue
        dom = int(s["domain_id"]) if domain_id_override is None else int(domain_id_override)
        img = s["image_input"].unsqueeze(0).to(device)
        mask = s["image_mask"].unsqueeze(0).to(device)
        ids = processor.encode_language([s["language_instruction"]])["input_ids"].to(device)
        proprio = s["proprio"].unsqueeze(0).to(device)
        target = s["action"].unsqueeze(0).to(device)  # [1,30,20]
        with torch.no_grad():
            pred = raw.generate_actions(
                input_ids=ids, image_input=img, image_mask=mask,
                domain_id=torch.tensor([dom], device=device), proprio=proprio, steps=steps,
            )
        trivial = proprio.unsqueeze(1).expand_as(target)  # "hold the current proprio"
        key = f"domain_{dom}"
        d = agg.setdefault(key, {"n": 0,
                                 "model": {h: np.zeros(20) for h in HOR},
                                 "trivial": {h: np.zeros(20) for h in HOR}})
        d["n"] += 1
        for h, sl in HOR.items():
            d["model"][h] += (pred[:, sl] - target[:, sl]).abs().mean(dim=(0, 1)).float().cpu().numpy()
            d["trivial"][h] += (trivial[:, sl] - target[:, sl]).abs().mean(dim=(0, 1)).float().cpu().numpy()
        n += 1

    out = {"n_samples": n, "denoise_steps": steps, "metas_path": metas_path, "per_domain": {}}
    grp = lambda v, idx: float(np.mean(v[idx]))  # noqa: E731
    for key, d in agg.items():
        nn_ = max(d["n"], 1)
        m = {h: d["model"][h] / nn_ for h in HOR}
        t = {h: d["trivial"][h] / nn_ for h in HOR}
        horizons = {}
        for h in HOR:
            horizons[h] = {
                "model_mae": {"xyz_m": grp(m[h], POS_IDX), "rot6d": grp(m[h], ROT_IDX),
                              "gripper": grp(m[h], GRIP_IDX),
                              "all10": grp(m[h], POS_IDX + ROT_IDX + GRIP_IDX)},
                "trivial_mae": {"xyz_m": grp(t[h], POS_IDX), "rot6d": grp(t[h], ROT_IDX),
                                "gripper": grp(t[h], GRIP_IDX),
                                "all10": grp(t[h], POS_IDX + ROT_IDX + GRIP_IDX)},
                "ratio_xyz": grp(m[h], POS_IDX) / max(grp(t[h], POS_IDX), 1e-12),
                "ratio_all10": (grp(m[h], POS_IDX + ROT_IDX + GRIP_IDX)
                                / max(grp(t[h], POS_IDX + ROT_IDX + GRIP_IDX), 1e-12)),
            }
        full_m, full_t = m["g1_30"], t["g1_30"]
        out["per_domain"][key] = {
            "n": d["n"],
            "model_mae": horizons["g1_30"]["model_mae"],
            "trivial_mae": horizons["g1_30"]["trivial_mae"],
            "ratio_model_over_trivial": {
                "xyz": horizons["g1_30"]["ratio_xyz"],
                "rot6d": grp(full_m, ROT_IDX) / max(grp(full_t, ROT_IDX), 1e-12),
                "all10": horizons["g1_30"]["ratio_all10"],
            },
            "per_dim_model_mae": [round(float(x), 6) for x in full_m[:10]],
            "per_dim_trivial_mae": [round(float(x), 6) for x in full_t[:10]],
            "horizons": horizons,
            "pass": bool(grp(full_m, POS_IDX + ROT_IDX + GRIP_IDX)
                         < grp(full_t, POS_IDX + ROT_IDX + GRIP_IDX)),
        }
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser("CCE closed-loop eval + G0'")
    ap.add_argument("--uid", default="panda", choices=["panda", "xarm6_robotiq"])
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--ckpt", default=None, help="dir holding cce_domain_state.safetensors")
    ap.add_argument("--lora", default=None, help="dir holding a peft adapter for the CCE run")
    ap.add_argument("--model-path", default=str(XB.XVLA_PT))
    ap.add_argument("--domain-id", type=int, default=None, help="override the registered row")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=20280101)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--replan-every", type=int, default=0,
                    help="re-query after K executed grid steps (0 = drain the whole chunk; "
                         "any K > 0 deviates from the pre-registered protocol)")
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--camera", default="base_camera")
    ap.add_argument("--sim-backend", default="physx_cpu",
                    help="physx_cpu matches the demo-collection physics; physx_cuda also supported")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--g0-metas", default=None, help="run G0' on this metas dir/file")
    ap.add_argument("--g0-samples", type=int, default=64, help="samples per meta file")
    ap.add_argument("--g0-stride", type=int, default=17)
    ap.add_argument("--skip-rollout", action="store_true")
    ap.add_argument("--tag", default=None, help="free-form label recorded in the JSON")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, processor, info = XB.load_xvla(model_path=args.model_path, device=args.device)
    model, lora_info = load_lora(model, args.lora, args.device)
    ck = load_cce_state(model, args.ckpt)
    ck.update(lora_info)
    domain_id = args.domain_id if args.domain_id is not None else XB.DOMAIN_IDS[args.uid]
    instruction = XB.TASK_INSTRUCTIONS[args.task]
    result = {
        "uid": args.uid,
        "task": args.task,
        "tag": args.tag,
        "domain_id": domain_id,
        "instruction": instruction,
        "model": info,
        "checkpoint": ck,
        "protocol": {
            "views": 3,
            "zero_padded_views": 2,
            "language_max_length": int(processor.language_max_length),
            "denoise_steps": args.denoise_steps,
            "chunk": 30,
            "chunk_seconds": 1.5,
            "control_hz": 20,
            "gripper_threshold": 0.5,
            "budget_steps": args.budget,
            "sim_backend": args.sim_backend,
            "proprio": "measured TCP root pose each chunk",
            "seed_base": args.seed_base,
            "replan_every": args.replan_every,
            "deviations": ([] if args.replan_every == 0 else
                           [f"chunk NOT drained: re-query every {args.replan_every} grid steps "
                            f"(pre-registered protocol is drain-all-30)"])
            + ([] if args.seed_base == 20280101 else
               [f"seed_base={args.seed_base} (held-out eval seeds are 20280101+; "
                f"20260101+ are the TRAINING seeds)"]),
        },
    }

    if args.g0_metas:
        t0 = time.time()
        p = Path(args.g0_metas)
        # one pass per meta file: the reader drains one dataset fully before the
        # next when training=False, so a shared pass would only cover one domain.
        metas = sorted(str(x) for x in p.glob("*.json")) if p.is_dir() else [str(p)]
        merged = {"n_samples": 0, "denoise_steps": args.denoise_steps,
                  "metas_path": args.g0_metas, "per_domain": {}}
        for m in metas:
            part = g0_prime(model, processor, m, args.domain_id, args.g0_samples,
                            args.device, args.denoise_steps, stride=args.g0_stride)
            merged["n_samples"] += part["n_samples"]
            merged["per_domain"].update(part["per_domain"])
        merged["wall_s"] = round(time.time() - t0, 1)
        result["g0_prime"] = merged
        print("[G0'] " + json.dumps(merged["per_domain"], indent=2), flush=True)

    if not args.skip_rollout:
        env = make_eval_env(args.task, args.uid, args.img, args.budget, args.sim_backend)
        policy = ChunkPolicy(model, processor, domain_id, instruction, args.device,
                             args.denoise_steps, args.replan_every)
        rows = []
        t0 = time.time()
        try:
            for i in range(args.n):
                r = rollout(env, policy, args.uid, args.seed_base + i, args.budget, args.camera)
                rows.append(r)
                print(json.dumps(r), flush=True)
        finally:
            env.close()
        sr = float(np.mean([r["success"] for r in rows])) if rows else 0.0
        mean_of = lambda k: (round(float(np.mean([r[k] for r in rows if r.get(k) is not None])), 4)  # noqa: E731
                             if any(r.get(k) is not None for r in rows) else None)
        result["rollout"] = {
            "n": len(rows),
            "success_rate": sr,
            "successes": int(sum(r["success"] for r in rows)),
            "mean_steps": round(float(np.mean([r["steps"] for r in rows])), 1) if rows else 0,
            "mean_tcp_tracking_error_m": mean_of("mean_tcp_tracking_error_m"),
            "mean_grip_closed_frac": mean_of("grip_closed_frac"),
            "mean_min_tcp_cube_dist_m": mean_of("min_tcp_cube_dist_m"),
            "mean_final_cube_goal_dist_m": mean_of("final_cube_goal_dist_m"),
            "mean_min_tcp_cube_step": mean_of("min_tcp_cube_step"),
            "mean_first_close_step": mean_of("first_close_step"),
            "mean_dist_at_first_close_m": mean_of("dist_at_first_close_m"),
            "mean_min_dist_while_closed_m": mean_of("min_dist_while_closed_m"),
            "mean_n_grip_switches": mean_of("n_grip_switches"),
            "mean_grasped_steps": mean_of("grasped_steps"),
            "mean_min_cube_goal_dist_m": mean_of("min_cube_goal_dist_m"),
            "n_ever_grasped": int(sum(1 for r in rows if (r.get("grasped_steps") or 0) > 0)),
            "n_ever_placed": int(sum(1 for r in rows if (r.get("placed_steps") or 0) > 0)),
            "n_placed_but_failed": int(sum(r.get("placed_but_failed", 0) for r in rows)),
            "n_lifted": int(sum(r.get("lifted", 0) for r in rows)),
            "n_reached_1cm": int(sum(1 for r in rows
                                     if r.get("min_tcp_cube_dist_m") is not None
                                     and r["min_tcp_cube_dist_m"] < 0.01)),
            "n_closed_within_1cm": int(sum(1 for r in rows
                                           if r.get("min_dist_while_closed_m") is not None
                                           and r["min_dist_while_closed_m"] < 0.01)),
            "policy_queries": policy.n_queries,
            "s_per_query": round(policy.query_s / max(policy.n_queries, 1), 3),
            "wall_s": round(time.time() - t0, 1),
            "episodes": rows,
        }
        print(json.dumps({k: v for k, v in result["rollout"].items() if k != "episodes"}, indent=2), flush=True)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[eval] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
