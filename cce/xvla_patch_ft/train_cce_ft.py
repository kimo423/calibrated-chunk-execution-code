# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------
#
# CCE (idea_v2) fork of X-VLA `train.py` for the *phase-2* leg of the official
# new-domain recipe: everything unfrozen (Florence2 + action transformer + the
# three domain-aware modules), starting from a phase-1 domain-row checkpoint.
#
# Relationship to upstream `train.py` (verified line by line, 2026-09-06):
#
#   * `build_optimizer` here is upstream's L116-129 verbatim: the same four
#     param groups (`vlm`, `transformer_core`, `soft_prompts`, `action_heads`),
#     the same membership rule, the same `lr * lr_coef_soft` on soft prompts.
#   * upstream's phase-2 base LRs (L156-161) are
#         vlm              = learning_rate * learning_coef
#         transformer_core = learning_rate
#         soft_prompts     = learning_rate * learning_coef
#         action_heads     = learning_rate
#     i.e. with the default `learning_coef = 1.0` all four groups run at the SAME
#     base LR once `step >= freeze_steps`. There is no per-group scaling to copy.
#   * upstream applies NO warmup at the phase-1 -> phase-2 boundary unless
#     `--use_cosine_decay` is passed (L171: `schedule(...) if use_cosine_decay
#     else base_lr`), so with the default flags the vlm/core groups jump from 0
#     to 1e-4 in one step. This fork keeps this repo's phase-1 convention
#     instead: a linear warmup from 0 to the base LR over `warmup_steps`, then
#     hold. That is the only intentional deviation from upstream's schedule and
#     it is recorded in the run status JSON.
#   * upstream runs phase-1 and phase-2 inside one process, switching at
#     `freeze_steps`. Here phase-1 is a *previous* 12000-step run
#     (`data/cce/ckpt/phase1_b32/ckpt-12000`) whose three domain-aware modules
#     are loaded before phase-2 starts, so `freeze_steps` defaults to 0 and all
#     four groups are live from step 0.
#   * the forward is upstream's `model(**inputs)` verbatim (no inlining, no
#     detach): with the VLM trainable there is nothing to detach.
#   * checkpoints are full `save_pretrained` + `processor.save_pretrained`
#     exactly as upstream L263-268, since phase-2 moves every parameter.
#
# What this file does NOT touch: `cce/xvla_patch/train_cce.py`, the patched
# `datasets` package, `cce/eval_xvla.py`, `cce/xvla_bridge.py`. The patched
# `datasets` package is imported read-only through the launcher's sys.path.
# ------------------------------------------------------------------------------

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import psutil
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator
from datasets import create_dataloader
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor

PHASE1_MODULES = ("soft_prompt_hub", "action_encoder", "action_decoder")


# ============================================================
# logger  (upstream L44-65 verbatim)
# ============================================================
def get_logger(name="train", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("XVLA Training (CCE phase-2 full fine-tune)", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, required=True, help="Path or HF repo for pretrained XVLA")
    parser.add_argument("--output_dir", type=str, default="runnings")

    # Data
    parser.add_argument("--train_metas_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)

    # Optimizer  (upstream defaults, except weight_decay which follows this
    # repo's phase-1 / LoRA runs at 0.01 -- upstream's default is 0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0, help="LR multiplier for vlm + soft prompts")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--freeze_steps", type=int, default=0,
                        help="steps before vlm/transformer_core open; 0 because phase-1 ran separately")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=50)

    # System
    parser.add_argument("--seed", type=int, default=0)

    # ---- CCE additions -------------------------------------------------
    parser.add_argument("--phase1_ckpt", type=str, default=None,
                        help="dir holding cce_domain_state.safetensors from the phase-1 run")
    parser.add_argument("--vlm_eval", type=int, default=0,
                        help="0 = upstream phase-2 (whole model in train()); "
                             "1 = keep Florence2 in eval() as this repo's phase-1/LoRA runs did")
    parser.add_argument("--status_json", type=str, default=None)
    parser.add_argument("--bench_steps", type=int, default=0,
                        help=">0: run only this many steps and report throughput, no checkpoints")
    parser.add_argument("--bench_save_dir", type=str, default=None,
                        help="bench mode only: save_pretrained here and reload it to prove the "
                             "full-model checkpoint round-trips bit-exactly")
    parser.add_argument("--safe_serialization", type=int, default=1,
                        help="1: safetensors (upstream); 0: torch .bin, needed only if safetensors "
                             "drops tied tensors and the round-trip check fails")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def load_phase1_domain_state(model: XVLA, ckpt: str | None, logger) -> dict:
    """Load the three domain-aware modules trained by the phase-1 true-freeze run."""
    if not ckpt:
        logger.info("phase1_ckpt: none (starting phase-2 from the untouched base rows)")
        return {"phase1_ckpt": None, "loaded_keys": 0}
    p = Path(ckpt)
    f = p / "cce_domain_state.safetensors"
    if f.exists():
        from safetensors.torch import load_file
        sd = load_file(str(f))
    else:
        f = p / "cce_domain_state.pt"
        if not f.exists():
            raise SystemExit(f"no phase-1 domain state under {p}")
        sd = torch.load(str(f), map_location="cpu")
    cur = model.state_dict()
    missing = [k for k in sd if k not in cur]
    if missing:
        raise SystemExit(f"phase-1 checkpoint keys absent from the model: {missing[:5]}")
    deltas = {k: float((sd[k].float() - cur[k].detach().cpu().float()).abs().max()) for k in sd}
    model.load_state_dict(sd, strict=False)
    after = model.state_dict()
    residual = max(float((sd[k].float() - after[k].detach().cpu().float()).abs().max()) for k in sd)
    if residual > 0:
        raise SystemExit(f"phase-1 load did not take: residual max|delta| = {residual}")
    step = None
    sj = p / "state.json"
    if sj.exists():
        step = json.loads(sj.read_text()).get("global_step")
    info = {"phase1_ckpt": str(p), "phase1_global_step": step, "loaded_keys": len(sd),
            "max_abs_delta_vs_base": {k: round(v, 6) for k, v in deltas.items()},
            "post_load_residual": residual}
    logger.info(f"phase-1 domain state loaded: {len(sd)} tensors from {p} (step {step}); "
                f"max|delta| vs base per tensor = "
                f"{ {k: round(v, 5) for k, v in deltas.items()} }")
    return info


def apply_full_unfreeze(model: XVLA, logger) -> dict:
    """Phase-2: every parameter trainable. The reverse of the phase-1 freeze check."""
    model.requires_grad_(True)
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_vlm = sum(p.numel() for p in model.vlm.parameters())
    n_phase1 = sum(p.numel()
                   for name in PHASE1_MODULES
                   for p in getattr(model.transformer, name).parameters())
    stats = {
        "trainable_params": int(n_train),
        "total_params": int(n_total),
        "vlm_params": int(n_vlm),
        "phase1_three_module_params": int(n_phase1),
        "transformer_core_params": int(n_total - n_vlm - n_phase1),
        "frozen_leaks": frozen[:8],
        "n_frozen": len(frozen),
        "all_trainable": bool(n_train == n_total and not frozen),
    }
    if not stats["all_trainable"]:
        raise SystemExit(f"phase-2 unfreeze incomplete: {len(frozen)} frozen tensors, e.g. {frozen[:8]}")
    logger.info(f"full-unfreeze: {stats}")
    return stats


def build_optimizer(model: XVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_soft=1.0):
    """Upstream `train.py` L116-129, verbatim."""
    vlm_params = list(model.vlm.parameters())
    soft_prompt_params = list(model.transformer.soft_prompt_hub.parameters())
    action_params = (list(model.transformer.action_decoder.parameters())
                     + list(model.transformer.action_encoder.parameters()))
    exclude = set(map(id, vlm_params + soft_prompt_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "soft_prompts", "params": soft_prompt_params, "lr": lr * lr_coef_soft, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups:
        if g["name"] == name:
            g["lr"] = lr


def update_group_lrs(optim, step, args):
    """Upstream's phase-2 base LRs (train.py L156-161), with this repo's linear warmup.

    Upstream jumps every group straight to its base LR at `freeze_steps` unless
    `--use_cosine_decay`; the four base LRs are identical when
    `learning_coef == 1.0`. Here the same four base LRs are reached through a
    linear warmup from 0 over `warmup_steps`, matching the phase-1 run in this
    repo, and then held (or cosine-decayed if `--use_cosine_decay`).
    """
    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
    }
    for name, base_lr in base.items():
        if name in ("vlm", "transformer_core") and step < args.freeze_steps:
            set_group_lr(optim, name, 0.0)
            continue
        start = args.freeze_steps if name in ("vlm", "transformer_core") else 0
        progress = step - start
        if progress < args.warmup_steps:
            lr = base_lr * (progress + 1) / max(1, args.warmup_steps)
        elif args.use_cosine_decay:
            remain = max(1, args.iters - (start + args.warmup_steps))
            ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - args.warmup_steps) / remain)))
            lr = base_lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * ratio)
        else:
            lr = base_lr
        set_group_lr(optim, name, lr)


def save_full_ckpt(model: XVLA, processor, save_dir: str, global_step: int, extra: dict,
                   safe: bool = True) -> dict:
    """Upstream train.py L263-268: full save_pretrained + processor + state.json."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=safe)
    processor.save_pretrained(save_dir)
    payload = {"global_step": global_step, **extra}
    with open(os.path.join(save_dir, "state.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def roundtrip_check(model: XVLA, processor, save_dir: str, logger, safe: bool = True) -> dict:
    """Save the full model, reload it with XVLA.from_pretrained, compare every tensor.

    A phase-2 checkpoint is only useful if `cce/eval_xvla.py --model-path <dir>`
    can read it back, so the bench run proves it before the long run starts.
    """
    os.makedirs(save_dir, exist_ok=True)
    t0 = time.time()
    model.save_pretrained(save_dir, safe_serialization=safe)
    processor.save_pretrained(save_dir)
    save_s = time.time() - t0
    t0 = time.time()
    reloaded = XVLA.from_pretrained(save_dir, torch_dtype=torch.float32)
    XVLAProcessor.from_pretrained(save_dir)
    load_s = time.time() - t0
    a = model.state_dict()
    b = reloaded.state_dict()
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    worst_k, worst_v = None, 0.0
    for k in set(a) & set(b):
        d = float((a[k].detach().cpu().float() - b[k].detach().cpu().float()).abs().max())
        if d > worst_v:
            worst_k, worst_v = k, d
    files = sorted(p.name for p in Path(save_dir).iterdir())
    bytes_ = sum(p.stat().st_size for p in Path(save_dir).iterdir() if p.is_file())
    out = {"safe_serialization": bool(safe),
           "save_s": round(save_s, 1), "load_s": round(load_s, 1),
           "n_tensors_saved": len(a), "n_tensors_reloaded": len(b),
           "keys_only_in_live_model": only_a[:8], "keys_only_in_reloaded": only_b[:8],
           "max_abs_delta": worst_v, "worst_key": worst_k,
           "files": files, "bytes": bytes_,
           "ok": bool(not only_a and not only_b and worst_v == 0.0)}
    logger.info(f"save/load round-trip: {out}")
    del reloaded
    torch.cuda.empty_cache()
    return out


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)
    accelerator = Accelerator(log_with="tensorboard", project_dir=output_dir)
    accelerator.init_trackers("XVLA-CCE-Phase2-FullFT")
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)

    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")
    logger.info(f"mixed_precision={accelerator.mixed_precision} (upstream trains in fp32 with no AMP)")

    model = XVLA.from_pretrained(args.models)
    processor = XVLAProcessor.from_pretrained(args.models)

    phase1_info = load_phase1_domain_state(model, args.phase1_ckpt, logger)
    freeze_stats = apply_full_unfreeze(model, logger)

    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
        num_workers=args.num_workers,
    )

    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_soft=args.learning_coef,
    )
    group_sizes = {g["name"]: int(sum(p.numel() for p in g["params"])) for g in optim.param_groups}
    logger.info(f"param groups (upstream membership): {group_sizes} "
                f"| sum={sum(group_sizes.values())} vs total={freeze_stats['total_params']}")

    model, optim = accelerator.prepare(model, optim)
    raw = accelerator.unwrap_model(model)

    model.train()
    if args.vlm_eval:
        raw.vlm.eval()
        logger.info("vlm kept in eval() (this repo's phase-1/LoRA convention, NOT upstream phase-2)")

    global_step, t0 = 0, time.time()
    status_path = Path(args.status_json) if args.status_json else output_dir / "train_status.json"
    logger.info(f"Start CCE phase-2 full fine-tune for {args.iters} iters "
                f"| world_size={accelerator.num_processes} | batch={args.batch_size}")
    domain_hist: Dict[int, int] = {}
    loss_hist = []
    save_log = []

    for batch in train_dataloader:
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
        for d in inputs["domain_id"].detach().cpu().tolist():
            domain_hist[int(d)] = domain_hist.get(int(d), 0) + 1

        update_group_lrs(optim, global_step, args)

        # upstream train.py L230-236, verbatim (nothing is detached in phase-2)
        loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss = sum(loss_dict.values())
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad(set_to_none=True)
        loss_hist.append(float(loss.detach().item()))

        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)
            if accelerator.is_main_process:
                dt = (time.time() - t0) / max(1, args.log_interval)
                t0 = time.time()
                cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                gpu_peak = torch.cuda.max_memory_allocated() / 1024**3
                gpu_res = torch.cuda.max_memory_reserved() / 1024**3
                logger.info(
                    f"[{global_step}/{args.iters}] loss={logs['loss_total']:.4f} "
                    f"pos={logs.get('position_loss', float('nan')):.4f} "
                    f"rot={logs.get('rotate6D_loss', float('nan')):.4f} "
                    f"grip={logs.get('gripper_loss', float('nan')):.4f} "
                    f"lr_vlm={logs['lr_vlm']:.2e} lr_core={logs['lr_transformer_core']:.2e} "
                    f"({dt:.3f}s/it) CPU={cpu_mem:.2f}GB GPU={gpu_mem:.2f}/{gpu_peak:.2f}"
                    f"(res {gpu_res:.2f})GB"
                )
                status = {
                    "mode": "phase2-full-ft",
                    "global_step": global_step,
                    "iters": args.iters,
                    "loss_total": logs["loss_total"],
                    "loss_recent_mean": float(np.mean(loss_hist[-200:])),
                    "losses": {k: logs[k] for k in loss_dict},
                    "s_per_it": round(dt, 4),
                    "gpu_alloc_gb": round(gpu_mem, 3),
                    "gpu_peak_gb": round(gpu_peak, 3),
                    "gpu_reserved_peak_gb": round(gpu_res, 3),
                    "cpu_rss_gb": round(cpu_mem, 3),
                    "batch_size": args.batch_size,
                    "lr": {g["name"]: g["lr"] for g in optim.param_groups},
                    "param_group_sizes": group_sizes,
                    "domain_sample_counts": domain_hist,
                    "trainable_params": freeze_stats["trainable_params"],
                    "total_params": freeze_stats["total_params"],
                    "phase1_ckpt": phase1_info.get("phase1_ckpt"),
                    "warmup_note": "linear warmup to upstream base LR then hold (upstream jumps)",
                    "saved_checkpoints": save_log,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text(json.dumps(status, indent=2) + "\n")

        global_step += 1
        if args.bench_steps and global_step >= args.bench_steps:
            logger.info(
                f"bench done: {global_step} steps | batch={args.batch_size} "
                f"| peak_alloc={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB "
                f"| peak_reserved={torch.cuda.max_memory_reserved() / 1024**3:.2f}GB "
                f"| trainable={freeze_stats['trainable_params']}/{freeze_stats['total_params']} "
                f"| take s/it from the periodic log lines above (the first includes warmup)")
            if args.bench_save_dir:
                rt = roundtrip_check(accelerator.unwrap_model(model), processor,
                                     args.bench_save_dir, logger,
                                     safe=bool(args.safe_serialization))
                logger.info(f"ROUNDTRIP_OK={rt['ok']}")
            break
        if accelerator.is_main_process and not args.bench_steps:
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"Saving full phase-2 model to {save_dir}")
                ts = time.time()
                extra = {"loss_recent_mean": float(np.mean(loss_hist[-200:])),
                         "domain_sample_counts": dict(domain_hist),
                         "base_model": args.models,
                         "phase1_ckpt": phase1_info.get("phase1_ckpt"),
                         "mode": "phase2-full-ft",
                         "batch_size": args.batch_size,
                         "learning_rate": args.learning_rate}
                save_full_ckpt(accelerator.unwrap_model(model), processor, save_dir, global_step, extra,
                               safe=bool(args.safe_serialization))
                save_log.append({"step": global_step, "dir": save_dir,
                                 "save_s": round(time.time() - ts, 1)})
                t0 = time.time()  # do not charge the save to the next s/it
        if global_step >= args.iters:
            break

    accelerator.end_training()


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("XVLA CCE phase-2 full fine-tune", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
