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
# CCE (idea_v2) fork of X-VLA `train.py`. Differences from upstream, all of them
# confined to phase-1 new-domain registration:
#
#   1. TRUE freeze. Upstream keeps every parameter trainable and only sets the
#      `vlm` / `transformer_core` group learning rates to 0.0, so the full
#      Florence2 backward still runs (~14 GB and ~2x the step time for nothing).
#      Here the frozen modules get `requires_grad_(False)` and the VLM features
#      are computed under `no_grad`, so the graph stops at `vlm_proj`.
#      Only `soft_prompt_hub`, `action_encoder`, `action_decoder` are trained --
#      exactly the three domain-aware modules that carry a domain row.
#   2. The forward is inlined (it is `XVLA.forward` verbatim apart from the
#      detached encoder output) so that the detach is explicit and auditable.
#   3. Phase-1 only by default: the group-LR schedule stays upstream's, but
#      `freeze_steps` is pinned to `iters` so the vlm/core groups never open.
#   4. Checkpoints store only the three domain-aware modules (~15 MB) instead of
#      re-serializing the 3.5 GB backbone every save_interval.
#   5. A `train_status.json` heartbeat is written for out-of-band monitoring.
#
#   6. `--lora 1` switches to the *upstream `peft_train.py` recipe* instead of
#      the true-freeze recipe: a peft `LoraConfig` with r=8, alpha=16, bias=none,
#      `modules_to_save = soft_prompt_hub / action_encoder / action_decoder`, and
#      base weights frozen. `--lora_scope official` reproduces upstream's
#      `target_modules="all-linear"` (which is what the released
#      `X-VLA-libero-spatial-peft` adapter_config resolves to, VLM included);
#      `--lora_scope core` restricts the adapters to the action transformer and
#      keeps Florence2 detached under `no_grad`, which is much cheaper per step.
#      Checkpoints in this mode are peft adapters (`adapter_model.safetensors`),
#      which already contain the three `modules_to_save` copies.
# ------------------------------------------------------------------------------

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator
from datasets import create_dataloader
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor

import logging
import os
import sys
import psutil

TRAINABLE_MODULES = ("soft_prompt_hub", "action_encoder", "action_decoder")
MODULES_TO_SAVE = ["transformer.soft_prompt_hub",
                   "transformer.action_encoder",
                   "transformer.action_decoder"]


# ============================================================
# logger
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
    parser = argparse.ArgumentParser("XVLA Training (CCE phase-1)", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, required=True, help="Path or HF repo for pretrained XVLA")
    parser.add_argument("--output_dir", type=str, default="runnings", help="Directory to save checkpoints")

    # Data
    parser.add_argument("--train_metas_path", type=str, required=True, help="Path to training metadata")
    parser.add_argument("--batch_size", type=int, default=16)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0, help="LR multiplier for soft prompts")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)

    # ---- CCE additions -------------------------------------------------
    parser.add_argument("--phase1_only", type=int, default=1,
                        help="1: pin freeze_steps=iters so vlm/core never unfreeze (true-freeze mode only)")
    parser.add_argument("--vlm_eval", type=int, default=1,
                        help="1: keep the Florence2 backbone in eval() so its features are deterministic")
    parser.add_argument("--save_full_final", action="store_true", default=False,
                        help="also write a full save_pretrained at the last step (~3.5 GB)")
    parser.add_argument("--status_json", type=str, default=None)
    parser.add_argument("--bench_steps", type=int, default=0,
                        help=">0: run only this many steps and report throughput, no checkpoints")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="dataloader workers; the gzip HDF5 decode is the throughput bound")

    # ---- LoRA route (upstream peft_train.py recipe) ---------------------
    parser.add_argument("--lora", type=int, default=0,
                        help="1: train a peft LoRA adapter instead of the true-freeze three-module set")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_scope", type=str, default="official", choices=["official", "core"],
                        help="official: target_modules='all-linear' exactly as upstream peft_train.py "
                             "(adapters inside Florence2 too, so the VLM backward runs); "
                             "core: adapters only on nn.Linear inside `transformer`, VLM stays detached")
    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def inner_xvla(model):
    """Return the raw XVLA module through any peft wrapper."""
    mod = model
    for _ in range(6):
        if hasattr(mod, "transformer") and hasattr(mod, "action_space"):
            return mod
        nxt = None
        for name in ("base_model", "model"):
            cand = getattr(mod, name, None)
            if cand is not None and cand is not mod:
                nxt = cand
                break
        if nxt is None:
            break
        mod = nxt
    return mod


def apply_true_freeze(model: XVLA) -> Dict[str, int]:
    """requires_grad_(False) everywhere except the three domain-aware modules."""
    model.requires_grad_(False)
    for name in TRAINABLE_MODULES:
        getattr(model.transformer, name).requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": int(n_train), "total_params": int(n_total)}


def apply_lora(model: XVLA, args, logger):
    """Wrap the model in the upstream peft recipe and report exactly what moved."""
    from peft import LoraConfig, get_peft_model

    if args.lora_scope == "core":
        targets = sorted(n for n, m in model.named_modules()
                         if isinstance(m, torch.nn.Linear) and n.startswith("transformer."))
        if not targets:
            raise SystemExit("lora_scope=core matched no nn.Linear under `transformer.`")
    else:
        targets = "all-linear"

    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=targets,
        modules_to_save=list(MODULES_TO_SAVE),
    )
    peft_model = get_peft_model(model, cfg)

    lora_names = sorted({n.rsplit(".lora_", 1)[0] for n, _ in peft_model.named_parameters() if ".lora_" in n})
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    # Auditable check that nothing outside {lora_*, modules_to_save} is trainable.
    leaks = [n for n, p in peft_model.named_parameters()
             if p.requires_grad and ".lora_" not in n and "modules_to_save" not in n]
    stats = {
        "trainable_params": int(n_train),
        "total_params": int(n_total),
        "n_lora_adapted_modules": len(lora_names),
        "lora_in_vlm": int(any(n.startswith("base_model.model.vlm") for n in lora_names)),
        "frozen_leaks": leaks[:8],
    }
    logger.info(f"lora: scope={args.lora_scope} r={args.lora_r} alpha={args.lora_alpha} | {stats}")
    logger.info(f"lora: first adapted modules {lora_names[:6]} ... last {lora_names[-3:]}")
    if leaks:
        raise SystemExit(f"LoRA freeze leaked: {leaks[:8]}")
    return peft_model, stats


def build_optimizer(model: XVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_soft=1.0):
    """Upstream's four param groups, so the group-LR schedule/logging is unchanged."""
    vlm_params = list(model.vlm.parameters())
    soft_prompt_params = list(model.transformer.soft_prompt_hub.parameters())
    action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    exclude = set(map(id, vlm_params + soft_prompt_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "soft_prompts", "params": soft_prompt_params, "lr": lr * lr_coef_soft, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, betas=betas)


def build_optimizer_lora(peft_model, lr, weight_decay, betas=(0.9, 0.95), lr_coef_soft=1.0):
    """Three live groups: the LoRA adapters and the two domain-aware families."""
    sp, ah, lo = [], [], []
    for n, p in peft_model.named_parameters():
        if not p.requires_grad:
            continue
        if "soft_prompt_hub" in n:
            sp.append(p)
        elif "action_encoder" in n or "action_decoder" in n:
            ah.append(p)
        else:
            lo.append(p)
    groups = [
        {"name": "lora", "params": lo, "lr": lr, "weight_decay": weight_decay},
        {"name": "soft_prompts", "params": sp, "lr": lr * lr_coef_soft, "weight_decay": weight_decay},
        {"name": "action_heads", "params": ah, "lr": lr, "weight_decay": weight_decay},
    ]
    return AdamW([g for g in groups if g["params"]], betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups:
        if g["name"] == name: g["lr"] = lr


def get_group_lr(optim: torch.optim.Optimizer, name: str) -> float:
    for g in optim.param_groups:
        if g["name"] == name: return g["lr"]
    return 0.0


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Linear warmup followed by cosine decay."""
    if step < start: return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optim, step, args):
    """Upstream group-wise LR scheduler, with warmup applied to the phase-1 groups."""
    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
    }
    if step < args.freeze_steps:
        # CCE: warm the two live groups up from 0 over `warmup_steps`, then hold.
        # Upstream jumps straight to the base LR here; with a brand-new domain row
        # that first step is a large, badly conditioned update.
        scale = min(1.0, (step + 1) / max(1, args.warmup_steps))
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "soft_prompts", base["soft_prompts"] * scale)
        set_group_lr(optim, "action_heads", base["action_heads"] * scale)
    else:
        for name, base_lr in base.items():
            new_lr = schedule_lr(step, base_lr, args) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


def update_group_lrs_lora(optim, step, args):
    """Linear warmup then hold (or cosine decay) on every live LoRA-mode group."""
    base = {
        "lora": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
    }
    for name, base_lr in base.items():
        if step < args.warmup_steps:
            lr = base_lr * (step + 1) / max(1, args.warmup_steps)
        elif args.use_cosine_decay:
            remain = max(1, args.iters - args.warmup_steps)
            ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup_steps) / remain)))
            lr = base_lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * ratio)
        else:
            lr = base_lr
        set_group_lr(optim, name, lr)


def schedule_lr(step, base_lr, args):
    return linear_warmup_cosine(step, args.freeze_steps, args.warmup_steps, args.iters, base_lr, args.min_lr_ratio)


def domain_state_dict(model: XVLA) -> Dict[str, torch.Tensor]:
    out = {}
    for name in TRAINABLE_MODULES:
        mod = getattr(model.transformer, name)
        for k, v in mod.state_dict().items():
            out[f"transformer.{name}.{k}"] = v.detach().cpu().clone()
    return out


def save_domain_ckpt(model: XVLA, save_dir: str, global_step: int, extra: dict) -> None:
    os.makedirs(save_dir, exist_ok=True)
    sd = domain_state_dict(model)
    try:
        from safetensors.torch import save_file
        save_file(sd, os.path.join(save_dir, "cce_domain_state.safetensors"))
    except Exception:
        torch.save(sd, os.path.join(save_dir, "cce_domain_state.pt"))
    with open(os.path.join(save_dir, "state.json"), "w") as f:
        json.dump({"global_step": global_step, **extra}, f, indent=2)


def save_lora_ckpt(peft_model, save_dir: str, global_step: int, extra: dict) -> None:
    """peft adapter: LoRA weights + the three `modules_to_save` copies in one dir."""
    os.makedirs(save_dir, exist_ok=True)
    peft_model.save_pretrained(save_dir, safe_serialization=True)
    with open(os.path.join(save_dir, "state.json"), "w") as f:
        json.dump({"global_step": global_step, **extra}, f, indent=2)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)
    accelerator = Accelerator(log_with="tensorboard", project_dir=output_dir)
    accelerator.init_trackers("XVLA-CCE-Phase1")
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)

    set_seed(args.seed + accelerator.process_index)
    if args.phase1_only and not args.lora:
        args.freeze_steps = max(args.freeze_steps, args.iters + 1)
    logger.info(f"Args: {args}")

    # Load model & processor
    model = XVLA.from_pretrained(args.models)
    processor = XVLAProcessor.from_pretrained(args.models)

    if args.lora:
        model, freeze_stats = apply_lora(model, args, logger)
    else:
        freeze_stats = apply_true_freeze(model)
        logger.info(f"true-freeze: {freeze_stats} | trainable modules: {TRAINABLE_MODULES}")

    xvla = inner_xvla(model)
    vlm_trainable = any(p.requires_grad for p in xvla.vlm.parameters())
    logger.info(f"vlm_trainable={vlm_trainable} (False => the VLM forward runs under no_grad and is detached)")

    # Iterable dataloader (don't wrap with prepare)
    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=xvla.num_actions,
        action_mode=xvla.action_mode,
        training=True,
        num_workers=args.num_workers,
    )

    if args.lora:
        optim = build_optimizer_lora(model, args.learning_rate, args.weight_decay,
                                     tuple(args.betas), args.learning_coef)
    else:
        optim = build_optimizer(
            model=model,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=tuple(args.betas),
            lr_coef_soft=args.learning_coef,
        )
    model, optim = accelerator.prepare(model, optim)
    raw = accelerator.unwrap_model(model)
    xvla = inner_xvla(raw)

    model.train()
    if args.vlm_eval:
        xvla.vlm.eval()

    global_step, t0 = 0, time.time()
    status_path = Path(args.status_json) if args.status_json else output_dir / "train_status.json"
    mode = f"LoRA({args.lora_scope}, r={args.lora_r})" if args.lora else "true-freeze"
    logger.info(f"Start CCE phase-1 training [{mode}] for {args.iters} iters "
                f"| world_size={accelerator.num_processes}")
    domain_hist: Dict[int, int] = {}
    loss_hist = []

    for batch in train_dataloader:
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
        for d in inputs["domain_id"].detach().cpu().tolist():
            domain_hist[int(d)] = domain_hist.get(int(d), 0) + 1

        if args.lora:
            update_group_lrs_lora(optim, global_step, args)
        else:
            update_group_lrs(optim, global_step, args)

        # ---- inlined XVLA.forward; the VLM is detached whenever it is frozen
        if vlm_trainable:
            enc = xvla.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
        else:
            with torch.no_grad():
                enc = xvla.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
            enc = {k: v.detach() for k, v in enc.items()}

        action = inputs["action"]
        proprio = inputs["proprio"]
        B = action.shape[0]
        t = (torch.rand(1, device=action.device) + torch.arange(B, device=action.device) / B) % (1 - 1e-5)
        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = xvla.action_space.preprocess(proprio, action_noisy)
        pred_action = xvla.transformer(
            domain_id=inputs["domain_id"],
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        loss_dict: Dict[str, torch.Tensor] = xvla.action_space.compute_loss(pred_action, action)
        # ---------------------------------------------------------------------

        loss = sum(loss_dict.values())
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
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
                lr_show = logs.get("lr_lora", logs.get("lr_soft_prompts", 0.0))
                logger.info(
                    f"[{global_step}/{args.iters}] loss={logs['loss_total']:.4f} "
                    f"pos={logs.get('position_loss', float('nan')):.4f} "
                    f"rot={logs.get('rotate6D_loss', float('nan')):.4f} "
                    f"grip={logs.get('gripper_loss', float('nan')):.4f} "
                    f"lr={lr_show:.2e} ({dt:.3f}s/it) "
                    f"CPU={cpu_mem:.2f}GB GPU={gpu_mem:.2f}/{gpu_peak:.2f}GB"
                )
                status = {
                    "mode": mode,
                    "global_step": global_step,
                    "iters": args.iters,
                    "loss_total": logs["loss_total"],
                    "loss_recent_mean": float(np.mean(loss_hist[-200:])),
                    "losses": {k: logs[k] for k in loss_dict},
                    "s_per_it": round(dt, 4),
                    "gpu_alloc_gb": round(gpu_mem, 3),
                    "gpu_peak_gb": round(gpu_peak, 3),
                    "cpu_rss_gb": round(cpu_mem, 3),
                    "batch_size": args.batch_size,
                    "lr": {g["name"]: g["lr"] for g in optim.param_groups},
                    "domain_sample_counts": domain_hist,
                    "trainable_params": freeze_stats["trainable_params"],
                    "total_params": freeze_stats["total_params"],
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text(json.dumps(status, indent=2) + "\n")

        global_step += 1
        if args.bench_steps and global_step >= args.bench_steps:
            logger.info(f"bench done: {global_step} steps | mode={mode} | batch={args.batch_size} "
                        f"| peak={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB "
                        f"| take s/it from the periodic log lines above (the first line includes warmup)")
            break
        if accelerator.is_main_process and not args.bench_steps:
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"Saving CCE {'LoRA adapter' if args.lora else 'domain state'} to {save_dir}")
                extra = {"loss_recent_mean": float(np.mean(loss_hist[-200:])),
                         "domain_sample_counts": domain_hist,
                         "base_model": args.models,
                         "mode": mode}
                if args.lora:
                    save_lora_ckpt(accelerator.unwrap_model(model), save_dir, global_step, extra)
                else:
                    save_domain_ckpt(inner_xvla(accelerator.unwrap_model(model)), save_dir, global_step, extra)
                    if args.save_full_final and global_step == args.iters:
                        inner_xvla(accelerator.unwrap_model(model)).save_pretrained(
                            os.path.join(save_dir, "full"), safe_serialization=True)
                        processor.save_pretrained(os.path.join(save_dir, "full"))
        if global_step >= args.iters: break

    accelerator.end_training()


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("XVLA CCE training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
