"""Finite single-GPU spatial adaptation, with exact loading and freeze audits."""
import _bootstrap  # noqa: F401
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import xvla_bridge as XB
XB.add_xvla_to_path()
sys.path.insert(0, str(Path(__file__).parent / 'xvla_patch'))
from cce.xvla_patch import train_cce as T
from cce.policyside_model_v4r1 import load_policy, frozen_digest, MODULES


def atomic(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--mode', choices=['ft_demo', 'prompt_fit'], required=True)
    ap.add_argument('--meta', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--freeze'); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if not args.smoke:
        from cce.freeze_training_v4r1 import validate_training_freeze
        spec = validate_training_freeze(args.freeze, data=True)
        assert str(Path(args.meta).resolve()) == spec['train_meta']
    else:
        assert Path(args.meta).resolve() == Path('/opt/cce/data/cce/policyside_v4r1_20260909/dev_demo/dataset_meta.json')
    root = Path(args.out_dir); root.mkdir(parents=True, exist_ok=False)
    seed = 20260909
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    model, raw, processor, stats = load_policy(args.mode, training=True)
    before = frozen_digest(model)
    atomic(root/'initialization.json', {**stats, 'frozen_sha256': before, 'seed': seed})
    model.cuda(); model.train(); raw.vlm.eval()
    optimizer = (T.build_optimizer_lora(model, 1e-4, .01) if args.mode == 'ft_demo'
                 else T.build_optimizer(model, 1e-4, .01))
    loader = T.create_dataloader(batch_size=32, metas_path=args.meta, num_actions=30,
                                 action_mode='ee6d', training=True, num_workers=4)
    steps = 3 if args.smoke else 5000
    initial_trainable = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    started = time.monotonic(); losses = []; first_gradient = None
    with (root/'losses.jsonl').open('x') as log:
        for step, batch in enumerate(loader):
            language = processor.encode_language(batch.pop('language_instruction'))
            inputs = {k: v.cuda(non_blocking=True) for k, v in {**batch, **language}.items()}
            assert set(inputs['domain_id'].tolist()) == {3}
            if args.smoke and step == 0:
                # Both smoke processes use the same data and RNG; compare these
                # outputs across merged prompt-fit and unmerged LoRA loading.
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(seed)
                    prediction = raw.generate_actions(**{k: v[:1] for k, v in inputs.items() if k != 'action'}, steps=10)
                np.save(root/'initial_prediction.npy', prediction.detach().cpu().numpy())
                model.train(); raw.vlm.eval()
            lr = 1e-4 * min(1., (step+1)/200.)
            for group in optimizer.param_groups:
                group['lr'] = lr if group['name'] not in ('vlm', 'transformer_core') else 0.
            action = inputs['action']; b = len(action)
            t = (torch.rand(1, device='cuda') + torch.arange(b, device='cuda')/b) % (1-1e-5)
            noisy_all = torch.randn_like(action)*t[:, None, None] + action*(1-t[:, None, None])
            parts = {}; total_loss = 0.
            # Two equal microbatches, one optimizer update, effective batch 32.
            # Sample time/noise for the whole batch before splitting.
            for lo in range(0, b, 16):
                hi = min(lo+16, b); weight = (hi-lo)/b
                sub = {k: v[lo:hi] for k, v in inputs.items()}
                if args.mode == 'ft_demo':
                    enc = raw.forward_vlm(sub['input_ids'], sub['image_input'], sub['image_mask'])
                else:
                    with torch.no_grad():
                        enc = raw.forward_vlm(sub['input_ids'], sub['image_input'], sub['image_mask'])
                    enc = {k: v.detach() for k, v in enc.items()}
                proprio, noisy = raw.action_space.preprocess(sub['proprio'], noisy_all[lo:hi])
                pred = raw.transformer(domain_id=sub['domain_id'], action_with_noise=noisy, t=t[lo:hi], proprio=proprio, **enc)
                micro_parts = raw.action_space.compute_loss(pred, action[lo:hi]); micro_loss = sum(micro_parts.values())
                assert torch.isfinite(micro_loss), 'Nonfinite training loss'
                (micro_loss*weight).backward()
                total_loss += float(micro_loss.detach())*weight
                for k, v in micro_parts.items(): parts[k] = parts.get(k, 0.) + float(v.detach())*weight
                del enc, pred, micro_loss, micro_parts
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad), 'Frozen gradient leak'
            grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            assert torch.isfinite(grad), 'Nonfinite gradient'
            if first_gradient is None:
                first_gradient = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
                assert any(v > 0 for v in first_gradient.values())
                atomic(root/'first_gradients.json', first_gradient)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            item = {'step': step+1, 'loss': total_loss, 'lr': lr, **parts}
            losses.append(item['loss']); log.write(json.dumps(item)+'\n'); log.flush()
            if (step+1) % 20 == 0 or step == 0 or step+1 == steps:
                state = {**item, 'total_steps': steps, 'mode': args.mode,
                         'wall_s': time.monotonic()-started, 's_per_step': (time.monotonic()-started)/(step+1),
                         'peak_gpu_gb': torch.cuda.max_memory_allocated()/1024**3,
                         'recent_loss': float(np.mean(losses[-100:])), 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')}
                atomic(root/'status.json', state); print(json.dumps(state), flush=True)
            if step+1 == steps:
                break
    assert step+1 == steps and len(losses) == steps, 'Training ended before the requested update count'
    changed = [n for n, p in model.named_parameters() if p.requires_grad and not torch.equal(p.detach().cpu(), initial_trainable[n])]
    assert changed, 'No trainable parameter changed'
    after = frozen_digest(model); assert before == after, 'Frozen parameters changed'
    checkpoint = root/f'ckpt-{steps}'
    checkpoint.mkdir()
    if args.mode == 'ft_demo':
        model.save_pretrained(str(checkpoint), safe_serialization=True)
    else:
        from safetensors.torch import save_file
        save_file(T.domain_state_dict(raw), str(checkpoint/'cce_domain_state.safetensors'))
    atomic(root/'training_audit.json', {'status': 'passed', 'steps': steps, 'changed_trainable_tensors': len(changed),
                                       'frozen_sha256_before': before, 'frozen_sha256_after': after,
                                       'checkpoint': str(checkpoint), 'smoke': args.smoke})
    (root/'training.exit').write_text('0\n')
    print('Training complete; frozen parameter digest unchanged', flush=True)


if __name__ == '__main__':
    main()
