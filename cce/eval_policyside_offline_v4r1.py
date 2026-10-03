"""Fixed teacher-forced observations: denoised actions versus persistence."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cce.policyside_model_v4r1 import load_policy


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--mode', choices=['ft_demo', 'prompt_fit'], required=True)
    ap.add_argument('--checkpoint'); ap.add_argument('--samples', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(); out = Path(args.out); assert not out.exists()
    model, raw, processor, stats = load_policy(args.mode, args.checkpoint, training=False)
    model.cuda().eval()
    samples = torch.load(args.samples, map_location='cpu', weights_only=False)
    if args.smoke:
        assert len(samples) == 3 and Path(args.samples).name == 'dev_offline_samples.pt'
        assert Path(args.checkpoint).name == 'ckpt-3'
    else: assert len(samples) == 200
    errors, trivial, per_sample = [], [], []
    for i, sample in enumerate(samples):
        inputs = {k: v[None].cuda() for k, v in sample.items() if isinstance(v, torch.Tensor) and k != 'action'}
        inputs.update({k: v.cuda() for k, v in processor.encode_language(sample['language_instruction']).items()})
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(20260909+i)
            pred = raw.generate_actions(**inputs, steps=10)[0, :, :10].cpu().numpy()
        target = sample['action'][:, :10].numpy()
        base = sample['proprio'][:10].numpy()[None, :]
        errors.append(np.abs(pred-target)); trivial.append(np.abs(base-target))
        per_sample.append({'index': i, 'position_mae_m': float(np.abs(pred[:, :3]-target[:, :3]).mean()),
                           'all10_mae': float(np.abs(pred-target).mean())})
    a, b = np.array(errors), np.array(trivial)
    horizons = {}
    for n in [1, 10, 30]:
        aa, bb = a[:, :n].mean(axis=(0, 1)), b[:, :n].mean(axis=(0, 1))
        horizons[str(n)] = {'model_mae_by_dim': aa.tolist(), 'persistence_mae_by_dim': bb.tolist(),
                            'position_mae_m': float(aa[:3].mean()), 'persistence_position_mae_m': float(bb[:3].mean()),
                            'all10_mae': float(aa.mean()), 'persistence_all10_mae': float(bb.mean())}
    full = horizons['30']
    passed = (full['position_mae_m'] < full['persistence_position_mae_m'] and
              full['all10_mae'] < full['persistence_all10_mae'])
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'complete', 'n_samples': len(samples), 'n_tasks': 1 if args.smoke else 10, 'mode': args.mode, 'smoke': args.smoke,
              'checkpoint': args.checkpoint, 'horizons': horizons, 'G0_prime_pass': bool(passed),
              'criterion': 'Full 30-step active-arm xyz MAE and all10 MAE both below persistence',
              'initialization': stats, 'samples': per_sample,
              'note': 'Teacher-forced observations/proprio; passing does not establish closed-loop success.'}
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'G0_prime_pass': bool(passed), 'horizon30': full}), flush=True)


if __name__ == '__main__': main()
