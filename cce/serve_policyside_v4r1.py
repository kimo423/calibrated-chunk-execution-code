"""Serve adapted policies with the unchanged seeded action endpoint."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
from cce.policyside_model_v4r1 import load_policy
from cce.serve_xvla_v4r1 import build_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['ft_demo', 'prompt_fit'], required=True)
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--port', required=True, type=int)
    args = ap.parse_args()
    model, raw, processor, stats = load_policy(args.mode, args.checkpoint, training=False)
    model.cuda().eval()
    stats.update(ready=True, supports_request_seed=True, num_actions=30,
                 checkpoint=str(Path(args.checkpoint).resolve()), policy_mode=args.mode)
    import uvicorn
    uvicorn.run(build_app(model, processor, stats), host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__': main()
