#!/usr/bin/env python
"""X-VLA policy server (official /act protocol) for CCE gate runs.

Same request/response contract as upstream `deploy.py` + `XVLA._build_app`, but
with an explicit LoRA merge, a readiness endpoint, and no `info.json` trap.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

import xvla_bridge as XB


def build_app(model, processor, stats: dict):
    import cv2
    import json_numpy
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from PIL import Image

    app = FastAPI()
    lock = threading.Lock()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    @app.get("/ready")
    def ready():
        return JSONResponse({"ready": True, **stats})

    @app.post("/act")
    def act(payload: Dict[str, Any]):
        try:
            images = []
            for key in ("image0", "image1", "image2"):
                if key not in payload:
                    continue
                v = json_numpy.loads(payload[key])
                if isinstance(v, np.ndarray):
                    if v.ndim == 1:
                        v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                    images.append(Image.fromarray(np.ascontiguousarray(v)))
                elif isinstance(v, (list, tuple)):
                    images.append(Image.fromarray(np.array(v)))
                elif isinstance(v, str):
                    images.append(Image.open(v))
            if not images:
                return JSONResponse({"error": "No valid images found."}, status_code=400)

            inputs = processor(images, payload["language_instruction"])
            proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))
            domain_id = torch.tensor([int(payload["domain_id"])], dtype=torch.long)

            def to_model(t):
                if not isinstance(t, torch.Tensor):
                    t = torch.as_tensor(t)
                return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

            inputs = {k: to_model(v) for k, v in inputs.items()}
            inputs["proprio"] = to_model(proprio.unsqueeze(0))
            inputs["domain_id"] = domain_id.to(device)
            steps = int(payload.get("steps", 10))
            with lock:
                if 'seed' in payload:
                    seed = int(payload['seed'])
                    with torch.random.fork_rng(devices=[device.index or 0]):
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        action = model.generate_actions(**inputs, steps=steps)
                else:
                    action = model.generate_actions(**inputs, steps=steps)
            action = action.squeeze(0).float().cpu().numpy()
            stats["calls"] = stats.get("calls", 0) + 1
            return JSONResponse({"action": action.tolist()})
        except Exception:
            logging.error(traceback.format_exc())
            return JSONResponse({"error": "Request failed"}, status_code=400)

    return app


def main() -> None:
    ap = argparse.ArgumentParser("X-VLA policy server")
    ap.add_argument("--model-path", default=str(XB.XVLA_PT))
    ap.add_argument("--lora-path", default=None)
    ap.add_argument("--no-merge-lora", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--info-json", default=None)
    args = ap.parse_args()

    print(f"[serve] torch={torch.__version__} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    model, processor, info = XB.load_xvla(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device=args.device,
        merge_lora=not args.no_merge_lora,
    )
    raw = XB.unwrap_xvla(model)
    stats = {
        **info,
        "supports_request_seed": True,
        "num_actions": int(raw.num_actions),
        "action_mode": str(raw.action_mode),
        "num_domains": int(raw.transformer.soft_prompt_hub.num_embeddings)
        if hasattr(raw.transformer.soft_prompt_hub, "num_embeddings")
        else None,
        "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
    }
    print("[serve] " + json.dumps(stats), flush=True)

    if args.info_json:
        p = Path(args.info_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"host": args.host, "port": args.port, **stats}, indent=2) + "\n")

    import uvicorn

    app = build_app(model, processor, stats)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
