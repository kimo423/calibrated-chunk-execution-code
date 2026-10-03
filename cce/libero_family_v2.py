#!/usr/bin/env python
"""Second-pass (prereg v3 amendment) driver for the LIBERO CCE plant family.

The closed loop itself is imported verbatim from ``cce/libero_family.py``
(``run_episode`` / ``ChunkClient``); nothing in the first-pass files is modified,
so ``cce/libero_family.py`` still reproduces the first pass at its own defaults.
This driver only changes what prereg v3 changed:

* the executor command bound defaults to **1.0** (0.1 m / 0.1 rad per 50 ms step)
  -- ``cce/executor.py``'s native working point, confirmed on the two development
  plants before the confirmatory run (``--set dev``);
* the plant pool is ``FAMILY_V2`` (the frozen first-pass 12 plus two
  over-responding plants), with the development / test split of prereg v3 item 1;
* ``--theta-file`` lets the run read the merged second-pass theta-hat file without
  overwriting the first-pass one.

Everything else -- arms, tasks, episodes, horizon, init seed, chunk protocol,
policy server, sharding -- is unchanged.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time
from pathlib import Path

from cce.common import json_default
from cce.libero_exec import METHODS, theta_from_hat
from cce.libero_family import ChunkClient, run_episode
from cce.libero_plant import NOMINAL_PID, SCREEN
from cce.libero_plant_v2 import DEV_SPECS, FAMILY_V2, TEST_SPECS
from cce.libero_probe import DATA, UID

XVLA_LIBERO = "/opt/CalibrationToPrompt/source/X-VLA/evaluation/libero"

# prereg v3 item 1: the executor command bound is fixed at the sim-side working point.
BOUND_POS_V2 = 1.0
BOUND_ROT_V2 = 1.0
HORIZON = 800


def main() -> None:
    ap = argparse.ArgumentParser("CCE LIBERO plant-family closed loop (prereg v3, pass 2)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--set", default="family_v2", choices=["family_v2", "dev", "test"])
    ap.add_argument("--plants", default="", help="csv override")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--init-seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--theta-file", default=str(DATA / "theta" / f"theta_hat_{UID}_v2.json"))
    ap.add_argument("--bound-pos", type=float, default=BOUND_POS_V2)
    ap.add_argument("--bound-rot", type=float, default=BOUND_ROT_V2)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool = {}
    for s in list(FAMILY_V2) + list(SCREEN):
        pool.setdefault(s.pid, s)
    if args.plants:
        specs = [pool[p] for p in args.plants.split(",") if p]
    elif args.set == "dev":
        specs = list(DEV_SPECS)
    elif args.set == "test":
        specs = list(TEST_SPECS)
    else:
        specs = list(FAMILY_V2)
    methods = [m for m in args.methods.split(",") if m]
    tasks = [int(x) for x in args.tasks.split(",") if x.strip() != ""]

    theta_path = Path(args.theta_file)
    hats = json.loads(theta_path.read_text())
    missing = [s.pid for s in specs if s.pid not in hats]
    if missing:
        raise SystemExit(f"no theta-hat for {missing} in {theta_path}")
    theta_n = theta_from_hat(hats[NOMINAL_PID])

    sys.path.insert(0, XVLA_LIBERO)
    import libero_client as LC

    ev = LC.LIBEROEval(task_suite_name=args.suite, eval_horizon=args.horizon, act_type="abs",
                       num_episodes=args.episodes, init_seed=args.init_seed)
    ev.base_dir = Path(args.out).parent
    ev.base_dir.mkdir(parents=True, exist_ok=True)
    ev._log_results = lambda metrics: None
    suite = [LC.benchmark_dict[t]() for t in LC.LIBERO_DATASETS[args.suite]][0]

    client = ChunkClient(args.host, args.port)

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["pid"], r["method"], r["task_id"], r["ep"]))
            except Exception:
                pass

    work = []
    idx = 0
    for sp in specs:
        for m in methods:
            for tid in tasks:
                for ep in range(args.episodes):
                    if idx % args.nshards == args.shard:
                        work.append((sp, m, tid, ep))
                    idx += 1

    fh = out_path.open("a", encoding="utf-8")
    t0 = time.time()
    n_run = 0
    for sp, m, tid, ep in work:
        if (sp.pid, m, tid, ep) in done:
            continue
        rec = run_episode(ev, suite, tid, ep, sp, m, client, hats.get(sp.pid), theta_n,
                          args.horizon, args.bound_pos, args.bound_rot)
        rec["suite"] = args.suite
        rec["shard"] = args.shard
        rec["bound_pos"] = args.bound_pos
        rec["bound_rot"] = args.bound_rot
        rec["pass"] = args.tag
        rec["theta_file"] = theta_path.name
        fh.write(json.dumps(rec, ensure_ascii=False, default=json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        n_run += 1
        print(json.dumps({k: rec[k] for k in ("pid", "method", "task_id", "ep", "success",
                                              "steps", "tracking_rmse_m", "n_bound_hits",
                                              "wall_s")}, default=json_default), flush=True)
    fh.close()
    print(json.dumps({"shard": args.shard, "n_run": n_run, "n_planned": len(work),
                      "bound_pos": args.bound_pos, "wall_s": round(time.time() - t0, 1)}),
          flush=True)


if __name__ == "__main__":
    main()
