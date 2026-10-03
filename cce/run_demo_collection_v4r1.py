"""Finite three-plant acquisition with one prespecified MX03 fallback."""
import _bootstrap  # noqa: F401
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import subprocess
import threading
from cce.freeze_demos_v4r1 import validate_demo_freeze, sha


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--freeze', required=True); ap.add_argument('--services', required=True)
    ap.add_argument('--out-dir', required=True); args = ap.parse_args()
    spec = validate_demo_freeze(args.freeze)
    subprocess.run(['git', 'ls-files', '--error-unmatch', args.freeze], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', args.freeze], check=True, stdout=subprocess.DEVNULL)
    root = Path(args.out_dir); root.mkdir(parents=True, exist_ok=False)
    services = json.loads(Path(args.services).read_text()); assert len(services) == 3
    (root / 'plan.json').write_text(json.dumps({'freeze': args.freeze, 'freeze_sha256': sha(args.freeze),
                                               'services': services, 'started': datetime.datetime.now().astimezone().isoformat(),
                                               'max_attempts': 400}, indent=2) + '\n')
    lock = threading.Lock(); outcomes = {}

    def one(plant, service):
        dest = root / plant
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(service['gpu']), MUJOCO_EGL_DEVICE_ID=str(service['gpu']),
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = ['bash', 'cce/env_libero.sh', 'cce/libero_collect_plant_demos_v4r1.py', '--stage', 'collect',
                   '--freeze', args.freeze, '--plant', plant, '--port', str(service['port']), '--out-dir', str(dest)]
        with (root / f'{plant}.log').open('x') as log:
            p = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (root / f'{plant}.exit').write_text(str(p.returncode) + '\n')
        if p.returncode:
            result = {'status': 'process_error', 'exit': p.returncode, 'plant': plant}
        else:
            result = json.loads((dest / 'collection.json').read_text())
            # Audit every available HDF5, including incomplete collections. An
            # incomplete but valid collection must never imply a training pass.
            if result['success_demos']:
                with (root / f'{plant}_handler.log').open('x') as log:
                    audit = subprocess.run(['bash', 'cce/env.sh', 'cce/audit_demo_handler_v4r1.py', '--demo-dir', str(dest)],
                                           env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
                result['handler_audit_exit'] = audit.returncode
                if audit.returncode:
                    result['status'] = 'handler_audit_error'
                    result['training_authorized_by_data_gate'] = False
        with lock:
            outcomes[plant] = result
            temp = root / 'status.tmp'
            temp.write_text(json.dumps({'plants': outcomes, 'completed_plants': len(outcomes)}, indent=2) + '\n')
            temp.replace(root / 'status.json')
        print(json.dumps(result), flush=True)
        return result

    def worker(plant, service):
        outcome = one(plant, service)
        if plant == 'MX02' and outcome['status'] == 'insufficient_successes':
            one(spec['fallback_plant'], service)

    code = 1
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker, p, s) for p, s in zip(spec['plants'], services)]
            for future in futures:
                future.result()
        errors = [p for p, r in outcomes.items() if r['status'] not in ('complete', 'insufficient_successes')]
        total_attempts = sum(r.get('attempts', 0) for r in outcomes.values())
        assert total_attempts <= 400
        result = {'status': 'execution_error' if errors else 'collection_finished', 'plants': outcomes,
                  'total_attempts': total_attempts, 'freeze_sha256': sha(args.freeze),
                  'training_complete': False, 'services_owned': False,
                  'note': 'Insufficient successes is a data-gate outcome, not evidence of ineffective fine-tuning.'}
        (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        lines = ['# CCE 策略侧演示采集结果', '', '| plant | 尝试 | 成功演示 | 状态 |', '|---|---:|---:|---|']
        for p, r in outcomes.items():
            lines.append(f"| {p} | {r.get('attempts', '未知')} | {r.get('success_demos', '未知')} | {r['status']} |")
        lines += ['', '每任务目标5成功；不足数据的plant不进入完整50条微调条件。尚未训练，不推断策略侧不可补偿。', '']
        (root / 'report.md').write_text('\n'.join(lines))
        code = 1 if errors else 0
    except BaseException as error:
        (root / 'failure.txt').write_text(repr(error) + '\n')
        raise
    finally:
        (root / 'supervisor.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    main()
