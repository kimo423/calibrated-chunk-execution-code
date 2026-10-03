"""Record oracle rollouts without changing the frozen execution loop."""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import cv2
import h5py
import numpy as np
from cce import libero_family_v3 as F
from cce.freeze_v4r1 import validate_freeze, sha

PARENT = 'cce/results/libero_family_spec_v4r1_fallback.json'


class RecordingEnv:
    def __init__(self, raw, obs, processor, flip):
        self.raw, self.env = raw, raw.env
        self.processor, self.flip = processor, flip
        self.frames = [[], []]
        self.poses, self.actions, self.joints, self.openings = [], [], [], []
        self.commands = []
        self.capture(obs, -1.)

    def capture(self, obs, grip):
        self.poses.append(F.measured_pose7(self.raw))
        c = self.env.robots[0].controller
        self.actions.append(np.r_[np.array(c.ee_pos).copy(), self.processor.Mat_to_Rotate6D(c.ee_ori_mat), grip])
        self.joints.append(np.asarray(obs['robot0_joint_pos']).copy())
        self.openings.append(F.gripper_opening(obs))
        for images, rgb in zip(self.frames, [self.flip(obs['agentview_image']), obs['robot0_eye_in_hand_image']]):
            # The upstream decoder returns cv2 channels directly to PIL, without
            # BGR->RGB conversion. Encode/decode without swapping preserves the
            # exact array seen by the policy, verified by the dataset audit.
            ok, encoded = cv2.imencode('.png', np.ascontiguousarray(rgb), [cv2.IMWRITE_PNG_COMPRESSION, 3])
            assert ok
            assert np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), rgb)
            images.append(encoded)

    def step(self, action):
        obs, reward, done, info = self.raw.step(action)
        self.commands.append(np.array(action).copy())
        self.capture(obs, float(action[-1]))
        return obs, reward, done, info

    def close(self):
        self.raw.close()


class RecordingEval:
    def __init__(self, evaluator, flip):
        self.evaluator, self.processor, self.flip = evaluator, evaluator.processor, flip
        self.record = None

    def _init_env(self, suite, task_id, ep):
        raw, language, obs = self.evaluator._init_env(suite, task_id, ep)
        self.record = RecordingEnv(raw, obs, self.processor, self.flip)
        return self.record, language, obs


def save_episode(path, record, row):
    assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'x') as f:
        f.create_dataset('abs_action_6d', data=np.asarray(record.actions, dtype=np.float64))
        f.create_dataset('achieved_pose7', data=np.asarray(record.poses))
        f.create_dataset('joint_pos', data=np.asarray(record.joints))
        f.create_dataset('gripper_opening', data=np.asarray(record.openings))
        f.create_dataset('applied_abs_command', data=np.asarray(record.commands))
        f.create_dataset('language_instruction', data=row['task'], dtype=h5py.string_dtype('utf-8'))
        for key, frames in zip(['agentview', 'eye_in_hand'], record.frames):
            # After the official img[1:] operation, frame index i is obs_i.
            padded = [frames[0]] + frames
            ds = f.create_dataset(key, shape=(len(padded),), dtype=h5py.vlen_dtype(np.dtype('uint8')))
            for i, frame in enumerate(padded):
                ds[i] = frame
        f.attrs.update(physical_hz=20., handler_index_hz=30., handler_query_duration_s=1.,
                       image_padding=1, task_id=row['task_id'], initial_state_index=row['ep'],
                       plant=row['pid'], success=int(row['success']), initial_state_sha256=row['initial_state_sha256'],
                       pose_label='achieved controller EE state; row 0 initial state, row t+1 after action t',
                       grip_label='applied hard command from preceding transition; initial row open',
                       source_sha256=sha(__file__))


def configure():
    spec = validate_freeze(PARENT)
    F.PACK = json.loads(Path(spec['pack']).read_text())
    with np.load('data/cce/libero/probes/probe_libero_panda_L00_nominal.npz') as z:
        F.GATE_REFERENCE = z['targets']
    F.GATE = F.gate_values(F.PACK, F.GATE_REFERENCE); F.RHO = spec['rho']
    assert F.GATE['H0'] == spec['H0']
    hats = json.loads(Path('data/cce/libero/theta/theta_hat_libero_panda_v2.json').read_text())
    hats.update(json.loads(Path('data/cce/takeover_v4_20260908/new_probes/probe_summary.json').read_text())['hats'])
    sys.path.insert(0, F.XVLA_LIBERO)
    import libero_client as LC
    ev = LC.LIBEROEval(task_suite_name='libero_spatial', eval_horizon=800, act_type='abs', num_episodes=20, init_seed=42)
    suite = LC.benchmark_dict['libero_spatial']()
    return spec, hats, LC, ev, suite


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--stage', choices=['dev', 'collect'], required=True)
    ap.add_argument('--freeze'); ap.add_argument('--plant', required=True)
    ap.add_argument('--port', type=int, required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--verify-replay', action='store_true')
    args = ap.parse_args()
    spec, hats, LC, ev, suite = configure()
    if args.stage == 'collect':
        from cce.freeze_demos_v4r1 import validate_demo_freeze
        fs = validate_demo_freeze(args.freeze)
        assert args.plant in fs['plants'] + [fs['fallback_plant']]
        tasks, candidates, target = fs['tasks'], fs['candidate_episodes'], fs['target_per_task']
    else:
        assert args.plant == 'L00_nominal'
        tasks, candidates, target = [0], [10], 1
    root = Path(args.out_dir); root.mkdir(parents=True, exist_ok=False)
    (root / 'traces').mkdir()
    F.STEP_DIR = root / 'traces'
    client = F.ChunkClient('127.0.0.1', args.port)
    client.suite_name = 'libero_spatial'; client.verify_repeat = args.stage == 'dev'
    import requests
    info = requests.get(f'http://127.0.0.1:{args.port}/ready', timeout=5).json()
    for k in ['model_path', 'lora_path', 'lora_merged', 'supports_request_seed', 'num_actions']:
        assert info[k] == spec['server'][k]
    metadata = {'stage': args.stage, 'args': vars(args), 'parent_sha256': sha(PARENT),
                'source_sha256': sha(__file__), 'server': info,
                'freeze_sha256': sha(args.freeze) if args.freeze else None,
                'candidate_initial_states': {}, 'eval_initial_state_indices': list(range(10))}
    for t in tasks:
        states = suite.get_task_init_states(t)
        assert len(states) > max(candidates), 'Would wrap initial-state index'
        state_hash = lambda e: hashlib.sha256(np.asarray(states[e]).tobytes()).hexdigest()
        eval_hashes = {state_hash(e) for e in range(10)}
        training = {str(e): state_hash(e) for e in candidates}
        assert not set(training.values()) & eval_hashes, 'Train/eval initial-state overlap'
        assert len(set(training.values())) == len(candidates), 'Duplicate training states'
        metadata['candidate_initial_states'][str(t)] = training
    (root / 'meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    recorder = RecordingEval(ev, LC._flip_agentview)
    rows, selected, counts = [], [], {}
    with (root / 'attempts.jsonl').open('x') as log:
        for task in tasks:
            counts[str(task)] = 0
            for ep in candidates:
                row = F.run_episode(recorder, suite, task, ep, F.BY_ID_V3[args.plant], 'oracle_v4', client,
                                    hats[args.plant], F.theta_from_hat(hats[F.NOMINAL_PID]), 800, 1., 1.)
                row['initial_state_sha256'] = metadata['candidate_initial_states'][str(task)][str(ep)]
                row['h5_path'] = None
                if row['success']:
                    path = root / 'episodes' / f'{args.plant}__t{task}__e{ep}.h5'
                    save_episode(path, recorder.record, row)
                    row['h5_path'] = str(path); row['h5_sha256'] = sha(path)
                    selected.append(str(path)); counts[str(task)] += 1
                rows.append(row); log.write(json.dumps(row) + '\n'); log.flush(); os.fsync(log.fileno())
                print(json.dumps({k: row[k] for k in ['pid', 'task_id', 'ep', 'success', 'steps', 'h5_path']}), flush=True)
                if counts[str(task)] == target:
                    break
    if args.verify_replay:
        assert args.stage == 'dev'
        (root / 'replay').mkdir(); F.STEP_DIR = root / 'replay'
        repeated = F.run_episode(ev, suite, 0, 10, F.BY_ID_V3[args.plant], 'oracle_v4', client,
                                hats[args.plant], F.theta_from_hat(hats[F.NOMINAL_PID]), 800, 1., 1.)
        a = np.load(root / 'traces' / f'{args.plant}__oracle_v4__t0__e10.npz')['steps']
        b = np.load(root / 'replay' / f'{args.plant}__oracle_v4__t0__e10.npz')['steps']
        assert np.array_equal(a, b), 'Recording changed rollout'
        assert repeated['success'] == rows[0]['success']
    meta = {'dataset_name': 'libero', 'robot_type': 'libero', 'datalist': selected,
            'observation_key': ['agentview', 'eye_in_hand'], 'language_instruction_key': 'language_instruction'}
    (root / 'dataset_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    outcome = {'status': 'complete' if all(v == target for v in counts.values()) else 'insufficient_successes',
               'plant': args.plant, 'attempts': len(rows), 'success_demos': len(selected), 'per_task': counts,
               'target_per_task': target, 'recording_replay_exact': bool(args.verify_replay),
               'training_authorized_by_data_gate': args.stage == 'collect' and all(v == target for v in counts.values())}
    (root / 'collection.json').write_text(json.dumps(outcome, indent=2) + '\n')
    print(json.dumps(outcome), flush=True)


if __name__ == '__main__':
    main()
