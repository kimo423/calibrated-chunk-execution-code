"""Read recorded demos through the unmodified upstream LIBERO training handler."""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
import xvla_bridge as XB


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--demo-dir', required=True); args = ap.parse_args()
    XB.add_xvla_to_path()
    from datasets.dataset import InfiniteDataReader
    from datasets.domain_handler.simulations import LiberoHandler
    from datasets.utils import decode_image_from_bytes
    from PIL import Image
    import inspect
    root = Path(args.demo_dir)
    meta = json.loads((root / 'dataset_meta.json').read_text())
    assert meta['datalist']
    checks = []
    for path in meta['datalist']:
        single = dict(meta, datalist=[path])
        single_path = root / 'handler_audit_meta.json'
        single_path.write_text(json.dumps(single) + '\n')
        reader = InfiniteDataReader(str(single_path), num_actions=30, num_views=3, training=False, action_mode='ee6d')
        sample = next(iter(reader))
        assert sample['domain_id'].item() == 3
        assert sample['action'].shape == (30, 20) and sample['proprio'].shape == (20,)
        assert sample['image_input'].shape == (3, 3, 224, 224)
        assert sample['image_mask'].tolist() == [True, True, False]
        assert torch.isfinite(sample['action']).all() and torch.isfinite(sample['image_input']).all()
        with h5py.File(path) as f:
            action = f['abs_action_6d'][()]
            action[:, 9] = action[:, 9] > 0
            n = len(action)
            assert len(f['agentview']) == n + 1 and len(f['eye_in_hand']) == n + 1
            assert len(f['applied_abs_command']) == n - 1
            assert np.array_equal(action[:, :3], f['achieved_pose7'][:, :3])
            i = int(np.argmin(np.max(np.abs(action[:, :9] - sample['proprio'].numpy()[:9]), axis=1)))
            assert np.max(np.abs(action[i, :9] - sample['proprio'].numpy()[:9])) < 1e-6
            # Verify the real handler's 30 Hz / 1 s index convention. In a full
            # window this gives exactly the next thirty 20 Hz control rows;
            # near the tail it compresses the remaining span, as upstream does.
            q = np.linspace(i / 30, min(i / 30 + 1., (n - 1) / 30), 31, dtype=np.float32)
            index = q * 30
            expected = np.stack([np.interp(index[1:], np.arange(n), action[:, k]) for k in range(10)], axis=-1)
            err = float(np.max(np.abs(sample['action'].numpy()[:, :10] - expected)))
            assert err < 2e-5 and torch.count_nonzero(sample['action'][:, 10:]) == 0
            image_errors = []
            for j, key in enumerate(['agentview', 'eye_in_hand']):
                encoded = f[key][i + 1]
                rgb = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                assert rgb.shape == (256, 256, 3)
                assert np.array_equal(np.array(decode_image_from_bytes(encoded)), rgb)
                transformed = reader.image_aug(Image.fromarray(rgb))
                ie = float((transformed - sample['image_input'][j]).abs().max())
                assert ie == 0.; image_errors.append(ie)
            checks.append({'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                           'initial_state_index': int(f.attrs['initial_state_index']), 'n_states': n,
                           'first_sample_index': i, 'action_max_abs_error': err,
                           'image_max_abs_errors': image_errors})
    out = {'status': 'passed', 'n_episodes': len(checks), 'checks': checks,
           'upstream_handler': inspect.getfile(LiberoHandler),
           'upstream_handler_sha256': hashlib.sha256(Path(inspect.getfile(LiberoHandler)).read_bytes()).hexdigest(),
           'physical_hz': 20, 'full_chunk_control_steps': 30,
           'note': 'The unmodified handler uses virtual 30 Hz and 1 s; physical chunk duration is 1.5 s. Tail interpolation is retained.'}
    (root / 'handler_audit.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
