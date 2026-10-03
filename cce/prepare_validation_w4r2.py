"""Cache a fixed, task-balanced validation panel before training."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import xvla_bridge as XB


def main():
    XB.add_xvla_to_path()
    from datasets.dataset import InfiniteDataReader
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True)
    root = Path(ap.parse_args().root)
    assert not (root/'validation_samples.pt').exists()
    meta = json.loads((root/'validation_meta.json').read_text())
    samples, records = [], []
    for path in meta['datalist']:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=root) as f:
            json.dump({**meta, 'datalist': [path]}, f); f.flush()
            reader = InfiniteDataReader(f.name, num_actions=30, num_views=3, training=False, action_mode='ee6d')
            episode = list(reader)
        assert len(episode) >= 20
        indices = np.linspace(0, len(episode)-1, 20, dtype=int).tolist()
        samples.extend(episode[i] for i in indices)
        records.append({'episode': path, 'available_samples': len(episode), 'selected_ordinals': indices})
    assert len(samples) == 200
    torch.save(samples, root/'validation_samples.pt')
    (root/'validation_panel.json').write_text(json.dumps({'n_samples': 200, 'episodes': records}, indent=2)+'\n')
    print('Cached 200 validation samples, 20 per task')


if __name__ == '__main__': main()
