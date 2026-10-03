"""Compare overresponse arms on the identical five initial states per task."""
import _bootstrap  # noqa: F401
import itertools
import json
from collections import Counter
from pathlib import Path
import numpy as np
from cce.freeze_v4r1 import sha, validate_freeze


def main():
    validate_freeze('cce/results/libero_family_spec_v4r1_fallback.json')
    source = json.loads(Path('cce/results/libero_fallback_v4r1.json').read_text())
    plants = ['OV2', 'D2_g130', 'MX_g130_d100']
    methods = ['naive', 'ours_v3', 'ours_v4', 'oracle_v4']
    rows, hashes = [], {}
    for path, digest in source['raw_sha256'].items():
        assert sha(path) == digest
        selected = [r for r in map(json.loads, Path(path).read_text().splitlines())
                    if r['pid'] in plants and r['method'] in methods and r['ep'] < 5]
        if selected:
            rows.extend(selected); hashes[path] = digest
    keys = Counter((r['pid'], r['method'], r['task_id'], r['ep']) for r in rows)
    assert set(keys) == set(itertools.product(plants, methods, range(10), range(5)))
    assert set(keys.values()) == {1}
    cells = {}
    for p in plants:
        cells[p] = {}
        for m in methods:
            rr = [r for r in rows if r['pid'] == p and r['method'] == m]
            cells[p][m] = dict(n=len(rr), sr=float(np.mean([r['success'] for r in rr])),
                              tracking_rmse_m=float(np.mean([r['tracking_rmse_m'] for r in rr])),
                              pooled_bound_share=sum(r['n_bound_hits'] for r in rr)/sum(r['steps'] for r in rr))
    out = dict(status='audited_matched_existing_data', n=len(rows), cells=cells,
               raw_sha256=hashes, note='Same task/ep0..4; descriptive; no newly completed rollouts. Earlier 50-vs-100 summary is superseded for comparisons.')
    Path('cce/results/CCE_overresponse_matched_v4r1.json').write_text(json.dumps(out, indent=2)+'\n')
    lines = ['# 过响应专项：相同初态比较', '',
             '三个条件，四臂，每臂10任务×相同ep0..4，600条已有记录；原始SHA和完整唯一键审计通过。不是新增闭环实验。', '',
             '| plant | 方法 | SR | 意图RMSE(mm) | 边界命中步比例 |', '|---|---|---:|---:|---:|']
    for p, cc in cells.items():
        for m, v in cc.items():
            lines.append(f"| {p} | {m} | {v['sr']:.2%} | {v['tracking_rmse_m']*1000:.3f} | {v['pooled_bound_share']:.2%} |")
    lines += ['', '旧摘要将ours_v3的50集与其他臂100集混列，本表统一初态后用于专项比较。r1与v3有多处实现差异，不能把两臂差值单独归因于单侧约束。', '',
              '2026-09-11补跑因样本数与主批冻结协议冲突在四分片入口失败，0集生成；空转服务已清理，失败日志原样保留。']
    Path('docs/reports/CCE_overresponse_matched_v4r1.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(cells, indent=2))


if __name__ == '__main__':
    main()
