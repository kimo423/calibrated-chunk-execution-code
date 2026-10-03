"""Corrected W4 trace analysis, spectra, matched saturation and gate table."""
import _bootstrap  # noqa: F401
import json
from pathlib import Path
import numpy as np
from scipy.signal import periodogram
from cce.freeze_v4r1 import validate_freeze, sha
from cce.bench_relative_v4r1 import gate_values
from cce.libero_exec_v4r1 import RelativeTheta, RelativeExecutor

ROOT = Path('/opt/cce/data/cce/v4r1_20260908/fallback_main')
DEST = Path('/opt/cce/data/cce/mechanism_w4r2_20260911')
LABELS = ['y:7','intent:7','u:6','applied:6','y_next:7','g_cmd:1','g_applied:1','opening:1']


def read(p):
    return json.loads(Path(p).read_text())


def split(a, labels):
    assert labels == LABELS and a.ndim == 2 and a.shape[1] == 36 and len(a) > 0
    assert np.isfinite(a).all()
    out = {}; start = 0
    for item in labels:
        name, width = item.split(':'); width = int(width)
        out[name] = a[:, start:start+width]; start += width
    return out


def metrics(a, labels):
    x = split(a, labels)
    command = .1*x['u'][:, :3]
    injected = .1*x['applied'][:, :3]
    achieved = x['y_next'][:, :3]-x['y'][:, :3]
    residuals = {'injected_minus_command_m': injected-command,
                 'achieved_minus_command_m': achieved-command}
    hit = (np.max(abs(x['u'][:, :3]), axis=1) >= 1-1e-9) | (np.linalg.norm(x['u'][:, 3:], axis=1) >= 1-1e-9)
    quarters = np.minimum(3, 4*np.arange(len(a))//len(a))
    bins = {}
    for b in range(4):
        v = hit[quarters == b]
        bins[str(b)] = {'n_steps': len(v), 'n_bound_hits': int(v.sum()), 'share': float(v.mean()) if len(v) else None}
    clock_bins = {}
    for b in np.unique(np.arange(len(a))//100):
        v = hit[np.arange(len(a))//100 == b]
        clock_bins[str(int(b)*5)] = {'n_steps': len(v), 'n_bound_hits': int(v.sum()), 'share': float(v.mean())}
    out = {'steps': len(a), 'n_bound_hits': int(hit.sum()), 'bound_share': float(hit.mean()),
           'quarters': bins, 'absolute_5s_bins': clock_bins,
           'intent_rmse_m': float(np.sqrt(np.mean(np.sum((x['y_next'][:, :3]-x['intent'][:, :3])**2, axis=1)))),
           'residuals': {}}
    spectra = {}
    for key, v in residuals.items():
        f, psd = periodogram(v, fs=20., window='boxcar', detrend=False, axis=0, scaling='density')
        energy = float(np.sum(psd)*(20./len(v)))
        actual = float(np.mean(np.sum(v*v, axis=1)))
        assert np.isclose(energy, actual, rtol=1e-10, atol=1e-14), 'PSD Parseval audit failed'
        norm = np.linalg.norm(v, axis=1)
        out['residuals'][key] = {'rms_m': actual**.5, 'mean_norm_m': float(norm.mean()),
                                'p95_norm_m': float(np.quantile(norm, .95)),
                                'spectral_energy_m2': energy,
                                'energy_above_2hz_fraction': float(psd[f > 2].sum()/psd.sum()) if psd.sum() else None}
        spectra[key+'_psd_m2_per_hz'] = psd
    spectra['frequency_hz'] = f
    return out, spectra


def selftest():
    a = np.zeros((100,36)); a[:,14] = .5; a[:,20] = .25; a[:,26] = .025
    m, _ = metrics(a, LABELS)
    assert np.isclose(m['residuals']['injected_minus_command_m']['rms_m'], .025)
    assert np.isclose(m['residuals']['achieved_minus_command_m']['rms_m'], .025)
    a[:,20] = .5; a[:,26] = .05
    m, _ = metrics(a, LABELS)
    assert m['residuals']['injected_minus_command_m']['rms_m'] == 0
    assert m['residuals']['achieved_minus_command_m']['rms_m'] == 0
    a[10,14] = 1.; a[55,17] = 1.
    m, _ = metrics(a, LABELS)
    assert m['n_bound_hits'] == 2 and m['quarters']['0']['n_bound_hits'] == 1 and m['quarters']['2']['n_bound_hits'] == 1
    return 'passed'


def main():
    spec = validate_freeze('cce/results/libero_family_spec_v4r1_fallback.json')
    summary = read('cce/results/libero_fallback_v4r1.json')
    rows = []; inputs = {}
    for p, h in summary['raw_sha256'].items():
        assert sha(p) == h
        rows.extend(map(json.loads, Path(p).read_text().splitlines())); inputs[p] = h
    lookup = {(r['pid'],r['method'],r['task_id'],r['ep']):r for r in rows}
    assert len(lookup) == len(rows) == 15200
    DEST.mkdir(exist_ok=True); (DEST/'spectra').mkdir(exist_ok=True)
    out = {'status':'passed','selftest':selftest(),'supersedes':'cce/results/CCE_mechanism_v4r1.json',
           'correction':'Old script used y_next quaternion columns as applied command and omitted 0.1m normalization scale; old residual values invalid. Old boundary counts unaffected.',
           'trace_population':'task0 ep0 only, one per plant/method; descriptive temporal and spectral diagnosis, not whole dataset trajectories',
           'traces':{},'matched5_cells':{},'gate':{},'inputs_sha256':inputs}
    paths = sorted((ROOT/'jobs').glob('*_steps/*.npz'))
    assert len(paths) == 19*9
    for p in paths:
        plant, method, t, e = p.stem.split('__'); assert t == 't0' and e == 'e0'
        with np.load(p) as z: m, spectra = metrics(z['steps'], z['columns'].tolist())
        r = lookup[plant,method,0,0]
        assert m['steps'] == r['steps'] and m['n_bound_hits'] == r['n_bound_hits']
        assert abs(m['intent_rmse_m']-r['tracking_rmse_m']) < 1e-12
        dest = DEST/'spectra'/p.name; np.savez_compressed(dest, **spectra)
        m.update(spectrum_path=str(dest),spectrum_sha256=sha(dest),source_sha256=sha(p))
        out['traces'].setdefault(plant,{})[method] = m
    for plant in spec['plant_ids']:
        out['matched5_cells'][plant] = {}
        for method in spec['episodes']:
            rr = [lookup[plant,method,t,e] for t in range(10) for e in range(5)]
            out['matched5_cells'][plant][method] = {'n':50,'sr':sum(r['success'] for r in rr)/50,
                'intent_rmse_m':float(np.mean([r['tracking_rmse_m'] for r in rr])),
                'bound_share':sum(r['n_bound_hits'] for r in rr)/sum(r['steps'] for r in rr)}
    pack = read(spec['pack'])
    with np.load('data/cce/libero/probes/probe_libero_panda_L00_nominal.npz') as rec:
        gate = gate_values(pack, rec['targets'])
    assert gate['H0'] == spec['H0']
    for p in spec['plant_ids']:
        th = RelativeTheta.from_fit(pack['plants'][p])
        ex = RelativeExecutor(th,pack['nominal_alpha'],pack['nominal_gain'],rho_max=spec['rho'],h=gate['H'][p],h0=gate['H0'])
        out['gate'][p] = {'H_m':gate['H'][p], 'H0_m':gate['H0'],'arm_bypass':ex.bypass,
                          'grip_lead_steps':ex.lead,'identifiable_axes':int(ex.mask.sum()),
                          'role':'dev' if p in spec['dev'] else 'test'}
    out['matched5_test_macro'] = {m:{key:float(np.mean([out['matched5_cells'][p][m][key] for p in spec['test']]))
                                       for key in ['sr','intent_rmse_m','bound_share']} for m in spec['episodes']}
    target=Path('cce/results/CCE_mechanism_w4r2.json'); target.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# W4机制与门控：修正后的分析','',out['correction'],'',
           '171条轨迹均为每个plant×方法的task0/ep0；逐条核对步数、边界命中与意图RMSE，和原始逐集记录一致；频谱积分与时域均方误差满足Parseval检查。','',
           '残差分别定义为注入后命令减提交命令、实际位移减提交命令，平移归一化命令乘0.1转换为米。PSD采用20Hz矩形窗、不去均值的单边periodogram，保留DC；频谱单位m²/Hz。','',
           '## 全测试条件匹配前5初态','', '| 方法 | 宏SR | 宏意图RMSE(mm) | plant等权边界命中步比例 |','|---|---:|---:|---:|']
    for m,v in out['matched5_test_macro'].items(): lines.append(f"| {m} | {v['sr']:.2%} | {1000*v['intent_rmse_m']:.3f} | {v['bound_share']:.2%} |")
    lines += ['', '不同臂成功终止时间不同，边界比例分母与轨迹长度不同；不能仅从比例下降推断失败的因果。54.3%历史值使用旧人群与旧实现，不能直接作此表基准。','',
              '## 19条件失配门','', '| plant | 角色 | H(mm) | H0(mm) | 机械臂回退naive | 夹爪提前(拍) |','|---|---|---:|---:|---|---:|']
    for p,v in out['gate'].items():lines.append(f"| {p} | {v['role']} | {1000*v['H_m']:.3f} | {1000*v['H0_m']:.3f} | {v['arm_bypass']} | {v['grip_lead_steps']} |")
    lines += ['', '机械臂回退与夹爪提前独立，H是位置模型量，不能反映纯夹爪失配。完整0–40秒按5秒分箱、归一化四段时间分布、频谱及SHA见JSON和/mnt/sda谱文件。','',
              '## 剩余饱和位置（task0/ep0诊断样本）','', '| plant / r1 | 总步数 | 边界比例 | 前/中前/中后/后四段比例 |','|---|---:|---:|---|']
    for p in spec['plant_ids']:
        v=out['traces'][p]['ours_v4']; q=' / '.join(f"{v['quarters'][str(k)]['share']:.1%}" for k in range(4))
        lines.append(f"| {p} | {v['steps']} | {v['bound_share']:.2%} | {q} |")
    Path('docs/reports/CCE_mechanism_w4r2.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'selftest':out['selftest'],'traces':len(paths),'gates':len(out['gate']),'macro':out['matched5_test_macro']},indent=2))


if __name__ == '__main__':main()
