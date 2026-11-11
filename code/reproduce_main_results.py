"""Recompute calendar-window results from the bundled daily returns."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from market.perf import nw_pvalue
from market.dsr import dsr

ROOT = Path(__file__).resolve().parents[1]
SEG = {
    '24.06~08': '2024-06-01', '24.12~02': '2024-12-01',
    '25.06~08': '2025-06-01', '25.12~02': '2025-12-01',
    'hold 26.03~05': '2026-03-01',
}


def sharpe(r):
    if r.isna().any() or not np.isfinite(r.to_numpy()).all() or r.std(ddof=1) <= 0:
        raise ValueError('missing, nonfinite or constant daily returns')
    return float(r.mean() / r.std(ddof=1) * np.sqrt(365))


def metrics(r):
    eq = (1 + r).cumprod()
    return {'sharpe': sharpe(r), 'return_pct': float((eq.iloc[-1] - 1) * 100),
            'mdd': float((eq / eq.cummax() - 1).min() * 100)}


def check(actual, expected, label, atol=1e-8):
    if not np.isclose(actual, expected, atol=atol, rtol=1e-10):
        raise AssertionError(f'{label}: computed={actual}, stored={expected}')


def main():
    manifest = json.loads((ROOT / 'results/checksums.json').read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f'input checksum mismatch: {name}')

    daily = pd.read_csv(ROOT / 'results/calendar/daily_returns.csv', index_col=0, parse_dates=True)
    indexes = [pd.date_range(a, pd.Timestamp(a) + pd.DateOffset(months=3), inclusive='left')
               for a in SEG.values()]
    expected_index = indexes[0].append(indexes[1:])
    if not daily.index.equals(expected_index):
        raise AssertionError('daily dates differ from the five fixed calendar windows')
    target = pd.read_csv(ROOT / 'results/calendar/performance.csv').set_index('variant')
    if set(daily.columns) != set(target.index):
        raise AssertionError('daily-return models and performance rows differ')
    rows = []
    for name in daily:
        r = daily.loc[daily.index < '2026-03-01', name]
        h = daily.loc[daily.index >= '2026-03-01', name]
        if len(r) != 364 or len(h) != 92:
            raise AssertionError('wrong evaluation length')
        result = {'variant': name}
        for seg, idx in zip(SEG, indexes):
            result[seg] = sharpe(daily.loc[idx, name])
        result.update({f'pooled_{k}': v for k, v in metrics(r).items()})
        hold = metrics(h)
        result.update(hold_return_pct=hold['return_pct'], hold_mdd=hold['mdd'],
                      pooled_nw_t=nw_pvalue(r.to_numpy(), 14)[0], days=len(r), hold_days=len(h))
        for k, v in result.items():
            if k != 'variant':
                check(v, target.at[name, k], f'{name}/{k}')
        rows.append(result)
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'performance.csv', index=False)

    extra = json.loads((ROOT / 'results/calendar/extra.json').read_text())
    additional = pd.read_csv(ROOT / 'data/extra_daily_returns.csv', index_col=0, parse_dates=True)
    if not additional.index.equals(expected_index[:364]):
        raise AssertionError('additional models use different evaluation dates')
    for name in additional:
        v = metrics(additional[name])
        for k, src in [('sharpe', 'sharpe'), ('return_pct', 'ret'), ('mdd', 'mdd')]:
            check(round(v[k], 2), extra[name][src], f'{name}/{k}')
        check(len(additional), extra[name]['days'], f'{name}/days')
    for name in daily:
        check(round(daily.loc[:'2026-02-28', name].std() * np.sqrt(365) * 100, 1),
              extra['ann_vol_pct'][name], f'{name}/annual volatility')

    candle = json.loads((ROOT / 'results/tables/candle.json').read_text())['candle']
    z = np.array([[r['sz'][k] for k in 'ABCD'] for r in candle])
    if z.shape != (25, 4) or not np.isfinite(z).all():
        raise AssertionError('invalid candle result grid')
    exported = pd.read_csv(ROOT / 'results/tables/candle_abcd.csv')
    np.testing.assert_allclose(z, exported[list('ABCD')], atol=1e-15)
    summary = {'negative_cells': int((z < 0).sum()), 'min': round(float(z.min()), 2),
               'max': round(float(z.max()), 2), 'mean': round(float(z.mean()), 2),
               'all_positive_settings': int((z > 0).all(axis=1).sum()),
               'accepted_settings': int(((z >= .2).sum(axis=1) >= 3).sum())}
    expected = {'negative_cells': 49, 'min': -.23, 'max': .29, 'mean': -.01,
                'all_positive_settings': 1, 'accepted_settings': 0}
    if summary != expected:
        raise AssertionError((summary, expected))
    (out / 'candle_summary.json').write_text(json.dumps(summary, indent=2))

    inverse = pd.read_csv(ROOT / 'results/calendar/inverse.csv')
    for _, row in inverse[inverse['방향'] == '원본'].iterrows():
        name = row['조건']
        for key, seg in zip('ABCD', list(SEG)[:4]):
            check(row[key], target.at[name, seg], f'{name}/inverse-original/{key}')
        check(row['hold'], target.at[name, 'hold 26.03~05'], f'{name}/inverse-original/hold')
        check(row['합산'], target.at[name, 'pooled_sharpe'], f'{name}/inverse-original/pooled')
        check(row['누적%'], target.at[name, 'pooled_return_pct'], f'{name}/inverse-original/return')

    v = json.loads((ROOT / 'results/tables/dsr_inputs.json').read_text())['v_pool_daily']
    r = daily.loc[:'2026-02-28', 'CNN']
    corrected = {str(n): dsr(r, v, n) for n in (142, 229)}
    (out / 'dsr.json').write_text(json.dumps(corrected, indent=2))
    print(pd.DataFrame(rows)[['variant', 'pooled_sharpe', 'pooled_return_pct', 'pooled_mdd']].round(2).to_string(index=False))
    print('Verified: 8 performance rows, 3 additional models, annual volatility, candle grid and input checksums.')
    print('Candle:', summary, 'DSR:', corrected)


if __name__ == '__main__':
    main()
