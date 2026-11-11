"""수정안 보조 수치(달력 기준): 3개 결합 CNN·MLP m2 성과, 고유 종목 수, 연 변동성 → evaluation_cal/extra.json"""
import json, sys, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from market.perf import daily_returns
W = str(Path(os.environ['RESEARCH_DATA']).resolve() / 'work/results') + '/'
os.chdir(Path(os.environ['RESEARCH_DATA']).resolve() / 'calendar')
WIN = [(pd.Timestamp(a), pd.Timestamp(a) + pd.DateOffset(months=3)) for a in ('2024-06-01', '2024-12-01', '2025-06-01', '2025-12-01')]
def dev(j):
    r = daily_returns(W + j)[0].sort_index(); r = pd.concat([r[(r.index >= a) & (r.index < b)] for a, b in WIN]); eq = (1 + r).cumprod()
    return {'sharpe': round(float(r.mean() / r.std() * np.sqrt(365)), 2), 'ret': round(float(eq.iloc[-1] - 1) * 100, 2),
            'mdd': round(float((eq / eq.cummax() - 1).min() * 100), 2), 'days': len(r)}
out = {'cnn3': dev('wf8_abcd_ext/t21_1_ens_abcd_cohort_H84.json'), 'm2': dev('wf8_abcd/m2x3_dev_cohort_H84.json'), 'm1': dev('wf8_abcd/m1x3_dev_cohort_H84.json')}
b, n = set(), []
for t in ('2406_08', '2412_02', '2506_08', '2512_02'):
    z = np.load(f'evaluation/CNN_{t}_pred.npz', allow_pickle=True); s = pd.Series(z['base'].astype(str)); b |= set(s)
    n += pd.Series(z['ts']).value_counts().tolist()
out.update(unique_bases=len(b), per_ts_min=min(n), per_ts_max=max(n))
d = pd.read_csv('evaluation_cal/daily_returns.csv', index_col=0, parse_dates=True); d = d[d.index < '2026-03-01']
out['ann_vol_pct'] = (d.std() * np.sqrt(365) * 100).round(1).to_dict()
json.dump(out, open('evaluation_cal/extra.json', 'w'), indent=1, ensure_ascii=False); print(json.dumps(out, ensure_ascii=False))
