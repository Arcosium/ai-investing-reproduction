"""Haiku 점수 부호를 뒤집어(p 의 상승·하락 열 교환) 같은 코호트 규칙·비용으로 돌린 성과 → evaluation_cal/inverse.csv. python3 inverse_cal.py"""
from pathlib import Path
import os, sys, tempfile
import numpy as np, pandas as pd
D = Path(os.environ['RESEARCH_DATA']).resolve() / 'calendar'; H = Path(os.environ['RESEARCH_DATA']).resolve()
sys.path.insert(0, str(H)); os.environ['MARKET_BARS'] = str(H / 'work/bars_hold')
from market import cohort
from market.perf import daily_returns
cohort.B.OUT = str(H / 'work/bars_hold')
SEG = {'A': ('2406_08', '2024-06-01'), 'B': ('2412_02', '2024-12-01'), 'C': ('2506_08', '2025-06-01'), 'D': ('2512_02', '2025-12-01'), 'hold': ('2603_05', '2026-03-01')}
sh = lambda r: float(r.mean() / r.std(ddof=1) * np.sqrt(365))
tmp = Path(tempfile.mkdtemp()); rows = []
for name in ('LLM_chart', 'LLM_context', 'LLM_heat', 'rule_zW'):
    for sign in ('원본', '반대'):
        d = {}
        for s, (t, a) in SEG.items():
            f = D / f'evaluation/{name}_{t}_pred.npz'
            if sign == '반대':
                z = dict(np.load(f, allow_pickle=True)); z['p'] = z['p'][:, [0, 2, 1]]; g = tmp / f'{name}_{t}.npz'; np.savez(g, **z); f = g
            r = cohort.run(str(f), '4h', 84, q=.1, cost=.001)[0].sort_index(); a = pd.Timestamp(a)
            d[s] = r[(r.index >= a) & (r.index < a + pd.DateOffset(months=3))]
        pooled = pd.concat([d[s] for s in 'ABCD']); eq = (1 + pooled).cumprod()
        rows.append({'조건': name, '방향': sign, **{s: sh(d[s]) for s in SEG}, '합산': sh(pooled), '누적%': (eq.iloc[-1] - 1) * 100})
o = pd.DataFrame(rows); o.to_csv(D / 'evaluation_cal/inverse.csv', index=False); print(o.round(2).to_string(index=False))
