"""DSR 재계산(수정안 제7장): 시도 분포·설정 수(142·229)는 MARKET results/dsr_inputs.json 그대로, 승자만 CNN 4분기(달력 기준) 일수익. python3 dsr_cal.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parents[2]
from market.dsr import dsr
v = json.load(open(H / 'results/tables/dsr_inputs.json'))['v_pool_daily']
r = pd.read_csv(H / 'results/calendar/daily_returns.csv', index_col=0, parse_dates=True).CNN.dropna()
r = r[r.index < '2026-03-01']
print(f"T {len(r)}일, 연율 Sharpe {r.mean()/r.std()*np.sqrt(365):.2f}")
for n in (142, 229): print(n, round(dsr(r, v, n), 4))
print('보정 없음', round(dsr(r, 0.0, 2), 4))
