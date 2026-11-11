"""달력 3개월 기준 r0 평가: python3 evaluate_cal.py → evaluation_cal/

research-data work/wf8/analyze_abcd.log 와 같은 기준으로 맞춘다. 장부를 돌린 뒤 분기 달력 창[시작, 끝)만 남겨서
마지막 판단 뒤 청산 꼬리(약 14일)를 빼고, 25.12~02 꼬리가 홀드아웃 3월로 넘어가 겹치는 일도 없앤다.
CNN·LightGBM·무작위(셔플 1·2)는 MARKET 코호트 결과(results/wf8_abcd)를 그대로 읽는다.
Haiku 세 조건·가격 위치 규칙은 evaluate_r0.py 가 만든 예측 파일(evaluation/*_pred.npz)을 같은 코호트에 넣고 같은 창으로 자른다.
점검: evaluation/CNN_*_pred.npz 를 구간별로 돌려 자른 값이 MARKET 연속 장부를 자른 값과 같아야 한다(구간별=연속의 근거).
"""
from pathlib import Path
import json, os, sys
import numpy as np, pandas as pd

D = Path(os.environ['RESEARCH_DATA']).resolve() / 'calendar'
H = Path(os.environ['RESEARCH_DATA']).resolve(); W = H / 'work/results/wf8_abcd'
sys.path.insert(0, str(H)); os.environ['MARKET_BARS'] = str(H / 'work/bars_hold')
from market import cohort
from market.perf import daily_returns, nw_pvalue
cohort.B.OUT = str(H / 'work/bars_hold')
EV, OUT = D / 'evaluation', D / 'evaluation_cal'; OUT.mkdir(exist_ok=True)
SEG = {'24.06~08': ('2406_08', '2024-06-01'), '24.12~02': ('2412_02', '2024-12-01'), '25.06~08': ('2506_08', '2025-06-01'),
       '25.12~02': ('2512_02', '2025-12-01'), 'hold 26.03~05': ('2603_05', '2026-03-01')}
WIN = {s: (pd.Timestamp(a), pd.Timestamp(a) + pd.DateOffset(months=3)) for s, (_, a) in SEG.items()}
EVAL = [s for s in SEG if not s.startswith('hold')]


def cut(r, seg):
    a, b = WIN[seg]; r = r.sort_index(); return r[(r.index >= a) & (r.index < b)]


def market(dev, hold):
    d = {s: cut(daily_returns(W / dev)[0], s) for s in EVAL}
    d['hold 26.03~05'] = cut(daily_returns(W / hold)[0], 'hold 26.03~05')
    return d


daily = {'CNN': market('cnn10_dev_cohort_H84.json', 'cnn10_q8_cohort_H84.json'),
         'GBM': market('gbm_dev_cohort_H84.json', 'gbm_q8_cohort_H84.json'),
         'random1': market('cnn10_dev_cohort_H84_shuf1.json', 'cnn10_q8_cohort_H84_shuf1.json'),
         'random2': market('cnn10_dev_cohort_H84_shuf2.json', 'cnn10_q8_cohort_H84_shuf2.json')}
for name in ('LLM_heat', 'LLM_chart', 'LLM_context', 'rule_zW', 'CNN'):
    key = name if name != 'CNN' else 'CNN_segment_check'
    daily[key] = {s: cut(cohort.run(str(EV / f'{name}_{t}_pred.npz'), '4h', 84, q=.1, cost=.001)[0], s) for s, (t, _) in SEG.items()}


def sh(r):
    return float(r.mean() / r.std(ddof=1) * np.sqrt(365))


for s in SEG:   # 구간별로 돌려 자른 CNN 과 MARKET 연속 장부를 자른 CNN 이 같은 날짜·같은 Sharpe 여야 한다
    a, b = daily['CNN'][s], daily['CNN_segment_check'][s]
    assert a.index.equals(b.index) and abs(sh(a) - sh(b)) < .02, (s, sh(a), sh(b), len(a), len(b))
del daily['CNN_segment_check']

rows = []
for name, d in daily.items():
    pooled = pd.concat([d[s] for s in EVAL]); eq = (1 + pooled).cumprod(); h = d['hold 26.03~05']; he = (1 + h).cumprod()
    rows.append({'variant': name, **{s: sh(d[s]) for s in SEG}, 'pooled_sharpe': sh(pooled), 'pooled_return_pct': float((eq.iloc[-1] - 1) * 100),
                 'pooled_mdd': float((eq / eq.cummax() - 1).min() * 100), 'pooled_nw_t': nw_pvalue(pooled.to_numpy(), 14)[0],
                 'hold_return_pct': float((he.iloc[-1] - 1) * 100), 'hold_mdd': float((he / he.cummax() - 1).min() * 100),
                 'days': len(pooled), 'hold_days': len(h)})
perf = pd.DataFrame(rows); perf.to_csv(OUT / 'performance.csv', index=False)

paired, rng = [], np.random.default_rng(20261019)
pool = {n: pd.concat([d[s] for s in EVAL]) for n, d in daily.items()}
for name in [n for n in daily if n.startswith('LLM')]:
    for ref in ['CNN', 'GBM', 'random1', 'random2', 'rule_zW']:
        x = pd.concat([pool[name], pool[ref]], axis=1).dropna().to_numpy(); n = len(x); v = []
        for _ in range(2000):
            st = rng.integers(0, n, size=int(np.ceil(n / 14))); a = x[((st[:, None] + np.arange(14)) % n).ravel()[:n]]
            s2 = a.mean(0) / a.std(0, ddof=1) * np.sqrt(365); v.append(s2[0] - s2[1])
        diff = x[:, 0] - x[:, 1]
        paired.append({'variant': name, 'reference': ref, 'days': n, 'sharpe_difference': float(sh(pd.Series(x[:, 0])) - sh(pd.Series(x[:, 1]))),
                       'bootstrap_95_low': float(np.quantile(v, .025)), 'bootstrap_95_high': float(np.quantile(v, .975)),
                       'daily_diff_sum_pctp': float(diff.sum() * 100), 'nw_t': nw_pvalue(diff, 14)[0]})
pd.DataFrame(paired).to_csv(OUT / 'paired.csv', index=False)
pd.DataFrame({n: pd.concat(d.values()) for n, d in daily.items()}).to_csv(OUT / 'daily_returns.csv')
print(perf.round(2).to_string(index=False)); print(pd.DataFrame(paired).round(2).to_string(index=False))
