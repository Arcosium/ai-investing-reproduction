"""7장 3절 <표 22>: CNN·LightGBM 다리별 Sharpe(MDD)와 숏 다리 구성. analyze.py 를 같은 WF8_DIR·WF8_QS 로 돌린 뒤 실행.
  MARKET_BARS="$RESEARCH_DATA/work/bars_hold" PYTHONPATH="$PWD/code" WF8_DIR=wf8_abcd WF8_QS=1,3,5,7 python3 code/walkforward/legs.py  →  $RESEARCH_DATA/results/<WF8_DIR>/legs.json
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from market import metrics as M
from market.paper_rules import select_legs
from market.perf import daily_returns, stats

import os
os.chdir(Path(os.environ.get('RESEARCH_DATA', '.')).resolve())
TAG = os.environ.get('WF8_DIR', 'wf8_abcd'); W = Path('work/results') / TAG
QS = [int(q) for q in os.environ.get('WF8_QS', '1,3,5,7').split(',')]
win = lambda q: (pd.Timestamp('2024-03-01') + pd.DateOffset(months=3 * q), pd.Timestamp('2024-03-01') + pd.DateOffset(months=3 * q + 3))
P = {'cnn': W / 'cnn10_dev_pred.npz', 'gbm': W / 'gbm_dev_pred.npz'}


def leg(pred, flag):
    j = Path(str(pred).replace('_pred.npz', f'_cohort_H84{"_" + flag if flag else ""}.json'))
    if not j.exists() or j.stat().st_mtime < Path(pred).stat().st_mtime:   # 예측이 바뀌었으면 다시 계산(analyze.py 와 같은 규칙)
        subprocess.run([sys.executable, '-m', 'market.cohort', '--pred', str(pred), '--res', '4h', '--H', '84', '--cost', '0.001000',
                        *([f'--{flag}_only'] if flag else [])], check=True, capture_output=True)
    r = daily_returns(j)[0].sort_index()
    return pd.concat([r[(r.index >= a) & (r.index < b)] for a, b in map(win, QS)])


def st(r):
    s = stats(r, (1 + r).cumprod(), 14); return {'sharpe': s['sharpe'], 'p': s['p'], 'mdd': s['mdd']}


out = {}
with ThreadPoolExecutor(6) as ex:
    R = dict(zip([(m, f) for m in P for f in ('', 'long', 'short')], ex.map(lambda k: leg(P[k[0]], k[1]), [(m, f) for m in P for f in ('', 'long', 'short')])))
for m in P:
    out[m] = {f or 'ls': st(R[(m, f)]) for f in ('', 'long', 'short')}
    out[m]['short_q'] = {q: st(R[(m, 'short')][(R[(m, 'short')].index >= win(q)[0]) & (R[(m, 'short')].index < win(q)[1])])['sharpe'] for q in QS}

# 숏 다리 구성: 고변동 = 같은 시각 sigH 상위 20%, 급등 = 이후 14일 상대수익 > +30%(로그 ln 1.3, 7개 분기 <표 22> 재현)
D = {}
for m, p in P.items():
    z = np.load(p, allow_pickle=True)
    d = pd.DataFrame({'ts': z['ts'], 'base': z['base'].astype(str), 's': M.score(z['p']), 'fwd': z['fwd'], 'sigH': z['sigH']})
    d['rel'] = d.fwd - d.groupby('ts').fwd.transform('mean')
    d['hv'] = d.groupby('ts').sigH.rank(pct=True) > 0.8
    lm = np.zeros(len(d), bool); sm = np.zeros(len(d), bool)
    for _, g in d.groupby('ts'):
        a, b = select_legs(g.s, 0.1); lm[g.index[a]] = True; sm[g.index[b]] = True
    d['L'], d['S'] = lm, sm; D[m] = d
    S_ = d[d.S]
    out[m].update(short_hv=round(float(S_.hv.mean()) * 100, 1), short_surge=round(float((S_.rel > np.log(1.3)).mean()) * 100, 1),
                  ls_gap=round(float(d[d.L].groupby('ts').rel.mean().mean() - S_.groupby('ts').rel.mean().mean()) * 100, 2))
d = D['cnn']
out['hv_mean_rel'] = round(float(d[d.hv].rel.mean()) * 100, 2)
out['hv_surge'] = {k: round(float(((d[d.hv] if k == 'hv' else d[~d.hv]).rel > np.log(1.3)).mean()) * 100, 1) for k in ('hv', 'rest')}
c, g = D['cnn'].set_index(['ts', 'base']), D['gbm'].set_index(['ts', 'base'])
j = c[['s', 'L', 'S']].join(g[['s', 'L', 'S']], lsuffix='_c', rsuffix='_g', how='inner').reset_index()
out['rank_corr'] = round(float(j.groupby('ts')[['s_c', 's_g']].apply(lambda x: x.s_c.corr(x.s_g, method='spearman')).mean()), 2)
out['overlap_top'] = round(float(j[j.L_c].L_g.mean()) * 100, 1); out['overlap_bottom'] = round(float(j[j.S_c].S_g.mean()) * 100, 1)
out['n_ts'] = int(D['cnn'].ts.nunique())
Path('results', TAG).mkdir(parents=True, exist_ok=True)
json.dump(out, open(Path('results', TAG) / 'legs.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, default=float))
