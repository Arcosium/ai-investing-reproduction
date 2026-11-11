import os
"""분기 A~D 캔들·이미지 단계·발생 예측 집계(학습 없음, 결과 JSON·예측만 읽는다). 분기 C 는 q7fill 결과(q7_*_q5), A·B·D 는 옛 폴드 s3·s2·s0.
  표 6(25개 설정 spread_z·상위 적중) · 표 7(십분위 평균 z·적중) · 3장 2절 모멘텀(점수와 직전 20일 수익률 Spearman, 모멘텀 spread_z)
  · 표 10(heat·heatx·heatf spread_z) · 표 B1(lift / 미학습 lift) · LightGBM 1d·20·20(abcd_gbm/screen, 같은 종목군·봉).
모멘텀·십분위 정의는 옛 D·A 값(+0.32·−0.19, −0.72·+0.08)을 재현하는 것으로 맞췄다. 없는 분기는 null.
usage: MARKET_BARS="$RESEARCH_DATA/work/bars_hold" PYTHONPATH="$PWD/code" python3 code/walkforward/abcd_candle.py → $RESEARCH_DATA/results/wf8_abcd/candle.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from market import bars as B
from market.metrics import evaluate

ROOT = Path(os.environ.get('RESEARCH_DATA', '.')).resolve(); R = ROOT / 'work/results'
QS = (1, 3, 5, 7); LET = dict(zip(QS, 'ABCD'))
MAP = json.load(open(ROOT / 'work/q7fill/mapping.json'))['entries']
J5 = {j['family']: ROOT / j['out'] for j in json.load(open(ROOT / 'work/q7fill/jobs.json'))['jobs'] if j['quarter'] == 5}
GBM = R / 'abcd_gbm'   # 분할 0~3 = A~D


def src(fam, sources, q):
    return J5.get(fam) if q == 5 else R / Path(sources[str(q)]).name


def get(path, *keys):
    if not path or not Path(path).exists():
        return None
    v = json.load(open(path))
    for k in keys:
        v = v[k]
    return float(v)


def preds(f):
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame({'ts': z['ts'], 'base': z['base'].astype(str), 's': z['p'][:, 1] - z['p'][:, 2], 'fwd': z['fwd'], 'sigH': z['sigH'], 'y': z['y']})


def deciles(f):
    """점수 상·하위 10%(전체 표본 분위) 의 평균 z(=fwd/σH) 와 적중률(상위 fwd>0, 하위 fwd<0) — metrics.evaluate 와 같은 경계."""
    if not Path(f).exists():
        return None
    d = preds(f); q = np.quantile(d.s, [.1, .9]); z = d.fwd / np.maximum(d.sigH, 1e-9); hi, lo = d.s >= q[1], d.s <= q[0]
    return {'top_z': float(z[hi].mean()), 'bot_z': float(z[lo].mean()), 'hit_top': float((d.fwd[hi] > 0).mean()), 'hit_bot': float((d.fwd[lo] < 0).mean())}


def momentum(f):
    """점수와 직전 20일 수익률(판단 봉 종가 / 20봉 전 종가 − 1)의 전체 표본 Spearman, 20일 수익률 자체를 점수로 한 spread_z."""
    if not Path(f).exists():
        return None
    d = preds(f); out = []
    for b, g in d.groupby('base'):
        x = pd.read_parquet(B.path('1d', b), columns=['ts', 'c']).sort_values('ts'); t, c = x.ts.to_numpy(), x.c.to_numpy()
        k = np.searchsorted(t, g.ts.to_numpy()); ok = (k >= 20) & (k < len(c)); r = np.full(len(g), np.nan); r[ok] = c[k[ok]] / c[k[ok] - 20] - 1
        out.append(g.assign(r20=r))
    e = pd.concat(out).dropna(subset=['r20']); pr = e.r20.rank(pct=True).to_numpy()
    sz = evaluate(e.y.to_numpy(), np.stack([np.zeros(len(e)), pr, 1 - pr], 1), e.fwd.to_numpy(), e.sigH.to_numpy())['spread_z']
    return {'rho': float(e[['s', 'r20']].rank().corr().iloc[0, 1]), 'mom_sz': float(sz)}


out = {'candle': [], 'table10': {}, 'B1': {}}
for e in MAP:
    fam = e['families'][0]; s = e['sources'][fam]
    if e['table'] == '1/6':
        out['candle'].append({'row': e['row'], 'label': e['label'], 'fam': fam,
                              'sz': {LET[q]: get(src(fam, s, q), 'test', 'spread_z') for q in QS}, 'hit': {LET[q]: get(src(fam, s, q), 'test', 'hit_top') for q in QS}})
    elif e['table'] == '10':
        out['table10'][e['row']] = {LET[q]: get(src(fam, s, q), 'test', 'spread_z') for q in QS}
    elif e['table'] == 'B1':
        out['B1'][e['row']] = {LET[q]: [get(src(fam, s, q), 'test', 'lift'), get(src(fam, s, q), 'unseen', 'lift')] for q in QS}
        out['B1'][str(e['row']) + '_ap'] = {LET[q]: get(src(fam, s, q), 'test', 'ap') for q in QS}
gev = pd.read_csv(R / 'wf8_gbm_event.csv').set_index('split')   # 발생 LightGBM 분기 C(= 분할 2)
out['B1_gbm_C'] = [float(gev.at[2, 'test_lift']), float(gev.at[2, 'unseen_lift'])]
c1d = {q: (R / f'q7_full_i1_relbin_1d_W20_H20_q5_pred.npz' if q == 5 else R / f"full_i1_relbin_1d_W20_H20_s{ {1: 3, 3: 2, 7: 0}[q]}_pred.npz") for q in QS}
out['momentum'] = {LET[q]: momentum(c1d[q]) for q in QS}
out['deciles_cnn'] = {LET[q]: deciles(c1d[q]) for q in QS}
if (GBM / 'screen.csv').exists():
    g = pd.read_csv(GBM / 'screen.csv').set_index('split')
    out['gbm1d'] = {LET[q]: {'sz': float(g.at[i, 'test_spread_z']), 'hit': float(g.at[i, 'test_hit_top'])} for i, q in enumerate(QS) if i in g.index}
    out['deciles_gbm'] = {LET[q]: deciles(GBM / f'screen_1d_W20_H20_s{i}_pred.npz') for i, q in enumerate(QS)}
    rho = {}
    for i, q in enumerate(QS):   # LightGBM 점수와 모멘텀 상관(본문 |ρ| ≤ 0.3 확인용)
        m = momentum(GBM / f'screen_1d_W20_H20_s{i}_pred.npz'); rho[LET[q]] = m and m['rho']
    out['gbm1d_mom_rho'] = rho
v2 = pd.read_csv(R / 'wf_xss_relbin.csv'); v2 = v2[(v2.res == '4h') & (v2.W == 60) & (v2.H == 84)].set_index('split')   # 분기 q = 분할 q
out['v2'] = {LET[q]: {'sz': float(v2.at[q, 'test_spread_z']), 'hit': float(v2.at[q, 'test_hit_top'])} for q in QS}
out['heatf_s0'] = {LET[q]: get(R / (f'wf8_i1_heatf_4h_W60_H84_q5.json' if q == 5 else f"full_i1_heatf_liq_4h_W60_H84_s{ {1: 3, 3: 2, 7: 0}[q]}.json"), 'test', 'spread_z') for q in QS}
json.dump(out, open(ROOT / 'results/wf8_abcd/candle.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != 'candle'}, ensure_ascii=False)[:3000])
