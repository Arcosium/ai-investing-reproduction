"""분기별 표 18(CNN 점수·발생 확률 Spearman), 표 25(다리 유동성), 9장 숏 청산 추정. 학습·코호트 없이 예측과 4시간봉만 읽는다.
원래 계산 코드가 남아 있지 않아(results/ext_orth.csv·ext_shortleg.csv·paper/liquidity_legs.csv) 같은 정의로 다시 썼다.
--check 는 옛 폴드 0(분기 D, ens_heatf10_direct_4h_s0) 로 옛 D 행을 재현하는지 본다.
usage: MARKET_BARS=work/bars_hold PYTHONPATH=. python3 work/wf8/abcd_light.py [--check]  →  results/wf8_abcd/light.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from market import bars as B

import os
os.chdir(Path(os.environ.get('RESEARCH_DATA', '.')).resolve())
R = Path('work/results'); W = R / 'wf8_abcd'
QS = {1: 'A', 3: 'B', 5: 'C', 7: 'D'}
EV = {1: 'full_gbm_event_15m_W120_H10_s3', 3: 'full_gbm_event_15m_W120_H10_s2', 5: 'wf8_gbm_event_15m_W120_H10_s2', 7: 'full_gbm_event_15m_W120_H10_s0'}
H = 84; MS4H = 4 * 3600_000


def win(q):
    a = pd.Timestamp('2024-03-01') + pd.DateOffset(months=3 * q)
    return a.value // 10**6, (a + pd.DateOffset(months=3)).value // 10**6


def load(f):
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame({'ts': z['ts'], 'base': z['base'].astype(str), 's': z['p'][:, 1] - z['p'][:, 2], 'fwd': z['fwd'], 'sigH': z['sigH']})


_bars = {}
def bars(b):
    if b not in _bars:
        g = pd.read_parquet(B.path('4h', b), columns=['ts', 'c', 'qv']).sort_values('ts').reset_index(drop=True); _bars[b] = g
    return _bars[b]


def orth(c, ev):
    e = load(R / f'{ev}_pred.npz')
    z = np.load(R / f'{ev}_pred.npz', allow_pickle=True); e['pev'] = z['p'][:, 1] + z['p'][:, 2]
    e = e[['ts', 'base', 'pev']].sort_values('ts')
    m = pd.merge_asof(c.sort_values('ts'), e, on='ts', by='base', direction='backward', tolerance=75 * 60_000).dropna(subset=['pev'])
    m['abss'] = m.s.abs(); m['absfwd'] = m.fwd.abs()
    rk = m.groupby('ts')[['s', 'abss', 'pev', 'absfwd', 'sigH', 'fwd']].rank()
    rk['ts'] = m.ts
    def rho(a, b):   # 판단 시각별 Spearman(=순위의 Pearson) 평균
        g = rk[rk.groupby('ts').s.transform('size') > 10]
        return float(g.groupby('ts').apply(lambda d: d[a].corr(d[b])).mean())
    return {'n': int(len(m)), 'cnn_pev': rho('s', 'pev'), 'abscnn_pev': rho('abss', 'pev'), 'pev_absfwd': rho('pev', 'absfwd'),
            'pev_sigH': rho('pev', 'sigH'), 'cnn_fwd': rho('s', 'fwd')}


def legs(c):
    c = c.copy(); c['pr'] = c.groupby('ts').s.rank(pct=True)
    return c[c.pr >= .9].assign(leg='long'), c[c.pr <= .1].assign(leg='short')


def path_stats(L):
    """진입 = 판단 시각 봉의 종가, 만기 = 84봉 뒤 종가, 고점·저점 = 보유 84봉 종가의 최대·최소(옛 D 행 8,265건·1.17%·4.22%·190% 재현)."""
    out = []
    for b, g in L.groupby('base'):
        x = bars(b); t = x.ts.to_numpy(); i = np.searchsorted(t, g.ts.to_numpy())
        ok = (i + H < len(x)) & (t[np.minimum(i, len(t) - 1)] == g.ts.to_numpy())
        c = x.c.to_numpy()
        for j, k in zip(np.flatnonzero(ok), i[ok]):
            out.append((g.index[j], c[k + H] / c[k] - 1, c[k + 1:k + H + 1].max() / c[k] - 1, c[k + 1:k + H + 1].min() / c[k] - 1))
    return pd.DataFrame(out, columns=['i', 'end', 'peak', 'trough']).set_index('i')


def squeeze(c):
    lg, sh = legs(c); ps, pl = path_stats(sh), path_stats(lg)
    pct = lambda s: float(s.mean() * 100)
    return {'n_short': int(len(ps)), 'short_end50': pct(ps.end > .5), 'short_end100': pct(ps.end > 1), 'short_peak50': pct(ps.peak > .5),
            'short_peak100': pct(ps.peak > 1), 'short_max_end': float(ps.end.max() * 100), 'short_mean_end': float(ps.end.mean() * 100),
            'long_mean_end': float(pl.end.mean() * 100)}


def liquidity(c):
    lg, sh = legs(c); x = pd.concat([lg, sh]); x['day'] = pd.to_datetime(x.ts, unit='ms').dt.floor('D')
    adv = {}
    for b in x.base.unique():
        g = bars(b).copy(); g['day'] = pd.to_datetime(g.ts, unit='ms').dt.floor('D')
        adv[b] = g.groupby('day').qv.sum().rolling(30, min_periods=10).mean().shift(1)   # analyze.py 운용 용량과 같은 ADV
    A = pd.DataFrame(adv); x['adv'] = [A.at[d, b] if d in A.index else np.nan for d, b in zip(x.day, x.base)]
    x = x.dropna(subset=['adv']); m = lambda s, p: float(np.percentile(s, p) / 1e6)
    lo, sh_ = x[x.leg == 'long'].adv, x[x.leg == 'short'].adv
    return {'long_med': m(lo, 50), 'long_p10': m(lo, 10), 'short_med': m(sh_, 50), 'short_p10': m(sh_, 10), 'all_med': m(x.adv, 50), 'all_p10': m(x.adv, 10)}


def quarter(pred, ev, q=None):
    c = load(pred)
    if q:
        a, b = win(q); c = c[(c.ts >= a) & (c.ts < b)]
    return {'orth': orth(c, ev), 'squeeze': squeeze(c), 'liq': liquidity(c)}


if __name__ == '__main__':
    if '--check' in sys.argv:   # 옛 D 행: orth +0.101 +0.038 −0.027 −0.147 +0.112 (n 85,659) · 숏 8,265건 1.17% 0.40% 4.22% 1.15% max 190% · 유동성 3.8 1.7 32.3 10.9
        print(json.dumps(quarter(R / 'ens_heatf10_direct_4h_s0_pred.npz', EV[7]), indent=1)); sys.exit()
    res = {QS[q]: quarter(W / f'cnn10_q{q}_pred.npz', EV[q], q) for q in QS}
    json.dump(res, open('results/wf8_abcd/light.json', 'w'), indent=1); print(json.dumps(res, indent=1))
