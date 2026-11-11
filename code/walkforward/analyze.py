"""11/09 분기 연속 walk-forward 집계. 7개 분기(2024-06~2026-02, 끊김 없음) + 홀드아웃(8번째 분기 2026-03~05)을 같은 규칙·같은 코드로.
24.03~05(q0)는 학습 자료가 11개월뿐이라 시험 분기에서 뺀다(첫 시험 분기 학습 2023.01~2024.02).
분기 q 의 예측: 새로 학습한 q0·q2·q4·q5 는 wf8_*, 기존 A·B·C·D(= q1·q3·q6·q7)와 홀드아웃(q8)은 9/8 학습 파일. 코호트는 현재 market.cohort 로 전부 다시 계산.
합산은 분기 예측을 이어 붙인 한 파일로 코호트를 한 번 돌린다(분기 경계에서 겹치는 보유 기간을 두 번 세지 않는다).
  MARKET_BARS="$RESEARCH_DATA/work/bars_hold" PYTHONPATH="$PWD/code" python3 code/walkforward/analyze.py  →  $RESEARCH_DATA/results/wf8_abcd/*.csv, facts.json
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(os.environ.get('RESEARCH_DATA', '.')).resolve())
from market import bars as B
from market.perf import daily_returns, stats

TAG = os.environ.get('WF8_DIR', 'wf8_abcd')
R, OUT = Path('work/results'), Path('results') / TAG; W = R / TAG   # 코호트 결과는 전부 W 아래에 새로(9/8 결과 파일을 다시 쓰지 않는다)
OUT.mkdir(parents=True, exist_ok=True); W.mkdir(exist_ok=True)
HOLD_END = pd.Timestamp('2026-06-01').value // 10**6
QS = [int(q) for q in os.environ.get('WF8_QS', '1,3,5,7').split(',')]; OLD = {1: 3, 3: 2, 6: 1, 7: 0}           # q → 9/8 폴드 번호(A=s3, B=s2, C=s1, D=s0)
LAB = {0: '24.03~05', 1: '24.06~08', 2: '24.09~11', 3: '24.12~02', 4: '25.03~05', 5: '25.06~08', 6: '25.09~11', 7: '25.12~02', 8: '26.03~05'}
FUND = 'work/funding.parquet'


def seed_file(model, sd, q):
    suf = '' if sd == 0 else f'_sd{sd}'
    if q == 8:
        return R / f'hold_{model}_heatf{suf}_4h_W60_H84_s4_pred.npz'
    if q in OLD:
        return R / f'full_{model}_heatf{suf}_liq_4h_W60_H84_s{OLD[q]}_pred.npz'
    return R / f'wf8_{model}_heatf{suf}_4h_W60_H84_q{q}_pred.npz'


def load(f):
    return dict(np.load(f, allow_pickle=True))


def write(d, path):
    np.savez_compressed(path, **d)
    return path


def cut(d, lo=None, hi=None):
    m = np.ones(len(d['ts']), bool)
    if lo is not None:
        m &= d['ts'] >= lo
    if hi is not None:
        m &= d['ts'] < hi
    return {k: v[m] for k, v in d.items()}


def concat(paths, out):
    ds = [load(p) for p in paths]
    d = {k: np.concatenate([x[k] for x in ds]) for k in ('ts', 'base', 'y', 'p', 'fwd', 'sigH')}
    o = np.argsort(d['ts'], kind='stable')
    return write({k: v[o] for k, v in d.items()}, out)


def zavg(files, out):
    if not out.exists():
        subprocess.run([sys.executable, '-m', 'market.zavg', *map(str, files), '--out', str(out)], check=True, capture_output=True)
    return out


def cohort(pred, H=84, cost=0.001, extra=(), suffix=''):
    """pred 를 코호트로. 출력 이름이 비용·변형을 담지 못하므로 다른 조건은 링크 이름(suffix)으로 가른다."""
    pred = Path(pred)
    if suffix:
        link = pred.with_name(pred.name.replace('_pred.npz', f'_{suffix}_pred.npz'))
        try:   # 여러 스레드가 같은 링크를 동시에 만들 수 있다
            link.symlink_to(pred.name)
        except FileExistsError:
            pass
        pred = link
    tag = ('_shuf' + extra[extra.index('--shuffle') + 1] if '--shuffle' in extra else '') + ('_fund' if '--funding' in extra else '') + \
          ('_w' + extra[extra.index('--tag') + 1] if '--tag' in extra else '')
    j = Path(str(pred).replace('_pred.npz', f'_cohort_H{H}{tag}.json'))
    if not j.exists() or j.stat().st_mtime < pred.stat().st_mtime:   # 예측이 바뀌었으면(분기 범위 변경 등) 다시 계산
        r = subprocess.run([sys.executable, '-m', 'market.cohort', '--pred', str(pred), '--res', '4h', '--H', str(H), '--cost', f'{cost:.6f}', *extra],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f'{pred}: {r.stderr[-300:]}')
    return j


def summ(j):
    return json.load(open(j))['summary']


def pooled(js):
    r = pd.concat([daily_returns(j)[0] for j in js]).sort_index()
    return stats(r, (1 + r).cumprod(), 14)


def dsharpe_ci(a, b, block=14, n=2000, seed=0):
    """work/ext/stack.py 와 같은 짝지은 14일 블록 부트스트랩: Sharpe 차(a − b), 95% 구간, P(Δ ≤ 0)."""
    a, b = a.align(b, join='inner'); rng = np.random.default_rng(seed); n_ = len(a); nb = int(np.ceil(n_ / block)); d = []
    sh = lambda x: x.mean() / (x.std() + 1e-12) * np.sqrt(365)
    for _ in range(n):
        ii = np.concatenate([np.arange(s, min(s + block, n_)) for s in rng.integers(0, n_ - block + 1, nb)])[:n_]
        d.append(sh(a.iloc[ii]) - sh(b.iloc[ii]))
    d = np.array(d)
    return round(float(sh(a) - sh(b)), 2), round(float(np.quantile(d, .025)), 2), round(float(np.quantile(d, .975)), 2), round(float((d <= 0).mean()), 3)


def par(fn, items, n=8):
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(fn, items))


# ---------- 1. 분기별 예측: 시드 파일(홀드아웃은 05월 말에서 자름) → 시드 10개 앙상블 · MLP 3개 앙상블 ----------
def link(f):
    o = W / f.name
    if not o.exists():
        o.symlink_to(f.resolve())
    return o


def seed_pred(model, sd, q):
    f = seed_file(model, sd, q)
    if q != 8:
        return link(f)
    o = W / f.name.replace('_pred.npz', '_cut_pred.npz')
    return o if o.exists() else write(cut(load(f), hi=HOLD_END), o)


have = {q: all(seed_file('i1', s, q).exists() for s in range(10)) for q in QS + [8]}
print('quarters ready', have, flush=True)
qs = [q for q in QS + [8] if have[q]]
ens = {q: zavg([seed_pred('i1', s, q) for s in range(10)], W / f'cnn10_q{q}_pred.npz') for q in qs}
mlp = {(m, q): zavg([seed_pred(m, s, q) for s in range(3)], W / f'{m}x3_q{q}_pred.npz')
       for m in ('m1', 'm2') for q in QS if all(seed_file(m, s, q).exists() for s in range(3))}
gbm = {q: link(R / f'ens_wfv2_4h_s{q}_pred.npz') for q in QS}
gbm[8] = W / 'gbm_q8_pred.npz'
if not gbm[8].exists():
    write(cut(load(R / 'ens_v2hold_4h_s4_pred.npz'), hi=HOLD_END), gbm[8])
dev_q = [q for q in QS if have[q]]
dev = {'cnn': concat([ens[q] for q in dev_q], W / 'cnn10_dev_pred.npz'), 'gbm': concat([gbm[q] for q in dev_q], W / 'gbm_dev_pred.npz')}
for m in ('m1', 'm2'):
    if all((m, q) in mlp for q in dev_q):
        dev[m] = concat([mlp[(m, q)] for q in dev_q], W / f'{m}x3_dev_pred.npz')

# ---------- 2. 장부 하나로 돌리고 달력 3개월로 자른다(개발 8개 분기 이어 붙임 / 홀드아웃 따로). 분기마다 정확히 같은 길이 ----------
def win(q):
    a = pd.Timestamp('2024-03-01') + pd.DateOffset(months=3 * q)
    return a, a + pd.DateOffset(months=3)


DEV_WIN = (pd.Timestamp('2024-06-01'), pd.Timestamp('2026-03-01'))
J = lambda p, t='': Path(str(p).replace('_pred.npz', f'_cohort_H84{t}.json'))


def r_of(j, a, b):
    r = daily_returns(j)[0].sort_index()
    return r[(r.index >= a) & (r.index < b)]


def r_dev(j):   # 개발 분기 창만 이어 붙인다(A~D 처럼 띄엄띄엄이면 사이 분기는 빠진다. 7개 분기면 DEV_WIN 과 같다)
    return pd.concat([r_of(j, *win(q)) for q in dev_q])


def st_of(r):
    s = stats(r, (1 + r).cumprod(), 14); s['ret'] = round(float((1 + r).prod() - 1) * 100, 2); return s


hold = {'cnn': ens.get(8), 'gbm': gbm[8]}
runs = [(p, ()) for p in dev.values()] + [(dev['cnn'], ('--shuffle', '1')), (dev['cnn'], ('--shuffle', '2')),
        (dev['cnn'], ('--funding', FUND)), (dev['gbm'], ('--funding', FUND))]
if hold['cnn']:
    runs += [(p, ()) for p in hold.values()] + [(hold['cnn'], ('--shuffle', '1')), (hold['cnn'], ('--shuffle', '2')),
             (hold['cnn'], ('--funding', FUND)), (hold['gbm'], ('--funding', FUND))]
par(lambda a: cohort(a[0], extra=a[1]), runs)
rows = []
for q in dev_q + ([8] if hold['cnn'] else []):
    src = hold if q == 8 else dev; a, b = (pd.Timestamp('2026-03-01'), pd.Timestamp('2026-06-01')) if q == 8 else win(q)
    r = {'q': q, 'period': LAB[q], 'set': 'holdout' if q == 8 else ('tuned' if q in OLD else 'new')}
    for k in ('cnn', 'gbm'):
        s_ = st_of(r_of(J(src[k]), a, b)); r.update({f'{k}_sharpe': s_['sharpe'], f'{k}_p': s_['p'], f'{k}_mdd': s_['mdd'], f'{k}_ret': s_['ret'],
                                                   f'{k}_fund': st_of(r_of(J(src[k], '_fund'), a, b))['sharpe']})
    r['shuf1'], r['shuf2'] = (st_of(r_of(J(src['cnn'], f'_shuf{i}'), a, b))['sharpe'] for i in (1, 2))
    for m in ('m1', 'm2'):
        r[m] = st_of(r_of(J(dev[m]), a, b))['sharpe'] if (m in dev and q != 8) else None
    rows.append(r)
qt = pd.DataFrame(rows); qt.to_csv(OUT / 'quarters.csv', index=False); print(qt.to_string(index=False), flush=True)
pool = {k: st_of(r_dev(J(p))) for k, p in dev.items()}
pool.update({'cnn_fund': st_of(r_dev(J(dev['cnn'], '_fund'))), 'gbm_fund': st_of(r_dev(J(dev['gbm'], '_fund'))),
             'shuf1': st_of(r_dev(J(dev['cnn'], '_shuf1'))), 'shuf2': st_of(r_dev(J(dev['cnn'], '_shuf2')))})
HW = (pd.Timestamp('2026-03-01'), pd.Timestamp('2026-06-01'))
if hold['cnn']:
    pool.update({f'hold_{k}': st_of(r_of(J(hold[k], t), *HW)) for k in ('cnn', 'gbm') for t in ('',)})
    pool.update({'hold_cnn_fund': st_of(r_of(J(hold['cnn'], '_fund'), *HW)), 'hold_gbm_fund': st_of(r_of(J(hold['gbm'], '_fund'), *HW))})
pool['diff_cnn_gbm'] = dsharpe_ci(r_dev(J(dev['cnn'])), r_dev(J(dev['gbm'])))   # CNN − LightGBM, 짝지은 블록 부트스트랩
pool.update({f'diff_cnn_{m}': dsharpe_ci(r_dev(J(dev['cnn'])), r_dev(J(dev[m]))) for m in ('m1', 'm2') if m in dev})
print('pooled', json.dumps(pool, default=float), flush=True)

# ---------- 3. 시드 10개: 시드마다 개발 장부 하나 + 홀드아웃 장부 하나 → 분기로 자른다 ----------
sd_dev = [concat([seed_pred('i1', s, q) for q in dev_q], W / f'cnn_sd{s}_dev_pred.npz') for s in range(10)]
par(cohort, sd_dev + ([seed_pred('i1', s, 8) for s in range(10)] if hold['cnn'] else []))
seeds = []
for s in range(10):
    r = {'seed': s, **{f'q{q}': st_of(r_of(J(sd_dev[s]), *win(q)))['sharpe'] for q in dev_q}}
    d_ = st_of(r_dev(J(sd_dev[s]))); r.update(dev=d_['sharpe'], dev_p=d_['p'])
    if hold['cnn']:
        h_ = st_of(r_of(J(seed_pred('i1', s, 8)), *HW)); r.update(q8=h_['sharpe'], q8_p=h_['p'])
    seeds.append(r)
seeds = pd.DataFrame(seeds); seeds.to_csv(OUT / 'seeds.csv', index=False); print(seeds.to_string(index=False), flush=True)

# ---------- 4. 보유 기간(점수 고정) · 운용 용량 ----------
HS = [12, 24, 42, 63, 84, 105, 126, 168, 252]
par(lambda h: cohort(dev['cnn'], H=h), HS); par(lambda h: cohort(dev['cnn'], H=h, cost=0.0, suffix='gross'), HS)
JH = lambda p, h: Path(str(p).replace('_pred.npz', f'_cohort_H{h}.json'))
gross = Path(str(dev['cnn']).replace('_pred.npz', '_gross_pred.npz'))
decay = pd.DataFrame([{'hold_days': h / 6, 'sharpe': st_of(r_dev(JH(dev['cnn'], h)))['sharpe'],
                       'gross_sharpe': st_of(r_dev(JH(gross, h)))['sharpe']} for h in HS])
decay.to_csv(OUT / 'decay.csv', index=False); print(decay.to_string(index=False), flush=True)

d = load(dev['cnn']); x = pd.DataFrame({'ts': d['ts'], 'base': d['base'].astype(str), 's': d['p'][:, 1] - d['p'][:, 2], 'sigH': d['sigH']})
x['pr'] = x.groupby('ts').s.rank(pct=True); legs = x[(x.pr >= .9) | (x.pr <= .1)].copy(); legs['day'] = pd.to_datetime(legs.ts, unit='ms').dt.floor('D')
adv = {}
for b in sorted(legs.base.unique()):
    p = B.path('4h', b)
    if os.path.exists(p):
        g = pd.read_parquet(p, columns=['ts', 'qv']); g['day'] = pd.to_datetime(g.ts, unit='ms').dt.floor('D')
        adv[b] = g.groupby('day').qv.sum().rolling(30, min_periods=10).mean().shift(1)   # 판단일 이전 30일 평균 거래대금
A = pd.DataFrame(adv); legs['adv'] = [A.at[dy, b] if (b in A.columns and dy in A.index) else np.nan for dy, b in zip(legs.day, legs.base)]
legs = legs.dropna(subset=['adv']); legs['sig_day'] = legs.sigH / np.sqrt(84) * np.sqrt(6)
cap = []
for F in (1e6, 3e6, 1e7, 3e7, 1e8, 3e8):
    imp = float((0.7 * legs.sig_day * np.sqrt(F * 6 / (2 * 84 * 20) / legs.adv)).mean())   # 제곱근 충격, 편도
    s_ = st_of(r_dev(cohort(dev['cnn'], cost=0.001 + imp, suffix=f'F{int(F / 1e6)}m')))
    cap.append({'fund_usd_m': F / 1e6, 'impact_bp_side': round(imp * 1e4, 1), 'sharpe': s_['sharpe'], 'p': s_['p']})
cap = pd.DataFrame(cap); cap.to_csv(OUT / 'capacity.csv', index=False); print(cap.to_string(index=False), flush=True)

# ---------- 5. BTC 대비 · 월별 · 국면 ----------
btc = pd.read_parquet('work/bars_ext/4h/BTC.parquet', columns=['ts', 'c']); btc.index = pd.to_datetime(btc.ts, unit='ms'); bd = btc.c.resample('D').last()
rd, rg = r_dev(J(dev['cnn'])), r_dev(J(dev['gbm']))
rh = r_of(J(hold['cnn']), *HW) if hold['cnn'] else None
curve = pd.DataFrame({'cnn': (1 + rd).cumprod(), 'gbm': (1 + rg.reindex(rd.index).fillna(0)).cumprod(), 'btc': (1 + bd.pct_change().reindex(rd.index).fillna(0)).cumprod()})
curve.to_csv(OUT / 'dev_curve.csv')
mdd = lambda e: float((e / e.cummax() - 1).min() * 100)
btc_cmp = {k: {'ret': round((curve[k].iloc[-1] - 1) * 100, 1), 'mdd': round(mdd(curve[k]), 1)} for k in curve}
qret = {q: {'cnn': round(((1 + rd[(rd.index >= win(q)[0]) & (rd.index < win(q)[1])]).prod() - 1) * 100, 1),
            'btc': round((bd.reindex([win(q)[1] - pd.Timedelta(days=1)]).iloc[0] / bd.reindex([win(q)[0] - pd.Timedelta(days=1)]).iloc[0] - 1) * 100, 1)} for q in dev_q}
mon = [{'m': str(k), 'v': round(((1 + g).prod() - 1) * 100, 2), 'seg': 'dev'} for k, g in rd.groupby(rd.index.to_period('M'))]
hcur = None
if rh is not None:
    hcur = pd.DataFrame({'cnn': (1 + rh).cumprod(), 'btc': (1 + bd.pct_change().reindex(rh.index).fillna(0)).cumprod()}); hcur.to_csv(OUT / 'hold_curve.csv')
    btc_cmp['hold'] = {k: {'ret': round((hcur[k].iloc[-1] - 1) * 100, 1), 'mdd': round(mdd(hcur[k]), 1)} for k in hcur}
    mon += [{'m': str(k), 'v': round(((1 + g).prod() - 1) * 100, 2), 'seg': 'holdout'} for k, g in rh.groupby(rh.index.to_period('M'))]
tr30 = (bd / bd.shift(30) - 1).shift(1)
g = pd.DataFrame({'r': rd, 'tr': tr30.reindex(rd.index)}).dropna()
g['trend'] = pd.cut(g.tr, [-1, -.10, .10, 10], labels=['BTC 30일 -10% 미만', '±10%', '+10% 초과'])
reg = pd.DataFrame([{'regime': k, 'days': len(sub), **stats(sub.r, (1 + sub.r).cumprod(), 14)} for k, sub in g.groupby('trend', observed=True)])
reg.to_csv(OUT / 'regime.csv', index=False); print(reg.to_string(index=False), btc_cmp, qret, flush=True)

# ---------- 6. 가설 3: 다리 안 비중(발생 확률·1/σ·곱) · q0 는 15분봉 자료가 2024-01 부터라 발생 모델이 없어 q1~q7 ----------
EV = {1: 'full_gbm_event_15m_W120_H10_s3', 3: 'full_gbm_event_15m_W120_H10_s2', 6: 'full_gbm_event_15m_W120_H10_s1', 7: 'full_gbm_event_15m_W120_H10_s0',
      2: 'wf8_gbm_event_15m_W120_H10_s0', 4: 'wf8_gbm_event_15m_W120_H10_s1', 5: 'wf8_gbm_event_15m_W120_H10_s2'}
stack = None
sq = [q for q in dev_q if q in EV and (R / f'{EV[q]}_pred.npz').exists()]
if sq:
    base = concat([ens[q] for q in sq], W / 'cnn10_ev_pred.npz'); c = load(base)
    c = pd.DataFrame({'ts': c['ts'], 'base': c['base'].astype(str), 'sigH': c['sigH']})
    ev = []
    for q in sq:
        z = load(R / f'{EV[q]}_pred.npz'); ev.append(pd.DataFrame({'ts': z['ts'], 'base': z['base'].astype(str), 'pev': z['p'][:, 1] + z['p'][:, 2]}))
    ev = pd.concat(ev).sort_values('ts')
    ce = pd.merge_asof(c.reset_index().sort_values('ts'), ev, on='ts', by='base', direction='backward', tolerance=75 * 60_000).sort_values('index')
    cov = float(ce.pev.notna().mean())
    ce['pev'] = ce.pev.fillna(ce.groupby('ts').pev.transform('median')).fillna(ce.pev.median()); rk = ce.groupby('ts').pev.rank(pct=True).to_numpy()
    iv = 1.0 / np.maximum(ce.sigH.to_numpy(), 1e-6)
    var = {'ev': rk, 'iv': iv, 'evi': rk * iv}
    for k, w in var.items():
        np.savez_compressed(W / f'w_{k}.npz', ts=ce.ts.to_numpy(), base=ce.base.to_numpy().astype(str), w=w)
    par(lambda k: cohort(base, extra=('--weight', str(W / f'w_{k}.npz'), '--tag', k)), list(var))
    cohort(base)
    r0 = r_dev(J(base))
    stack = [{'variant': 'cnn', **{k: v for k, v in st_of(r0).items() if k in ('sharpe', 'p', 'mdd')}}]
    for k in var:
        rr = r_dev(J(base, f'_w{k}')); dd = dsharpe_ci(rr, r0)
        stack.append({'variant': k, **{a: v for a, v in st_of(rr).items() if a in ('sharpe', 'p', 'mdd')}, 'd': dd[0], 'lo': dd[1], 'hi': dd[2], 'p_le0': dd[3],
                      'per_q': [st_of(rr[(rr.index >= win(q)[0]) & (rr.index < win(q)[1])])['sharpe'] - st_of(r0[(r0.index >= win(q)[0]) & (r0.index < win(q)[1])])['sharpe'] for q in sq]})
    stack = pd.DataFrame(stack); stack.to_csv(OUT / 'stack.csv', index=False); print('stack quarters', sq, 'coverage', round(cov, 3)); print(stack.to_string(index=False), flush=True)

json.dump({'quarters': rows, 'pooled': pool, 'dev_quarters': dev_q, 'seeds': seeds.to_dict('records'), 'decay': decay.to_dict('records'),
           'capacity': cap.to_dict('records'), 'btc': btc_cmp, 'quarter_ret': qret, 'monthly': mon, 'regime': reg.to_dict('records'),
           'stack': None if stack is None else stack.to_dict('records'), 'stack_quarters': sq,
           'curve': {'x': [str(i.date()) for i in curve.index], **{k: [round(v, 4) for v in curve[k]] for k in curve}},
           'hold_curve': None if hcur is None else {'x': [str(i.date()) for i in hcur.index], **{k: [round(v, 4) for v in hcur[k]] for k in hcur}}},
          open(OUT / 'facts.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('WF8_ANALYZE_DONE', flush=True)
