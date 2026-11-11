"""수정안 그림 5·8·9를 달력 기준 평가값(haiku_r1/evaluation_cal)으로 다시 그린다. python3 figures_cal.py
기존 그림(source-experiments figures.py)과 같은 색·축 규칙, 서체만 맑은 고딕. assets 파일 이름은 원고 참조 그대로."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'results/calendar'
A = ROOT / 'output/figures'
A.mkdir(parents=True, exist_ok=True)
import os
font = os.environ.get('PAPER_FONT')
if font:
    font_manager.fontManager.addfont(font)
family = font_manager.FontProperties(fname=font).get_name() if font else 'DejaVu Sans'
plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220, 'hatch.color': 'white'})
INK, GRAY, LIGHT, RED = '#222222', '#888888', '#cccccc', '#a83b32'
SEG = ['24.06~08', '24.12~02', '25.06~08', '25.12~02']
LAB = ['A 2024.6~8', 'B 2024.12~2025.2', 'C 2025.6~8', 'D 2025.12~2026.2']
perf = pd.read_csv(E / 'performance.csv').set_index('variant')
neg = lambda v: f'{v:.2f}'  # 맑은 고딕에는 U+2212 글리프가 없어 하이픈 그대로

# 그림 5: 조건별 구간 A~D Sharpe
series = [('Haiku · 차트 이미지', 'LLM_chart', RED, '////'), ('Haiku · 지표 37개', 'LLM_context', RED, ''),
          ('Haiku · 열지도', 'LLM_heat', RED, '....'), ('최근 가격 위치 순 규칙', 'rule_zW', LIGHT, ''),
          ('LightGBM', 'GBM', GRAY, ''), ('열지도 CNN', 'CNN', INK, '')]
fig, ax = plt.subplots(figsize=(6.3, 3.1))
x = np.arange(4); w = 0.8 / len(series)
for i, (name, key, color, hatch) in enumerate(series):
    ax.bar(x + (i - (len(series) - 1) / 2) * w, perf.loc[key, SEG].to_numpy(float), w * 0.92, label=name, color=color, hatch=hatch, linewidth=0)
ax.axhline(0, color='#555555', lw=.8)
ax.set_xticks(x, LAB, fontsize=8)
ax.set_ylabel('연율 Sharpe'); ax.grid(axis='y', color='#dddddd', lw=.5)
ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.3), columnspacing=1.2, handlelength=1.6)
fig.tight_layout(); fig.savefig(A / 'llm_fold_sharpe_v3.png'); plt.close(fig)

# 그림 8: CNN·LightGBM 구간별·합산 Sharpe(값 표기)
fig, ax = plt.subplots(figsize=(6.3, 2.3))
cols = SEG + ['pooled_sharpe']; x = np.arange(len(cols))
for off, (name, key, color) in zip((-.18, .18), (('LightGBM', 'GBM', GRAY), ('열지도 CNN (10개 초기화)', 'CNN', RED))):
    v = perf.loc[key, cols].to_numpy(float)
    ax.bar(x + off, v, .34, label=name, color=color, linewidth=0)
    for xi, vi in zip(x + off, v):
        ax.text(xi, vi + .08, neg(vi), ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x, ['A', 'B', 'C', 'D', '합산'], fontsize=8); ax.set_ylim(0, 6.6)
ax.set_ylabel('연율 Sharpe', fontsize=8); ax.tick_params(axis='y', labelsize=8); ax.grid(axis='y', color='#dddddd', lw=.5)
ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper right', handlelength=1.6)
fig.tight_layout(); fig.savefig(A / 'gbm_cnn_native.png'); plt.close(fig)

# 그림 9: 열지도 CNN 롱숏과 BTC 단순 보유의 누적 자산(BTC = 4시간봉 일별 마지막 종가의 전일 대비 변화)
btc = pd.read_csv(ROOT / 'data/btc_daily_returns.csv', index_col=0).BTC
d = pd.read_csv(E / 'daily_returns.csv', index_col=0).CNN
ev, ho = d.loc[:'2026-02-28'], d.loc['2026-03-01':]
mdd = lambda s: (s / np.maximum.accumulate(np.r_[1.0, s.to_numpy()])[1:] - 1).min() * 100
S, B = (1 + ev.reset_index(drop=True)).cumprod(), (1 + btc.reindex(ev.index).reset_index(drop=True)).cumprod()
h, hb = (1 + ho).cumprod(), (1 + btc.reindex(ho.index)).cumprod()
starts = [ev.index.get_loc(ev.loc[f'20{s[:2]}-{s[3:5]}-01':].index[0]) for s in SEG]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 3.1), gridspec_kw={'width_ratios': [2.2, 1]})
lab = lambda name, s: f'{name} ×{s.iloc[-1]:.2f}, MDD {mdd(s):.1f}%'
a1.plot(np.r_[0, S.index + 1], np.r_[1, S], color=RED, lw=1.3, label=lab('열지도 CNN 롱숏', S))
a1.plot(np.r_[0, B.index + 1], np.r_[1, B], color=GRAY, lw=1, label=lab('BTC 보유', B))
for k, x0 in zip('ABCD', starts):
    if x0: a1.axvline(x0, color='#bbbbbb', lw=.6, ls=':')
    a1.text(x0 + 4, 1.72, f'구간 {k}', fontsize=7.5, color='#555555')
a1.set_ylim(.4, 1.8); a1.set_xlim(0, len(S) + 2)
a1.set_xlabel('네 평가 구간을 이은 일수(구간 사이는 불연속)', fontsize=8); a1.set_ylabel('누적 자산(시작 = 1)', fontsize=8)
a1.set_title('네 평가 구간 A→D', fontsize=9, loc='left')
a2.plot(range(len(h)), h.to_numpy(), color=RED, lw=1.3, label=lab('열지도 CNN 롱숏', h.reset_index(drop=True)))
a2.plot(range(len(hb)), hb.to_numpy(), color=GRAY, lw=1, label=lab('BTC 보유', hb.reset_index(drop=True)))
a2.set_xticks([0, len(h) - 1], [h.index[0].replace('-', '.'), h.index[-1].replace('-', '.')], fontsize=7.5)
a2.set_ylim(min(h.min(), hb.min()) - .03, max(h.max(), hb.max()) + .09)
a2.set_title('남겨 둔 기간 2026년 3~5월', fontsize=9, loc='left')
for ax in (a1, a2):
    ax.axhline(1, color='#555555', lw=.7); ax.grid(axis='y', color='#e4e4e4', lw=.5); ax.tick_params(labelsize=7.5)
    ax.legend(frameon=False, fontsize=7.5, loc='lower left' if ax is a1 else 'upper left', handlelength=1.4)
fig.tight_layout(w_pad=2); fig.savefig(A / 'heatmap_nav_v3.png'); plt.close(fig)

# 값 대조: 그림이 쓴 값 = performance.csv, 누적 = 원고 문장(+57.66%, 별도 기간 +12.14%)
assert abs((S.iloc[-1] - 1) * 100 - perf.loc['CNN', 'pooled_return_pct']) < 1e-6
assert abs((h.iloc[-1] - 1) * 100 - perf.loc['CNN', 'hold_return_pct']) < 1e-6
assert len(S) == 364 and len(h) == 92 and B.notna().all() and hb.notna().all()
print('CNN', round(S.iloc[-1], 4), round(mdd(S), 2), 'BTC', round(B.iloc[-1], 4), round(mdd(B), 2),
      'hold CNN', round(h.iloc[-1], 4), 'BTC', round(hb.iloc[-1], 4), 'starts', starts)

# 그림 2: 캔들 CNN 25개 설정의 구간 A~D 표준화 상대수익 차(research-data results/wf8_abcd/candle.json, 읽기만)
import json
cd = json.load(open(ROOT / 'results/tables/candle.json'))['candle']
Z = np.array([[r['sz'][k] for k in 'ABCD'] for r in cd])
assert Z.shape == (25, 4) and (Z < 0).sum() == 49 and ((Z >= .2).sum(1) >= 3).sum() == 0  # 원고 문장과 대조
assert round(Z.min(), 2) == -0.23 and round(Z.max(), 2) == 0.29 and round(Z.mean(), 2) == -0.01
fig, ax = plt.subplots(figsize=(6.3, 5.6))
y = np.arange(25)[::-1]
for j, (k, mk, c) in enumerate(zip('ABCD', 'osD^', (INK, GRAY, RED, '#4a6fa5'))):
    ax.scatter(Z[:, j], y, s=16, marker=mk, color=c, label=f'구간 {k}', zorder=3)
ax.scatter(Z.mean(1), y, s=40, marker='|', color=INK, lw=1.4, label='네 구간 평균', zorder=4)
ax.axvline(0, color='#555555', lw=.8); ax.axvline(.2, color=RED, lw=.7, ls='--')
ax.text(.205, 24.6, '채택 기준 0.2', fontsize=7, color=RED, va='bottom')
ax.set_yticks(y, [r['label'] for r in cd], fontsize=6.8)
ax.set_xlabel('표준화 상대수익 차', fontsize=8); ax.tick_params(axis='x', labelsize=7.5)
ax.grid(axis='x', color='#e4e4e4', lw=.5); ax.set_ylim(-.8, 25.4)
ax.legend(frameon=False, fontsize=7.5, ncol=5, loc='upper center', bbox_to_anchor=(.42, 1.06), handlelength=1)
fig.tight_layout(); fig.savefig(A / 'candle_performance.png'); plt.close(fig)
