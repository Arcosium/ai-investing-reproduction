"""Within-decision Spearman correlations, averaged by evaluation window."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEG = {3: '24.06~08', 2: '24.12~02', 5: '25.06~08', 0: '25.12~02', 8: 'hold 26.03~05'}


def main():
    data = pd.read_csv(ROOT / 'data/score_inputs.csv.gz')
    if data.duplicated(['ts', 'base']).any():
        raise ValueError('duplicate decision/asset')
    rows = []
    for (fold, ts), x in data.groupby(['fold', 'ts'], sort=False):
        for model in ('gbm', 'cnn10'):
            # Original alignment first selects rows with a model prediction.
            g = x.dropna(subset=[model])
            if g.empty:
                continue
            rows.append({'model': model, 'seg': SEG[fold], 'ts': int(ts),
                         'ret_W': g[model].corr(g.ret_W, method='spearman'),
                         'vol_W': g[model].corr(g.vol_W, method='spearman'),
                         'n': int(g.ret_W.notna().sum())})
    result = pd.DataFrame(rows).groupby(['model', 'seg'], sort=False).agg(
        ret_W=('ret_W', 'mean'), vol_W=('vol_W', 'mean'), ts=('ts', 'size'), n=('n', 'median'))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    result.to_csv(out / 'score_correlations.csv')
    print(result.round(3).to_string())


if __name__ == '__main__':
    main()
