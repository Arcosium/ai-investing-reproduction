"""동결 가중치로 heatf CNN 점수만 다시 낸다(학습·선택 없음) — 표본 외 구간을 뒤로 늘릴 때.
데이터 준비·횡단면 순위·표준화(json 의 feat_mu/feat_sd)·예측은 train_cnn 과 같은 함수를 쓰므로 같은 봉이면 같은 점수가 나온다.
usage: (cd "$RESEARCH_DATA" && MARKET_BARS="$RESEARCH_DATA/work/bars_hold" python3 -m market.rescore_cnn --stem work/results/hold_i1_heatf{,_sd1}_4h_W60_H84_s5 --test 2026-06 --test_end 2026-10 --out_dir work/results/ext)
  → <out_dir>/<stem 이름>_pred.npz. 여러 stem 은 같은 종목·설정(시드만 다름)이어야 하고 창은 한 번만 만든다.
"""
import argparse, json, os, types
import numpy as np
import torch
from market import train_cnn as T
from market.pilot_gbm import add_xs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stem", nargs="+", required=True); ap.add_argument("--test", required=True); ap.add_argument("--test_end", required=True); ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(); metas = [json.load(open(s + ".json")) for s in a.stem]; ar = types.SimpleNamespace(**metas[0]["args"])
    same = lambda m: {k: v for k, v in m["args"].items() if k not in ("seed", "out")}
    if any(same(m) != same(metas[0]) for m in metas) or ar.render != "heatf" or ar.model != "i1" or not ar.bases.startswith("@"):
        raise ValueError("같은 설정(heatf i1 + @bases)의 시드만 함께 채점한다")
    if not all(m.get("feat_cols") for m in metas):
        raise ValueError("json 에 표준화 통계(feat_cols·feat_mu·feat_sd)가 없는 초기 모델")
    T.PX, T.IMG_H = ar.px, ar.img_h
    bases = [l.strip() for l in open(ar.bases[1:]) if l.strip()]
    wins = T.Windows(bases, ar.res, ar.W, ar.H, ar.k, "cuda", ar.start, a.test_end, ar.label, ar.fixed, need_feats=True, stride=ar.stride or None)
    ix = add_xs(wins.idx); wins.idx = ix
    te = ix[(ix.ts >= T.ms(a.test)) & (ix.ts < T.ms(a.test_end))]; g = torch.tensor(te.g.to_numpy(), device="cuda")
    os.makedirs(a.out_dir, exist_ok=True)
    for stem, meta in zip(a.stem, metas):
        f = torch.tensor(((te[meta["feat_cols"]] - np.array(meta["feat_mu"])) / np.array(meta["feat_sd"])).to_numpy(np.float32), device="cuda")
        model = T.build_model(ar, ar.W).to("cuda"); model.load_state_dict(torch.load(stem + ".pt", map_location="cuda"))
        pt = T.predict(model, wins, g, f, ar); out = os.path.join(a.out_dir, os.path.basename(stem) + "_pred.npz")
        np.savez_compressed(out, ts=te.ts.to_numpy(), base=te.base.to_numpy().astype(str), y=te.label.to_numpy(), p=pt, fwd=te.fwd.to_numpy(), sigH=te.sigH.to_numpy())
        print("RESCORE", out, len(te), flush=True)


if __name__ == "__main__":
    main()
