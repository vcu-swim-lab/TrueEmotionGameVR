"""Weak acted windows in training: a participant-held-out diagnostic with a simple classifier. D unchanged.

Weak windows: ungated, non-overlapping 5-frame windows of the training participants' EMOTION recordings
that FAIL the 0.25 gate (same raw pipeline as the notebook). Each is put in one category; thresholds
that need data come from the training fold only:
  tracking   a frame with all 70 AUs = 0, or a frame-to-frame jump (sum |change|) above the 99.9th
             percentile of jumps inside the fold's gated training windows
  edge       within the first or last 20% of its recording (onset / offset)
  changing   |strength(last frame) - strength(first frame)| > 0.5 x mean strength (a transition)
  resting    window strength (sum of expressive AUs, mean over frames) <= 90th percentile of the
             fold's Neutral training windows
  sustained  everything else -> ELIGIBLE (a held weak expression; labelled with its recording's emotion)
Models (same features = 350 raw values; same logistic regression; C chosen from the same 12-value grid
by inner GroupKFold(3) on participants, scored on the inner held-out normal + eligible weak windows):
  N  normal (gated) training windows only
  W  N + eligible weak windows of the fold's TRAINING participants
Outer GroupKFold(7) on participants (the same folds as strength_pattern_diagnostic.py).
Both models are scored on identical held-out windows: normal gated windows, and weak windows of each
category of the held-out participants.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
sys.path.insert(0, str(REPO / "Analysis/exploratory"))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import teacher_preprocessing as tp
from strength_pattern_diagnostic import recordings, build, EXPR, E, ANG, NEU

CACHE = REPO / "Analysis/data/weak_windows_with_position.npz"


def weak_windows():
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        return {k: c[k] for k in c.files}
    X, Y, P, S, POS = [], [], [], [], []
    for study, p, emo, frames in recordings():
        if emo == "Neutral":
            continue
        W = tp.make_windows(frames.astype(np.float32), emo, gate=False)
        for i, w in enumerate(W):
            if not tp.window_passes_gate(w, emo):
                X.append(w); Y.append(E.index(emo)); P.append(f"{study}:{p}"); S.append(study)
                POS.append(i / max(len(W) - 1, 1))
    out = {"X": np.stack(X), "Y": np.array(Y), "P": np.array(P), "S": np.array(S), "pos": np.array(POS)}
    np.savez_compressed(CACHE, **out)
    return out


def strength(X):
    return X[..., EXPR].sum(-1)                     # (n, 5)


def categorise(Xw, pos, Xg_train, Yg_train):
    jumps_g = np.abs(np.diff(Xg_train, axis=1)).sum(-1).ravel()
    jump_cut = np.quantile(jumps_g, 0.999)
    rest_cut = np.quantile(strength(Xg_train[Yg_train == NEU]).mean(1), 0.9)
    s = strength(Xw)
    zero = (Xw.max(-1) == 0).any(1)
    jump = np.abs(np.diff(Xw, axis=1)).sum(-1).max(1) > jump_cut
    edge = (pos < 0.2) | (pos > 0.8)
    changing = np.abs(s[:, -1] - s[:, 0]) > 0.5 * np.maximum(s.mean(1), 1e-6)
    resting = s.mean(1) <= rest_cut
    return np.select([zero | jump, edge, changing, resting], ["tracking", "edge", "changing", "resting"], "sustained")


def model(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


flat = lambda X: X.reshape(len(X), -1).astype(np.float64)


def fit_eval(Xtr, Ytr, Ptr, Xte_sets, inner_extra):
    """Choose C by inner GroupKFold(3) on participants; score on inner held-out normal + eligible weak."""
    best, bestC = -1, None
    for C in np.logspace(-3, 1, 12):
        sc = []
        for a, b in GroupKFold(3).split(Xtr, Ytr, Ptr):
            m = model(C).fit(flat(Xtr[a]), Ytr[a])
            held = set(Ptr[b])
            xs, ys = [Xtr[b]], [Ytr[b]]
            ex = np.isin(inner_extra[2], list(held))
            if ex.any():
                xs.append(inner_extra[0][ex]); ys.append(inner_extra[1][ex])
            Xi, Yi = np.concatenate(xs), np.concatenate(ys)
            sc.append(f1_score(Yi, m.predict(flat(Xi)), labels=range(7), average="macro", zero_division=0))
        if np.mean(sc) > best:
            best, bestC = np.mean(sc), C
    m = model(bestC).fit(flat(Xtr), Ytr)
    return [m.predict(flat(x)) for x in Xte_sets], bestC


def main():
    pd.set_option("display.width", 250)
    d = build()
    ok = d["participant"] != ""
    X, Y, S, P = d["X"][ok], d["Y"][ok], d["S"][ok], d["participant"][ok]
    w = weak_windows()
    keep = np.isin(w["P"], list(set(P)))
    Xw, Yw, Pw, Sw, pos = w["X"][keep], w["Y"][keep], w["P"][keep], w["S"][keep], w["pos"][keep]

    cat_all = categorise(Xw, pos, X, Y)   # report only (all training data); folds recompute
    print(f"weak windows (fail gate, emotion recordings, 28 participants): {len(Yw)}")
    print(pd.crosstab(pd.Series(Yw).map(dict(enumerate(E))), cat_all, margins=True).to_string())
    print("sustained per participant: min/median/max",
          pd.Series(Pw[cat_all == "sustained"]).value_counts().reindex(sorted(set(P)), fill_value=0).describe()[["min", "50%", "max"]].to_dict())

    pN, pW = np.zeros(len(Y), int), np.zeros(len(Y), int)
    wN, wW = np.full(len(Yw), -1), np.full(len(Yw), -1)
    cat_fold = np.empty(len(Yw), object)
    chosen = []
    for k, (tr, te) in enumerate(GroupKFold(7).split(X, Y, P)):
        trP, teP = set(P[tr]), set(P[te])
        cat = categorise(Xw, pos, X[tr], Y[tr])
        wtr = np.isin(Pw, list(trP)) & (cat == "sustained")
        wte = np.isin(Pw, list(teP))
        cat_fold[wte] = cat[wte]
        extra = (Xw[wtr], Yw[wtr], Pw[wtr])
        (a, b), cN = fit_eval(X[tr], Y[tr], P[tr], [X[te], Xw[wte]], extra)
        pN[te], wN[wte] = a, b
        (a, b), cW = fit_eval(np.concatenate([X[tr], Xw[wtr]]), np.concatenate([Y[tr], Yw[wtr]]),
                              np.concatenate([P[tr], Pw[wtr]]), [X[te], Xw[wte]], extra)
        pW[te], wW[wte] = a, b
        chosen.append((k, round(cN, 4), round(cW, 4), int(wtr.sum())))
        print(f"fold {k + 1}/7: weak windows added {wtr.sum()}", flush=True)
    print("chosen C (N, W) and weak windows added per fold:", chosen)

    src = np.where(S == 0, "old", "new")
    print("\n== normal (gated) held-out windows")
    rows = []
    for name, m in (("old", src == "old"), ("new", src == "new"), ("pooled", np.ones(len(Y), bool))):
        for arm, p in (("N", pN), ("W", pW)):
            y, q = Y[m], p[m]
            emo = y != NEU
            rows.append({"data": name, "model": arm,
                         "macro-F1": f1_score(y, q, labels=range(7), average="macro", zero_division=0),
                         "emotion->Neutral": int((emo & (q == NEU)).sum()), "Neutral->Anger": int(((y == NEU) & (q == ANG)).sum()),
                         "other emotion->Anger": int((emo & (y != ANG) & (q == ANG)).sum()),
                         "Neutral correct": f"{int(((y == NEU) & (q == NEU)).sum())}/{int((y == NEU).sum())}"})
    print(pd.DataFrame(rows).set_index(["data", "model"]).round(3).to_string())

    print("\n== held-out weak windows, by category (correct / ->Neutral / other->Anger)")
    rows = []
    for c in ("sustained", "resting", "changing", "edge", "tracking"):
        for s in ("old", "new"):
            m = (cat_fold == c) & (Sw == s)
            if not m.any():
                continue
            for arm, p in (("N", wN), ("W", wW)):
                rows.append({"category": c, "source": s, "model": arm, "n": int(m.sum()),
                             "correct": int((p[m] == Yw[m]).sum()), "->Neutral": int((p[m] == NEU).sum()),
                             "other->Anger": int(((Yw[m] != ANG) & (p[m] == ANG)).sum())})
    print(pd.DataFrame(rows).set_index(["category", "source", "model"]).to_string())

    print("\n== held-out SUSTAINED weak windows, per emotion (correct of n; ->Neutral)")
    m = cat_fold == "sustained"
    t = pd.DataFrame({e: {"n": int((Yw[m] == i).sum()),
                          "N correct": int(((Yw == i) & m & (wN == i)).sum()), "W correct": int(((Yw == i) & m & (wW == i)).sum()),
                          "N ->Neutral": int(((Yw == i) & m & (wN == NEU)).sum()), "W ->Neutral": int(((Yw == i) & m & (wW == NEU)).sum())}
                      for i, e in enumerate(E) if e != "Neutral"}).T
    print(t.to_string())
    print("\nper-class F1 on normal windows, pooled")
    print(pd.DataFrame({a: f1_score(Y, p, labels=range(7), average=None, zero_division=0) for a, p in (("N", pN), ("W", pW))},
                       index=E).T.round(3).to_string())

    print("\n== consistency per held-out participant")
    accs = pd.DataFrame({q: {"sustained N": np.mean(wN[m & (Pw == q)] == Yw[m & (Pw == q)]) if (m & (Pw == q)).any() else np.nan,
                             "sustained W": np.mean(wW[m & (Pw == q)] == Yw[m & (Pw == q)]) if (m & (Pw == q)).any() else np.nan,
                             "normal F1 N": f1_score(Y[P == q], pN[P == q], labels=range(7), average="macro", zero_division=0),
                             "normal F1 W": f1_score(Y[P == q], pW[P == q], labels=range(7), average="macro", zero_division=0)}
                         for q in sorted(set(P))}).T
    dw = accs["sustained W"] - accs["sustained N"]
    dn = accs["normal F1 W"] - accs["normal F1 N"]
    for name, dd in (("sustained weak accuracy", dw), ("normal macro-F1", dn)):
        dd = dd.dropna()
        print(f"  {name}: participants better / worse {int((dd > 0.005).sum())} / {int((dd < -0.005).sum())} of {len(dd)}"
              f" | old mean {dd[[i.startswith('old') for i in dd.index]].mean():+.3f}, new mean {dd[[i.startswith('new') for i in dd.index]].mean():+.3f}")


if __name__ == "__main__":
    main()
