"""Anger vs Disgust and Fear vs Surprise: is there a separating pattern, and does it hold across sources?

Groups (test windows, model D): correct A, A->B (D's error), correct B.
Features per window (L/R averaged into one muscle, gaze and tongue left out):
  peak     max over the 5 frames, per muscle
  change   last frame - first frame, per muscle (rising / falling)
  size     face size (mean over frames of the sum of the 63 non-tongue AUs), size change, and the
           share of the size carried by each face region (brows, eyes, nose/cheek, mouth)
Steps
  1. Group profiles: median of every feature in the three groups; for each feature, where the error
     group sits between correct A (0) and correct B (1).
  2. Per-source effect: standardized difference (A - B) on TRAINING windows labelled A vs B, for
     each source separately. A pattern "holds across sources" if the sign agrees in all three and
     |d| >= 0.3 in each.
  3. Leave-one-source-out: logistic regression on training windows labelled A vs B from two
     sources, scored (AUC) on the third source's test windows labelled A or B.
  4. Rescue: the same rule fitted on all training sources, applied to D's A->B error windows.
     Share called A = how many errors a pattern of this kind could recover.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
AU = [str(a) for a in Z["au_names"]]
E = [str(c) for c in Z["class_names"]]
SRC_TRAIN = {0: "old", 1: "eh", 2: "new"}
SRCS = ["old", "new", "eh"]


def muscles():
    groups = {}
    for i, a in enumerate(AU):
        if a.startswith(("EyesLook", "Tongue")):
            continue
        base = a
        if a[-1] in "LR" and (a[:-1] + {"L": "R", "R": "L"}[a[-1]]) in AU:
            base = a[:-1]
        for x in ("LB", "LT", "RB", "RT"):
            if a.startswith(("LipFunneler", "LipSuck")) and a.endswith(x):
                base = a[:-2]
        groups.setdefault(base, []).append(i)
    return groups


M = muscles()
REGIONS = {
    "brows": ["BrowLowerer", "InnerBrowRaiser", "OuterBrowRaiser"],
    "eyes": ["EyesClosed", "LidTightener", "UpperLidRaiser", "CheekRaiser"],
    "nose": ["NoseWrinkler", "UpperLipRaiser"],
    "mouth": ["JawDrop", "LipStretcher", "LipCornerPuller", "LipCornerDepressor", "LipPressor",
              "LipTightener", "LipFunneler", "LipPucker", "ChinRaiserB", "ChinRaiserT", "LowerLipDepressor",
              "LipSuck", "MouthLeft", "MouthRight", "JawSidewaysLeft", "JawSidewaysRight", "JawThrust",
              "LipsToward", "DimplerL", "Dimpler", "CheekPuff", "CheekSuck"],
}


def features(X):
    per = {k: X[..., v].mean(-1) for k, v in M.items()}          # (n, 5) per muscle
    f = {}
    for k, s in per.items():
        f[f"peak {k}"] = s.max(1)
        f[f"change {k}"] = s[:, -1] - s[:, 0]
    size = X[..., :63].sum(-1)
    f["size"] = size.mean(1)
    f["size change"] = size[:, -1] - size[:, 0]
    f["size wobble"] = np.abs(np.diff(size, axis=1)).mean(1)
    tot = sum(per[k].mean(1) for k in per) + 1e-6
    for r, names in REGIONS.items():
        f[f"share {r}"] = sum(per[k].mean(1) for k in names if k in per) / tot
    return pd.DataFrame(f)


def analyse(A, B):
    a, b = E.index(A), E.index(B)
    print("=" * 78 + f"\n{A} vs {B}\n" + "=" * 78)
    df = pd.read_csv(REPO / "Analysis/results/test_error_windows.csv")
    Xte = np.concatenate([Z[f"X_test_{s}"] for s in SRCS])
    Fte = features(Xte)
    g = {f"{A} ok": (df.true == A) & (df.pred_D == A), f"{A}->{B}": (df.true == A) & (df.pred_D == B),
         f"{B} ok": (df.true == B) & (df.pred_D == B), f"{B}->{A}": (df.true == B) & (df.pred_D == A)}

    # ---- training windows labelled A or B, per source
    Xtr, Ytr, Str = Z["X_train"], Z["Y_train"], Z["S_train"]
    m = np.isin(Ytr, [a, b])
    Ftr, ytr, str_ = features(Xtr[m]), (Ytr[m] == a).astype(int), np.array([SRC_TRAIN[s] for s in Str[m]])

    # ---- step 2: per-source standardized difference
    d = {}
    for s in SRCS:
        k = str_ == s
        fa, fb = Ftr[k & (ytr == 1)], Ftr[k & (ytr == 0)]
        pooled = np.sqrt((fa.var() + fb.var()) / 2) + 1e-6
        d[s] = (fa.mean() - fb.mean()) / pooled
    d = pd.DataFrame(d)
    consistent = (np.sign(d).nunique(axis=1) == 1) & (d.abs().min(axis=1) >= 0.3)
    d["holds"] = consistent
    d["mean d"] = d[SRCS].mean(1)
    keep = d[consistent].sort_values("mean d", key=abs, ascending=False)
    print(f"\nfeatures that separate {A} (+) from {B} (-) in ALL three sources (|d| >= 0.3, same sign):"
          f" {len(keep)} of {len(d)}")
    print(keep[SRCS + ["mean d"]].round(2).head(14).to_string())
    print("\nstrongest features that do NOT hold across sources (sign flips or weak in one):")
    print(d[~consistent].assign(a=d[SRCS].abs().max(1)).sort_values("a", ascending=False)[SRCS].round(2).head(8).to_string())

    # ---- step 1: where does the error group sit on the consistent features? 0 = like A ok, 1 = like B ok
    cols = list(keep.index)
    if cols:
        med = pd.DataFrame({k: Fte[v.values][cols].median() for k, v in g.items()}).T
        pos = (med.loc[f"{A}->{B}"] - med.loc[f"{A} ok"]) / (med.loc[f"{B} ok"] - med.loc[f"{A} ok"] + 1e-9)
        med.loc["error position (0=A ok, 1=B ok)"] = pos
        med["n"] = [int(v.sum()) for v in g.values()] + [np.nan]
        print(f"\nmedians on those features, test windows (D):")
        print(med[cols[:8] + ["n"]].round(2).to_string())
        print(f"error group position, median over the consistent features: {pos.median():.2f}"
              f"  (share of features where it is closer to {B} ok: {(pos > 0.5).mean():.0%})")

    print("\nstrength and timing, test windows (median):")
    print(pd.DataFrame({k: Fte[v.values][["size", "size change", "size wobble", "share brows", "share eyes",
                                           "share nose", "share mouth"]].median() for k, v in g.items()}).T.round(3).to_string())

    # ---- step 3: leave-one-source-out rule
    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    yte_all = df.true.values
    src_te = df.source.values
    print("\nleave-one-source-out: rule fitted on training windows of two sources, AUC on the third (test):")
    for held in SRCS:
        for feats, name in ((list(d.index), "all features"), (cols, "consistent only")):
            if not feats:
                continue
            clf = model().fit(Ftr[str_ != held][feats], ytr[str_ != held])
            k = (src_te == held) & np.isin(yte_all, [A, B])
            auc = roc_auc_score(yte_all[k] == A, clf.predict_proba(Fte[k][feats])[:, 1])
            print(f"  held-out {held:<4} {name:<16} AUC {auc:.3f}  (n={k.sum()})")
    # D's own AUC on the same pairs, for reference
    P = np.concatenate([np.zeros(0)])  # placeholder, D's probabilities are not in the CSV
    print("  (for reference, D's p_true-based ranking is not comparable; see rescue below)")

    # ---- step 4: rescue of D's errors
    for feats, name in ((list(d.index), "all features"), (cols, "consistent only")):
        if not feats:
            continue
        clf = model().fit(Ftr[feats], ytr)
        p = clf.predict_proba(Fte[feats])[:, 1]
        for k, v in g.items():
            print(f"  rule ({name:<15}) calls {A}: {k:<18} {np.mean(p[v.values] > 0.5):.0%}  (n={v.sum()})")
        e = g[f"{A}->{B}"].values
        for s in SRCS:
            k = e & (src_te == s)
            if k.sum():
                print(f"      {A}->{B} errors from {s:<4}: rule calls {A} in {np.mean(p[k] > 0.5):.0%} (n={k.sum()})")
        if name == "consistent only":
            coef = pd.Series(clf[-1].coef_[0], index=feats).sort_values()
            print(f"  rule weights (consistent features; + = {A}, - = {B}):")
            print("   ", ", ".join(f"{k} {v:+.2f}" for k, v in coef.items() if abs(v) > 0.05))

    # ---- combinations: the two-cue pattern
    return d


def combos():
    print("\n" + "=" * 78 + "\nAU combinations (test, model D): share of windows with each cue pattern\n" + "=" * 78)
    df = pd.read_csv(REPO / "Analysis/results/test_error_windows.csv")
    X = np.concatenate([Z[f"X_test_{s}"] for s in SRCS])
    F = features(X)
    on = lambda k: F[f"peak {k}"] > 0.15
    pats = {
        "Anger/Disgust": {
            "nose only (NoseWrinkler or UpperLipRaiser, no lip press/tighten)": (on("NoseWrinkler") | on("UpperLipRaiser")) & ~(on("LipPressor") | on("LipTightener")),
            "lips only (LipPressor or LipTightener, no nose)": ~(on("NoseWrinkler") | on("UpperLipRaiser")) & (on("LipPressor") | on("LipTightener")),
            "nose + lips": (on("NoseWrinkler") | on("UpperLipRaiser")) & (on("LipPressor") | on("LipTightener")),
            "brow lowerer + upper lid raiser (glare)": on("BrowLowerer") & on("UpperLidRaiser"),
            "brow lowerer + lid tightener": on("BrowLowerer") & on("LidTightener"),
        },
        "Fear/Surprise": {
            "brows up + lids up + jaw drop (surprise)": (on("InnerBrowRaiser") | on("OuterBrowRaiser")) & on("UpperLidRaiser") & on("JawDrop"),
            "brows up + brow lowerer (fear brow)": (on("InnerBrowRaiser")) & on("BrowLowerer"),
            "lip stretcher": on("LipStretcher"),
            "lids up, no jaw drop": on("UpperLidRaiser") & ~on("JawDrop"),
            "jaw drop, brows not up": on("JawDrop") & ~(on("InnerBrowRaiser") | on("OuterBrowRaiser")),
        },
    }
    for pair, A, B in (("Anger/Disgust", "Anger", "Disgust"), ("Fear/Surprise", "Fear", "Surprise")):
        g = {f"{A} ok": (df.true == A) & (df.pred_D == A), f"{A}->{B}": (df.true == A) & (df.pred_D == B),
             f"{B} ok": (df.true == B) & (df.pred_D == B)}
        t = pd.DataFrame({k: {p: v[m.values].mean() for p, v in pats[pair].items()} for k, m in g.items()})
        print(f"\n{pair}")
        print(t.round(2).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    analyse("Anger", "Disgust")
    analyse("Fear", "Surprise")
    combos()
