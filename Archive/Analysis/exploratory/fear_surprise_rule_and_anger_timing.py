"""(A) The Fear/Surprise jaw-drop tie-breaker, tested widely. (B) When in an Anger expression does D say Disgust?

(A) Pipeline: model D decides; if D's top class is Fear or Surprise, a 2-feature logistic rule
    (peak JawDrop over the 5 frames, share of the face movement in the eye region; see
    pair_separation.py) re-decides between Fear and Surprise. Nothing else changes.
    The rule is fitted on TRAINING windows labelled Fear or Surprise, from one of:
        VR (old + new acted), all sources, old only, new only, EmojiHero only.
    Scored, 7-class macro-F1, against D alone:
        validation and test, per source (window level);
        headset replay, every session-local-v1 session, evaluation block, raw tick macro-F1,
        per session (better / worse), and Fear / Surprise tick accuracy.
    Threshold check: the rule's Fear probability cut at 0.3 / 0.4 / 0.5 / 0.6 / 0.7 (VR fit).

(B) Anger timing.
    Headset: every trial (calibration + evaluation blocks) prompted Anger; D's prediction per tick
    (1-9, 1 tick per second); share of Disgust per tick; face size and nose share per tick; and
    whether a trial-majority vote fixes the trial.
    Acted test (old, new): windows are stored in recording order; a stretch is cut where the jump
    between one window's last frame and the next window's first frame is larger than the 99th
    percentile of the frame-to-frame jumps inside windows, or the label changes. Position of each
    Anger window in its stretch (first third / middle / last third) and its size change (rising,
    steady, falling) against D's prediction.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
sys.path.insert(0, str(REPO / "Analysis/exploratory"))
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import calibration_study as cs
import session_analysis as sa
from jitter_headset_replay import sessions
from pair_separation import features, Z, E

FEAR, SUR, ANG, DIS = E.index("Fear"), E.index("Surprise"), E.index("Anger"), E.index("Disgust")
FEATS = ["peak JawDrop", "share eyes"]
D_PATH = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
SESS = ort.InferenceSession(str(D_PATH))
run_d = lambda X: SESS.run(None, {SESS.get_inputs()[0].name: X.astype(np.float32)})[0]
FITS = {"VR (old+new)": [0, 2], "all sources": [0, 1, 2], "old only": [0], "new only": [2], "EmojiHero only": [1]}
mf1 = lambda y, p: f1_score(y, p, labels=range(7), average="macro", zero_division=0)


def fit_rule(sources):
    m = np.isin(Z["Y_train"], [FEAR, SUR]) & np.isin(Z["S_train"], sources)
    F = features(Z["X_train"][m])[FEATS]
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(F, Z["Y_train"][m] == FEAR)


def apply_rule(rule, X, pred, cut=0.5):
    out = pred.copy()
    k = np.isin(pred, [FEAR, SUR])
    if k.any():
        pf = rule.predict_proba(features(X[k])[FEATS])[:, 1]
        out[k] = np.where(pf > cut, FEAR, SUR)
    return out


def headset_data():
    out = []
    for d in sessions():
        data = cs.replay(sa.load_session(d / "predictions.csv"), D_PATH)
        out.append((d.name[8:], data))
    return out


def part_a(HS):
    print("=" * 78 + "\n(A) FEAR/SURPRISE TIE-BREAKER, 7-class macro-F1, D alone vs D + rule\n" + "=" * 78)
    rules = {k: fit_rule(v) for k, v in FITS.items()}
    windows = {f"{sp} {s}": (Z[f"X_{sp}_{s}"], Z[f"Y_{sp}_{s}"]) for sp in ("val", "test") for s in ("old", "new", "eh")}
    rows = []
    for name, (X, y) in windows.items():
        p = run_d(X).argmax(1)
        r = {"data": name, "D": mf1(y, p)}
        for k, rule in rules.items():
            r[k] = mf1(y, apply_rule(rule, X, p))
        rows.append(r)

    hs_rows, fs = [], []
    for name, data in HS:
        _, te = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
        X, y = data.windows[te], data.prompted[te]
        p = data.scores[te].argmax(1)
        yt = [E[i] for i in y]
        r = {"data": f"headset {name}", "D": sa.macro_f1(yt, [E[i] for i in p], E)["macro_f1"]}
        for k, rule in rules.items():
            q = apply_rule(rule, X, p)
            r[k] = sa.macro_f1(yt, [E[i] for i in q], E)["macro_f1"]
            for c in (FEAR, SUR):
                fs.append({"session": name, "fit": k, "class": E[c], "n": int((y == c).sum()),
                           "D": int(((p == c) & (y == c)).sum()), "rule": int(((q == c) & (y == c)).sum())})
        hs_rows.append(r)
    tab = pd.DataFrame(rows + hs_rows).set_index("data")
    print(tab.round(3).to_string())
    hs = tab[tab.index.str.startswith("headset")]
    print("\nheadset mean      ", hs.mean().round(3).to_dict())
    for k in FITS:
        print(f"headset sessions better / worse than D, fit {k:<15}: {(hs[k] > hs.D).sum()} / {(hs[k] < hs.D).sum()}")
    f = pd.DataFrame(fs).groupby(["fit", "class"])[["n", "D", "rule"]].sum()
    print("\nheadset Fear and Surprise ticks correct (all sessions), D vs D + rule:")
    print(f.to_string())

    print("\nthreshold check (VR fit): mean 7-class macro-F1")
    rule = rules["VR (old+new)"]
    thr = []
    for cut in (0.3, 0.4, 0.5, 0.6, 0.7):
        r = {"cut": cut}
        for name, (X, y) in windows.items():
            if name.startswith("test"):
                r[name] = mf1(y, apply_rule(rule, X, run_d(X).argmax(1), cut))
        vals = []
        for _, data in HS:
            _, te = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
            q = apply_rule(rule, data.windows[te], data.scores[te].argmax(1), cut)
            vals.append(sa.macro_f1([E[i] for i in data.prompted[te]], [E[i] for i in q], E)["macro_f1"])
        r["headset"] = np.mean(vals)
        thr.append(r)
    print(pd.DataFrame(thr).round(3).to_string(index=False))
    tab.to_csv(REPO / "Analysis/results/fear_surprise_rule.csv")


def part_b(HS):
    print("\n" + "=" * 78 + "\n(B) ANGER TIMING\n" + "=" * 78)
    rows = []
    for name, data in HS:
        k = data.prompted == ANG
        p = data.scores.argmax(1)
        F = features(data.windows[k])
        for i, j in enumerate(np.where(k)[0]):
            rows.append({"session": name, "trial": data.trial[j], "tick": int(data.tick[j]), "pred": E[p[j]],
                         "size": F["size"].iloc[i], "share nose": F["share nose"].iloc[i],
                         "size change": F["size change"].iloc[i]})
    h = pd.DataFrame(rows)
    h = h[h.tick >= 1]
    print(f"headset: {h.groupby(['session', 'trial']).ngroups} Anger trials ({h.session.nunique()} sessions), {len(h)} ticks")
    t = pd.crosstab(h.tick, h.pred, normalize="index").round(2)
    t["n"] = h.groupby("tick").size()
    t["size"] = h.groupby("tick")["size"].median().round(2)
    t["nose share"] = h.groupby("tick")["share nose"].median().round(2)
    print(t.to_string())
    h["phase"] = pd.cut(h.tick, [0, 3, 6, 9], labels=["ticks 1-3", "ticks 4-6", "ticks 7-9"])
    print("\nheadset Anger, share predicted Anger / Disgust by phase:")
    print(pd.crosstab(h.phase, h.pred, normalize="index").reindex(columns=["Anger", "Disgust", "Neutral"], fill_value=0).round(2).to_string())
    print("\nper session, Disgust share of Anger ticks:", h.groupby("session").pred.apply(lambda s: round((s == 'Disgust').mean(), 2)).to_dict())
    votes = h.groupby(["session", "trial"]).pred.agg(lambda s: s.value_counts().index[0])
    print("trial-majority vote of Anger trials:", votes.value_counts().to_dict(),
          " | tick accuracy:", round((h.pred == "Anger").mean(), 2))

    # acted test windows
    df = pd.read_csv(REPO / "Analysis/results/test_error_windows.csv")
    rows = []
    for src in ("old", "new"):
        X, Y = Z[f"X_test_{src}"], Z[f"Y_test_{src}"]
        inner = np.quantile(np.abs(np.diff(X, axis=1)).sum(-1), 0.99)
        jump = np.abs(X[1:, 0] - X[:-1, -1]).sum(-1)
        cut = np.r_[True, (jump > inner) | (Y[1:] != Y[:-1])]
        stretch = np.cumsum(cut)
        d = df[df.source == src].reset_index(drop=True)
        F = features(X)
        for sid in np.unique(stretch):
            idx = np.where(stretch == sid)[0]
            if Y[idx[0]] != ANG:
                continue
            for pos, i in enumerate(idx):
                rows.append({"source": src, "stretch": f"{src}{sid}", "len": len(idx), "pos": pos,
                             "rel": pos / max(len(idx) - 1, 1), "pred": d.pred_D[i],
                             "size change": F["size change"][i], "size": F["size"][i]})
    a = pd.DataFrame(rows)
    print(f"\nacted test Anger: {len(a)} windows in {a.stretch.nunique()} stretches "
          f"(median length {a.groupby('stretch').len.first().median():.0f} windows = "
          f"{a.groupby('stretch').len.first().median() * 0.5:.1f} s)")
    long = a[a.len >= 3].copy()
    long["third"] = pd.cut(long.rel, [-0.01, 1 / 3, 2 / 3, 1.0], labels=["first third", "middle", "last third"])
    print("stretches of 3+ windows, share predicted Anger / Disgust by position:")
    print(pd.crosstab(long.third, long.pred, normalize="index").reindex(columns=["Anger", "Disgust", "Neutral"], fill_value=0).round(2).to_string(),
          "\n n:", long.third.value_counts().sort_index().to_dict())
    a["motion"] = pd.cut(a["size change"], [-99, -0.5, 0.5, 99], labels=["falling", "steady", "rising"])
    print("\nby size change inside the window (all acted test Anger):")
    t = pd.crosstab(a.motion, a.pred, normalize="index").reindex(columns=["Anger", "Disgust", "Neutral"], fill_value=0).round(2)
    t["n"] = a.motion.value_counts()
    print(t.to_string())
    eh = df[(df.source == "eh") & (df.true == "Anger")].copy()
    eh["motion"] = pd.cut(features(Z["X_test_eh"][(df[df.source == "eh"].true == "Anger").values])["size change"].values,
                          [-99, -0.5, 0.5, 99], labels=["falling", "steady", "rising"])
    print("\nEmojiHero test Anger (shuffled, no order), by size change:")
    t = pd.crosstab(eh.motion, eh.pred_D, normalize="index").reindex(columns=["Anger", "Disgust", "Neutral"], fill_value=0).round(2)
    t["n"] = eh.motion.value_counts()
    print(t.to_string())
    votes = a.groupby("stretch").pred.agg(lambda s: s.value_counts().index[0])
    print("\nacted test stretch-majority vote for Anger:", votes.value_counts().to_dict(),
          "| window accuracy:", round((a.pred == "Anger").mean(), 2))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    HS = headset_data()
    if "--b-only" not in sys.argv:
        part_a(HS)
    part_b(HS)
