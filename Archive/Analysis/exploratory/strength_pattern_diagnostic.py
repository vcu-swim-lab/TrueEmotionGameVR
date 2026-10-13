"""Expression strength vs expression pattern: a participant-held-out diagnostic with a simple classifier.

No neural network, no change to D. Training windows only (old + new acted, the npz TRAIN windows).

Participants
  Each npz training window is matched EXACTLY to a window rebuilt from the raw recordings with the
  notebook's pipeline (old: as stored; new: clean + mean of 6 rows; both gated). Windows that match
  no participant or more than one are excluded and counted.

Features (one window = 5 frames x 70 AUs)
  EXPR      expressive channels: all except gaze (EyesLook*), blink (EyesClosed*) and tongue (Tongue*).
  A  raw            the 350 raw values (gaze kept).
  B  raw+strength   A + per-frame strength (5) + window strength + log window strength.
  C  raw+str+rel    B + relative expressive AUs: X[:, EXPR] / max(window strength, floor) (5 x 53).
  strength  per frame: L1 = sum of EXPR values, or L2 = sqrt(sum of squares); window strength =
            mean of the 5 frame strengths. Normalisation is PER WINDOW (one denominator for all 5
            frames), so the movement over the 5 frames is kept and single noisy frames are not
            rescaled on their own.
  floor     quantile q of the window strength of the training fold's NEUTRAL windows (resting faces),
            q in {0.5, 0.9}. A face weaker than the floor is divided by the floor, so a resting face or
            tracking noise is never blown up to a full-size pattern.

Classifier: StandardScaler + multinomial logistic regression, class_weight balanced. Every feature
set gets the same tuning budget, 12 configurations chosen by inner GroupKFold(3) on participants:
  A: C in 12 log-spaced values;  B: C (6) x strength (2);  C: C (3) x strength (2) x floor (2).
Outer: GroupKFold(7) on participants; all preprocessing (scaler, floor, weak threshold) fitted on
the training fold only.

Evaluation (held-out participants, pooled over outer folds)
  gated windows (the training distribution): macro-F1 per source, per-class F1, emotion->Neutral,
    Neutral->Anger, other emotion->Anger, per participant.
  weak windows: emotion windows whose L1 window strength is below the training fold's 25th percentile
    of emotion-window strength; weak Anger vs Neutral / Happiness / Disgust.
  ungated weak windows: windows of the held-out participants' emotion recordings that FAIL the 0.25
    gate but have an expressive AU > 0.10 (real weak acting, never in any training set here).
  matched strength: accuracy of A vs C inside strength tertiles (training-fold cut points).
Headset replay (exploratory, after the CV): models fitted on all mapped participants with the config
chosen by participant CV on all of them; developer sessions and 26908f separately.
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

Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
AU = [str(a) for a in Z["au_names"]]
E = [str(c) for c in Z["class_names"]]
ANG, DIS, HAP, NEU = E.index("Anger"), E.index("Disgust"), E.index("Happiness"), E.index("Neutral")
EXPR = np.array([not a.startswith(("EyesLook", "EyesClosed", "Tongue")) for a in AU])
OLD = REPO / "Analysis/data/filtered-dataset-20260924T162300Z-1-001/filtered-dataset"
NEW = Path(r"C:\Users\alime\multimodal-vr-emotion-study\multimodal-vr-emotion-study-main")
CACHE = REPO / "Analysis/data/train_window_participants.npz"
key = lambda w: np.round(w, 5).tobytes()


# ---------------------------------------------------------------- data
def recordings():
    for p in sorted(d for d in OLD.iterdir() if d.is_dir()):
        for f in sorted(p.glob("ACT_*.csv")):
            df = pd.read_csv(f)
            yield "old", p.name, f.stem.split("_", 1)[1].split("(")[0], np.nan_to_num(df[AU].to_numpy(np.float64))
    for f in sorted((NEW / "ActingRecordings").glob("*/*/*.csv")):
        raw = pd.read_csv(f)[AU].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        yield "new", f.parents[1].name, f.parent.name, tp.downsample(tp.clean(raw))


def build():
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        return {k: c[k] for k in c.files}
    owner, weak = {}, []
    for study, p, emo, X in recordings():
        X = X.astype(np.float32)
        for w in tp.make_windows(X, emo):
            owner.setdefault(key(w), set()).add(f"{study}:{p}")
        if emo != "Neutral":
            for w in tp.make_windows(X, emo, gate=False):
                if not tp.window_passes_gate(w, emo) and w[..., EXPR].max() > 0.10:
                    weak.append((f"{study}:{p}", E.index(emo), w))
    m = np.isin(Z["S_train"], [0, 2])
    X, Y, S = Z["X_train"][m], Z["Y_train"][m], Z["S_train"][m]
    part = np.array([next(iter(owner[key(w)])) if len(owner.get(key(w), ())) == 1 else "" for w in X])
    out = {"X": X, "Y": Y, "S": S, "participant": part,
           "Xw": np.stack([w for _, _, w in weak]).astype(np.float32),
           "Yw": np.array([y for _, y, _ in weak]), "Pw": np.array([p for p, _, _ in weak])}
    np.savez_compressed(CACHE, **out)
    return out


# ---------------------------------------------------------------- features
def strength(X, kind):
    x = X[..., EXPR]
    f = x.sum(-1) if kind == "L1" else np.sqrt((x ** 2).sum(-1))
    return f, f.mean(1)                                    # (n, 5), (n,)


def feats(X, fs, kind="L1", floor=None):
    X = X.astype(np.float64)
    raw = X.reshape(len(X), -1)
    if fs == "A":
        return raw
    fr, ws = strength(X, kind)
    b = np.hstack([raw, fr, ws[:, None], np.log(ws + 1e-3)[:, None]])
    if fs == "B":
        return b
    rel = X[..., EXPR] / np.maximum(ws, floor)[:, None, None]
    return np.hstack([b, rel.reshape(len(X), -1)])


def grid(fs):
    if fs == "A":
        return [{"C": c} for c in np.logspace(-3, 1, 12)]
    if fs == "B":
        return [{"C": c, "kind": k} for c in np.logspace(-3, 1, 6) for k in ("L1", "L2")]
    return [{"C": c, "kind": k, "q": q} for c in (0.003, 0.03, 0.3) for k in ("L1", "L2") for q in (0.5, 0.9)]


def fit(fs, cfg, X, Y):
    floor = None
    if fs == "C":
        floor = max(float(np.quantile(strength(X[Y == NEU].astype(np.float64), cfg["kind"])[1], cfg["q"])), 1e-3)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg["C"], max_iter=3000, class_weight="balanced"))
    clf.fit(feats(X, fs, cfg.get("kind", "L1"), floor), Y)
    return lambda Xn: clf.predict(feats(Xn, fs, cfg.get("kind", "L1"), floor)), floor


def choose(fs, X, Y, P):
    best, best_cfg = -1, None
    for cfg in grid(fs):
        sc = []
        for tr, te in GroupKFold(3).split(X, Y, P):
            pred, _ = fit(fs, cfg, X[tr], Y[tr])
            sc.append(f1_score(Y[te], pred(X[te]), labels=range(7), average="macro", zero_division=0))
        if np.mean(sc) > best:
            best, best_cfg = np.mean(sc), cfg
    return best_cfg


# ---------------------------------------------------------------- evaluation helpers
def errs(y, p):
    emo = y != NEU
    return {"macro-F1": f1_score(y, p, labels=range(7), average="macro", zero_division=0),
            "emotion->Neutral": int((emo & (p == NEU)).sum()), "Neutral->Anger": int(((y == NEU) & (p == ANG)).sum()),
            "other emotion->Anger": int((emo & (y != ANG) & (p == ANG)).sum()), "n": len(y)}


def main():
    pd.set_option("display.width", 250)
    d = build()
    X, Y, S, P = d["X"], d["Y"], d["S"], d["participant"]
    ok = P != ""
    print(f"training windows: {len(X)} | mapped to exactly one participant: {ok.sum()} ({ok.mean():.1%}) | "
          f"excluded: old {int((~ok & (S == 0)).sum())}, new {int((~ok & (S == 2)).sum())}")
    X, Y, S, P = X[ok], Y[ok], S[ok], P[ok]
    print(f"participants: old {len({p for p in P if p.startswith('old')})}, new {len({p for p in P if p.startswith('new')})}")
    print(f"ungated weak emotion windows (fail the gate, expressive AU > 0.10): {len(d['Yw'])} from "
          f"{len(set(d['Pw']) & set(P))} of these participants; per class:",
          pd.Series(d['Yw']).map(dict(enumerate(E))).value_counts().to_dict())

    preds = {fs: np.zeros(len(Y), int) for fs in "ABC"}
    pw = {fs: np.full(len(d["Yw"]), -1) for fs in "ABC"}
    weak_flag, tert, chosen, floors = np.zeros(len(Y), bool), np.zeros(len(Y), int), [], []
    for k, (tr, te) in enumerate(GroupKFold(7).split(X, Y, P)):
        ws_tr = strength(X[tr], "L1")[1]
        emo_tr = ws_tr[Y[tr] != NEU]
        thr = np.quantile(emo_tr, 0.25)
        cuts = np.quantile(emo_tr, [1 / 3, 2 / 3])
        ws_te = strength(X[te], "L1")[1]
        weak_flag[te] = (Y[te] != NEU) & (ws_te < thr)
        tert[te] = np.digitize(ws_te, cuts)
        held = set(P[te])
        wmask = np.isin(d["Pw"], list(held))
        for fs in "ABC":
            cfg = choose(fs, X[tr], Y[tr], P[tr])
            pred, floor = fit(fs, cfg, X[tr], Y[tr])
            preds[fs][te] = pred(X[te])
            if wmask.any():
                pw[fs][wmask] = pred(d["Xw"][wmask])
            chosen.append((k, fs, cfg)); floors.append((k, fs, floor))
        print(f"fold {k + 1}/7 done: held-out {sorted(held)}", flush=True)

    print("\nchosen configurations:", [(k, fs, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in c.items()})
                                      for k, fs, c in chosen])
    print("floors (C):", [(k, round(f, 3)) for k, fs, f in floors if fs == "C"])

    src = np.where(S == 0, "old", "new")
    print("\n== gated windows of held-out participants")
    rows = []
    for name, m in (("old", src == "old"), ("new", src == "new"), ("pooled", np.ones(len(Y), bool))):
        for fs in "ABC":
            rows.append({"data": name, "features": fs, **errs(Y[m], preds[fs][m])})
    print(pd.DataFrame(rows).set_index(["data", "features"]).round(3).to_string())
    print("\nper-class F1, pooled")
    print(pd.DataFrame({fs: f1_score(Y, preds[fs], labels=range(7), average=None, zero_division=0) for fs in "ABC"},
                       index=E).T.round(3).to_string())

    print("\n== per participant macro-F1 (held out)")
    pp = pd.DataFrame({fs: pd.Series({p: f1_score(Y[P == p], preds[fs][P == p], labels=range(7), average="macro",
                                                   zero_division=0) for p in sorted(set(P))}) for fs in "ABC"})
    pp["weak windows"] = pd.Series({p: int(weak_flag[P == p].sum()) for p in sorted(set(P))})
    for a, b in (("B", "A"), ("C", "A"), ("C", "B")):
        dd = pp[a] - pp[b]
        print(f"  {a} - {b}: mean {dd.mean():+.3f}, participants better / worse {int((dd > 0.005).sum())} / {int((dd < -0.005).sum())}"
              f" | old {dd[[p.startswith('old') for p in dd.index]].mean():+.3f}, new {dd[[p.startswith('new') for p in dd.index]].mean():+.3f}")
    print(pp.round(3).to_string())

    print("\n== weak windows (L1 strength < training 25th percentile of emotion windows)")
    print(f"weak windows: {weak_flag.sum()} from {len(set(P[weak_flag]))} participants; per class:",
          pd.Series(Y[weak_flag]).map(dict(enumerate(E))).value_counts().to_dict())
    wa = weak_flag & (Y == ANG)
    print(f"weak Anger: {wa.sum()} windows from {len(set(P[wa]))} participants")
    t = pd.DataFrame({fs: pd.Series(preds[fs][wa]).map(dict(enumerate(E))).value_counts() for fs in "ABC"}).fillna(0).astype(int)
    print(t.to_string())
    rows = []
    for fs in "ABC":
        p = preds[fs]
        rows.append({"features": fs, "weak emotion acc": np.mean(p[weak_flag] == Y[weak_flag]),
                     "weak emotion->Neutral": int((p[weak_flag] == NEU).sum()),
                     "weak Anger acc": np.mean(p[wa] == ANG) if wa.any() else np.nan,
                     "Neutral->Anger (all)": int(((Y == NEU) & (p == ANG)).sum()),
                     "Happiness->Anger": int(((Y == HAP) & (p == ANG)).sum()),
                     "Disgust->Anger": int(((Y == DIS) & (p == ANG)).sum())})
    print(pd.DataFrame(rows).set_index("features").round(3).to_string())

    print("\n== matched strength: accuracy inside strength tertiles (training-fold cut points), emotion windows")
    emo = Y != NEU
    print(pd.DataFrame({fs: [np.mean(preds[fs][emo & (tert == t)] == Y[emo & (tert == t)]) for t in range(3)]
                        for fs in "ABC"}, index=["weakest third", "middle", "strongest third"]).assign(
        n=[int((emo & (tert == t)).sum()) for t in range(3)]).round(3).to_string())

    print("\n== ungated weak acting (fail the gate, expressive AU > 0.10), held-out participants")
    yw = d["Yw"]
    have = pw["A"] >= 0
    rows = []
    for fs in "ABC":
        p = pw[fs][have]
        rows.append({"features": fs, "n": int(have.sum()), "accuracy": np.mean(p == yw[have]),
                     "->Neutral": int((p == NEU).sum()), "Anger acc": np.mean(p[yw[have] == ANG] == ANG),
                     "other->Anger": int(((yw[have] != ANG) & (p == ANG)).sum())})
    print(pd.DataFrame(rows).set_index("features").round(3).to_string())
    pr = {fs: pd.Series({q: np.mean(pw[fs][have & (d['Pw'] == q)] == yw[have & (d['Pw'] == q)])
                         for q in sorted(set(d["Pw"][have]))}) for fs in "ABC"}
    dd = pr["C"] - pr["A"]
    print(f"per participant, C - A accuracy: better / worse {int((dd > 0.005).sum())} / {int((dd < -0.005).sum())} of {len(dd)}")

    # noise amplification check, on all mapped training windows
    print("\n== noise check: peak relative value (C features) by window type, L1 strength, q = 0.9 floor")
    floor = float(np.quantile(strength(X[Y == NEU], "L1")[1], 0.9))
    ws = strength(X, "L1")[1]
    rel = (X[..., EXPR] / np.maximum(ws, floor)[:, None, None]).max((1, 2))
    for name, m in (("Neutral", Y == NEU), ("weak emotion", weak_flag), ("other emotion", (Y != NEU) & ~weak_flag)):
        print(f"  {name:<14} n={m.sum():>5}  strength median {np.median(ws[m]):.3f}  share below floor "
              f"{np.mean(ws[m] < floor):.2f}  peak relative q50/q90 {np.quantile(rel[m], .5):.3f}/{np.quantile(rel[m], .9):.3f}")
    print(f"  floor = {floor:.3f}")

    np.savez_compressed(REPO / "Analysis/results/strength_pattern_cv_predictions.npz",
                        Y=Y, P=P, S=S, weak=weak_flag, **{f"pred_{fs}": preds[fs] for fs in "ABC"})


if __name__ == "__main__":
    main()
