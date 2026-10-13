"""Do our synthetic quiet copies look like REAL quiet faces?

Real quiet faces  = headset expressive ticks (scored ticks, all 9 session-local-v1 sessions) whose
                    face size is in the quiet band.
Candidates, all restricted to the same quiet band and the same emotions:
  scaled      the current augmentation: copy = window * g (all 70 signals), clipped to [0, 1]
  blend       copy = neutral + g * (window - neutral), gaze kept from the window
              (neutral = a real gated Neutral window of the same person)
  real weak   real acted training windows that FAIL the notebook's 0.25 rule (ungated windows)
  real loud training vs real loud headset (face size 2-4) is the reference domain gap.

Face size = mean over frames of the sum of the 63 non-tongue AUs. Quiet band = [0.5, 1.5].

Tests:
  1. simple statistics: gaze share of the face size, number of active channels, resting floor
  2. a two-sample classifier: can logistic regression tell the candidate from the real quiet
     headset ticks? AUC 0.5 = indistinguishable, 1.0 = trivially different
  3. shape: cosine of the per-emotion mean profile with the real quiet headset profile
  4. model C's accuracy on each candidate vs on real quiet headset ticks
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import acting_dataset as ad
import calibration_study as cs
import emojihero_calibration_study as eh
import session_analysis as sa
import teacher_preprocessing as tp

rng = np.random.default_rng(0)
E = list(eh.EMOTIONS)
NEUTRAL = E.index("Neutral")
EXPR = [e for e in range(7) if e != NEUTRAL]
M63 = slice(0, 63)
GAZE = slice(14, 22)
QUIET = (0.5, 1.5)
MODEL_C = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only_vol-aug-k1_with-EmojiHero.onnx"

size = lambda w: w[..., M63].sum(-1).mean(-1)
feat = lambda w: w.mean(axis=1)                         # (N, 70): mean over the 5 frames

# ---------------------------------------------------------------- training windows per person
gated, ungated = {}, {}                                  # (person, emotion) -> windows
old_root = REPO / "Analysis/data/filtered-dataset-20260924T162300Z-1-001/filtered-dataset"
for folder in sorted(old_root.iterdir()):
    for f in sorted(folder.glob("ACT_*.csv")):
        rows = list(csv.reader(open(f, encoding="utf-8-sig")))
        fr = np.array([[float(v) for v in r[1:]] for r in rows[1:] if len(r) == len(rows[0])])
        emo = f.stem[4:]
        gated[("old:" + folder.name, E.index(emo))] = tp.make_windows(fr, emo)
        ungated[("old:" + folder.name, E.index(emo))] = tp.make_windows(fr, emo, gate=False)
for r in ad.load_corpus(ad.find_corpus_root()).recordings:
    fr = tp.downsample(r.au.astype(np.float64), tp.DOWNSAMPLE_FACTOR)
    gated[("new:" + r.participant, E.index(r.emotion))] = tp.make_windows(fr, r.emotion)
    ungated[("new:" + r.participant, E.index(r.emotion))] = tp.make_windows(fr, r.emotion, gate=False)
for s in eh.load_split(REPO / "Analysis/data", "training")[0]:
    k = ("eh:" + str(s.participant), s.emotion)
    gated[k] = np.concatenate([gated[k], eh.to_model_windows(s.au)]) if k in gated else eh.to_model_windows(s.au)

people = sorted({p for p, _ in gated})
neutral_of = {p: gated.get((p, NEUTRAL)) for p in people}
all_neutral = np.concatenate([w for w in neutral_of.values() if w is not None and len(w)])

X_tr, y_tr, p_tr = [], [], []
for (p, e), w in gated.items():
    if e in EXPR and len(w):
        X_tr.append(w); y_tr += [e] * len(w); p_tr += [p] * len(w)
X_tr = np.concatenate(X_tr).astype(np.float32); y_tr = np.array(y_tr); p_tr = np.array(p_tr)
print(f"training expressive windows (gated): {len(X_tr)} from {len(set(p_tr))} people")


# ---------------------------------------------------------------- candidates
def scaled_copies(X, draws=6):
    s = size(X)
    lo = np.clip(0.8 / s, 1 / 3, 3); hi = np.clip(5.5 / s, 1 / 3, 3)
    out = []
    for _ in range(draws):
        g = np.exp(rng.uniform(np.log(lo), np.log(hi)))
        out.append(np.clip(X * g[:, None, None], 0, 1))
    return np.concatenate(out), np.tile(np.arange(len(X)), draws)


def blend_copies(X, persons, draws=6):
    out, idx = [], []
    for _ in range(draws):
        n = np.stack([
            (neutral_of[p] if neutral_of.get(p) is not None and len(neutral_of[p]) else all_neutral)[
                rng.integers(len(neutral_of[p]) if neutral_of.get(p) is not None and len(neutral_of[p])
                             else len(all_neutral))]
            for p in persons])
        g = np.exp(rng.uniform(np.log(1 / 3), np.log(1.0), size=len(X)))
        c = n + g[:, None, None] * (X - n)
        c[..., GAZE] = X[..., GAZE]                      # gaze is where you look, not intensity
        out.append(np.clip(c, 0, 1)); idx.append(np.arange(len(X)))
    return np.concatenate(out), np.concatenate(idx)


Xs, is_ = scaled_copies(X_tr); ys = y_tr[is_]
Xb, ib = blend_copies(X_tr, p_tr); yb = y_tr[ib]

weak_real, weak_y = [], []
for (p, e), w in ungated.items():
    if e in EXPR and len(w):
        m = ~(w[..., M63] > 0.25).any(axis=(1, 2))
        weak_real.append(w[m]); weak_y += [e] * int(m.sum())
Xw = np.concatenate(weak_real).astype(np.float32); yw = np.array(weak_y)

# ---------------------------------------------------------------- headset
H, Hy = [], []
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if (meta.get("protocol_version") != "session-local-v1"
            or meta.get("calibration", {}).get("calibration_valid") not in ("true", None)):
        continue
    data = cs.replay(sa.load_session(d / "predictions.csv"), MODEL_C)
    k = np.isin(data.tick, list(cs.SCORED_TICKS)) & (data.prompted != NEUTRAL)
    H.append(data.windows[k]); Hy.append(data.prompted[k])
H = np.concatenate(H).astype(np.float32); Hy = np.concatenate(Hy)


def band(X, y, lo, hi):
    s = size(X)
    m = (s >= lo) & (s < hi)
    return X[m], y[m]


Hq, Hqy = band(H, Hy, *QUIET)
cands = {
    "scaled (current)": band(Xs, ys, *QUIET),
    "blend with Neutral": band(Xb, yb, *QUIET),
    "real weak training": band(Xw, yw, *QUIET),
}
print(f"real quiet headset ticks (size {QUIET[0]}-{QUIET[1]}): {len(Hq)}")
for n, (X, y) in cands.items():
    print(f"  {n:<20} {len(X)} windows")

# ---------------------------------------------------------------- 1. simple statistics
print("\n1. SIMPLE STATISTICS (quiet band; median)")
def stats(X):
    s = size(X)
    f = feat(X)[:, M63]
    return {
        "face size": np.median(s),
        "gaze share": np.median(X[..., GAZE].sum(-1).mean(-1) / s),
        "active channels (>0.02)": np.median((f > 0.02).sum(1)),
        "resting floor (mean of 40 smallest)": np.median(np.sort(f, axis=1)[:, :40].mean(1)),
        "largest signal": np.median(X[..., M63].max(axis=(1, 2))),
    }
rows = [{"source": "REAL quiet headset", **stats(Hq)}]
rows += [{"source": n, **stats(X)} for n, (X, _) in cands.items()]
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------------------------------------------------------- 2. two-sample classifier
print("\n2. CAN A CLASSIFIER TELL IT FROM REAL QUIET HEADSET TICKS?  (AUC; 0.5 = looks real)")
def auc(A, B, n=1500):
    a = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    b = B[rng.choice(len(B), min(n, len(B), len(a) * 3), replace=False)]
    X = np.concatenate([feat(a), feat(b)]); y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    return cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           scoring="roc_auc").mean()
for n, (X, _) in cands.items():
    print(f"  {n:<20} AUC {auc(Hq, X):.3f}")
Hl, _ = band(H, Hy, 2.0, 4.0)
Tl, Tli = band(X_tr, np.arange(len(X_tr)), 2.0, 4.0)
print(f"  reference: real LOUD headset vs real LOUD training (size 2-4): AUC {auc(Hl, Tl):.3f}")
# Control: is a high AUC just "different people"? Split training people of one collection.
pl = p_tr[Tli]
for coll in ("old:", "new:"):
    ppl = sorted({p for p in pl if p.startswith(coll)})
    res = []
    for _ in range(5):
        grp = set(rng.choice(ppl, size=min(5, len(ppl) // 2), replace=False))
        res.append(auc(Tl[np.array([p in grp for p in pl])],
                       Tl[np.array([p.startswith(coll) and p not in grp for p in pl])]))
    print(f"  control: 5 {coll[:-1]} acted people vs the other {coll[:-1]} acted people (loud): "
          f"AUC {np.mean(res):.3f}")

# ---------------------------------------------------------------- 3. shape per emotion
print("\n3. SHAPE: cosine of mean profile (63 signals) with REAL quiet headset, per emotion")
unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
rows = []
for e in EXPR:
    if (Hqy == e).sum() < 5:
        continue
    ref = unit(feat(Hq[Hqy == e])[:, M63].mean(0))
    row = {"emotion": E[e], "real quiet ticks": int((Hqy == e).sum())}
    for n, (X, y) in cands.items():
        row[n] = float(ref @ unit(feat(X[y == e])[:, M63].mean(0))) if (y == e).sum() else np.nan
    rows.append(row)
print(pd.DataFrame(rows).round(2).to_string(index=False))

# ---------------------------------------------------------------- 4. model C accuracy
print("\n4. MODEL C ACCURACY in the quiet band (if C is right on copies but wrong on real quiet")
print("   faces, the copies teach something the headset does not show)")
sess = ort.InferenceSession(str(MODEL_C))
predict = lambda X: np.concatenate([sess.run(None, {"seq": X[i:i + 512]})[0]
                                    for i in range(0, len(X), 512)]).argmax(1)
rows = [{"source": "REAL quiet headset", "accuracy": np.mean(predict(Hq) == Hqy),
         "called Neutral": np.mean(predict(Hq) == NEUTRAL)}]
for n, (X, y) in cands.items():
    p = predict(X.astype(np.float32))
    rows.append({"source": n, "accuracy": np.mean(p == y), "called Neutral": np.mean(p == NEUTRAL)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
