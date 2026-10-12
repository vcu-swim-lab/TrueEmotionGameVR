"""Day-to-day variation of ONE person vs differences BETWEEN people.

Headset: 9 session-local-v1 sessions. 8 are the same person (the developer) on different days;
session 20260930_144728_26908f is a different person.

Unit = one session's (or one training person's) mean expression shape per emotion
(55 channels: 63 non-tongue minus the 8 gaze channels, unit length).

  1. same-emotion shape similarity:
       same person, different day   (developer session vs developer session)
       different people, headset    (developer session vs the other person)
       different people, training   (VR training person vs VR training person)
  2. face size per session
  3. two-sample classifier AUC (window level): one developer session vs the other developer
     sessions, compared with 5 training people vs other training people
  4. 1-NN emotion from shapes: a developer session recognised from (a) the developer's OTHER days,
     (b) the training people
"""
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import acting_dataset as ad
import calibration_study as cs
import emojihero_calibration_study as eh
import session_analysis as sa
import teacher_preprocessing as tp

rng = np.random.default_rng(0)
E = list(eh.EMOTIONS)
NEUTRAL = E.index("Neutral")
EXPR = [e for e in range(7) if e != NEUTRAL]
CH = [i for i in range(63) if not 14 <= i < 22]
OTHER_PERSON = "26908f"
MODEL = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
prof = lambda w: unit(w[..., CH].mean(axis=(0, 1)).astype(np.float64))
size = lambda w: w[..., :63].sum(-1).mean(-1)

# ---------------------------------------------------------------- headset
S, SW = {}, {}                                           # session -> {emotion: shape}, windows
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if (meta.get("protocol_version") != "session-local-v1"
            or meta.get("calibration", {}).get("calibration_valid") not in ("true", None)):
        continue
    data = cs.replay(sa.load_session(d / "predictions.csv"), MODEL)
    k = np.isin(data.tick, list(cs.SCORED_TICKS))
    s = d.name[-6:]
    S[s] = {e: prof(data.windows[k & (data.prompted == e)]) for e in EXPR}
    SW[s] = (data.windows[k], data.prompted[k])
dev = [s for s in S if s != OTHER_PERSON]
print(f"developer sessions: {len(dev)}   other person: {OTHER_PERSON}")

# ---------------------------------------------------------------- VR training people
P, PW = {}, {}
old_root = REPO / "Analysis/data/filtered-dataset-20260924T162300Z-1-001/filtered-dataset"
for folder in sorted(old_root.iterdir()):
    for f in sorted(folder.glob("ACT_*.csv")):
        rows = list(csv.reader(open(f, encoding="utf-8-sig")))
        fr = np.array([[float(v) for v in r[1:]] for r in rows[1:] if len(r) == len(rows[0])])
        w = tp.make_windows(fr, f.stem[4:])
        if len(w):
            P.setdefault("old:" + folder.name, {})[E.index(f.stem[4:])] = prof(w)
            PW.setdefault("old:" + folder.name, []).append(w)
for r in ad.load_corpus(ad.find_corpus_root()).recordings:
    w = tp.make_windows(tp.downsample(r.au.astype(np.float64), tp.DOWNSAMPLE_FACTOR), r.emotion)
    if len(w):
        P.setdefault("new:" + r.participant, {})[E.index(r.emotion)] = prof(w)
        PW.setdefault("new:" + r.participant, []).append(w)
people = [p for p in P if all(e in P[p] for e in EXPR)]
PW = {p: np.concatenate(PW[p]) for p in people}

# ---------------------------------------------------------------- 1. shape similarity
def mean_cos(pairs, A, B):
    return np.mean([A[a][e] @ B[b][e] for a, b in pairs for e in EXPR])

rows = [
    {"comparison": "SAME person, different day (developer vs developer)",
     "pairs": len(list(combinations(dev, 2))),
     "same-emotion cosine": mean_cos(list(combinations(dev, 2)), S, S)},
    {"comparison": "DIFFERENT people, headset (developer vs other person)",
     "pairs": len(dev), "same-emotion cosine": mean_cos([(s, OTHER_PERSON) for s in dev], S, S)},
    {"comparison": "DIFFERENT people, VR training (person vs person)",
     "pairs": len(list(combinations(people, 2))),
     "same-emotion cosine": mean_cos(list(combinations(people, 2)), P, P)},
]
print("\n1. HOW ALIKE IS THE SAME EMOTION?  (cosine of shapes; 1 = identical shape)")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("   per emotion, same person different day vs different people (training):")
for e in EXPR:
    a = np.mean([S[x][e] @ S[y][e] for x, y in combinations(dev, 2)])
    b = np.mean([P[x][e] @ P[y][e] for x, y in combinations(people, 2)])
    print(f"   {E[e]:<10} same person {a:.2f}   different people {b:.2f}")

# ---------------------------------------------------------------- 2. face size
print("\n2. FACE SIZE PER SESSION (median, expressive ticks)")
for s in S:
    w, y = SW[s]
    print(f"   {s}{'  <- other person' if s == OTHER_PERSON else ''}  {np.median(size(w[y != NEUTRAL])):.2f}")
tr = [np.median(size(PW[p])) for p in people]
print(f"   VR training people: median {np.median(tr):.2f}, range {min(tr):.2f}-{max(tr):.2f}")

# ---------------------------------------------------------------- 3. classifier
def auc(a, b, n=900):
    a = a[rng.choice(len(a), min(n, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(n * 3, len(b)), replace=False)]
    X = np.concatenate([a.mean(1), b.mean(1)]); y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    return cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           scoring="roc_auc").mean()

day = [auc(SW[s][0], np.concatenate([SW[t][0] for t in dev if t != s])) for s in dev]
per = []
for _ in range(8):
    p1 = rng.choice(people)
    per.append(auc(PW[p1], np.concatenate([PW[q] for q in people if q != p1])))
print("\n3. CAN A CLASSIFIER TELL IT APART?  (AUC; 0.5 = looks the same)")
print(f"   one developer DAY vs the developer's other days:   AUC {np.mean(day):.3f}  "
      f"(range {min(day):.2f}-{max(day):.2f})")
print(f"   one training PERSON vs the other training people:  AUC {np.mean(per):.3f}")
print(f"   the other person vs the developer's days:          "
      f"AUC {auc(SW[OTHER_PERSON][0], np.concatenate([SW[t][0] for t in dev])):.3f}")

# ---------------------------------------------------------------- 4. 1-NN
def nn_acc(query, pool):
    shapes = np.array([v for d in pool for v in d.values()])
    labels = np.array([e for d in pool for e in d])
    return np.mean([labels[np.argmax(shapes @ v)] == e for e, v in query.items()])

own = [nn_acc(S[s], [S[t] for t in dev if t != s]) for s in dev]
oth = [nn_acc(S[s], [P[p] for p in people]) for s in dev]
print("\n4. RECOGNISE A DEVELOPER SESSION'S EMOTIONS FROM ... (1-NN on shapes, 6 emotions)")
print(f"   the developer's OTHER days:   {np.mean(own):.2f}")
print(f"   the VR training people:       {np.mean(oth):.2f}")
print(f"   other person ({OTHER_PERSON}) from developer days: "
      f"{nn_acc(S[OTHER_PERSON], [S[t] for t in dev]):.2f}   from training people: "
      f"{nn_acc(S[OTHER_PERSON], [P[p] for p in people]):.2f}")
