"""Would "person-style" copies cover new people better than volume copies?

Unit = one person's mean expression SHAPE for one emotion (mean AU profile over that person's
windows, 55 channels = 63 non-tongue minus the 8 gaze channels, scaled to unit length). Volume
copies (window * g) do not change the shape, so they add NO shape coverage: the pool of shapes
they show is exactly the pool of real training people.

Person style of person p = how p uses each muscle compared with the average person, averaged over
the six expressive emotions:  r_p = exp(mean_e log((P[p,e] + eps) / (M[e] + eps))), clipped to
[1/4, 4]. A person-style copy of person q's emotion e "made by" p is  P[q,e] * r_p.

Question: for a NEW person (query), is the nearest pool shape of the RIGHT emotion closer, and is
the 1-nearest-neighbour emotion more often right, when the pool also has person-style copies?

Pools (all built only from people other than the query):
  real          the real training people's shapes (= what volume augmentation shows)
  + style       real + person-style copies (every other person's style applied)
  + shuffled    real + copies whose style ratios are shuffled across muscles (same size of the
                changes, no muscle structure) -- control: is it the structure or just more points?
  + random      real + copies with random log-normal per-muscle gains of the same spread
  + same-emotion mix  real + copies moved part of the way (30-70%) toward ANOTHER person's shape
                for the SAME emotion: new people's version of that emotion, between real ones

Queries:
  A. leave-one-person-out over the VR training people (old + new acted): no headset data used
  B. the 9 headset sessions (one query per session and emotion)
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import pandas as pd
import acting_dataset as ad
import calibration_study as cs
import emojihero_calibration_study as eh
import session_analysis as sa
import teacher_preprocessing as tp

rng = np.random.default_rng(0)
E = list(eh.EMOTIONS)
NEUTRAL = E.index("Neutral")
EXPR = [e for e in range(7) if e != NEUTRAL]
CH = [i for i in range(63) if not 14 <= i < 22]        # 55 channels: no tongue, no gaze
EPS = 0.01
MODEL = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"


def prof(w):
    return w[..., CH].mean(axis=(0, 1)).astype(np.float64)


unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

# ---------------------------------------------------------------- training people
P = {}                                                   # person -> {emotion: profile}
old_root = REPO / "Analysis/data/filtered-dataset-20260924T162300Z-1-001/filtered-dataset"
for folder in sorted(old_root.iterdir()):
    for f in sorted(folder.glob("ACT_*.csv")):
        rows = list(csv.reader(open(f, encoding="utf-8-sig")))
        fr = np.array([[float(v) for v in r[1:]] for r in rows[1:] if len(r) == len(rows[0])])
        w = tp.make_windows(fr, f.stem[4:])
        if len(w):
            P.setdefault("old:" + folder.name, {})[E.index(f.stem[4:])] = prof(w)
for r in ad.load_corpus(ad.find_corpus_root()).recordings:
    w = tp.make_windows(tp.downsample(r.au.astype(np.float64), tp.DOWNSAMPLE_FACTOR), r.emotion)
    if len(w):
        P.setdefault("new:" + r.participant, {})[E.index(r.emotion)] = prof(w)
eh_w = {}
for s in eh.load_split(REPO / "Analysis/data", "training")[0]:
    eh_w.setdefault(("eh:" + str(s.participant), s.emotion), []).append(eh.to_model_windows(s.au))
for (p, e), ws in eh_w.items():
    P.setdefault(p, {})[e] = prof(np.concatenate(ws))
people = [p for p in P if all(e in P[p] for e in EXPR)]
vr_people = [p for p in people if not p.startswith("eh:")]
print(f"training people with all 6 expressive emotions: {len(people)} ({len(vr_people)} VR)")

# ---------------------------------------------------------------- headset sessions
Q_hs = {}
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
    Q_hs[d.name[-6:]] = {e: prof(data.windows[k & (data.prompted == e)]) for e in EXPR}


# ---------------------------------------------------------------- pools
def style_of(p, pool_people):
    M = {e: np.mean([P[q][e] for q in pool_people], axis=0) for e in EXPR}
    lr = np.mean([np.log((P[p][e] + EPS) / (M[e] + EPS)) for e in EXPR], axis=0)
    return np.exp(np.clip(lr, np.log(0.25), np.log(4)))


def build_pools(pool_people):
    real_s, real_y = [], []
    for q in pool_people:
        for e in EXPR:
            real_s.append(P[q][e]); real_y.append(e)
    real_s = np.array(real_s); real_y = np.array(real_y)
    styles = np.array([style_of(p, pool_people) for p in pool_people])
    spread = np.log(styles).std()
    out = {"real": (real_s, real_y)}
    for name in ("+ style", "+ shuffled", "+ random", "+ same-emotion mix"):
        syn_s, syn_y = [], []
        for i, (s, y) in enumerate(zip(real_s, real_y)):
            for j in rng.choice(len(styles), size=8, replace=False):     # 8 copies per shape
                if name == "+ same-emotion mix":
                    # move toward another person's shape for the SAME emotion (geometric mix)
                    other = P[pool_people[j]][y]
                    a = rng.uniform(0.3, 0.7)
                    r = ((other + EPS) / (s + EPS)) ** a
                elif name == "+ style":
                    r = styles[j]
                elif name == "+ shuffled":
                    r = rng.permutation(styles[j])
                else:
                    r = np.exp(rng.normal(0, spread, size=len(CH)))
                syn_s.append(s * r); syn_y.append(y)
        out[name] = (np.concatenate([real_s, syn_s]), np.concatenate([real_y, syn_y]))
    return out


def score(query, pools):
    """query: {emotion: profile}. Returns per pool: (1-NN correct, cosine to nearest same-emotion)."""
    res = {}
    for name, (S, Y) in pools.items():
        U = unit(S)
        correct, near = [], []
        for e, v in query.items():
            c = U @ unit(v)
            correct.append(Y[np.argmax(c)] == e)
            near.append(c[Y == e].max())
        res[name] = (np.mean(correct), np.mean(near))
    return res


# ---------------------------------------------------------------- A. leave one VR person out
rows = []
for p in vr_people:
    pools = build_pools([q for q in people if q != p])
    for name, (acc, near) in score({e: P[p][e] for e in EXPR}, pools).items():
        rows.append({"query": p, "pool": name, "1-NN emotion correct": acc, "nearest same-emotion cosine": near})
A = pd.DataFrame(rows)
print("\nA. LEAVE ONE VR TRAINING PERSON OUT (mean over", len(vr_people), "people; no headset data)")
print(A.groupby("pool", sort=False)[["1-NN emotion correct", "nearest same-emotion cosine"]].mean().round(3).to_string())

# ---------------------------------------------------------------- B. headset sessions
pools = build_pools(people)
rows = []
for s, q in Q_hs.items():
    for name, (acc, near) in score(q, pools).items():
        rows.append({"session": s, "pool": name, "1-NN emotion correct": acc, "nearest same-emotion cosine": near})
B = pd.DataFrame(rows)
print("\nB. HEADSET SESSIONS (mean over", len(Q_hs), "sessions)")
print(B.groupby("pool", sort=False)[["1-NN emotion correct", "nearest same-emotion cosine"]].mean().round(3).to_string())
print("\n   per session, 1-NN emotion correct (of 6 emotions):")
print(B.pivot(index="session", columns="pool", values="1-NN emotion correct")[
    ["real", "+ style", "+ shuffled", "+ random", "+ same-emotion mix"]].mul(6).round(0).astype(int).to_string())
