"""Acted-only teacher, 6 session-local-v1 sessions, offline.

Part 1  personal calibration: 7 centroids from the calibration block, scored on evaluation.
Part 2  one-emotion calibration: only emotion e's calibration trial is used. The person's
        offset for e, d = K_e - R_e, is added to every reference centroid (R + d), and the
        evaluation block is classified against that. R is either the other five headset
        sessions (leave-one-session-out) or the EmojiHero population.

Same arithmetic as the study: log(clip(score, 1e-12)), cosine nearest centroid, ticks 1-9,
trial majority with the frozen tie rule.
"""
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
SP = Path(__file__).parent
sys.path.insert(0, str(REPO / "Analysis"))

import numpy as np
import calibration_study as cs
import emojihero_calibration_study as eh
import session_analysis as sa

MODEL = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
E = list(eh.EMOTIONS)
G_eh, _ = cs.load_population_profile(str(SP / "pop_acted_only.json"), MODEL)

S = []
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if meta.get("protocol_version") != "session-local-v1":
        continue
    r = cs.replay(sa.load_session(d / "predictions.csv"), MODEL)
    assert r.classes == E
    keep = np.isin(r.tick, list(cs.SCORED_TICKS))
    S.append({
        "name": d.name[-6:],
        "recorded": meta["models"]["face_model_asset"] == "sparseVR_teacher_model_acted_only",
        "faithful": r.faithful,
        "lf": cs.logscore(r.scores[keep].astype(np.float64)), "sc": r.scores[keep],
        "y": r.prompted[keep], "role": r.role[keep], "trial": r.trial[keep], "tick": r.tick[keep],
    })
for s in S:
    if s["recorded"] and not s["faithful"]:
        raise SystemExit(f"{s['name']}: replay does not match the headset log")


def class_centroids(lf, y):
    return np.stack([lf[y == c].mean(0) for c in range(7)])


def score(s, pred):
    """Tick accuracy, trial-majority accuracy and macro-F1 on the evaluation block."""
    ev = s["role"] == sa.ROLE_EVALUATION
    names = np.array(E)
    _, ty, tp = cs.trial_vote(names[s["y"][ev]], names[pred], s["trial"][ev], s["tick"][ev], E)
    ty_i = np.array([E.index(v) for v in ty]); tp_i = np.array([E.index(v) for v in tp])
    return (float(np.mean(pred == s["y"][ev])), float(np.mean(ty_i == tp_i)),
            eh.fast_macro_f1(ty_i, tp_i), ty_i, tp_i)


def classify(M, s):
    ev = s["role"] == sa.ROLE_EVALUATION
    return cs.classify(M, list(range(7)), s["lf"][ev], "cosine")[0]


def reference_loso(i):
    """Participant-balanced centroids of the OTHER sessions (all ticks of both blocks)."""
    return np.stack([class_centroids(o["lf"], o["y"]) for j, o in enumerate(S) if j != i]).mean(0)


res = {}
for i, s in enumerate(S):
    ev = s["role"] == sa.ROLE_EVALUATION
    cal = s["role"] == sa.ROLE_CALIBRATION
    K = class_centroids(s["lf"][cal], s["y"][cal])
    res[(s["name"], "raw")] = score(s, s["sc"][ev].argmax(1))
    res[(s["name"], "personal")] = score(s, classify(K, s))
    for rname, R in (("loso", reference_loso(i)), ("eh", G_eh)):
        res[(s["name"], f"{rname}:none")] = score(s, classify(R, s))
        for e in range(7):
            res[(s["name"], f"{rname}:{E[e]}")] = score(s, classify(R + (K[e] - R[e]), s))

names = [s["name"] for s in S]


def table(title, methods):
    print("=" * 96)
    print(title)
    print("=" * 96)
    print(f"  {'method':<28}" + "".join(f"{n:>9}" for n in names)
          + f"{'mean':>9}{'vs raw':>12}")
    raw_f1 = np.array([res[(n, 'raw')][2] for n in names])
    for m, label in methods:
        f1 = np.array([res[(n, m)][2] for n in names])
        acc = np.array([res[(n, m)][1] for n in names])
        d = f1 - raw_f1
        wl = f"{int((d > 0).sum())}+/{int((d < 0).sum())}-"
        print(f"  {label:<28}" + "".join(f"{v:>9.3f}" for v in f1)
              + f"{f1.mean():>9.3f}{wl:>12}")
    print("  (trial-majority macro-F1 on the 14 evaluation trials; 'vs raw' = sessions better/worse)")
    print()


print("\nsessions:", ", ".join(f"{s['name']} ({'recorded' if s['recorded'] else 'replayed'})"
                               for s in S))
print()
print("=" * 96)
print("PART 1  personal calibration, acted-only model")
print("=" * 96)
print(f"  {'session':<10}{'raw tick':>10}{'raw trial':>11}{'raw F1':>9}"
      f"{'pers tick':>11}{'pers trial':>12}{'pers F1':>9}")
for n in names:
    r, p = res[(n, "raw")], res[(n, "personal")]
    print(f"  {n:<10}{r[0]:>10.3f}{r[1]:>11.3f}{r[2]:>9.3f}{p[0]:>11.3f}{p[1]:>12.3f}{p[2]:>9.3f}")
m = lambda k, j: np.mean([res[(n, k)][j] for n in names])
print(f"  {'mean':<10}{m('raw',0):>10.3f}{m('raw',1):>11.3f}{m('raw',2):>9.3f}"
      f"{m('personal',0):>11.3f}{m('personal',1):>12.3f}{m('personal',2):>9.3f}")

print("\n  per emotion, trial recall pooled over sessions (hits / trials):")
print(f"    {'emotion':<10}{'raw':>10}{'personal':>10}")
for c in range(7):
    row = f"    {E[c]:<10}"
    for k in ("raw", "personal"):
        hit = sum(int(((res[(n, k)][3] == c) & (res[(n, k)][4] == c)).sum()) for n in names)
        tot = sum(int((res[(n, k)][3] == c).sum()) for n in names)
        row += f"{f'{hit}/{tot}':>10}"
    print(row)
print()

base = [("raw", "raw (no calibration)"), ("personal", "personal (all 7 emotions)")]
for rname, rlabel in (("loso", "reference = other 5 headset sessions"),
                      ("eh", "reference = EmojiHero population")):
    table(f"PART 2  one-emotion calibration, {rlabel}",
          base + [(f"{rname}:none", "reference only, no shift")]
          + [(f"{rname}:{e}", f"shift from {e} only") for e in E])
