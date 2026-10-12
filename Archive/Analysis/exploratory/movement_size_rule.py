"""Can a face-size term or a Neutral gate fix Neutral in movement calibration, without losing Fear?

Offline, on every session-local-v1 recording with a valid calibration block, model D. Fitted on the
calibration block, scored on the evaluation block. All choices are fixed before looking at results:

  current    movement calibration as deployed: score cosine + movement cosine
  A (size)   + size match per emotion: exp(-(log s - m)^2 / (2 sd^2)), where s is the tick's face size
             (sum of the 63 face AUs, frame-averaged), m the median log size of the emotion's 9
             calibration ticks, and sd their log standard deviation, at least 0.1. Equal weight.
  C (gate)   current, but a tick whose raw Neutral probability is >= 0.9 is Neutral
  A + C      both

Success, set in advance: Neutral tick accuracy clearly up, Fear not down, overall tick macro-F1 not
down, and better than current in most sessions.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
sys.path.insert(0, str(REPO / "Analysis/exploratory"))
import numpy as np
import pandas as pd

import calibration_study as cs
import session_analysis as sa
import jitter_headset_replay as jr

E = jr.E
EVERY = list(range(7))
NEUTRAL = E.index("Neutral")
SD_FLOOR = 0.1
GATE = 0.9
OTHER_PERSON = "20260930_144728_26908f"


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def methods(data):
    enrol, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
    y = data.prompted
    L = cs.logscore(data.scores)
    A = cs.movement_feature(data.windows)
    logsize = np.log(np.maximum(A.sum(1), 1e-6))

    score_M = cs.centroids(L[enrol], y[enrol], EVERY)
    move_M = cs.centroids(A[enrol], y[enrol], EVERY)
    m = np.array([np.median(logsize[enrol & (y == c)]) for c in EVERY])
    sd = np.array([max(np.std(logsize[enrol & (y == c)]), SD_FLOOR) for c in EVERY])

    base = unit(L[test]) @ unit(score_M).T + unit(A[test]) @ unit(move_M).T
    size = np.exp(-((logsize[test][:, None] - m[None, :]) ** 2) / (2 * sd[None, :] ** 2))
    gate = data.scores[test][:, NEUTRAL] >= GATE

    def gated(pred):
        out = pred.copy()
        out[gate] = NEUTRAL
        return out

    preds = {
        "raw": data.scores[test].argmax(1),
        "current": base.argmax(1),
        "A (size)": (base + size).argmax(1),
        "C (gate)": gated(base.argmax(1)),
        "A + C": gated((base + size).argmax(1)),
    }
    # sanity: "current" must equal the production function
    check = cs.classify_movement(score_M, L[test], move_M, A[test], EVERY)[0]
    assert np.array_equal(check, preds["current"])
    return preds, y[test], data.trial[test], data.tick[test], int(gate.sum())


rows = []
for d in jr.sessions():
    data = cs.replay(sa.load_session(d / "predictions.csv"), jr.ONNX_MODELS["D (none, EH)"])
    preds, yt, trials, ticks, n_gate = methods(data)
    yn = np.array([E[i] for i in yt])
    for name, p in preds.items():
        pn = np.array([E[i] for i in p])
        _, ty, tp = cs.trial_vote(yn, pn, trials, ticks, E)
        row = {"session": d.name[8:], "method": name, "gated ticks": n_gate,
               "tick F1": sa.macro_f1(yn, pn, E)["macro_f1"], "trial F1": sa.macro_f1(ty, tp, E)["macro_f1"],
               "called Neutral (expressive)": int(((p == NEUTRAL) & (yt != NEUTRAL)).sum())}
        row.update({e: float(np.mean(p[yt == i] == i)) for i, e in enumerate(E)})
        rows.append(row)

df = pd.DataFrame(rows)
order = ["raw", "current", "A (size)", "C (gate)", "A + C"]
pd.set_option("display.width", 250)

print(f"MEAN over {df.session.nunique()} sessions (model D, evaluation block)")
print(df.groupby("method")[["tick F1", "trial F1", "called Neutral (expressive)"] + E].mean().loc[order].round(3).to_string())

piv = df.pivot(index="session", columns="method", values="tick F1")
print("\nsessions better / worse / tied than current, tick macro-F1")
for mth in order[2:]:
    dl = piv[mth] - piv["current"]
    print(f"  {mth:<9} {int((dl > 1e-9).sum())} / {int((dl < -1e-9).sum())} / {int((dl.abs() <= 1e-9).sum())}"
          f"   mean change {dl.mean():+.3f}")

print("\nper session, tick macro-F1")
print(piv[order].round(3).to_string())
print(f"\nticks gated to Neutral per session (raw Neutral >= {GATE}):",
      df[df.method == 'current'].set_index('session')['gated ticks'].to_dict())
print(f"\nother person ({OTHER_PERSON}):")
print(df[df.session == OTHER_PERSON].set_index("method").loc[order, ["tick F1", "Neutral", "Fear"]].round(3).to_string())

df.to_csv(REPO / "Analysis/results/movement_size_rule.csv", index=False)
