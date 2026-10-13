"""Two signal-processing ideas on model D, offline, no retraining.

A. Output smoothing: average the per-tick decision scores over consecutive ticks before argmax.
   Causal (only past ticks), exponential with alpha 0.5 / 0.3, or the mean of the last 3 ticks.
   "trial" resets at each trial start (the game knows when a prompt starts); "stream" never
   resets (the realistic case when the game does not know). Applied to the raw model scores and
   to movement calibration's similarity sums.

B. Peak selection: decide only from the ticks where the face moves most.
   B1 per trial: vote over the k most active ticks instead of all ticks (every trial answered).
   B2 per tick: keep the most active half and score only those, compared with keeping the most
      confident half (model probability) and with keeping all. Activity = mean of the 63 face
      AUs over the window (the movement feature, summed). Headset: per-session median. Test set:
      overall median (no person IDs).

Headset: every session-local-v1 session, evaluation block. Test set: old/new VR + EmojiHero.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
sys.path.insert(0, str(REPO / "Analysis/exploratory"))
import numpy as np
import onnxruntime as ort
import pandas as pd

import calibration_study as cs
import session_analysis as sa
from jitter_headset_replay import E, EVERY, ONNX_MODELS, sessions

MODEL = ONNX_MODELS["D (none, EH)"]
session = ort.InferenceSession(str(MODEL))


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def smooth(S, trial, how, reset):
    """Causal smoothing of the rows of S, in time order."""
    out = np.empty_like(S)
    for i in range(len(S)):
        new = reset and (i == 0 or trial[i] != trial[i - 1])
        if how == "none" or i == 0 or new:
            out[i] = S[i]
        elif how.startswith("ema"):
            a = float(how[3:])
            out[i] = a * S[i] + (1 - a) * out[i - 1]
        elif how == "last3":
            lo = i
            while lo > max(0, i - 2) and not (reset and trial[lo - 1] != trial[i]):
                lo -= 1
            out[i] = S[lo:i + 1].mean(0)
    return out


def f1(y, p):
    return sa.macro_f1([E[i] for i in y], [E[i] for i in p], E)["macro_f1"]


def trial_f1(y, p, trial, tick):
    _, ty, tp = cs.trial_vote(np.array([E[i] for i in y]), np.array([E[i] for i in p]), trial, tick, E)
    return sa.macro_f1(ty, tp, E)["macro_f1"]


def headset():
    rowsA, rowsB = [], []
    for d in sessions():
        data = cs.replay(sa.load_session(d / "predictions.csv"), MODEL)
        enrol, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
        y = data.prompted
        L = cs.logscore(data.scores)
        A = cs.movement_feature(data.windows)
        score_M = cs.centroids(L[enrol], y[enrol], EVERY)
        move_M = cs.centroids(A[enrol], y[enrol], EVERY)
        sims = unit(L) @ unit(score_M).T + unit(A) @ unit(move_M).T
        decision = {"raw": data.scores, "movement": sims}
        trial, tick, yt = data.trial, data.tick, y[test]

        # A. output smoothing (smoothing runs over the whole session in time order, scored on test)
        for method, S in decision.items():
            for how in ("none", "ema0.5", "ema0.3", "last3"):
                for reset in ((True,) if how == "none" else (True, False)):
                    p = smooth(S, trial, how, reset).argmax(1)[test]
                    rowsA.append({"session": d.name[8:], "method": method,
                                  "smoothing": how if how == "none" else f"{how} ({'trial' if reset else 'stream'})",
                                  "tick": f1(yt, p), "trial": trial_f1(yt, p, trial[test], tick[test])})

        # B. peak selection
        act = A.sum(1)[test]
        for method, S in decision.items():
            p = S.argmax(1)[test]
            row = {"session": d.name[8:], "method": method, "all ticks, trial": trial_f1(yt, p, trial[test], tick[test])}
            for k in (3, 5):
                keep = np.zeros(len(p), bool)
                for t in np.unique(trial[test]):
                    idx = np.where(trial[test] == t)[0]
                    keep[idx[np.argsort(-act[idx])[:k]]] = True
                row[f"top{k} active, trial"] = trial_f1(yt[keep], p[keep], trial[test][keep], tick[test][keep])
            conf = S[test].max(1) if method == "raw" else np.sort(S[test], 1)[:, -1] - np.sort(S[test], 1)[:, -2]
            row["all ticks, tick"] = f1(yt, p)
            row["active half, tick"] = f1(yt[act >= np.median(act)], p[act >= np.median(act)])
            row["confident half, tick"] = f1(yt[conf >= np.median(conf)], p[conf >= np.median(conf)])
            row["quiet half, tick"] = f1(yt[act < np.median(act)], p[act < np.median(act)])
            rowsB.append(row)
    return pd.DataFrame(rowsA), pd.DataFrame(rowsB)


def test_set():
    z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
    rows = []
    for split in ("test_old", "test_new", "test_eh"):
        X, Y = z[f"X_{split}"], z[f"Y_{split}"]
        S = session.run(None, {session.get_inputs()[0].name: X.astype(np.float32)})[0]
        p = S.argmax(1)
        act = cs.movement_feature(X).sum(1)
        conf = S.max(1)
        a, c = act >= np.median(act), conf >= np.median(conf)
        rows.append({"data": split, "all": f1(Y, p), "active half": f1(Y[a], p[a]),
                     "confident half": f1(Y[c], p[c]), "quiet half": f1(Y[~a], p[~a]),
                     "Neutral share, active half": float(np.mean(Y[a] == E.index("Neutral"))),
                     "Neutral share, all": float(np.mean(Y == E.index("Neutral")))})
    return pd.DataFrame(rows).set_index("data")


def main():
    A, B = headset()
    T = test_set()
    A.to_csv(REPO / "Analysis/results/output_smoothing_headset.csv", index=False)
    B.to_csv(REPO / "Analysis/results/peak_selection_headset.csv", index=False)
    T.to_csv(REPO / "Analysis/results/peak_selection_test.csv")
    pd.set_option("display.width", 220)

    n = A.session.nunique()
    print(f"A. OUTPUT SMOOTHING, headset, {n} sessions, mean macro-F1 (better/worse sessions vs none, tick)")
    for method in ("raw", "movement"):
        a = A[A.method == method]
        base = a[a.smoothing == "none"].set_index("session").tick
        print(f"\n  {method}")
        for how, g in a.groupby("smoothing", sort=False):
            g = g.set_index("session")
            print(f"    {how:<20} tick {g.tick.mean():.3f}  trial {g.trial.mean():.3f}   "
                  f"{int((g.tick > base).sum())}/{int((g.tick < base).sum())}")

    print(f"\nB. PEAK SELECTION, headset, {n} sessions, mean macro-F1")
    cols = ["all ticks, trial", "top3 active, trial", "top5 active, trial",
            "all ticks, tick", "active half, tick", "confident half, tick", "quiet half, tick"]
    print(B.groupby("method")[cols].mean().round(3).T.to_string())
    for method in ("raw", "movement"):
        b = B[B.method == method]
        print(f"  {method}: top3 vs all trial {int((b['top3 active, trial'] > b['all ticks, trial']).sum())}/"
              f"{int((b['top3 active, trial'] < b['all ticks, trial']).sum())}, active-half vs all tick "
              f"{int((b['active half, tick'] > b['all ticks, tick']).sum())}/{int((b['active half, tick'] < b['all ticks, tick']).sum())}")

    print("\nB. PEAK SELECTION, test set (raw D), macro-F1")
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
