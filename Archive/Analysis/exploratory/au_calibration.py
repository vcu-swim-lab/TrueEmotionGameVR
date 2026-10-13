"""Movement (AU-space) calibration vs the current score-space calibration, offline.

Every session-local-v1 recording, replayed through the deployed teacher D. Each method is fitted
on the calibration block only (1 trial x 9 ticks per emotion) and scored on the evaluation block
(2 trials x 9 ticks per emotion), with the same trial-majority vote as calibration_study.py.

Methods
  raw            D argmax, no calibration
  score-cos      current runtime calibration: log-scores, nearest centroid, cosine
  au-cos         AU templates: mean over the 5 frames of the 63 non-tongue AUs, nearest centroid, cosine
  au-euc         same templates, Euclidean distance
  au-neu-euc     AUs minus the person's Neutral template, Euclidean
  au-z-euc       AUs standardised with the person's calibration mean/std, Euclidean
  au-lda         shrinkage LDA on the AU features
  au-knn5        5 nearest calibration ticks (standardised AUs)
  fuse-lda       shrinkage LDA on standardised AUs + standardised log-scores
  fuse-sim       mean of the score-cos and au-cos similarities

Nothing is tuned on the evaluation block. With 9 sessions (8 of one person) and 10 methods, small
differences between methods are not evidence.
"""
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

import calibration_study as cs
import session_analysis as sa

MODEL = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
E = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
NEUTRAL = E.index("Neutral")
N_FACE = 63  # non-tongue AUs


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cos_sim(M, F):
    return unit(F) @ unit(M).T


def nearest_cos(Ftr, ytr, Fte):
    M = np.stack([Ftr[ytr == c].mean(0) for c in range(7)])
    return cos_sim(M, Fte).argmax(1)


def nearest_euc(Ftr, ytr, Fte):
    M = np.stack([Ftr[ytr == c].mean(0) for c in range(7)])
    return np.linalg.norm(Fte[:, None] - M[None], axis=2).argmin(1)


def zfit(Ftr):
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd = np.where(sd < 1e-3, 1e-3, sd)  # AUs that never move in calibration stay ~0
    return lambda F: (F - mu) / sd


def lda(Ftr, ytr, Fte):
    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Ftr, ytr)
    return m.predict(Fte)


def methods(scores, au, ytr_mask, te_mask, y):
    """Predictions on the evaluation ticks, per method."""
    ytr = y[ytr_mask]
    ls = cs.logscore(scores)
    Atr, Ate = au[ytr_mask], au[te_mask]
    Str, Ste = ls[ytr_mask], ls[te_mask]
    z = zfit(Atr)
    zs = zfit(Str)
    neu = Atr[ytr == NEUTRAL].mean(0)

    out = {
        "raw": scores[te_mask].argmax(1),
        "score-cos": nearest_cos(Str, ytr, Ste),
        "au-cos": nearest_cos(Atr, ytr, Ate),
        "au-euc": nearest_euc(Atr, ytr, Ate),
        "au-neu-euc": nearest_euc(Atr - neu, ytr, Ate - neu),
        "au-z-euc": nearest_euc(z(Atr), ytr, z(Ate)),
        "au-lda": lda(Atr, ytr, Ate),
        "au-knn5": KNeighborsClassifier(5).fit(z(Atr), ytr).predict(z(Ate)),
    }
    fz_tr = np.hstack([z(Atr) / np.sqrt(N_FACE), zs(Str) / np.sqrt(7)])
    fz_te = np.hstack([z(Ate) / np.sqrt(N_FACE), zs(Ste) / np.sqrt(7)])
    out["fuse-lda"] = lda(fz_tr, ytr, fz_te)
    Ms = np.stack([Str[ytr == c].mean(0) for c in range(7)])
    Ma = np.stack([Atr[ytr == c].mean(0) for c in range(7)])
    out["fuse-sim"] = (cos_sim(Ms, Ste) + cos_sim(Ma, Ate)).argmax(1)
    return out


def trial_f1(y, pred, trials, ticks):
    y_n = np.array([E[i] for i in y])
    p_n = np.array([E[i] for i in pred])
    _, ty, tp = cs.trial_vote(y_n, p_n, trials, ticks, E)
    return sa.macro_f1(ty, tp, E)["macro_f1"]


def tick_f1(y, pred):
    return sa.macro_f1(np.array([E[i] for i in y]), np.array([E[i] for i in pred]), E)["macro_f1"]


sessions = []
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if (meta.get("protocol_version") == "session-local-v1"
            and meta.get("calibration", {}).get("calibration_valid") in ("true", None)):
        sessions.append(d)

rows, per_emotion = [], []
for d in sessions:
    data = cs.replay(sa.load_session(d / "predictions.csv"), MODEL)
    enrol, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
    au = data.windows[..., :N_FACE].mean(axis=1)
    y = data.prompted
    preds = methods(data.scores, au, enrol, test, y)
    yt = y[test]
    for name, p in preds.items():
        rows.append({
            "session": d.name[8:], "method": name,
            "trial macro-F1": trial_f1(yt, p, data.trial[test], data.tick[test]),
            "tick macro-F1": tick_f1(yt, p),
            "tick accuracy": float(np.mean(p == yt)),
        })
        per_emotion.append({"session": d.name[8:], "method": name,
                            **{E[e]: float(np.mean(p[yt == e] == e)) for e in range(7)}})

df = pd.DataFrame(rows)
order = list(preds)
pd.set_option("display.width", 250)

print("TRIAL macro-F1 per session (fit on calibration block, scored on evaluation block)")
print(df.pivot(index="session", columns="method", values="trial macro-F1")[order].round(2).to_string())

summary = df.groupby("method")[["trial macro-F1", "tick macro-F1", "tick accuracy"]].mean().loc[order]
for metric, ref in (("tick macro-F1", "score-cos"), ("tick macro-F1", "raw")):
    piv = df.pivot(index="session", columns="method", values=metric)
    col = f"{metric} vs {ref}: better/worse"
    summary[col] = [f"{int((piv[m] > piv[ref]).sum())}/{int((piv[m] < piv[ref]).sum())}"
                    if m != ref else "" for m in order]
print(f"\nMEANS over {len(sessions)} sessions")
print(summary.round(3).to_string())

piv = df.pivot(index="session", columns="method", values="tick macro-F1")
print("\nWilcoxon signed-rank, tick macro-F1, vs score-cos (current calibration):")
for m in order:
    if m in ("score-cos",):
        continue
    dlt = piv[m] - piv["score-cos"]
    p = wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
    print(f"  {m:<11} mean {dlt.mean():+.3f}  p {p:.3f}")

pe = pd.DataFrame(per_emotion).groupby("method")[E].mean().loc[order]
print("\nEVALUATION tick accuracy per emotion, mean over sessions")
print(pe.round(2).to_string())

out = REPO / "Analysis/results"
df.to_csv(out / "au_calibration_sessions.csv", index=False)
pe.to_csv(out / "au_calibration_per_emotion.csv")
