"""Every session-local-v1 recording replayed through several teachers.

D = deployed acted-only teacher (old + new acted + EmojiHero, no augmentation)
B = volume augmentation k=1, EmojiHero excluded
C = volume augmentation k=1, EmojiHero kept (D + augmentation)

Participants see no prediction, so a replay through a model a session was not recorded with is a
fair counterfactual. Evaluation block, trial-majority macro-F1, raw argmax and personal calibration
(same functions as calibration_study.py). Also: tick accuracy, accuracy on weak ticks (no AU > 0.25
anywhere in the window), share of expressive ticks called Neutral, per-emotion tick accuracy.
"""
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import pandas as pd
import calibration_study as cs
import session_analysis as sa

MODELS = REPO.parent / "VREmotionDetection/Runtime/Models"
RUNS = {
    "D": (MODELS / "sparseVR_teacher_model_acted_only.onnx",
          "sparseVR_teacher_model_acted_only"),
    "B": (MODELS / "sparseVR_teacher_model_acted_only_vol-aug-k1_excluding-EmojiHero.onnx",
          "sparseVR_teacher_model_acted_only_vol-aug-k1_excluding-EmojiHero"),
    "C": (MODELS / "sparseVR_teacher_model_acted_only_vol-aug-k1_with-EmojiHero.onnx",
          "sparseVR_teacher_model_acted_only_vol-aug-k1_with-EmojiHero"),
}
E = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
NEUTRAL = E.index("Neutral")

sessions = []
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if (meta.get("protocol_version") == "session-local-v1"
            and meta.get("calibration", {}).get("calibration_valid") in ("true", None)):
        sessions.append((d, meta["models"]["face_model_asset"]))

cond = cs.Condition()
rows = []
for d, recorded_with in sessions:
    log = sa.load_session(d / "predictions.csv")
    row = {"session": d.name[8:], "recorded with": recorded_with}
    for name, (path, asset) in RUNS.items():
        data = cs.replay(log, path)
        if recorded_with == asset and not data.faithful:
            raise SystemExit(f"{d.name}: own-model replay does not match the log")
        enrol, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
        r = cs.evaluate(data, cond, enrol, test, required_ticks=cs.TICKS_REQUIRED_PER_CENTROID,
                        test_role=sa.ROLE_EVALUATION)
        tm = r["trial_majority"]
        row[f"{name} raw"] = tm["argmax_full"]["macro_f1"]
        row[f"{name} pers"] = tm["adapter"]["macro_f1"] if r["calibration_valid"] else np.nan
        k = np.isin(data.tick, list(cs.SCORED_TICKS)) & test
        pred = data.scores.argmax(1)
        expressive = k & (data.prompted != NEUTRAL)
        weak = expressive & ~(data.windows[..., :63] > 0.25).any(axis=(1, 2))
        row[f"{name} tick acc"] = float(np.mean(pred[k] == data.prompted[k]))
        row[f"{name} weak correct"] = int(np.sum(pred[weak] == data.prompted[weak]))
        row[f"{name} ->Neu"] = float(np.mean(pred[expressive] == NEUTRAL))
        row["n weak"] = int(weak.sum())
        row["weak share"] = float(weak.sum() / expressive.sum())
        row["face size"] = float(np.median(data.windows[expressive][..., :63].sum(-1).mean(-1)))
        for e in range(7):
            m = k & (data.prompted == e)
            row[f"{name} {E[e]}"] = float(np.mean(pred[m] == e))
    rows.append(row)

df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
names = list(RUNS)
print(df[["session", "face size", "weak share"]
         + [f"{n} raw" for n in names] + [f"{n} pers" for n in names]].round(2).to_string(index=False))

print("\nmeans over", len(df), "sessions:")
summary = pd.DataFrame({
    n: {
        "trial macro-F1 raw": df[f"{n} raw"].mean(),
        "trial macro-F1 personal": df[f"{n} pers"].mean(),
        "tick accuracy": df[f"{n} tick acc"].mean(),
        "weak-tick accuracy (pooled)": df[f"{n} weak correct"].sum() / df["n weak"].sum(),
        "expressive ticks called Neutral": df[f"{n} ->Neu"].mean(),
    } for n in names
})
print(summary.round(3).to_string())
print("weak ticks pooled:", int(df["n weak"].sum()))

for other in names[1:]:
    for c in ("raw", "pers"):
        dl = df[f"{other} {c}"] - df[f"D {c}"]
        print(f"{other} - D trial macro-F1 {c:<4}: mean {dl.mean():+.3f}  "
              f"better {(dl > 0).sum()}  worse {(dl < 0).sum()}  tied {(dl == 0).sum()}")

print("\nper-emotion evaluation tick accuracy, mean over sessions:")
print(pd.DataFrame({n: [df[f"{n} {e}"].mean() for e in E] for n in names}, index=E).round(2).to_string())
