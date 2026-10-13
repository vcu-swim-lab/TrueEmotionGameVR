"""Every session-local-v1 recording replayed through D (deployed acted-only) and B (vol-aug k1, no
EmojiHero). Participants see no prediction, so a replay through a model a session was not recorded
with is a fair counterfactual. Evaluation block, trial-majority, raw argmax and personal calibration.
Also: face size of the session and accuracy on weak ticks (no AU > 0.25 in the window)."""
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import calibration_study as cs
import session_analysis as sa

MODELS = REPO.parent / "VREmotionDetection/Runtime/Models"
RUNS = {
    "D": (MODELS / "sparseVR_teacher_model_acted_only.onnx", "sparseVR_teacher_model_acted_only"),
    "B": (MODELS / "sparseVR_teacher_model_acted_only_vol-aug-k1_excluding-EmojiHero.onnx",
          "sparseVR_teacher_model_acted_only_vol-aug-k1_excluding-EmojiHero"),
}
E = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

sessions = []
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if meta.get("protocol_version") == "session-local-v1" and meta.get("calibration", {}).get("calibration_valid") in ("true", None):
        sessions.append((d, meta["models"]["face_model_asset"]))

cond = cs.Condition()
rows = []
for d, recorded_with in sessions:
    try:
        log = sa.load_session(d / "predictions.csv")
    except (Exception, SystemExit) as exc:
        print(f"skipped {d.name}: {exc}")
        continue
    row = {"session": d.name[8:], "recorded": recorded_with}
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
        expressive = k & (data.prompted != 4)
        weak = expressive & ~(data.windows[..., :63] > 0.25).any(axis=(1, 2))
        row[f"{name} tick acc"] = float(np.mean(pred[k] == data.prompted[k]))
        row[f"{name} weak acc"] = float(np.mean(pred[weak] == data.prompted[weak])) if weak.any() else np.nan
        row[f"{name} ->Neu"] = float(np.mean(pred[expressive] == 4))
        row["weak share"] = float(weak.sum() / expressive.sum())
        row["n weak"] = int(weak.sum())
        row["face size"] = float(np.median(data.windows[expressive][..., :63].sum(-1).mean(-1)))
        for e in range(7):
            m = k & (data.prompted == e)
            row[f"{name} {E[e]}"] = float(np.mean(pred[m] == e))
    rows.append(row)

import pandas as pd
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df[["session", "face size", "weak share", "n weak",
          "D raw", "B raw", "D pers", "B pers",
          "D tick acc", "B tick acc", "D weak acc", "B weak acc", "D ->Neu", "B ->Neu"]].round(2).to_string(index=False))
print("\nmeans over sessions:")
print(df[["D raw", "B raw", "D pers", "B pers", "D tick acc", "B tick acc", "D ->Neu", "B ->Neu"]].mean().round(3).to_string())
w = df["n weak"]
for n in ("D", "B"):
    print(f"{n} weak-tick acc pooled: {np.nansum(df[f'{n} weak acc'] * w) / w.sum():.3f} (n={w.sum()})")
for c in ("raw", "pers"):
    dl = df[f"B {c}"] - df[f"D {c}"]
    print(f"B - D trial macro-F1 {c}: mean {dl.mean():+.3f}  better {(dl > 0).sum()}  worse {(dl < 0).sum()}  tied {(dl == 0).sum()}")
print("\nper-emotion eval tick accuracy, mean over sessions (D / B):")
for e in E:
    print(f"  {e:<10} {df[f'D {e}'].mean():.2f} / {df[f'B {e}'].mean():.2f}")
print("\nnewest session per emotion (D / B):")
last = df.iloc[-1]
for e in E:
    print(f"  {e:<10} {last[f'D {e}']:.2f} / {last[f'B {e}']:.2f}")
