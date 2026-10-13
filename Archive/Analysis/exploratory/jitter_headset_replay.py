"""Jitter models on the headset sessions: raw, personal calibration and movement calibration.

The jitter models (Colab, seed 42, notebooks/teacher_jitter_volume_augmentation.ipynb) are
.keras files from Analysis/data/jitter_volume_keras.zip. They are rebuilt with the notebook's
architecture (cell 74 of the volume-augmentation notebook), their weights loaded, and the gate
temperature set to 0.7 as in the deployed ONNX (it is not stored in .keras files). The deployed
ONNX models D, C and B are replayed through calibration_study.replay as before.

Every session-local-v1 recording with a valid calibration block. Evaluation block only.
Methods: raw argmax, personal (log-score centroids), movement (score + AU cosines).
"""
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import pandas as pd

import calibration_study as cs
import session_analysis as sa

E = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
EVERY = list(range(7))
ONNX = REPO.parent / "VREmotionDetection/Runtime/Models"
ONNX_MODELS = {
    "D (none, EH)": ONNX / "sparseVR_teacher_model_acted_only.onnx",
    "C (volume, EH)": ONNX / "sparseVR_teacher_model_acted_only_vol-aug-k1_with-EmojiHero.onnx",
    "B (volume, no EH)": ONNX / "sparseVR_teacher_model_acted_only_vol-aug-k1_excluding-EmojiHero.onnx",
}
ZIP = REPO / "Analysis/data/jitter_volume_keras.zip"
KERAS_DIR = REPO / "Analysis/data/jitter_volume_keras"
KERAS_MODELS = {
    "jitter, EH": "with_emojihero__jitter__seed42.keras",
    "volume+jitter, EH": "with_emojihero__volume-jitter__seed42.keras",
    "jitter, no EH": "without_emojihero__jitter__seed42.keras",
    "volume+jitter, no EH": "without_emojihero__volume-jitter__seed42.keras",
}


def keras_predictors():
    import tensorflow as tf  # noqa: F401  (needed by the architecture cell)

    if not KERAS_DIR.exists():
        zipfile.ZipFile(ZIP).extractall(KERAS_DIR)

    nb = json.load(open(REPO / "notebooks/teacher_notebook_only_acted_data_volume_augmentation.ipynb",
                        encoding="utf-8"))["cells"]
    arch = "".join(nb[74]["source"])
    assert "def build_moe_single_input" in arch
    scope = {}
    exec(arch, scope)

    out = {}
    for label, name in KERAS_MODELS.items():
        model, _ = scope["build_moe_single_input"](
            input_shape=(5, 70), num_classes=7, tau_init=2.0, tau_final=0.7,
            gate_hidden=128, classifier_hidden=256, lb_coef=0.02, ent_coef=0.01)
        model.load_weights(str(KERAS_DIR / name))
        model.get_layer("gate").tau.assign(0.7)
        out[label] = model
    return out


def sessions():
    found = []
    for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
        try:
            meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        if (meta.get("protocol_version") == "session-local-v1"
                and meta.get("calibration", {}).get("calibration_valid") in ("true", None)):
            found.append(d)
    return found


def evaluate(data, scores):
    """Tick and trial macro-F1 of raw / personal / movement on the evaluation block."""
    enrol, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
    y = data.prompted
    L = cs.logscore(scores)
    A = cs.movement_feature(data.windows)
    personal_M = cs.centroids(L[enrol], y[enrol], EVERY)
    preds = {
        "raw": scores[test].argmax(1),
        "personal": cs.classify(personal_M, EVERY, L[test], "cosine")[0],
        "movement": cs.classify_movement(personal_M, L[test], cs.centroids(A[enrol], y[enrol], EVERY),
                                         A[test], EVERY)[0],
    }
    yt = np.array([E[i] for i in y[test]])
    out = {}
    for method, p in preds.items():
        pn = np.array([E[i] for i in p])
        _, ty, tp = cs.trial_vote(yt, pn, data.trial[test], data.tick[test], E)
        out[f"{method} tick"] = sa.macro_f1(yt, pn, E)["macro_f1"]
        out[f"{method} trial"] = sa.macro_f1(ty, tp, E)["macro_f1"]
        for e in range(7):
            out[f"{method} {E[e]}"] = float(np.mean(p[y[test] == e] == e))
    return out


def main():
    keras = keras_predictors()
    rows = []
    for d in sessions():
        log = sa.load_session(d / "predictions.csv")
        data = None
        for label, path in ONNX_MODELS.items():
            data = cs.replay(log, path)
            rows.append({"session": d.name[8:], "model": label, **evaluate(data, data.scores)})
        for label, model in keras.items():
            scores = model.predict(data.windows.astype(np.float32), batch_size=256, verbose=0)
            rows.append({"session": d.name[8:], "model": label, **evaluate(data, scores)})

    df = pd.DataFrame(rows)
    order = list(ONNX_MODELS) + list(KERAS_MODELS)
    pd.set_option("display.width", 250)

    cols = [f"{m} {lvl}" for m in ("raw", "personal", "movement") for lvl in ("tick", "trial")]
    print(f"MEAN over {df.session.nunique()} sessions (macro-F1, evaluation block)")
    print(df.groupby("model")[cols].mean().loc[order].round(3).to_string())

    print("\nsessions better / worse than D, tick macro-F1")
    piv = {m: df.pivot(index="session", columns="model", values=f"{m} tick") for m in ("raw", "personal", "movement")}
    for model in order[1:]:
        text = "  ".join(f"{m}: {int((p[model] > p[order[0]]).sum())}/{int((p[model] < p[order[0]]).sum())}"
                         for m, p in piv.items())
        print(f"  {model:<22} {text}")

    print("\nmovement calibration, tick accuracy per emotion (mean over sessions)")
    print(df.groupby("model")[[f"movement {e}" for e in E]].mean().loc[order].round(2)
          .rename(columns=lambda c: c.split()[1]).to_string())
    print("\nraw, tick accuracy per emotion (mean over sessions)")
    print(df.groupby("model")[[f"raw {e}" for e in E]].mean().loc[order].round(2)
          .rename(columns=lambda c: c.split()[1]).to_string())

    df.to_csv(REPO / "Analysis/results/jitter_headset_replay.csv", index=False)


if __name__ == "__main__":
    main()
