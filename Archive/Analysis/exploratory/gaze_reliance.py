"""How much does model D rely on the 8 gaze AUs (EyesLook*)? No training, model D as deployed.

Two probes on the input, everything else unchanged:
  zero     gaze AUs set to 0 (out of distribution: real gaze is rarely all 0)
  shuffle  gaze AUs of each window replaced by those of a random other window of the same set
           (keeps realistic gaze values, breaks any link between gaze and label)
Measured on the test sets (window macro-F1) and on headset replay (raw argmax, evaluation block,
tick macro-F1). Shuffle is repeated 5 times and averaged. Also: the share of predictions that flip.
Blink (EyesClosed) is left in: it is part of real expressions (e.g. squinting in Happiness).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
sys.path.insert(0, str(REPO / "Analysis/exploratory"))
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import f1_score

import calibration_study as cs
import session_analysis as sa
from jitter_headset_replay import sessions, E

Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
AU = list(Z["au_names"])
GAZE = np.array([a.startswith("EyesLook") for a in AU])
D = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
sess = ort.InferenceSession(str(D))
run = lambda X: sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})[0]
REPEATS = 5


def probes(X, rng):
    z = X.copy(); z[..., GAZE] = 0
    yield "zero", z
    for _ in range(REPEATS):
        s = X.copy(); s[..., GAZE] = X[rng.permutation(len(X))][..., GAZE]
        yield "shuffle", s


def main():
    rng = np.random.default_rng(0)
    rows = []
    for src, name in (("old", "old acted"), ("new", "new acted"), ("eh", "EmojiHero")):
        X, Y = Z[f"X_test_{src}"], Z[f"Y_test_{src}"]
        base = run(X).argmax(1)
        rows.append({"data": f"test {name}", "probe": "none", "F1": f1_score(Y, base, average="macro"), "flip": 0.0})
        for probe, Xp in probes(X, rng):
            p = run(Xp).argmax(1)
            rows.append({"data": f"test {name}", "probe": probe, "F1": f1_score(Y, p, average="macro"),
                         "flip": np.mean(p != base)})

    for d in sessions():
        data = cs.replay(sa.load_session(d / "predictions.csv"), D)
        _, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
        X, y = data.windows[test].astype(np.float32), data.prompted[test]
        yt = [E[i] for i in y]
        base = run(X).argmax(1)
        tag = f"headset {d.name[8:]}"
        rows.append({"data": tag, "probe": "none", "F1": sa.macro_f1(yt, [E[i] for i in base], E)["macro_f1"], "flip": 0.0})
        for probe, Xp in probes(X, rng):
            p = run(Xp).argmax(1)
            rows.append({"data": tag, "probe": probe, "F1": sa.macro_f1(yt, [E[i] for i in p], E)["macro_f1"],
                         "flip": np.mean(p != base)})

    df = pd.DataFrame(rows)
    df["group"] = np.where(df.data.str.startswith("headset"), "headset replay (mean of sessions)", df.data)
    t = df.groupby(["group", "probe"])[["F1", "flip"]].mean().unstack("probe")
    pd.set_option("display.width", 200)
    print(t.round(3))
    hs = df[df.data.str.startswith("headset")].groupby(["data", "probe"]).F1.mean().unstack()
    print("\nheadset sessions where shuffled gaze is better / worse than none:",
          int((hs["shuffle"] > hs["none"]).sum()), "/", int((hs["shuffle"] < hs["none"]).sum()))
    print("headset sessions where zero gaze is better / worse than none:",
          int((hs["zero"] > hs["none"]).sum()), "/", int((hs["zero"] < hs["none"]).sum()))

    # gaze level by source: is gaze itself different between data sets?
    print("\nmean gaze AU value (all frames), by data set:")
    for k, s in enumerate(["old", "eh", "new"]):
        print(f"  train {s:<4}", round(float(Z["X_train"][Z["S_train"] == k][..., GAZE].mean()), 3))
    hsX = np.concatenate([cs.replay(sa.load_session(d / "predictions.csv"), D).windows for d in sessions()])
    print("  headset   ", round(float(hsX[..., GAZE].mean()), 3))
    df.to_csv(REPO / "Analysis/results/gaze_reliance.csv", index=False)


if __name__ == "__main__":
    main()
