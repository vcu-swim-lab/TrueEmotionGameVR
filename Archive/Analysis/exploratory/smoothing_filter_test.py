"""Idea 2: does smoothing the AU values before the model help? Offline, deployed model D only.

The logs keep 5 frames per tick (about 0.1 s apart) and nothing between ticks, so every filter
works inside one 5-frame window, along the frame axis, per AU. No retraining: D sees filtered
input at test time only.

Filters: median3 / mean3 (3-frame sliding, edge-padded), ema0.5 (causal exponential average,
which is what a One-Euro filter does on a slow signal), median5 / mean5 (all 5 frames replaced
by the window median / mean).

Data: test set (old VR, new VR, EmojiHero) from notebooks/teacher_arrays_with_emojihero.npz, and
every session-local-v1 headset session (evaluation block: raw and movement calibration).
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
from jitter_headset_replay import E, ONNX_MODELS, evaluate, sessions


def _sliding(x, fn):
    p = np.concatenate([x[:, :1], x, x[:, -1:]], axis=1)
    return fn(np.stack([p[:, :-2], p[:, 1:-1], p[:, 2:]], axis=0), axis=0)


def _ema(x, a=0.5):
    y = x.copy()
    for t in range(1, x.shape[1]):
        y[:, t] = a * x[:, t] + (1 - a) * y[:, t - 1]
    return y


FILTERS = {
    "raw": lambda x: x,
    "median3": lambda x: _sliding(x, np.median),
    "mean3": lambda x: _sliding(x, np.mean),
    "ema0.5": _ema,
    "median5": lambda x: np.repeat(np.median(x, axis=1, keepdims=True), x.shape[1], axis=1),
    "mean5": lambda x: np.repeat(x.mean(axis=1, keepdims=True), x.shape[1], axis=1),
}

session = ort.InferenceSession(str(ONNX_MODELS["D (none, EH)"]))
INPUT = session.get_inputs()[0].name


def predict(x):
    return session.run(None, {INPUT: x.astype(np.float32)})[0]


def main():
    z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
    assert list(z["class_names"]) == E
    rows = []
    for split in ("test_old", "test_new", "test_eh"):
        X, Y = z[f"X_{split}"], z[f"Y_{split}"]
        for name, f in FILTERS.items():
            p = predict(f(X)).argmax(1)
            yt, pn = [E[i] for i in Y], [E[i] for i in p]
            rows.append({"data": split, "filter": name, "macro_f1": sa.macro_f1(yt, pn, E)["macro_f1"]})
    test = pd.DataFrame(rows).pivot(index="filter", columns="data", values="macro_f1").loc[list(FILTERS)]

    hrows = []
    for d in sessions():
        data = cs.replay(sa.load_session(d / "predictions.csv"), ONNX_MODELS["D (none, EH)"])
        raw_windows = data.windows
        for name, f in FILTERS.items():
            data.windows = f(raw_windows)  # movement calibration reads the filtered AUs too
            hrows.append({"session": d.name[8:], "filter": name, **evaluate(data, predict(data.windows))})
        data.windows = raw_windows
    h = pd.DataFrame(hrows)
    h.to_csv(REPO / "Analysis/results/smoothing_filter_headset.csv", index=False)
    test.to_csv(REPO / "Analysis/results/smoothing_filter_test.csv")

    pd.set_option("display.width", 200)
    print("TEST SET, macro-F1 (model D)")
    print(test.round(3).to_string())
    cols = ["raw tick", "raw trial", "movement tick", "movement trial"]
    print(f"\nHEADSET, {h.session.nunique()} sessions, evaluation block, mean macro-F1")
    print(h.groupby("filter")[cols].mean().loc[list(FILTERS)].round(3).to_string())
    print("\nsessions better / worse than raw input (tick macro-F1)")
    for m in ("raw", "movement"):
        piv = h.pivot(index="session", columns="filter", values=f"{m} tick")
        print(f"  {m:<9}" + "  ".join(f"{n}: {int((piv[n] > piv['raw']).sum())}/{int((piv[n] < piv['raw']).sum())}"
                                     for n in list(FILTERS)[1:]))


if __name__ == "__main__":
    main()
