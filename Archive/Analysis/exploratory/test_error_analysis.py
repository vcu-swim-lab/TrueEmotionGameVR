"""Error analysis of the teacher on the held-out test windows, with each error put in one category.

Data: notebooks/teacher_arrays_with_emojihero.npz (model D's exact arrays). Test sets: old acted,
new acted, EmojiHero. Window level (5 frames x 70 AUs), not tick level.

Models: D (deployed ONNX) is the main one. C (volume, ONNX) and V+J (volume then jitter, .keras,
gate tau set to 0.7 after load_weights) are run on the same windows for comparison.

Two labels per error:
  kind  - what went wrong: emotion->Neutral, Neutral->emotion, emotion->emotion
  cause - one bucket, the first rule that matches:
    1 no_expression : true label is an emotion, but every AU > 0.25 in the window is gaze, blink
                      or tongue. The self-filter let it in without a facial expression.
    2 unlike_train  : distance to the nearest training window is above the 95th percentile of
                      the validation windows' nearest-training distance.
    3 looks_like_pred: of the 15 nearest training windows (class-balanced vote), the predicted
                      class wins. The face really resembles the predicted class in training data.
    4 near_miss     : the true class is ranked 2nd and p(pred) - p(true) < 0.2.
    5 model_miss    : the training neighbours vote for the true class, but the model is wrong.
    6 unclear       : none of the above (neighbours vote for a third class).
Training and test are participant-exclusive, so the neighbour search does not see the test person.

Persistence (old and new acted only; their windows are stored in recording order, EmojiHero is
shuffled): an error is "persistent" if a neighbouring window with the same label is wrong with the
same predicted class. Recording boundaries are not stored, so this is approximate.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "Analysis/results"
E = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
NEU = 4
ONNX = REPO.parent / "VREmotionDetection/Runtime/Models"
ONNX_MODELS = {
    "D": ONNX / "sparseVR_teacher_model_acted_only.onnx",
    "C": ONNX / "sparseVR_teacher_model_acted_only_vol-aug-k1_with-EmojiHero.onnx",
}
VJ_KERAS = REPO / "Analysis/data/jitter_volume_keras/with_emojihero__volume-jitter__seed42.keras"
SOURCES = {"old": "Old acted", "new": "New acted", "eh": "EmojiHero"}

Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
AU = list(Z["au_names"])
assert list(Z["class_names"]) == E
NON_EXPRESSIVE = np.array([a.startswith(("EyesLook", "EyesClosed", "Tongue")) for a in AU])
FACE_63 = slice(0, 63)
K = 15


def predict_onnx(path, X):
    s = ort.InferenceSession(str(path))
    return s.run(None, {s.get_inputs()[0].name: X})[0]


def predict_vj(X_all):
    import tensorflow as tf  # noqa: F401
    nb = json.load(open(REPO / "notebooks/teacher_notebook_only_acted_data_volume_augmentation.ipynb",
                        encoding="utf-8"))["cells"]
    arch = "".join(nb[74]["source"])
    assert "def build_moe_single_input" in arch
    scope = {}
    exec(arch, scope)
    model, _ = scope["build_moe_single_input"](
        input_shape=(5, 70), num_classes=7, tau_init=2.0, tau_final=0.7,
        gate_hidden=128, classifier_hidden=256, lb_coef=0.02, ent_coef=0.01)
    model.load_weights(str(VJ_KERAS))
    model.get_layer("gate").tau.assign(0.7)
    return model.predict(X_all, batch_size=512, verbose=0)


def main():
    # ---------- test windows ----------
    rows, Xs = [], []
    for src in SOURCES:
        X, Y = Z[f"X_test_{src}"], Z[f"Y_test_{src}"]
        Xs.append(X)
        for i, y in enumerate(Y):
            rows.append({"source": src, "order": i, "true": int(y)})
    X = np.concatenate(Xs)
    df = pd.DataFrame(rows)

    # ---------- predictions ----------
    P = {m: predict_onnx(p, X) for m, p in ONNX_MODELS.items()}
    P["V+J"] = predict_vj(X)
    for m, p in P.items():
        df[f"pred_{m}"] = p.argmax(1)
    pD = P["D"]
    df["p_pred"] = pD.max(1)
    df["p_true"] = pD[np.arange(len(df)), df.true]
    df["true_rank"] = (pD > df.p_true.values[:, None]).sum(1) + 1

    # ---------- window descriptors ----------
    df["face_size"] = X[..., FACE_63].sum(-1).mean(-1)          # same definition as volume aug
    df["max_au"] = X.max((1, 2))
    expressive_peak = X[..., ~NON_EXPRESSIVE].max((1, 2))
    df["max_expressive_au"] = expressive_peak
    df["no_expression"] = (df.true != NEU) & (expressive_peak <= 0.25)
    df["motion"] = np.abs(np.diff(X, axis=1)).sum(-1).mean(-1)  # mean frame-to-frame change
    top = X.mean(1).argsort(1)[:, ::-1][:, :3]
    df["top_aus"] = [", ".join(AU[j] for j in t) for t in top]

    # ---------- training neighbours ----------
    Xtr, Ytr = Z["X_train"].reshape(len(Z["X_train"]), -1), Z["Y_train"]
    nn = NearestNeighbors(n_neighbors=K).fit(Xtr)
    dist, idx = nn.kneighbors(X.reshape(len(X), -1))
    Xval = np.concatenate([Z[f"X_val_{s}"] for s in SOURCES]).reshape(-1, 350)
    dval = nn.kneighbors(Xval, n_neighbors=1)[0][:, 0]
    ood_cut = np.quantile(dval, 0.95)
    df["nn_dist"] = dist[:, 0]
    w_cls = 1.0 / np.bincount(Ytr, minlength=7)
    votes = np.zeros((len(X), 7))
    for j in range(K):
        np.add.at(votes, (np.arange(len(X)), Ytr[idx[:, j]]), w_cls[Ytr[idx[:, j]]])
    votes /= votes.sum(1, keepdims=True)
    df["nn_vote"] = votes.argmax(1)
    df["nn_share_true"] = votes[np.arange(len(X)), df.true]

    # ---------- categories for D ----------
    wrong = df.pred_D != df.true
    df["correct_D"] = ~wrong
    kind = np.where(df.pred_D == NEU, "emotion->Neutral",
                    np.where(df.true == NEU, "Neutral->emotion", "emotion->emotion"))
    df["kind"] = np.where(wrong, kind, "")
    cause = np.select(
        [df.no_expression,
         df.nn_dist > ood_cut,
         df.nn_vote == df.pred_D,
         (df.true_rank == 2) & (df.p_pred - df.p_true < 0.2),
         df.nn_vote == df.true],
        ["1 no_expression", "2 unlike_train", "3 looks_like_pred", "4 near_miss", "5 model_miss"],
        "6 unclear")
    df["cause"] = np.where(wrong, cause, "")
    df["pair"] = np.where(wrong, [f"{E[t]}->{E[p]}" for t, p in zip(df.true, df.pred_D)], "")

    # persistence, ordered sources only
    df["persistent"] = ""
    for src in ("old", "new"):
        s = df[df.source == src]
        t, p = s.true.values, s.pred_D.values
        flag = []
        for i in range(len(s)):
            if p[i] == t[i]:
                flag.append("")
                continue
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(s) and t[j] == t[i]]
            flag.append("persistent" if any(p[j] == p[i] for j in nb) else "isolated")
        df.loc[s.index, "persistent"] = flag

    df["true"] = df.true.map(dict(enumerate(E)))
    for c in ["pred_D", "pred_C", "pred_V+J", "nn_vote"]:
        df[c] = df[c].map(dict(enumerate(E)))
    df.to_csv(OUT / "test_error_windows.csv", index=False, float_format="%.4f")

    # ---------- report ----------
    pd.set_option("display.width", 200)
    print(f"OOD cut (95th pct of val nearest-train distance): {ood_cut:.3f}")
    print("\n== Window-level macro-F1 and errors per source ==")
    for src in SOURCES:
        s = df[df.source == src]
        line = [f"{SOURCES[src]:<10} n={len(s):>4}"]
        for m in ("D", "C", "V+J"):
            line.append(f"{m}: F1 {f1_score(s.true, s[f'pred_{m}'], average='macro'):.3f}"
                        f" err {(s.true != s[f'pred_{m}']).sum():>4}")
        print("  ".join(line))

    err = df[~df.correct_D]
    print(f"\nD errors: {len(err)} of {len(df)}")
    print("\n== Kind x source (D) ==")
    print(pd.crosstab(err.kind, err.source, margins=True))
    print("\n== Cause x source (D) ==")
    print(pd.crosstab(err.cause, err.source, margins=True))
    print("\n== Cause x kind (D) ==")
    print(pd.crosstab(err.cause, err.kind, margins=True))
    print("\n== Top 15 confusion pairs (D) ==")
    print(err.groupby(["pair"]).size().sort_values(ascending=False).head(15).to_string())
    print("\n== Top pairs per source ==")
    for src in SOURCES:
        e = err[err.source == src]
        print(SOURCES[src], e.pair.value_counts().head(6).to_dict())
    print("\n== Per true class: error rate and main cause (D) ==")
    g = df.groupby("true")
    tab = pd.DataFrame({"n": g.size(), "err_rate": 1 - g.correct_D.mean()})
    tab["main_kind"] = err.groupby("true").kind.agg(lambda s: s.value_counts().index[0])
    tab["main_cause"] = err.groupby("true").cause.agg(lambda s: s.value_counts().index[0])
    print(tab.round(3))
    print("\n== Cause x true class (D) ==")
    print(pd.crosstab(err.cause, err.true))
    print("\n== Persistence (old+new acted) ==")
    print(pd.crosstab(err[err.source != "eh"].persistent, err[err.source != "eh"].source, margins=True))
    print(pd.crosstab(err[err.source != "eh"].cause, err[err.source != "eh"].persistent))

    print("\n== Confidence of errors (D) ==")
    conf = pd.cut(err.p_pred, [0, .5, .8, 1.0001], labels=["<0.5", "0.5-0.8", ">=0.8"])
    print(pd.crosstab(err.cause, conf, margins=True))
    print("\n== Descriptors: correct vs wrong (median) ==")
    print(df.groupby(["source", "correct_D"])[["face_size", "max_expressive_au", "motion", "nn_dist", "p_pred"]]
          .median().round(3))
    print("\n== Face size: emotion->Neutral errors vs correct emotion windows (median) ==")
    emo = df[df.true != "Neutral"]
    print("correct", emo[emo.correct_D].face_size.median().round(3),
          " ->Neutral", err[err.kind == "emotion->Neutral"].face_size.median().round(3))

    print("\n== Do C or V+J fix D's errors? (per cause; share of D errors each model gets right) ==")
    fix = pd.DataFrame({m: (err[f"pred_{m}"] == err.true).groupby(err.cause).mean() for m in ("C", "V+J")})
    fix["n"] = err.cause.value_counts()
    print(fix.round(3))
    print("new errors created (D right, model wrong):",
          {m: int(((df[f"pred_{m}"] != df.true) & df.correct_D).sum()) for m in ("C", "V+J")})
    print("\nno_expression windows in test (any outcome):",
          df.groupby("source").no_expression.sum().to_dict(),
          " D accuracy on them:", round(df[df.no_expression].correct_D.mean(), 3))


if __name__ == "__main__":
    main()
