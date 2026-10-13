"""Follow-up checks on the test error categories (test_error_analysis.py).

1. No-expression windows: emotion-labelled windows whose only AUs > 0.25 are gaze, blink or
   tongue. Where they come from (split, source, class), which AU let them through the filter,
   and how the test score moves if they are left out.
2. Fear->Surprise and Anger->Disgust: do the misread windows carry the AUs that separate the two
   classes (FACS prototypes mapped to Quest AUs)? Same check on the training windows, to see
   whether the training labels themselves contain the conflict.

FACS cues (mean of left/right, peak over the 5 frames):
  Fear only (vs Surprise) : BrowLowerer (AU4), LidTightener (AU7), LipStretcher (AU20)
  Surprise only (vs Fear) : JawDrop (AU26/27) with brows up and no AU4/AU20
  Anger only (vs Disgust) : LipPressor (AU24), LipTightener (AU23), UpperLidRaiser (AU5)
  Disgust only (vs Anger) : NoseWrinkler (AU9), UpperLipRaiser (AU10)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
E = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
NEU = 4
Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
AU = list(Z["au_names"])
NON_EXPR = np.array([a.startswith(("EyesLook", "EyesClosed", "Tongue")) for a in AU])
ON = 0.25
CUE = 0.15   # a FACS cue counts as present above this peak value


def pair(name):
    return [AU.index(name + "L"), AU.index(name + "R")]


CUES = {
    "BrowLowerer": pair("BrowLowerer"), "LidTightener": pair("LidTightener"),
    "LipStretcher": pair("LipStretcher"), "JawDrop": [AU.index("JawDrop")],
    "InnerBrowRaiser": pair("InnerBrowRaiser"), "OuterBrowRaiser": pair("OuterBrowRaiser"),
    "UpperLidRaiser": pair("UpperLidRaiser"), "LipPressor": pair("LipPressor"),
    "LipTightener": pair("LipTightener"), "NoseWrinkler": pair("NoseWrinkler"),
    "UpperLipRaiser": pair("UpperLipRaiser"), "LipCornerPuller": pair("LipCornerPuller"),
}
FEAR_ONLY = ["BrowLowerer", "LidTightener", "LipStretcher"]
ANGER_ONLY = ["LipPressor", "LipTightener", "UpperLidRaiser"]
DISGUST_ONLY = ["NoseWrinkler", "UpperLipRaiser"]


def cue_table(X):
    return pd.DataFrame({k: X[..., v].mean(-1).max(-1) for k, v in CUES.items()})


def split_frames():
    parts = [("train", s, Z["X_train"][Z["S_train"] == k], Z["Y_train"][Z["S_train"] == k])
             for k, s in enumerate(["old", "eh", "new"])]          # S_train: 0 old, 1 eh, 2 new
    for sp in ("val", "test"):
        for s in ("old", "new", "eh"):
            parts.append((sp, s, Z[f"X_{sp}_{s}"], Z[f"Y_{sp}_{s}"]))
    return parts


def no_expression():
    print("=" * 70, "\n1. NO-EXPRESSION WINDOWS\n" + "=" * 70)
    rows = []
    for sp, s, X, Y in split_frames():
        emo = Y != NEU
        ne = emo & (X[..., ~NON_EXPR].max((1, 2)) <= ON)
        rows.append({"split": sp, "source": s, "emotion_windows": int(emo.sum()),
                     "no_expr": int(ne.sum()), "share": ne.sum() / max(emo.sum(), 1)})
    t = pd.DataFrame(rows)
    print(t.pivot(index="source", columns="split", values="no_expr"))
    print("share of emotion windows:")
    print(t.pivot(index="source", columns="split", values="share").round(3))

    # which class and which AU let them in (train + test pooled)
    Xa = np.concatenate([X for _, _, X, _ in split_frames()])
    Ya = np.concatenate([Y for _, _, _, Y in split_frames()])
    Sa = np.concatenate([[f"{sp}/{s}"] * len(Y) for sp, s, _, Y in split_frames()])
    ne = (Ya != NEU) & (Xa[..., ~NON_EXPR].max((1, 2)) <= ON)
    print("\nno-expression windows per class (all splits):")
    print(pd.Series(Ya[ne]).map(dict(enumerate(E))).value_counts().to_string())
    trigger = Xa[ne].max(1)                       # (n, 70) peak per AU
    trig_names = [AU[j] for j in trigger.argmax(1)]
    grp = pd.Series(trig_names).str.replace(r"(L|R)$", "", regex=True)
    print("\nthe AU that passed the filter (largest one):")
    print(grp.value_counts().head(8).to_string())
    print("\nmedian peak of the expressive AUs in these windows:",
          round(float(np.median(Xa[ne][..., ~NON_EXPR].max((1, 2)))), 3),
          "| in other emotion windows:",
          round(float(np.median(Xa[(Ya != NEU) & ~ne][..., ~NON_EXPR].max((1, 2)))), 3))

    # D's predictions on them, and test score with and without them
    df = pd.read_csv(REPO / "Analysis/results/test_error_windows.csv")
    print("\nD predicts, on the test no-expression windows:")
    print(df[df.no_expression].pred_D.value_counts().to_string())
    print("\ntest macro-F1 with / without no-expression windows (D):")
    for s in ("old", "new", "eh"):
        d = df[df.source == s]
        k = d[~d.no_expression]
        print(f"  {s:<4} all {f1_score(d.true, d.pred_D, average='macro'):.3f}"
              f"  without {f1_score(k.true, k.pred_D, average='macro'):.3f}  (removed {d.no_expression.sum()})")

    # position inside the recording (ordered sources): start / end of a label run?
    print("\nposition of test no-expression windows inside their label run (old+new acted):")
    pos = []
    for s in ("old", "new"):
        d = df[df.source == s].reset_index(drop=True)
        run = (d.true != d.true.shift()).cumsum()
        for _, g in d.groupby(run):
            n = len(g)
            for i, flag in enumerate(g.no_expression):
                if flag:
                    pos.append("edge (first/last 2)" if i < 2 or i >= n - 2 else "middle")
    print(pd.Series(pos).value_counts().to_string())


def compare(title, true_c, pred_c, only_true, only_pred, df, Xtest):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    t, p = E.index(true_c), E.index(pred_c)
    C = cue_table(Xtest)
    groups = {
        f"{true_c} correct": (df.true == true_c) & (df.pred_D == true_c),
        f"{true_c}->{pred_c}": (df.true == true_c) & (df.pred_D == pred_c),
        f"{pred_c} correct": (df.true == pred_c) & (df.pred_D == pred_c),
        f"{pred_c}->{true_c}": (df.true == pred_c) & (df.pred_D == true_c),
    }
    cols = only_true + only_pred + [c for c in ("JawDrop", "InnerBrowRaiser", "OuterBrowRaiser",
                                                "BrowLowerer", "UpperLidRaiser") if c not in only_true + only_pred]
    med = pd.DataFrame({g: C[m.values][cols].median() for g, m in groups.items()}).T
    med["n"] = [int(m.sum()) for m in groups.values()]
    print("median peak AU per group (test, model D):")
    print(med.round(2))

    def kind(Cm):
        a = (Cm[only_true] > CUE).any(axis=1)
        b = (Cm[only_pred] > CUE).any(axis=1)
        return np.select([a & ~b, ~a & b, a & b], [f"{true_c} cues only", f"{pred_c} cues only", "both"],
                         "neither")

    print(f"\ncue check (cue present = peak > {CUE}):")
    out = {}
    for g, m in groups.items():
        out[g] = pd.Series(kind(C[m.values])).value_counts(normalize=True)
    print(pd.DataFrame(out).T.fillna(0).round(2))

    e = df[(df.true == true_c) & (df.pred_D == pred_c)]
    print(f"\n{true_c}->{pred_c} by source:", e.source.value_counts().to_dict(),
          "| by cause:", e.cause.value_counts().to_dict())
    print("confidence (p_pred) median:", round(e.p_pred.median(), 3),
          "| share with p_pred >= 0.8:", round((e.p_pred >= 0.8).mean(), 2))

    # the same cue check on the training labels themselves
    print(f"\ntraining windows labelled {true_c}, cue check per source:")
    for k, s in enumerate(["old", "eh", "new"]):
        m = (Z["Y_train"] == t) & (Z["S_train"] == k)
        print(f"  {s:<4} n={m.sum():>4}", pd.Series(kind(cue_table(Z['X_train'][m]))).value_counts(normalize=True)
              .round(2).to_dict())
    print(f"training windows labelled {pred_c}, cue check per source:")
    for k, s in enumerate(["old", "eh", "new"]):
        m = (Z["Y_train"] == p) & (Z["S_train"] == k)
        print(f"  {s:<4} n={m.sum():>4}", pd.Series(kind(cue_table(Z['X_train'][m]))).value_counts(normalize=True)
              .round(2).to_dict())

    # examples: the most confident errors, with their top AUs
    print(f"\nexamples ({true_c}->{pred_c}, most confident first): source, p_pred, top AUs, cue peaks")
    ex = e.sort_values("p_pred", ascending=False)
    pick = pd.concat([ex.head(4), ex.iloc[len(ex) // 2: len(ex) // 2 + 2], ex.tail(2)])
    for i, r in pick.iterrows():
        cues = {c: round(float(C.loc[i, c]), 2) for c in only_true + only_pred}
        print(f"  #{i:<5} {r.source:<4} p={r.p_pred:.2f}  top: {r.top_aus:<55} {cues}")


def main():
    pd.set_option("display.width", 220)
    no_expression()
    df = pd.read_csv(REPO / "Analysis/results/test_error_windows.csv")
    Xtest = np.concatenate([Z[f"X_test_{s}"] for s in ("old", "new", "eh")])
    compare("2. FEAR -> SURPRISE", "Fear", "Surprise", FEAR_ONLY, ["JawDrop"], df, Xtest)
    compare("3. ANGER -> DISGUST", "Anger", "Disgust", ANGER_ONLY, DISGUST_ONLY, df, Xtest)


if __name__ == "__main__":
    main()
