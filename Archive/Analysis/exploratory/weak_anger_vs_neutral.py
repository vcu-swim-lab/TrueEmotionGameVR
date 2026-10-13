"""Weak Anger vs Neutral on the headset. Model D unchanged, no calibration.

Ticks: every session-local-v1 session, both blocks (used only as labelled data), ticks 1-9.
Groups by D's raw argmax: Anger ok, Anger->Neutral, Neutral ok, Neutral->Anger.

1. Group profiles: face size, and for lip press, lip tightening, brow lowering (and a few
   neighbours: lid tightener, chin raiser, lip corner depressor, nose wrinkler) the peak over the
   5 frames and the change (last - first frame). L/R averaged.
2. Same strength: each Anger->Neutral tick is matched with the Neutral-ok tick of the SAME
   session with the closest face size. Paired differences per cue, per session.
3. Candidate rules, applied only to ticks D calls Neutral ("flip to Anger if ..."), fitted with
   whole sessions held out (leave-one-session-out over the developer's 9 sessions; 26908f, the only
   other person, is never used for fitting and is scored separately):
     scores  D's own probabilities: flip if log p(Anger) - log p(Neutral) > t
     cues    logistic regression on the cue features above (+ face size)
     both    cues + D's log-ratio
   The threshold t is chosen inside the training sessions (the largest Anger gain with at most
   +1 Neutral->Anger per training session on average). Outcome on the held-out session: Anger
   ticks recognised, Neutral ticks called Anger, and other emotions turned into Anger.
4. Acted test windows (old, new: different people) as a further check of the same rules, with
   the caveat that their windows passed the 0.25 filter.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
sys.path.insert(0, str(REPO / "Analysis/exploratory"))
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import calibration_study as cs
import session_analysis as sa
from jitter_headset_replay import sessions

Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
AU = [str(a) for a in Z["au_names"]]
E = [str(c) for c in Z["class_names"]]
ANG, NEU = E.index("Anger"), E.index("Neutral")
OTHER = "26908f"
D_PATH = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
CUES = ["LipPressor", "LipTightener", "BrowLowerer", "LidTightener", "ChinRaiser", "LipCornerDepressor",
        "NoseWrinkler"]


def cue_idx(name):
    if name == "ChinRaiser":
        return [AU.index("ChinRaiserB"), AU.index("ChinRaiserT")]
    return [AU.index(name + "L"), AU.index(name + "R")]


def feats(X, P):
    f = {"size": X[..., :63].sum(-1).mean(1)}
    for c in CUES:
        s = X[..., cue_idx(c)].mean(-1)
        f[f"peak {c}"] = s.max(1)
        f[f"change {c}"] = s[:, -1] - s[:, 0]
    f["D log A/N"] = np.log(P[:, ANG] + 1e-6) - np.log(P[:, NEU] + 1e-6)
    return pd.DataFrame(f).astype("float64")


def load():
    rows, Xs, Ps = [], [], []
    for d in sessions():
        data = cs.replay(sa.load_session(d / "predictions.csv"), D_PATH)
        k = data.tick >= 1
        Xs.append(data.windows[k]); Ps.append(data.scores[k])
        for y, p, tr, tk in zip(data.prompted[k], data.scores[k].argmax(1), data.trial[k], data.tick[k]):
            rows.append({"session": d.name[-6:], "true": E[y], "pred": E[p], "trial": tr, "tick": tk})
    df = pd.DataFrame(rows)
    X, P = np.concatenate(Xs), np.concatenate(Ps)
    out = pd.concat([df, feats(X, P)], axis=1)
    finite = np.isfinite(out.select_dtypes("number")).all(axis=1) & np.isfinite(X).all(axis=(1, 2))
    if (~finite).any():
        print(f"dropped {int((~finite).sum())} ticks with non-finite values:",
              out[~finite].groupby(["session", "true"]).size().to_dict())
    return out[finite].reset_index(drop=True)


def groups(df):
    return {"Anger ok": (df.true == "Anger") & (df.pred == "Anger"),
            "Anger->Neutral": (df.true == "Anger") & (df.pred == "Neutral"),
            "Neutral ok": (df.true == "Neutral") & (df.pred == "Neutral"),
            "Neutral->Anger": (df.true == "Neutral") & (df.pred == "Anger")}


def step1(df):
    print("=" * 78 + "\n1. GROUP PROFILES (headset ticks, D raw), medians\n" + "=" * 78)
    cols = ["size"] + [f"peak {c}" for c in CUES] + [f"change {c}" for c in CUES[:3]]
    for who, m in (("developer (9 sessions)", df.session != OTHER), (f"other person ({OTHER})", df.session == OTHER)):
        sub = df[m]
        t = pd.DataFrame({k: sub[v[m]][cols].median() for k, v in groups(df).items()}).T
        t["n"] = [int(v[m].sum()) for v in groups(df).values()]
        print(f"\n{who}")
        print(t.round(3).to_string())


def step2(df):
    print("\n" + "=" * 78 + "\n2. SAME STRENGTH: each Anger->Neutral tick vs the closest-size Neutral-ok tick, same session\n" + "=" * 78)
    g = groups(df)
    pairs = []
    for s, sub in df.groupby("session"):
        a = sub[g["Anger->Neutral"][sub.index]]
        n = sub[g["Neutral ok"][sub.index]]
        if a.empty or n.empty:
            continue
        for _, r in a.iterrows():
            j = (n["size"] - r["size"]).abs().idxmin()
            d = {"session": s, "size gap": r["size"] - n.loc[j, "size"]}
            for c in CUES:
                d[f"peak {c}"] = r[f"peak {c}"] - n.loc[j, f"peak {c}"]
                d[f"change {c}"] = r[f"change {c}"] - n.loc[j, f"change {c}"]
            pairs.append(d)
    p = pd.DataFrame(pairs)
    print(f"{len(p)} pairs; median |size gap| {p['size gap'].abs().median():.3f}")
    cols = [c for c in p.columns if c.startswith(("peak", "change"))]
    per = p.groupby("session")[cols].median()
    summ = pd.DataFrame({"median diff (Anger - Neutral)": p[cols].median(),
                         "share of pairs Anger higher": (p[cols] > 0).mean(),
                         "sessions +": (per > 0.005).sum(), "sessions -": (per < -0.005).sum()})
    print(summ.round(3).to_string())
    print(f"\nper-session median differences for the three main cues (+ = higher in weak Anger); {OTHER} = other person")
    print(per[[f"peak {c}" for c in CUES[:3]] + [f"change {c}" for c in CUES[:3]]].round(3).to_string())
    return summ


def choose_t(score, y_anger, y_neutral, sessions_n):
    """Largest Anger gain with at most +1 Neutral->Anger per training session on average."""
    best = np.inf
    for t in np.unique(np.quantile(score, np.linspace(0, 1, 201))):
        if (score[y_neutral] > t).sum() <= sessions_n:
            best = min(best, t)
    return best


def step3(df):
    print("\n" + "=" * 78 + "\n3. CANDIDATE RULES, only on ticks D calls Neutral, whole sessions held out\n" + "=" * 78)
    cue_cols = ["size"] + [f"peak {c}" for c in CUES] + [f"change {c}" for c in CUES]
    variants = {"scores": ["D log A/N"], "cues": cue_cols, "both": cue_cols + ["D log A/N"]}
    devs = [s for s in df.session.unique() if s != OTHER]
    calledN = df.pred == "Neutral"
    rows = []
    for name, all_cols in variants.items():
        for held in devs + [OTHER]:
            cols = all_cols
            train = calledN & ~df.session.isin([held, OTHER])
            tr = df[train & df.true.isin(["Anger", "Neutral"])]
            if name == "scores":
                score_tr = tr["D log A/N"].values
                model = None
            else:
                cols = [c for c in cols if tr[c].std() > 1e-3]      # cues that never move are left out
                model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))
                model.fit(tr[cols], tr.true == "Anger")
                score_tr = model.decision_function(tr[cols])
            t = choose_t(score_tr, (tr.true == "Anger").values, (tr.true == "Neutral").values,
                         tr.session.nunique())
            te = df[(df.session == held)]
            sc = te["D log A/N"].values if model is None else model.decision_function(te[cols])
            flip = (te.pred == "Neutral").values & (sc > t)
            new = np.where(flip, "Anger", te.pred.values)
            ya, yn = (te.true == "Anger").values, (te.true == "Neutral").values
            rows.append({"rule": name, "session": held, "Anger n": ya.sum(),
                         "Anger D": int((te.pred.values[ya] == "Anger").sum()), "Anger rule": int((new[ya] == "Anger").sum()),
                         "Neutral n": yn.sum(),
                         "Neutral->Anger D": int((te.pred.values[yn] == "Anger").sum()),
                         "Neutral->Anger rule": int((new[yn] == "Anger").sum()),
                         "other->Anger added": int((flip & ~ya & ~yn).sum())})
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for who, m in (("developer sessions (each held out in turn)", r.session != OTHER),
                   (f"other person {OTHER} (never used for fitting; rule fitted on all 9 developer sessions)", r.session == OTHER)):
        print(f"\n{who}: totals")
        print(r[m].groupby("rule")[["Anger n", "Anger D", "Anger rule", "Neutral n", "Neutral->Anger D",
                                    "Neutral->Anger rule", "other->Anger added"]].sum().to_string())
    dev = r[r.session != OTHER].copy()
    dev["anger gain"] = dev["Anger rule"] - dev["Anger D"]
    dev["false gain"] = dev["Neutral->Anger rule"] - dev["Neutral->Anger D"]
    print("\nper developer session: Anger gain / extra false Anger (Neutral) / other emotions turned Anger")
    print(dev.pivot(index="session", columns="rule", values=["anger gain", "false gain", "other->Anger added"]).to_string())
    r.to_csv(REPO / "Analysis/results/weak_anger_vs_neutral_rules.csv", index=False)

    # rule weights, fitted on all developer sessions
    tr = df[calledN & (df.session != OTHER) & df.true.isin(["Anger", "Neutral"])]
    cols = [c for c in variants["cues"] if tr[c].std() > 1e-3]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")).fit(
        tr[cols], tr.true == "Anger")
    w = pd.Series(m[-1].coef_[0], index=cols).sort_values()
    print("cues left out (never move on the headset):", [c for c in variants["cues"] if c not in cols])
    print("\ncue rule weights (all developer sessions; + = Anger):",
          ", ".join(f"{k} {v:+.2f}" for k, v in w.items() if abs(v) >= 0.1))
    return variants


def step4(df, variants):
    print("\n" + "=" * 78 + "\n4. ACTED TEST WINDOWS (other people), rules fitted on all developer sessions\n" + "=" * 78)
    sess = ort.InferenceSession(str(D_PATH))
    calledN = df.pred == "Neutral"
    tr = df[calledN & (df.session != OTHER) & df.true.isin(["Anger", "Neutral"])]
    for src in ("old", "new"):
        X, Y = Z[f"X_test_{src}"], Z[f"Y_test_{src}"]
        P = sess.run(None, {sess.get_inputs()[0].name: X})[0]
        F, p = feats(X, P), P.argmax(1)
        for name, cols in variants.items():
            if name == "scores":
                sc_tr, sc = tr["D log A/N"].values, F["D log A/N"].values
            else:
                cols = [c for c in cols if tr[c].std() > 1e-3]
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")).fit(
                    tr[cols], tr.true == "Anger")
                sc_tr, sc = m.decision_function(tr[cols]), m.decision_function(F[cols])
            t = choose_t(sc_tr, (tr.true == "Anger").values, (tr.true == "Neutral").values, tr.session.nunique())
            new = np.where((p == NEU) & (sc > t), ANG, p)
            print(f"  {src:<4} {name:<7} Anger {int(((p == ANG) & (Y == ANG)).sum())} -> {int(((new == ANG) & (Y == ANG)).sum())}"
                  f" of {(Y == ANG).sum()} | Neutral called Anger {int(((p == ANG) & (Y == NEU)).sum())} -> "
                  f"{int(((new == ANG) & (Y == NEU)).sum())} of {(Y == NEU).sum()} | Anger windows D called Neutral: "
                  f"{int(((p == NEU) & (Y == ANG)).sum())}")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    df = load()
    print(df.groupby("session").size().to_dict())
    step1(df)
    step2(df)
    v = step3(df)
    step4(df, v)
