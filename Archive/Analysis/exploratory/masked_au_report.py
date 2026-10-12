"""Report for masked_au_aux.py: beta chosen on VALIDATION only (same rule for B and C), then the
selected B and C against A, paired by repeat. Test and headset numbers are exploratory."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

REPO = Path(__file__).resolve().parents[2]
E = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
import sys
CSV = sys.argv[1] if len(sys.argv) > 1 else "Analysis/results/masked_au_aux.csv"
df = pd.read_csv(REPO / CSV)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

print("== beta selection, VALIDATION mean of old and new acted macro-F1 (mean of 3 repeats)")
sel = df[df.arm.isin(["B", "C"])].groupby(["arm", "beta"])["val mean"].mean().unstack().round(4)
print(sel.to_string())
best = {a: float(sel.loc[a].idxmax()) for a in ("B", "C")}
print("selected beta:", best, "| A val mean", round(df[df.arm == "A"]["val mean"].mean(), 4))

runs = {"A": df[df.arm == "A"], "B": df[(df.arm == "B") & (df.beta == best["B"])],
        "C": df[(df.arm == "C") & (df.beta == best["C"])]}
runs = {k: v.set_index("repeat").sort_index() for k, v in runs.items()}
D = df[df.arm == "D"].iloc[0]

if "pull log" in df.columns:
    print("\n== shared-encoder gradients (median over probes, mean over 3 repeats), every beta")
    rows = []
    for _, r in df[df.arm.isin(["A", "B", "C"])].iterrows():
        for e in json.loads(r["pull log"]) if isinstance(r["pull log"], str) else []:
            rows.append({"arm": r.arm, "beta": r.beta, **e})
    g = pd.DataFrame(rows)
    if len(g):
        t = g.groupby(["arm", "beta", "epoch"])[["g_cls", "beta*g_rec", "g_total", "ratio", "cosine"]].mean()
        print(t.round(3).to_string())

main = ["val old", "val new", "val mean", "test old", "test new", "test VR pooled", "test eh",
        "VR emotion->Neutral", "VR Neutral->Anger", "VR other emotion->Anger",
        "headset developer tick F1 (mean of 9)", "headset 26908f tick F1",
        "hs dev Anger acc", "hs dev emotion->Neutral", "hs dev Neutral->Anger", "hs dev other emotion->Anger",
        "hs other Anger acc", "hs other emotion->Neutral", "hs other Neutral->Anger"]
print("\n== per run (selected beta)")
per = pd.concat({k: v[main] for k, v in runs.items()}, names=["arm"]).T.round(3)
print(per.to_string())
print("\n== means; D = deployed reference (trained WITH EmojiHero, a separate change)")
m = pd.DataFrame({k: v[main].mean() for k, v in runs.items()})
m["D"] = [D[c] for c in main]
print(m.round(3).to_string())

print("\n== paired differences by repeat (mean, per-repeat values, paired t p)")
for a, b in (("B", "A"), ("B", "C"), ("C", "A")):
    print(f"-- {a} - {b}")
    for c in main:
        d = (runs[a][c] - runs[b][c]).values
        p = ttest_rel(runs[a][c], runs[b][c]).pvalue if np.any(d != 0) else 1.0
        print(f"   {c:<40} {d.mean():+.3f}   {np.round(d, 3).tolist()}   p={p:.2f}")

print("\n== per-class F1 (mean of 3 repeats)")
for src in ("old", "new", "VR"):
    cols = [f"{src} F1 {e}" for e in E]
    t = pd.DataFrame({k: v[cols].mean().values for k, v in runs.items()}, index=E)
    t["D"] = [D[c] for c in cols]
    print(f"-- {src}"); print(t.round(3).T.to_string())
cols = [f"eh F1 {e}" for e in E]
t = pd.DataFrame({k: v[cols].mean().values for k, v in runs.items()}, index=E); t["D"] = [D[c] for c in cols]
print("-- EmojiHero (not in A/B/C training)"); print(t.round(3).T.to_string())

print("\n== VR pooled confusion counts, summed over the 3 repeats (rows = true, cols = predicted)")
for k, v in runs.items():
    C = sum(np.array(json.loads(s)) for s in v["VR confusion"])
    print(f"-- {k}"); print(pd.DataFrame(C, index=E, columns=[e[:4] for e in E]).to_string())
C = np.array(json.loads(D["VR confusion"])); print("-- D (one model)"); print(pd.DataFrame(C, index=E, columns=[e[:4] for e in E]).to_string())

print("\n== headset sessions, tick macro-F1 (mean of 3 repeats)")
hs = [c for c in df.columns if c.startswith("hs ") and len(c) == 9]
t = pd.DataFrame({k: v[hs].mean() for k, v in runs.items()}); t["D"] = [D[c] for c in hs]
print(t.round(3).to_string())
for a in ("B", "C"):
    d = pd.Series({c: (runs[a][c] - runs["A"][c]).mean() for c in hs if c != "hs 26908f"})
    print(f"developer sessions {a} better / worse than A: {(d > 0).sum()} / {(d < 0).sum()}")
