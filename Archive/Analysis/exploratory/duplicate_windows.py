"""Duplicate and near-duplicate windows in teacher_arrays_with_emojihero.npz.

Exact: identical after rounding to 3 decimals. Near: Euclidean distance on the flat 350 values
< 0.05 (about 0.003 per value). Checked inside each split, and test/val against train.
Also: duplicates that carry DIFFERENT labels (the same face, two answers).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[2]
Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
E = list(Z["class_names"])
NEAR = 0.05

sets = {"train/" + s: (Z["X_train"][Z["S_train"] == k], Z["Y_train"][Z["S_train"] == k])
        for k, s in enumerate(["old", "eh", "new"])}
for sp in ("val", "test"):
    for s in ("old", "new", "eh"):
        sets[f"{sp}/{s}"] = (Z[f"X_{sp}_{s}"], Z[f"Y_{sp}_{s}"])

print("== inside each set ==")
for name, (X, Y) in sets.items():
    F = X.reshape(len(X), -1)
    _, inv, cnt = np.unique(np.round(F, 3), axis=0, return_inverse=True, return_counts=True)
    in_dup = cnt[inv.ravel()] > 1
    d, i = NearestNeighbors(n_neighbors=2).fit(F).kneighbors(F)
    near = d[:, 1] < NEAR
    conflict = near & (Y[i[:, 1]] != Y)
    print(f"{name:<10} n={len(X):>5}  exact-dup windows {in_dup.sum():>5} ({in_dup.mean():.0%})  "
          f"distinct {len(cnt):>5}  near-dup {near.mean():.0%}  near-dup with other label {conflict.sum()}")

print("\n== val/test against train (nearest training window) ==")
Ftr = Z["X_train"].reshape(len(Z["X_train"]), -1)
Ytr = Z["Y_train"]
nn = NearestNeighbors(n_neighbors=1).fit(Ftr)
rows = []
for name, (X, Y) in sets.items():
    if name.startswith("train"):
        continue
    d, i = nn.kneighbors(X.reshape(len(X), -1))
    near = d[:, 0] < NEAR
    same = near & (Ytr[i[:, 0]] == Y)
    print(f"{name:<10} near-copy of a train window: {near.sum():>5} ({near.mean():.0%})  "
          f"same label {same.sum()}  other label {(near & ~same).sum()}")

# the test EH copies: how many distinct faces, and how D did on them
df = pd.read_csv(REPO / "Analysis/results/test_error_windows.csv")
eh = df[df.source == "eh"].reset_index(drop=True)
X = Z["X_test_eh"].reshape(len(Z["X_test_eh"]), -1)
_, inv, cnt = np.unique(np.round(X, 3), axis=0, return_inverse=True, return_counts=True)
eh["group"], eh["copies"] = inv.ravel(), cnt[inv.ravel()]
print("\n== EmojiHero test: copies per distinct window ==")
print(pd.Series(cnt).value_counts().sort_index().head(12).to_string())
print("D errors on EH test:", (~eh.correct_D).sum(), " distinct erroneous windows:",
      eh[~eh.correct_D].group.nunique())
for pair in ("Fear->Surprise", "Anger->Disgust"):
    e = eh[eh.pair == pair]
    print(f"{pair}: {len(e)} EH errors from {e.group.nunique()} distinct windows; "
          f"largest group {e.group.value_counts().iloc[0]}")
