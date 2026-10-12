"""Before any label-distribution training: what would AU-neighbour target distributions contain?

No training, no test or headset data. Training windows only (validation used once, for D's
confusion matrix as a COMPARISON, never as a target).

Candidate neighbour distribution for training window i (K=8, as in Le et al. 2023, but in AU space
because we have no valence-arousal):
  neighbours: K nearest training windows on the flat 5x70 AU values, excluding i itself and any
  near-copy (distance < 0.05). Two pools:
    any source      all training windows
    other sources   only windows from a DIFFERENT source than i. Sources are different people, so
                    these neighbours cannot be the same recording or the same person (the npz
                    has no recording or participant IDs, so this is the only guaranteed separation)
  weights: exp(-d^2 / delta^2), delta = median 1st-neighbour distance.
  distribution: weighted share of neighbour LABELS (one-hot), divided by the class prior of the
  pool so big classes do not dominate.

Questions
  1. Same-recording echo: do same-source neighbours agree with the label much more than
     other-source neighbours? (If yes, same-source neighbours mostly reflect the same person.)
  2. Where does the off-label mass go? Class -> class matrix.
  3. Does that matrix just repeat D's known confusions? Correlation with D's VALIDATION confusion.
  4. How much would the targets move: off-label mass per window, and share of windows whose
     neighbour distribution puts its maximum on another class.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[2]
Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
E = [str(c) for c in Z["class_names"]]
K, NEAR = 8, 0.05
X = Z["X_train"].reshape(len(Z["X_train"]), -1).astype(np.float64)
Y, S = Z["Y_train"], Z["S_train"]
SRC = {0: "old", 1: "eh", 2: "new"}


def neighbour_dist(pool_mask_fn):
    D = np.zeros((len(X), 7))
    agree, offmass = np.zeros(len(X)), np.zeros(len(X))
    for s in range(3):
        q = np.where(S == s)[0]
        pool = np.where(pool_mask_fn(s))[0]
        nn = NearestNeighbors(n_neighbors=K + 20).fit(X[pool])
        d, j = nn.kneighbors(X[q])
        for r, i in enumerate(q):
            keep = [(dd, pool[jj]) for dd, jj in zip(d[r], j[r]) if pool[jj] != i and dd >= NEAR][:K]
            dist = np.array([k[0] for k in keep]); lab = Y[[k[1] for k in keep]]
            D[i, :] = 0
            np.add.at(D[i], lab, 1.0)  # weights set below
            D[i] = 0
            for dd, l in zip(dist, lab):
                D[i, l] += np.exp(-dd ** 2 / DELTA ** 2)
    return D


# delta = median first-neighbour distance (excluding self / near-copies), any source
nn = NearestNeighbors(n_neighbors=3).fit(X)
d, j = nn.kneighbors(X)
first = np.where(d[:, 1] >= NEAR, d[:, 1], d[:, 2])
DELTA = float(np.median(first))


def finish(D, pool_prior):
    D = D / pool_prior                  # remove class-size effect
    D = D / np.maximum(D.sum(1, keepdims=True), 1e-12)
    return D


rows = {}
prior_all = np.bincount(Y, minlength=7) / len(Y)
D_any = finish(neighbour_dist(lambda s: np.ones(len(X), bool)), prior_all)
D_oth = np.zeros_like(D_any)
raw_oth = neighbour_dist(lambda s: S != s)
for s in range(3):
    m = S == s
    pr = np.bincount(Y[S != s], minlength=7) / (S != s).sum()
    D_oth[m] = finish(raw_oth[m], pr)
raw_same = neighbour_dist(lambda s: S == s)
D_same = np.zeros_like(D_any)
for s in range(3):
    m = S == s
    pr = np.bincount(Y[m], minlength=7) / m.sum()
    D_same[m] = finish(raw_same[m], pr)

print(f"delta (median 1st-neighbour distance) = {DELTA:.3f}, K = {K}\n")
print("1. mass on the window's OWN label (mean), by source of the window")
t = pd.DataFrame({name: [Dm[S == s, :][np.arange((S == s).sum()), Y[S == s]].mean() for s in range(3)]
                  for name, Dm in (("same-source pool", D_same), ("other-source pool", D_oth), ("any pool", D_any))},
                 index=[SRC[s] for s in range(3)])
print(t.round(3).to_string())
print("chance level (uniform over 7) = 0.143")

for name, Dm in (("OTHER-SOURCE pool", D_oth), ("ANY pool", D_any)):
    print(f"\n2. {name}: mean neighbour distribution per true class (rows = label, cols = neighbour mass)")
    M = pd.DataFrame([Dm[Y == c].mean(0) for c in range(7)], index=E, columns=[e[:4] for e in E])
    print(M.round(2).to_string())
    own = Dm[np.arange(len(Y)), Y]
    print(f"   windows whose neighbour maximum is ANOTHER class: {np.mean(Dm.argmax(1) != Y):.0%}"
          f" | off-label mass mean {1 - own.mean():.2f}")
    print("   per source, max on another class:",
          {SRC[s]: f"{np.mean(Dm[S == s].argmax(1) != Y[S == s]):.0%}" for s in range(3)})
    rows[name] = M.values

# 3. compare with D's validation confusion
sess = ort.InferenceSession(str(REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"))
Xv = np.concatenate([Z[f"X_val_{s}"] for s in ("old", "new", "eh")])
Yv = np.concatenate([Z[f"Y_val_{s}"] for s in ("old", "new", "eh")])
pv = sess.run(None, {sess.get_inputs()[0].name: Xv})[0].argmax(1)
C = np.zeros((7, 7))
np.add.at(C, (Yv, pv), 1)
C = C / C.sum(1, keepdims=True)
off = ~np.eye(7, dtype=bool)
print("\n3. does the off-label neighbour mass repeat D's VALIDATION confusions? (off-diagonal cells)")
for name, M in rows.items():
    r = np.corrcoef(M[off], C[off])[0, 1]
    top_n = sorted(zip(M[off], [f"{E[i]}->{E[j]}" for i in range(7) for j in range(7) if i != j]), reverse=True)[:6]
    print(f"   {name:<18} correlation with D's val confusion r = {r:.2f}; top off-label cells:",
          ", ".join(f"{p} {v:.2f}" for v, p in top_n))
top_c = sorted(zip(C[off], [f"{E[i]}->{E[j]}" for i in range(7) for j in range(7) if i != j]), reverse=True)[:6]
print("   D's top validation confusions:", ", ".join(f"{p} {v:.2f}" for v, p in top_c))
