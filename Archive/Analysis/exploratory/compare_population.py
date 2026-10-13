"""Training-built vs held-out-built population centroids, for both teachers. Read-only."""
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))

import numpy as np
import calibration_study as cs
import emojihero_calibration_study as eh

MODELS = {
    "mixed": REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model.onnx",
    "acted_only": REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx",
}
E = eh.EMOTIONS
data = REPO / "Analysis" / "data"

loaded, prov = {}, {}
for split in eh.SPLITS:
    loaded[split], prov[split] = eh.load_split(data, split)
eh.check_corpus(loaded, prov)
heldout = loaded["validation"] + loaded["test"]


def seq_predict(M, logf):
    """Population-only rule: cosine nearest centroid per window, then majority vote."""
    n, w, k = logf.shape
    win = cs.classify(M, list(range(k)), logf.reshape(n * w, k), "cosine")[0]
    return eh.fast_majority_vote(np.asarray(win).reshape(n, w))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def softmax(v):
    z = np.exp(v - v.max())
    return z / z.sum()


for name, path in MODELS.items():
    print("=" * 78)
    print(f"MODEL {name}")
    print("=" * 78)
    scores, logf = {}, {}
    for split in eh.SPLITS:
        scores[split] = eh.run_model(loaded[split], path)
        logf[split] = cs.logscore(scores[split], cs.DEFAULT_EPSILON)

    # 1. How differently the model behaves on people it trained on.
    print("\n1. model behaviour by split (window level)")
    print(f"   {'split':<11} {'raw acc':>8} {'p(true)':>8} {'top score':>10}")
    for split in eh.SPLITS:
        s = scores[split].reshape(-1, 7)
        y = np.repeat([q.emotion for q in loaded[split]], eh.WINDOWS_PER_SEQUENCE)
        print(f"   {split:<11} {np.mean(s.argmax(1) == y):>8.3f} "
              f"{s[np.arange(len(y)), y].mean():>8.3f} {s.max(1).mean():>10.3f}")

    # 2. The two population profiles side by side.
    idx_tr = {s.sequence_id: i for i, s in enumerate(loaded["training"])}
    G_tr = eh.build_population_centroids(loaded["training"], logf["training"], idx_tr)["matrix"]
    lf_ho = np.concatenate([logf["validation"], logf["test"]])
    sc_ho = np.concatenate([scores["validation"], scores["test"]])
    idx_ho = {s.sequence_id: i for i, s in enumerate(heldout)}
    G_ho = eh.build_population_centroids(heldout, lf_ho, idx_ho)["matrix"]

    print("\n2. population centroid per emotion: training-built vs held-out-built")
    print("   'own-class share' = how much of the centroid's probability sits on its own emotion")
    print(f"   {'emotion':<10} {'cosine':>7}   {'own share (train)':>17} {'own share (held-out)':>20}")
    for e in range(7):
        print(f"   {E[e]:<10} {cos(G_tr[e], G_ho[e]):>7.4f}   "
              f"{softmax(G_tr[e])[e]:>17.3f} {softmax(G_ho[e])[e]:>20.3f}")

    # 3. Does it change predictions? Population-only rule on each held-out person,
    #    training population vs a population of the OTHER held-out people.
    print("\n3. population-only method on held-out people (sequence level, majority vote)")
    participants = sorted({(sp, s.participant) for sp in ("validation", "test")
                           for s in loaded[sp]})
    rows = []
    for sp, p in participants:
        mine = [i for i, s in enumerate(heldout) if s.participant == p and
                (i < len(loaded["validation"])) == (sp == "validation")]
        others = [i for i in range(len(heldout)) if i not in set(mine)]
        o_seqs = [heldout[i] for i in others]
        G_lopo = eh.build_population_centroids(
            o_seqs, lf_ho[others], {s.sequence_id: j for j, s in enumerate(o_seqs)})["matrix"]
        y = np.array([heldout[i].emotion for i in mine])
        F = lf_ho[mine]
        raw = eh.fast_majority_vote(sc_ho[mine].argmax(2))
        a_tr, a_lo = seq_predict(G_tr, F), seq_predict(G_lopo, F)
        rows.append((sp, p, len(mine),
                     eh.fast_macro_f1(y, raw), eh.fast_macro_f1(y, a_tr), eh.fast_macro_f1(y, a_lo)))

    print(f"   {'split':<11}{'person':>7}{'seqs':>6}{'raw':>8}{'pop(train)':>12}{'pop(held-out)':>15}")
    for sp, p, n, r, t, l in rows:
        print(f"   {sp:<11}{p:>7}{n:>6}{r:>8.3f}{t:>12.3f}{l:>15.3f}")
    r, t, l = (np.array([row[i] for row in rows]) for i in (3, 4, 5))
    print(f"   {'mean':<24}{r.mean():>8.3f}{t.mean():>12.3f}{l.mean():>15.3f}")
    d = l - t
    print(f"   held-out minus train: mean {d.mean():+.3f}; held-out better for "
          f"{(d > 0).sum()}, worse for {(d < 0).sum()}, tied {(d == 0).sum()} of {len(d)}")
    print()
