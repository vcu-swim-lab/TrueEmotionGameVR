"""Acted-only teacher with a masked-AU reconstruction task on unlabelled windows. Arms A, B, C.

  A  old + new acted, hard labels (EmojiHero removed). No auxiliary task.
  B  A + masked-AU reconstruction on NATURAL windows of train participants (unlabelled).
  C  A + the same reconstruction on ACTED windows of train participants (ungated, labels unused).
  D  deployed model, reference only (it also has EmojiHero: removing EmojiHero is a separate change).

Pools: Analysis/data/aux_pools.npz (natural_windows_build.py). Supervised data: the npz windows
with S_train in {old, new}; sample weights = D's class x source weights for those windows, rescaled
to mean 1.

One optimizer update (all arms, same code path, SGD 1e-3):
    sup batch (10 windows, the same epoch order for every arm with the same repeat number)
      -> model(x_sup, training=True)        # unmasked inputs; dropout + BN batch statistics
      L_cls = sum_i w_i * CE(y_i, p_i) / 10 # = Keras sparse CE with sample weights
      L_gate = load-balance + entropy terms of THIS pass only (the model's existing regularisers)
    aux batch (10 windows, B and C only)
      -> rec_model(x_masked, training=False)  # no dropout, BN uses running statistics and does
                                              # NOT update them; gradients still flow to weights
      L_rec = Huber(pred - target) over MASKED target entries only (mean over those entries)
    loss = L_cls + L_gate + beta * L_rec      (A: beta term absent)
Same number of updates per epoch (ceil(2397/10) = 240) in every arm; aux sampling uses its own
random generator, so supervised batches and dropout draws are identical across arms.

Reconstruction (B and C identical):
  targets   channel groups (L/R and quadrant channels of one muscle grouped). Gaze (EyesLook*) and
            blink (EyesClosed*) are never targets (design: they reflect looking and blinking, not
            expression). Channels with spread < 0.02 in the TRAINING pools are not targets.
  stats     mean and spread (floor 0.02) per channel from the union of both training pools.
  mask      per window, 25% of target groups (at least 1), hidden in all 5 frames, filled with
            the channel's training mean. Every other channel (other targets, gaze, blink, tongue,
            non-targets) stays visible.
  target    (x - mean) / spread of the hidden channels; Huber delta 1.
  head      Dense(256, relu) -> Dense(5 * n_targets) on the 'fuse' output (training only).
  sampling  participant uniform -> recording uniform -> window uniform.
After training (best validation epoch restored), every arm's BatchNorm statistics are re-estimated
exactly on the acted supervised training windows, layer by layer.
Selection: early stopping and beta use VALIDATION ONLY (mean of old and new acted macro-F1).
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
sys.path.insert(0, str(REPO / "Analysis/exploratory"))
parser = argparse.ArgumentParser()
parser.add_argument("--runs", default="A:0,B:0.1,B:0.3,B:1.0,C:0.1,C:0.3,C:1.0")
parser.add_argument("--repeats", default="42,1,7")
parser.add_argument("--out", default=str(REPO / "Analysis/results/masked_au_aux.csv"))
parser.add_argument("--check", action="store_true", help="only run the input/mask checks")
parser.add_argument("--save-dir", default="", help="save each run's classifier weights here (pilot)")
parser.add_argument("--no-log", action="store_true", help="skip gradient logging (equality check)")
args = parser.parse_args()

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix

E = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
ANG, NEU = 0, 4
Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
AU = [str(a) for a in Z["au_names"]]
POOLS = np.load(REPO / "Analysis/data/aux_pools.npz")

# ---------------------------------------------------------------- supervised data
sup = np.isin(Z["S_train"], [0, 2])
X_SUP = Z["X_train"][sup].astype(np.float32)
Y_SUP = Z["Y_train"][sup].astype(np.int64)
W_SUP = Z["W_train"][sup].astype(np.float32)
W_SUP = W_SUP / W_SUP.mean()
VAL = {s: (Z[f"X_val_{s}"], Z[f"Y_val_{s}"]) for s in ("old", "new")}
TEST = {s: (Z[f"X_test_{s}"], Z[f"Y_test_{s}"]) for s in ("old", "new", "eh")}
BATCH = 10

# ---------------------------------------------------------------- reconstruction targets
def groups():
    g = {}
    for i, a in enumerate(AU):
        if a.startswith(("EyesLook", "EyesClosed")):
            continue
        base = a
        if a[-1] in "LR" and (a[:-1] + {"L": "R", "R": "L"}[a[-1]]) in AU:
            base = a[:-1]
        for q in ("LB", "LT", "RB", "RT"):
            if a.startswith(("LipFunneler", "LipSuck")) and a.endswith(q):
                base = a[:-2]
        if a.startswith("ChinRaiser"):
            base = "ChinRaiser"
        g.setdefault(base, []).append(i)
    return g


frames = np.concatenate([POOLS["X_natural"], POOLS["X_acted"]]).reshape(-1, 70)
MEAN, SPREAD_RAW = frames.mean(0), frames.std(0)
SPREAD = np.maximum(SPREAD_RAW, 0.02)
G = {k: [i for i in v if SPREAD_RAW[i] >= 0.02] for k, v in groups().items()}
G = {k: v for k, v in G.items() if v}
TARGETS = sorted(i for v in G.values() for i in v)
T_POS = {c: j for j, c in enumerate(TARGETS)}
GROUP_LIST = list(G.values())
N_MASK = max(1, round(0.25 * len(GROUP_LIST)))


class AuxSampler:
    def __init__(self, pool, seed):
        self.X = POOLS[f"X_{pool}"]
        P, R = POOLS[f"participant_{pool}"], POOLS[f"recording_{pool}"]
        self.index = {}
        for i, (p, r) in enumerate(zip(P, R)):
            self.index.setdefault(p, {}).setdefault(r, []).append(i)
        self.parts = sorted(self.index)
        self.rng = np.random.default_rng(seed + 1000)

    def batch(self, n=BATCH):
        idx = []
        for _ in range(n):
            p = self.parts[self.rng.integers(len(self.parts))]
            recs = sorted(self.index[p])
            r = recs[self.rng.integers(len(recs))]
            w = self.index[p][r]
            idx.append(w[self.rng.integers(len(w))])
        X = self.X[idx].copy()
        target = ((X[..., TARGETS] - MEAN[TARGETS]) / SPREAD[TARGETS]).astype(np.float32)
        tmask = np.zeros_like(target)
        for b in range(n):
            for gi in self.rng.choice(len(GROUP_LIST), N_MASK, replace=False):
                ch = GROUP_LIST[gi]
                X[b][:, ch] = MEAN[ch]
                tmask[b][:, [T_POS[c] for c in ch]] = 1.0
        return X.astype(np.float32), target, tmask


# ---------------------------------------------------------------- model
nb = json.load(open(REPO / "notebooks/teacher_notebook_only_acted_data_volume_augmentation.ipynb",
                    encoding="utf-8"))["cells"]
arch = "".join(nb[74]["source"])
assert "def build_moe_single_input" in arch
exec(arch)


def build(seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model, _ = build_moe_single_input(input_shape=(5, 70), num_classes=7, tau_init=2.0, tau_final=0.7,
                                      gate_hidden=128, classifier_hidden=256, lb_coef=0.02, ent_coef=0.01)
    fused = model.get_layer("fuse").output
    h = tf.keras.layers.Dense(256, activation="relu", name="rec_d1")(fused)
    h = tf.keras.layers.Dense(5 * len(TARGETS), name="rec_out")(h)
    rec = tf.keras.layers.Reshape((5, len(TARGETS)), name="rec")(h)
    rec_model = tf.keras.Model(model.input, rec)
    return model, rec_model


def macro(y, p):
    return f1_score(y, p, labels=range(7), average="macro", zero_division=0)


def predict(model, X):
    return model.predict(X.astype(np.float32), batch_size=512, verbose=0)


def val_score(model):
    s = {k: macro(y, predict(model, X).argmax(1)) for k, (X, y) in VAL.items()}
    return (s["old"] + s["new"]) / 2, s


def reestimate_bn(model):
    """Exact BatchNorm statistics on the acted supervised training windows, layer by layer."""
    bns = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    for bn in bns:
        sub = tf.keras.Model(model.input, bn.input)
        a = sub.predict(X_SUP, batch_size=512, verbose=0)
        axes = tuple(range(a.ndim - 1))
        bn.moving_mean.assign(a.mean(axis=axes))
        bn.moving_variance.assign(a.var(axis=axes))
    return len(bns)


def train(arm, beta, seed):
    model, rec_model = build(seed)
    gate = model.get_layer("gate")
    opt = tf.keras.optimizers.SGD(1e-3)
    huber = tf.keras.losses.Huber(delta=1.0, reduction="none")
    sampler = AuxSampler("natural" if arm == "B" else "acted", seed) if arm in ("B", "C") else None
    use_aux = sampler is not None
    beta_t = tf.constant(float(beta))

    @tf.function
    def step(x, y, w, xa, ta, ma):
        with tf.GradientTape() as tape:
            p = model(x, training=True)
            ce = tf.keras.losses.sparse_categorical_crossentropy(y, p)
            l_cls = tf.reduce_sum(w * ce) / tf.cast(tf.shape(x)[0], tf.float32)
            l_gate = tf.add_n(model.losses) if model.losses else 0.0
            loss = l_cls + l_gate
            l_rec = tf.constant(0.0)
            if use_aux:
                r = rec_model(xa, training=False)
                hl = huber(ta[..., None], r[..., None])            # (B, 5, n_targets)
                l_rec = tf.reduce_sum(hl * ma) / tf.maximum(tf.reduce_sum(ma), 1.0)
                loss = loss + beta_t * l_rec
        grads = tape.gradient(loss, model.trainable_variables + (rec_model.trainable_variables if use_aux else []))
        varis = model.trainable_variables + (rec_model.trainable_variables if use_aux else [])
        pairs = [(g, v) for g, v in zip(grads, varis) if g is not None]
        seen, uniq = set(), []
        for g, v in pairs:                       # rec_model shares the encoder variables
            if id(v) not in seen:
                seen.add(id(v)); uniq.append((g, v))
        opt.apply_gradients(uniq)
        return l_cls, l_rec

    # pull log: beta * |grad L_rec| / |grad (L_cls + L_gate)| on the SHARED encoder variables, median
    # over 8 fixed probe batch pairs drawn with their own generators (training draws are untouched)
    shared = [v for v in model.trainable_variables
              if not v.path.startswith(("y/", "clf_dense/", "rec_d1/", "rec_out/"))]
    pull_log, trivial = [], np.nan
    # probes for every arm (A logs only its classification gradient); own generators, so the
    # training draws are untouched
    probe_rng = np.random.default_rng(999)
    probe_sampler = AuxSampler("acted" if arm == "C" else "natural", 5000)
    probes = [(probe_rng.choice(len(X_SUP), BATCH, replace=False), probe_sampler.batch()) for _ in range(8)]
    if use_aux:
        trivial = float(np.mean([np.sum(huber(t[..., None], np.zeros_like(t)[..., None]).numpy() * mk) / mk.sum()
                                 for _, (_, t, mk) in probes]))

    def measure_pull(epoch):
        """Shared-encoder gradients, median over the 8 probe pairs (L2 norms over all shared variables):
          g_cls      = |grad(L_cls + L_gate)|
          beta*g_rec = |beta * grad L_rec|
          g_total    = |grad(L_cls + L_gate) + beta * grad L_rec|   (the size of the real update direction)
          ratio      = (beta*g_rec) / g_cls
          cosine     = cos(grad(L_cls + L_gate), beta * grad L_rec)  (alignment; 0 = orthogonal)
        Norms do not split into percentage shares, so none is reported.
        Every non-trainable variable (BatchNorm moving statistics, dropout seed-generator states) is
        saved before the probes and restored after, so logging does not change training."""
        if args.no_log:
            return
        saved = [v.numpy() for v in model.non_trainable_variables]
        rows = []
        for i, (xa, ta, ma) in probes:
            with tf.GradientTape(persistent=True) as tape:
                p = model(X_SUP[i], training=True)
                ce = tf.keras.losses.sparse_categorical_crossentropy(Y_SUP[i], p)
                l_sup = tf.reduce_sum(W_SUP[i] * ce) / BATCH + (tf.add_n(model.losses) if model.losses else 0.0)
                r = rec_model(xa, training=False)
                l_rec = tf.reduce_sum(huber(ta[..., None], r[..., None]) * ma) / tf.reduce_sum(ma)
            g1 = [g if g is not None else tf.zeros_like(v) for g, v in zip(tape.gradient(l_sup, shared), shared)]
            n1 = float(tf.linalg.global_norm(g1))
            if not use_aux:
                rows.append((n1, 0.0, n1, 0.0, 0.0)); continue
            g2 = [float(beta) * (g if g is not None else tf.zeros_like(v)) for g, v in zip(tape.gradient(l_rec, shared), shared)]
            n2 = float(tf.linalg.global_norm(g2))
            nt = float(tf.linalg.global_norm([a + b for a, b in zip(g1, g2)]))
            dot = float(sum(tf.reduce_sum(a * b) for a, b in zip(g1, g2)))
            rows.append((n1, n2, nt, n2 / n1, dot / max(n1 * n2, 1e-12)))
        for v, val in zip(model.non_trainable_variables, saved):
            v.assign(val)
        med = np.median(np.array(rows), axis=0)
        pull_log.append({"epoch": epoch, "g_cls": round(float(med[0]), 3), "beta*g_rec": round(float(med[1]), 3),
                         "g_total": round(float(med[2]), 3), "ratio": round(float(med[3]), 3),
                         "cosine": round(float(med[4]), 3)})

    measure_pull(0)
    order_rng = np.random.default_rng(seed)
    dummy = (np.zeros((BATCH, 5, 70), np.float32), np.zeros((BATCH, 5, len(TARGETS)), np.float32),
             np.zeros((BATCH, 5, len(TARGETS)), np.float32))
    best, best_w, best_ep, wait, hist = -1, None, 0, 0, []
    for ep in range(30):
        gate.tau.assign(2.0 + min(1.0, ep / 20) * (0.7 - 2.0))
        order = order_rng.permutation(len(X_SUP))
        lc, lr = [], []
        for b in range(0, len(order), BATCH):
            i = order[b:b + BATCH]
            aux = sampler.batch() if use_aux else dummy
            c, r = step(X_SUP[i], Y_SUP[i], W_SUP[i], *aux)
            lc.append(float(c)); lr.append(float(r))
        v, _ = val_score(model)
        hist.append((ep + 1, np.mean(lc), np.mean(lr), v))
        if ep + 1 in (1, 5, 10, 15, 20, 25, 30):
            measure_pull(ep + 1)
        if v > best:
            best, best_w, best_ep, wait = v, model.get_weights(), ep + 1, 0
        else:
            wait += 1
            if wait >= 8:
                break
    model.set_weights(best_w)
    gate.tau.assign(0.7)
    n_bn = reestimate_bn(model)
    return model, best_ep, hist, n_bn, pull_log, trivial


# ---------------------------------------------------------------- evaluation
def counts(y, p):
    emo = y != NEU
    return {"emotion->Neutral": int((emo & (p == NEU)).sum()),
            "Neutral->Anger": int(((y == NEU) & (p == ANG)).sum()),
            "other emotion->Anger": int((emo & (y != ANG) & (p == ANG)).sum())}


HEADSET = None


def headset():
    global HEADSET
    if HEADSET is None:
        import calibration_study as cs
        import session_analysis as sa
        from jitter_headset_replay import sessions
        D = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
        HEADSET = []
        for d in sessions():
            data = cs.replay(sa.load_session(d / "predictions.csv"), D)
            _, te = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
            HEADSET.append((d.name[-6:], data.windows[te].astype(np.float32), data.prompted[te]))
    return HEADSET


def evaluate(pred_fn, row):
    pv, yv = [], []
    for s, (X, y) in TEST.items():
        p = pred_fn(X).argmax(1)
        row[f"test {s}"] = macro(y, p)
        f1c = f1_score(y, p, labels=range(7), average=None, zero_division=0)
        for i, e in enumerate(E):
            row[f"{s} F1 {e}"] = f1c[i]
        for k, v in counts(y, p).items():
            row[f"{s} {k}"] = v
        row[f"{s} confusion"] = json.dumps(confusion_matrix(y, p, labels=range(7)).tolist())
        if s != "eh":
            pv.append(p); yv.append(y)
    p, y = np.concatenate(pv), np.concatenate(yv)
    row["test VR pooled"] = macro(y, p)
    for i, e in enumerate(E):
        row[f"VR F1 {e}"] = f1_score(y, p, labels=range(7), average=None, zero_division=0)[i]
    for k, v in counts(y, p).items():
        row[f"VR {k}"] = v
    row["VR confusion"] = json.dumps(confusion_matrix(y, p, labels=range(7)).tolist())
    import session_analysis as sa
    dev_f1, hp, hy = [], {"dev": [], "other": []}, {"dev": [], "other": []}
    for name, X, y in headset():
        p = pred_fn(X).argmax(1)
        f = sa.macro_f1([E[i] for i in y], [E[i] for i in p], E)["macro_f1"]
        who = "other" if name == "26908f" else "dev"
        if who == "dev":
            dev_f1.append(f)
        else:
            row["headset 26908f tick F1"] = f
        hp[who].append(p); hy[who].append(y)
        row[f"hs {name}"] = f
    row["headset developer tick F1 (mean of 9)"] = float(np.mean(dev_f1))
    for who in ("dev", "other"):
        p, y = np.concatenate(hp[who]), np.concatenate(hy[who])
        row[f"hs {who} Anger acc"] = float(np.mean(p[y == ANG] == ANG))
        for k, v in counts(y, p).items():
            row[f"hs {who} {k}"] = v
    return row


def checks():
    s = AuxSampler("natural", 42)
    X, t, m = s.batch()
    print(f"targets: {len(TARGETS)} channels in {len(GROUP_LIST)} groups; masked per window: {N_MASK} groups")
    print("target channels:", [AU[i] for i in TARGETS])
    print("never targets (gaze/blink by design, or spread < 0.02 in training pools):",
          [AU[i] for i in range(70) if i not in TARGETS])
    # masked entries carry the fill value, unmasked entries are untouched, loss mask only on hidden targets
    ok_fill = all(np.allclose(X[b][:, [TARGETS[j] for j in np.where(m[b, 0] > 0)[0]]],
                              MEAN[[TARGETS[j] for j in np.where(m[b, 0] > 0)[0]]]) for b in range(len(X)))
    print("masked inputs equal the training mean:", ok_fill)
    # unmasked channels equal the pool window: re-draw with a fresh sampler of the same seed and compare
    s2 = AuxSampler("natural", 42)
    p = s2.parts[s2.rng.integers(len(s2.parts))]; recs = sorted(s2.index[p]); r = recs[s2.rng.integers(len(recs))]
    w = s2.index[p][r][s2.rng.integers(len(s2.index[p][r]))]
    hidden = [TARGETS[j] for j in np.where(m[0, 0] > 0)[0]]
    vis = [c for c in range(70) if c not in hidden]
    print("first window: visible channels unchanged:", bool(np.array_equal(X[0][:, vis], s2.X[w][:, vis])),
          "| hidden channels:", [AU[c] for c in hidden])
    print("loss-mask entries per window (targets x 5 frames):", m.sum((1, 2)).tolist())
    print("loss mask is constant over the 5 frames:", bool((m == m[:, :1]).all()))
    print("supervised inputs are the npz windows unchanged:", bool(np.shares_memory(X_SUP, X_SUP) and
          np.array_equal(X_SUP, Z['X_train'][sup])))
    print(f"supervised windows {len(X_SUP)} (old {int((Z['S_train'] == 0).sum())}, new {int((Z['S_train'] == 2).sum())}),"
          f" updates per epoch {int(np.ceil(len(X_SUP) / BATCH))}")
    for pool in ("natural", "acted"):
        sp = AuxSampler(pool, 0)
        draws = [sp.batch()[0] for _ in range(1)]
        P = POOLS[f"participant_{pool}"]
        idx = []
        for _ in range(5000):
            p = sp.parts[sp.rng.integers(len(sp.parts))]
            idx.append(p)
        print(f"{pool}: participants {len(sp.parts)}, recordings {len(set(POOLS[f'recording_{pool}']))}, "
              f"windows {len(P)}; sampled share per participant min/max "
              f"{pd.Series(idx).value_counts(normalize=True).min():.3f}/{pd.Series(idx).value_counts(normalize=True).max():.3f}")


def main():
    checks()
    if args.check:
        return
    rows = []
    out = Path(args.out)
    if out.exists():
        rows = pd.read_csv(out).to_dict("records")
    done = {(r["arm"], float(r["beta"]), int(r["repeat"])) for r in rows}
    if not any(r["arm"] == "D" for r in rows):
        import onnxruntime as ort
        sess = ort.InferenceSession(str(REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"))
        f = lambda X: sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})[0]
        r = {"arm": "D", "beta": 0.0, "repeat": 0, "best epoch": np.nan}
        s = {k: macro(y, f(X).argmax(1)) for k, (X, y) in VAL.items()}
        r.update({"val old": s["old"], "val new": s["new"], "val mean": (s["old"] + s["new"]) / 2})
        rows.append(evaluate(f, r))
        pd.DataFrame(rows).to_csv(out, index=False)
    for spec in args.runs.split(","):
        arm, beta = spec.split(":")
        for seed in (int(x) for x in args.repeats.split(",")):
            if (arm, float(beta), seed) in done:
                continue
            t = time.time()
            model, ep, hist, n_bn, pull_log, trivial = train(arm, float(beta), seed)
            v, s = val_score(model)
            r = {"arm": arm, "beta": float(beta), "repeat": seed, "best epoch": ep,
                 "val old": s["old"], "val new": s["new"], "val mean": v,
                 "final L_rec": hist[ep - 1][2], "trivial L_rec": trivial, "pull log": json.dumps(pull_log),
                 "train L_cls by epoch": json.dumps([round(h[1], 4) for h in hist]),
                 "minutes": (time.time() - t) / 60}
            if args.save_dir:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                model.save_weights(str(Path(args.save_dir) / f"{arm}_{beta}_{seed}.weights.h5"))
            rows.append(evaluate(lambda X: predict(model, X), r))
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{arm} beta={beta} r{seed}: best epoch {ep}, val {v:.3f}, test VR {r['test VR pooled']:.3f}, "
                  f"L_rec {r['final L_rec']:.3f} (trivial {trivial:.3f}), {r['minutes']:.1f} min "
                  f"(BN layers re-estimated: {n_bn})", flush=True)
            for row_ in pull_log:
                print("   ", row_, flush=True)


if __name__ == "__main__":
    main()
