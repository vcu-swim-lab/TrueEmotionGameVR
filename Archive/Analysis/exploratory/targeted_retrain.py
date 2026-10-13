"""Targeted retrains from the test error analysis: gaze off, Fear weighting, left/right mirror.

Data, architecture and training are exactly as in aug_amount_strength.py (the notebook's MoE-TCN,
SGD 1e-3, batch 10, <= 30 epochs, early stopping on domain-balanced validation macro-F1, gate tau
0.7 at evaluation). Each variant is trained REPEATS times and the scores are averaged.

Variants
  base      D's data as it is (a D twin trained here).
  gaze_off  the 8 EyesLook AUs set to 0 in train, validation, test and headset windows.
  fear_w    EmojiHero Fear training windows with no Fear cue (peak of BrowLowerer, LidTightener and
            LipStretcher, mean of L/R, all <= 0.15) get weight x0.3. The other Fear windows are
            scaled up so the Fear class keeps its total weight.
  mirror    every epoch shows every window and its left/right mirror (L<->R AUs, gaze left<->right,
            JawSideways and Mouth left<->right). Labels and weights unchanged.

Scores: test macro-F1 (old acted, new acted, VR combined, EmojiHero), false Neutral, Fear F1, and
headset replay of the evaluation block, raw argmax (tick and trial macro-F1, 10 sessions).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
sys.path.insert(0, str(REPO / "Analysis/exploratory"))
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--variants", default="base,gaze_off,fear_w,mirror")
parser.add_argument("--repeats", default="42,1,7")
args = parser.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import callbacks

import calibration_study as cs
import session_analysis as sa
from jitter_headset_replay import sessions

Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
AU = [str(a) for a in Z["au_names"]]
E = [str(c) for c in Z["class_names"]]
NEU, FEAR = E.index("Neutral"), E.index("Fear")
LABELS = np.arange(7)
GAZE = np.array([a.startswith("EyesLook") for a in AU])


# ---------------------------------------------------------------- mirror permutation
def mirror_index():
    def partner(a):
        for x, y in (("EyesLookLeft", "EyesLookRight"), ("EyesLookRight", "EyesLookLeft")):
            if a.startswith(x):                      # look left with the left eye -> look right with the right eye
                return y + {"L": "R", "R": "L"}[a[-1]]
        for x, y in (("JawSidewaysLeft", "JawSidewaysRight"), ("JawSidewaysRight", "JawSidewaysLeft"),
                     ("MouthLeft", "MouthRight"), ("MouthRight", "MouthLeft")):
            if a == x:
                return y
        for x, y in (("LB", "RB"), ("RB", "LB"), ("LT", "RT"), ("RT", "LT")):
            if a.startswith(("LipFunneler", "LipSuck")) and a.endswith(x):
                return a[:-2] + y
        if a[-1] in "LR" and a[:-1] + {"L": "R", "R": "L"}[a[-1]] in AU:
            return a[:-1] + {"L": "R", "R": "L"}[a[-1]]
        return a
    idx = np.array([AU.index(partner(a)) for a in AU])
    assert sorted(idx) == list(range(70)) and (idx[idx] == np.arange(70)).all()
    return idx


MIRROR = mirror_index()

# ---------------------------------------------------------------- architecture (notebook cell 74)
nb = json.load(open(REPO / "notebooks/teacher_notebook_only_acted_data_volume_augmentation.ipynb",
                    encoding="utf-8"))["cells"]
arch = "".join(nb[74]["source"])
assert "def build_moe_single_input" in arch
exec(arch)


class Epochs(tf.keras.utils.PyDataset):
    """Every window once per epoch, plus (mirror) its mirrored copy. Reshuffled every epoch."""

    def __init__(self, X, y, w, mirror, seed, batch_size=10, **kw):
        super().__init__(**kw)
        self.X, self.y, self.w = X.astype(np.float32), y, w.astype(np.float32)
        self.mirror, self.bs = mirror, batch_size
        self.rng = np.random.default_rng(seed)
        self._new_epoch()

    def _new_epoch(self):
        n = len(self.X)
        src = np.arange(n)
        flip = np.zeros(n, bool)
        if self.mirror:
            src, flip = np.concatenate([src, src]), np.concatenate([flip, ~flip])
        o = self.rng.permutation(len(src))
        self.src, self.flip = src[o], flip[o]

    def __len__(self):
        return math.ceil(len(self.src) / self.bs)

    def __getitem__(self, i):
        b = slice(i * self.bs, (i + 1) * self.bs)
        rows, flip = self.src[b], self.flip[b]
        X = self.X[rows].copy()
        X[flip] = X[flip][..., MIRROR]
        return X, self.y[rows], self.w[rows]

    def on_epoch_end(self):
        self._new_epoch()


def prep(X, variant):
    X = X.astype(np.float32).copy()
    if variant == "gaze_off":
        X[..., GAZE] = 0
    return X


def fear_weights(X, y, s, w):
    cue = np.stack([X[..., [AU.index(n + "L"), AU.index(n + "R")]].mean(-1).max(-1)
                    for n in ("BrowLowerer", "LidTightener", "LipStretcher")], 1).max(1)
    w = w.copy()
    fear = y == FEAR
    weak = fear & (s == 1) & (cue <= 0.15)                  # S_train: 0 old, 1 EmojiHero, 2 new
    total = w[fear].sum()
    w[weak] *= 0.3
    w[fear] *= total / w[fear].sum()
    print(f"  fear_w: {weak.sum()} EmojiHero Fear windows down-weighted of {(fear & (s == 1)).sum()}", flush=True)
    return w


VAL = {k: (Z[f"X_val_{k}"], Z[f"Y_val_{k}"]) for k in ("old", "new", "eh")}
TEST = {k: (Z[f"X_test_{k}"], Z[f"Y_test_{k}"]) for k in ("old", "new", "eh")}
mf1 = lambda y, p: f1_score(y, p, labels=LABELS, average="macro", zero_division=0)

# headset evaluation windows, once (they do not depend on the model)
HEADSET = []
D_ONNX = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
for d in sessions():
    data = cs.replay(sa.load_session(d / "predictions.csv"), D_ONNX)
    _, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
    HEADSET.append((d.name[8:], data.windows[test].astype(np.float32), data.prompted[test],
                    data.trial[test], data.tick[test]))
print(f"{len(HEADSET)} headset sessions loaded", flush=True)


class ValF1(callbacks.Callback):
    def __init__(self, variant):
        super().__init__()
        self.val = {k: (prep(X, variant), y) for k, (X, y) in VAL.items()}

    def on_epoch_end(self, epoch, logs=None):
        s = {k: mf1(y, self.model.predict(X, batch_size=256, verbose=0).argmax(1)) for k, (X, y) in self.val.items()}
        logs["val_domain_balanced_macro_f1"] = float(np.mean(list(s.values())))


def train(variant, seed):
    tf.keras.backend.clear_session()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model, anneal = build_moe_single_input(
        input_shape=(5, 70), num_classes=7, tau_init=2.0, tau_final=0.7,
        gate_hidden=128, classifier_hidden=256, lb_coef=0.02, ent_coef=0.01)
    X, y, w = prep(Z["X_train"], variant), Z["Y_train"], Z["W_train"]
    if variant == "fear_w":
        w = fear_weights(Z["X_train"], y, Z["S_train"], w)
    data = Epochs(X, y, w, mirror=(variant == "mirror"), seed=seed)
    Xv = np.concatenate([prep(v[0], variant) for v in VAL.values()])
    Yv = np.concatenate([v[1] for v in VAL.values()])
    t = time.time()
    model.fit(data, validation_data=(Xv, Yv), epochs=30, verbose=0,
              callbacks=[ValF1(variant),
                         callbacks.EarlyStopping(monitor="val_domain_balanced_macro_f1", mode="max",
                                                 patience=8, restore_best_weights=True),
                         anneal])
    model.save(str(OUT / f"{variant}_r{seed}.keras"))
    print(f"  {variant} r{seed}: trained in {(time.time() - t) / 60:.1f} min", flush=True)
    model.get_layer("gate").tau.assign(0.7)
    return model


def evaluate(model, variant, seed):
    row = {"variant": variant, "repeat": seed}
    preds = {k: model.predict(prep(X, variant), batch_size=256, verbose=0).argmax(1) for k, (X, _) in TEST.items()}
    for k, (_, y) in TEST.items():
        row[f"test {k}"] = mf1(y, preds[k])
    yv = np.concatenate([TEST["old"][1], TEST["new"][1]]); pv = np.concatenate([preds["old"], preds["new"]])
    row["test VR"] = mf1(yv, pv)
    row["VR false Neutral"] = int(((yv != NEU) & (pv == NEU)).sum())
    ya = np.concatenate([t[1] for t in TEST.values()]); pa = np.concatenate(list(preds.values()))
    f1e = f1_score(ya, pa, labels=LABELS, average=None, zero_division=0)
    for i, e in enumerate(E):
        row[f"test F1 {e}"] = f1e[i]
    f1e_vr = f1_score(yv, pv, labels=LABELS, average=None, zero_division=0)
    row["VR F1 Fear"] = f1e_vr[FEAR]
    tick, trial, per = [], [], {e: [] for e in E}
    for name, X, y, tr, tk in HEADSET:
        p = model.predict(prep(X, variant), batch_size=256, verbose=0).argmax(1)
        yt, pt = np.array([E[i] for i in y]), np.array([E[i] for i in p])
        tick.append(sa.macro_f1(yt, pt, E)["macro_f1"])
        _, ty, tp = cs.trial_vote(yt, pt, tr, tk, E)
        trial.append(sa.macro_f1(ty, tp, E)["macro_f1"])
        for i, e in enumerate(E):
            if (y == i).any():
                per[e].append(np.mean(p[y == i] == i))
        row[f"hs {name}"] = tick[-1]
    row["headset tick"], row["headset trial"] = np.mean(tick), np.mean(trial)
    for e in E:
        row[f"hs acc {e}"] = np.mean(per[e])
    return row


rows = []
for variant in args.variants.split(","):
    for seed in (int(s) for s in args.repeats.split(",")):
        print(f"\n=== {variant} r{seed}", flush=True)
        rows.append(evaluate(train(variant, seed), variant, seed))
        pd.DataFrame(rows).to_csv(REPO / "Analysis/results/targeted_retrain.csv", index=False)

df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
main_cols = ["test old", "test new", "test VR", "test eh", "VR false Neutral", "VR F1 Fear", "test F1 Fear",
             "headset tick", "headset trial"]
print("\nMEAN over repeats")
print(df.groupby("variant")[main_cols].mean().round(3).to_string())
print("\nSPREAD (max - min) over repeats")
print(df.groupby("variant")[main_cols].agg(lambda s: s.max() - s.min()).round(3).to_string())
print("\nheadset raw accuracy per emotion (mean)")
print(df.groupby("variant")[[f"hs acc {e}" for e in E]].mean().round(2).to_string())
print("\ntest F1 per emotion, all sources pooled (mean)")
print(df.groupby("variant")[[f"test F1 {e}" for e in E]].mean().round(3).to_string())
