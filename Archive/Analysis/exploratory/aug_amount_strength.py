"""Volume augmentation: how many copies (step A) and how strong (step B). Trained offline.

Data: the exact arrays of the deployed teacher D, exported from Colab
(notebooks/teacher_arrays_with_emojihero.npz: old acted + EmojiHero + new acted, 8,181 training
windows, D's class x source sample weights, D's validation and test windows).

Model and training: exactly the notebook's. The architecture is executed from cell 74 of
notebooks/teacher_notebook_only_acted_data_volume_augmentation.ipynb. SGD 1e-3, batch 10, at most
30 epochs, gate temperature annealed 2.0 -> 0.7, early stopping (patience 8, best weights restored)
on the domain-balanced validation macro-F1 of old acted, new acted and EmojiHero. Seed 42.

Augmentation, "add" mode: every epoch shows every original window once, plus a scaled copy of a
random FRACTION of the windows (a new random choice every epoch).
  strong: multiplier = target size / window size, target 0.8-5.5, kept in x1/3 - x3 (model C's rule)
  mild:   the same, then also kept in x0.6 - x1.6
Neutral copies: x0.5 - x1.5, lowered so the largest signal is <= 0.25.

Runs: A0, A20, A50, A100 (strong), then B = mild at the best non-zero amount of step A (chosen on
VALIDATION). The test sets are scored once per run, with gate temperature 0.7 (as deployed).
Outputs: models and logs in --out, the comparison table in Analysis/results/.
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
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, help="folder for models and training logs")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fractions", default="0,0.2,0.5,1.0",
                    help="step A amounts to train, comma-separated (strong strength)")
parser.add_argument("--no-step-b", action="store_true", help="skip the mild run of step B")
args = parser.parse_args()
FRACTIONS = [float(f) for f in args.fractions.split(",")]
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support
from tensorflow.keras import callbacks

# ---------------------------------------------------------------- data
Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
X_train, Y_train, W_train = Z["X_train"], Z["Y_train"], Z["W_train"]
CLASS_NAMES = [str(c) for c in Z["class_names"]]
NEUTRAL_ID = CLASS_NAMES.index("Neutral")
CLASS_LABELS = np.arange(len(CLASS_NAMES))
SEQ_LENGTH, input_size, num_classes = 5, X_train.shape[-1], len(CLASS_NAMES)
VAL = {"old_acted": (Z["X_val_old"], Z["Y_val_old"]), "new_acted": (Z["X_val_new"], Z["Y_val_new"]),
       "emojihero": (Z["X_val_eh"], Z["Y_val_eh"])}
TEST = {"old VR": (Z["X_test_old"], Z["Y_test_old"]), "new VR": (Z["X_test_new"], Z["Y_test_new"]),
        "EmojiHero": (Z["X_test_eh"], Z["Y_test_eh"])}
X_val_all = np.concatenate([v[0] for v in VAL.values()]); Y_val_all = np.concatenate([v[1] for v in VAL.values()])
assert len(X_train) == 8181 and np.isclose(W_train.mean(), 1.0)

# ---------------------------------------------------------------- architecture (notebook cell 74)
nb = json.load(open(REPO / "notebooks/teacher_notebook_only_acted_data_volume_augmentation.ipynb",
                    encoding="utf-8"))["cells"]
arch = "".join(nb[74]["source"])
assert "def build_moe_single_input" in arch
exec(arch)

# ---------------------------------------------------------------- augmentation
TARGET_LOW, TARGET_HIGH, GAIN_MIN, GAIN_MAX = 0.8, 5.5, 1 / 3, 3.0
MILD_MIN, MILD_MAX = 0.6, 1.6
size_of = lambda X: X[..., :63].sum(-1).mean(-1)


def gain_ranges(X, y, strength):
    s = np.maximum(size_of(X), 1e-6)
    low = np.clip(TARGET_LOW / s, GAIN_MIN, GAIN_MAX)
    high = np.clip(TARGET_HIGH / s, GAIN_MIN, GAIN_MAX)
    if strength == "mild":
        low, high = np.clip(low, MILD_MIN, MILD_MAX), np.clip(high, MILD_MIN, MILD_MAX)
    low[y == NEUTRAL_ID], high[y == NEUTRAL_ID] = 0.5, 1.5
    return low, high


def draw_gains(X, y, low, high, rng):
    g = np.exp(rng.uniform(np.log(low), np.log(high)))
    neu = y == NEUTRAL_ID
    if neu.any():
        g[neu] = np.minimum(g[neu], 0.25 / np.maximum(X[neu].max(axis=(1, 2)), 1e-6))
    return g.astype(np.float32)


class AddCopies(tf.keras.utils.PyDataset):
    def __init__(self, X, y, w, fraction, strength, batch_size, seed, **kw):
        super().__init__(**kw)
        self.X, self.y, self.w = X.astype(np.float32), y.astype(np.int64), w.astype(np.float32)
        self.fraction, self.batch_size = fraction, batch_size
        self.rng = np.random.default_rng(seed)
        self.low, self.high = gain_ranges(self.X, self.y, strength)
        self._new_epoch()

    def _new_epoch(self):
        n = len(self.X)
        k = int(round(self.fraction * n))
        chosen = np.sort(self.rng.choice(n, size=k, replace=False))
        self.source = np.concatenate([np.arange(n), chosen])
        self.gain = np.ones(len(self.source), np.float32)
        if k:
            self.gain[n:] = draw_gains(self.X[chosen], self.y[chosen], self.low[chosen],
                                       self.high[chosen], self.rng)
        order = self.rng.permutation(len(self.source))
        self.source, self.gain = self.source[order], self.gain[order]

    def __len__(self):
        return math.ceil(len(self.source) / self.batch_size)

    def __getitem__(self, i):
        b = slice(i * self.batch_size, (i + 1) * self.batch_size)
        rows = self.source[b]
        return np.clip(self.X[rows] * self.gain[b, None, None], 0, 1), self.y[rows], self.w[rows]

    def on_epoch_end(self):
        self._new_epoch()


# ---------------------------------------------------------------- training callback (notebook)
def macro_f1(y, p):
    return f1_score(y, p, labels=CLASS_LABELS, average="macro", zero_division=0)


class DomainBalancedMacroF1(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        scores = {}
        for name, (X, y) in VAL.items():
            scores[name] = macro_f1(y, self.model.predict(X, batch_size=256, verbose=0).argmax(1))
            logs[f"val_{name}_macro_f1"] = scores[name]
        logs["val_domain_balanced_macro_f1"] = float(np.mean(list(scores.values())))
        print(f"  epoch {epoch + 1:2d}  val old {scores['old_acted']:.4f}  new {scores['new_acted']:.4f}"
              f"  EH {scores['emojihero']:.4f}  domain-bal {logs['val_domain_balanced_macro_f1']:.4f}",
              flush=True)


def train(tag, fraction, strength, seed):
    tf.keras.backend.clear_session()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model, anneal_cb = build_moe_single_input(
        input_shape=(SEQ_LENGTH, input_size), num_classes=num_classes, tau_init=2.0, tau_final=0.7,
        gate_hidden=128, classifier_hidden=256, lb_coef=0.02, ent_coef=0.01)
    data = AddCopies(X_train, Y_train, W_train, fraction, strength, batch_size=10, seed=seed)
    print(f"\n=== {tag}: +{fraction:.0%} copies, {strength}, {len(data.source)} windows per epoch,"
          f" {len(data)} steps", flush=True)
    t = time.time()
    model.fit(
        data, validation_data=(X_val_all, Y_val_all), epochs=30, verbose=0,
        callbacks=[DomainBalancedMacroF1(),
                   callbacks.EarlyStopping(monitor="val_domain_balanced_macro_f1", mode="max", patience=8,
                                           restore_best_weights=True, verbose=0),
                   callbacks.CSVLogger(str(OUT / f"{tag}_training_log.csv")),
                   anneal_cb])
    model.save(str(OUT / f"{tag}.keras"))
    print(f"  trained in {(time.time() - t) / 60:.1f} min", flush=True)
    return model


def evaluate(label, tag, model):
    model.get_layer("gate").tau.assign(0.7)          # as the deployed ONNX; not stored in .keras
    log = pd.read_csv(OUT / f"{tag}_training_log.csv")
    best = log.loc[log["val_domain_balanced_macro_f1"].idxmax()]
    row = {"run": label, "best epoch": int(best["epoch"]) + 1,
           "val domain-bal": best["val_domain_balanced_macro_f1"],
           "val old": best["val_old_acted_macro_f1"], "val new": best["val_new_acted_macro_f1"],
           "val EmojiHero": best["val_emojihero_macro_f1"]}
    res = {k: (X, y, model.predict(X, batch_size=256, verbose=0)) for k, (X, y) in TEST.items()}
    res["VR combined"] = tuple(np.concatenate([res["old VR"][i], res["new VR"][i]]) for i in range(3))
    res["pooled"] = tuple(np.concatenate([res[s][i] for s in TEST]) for i in range(3))
    f1 = {k: macro_f1(y, p.argmax(1)) for k, (X, y, p) in res.items()}

    def ll(y, p):
        p = np.clip(p, 1e-12, 1); return log_loss(y, p / p.sum(1, keepdims=True), labels=CLASS_LABELS)

    X, y, p = res["VR combined"]; pred = p.argmax(1)
    expr = y != NEUTRAL_ID; quiet = expr & (size_of(X) < 1.5)
    row.update({
        "test VR combined": f1["VR combined"], "test new VR": f1["new VR"], "test EmojiHero": f1["EmojiHero"],
        "test pooled": f1["pooled"], "test domain-bal": np.mean([f1[s] for s in TEST]),
        "VR correct (of 607)": int(np.sum(pred == y)),
        "false Neutral (of 520)": int(np.sum(expr & (pred == NEUTRAL_ID))),
        "quiet accuracy (n=44)": float(np.mean(pred[quiet] == y[quiet])),
        "log loss VR": ll(y, p), "log loss domain-bal": np.mean([ll(res[s][1], res[s][2]) for s in TEST]),
    })
    f1e = precision_recall_fscore_support(y, pred, labels=CLASS_LABELS, zero_division=0)[2]
    return row, {"run": label, **dict(zip(CLASS_NAMES, f1e))}


rows, emo = [], []
def run(label, fraction, strength):
    tag = f"add{round(fraction * 100)}-{strength}-seed{args.seed}"
    model = train(tag, fraction, strength, args.seed)
    r, e = evaluate(label, tag, model)
    rows.append(r); emo.append(e)
    pd.DataFrame(rows).to_csv(OUT / "progress.csv", index=False)
    return r

for frac in FRACTIONS:
    run(f"A{round(frac * 100)} (+{frac:.0%}, strong)" if frac else "A0 (no copies)", frac, "strong")

if not args.no_step_b:
    stepA = [r for r in rows if not r["run"].startswith("A0")]
    best = max(stepA, key=lambda r: r["val domain-bal"])
    best_frac = float(best["run"].split()[0][1:]) / 100
    print(f"\nbest amount on VALIDATION: {best['run']} -> step B at +{best_frac:.0%}")
    run(f"B{round(best_frac * 100)} (+{best_frac:.0%}, mild)", best_frac, "mild")

table = pd.DataFrame(rows)
reference = {"run": "D deployed (Colab, reference)", "test VR combined": 0.4545, "test new VR": 0.4137,
             "test EmojiHero": 0.5657, "test pooled": 0.5460, "test domain-bal": 0.4910,
             "VR correct (of 607)": 308, "false Neutral (of 520)": 39, "quiet accuracy (n=44)": 0.25,
             "log loss VR": 1.3530, "log loss domain-bal": 1.3412}
table = pd.concat([pd.DataFrame([reference]), table], ignore_index=True)
pd.set_option("display.width", 250)
print("\n" + table.round(4).to_string(index=False))
print("\nVR combined, F1 per emotion")
print(pd.DataFrame(emo).round(3).to_string(index=False))
results = REPO / "Analysis/results"
table.to_csv(results / f"aug_amount_strength_seed{args.seed}.csv", index=False)
pd.DataFrame(emo).to_csv(results / f"aug_amount_strength_per_emotion_seed{args.seed}.csv", index=False)
print("saved to", results)
