"""Does applying our 0.25 self-filter to EmojiHero help the acted-only teacher?

Our VR recordings are filtered when windows are cut: a window is kept only if it holds 3 frames in a
row that pass (emotion: some AU > 0.25; Neutral: every AU <= 0.25). EmojiHero has never been
filtered. This trains model D's setup with EmojiHero filtered by the same rule and compares it with
the unfiltered baseline (the A0 runs of aug_amount_strength.py, same seeds, same code path).

  --filter all      drop every EmojiHero train/validation window that fails the rule
  --filter neutral  drop only the EmojiHero Neutral windows that fail it

No augmentation. Sample weights are recomputed with the notebook's class x source formula (which
reproduces D's weights exactly on the unfiltered data). Our VR data and all test sets are unchanged;
EmojiHero test is reported both unfiltered (comparable with earlier numbers) and filtered.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--filter", choices=["all", "neutral"], required=True)
parser.add_argument("--seeds", default="42,1,7,123,2024")
parser.add_argument("--out", required=True, help="folder for models and training logs")
args = parser.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks

Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
CLASS_NAMES = [str(c) for c in Z["class_names"]]
NEUTRAL = CLASS_NAMES.index("Neutral")
LABELS = np.arange(7)
EH = 1


def passes(X, y):
    """Our VR self-filter, applied to whole windows: 3 passing frames in a row."""
    peak = X.max(axis=2)                                   # [n, 5] largest value per frame
    ok = np.where((y == NEUTRAL)[:, None], peak <= 0.25, peak > 0.25)
    return (ok[:, 0:3].all(1)) | (ok[:, 1:4].all(1)) | (ok[:, 2:5].all(1))


def keep_mask(X, y):
    p = passes(X, y)
    return p if args.filter == "all" else (p | (y != NEUTRAL))


def weights(y, sources):
    classes = np.unique(y)
    cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y)))
    ids, counts = np.unique(sources, return_counts=True)
    sw = dict(zip(ids, len(sources) / (len(ids) * counts)))
    w = np.array([sw[s] * cw[c] for s, c in zip(sources, y)], np.float32)
    return w / w.mean()


assert np.allclose(weights(Z["Y_train"], Z["S_train"]), Z["W_train"], atol=1e-5)

S = Z["S_train"]
keep = (S != EH) | keep_mask(Z["X_train"], Z["Y_train"])
X_train, Y_train, S_train = Z["X_train"][keep], Z["Y_train"][keep], S[keep]
W_train = weights(Y_train, S_train)

eh_val_keep = keep_mask(Z["X_val_eh"], Z["Y_val_eh"])
VAL = {"old_acted": (Z["X_val_old"], Z["Y_val_old"]), "new_acted": (Z["X_val_new"], Z["Y_val_new"]),
       "emojihero": (Z["X_val_eh"][eh_val_keep], Z["Y_val_eh"][eh_val_keep])}
eh_test_keep = passes(Z["X_test_eh"], Z["Y_test_eh"])
TEST = {"old VR": (Z["X_test_old"], Z["Y_test_old"]), "new VR": (Z["X_test_new"], Z["Y_test_new"]),
        "EmojiHero": (Z["X_test_eh"], Z["Y_test_eh"]),
        "EmojiHero filtered": (Z["X_test_eh"][eh_test_keep], Z["Y_test_eh"][eh_test_keep])}
X_val_all = np.concatenate([v[0] for v in VAL.values()]); Y_val_all = np.concatenate([v[1] for v in VAL.values()])

print(f"filter={args.filter}: EmojiHero train {int((S == EH).sum())} -> {int((S_train == EH).sum())} windows, "
      f"EmojiHero validation {len(Z['Y_val_eh'])} -> {int(eh_val_keep.sum())}; total train {len(Y_train)}", flush=True)
print("EmojiHero train per class after filter:",
      {CLASS_NAMES[c]: int(((S_train == EH) & (Y_train == c)).sum()) for c in range(7)}, flush=True)

nb = json.load(open(REPO / "notebooks/teacher_notebook_only_acted_data_volume_augmentation.ipynb", encoding="utf-8"))["cells"]
arch = "".join(nb[74]["source"]); assert "def build_moe_single_input" in arch
exec(arch)

size_of = lambda X: X[..., :63].sum(-1).mean(-1)
macro = lambda y, p: f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


class Epochs(tf.keras.utils.PyDataset):
    """The data path of aug_amount_strength.py's AddCopies at +0% copies, so the baseline (A0) and
    these runs differ only in the filter: same rng calls, same per-epoch shuffle, batch 10."""

    def __init__(self, X, y, w, seed, **kw):
        super().__init__(**kw)
        self.X, self.y, self.w = X.astype(np.float32), y.astype(np.int64), w.astype(np.float32)
        self.rng = np.random.default_rng(seed)
        self._new_epoch()

    def _new_epoch(self):
        n = len(self.X)
        chosen = np.sort(self.rng.choice(n, size=0, replace=False))
        self.source = np.concatenate([np.arange(n), chosen])
        self.source = self.source[self.rng.permutation(len(self.source))]

    def __len__(self):
        return int(np.ceil(len(self.source) / 10))

    def __getitem__(self, i):
        rows = self.source[i * 10:(i + 1) * 10]
        return self.X[rows], self.y[rows], self.w[rows]

    def on_epoch_end(self):
        self._new_epoch()


class DomainBalanced(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        s = {k: macro(y, self.model.predict(X, batch_size=256, verbose=0).argmax(1)) for k, (X, y) in VAL.items()}
        for k, v in s.items():
            logs[f"val_{k}_macro_f1"] = v
        logs["val_domain_balanced_macro_f1"] = float(np.mean(list(s.values())))


rows = []
for seed in SEEDS:
    tag = f"ehfilter-{args.filter}-seed{seed}"
    tf.keras.backend.clear_session()
    os.environ["PYTHONHASHSEED"] = str(seed); random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model, anneal = build_moe_single_input(input_shape=(5, 70), num_classes=7, tau_init=2.0, tau_final=0.7,
                                           gate_hidden=128, classifier_hidden=256, lb_coef=0.02, ent_coef=0.01)
    t = time.time()
    model.fit(Epochs(X_train, Y_train, W_train, seed), epochs=30,
              validation_data=(X_val_all, Y_val_all), verbose=0,
              callbacks=[DomainBalanced(),
                         callbacks.EarlyStopping(monitor="val_domain_balanced_macro_f1", mode="max", patience=8,
                                                 restore_best_weights=True),
                         callbacks.CSVLogger(str(OUT / f"{tag}_training_log.csv")), anneal])
    model.save(str(OUT / f"{tag}.keras"))
    model.get_layer("gate").tau.assign(0.7)

    log = pd.read_csv(OUT / f"{tag}_training_log.csv")
    best = log.loc[log["val_domain_balanced_macro_f1"].idxmax()]
    res = {k: (X, y, model.predict(X, batch_size=256, verbose=0)) for k, (X, y) in TEST.items()}
    res["VR combined"] = tuple(np.concatenate([res["old VR"][i], res["new VR"][i]]) for i in range(3))
    f1 = {k: macro(y, p.argmax(1)) for k, (X, y, p) in res.items()}
    X, y, p = res["VR combined"]; pred = p.argmax(1); expr = y != NEUTRAL; quiet = expr & (size_of(X) < 1.5)
    pc = np.clip(p, 1e-12, 1)
    row = {"filter": args.filter, "seed": seed, "best epoch": int(best["epoch"]) + 1,
           "val domain-bal": best["val_domain_balanced_macro_f1"],
           "test VR combined": f1["VR combined"], "test new VR": f1["new VR"], "test old VR": f1["old VR"],
           "test EmojiHero": f1["EmojiHero"], "test EmojiHero filtered": f1["EmojiHero filtered"],
           "test domain-bal": np.mean([f1["old VR"], f1["new VR"], f1["EmojiHero"]]),
           "VR correct (of 607)": int((pred == y).sum()),
           "false Neutral (of 520)": int((expr & (pred == NEUTRAL)).sum()),
           "quiet accuracy (n=44)": float(np.mean(pred[quiet] == y[quiet])),
           "log loss VR": log_loss(y, pc / pc.sum(1, keepdims=True), labels=LABELS),
           **{f"F1 {CLASS_NAMES[c]}": f1_score(y, pred, labels=[c], average="macro", zero_division=0) for c in range(7)}}
    rows.append(row)
    print(f"  {tag}: {(time.time() - t) / 60:.1f} min, best epoch {row['best epoch']}, VR combined "
          f"{row['test VR combined']:.4f}, new VR {row['test new VR']:.4f}, EmojiHero {row['test EmojiHero']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(REPO / f"Analysis/results/eh_filter_{args.filter}.csv", index=False)

print("done", flush=True)
