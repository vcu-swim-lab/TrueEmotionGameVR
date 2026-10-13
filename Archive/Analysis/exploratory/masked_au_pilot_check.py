"""Pilot criterion 3: did the encoder change? Validation windows (old + new acted), saved pilot weights.
Prediction disagreement with A, and linear CKA of the 'fuse' features with A (1.0 = same representation)."""
import sys
from pathlib import Path
sys.argv = [sys.argv[0], "--check"]
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis/exploratory"))
import numpy as np
import tensorflow as tf
import masked_au_aux as m

Xv = np.concatenate([m.VAL["old"][0], m.VAL["new"][0]]).astype(np.float32)
D = REPO / "Analysis/data/masked_au_pilot"


def load(name):
    model, _ = m.build(42)
    model.load_weights(str(D / f"{name}.weights.h5"))
    model.get_layer("gate").tau.assign(0.7)
    feat = tf.keras.Model(model.input, model.get_layer("fuse").output)
    return model.predict(Xv, verbose=0).argmax(1), feat.predict(Xv, verbose=0)


def cka(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return float(np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


pA, fA = load("A_0_42")
for name in ("B_200_42", "C_200_42", "B_600_42"):
    p, f = load(name)
    print(f"{name:<10} disagreement with A on validation {np.mean(p != pA):.1%} | CKA with A {cka(f, fA):.3f}")
