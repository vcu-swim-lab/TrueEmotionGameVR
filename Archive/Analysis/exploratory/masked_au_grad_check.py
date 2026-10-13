"""Did the reconstruction task have any pull on the shared encoder? Diagnostic for masked_au_aux.py.

For a freshly built model (same seed and setup as the experiment) and for 50 random batch pairs,
the gradient norms on the SHARED variables (experts, gate; not the classifier or reconstruction
heads) of
    L_cls + L_gate        (supervised batch, training=True)
    L_rec                 (natural or acted aux batch, training=False, beta = 1)
and the cosine between the two gradients. Median over the batches.
"""
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--check"]
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis/exploratory"))
import numpy as np
import tensorflow as tf

import masked_au_aux as m


def main():
    model, rec_model = m.build(42)
    shared = [v for v in model.trainable_variables
              if not v.path.startswith(("y/", "clf_dense/", "rec_d1/", "rec_out/"))]
    huber = tf.keras.losses.Huber(delta=1.0, reduction="none")
    rng = np.random.default_rng(0)
    for pool in ("natural", "acted"):
        s = m.AuxSampler(pool, 42)
        ratios, cos = [], []
        for _ in range(50):
            i = rng.choice(len(m.X_SUP), m.BATCH, replace=False)
            xa, ta, ma = s.batch()
            with tf.GradientTape(persistent=True) as tape:
                p = model(m.X_SUP[i], training=True)
                ce = tf.keras.losses.sparse_categorical_crossentropy(m.Y_SUP[i], p)
                l_sup = tf.reduce_sum(m.W_SUP[i] * ce) / m.BATCH + tf.add_n(model.losses)
                r = rec_model(xa, training=False)
                l_rec = tf.reduce_sum(huber(ta[..., None], r[..., None]) * ma) / tf.reduce_sum(ma)
            g1 = tape.gradient(l_sup, shared)
            g2 = tape.gradient(l_rec, shared)
            f1 = tf.concat([tf.reshape(g, [-1]) for g in g1 if g is not None], 0)
            f2 = tf.concat([tf.reshape(g, [-1]) for g in g2 if g is not None], 0)
            n1, n2 = float(tf.norm(f1)), float(tf.norm(f2))
            ratios.append(n2 / n1)
            cos.append(float(tf.reduce_sum(f1 * f2)) / (n1 * n2 + 1e-12))
        print(f"{pool:<8} |grad L_rec| / |grad L_cls| on shared encoder: median {np.median(ratios):.3f} "
              f"(q10 {np.quantile(ratios, .1):.3f}, q90 {np.quantile(ratios, .9):.3f}); "
              f"cosine median {np.median(cos):+.3f}")


if __name__ == "__main__":
    main()
