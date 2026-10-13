"""Where should the volume-augmentation range be?  Data-driven, before any training.

Training data = old acted + new acted, windowed and GATED exactly as the notebook does
(teacher_preprocessing.make_windows: some AU > 0.25 for 3 frames; Neutral: all <= 0.25).
Augmentation multiplies those gated windows. So the question per emotion is:

    how much would a training window have to shrink / grow to look like a real user?

    ratio = (a user's face size for emotion e) / (typical gated training face size for e)

Small users = the 6 headset sessions (ungated, as the runtime sees them).
Big users   = EmojiHero validation+test people (NOT trained on; a yardstick for exaggerators).
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import acting_dataset as ad
import calibration_study as cs
import emojihero_calibration_study as eh
import session_analysis as sa
import teacher_preprocessing as tp

E = list(eh.EMOTIONS)
M63 = slice(0, 63)
MODEL = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
size = lambda w: w[..., M63].sum(-1).mean()          # mean per-frame sum of 63 channels

# ---------------------------------------------------------------- training windows (gated)
train = []                                            # (person, emotion, windows)
old_root = REPO / "Analysis/data/filtered-dataset-20260924T162300Z-1-001/filtered-dataset"
for folder in sorted(old_root.iterdir()):
    for f in sorted(folder.glob("ACT_*.csv")):
        rows = list(csv.reader(open(f, encoding="utf-8-sig")))
        fr = np.array([[float(v) for v in r[1:]] for r in rows[1:] if len(r) == len(rows[0])])
        emo = f.stem[4:]
        train.append(("old:" + folder.name, E.index(emo), tp.make_windows(fr, emo)))
for r in ad.load_corpus(ad.find_corpus_root()).recordings:
    fr = tp.downsample(r.au.astype(np.float64), tp.DOWNSAMPLE_FACTOR)
    train.append(("new:" + r.participant, E.index(r.emotion), tp.make_windows(fr, r.emotion)))

Xtr = np.concatenate([w for _, _, w in train if len(w)])
ytr = np.concatenate([[e] * len(w) for _, e, w in train if len(w)])
print(f"training windows after the gate: {len(Xtr)} from {len({p for p, _, _ in train})} people")
lost = [(p, E[e]) for p, e, w in train if len(w) == 0]
print(f"recordings that gave NO window (gate removed them entirely): {len(lost)}")

# typical training size per emotion = median over people of their gated window size
T = {}
for e in range(7):
    per = [size(w) for _, ee, w in train if ee == e and len(w)]
    T[e] = float(np.median(per))

# ---------------------------------------------------------------- users
hs = {}                                               # (session, emotion) -> ticks
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if meta.get("protocol_version") != "session-local-v1":
        continue
    r = cs.replay(sa.load_session(d / "predictions.csv"), MODEL)
    k = np.isin(r.tick, list(cs.SCORED_TICKS))
    for e in range(7):
        hs[(d.name[-6:], e)] = r.windows[k & (r.prompted == e)]
seqs = eh.load_split(REPO / "Analysis/data", "validation")[0] + \
       eh.load_split(REPO / "Analysis/data", "test")[0]
emo_users = {}
for s in seqs:
    emo_users.setdefault((s.participant, s.emotion), []).append(eh.to_model_windows(s.au))
emo_users = {k: np.concatenate(v) for k, v in emo_users.items()}


def ratios(users, e):
    return np.array([size(w) / T[e] for (p, ee), w in users.items() if ee == e and len(w)])


own = {e: np.array([size(w) / T[e] for _, ee, w in train if ee == e and len(w)]) for e in range(7)}

print("\n" + "=" * 92)
print("1. HOW BIG IS EACH GROUP'S FACE COMPARED WITH A TYPICAL (GATED) TRAINING FACE?")
print("   1.00 = same size as training; 0.50 = half; 2.00 = twice. Range = 10th-90th percentile.")
print("=" * 92)
print(f"  {'emotion':<10}{'train size':>11}   {'training people':>17}   {'headset users':>17}   {'EmojiHero users':>17}")
for e in range(7):
    cells = []
    for v in (own[e], ratios(hs, e), ratios(emo_users, e)):
        cells.append(f"{np.median(v):.2f} ({np.quantile(v, .1):.2f}-{np.quantile(v, .9):.2f})")
    print(f"  {E[e]:<10}{T[e]:>11.2f}   " + "   ".join(f"{c:>17}" for c in cells))

expr = [e for e in range(7) if e != 4]
hs_all = np.concatenate([ratios(hs, e) for e in expr])
emo_all = np.concatenate([ratios(emo_users, e) for e in expr])
tr_all = np.concatenate([own[e] for e in expr])
print("\n  expressive emotions pooled (Neutral separately):")
for lab, v in (("training people", tr_all), ("headset users", hs_all), ("EmojiHero users", emo_all)):
    print(f"    {lab:<16} 5%: {np.quantile(v, .05):.2f}   median: {np.median(v):.2f}   95%: {np.quantile(v, .95):.2f}"
          f"   min {v.min():.2f}  max {v.max():.2f}")
low = float(np.quantile(hs_all, .05)); high = float(np.quantile(emo_all, .95))
print(f"\n  => range that covers small users (headset 5%) to big users (EmojiHero 95%): "
      f"x{low:.2f} to x{high:.2f}")
n_low, n_high = np.quantile(ratios(hs, 4), .05), np.quantile(ratios(emo_users, 4), .95)
print(f"  => Neutral on its own: x{n_low:.2f} to x{n_high:.2f}")

print("\n" + "=" * 92)
print("2. WHAT HAPPENS TO TRAINING WINDOWS AT EACH VOLUME")
print("=" * 92)
print("   'clipped' = muscle values pushed to the 1.0 ceiling (information lost)")
print("   'below gate' = expressive windows with no muscle above 0.25 (the headset has 32% of these)")
print(f"  {'volume':>7}{'clipped values':>16}{'windows w/ clipping':>21}{'below gate':>12}")
ex = ytr != 4
for g in (0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
    x = Xtr[ex][..., M63] * g
    clip_vals = np.mean(x > 1.0)
    clip_win = np.mean((x > 1.0).any(axis=(1, 2)))
    below = np.mean(~(x > 0.25).any(axis=-1).any(axis=-1))
    print(f"  {g:>7.2f}{clip_vals:>16.2%}{clip_win:>21.1%}{below:>12.0%}")

print("\n" + "=" * 92)
print("3. DOES A CANDIDATE RANGE COVER REAL USERS?  (expressive emotions)")
print("   Share of each group's face sizes that fall inside what augmented training shows.")
print("=" * 92)
rng = np.random.default_rng(0)
tr_sizes = Xtr[ex][..., M63].sum(-1).mean(-1)
hs_sizes = np.concatenate([w[..., M63].sum(-1).mean(-1) for (p, e), w in hs.items() if e != 4])
emo_sizes = np.concatenate([w[..., M63].sum(-1).mean(-1) for (p, e), w in emo_users.items() if e != 4])
cands = [(1.0, 1.0), (0.5, 2.0), (0.4, 2.0), (0.3, 2.0), (low, high), (0.3, 2.5)]
print(f"  {'range':<14}{'headset covered':>17}{'EmojiHero covered':>19}")
for lo, hi in cands:
    g = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(20, len(tr_sizes)))) if hi > lo else np.ones((1, len(tr_sizes)))
    aug = (np.minimum(tr_sizes[None] * g, 63)).ravel()   # size scales linearly until clipping
    a_lo, a_hi = np.quantile(aug, .02), np.quantile(aug, .98)
    cov = lambda s: np.mean((s >= a_lo) & (s <= a_hi))
    print(f"  x{lo:.2f}-x{hi:.2f}{cov(hs_sizes):>15.0%}{cov(emo_sizes):>19.0%}")
