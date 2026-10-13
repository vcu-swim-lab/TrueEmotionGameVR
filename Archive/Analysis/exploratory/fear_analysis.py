"""Why is Fear about 0% on the headset for every teacher?

Compares the headset Fear ticks with the training data:
  1. how big the face is (face size = mean over frames of the sum of the 63 non-tongue AUs);
  2. which muscles move (top AUs);
  3. which training emotion the headset Fear face looks most like (cosine similarity of
     size-normalised AU profiles, so only the SHAPE counts, not the volume);
  4. what each model predicts for headset Fear, and the score it gives to Fear.

Training data: old acted (all people, gated like the notebook), new acted (local corpus, gated)
and EmojiHero validation + test people (ungated, as eh.to_model_windows gives them).
Headset: every session-local-v1 recording, scored ticks 1-9 of both blocks.
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import pandas as pd
import acting_dataset as ad
import calibration_study as cs
import emojihero_calibration_study as eh
import session_analysis as sa
import teacher_preprocessing as tp

E = list(eh.EMOTIONS)
FEAR, SURPRISE, NEUTRAL = E.index("Fear"), E.index("Surprise"), E.index("Neutral")
M63 = slice(0, 63)
AU = [
    "BrowLowererL", "BrowLowererR", "CheekPuffL", "CheekPuffR", "CheekRaiserL", "CheekRaiserR",
    "CheekSuckL", "CheekSuckR", "ChinRaiserB", "ChinRaiserT", "DimplerL", "DimplerR", "EyesClosedL",
    "EyesClosedR", "EyesLookDownL", "EyesLookDownR", "EyesLookLeftL", "EyesLookLeftR",
    "EyesLookRightL", "EyesLookRightR", "EyesLookUpL", "EyesLookUpR", "InnerBrowRaiserL",
    "InnerBrowRaiserR", "JawDrop", "JawSidewaysLeft", "JawSidewaysRight", "JawThrust",
    "LidTightenerL", "LidTightenerR", "LipCornerDepressorL", "LipCornerDepressorR",
    "LipCornerPullerL", "LipCornerPullerR", "LipFunnelerLB", "LipFunnelerLT", "LipFunnelerRB",
    "LipFunnelerRT", "LipPressorL", "LipPressorR", "LipPuckerL", "LipPuckerR", "LipStretcherL",
    "LipStretcherR", "LipSuckLB", "LipSuckLT", "LipSuckRB", "LipSuckRT", "LipTightenerL",
    "LipTightenerR", "LipsToward", "LowerLipDepressorL", "LowerLipDepressorR", "MouthLeft",
    "MouthRight", "NoseWrinklerL", "NoseWrinklerR", "OuterBrowRaiserL", "OuterBrowRaiserR",
    "UpperLidRaiserL", "UpperLidRaiserR", "UpperLipRaiserL", "UpperLipRaiserR",
]
MODELS = REPO.parent / "VREmotionDetection/Runtime/Models"
RUNS = {
    "D": MODELS / "sparseVR_teacher_model_acted_only.onnx",
    "B": MODELS / "sparseVR_teacher_model_acted_only_vol-aug-k1_excluding-EmojiHero.onnx",
    "C": MODELS / "sparseVR_teacher_model_acted_only_vol-aug-k1_with-EmojiHero.onnx",
}

size = lambda w: w[..., M63].sum(-1).mean(-1)          # per window
profile = lambda w: w[..., M63].mean(axis=(0, 1))       # mean AU vector over windows and frames


def shape(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


# ---------------------------------------------------------------- training data
groups = {}                                              # (collection, emotion) -> windows
old_root = REPO / "Analysis/data/filtered-dataset-20260924T162300Z-1-001/filtered-dataset"
for folder in sorted(old_root.iterdir()):
    for f in sorted(folder.glob("ACT_*.csv")):
        rows = list(csv.reader(open(f, encoding="utf-8-sig")))
        fr = np.array([[float(v) for v in r[1:]] for r in rows[1:] if len(r) == len(rows[0])])
        w = tp.make_windows(fr, f.stem[4:])
        if len(w):
            groups.setdefault(("old acted", E.index(f.stem[4:])), []).append(w)
for r in ad.load_corpus(ad.find_corpus_root()).recordings:
    w = tp.make_windows(tp.downsample(r.au.astype(np.float64), tp.DOWNSAMPLE_FACTOR), r.emotion)
    if len(w):
        groups.setdefault(("new acted", E.index(r.emotion)), []).append(w)
for split in ("validation", "test"):
    for s in eh.load_split(REPO / "Analysis/data", split)[0]:
        groups.setdefault(("EmojiHero", s.emotion), []).append(eh.to_model_windows(s.au))
groups = {k: np.concatenate(v) for k, v in groups.items()}

# ---------------------------------------------------------------- headset
hs_w, hs_y, hs_sess, hs_pred, hs_pfear = [], [], [], {n: [] for n in RUNS}, {n: [] for n in RUNS}
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if (meta.get("protocol_version") != "session-local-v1"
            or meta.get("calibration", {}).get("calibration_valid") not in ("true", None)):
        continue
    log = sa.load_session(d / "predictions.csv")
    first = True
    for n, path in RUNS.items():
        data = cs.replay(log, path)
        k = np.isin(data.tick, list(cs.SCORED_TICKS))
        if first:
            hs_w.append(data.windows[k]); hs_y.append(data.prompted[k])
            hs_sess += [d.name[-6:]] * int(k.sum())
            first = False
        hs_pred[n].append(data.scores[k].argmax(1))
        hs_pfear[n].append(data.scores[k][:, FEAR])
hs_w = np.concatenate(hs_w); hs_y = np.concatenate(hs_y); hs_sess = np.array(hs_sess)
hs_pred = {n: np.concatenate(v) for n, v in hs_pred.items()}
hs_pfear = {n: np.concatenate(v) for n, v in hs_pfear.items()}
fear = hs_y == FEAR
print(f"headset: {len(set(hs_sess))} sessions, {len(hs_y)} scored ticks, {fear.sum()} Fear ticks")

# ---------------------------------------------------------------- 1. face size
print("\n1. FACE SIZE (median per window; 1.0 = resting face level)")
rows = []
for e in (FEAR, SURPRISE, NEUTRAL):
    row = {"emotion": E[e], "headset": np.median(size(hs_w[hs_y == e]))}
    for c in ("old acted", "new acted", "EmojiHero"):
        row[c] = np.median(size(groups[(c, e)])) if (c, e) in groups else np.nan
    rows.append(row)
print(pd.DataFrame(rows).round(2).to_string(index=False))
weak = ~(hs_w[..., M63] > 0.25).any(axis=(1, 2))
print(f"   headset Fear ticks with no AU above 0.25 (weak): {weak[fear].mean():.0%}"
      f"   (Surprise {weak[hs_y == SURPRISE].mean():.0%}, Neutral {weak[hs_y == NEUTRAL].mean():.0%})")

# ---------------------------------------------------------------- 2. top muscles
print("\n2. TOP 8 MUSCLES (mean value)")
top = {}
for name, w in (("headset Fear", hs_w[fear]), ("headset Neutral", hs_w[hs_y == NEUTRAL]),
                ("headset Surprise", hs_w[hs_y == SURPRISE]),
                ("old acted Fear", groups[("old acted", FEAR)]),
                ("new acted Fear", groups[("new acted", FEAR)]),
                ("EmojiHero Fear", groups[("EmojiHero", FEAR)]),
                ("old acted Surprise", groups[("old acted", SURPRISE)])):
    p = profile(w)
    idx = np.argsort(p)[::-1][:8]
    top[name] = idx
    print(f"  {name:<19} " + ", ".join(f"{AU[i]} {p[i]:.2f}" for i in idx))

# ---------------------------------------------------------------- 3. shape similarity
print("\n3. WHICH TRAINING EMOTION DOES HEADSET FEAR LOOK LIKE?  (cosine of size-normalised profiles)")
hf = shape(profile(hs_w[fear]))
rows = []
for c in ("old acted", "new acted", "EmojiHero"):
    rows.append({"collection": c, **{E[e]: float(hf @ shape(profile(groups[(c, e)])))
                                     for e in range(7) if (c, e) in groups}})
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n   per session: headset Fear vs training (old + new acted pooled) Fear / Surprise / Neutral")
pool = {e: shape(profile(np.concatenate([groups[(c, e)] for c in ("old acted", "new acted")])))
        for e in range(7)}
for s in sorted(set(hs_sess)):
    m = fear & (hs_sess == s)
    v = shape(profile(hs_w[m]))
    sims = {E[e]: float(v @ pool[e]) for e in range(7)}
    best = max(sims, key=sims.get)
    print(f"   {s}  size {np.median(size(hs_w[m])):.2f}  Fear {sims['Fear']:.2f}  "
          f"Surprise {sims['Surprise']:.2f}  Neutral {sims['Neutral']:.2f}  -> most like {best}")

print("\n   training: how alike are Fear and Surprise themselves? (same collection)")
for c in ("old acted", "new acted", "EmojiHero"):
    print(f"   {c:<10} Fear vs Surprise {shape(profile(groups[(c, FEAR)])) @ shape(profile(groups[(c, SURPRISE)])):.2f}"
          f"   Fear vs Neutral {shape(profile(groups[(c, FEAR)])) @ shape(profile(groups[(c, NEUTRAL)])):.2f}")

# ---------------------------------------------------------------- 4. predictions
print("\n4. WHAT THE MODELS SAY FOR HEADSET FEAR")
for n in RUNS:
    p = hs_pred[n][fear]
    dist = {E[e]: f"{np.mean(p == e):.0%}" for e in range(7) if np.mean(p == e) > 0}
    print(f"   {n}: {dist}   mean Fear score {hs_pfear[n][fear].mean():.3f}   "
          f"Fear score on loud Fear ticks {hs_pfear[n][fear & ~weak].mean():.3f}")
for n in RUNS:
    p = hs_pred[n]
    called = p == FEAR
    print(f"   {n}: ticks called Fear: {called.sum()}  of which truly Fear {np.sum(called & fear)}"
          f"   (prompted Surprise called Fear: {np.sum(called & (hs_y == SURPRISE))})")
