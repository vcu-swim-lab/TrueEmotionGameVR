"""Old acted (2025) vs EmojiHero vs new acted (15, ActingRecordings) vs headset (6 sessions).
Acted-only teacher. Read-only.

Face size = mean over frames of the sum of the 63 measured channels (EmojiHero has no tongue).
Model windows = 5 consecutive ~10 Hz frames, non-overlapping, no gate (what the headset does);
EmojiHero uses the study's own pooled windows; headset uses the logged live windows.
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import onnxruntime as ort
import acting_dataset as ad
import au_names
import calibration_study as cs
import emojihero_calibration_study as eh
import session_analysis as sa
import teacher_preprocessing as tp

E = list(eh.EMOTIONS)
MODEL = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
rn = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
INP = rn.get_inputs()[0].name
M63 = slice(0, 63)


def windows_from_frames(f):
    n = len(f) // 5
    return f[: n * 5].reshape(n, 5, 70).astype(np.float32)


# Each dataset: list of items {person, emotion, frames [n,70] ~10 Hz, windows [k,5,70]}
D = {}

# --- old acted 2025
old_root = REPO / "Analysis/data/filtered-dataset-20260924T162300Z-1-001/filtered-dataset"
items = []
for folder in sorted(old_root.iterdir()):
    for f in sorted(folder.glob("ACT_*.csv")):
        rows = list(csv.reader(open(f, encoding="utf-8-sig")))
        good = [r for r in rows[1:] if len(r) == len(rows[0])]
        fr = np.array([[float(v) for v in r[1:]] for r in good])
        items.append({"p": folder.name, "e": E.index(f.stem[4:]), "frames": fr,
                      "win": windows_from_frames(fr)})
D["old acted"] = items

# --- new acted 15 (60 Hz -> mean of 6 rows, the teacher's own operation)
corpus = ad.load_corpus(ad.find_corpus_root())
items = []
for r in corpus.recordings:
    fr = tp.downsample(r.au.astype(np.float64), tp.DOWNSAMPLE_FACTOR)
    items.append({"p": r.participant, "e": E.index(r.emotion), "frames": fr,
                  "win": windows_from_frames(fr)})
D["new acted"] = items

# --- EmojiHero, held-out people only (validation + test)
seqs = eh.load_split(REPO / "Analysis/data", "validation")[0] + \
       eh.load_split(REPO / "Analysis/data", "test")[0]
items = []
for s in seqs:
    fr = np.zeros((len(s.au), 70)); fr[:, :63] = s.au
    items.append({"p": s.participant, "e": s.emotion, "frames": fr,
                  "win": eh.to_model_windows(s.au)})
D["EmojiHero"] = items

# --- headset: each tick's live window is one item (ticks 1-9)
items = []
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if meta.get("protocol_version") != "session-local-v1":
        continue
    rp = cs.replay(sa.load_session(d / "predictions.csv"), MODEL)
    for w, y, t in zip(rp.windows, rp.prompted, rp.tick):
        if t in cs.SCORED_TICKS:
            items.append({"p": d.name[-6:], "e": int(y), "frames": w, "win": w[None]})
D["headset"] = items

names = list(D)
for k in names:
    for it in D[k]:
        it["pred"] = rn.run(None, {INP: it["win"]})[0].argmax(1)

# ================================================================ 1. size
print("=" * 90)
print("1. SIZE")
print("=" * 90)
print(f"  {'':<12}{'people':>8}{'recordings':>12}{'frames (~10 Hz)':>17}   in acted-only training?")
train_note = {"old acted": "yes (most people)", "new acted": "yes (about 10 of 15)",
              "EmojiHero": "no (held-out people only)", "headset": "no"}
for k in names:
    unit = "ticks" if k == "headset" else ""
    print(f"  {k:<12}{len({i['p'] for i in D[k]}):>8}{len(D[k]):>12}"
          f"{sum(len(i['frames']) for i in D[k]):>17}   {train_note[k]}")
print("  headset: 'people' = sessions (some may be the same person); recordings = ticks")

# ================================================================ 2. face size
print("\n" + "=" * 90)
print("2. FACE SIZE -- mean sum of 63 channels per frame")
print("=" * 90)
print(f"  {'emotion':<10}" + "".join(f"{k:>12}" for k in names))
act = {k: {e: np.concatenate([i["frames"][:, M63] for i in D[k] if i["e"] == e]).sum(1).mean()
           for e in range(7)} for k in names}
for e in range(7):
    print(f"  {E[e]:<10}" + "".join(f"{act[k][e]:>12.2f}" for k in names))
print(f"  {'ALL':<10}" + "".join(
    f"{np.concatenate([i['frames'][:, M63] for i in D[k]]).sum(1).mean():>12.2f}" for k in names))
print(f"  {'expr-Neut':<10}" + "".join(
    f"{np.mean([act[k][e] for e in range(7) if e != 4]) - act[k][4]:>+12.2f}" for k in names))

print("\n  spread between people (mean face size per person):")
pp = {}
for k in names:
    per = {}
    for i in D[k]:
        per.setdefault(i["p"], []).append(i["frames"][:, M63].sum(1).mean())
    pp[k] = {p: np.mean(v) for p, v in per.items()}
    v = np.array(list(pp[k].values()))
    print(f"    {k:<12} min {v.min():.2f}   median {np.median(v):.2f}   max {v.max():.2f}")

# ================================================================ 3. same muscles?
print("\n" + "=" * 90)
print("3. DO THEY USE THE SAME MUSCLES?  correlation of the 63-channel pattern, per emotion")
print("   (1.0 = identical pattern; ignores overall size)")
print("=" * 90)
pat = {k: {e: np.concatenate([i["frames"][:, M63] for i in D[k] if i["e"] == e]).mean(0)
           for e in range(7)} for k in names}
pairs = [(a, b) for ia, a in enumerate(names) for b in names[ia + 1:]]
print(f"  {'emotion':<10}" + "".join(f"{(a[:3] + '-' + b[:3]):>10}" for a, b in pairs))
for e in range(7):
    print(f"  {E[e]:<10}" + "".join(f"{np.corrcoef(pat[a][e], pat[b][e])[0, 1]:>10.2f}"
                                    for a, b in pairs))
print(f"  {'mean':<10}" + "".join(
    f"{np.mean([np.corrcoef(pat[a][e], pat[b][e])[0, 1] for e in range(7)]):>10.2f}"
    for a, b in pairs))
print("  (old=old acted, new=new acted, Emo=EmojiHero, hea=headset)")

# ================================================================ 4. model
print("\n" + "=" * 90)
print("4. HOW THE ACTED-ONLY MODEL READS EACH ONE (window accuracy, raw argmax)")
print("=" * 90)
print(f"  {'emotion':<10}" + "".join(f"{k:>12}" for k in names))
for e in range(7):
    row = f"  {E[e]:<10}"
    for k in names:
        p = np.concatenate([i["pred"] for i in D[k] if i["e"] == e])
        row += f"{np.mean(p == e):>12.2f}"
    print(row)
allacc = {k: np.mean(np.concatenate([i["pred"] == i["e"] for i in D[k]])) for k in names}
print(f"  {'ALL':<10}" + "".join(f"{allacc[k]:>12.2f}" for k in names))
print(f"  {'% Neutral':<10}" + "".join(
    f"{np.mean(np.concatenate([i['pred'] for i in D[k]]) == 4):>12.0%}" for k in names))
print("  (chance = 0.14; old/new acted are mostly training data, so their numbers are flattering)")

# ================================================================ 5. face size vs accuracy
print("\n" + "=" * 90)
print("5. DOES A BIGGER FACE MEAN A BETTER RESULT?  one point per person, all datasets")
print("=" * 90)
xs, ys, src = [], [], []
for k in names:
    per = {}
    for i in D[k]:
        per.setdefault(i["p"], []).append(i["pred"] == i["e"])
    for p, v in per.items():
        xs.append(pp[k][p]); ys.append(np.mean(np.concatenate(v))); src.append(k)
xs, ys, src = np.array(xs), np.array(ys), np.array(src)
print(f"  all {len(xs)} people: correlation {np.corrcoef(xs, ys)[0, 1]:+.2f}")
for k in names:
    m = src == k
    if m.sum() > 2:
        print(f"    within {k:<12} ({m.sum():>2} people): correlation {np.corrcoef(xs[m], ys[m])[0, 1]:+.2f}")
bins = [0, 2, 3, 4, 99]
print("  accuracy by face size, all people pooled:")
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (xs >= lo) & (xs < hi)
    if m.any():
        by = ", ".join(f"{k} {int((src[m] == k).sum())}" for k in names if (src[m] == k).any())
        print(f"    face size {lo}-{hi if hi < 99 else '+'}: {m.sum():>3} people, accuracy {ys[m].mean():.2f}   ({by})")
