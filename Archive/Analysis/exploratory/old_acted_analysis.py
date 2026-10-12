"""Old 2025 recordings (filtered-dataset): inventory, quality, expression intensity, and how the
acted-only teacher reads them. ACT_* = acted ~10 s; weights_* = natural reaction (kept only for
emotions the participant reported feeling, hence 'filtered'). Read-only."""
import csv
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import onnxruntime as ort
import au_names
import calibration_study as cs
import emojihero_calibration_study as eh
import session_analysis as sa

ROOT = REPO / "Analysis/data/filtered-dataset-20260924T162300Z-1-001/filtered-dataset"
E = list(eh.EMOTIONS)
MODEL = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
rn = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
INP = rn.get_inputs()[0].name
M63 = slice(0, 63)


def load(path):
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    head = rows[0]
    assert tuple(head[1:]) == tuple(au_names.NAMES), f"{path}: unexpected columns"
    body = [r for r in rows[1:] if r and r[0] != ""]
    good = [r for r in body if len(r) == len(head)]
    BAD_ROWS.append((path, len(body) - len(good), len(body)))
    data = np.array([[float(v) for v in r] for r in good])
    return data[:, 0], data[:, 1:]


BAD_ROWS = []


rec = {"ACT": [], "NAT": []}
for folder in sorted(ROOT.iterdir()):
    for f in sorted(folder.glob("*.csv")):
        kind = "ACT" if f.name.startswith("ACT_") else "NAT" if f.name.startswith("weights_") else None
        emo = f.stem.split("_", 1)[1].split("(")[0] if kind else None
        if kind is None or emo not in E:
            print(f"  skipped: {folder.name}/{f.name}")
            continue
        t, au = load(f)
        rec[kind].append({"p": folder.name, "e": E.index(emo), "t": t, "au": au})

# ------------------------------------------------------------------ 1. inventory
print("=" * 84)
print("1. WHAT IS IN THE FOLDER")
print("=" * 84)
for k, lab in (("ACT", "acted"), ("NAT", "natural (weights_)")):
    ps = {r["p"] for r in rec[k]}
    print(f"  {lab:<20} {len(rec[k]):>4} recordings from {len(ps)} folders")
    print("    per emotion: " + "  ".join(f"{E[e][:4]} {sum(r['e'] == e for r in rec[k])}"
                                         for e in range(7)))
act_p = {r["p"] for r in rec["ACT"]}
nat_p = {r["p"] for r in rec["NAT"]}
print(f"  folders with acted: {len(act_p)}, natural: {len(nat_p)}, both: {len(act_p & nat_p)}")
print("  NOTE: a folder is a recording session; whether two folders are the same person is unknown.")

# ------------------------------------------------------------------ 2. quality
print("\n" + "=" * 84)
print("2. RECORDING QUALITY")
print("=" * 84)
for k, lab in (("ACT", "acted"), ("NAT", "natural")):
    durs = [r["t"][-1] - r["t"][0] for r in rec[k]]
    rates = [(len(r["t"]) - 1) / (r["t"][-1] - r["t"][0]) for r in rec[k]]
    gaps = [np.diff(r["t"]) for r in rec[k]]
    big = np.mean([np.mean(g > 0.15) for g in gaps])
    zero = np.mean([np.mean(r["au"][:, M63].sum(1) < 1e-3) for r in rec[k]])
    frozen = np.mean([np.mean(np.all(np.diff(r["au"], axis=0) == 0, axis=1)) for r in rec[k]])
    print(f"  {lab:<8} duration median {np.median(durs):5.1f} s   frame rate median {np.median(rates):4.1f}/s"
          f"   frames after a gap >0.15 s {big:5.1%}")
    print(f"  {'':<8} all-zero frames (tracking lost) {zero:5.1%}   repeated identical frames {frozen:5.1%}")
broken = [(p, b, n) for p, b, n in BAD_ROWS if b]
print(f"  files with malformed rows (wrong column count, skipped): {len(broken)} of {len(BAD_ROWS)}"
      f"; {sum(b for _, b, _ in broken)} rows of {sum(n for _, _, n in BAD_ROWS)}")
for p, b, n in broken[:5]:
    print(f"    {p.parent.name}/{p.name}: {b} of {n}")
allf = np.concatenate([r["au"] for r in rec["ACT"] + rec["NAT"]])
dead = [au_names.name(i) for i in range(63) if allf[:, i].max() < 1e-6]
print(f"  channels that never move (of 63): {len(dead)} {dead if dead else ''}")
print(f"  tongue channels: max {allf[:, 63:].max():.3f}")

# ------------------------------------------------------------------ 3. intensity
print("\n" + "=" * 84)
print("3. BIG FACES OR SMALL FACES?  mean sum of 63 channels per frame")
print("=" * 84)
# comparison sets
eh_seqs = eh.load_split(REPO / "Analysis/data", "validation")[0] + \
          eh.load_split(REPO / "Analysis/data", "test")[0]
eh_raw = {e: np.concatenate([s.au[:, :63] for s in eh_seqs if s.emotion == e]) for e in range(7)}
hs = []
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if meta.get("protocol_version") == "session-local-v1":
        r = cs.replay(sa.load_session(d / "predictions.csv"), MODEL)
        keep = np.isin(r.tick, list(cs.SCORED_TICKS))
        hs.append((r.windows[keep], r.prompted[keep]))
hs_w = np.concatenate([w for w, _ in hs]); hs_y = np.concatenate([y for _, y in hs])


def act(k, e):
    return np.concatenate([r["au"][:, M63] for r in rec[k] if r["e"] == e]).sum(1).mean() \
        if any(r["e"] == e for r in rec[k]) else float("nan")


print(f"  {'emotion':<10}{'old acted':>10}{'old natural':>12}{'headset':>9}{'EmojiHero':>11}")
for e in range(7):
    print(f"  {E[e]:<10}{act('ACT', e):>10.2f}{act('NAT', e):>12.2f}"
          f"{hs_w[hs_y == e][..., M63].sum(-1).mean():>9.2f}{eh_raw[e].sum(1).mean():>11.2f}")
tot = lambda k: np.concatenate([r["au"][:, M63] for r in rec[k]]).sum(1).mean()
print(f"  {'all':<10}{tot('ACT'):>10.2f}{tot('NAT'):>12.2f}{hs_w[..., M63].sum(-1).mean():>9.2f}"
      f"{np.concatenate(list(eh_raw.values())).sum(1).mean():>11.2f}")
print("  (EmojiHero here is raw frames, not pooled, so it is directly comparable to the others)")

# spread between people inside the old acted data
per_p = {}
for r in rec["ACT"]:
    per_p.setdefault(r["p"], []).append(r["au"][:, M63].sum(1).mean())
pp = np.array([np.mean(v) for v in per_p.values()])
print(f"\n  old acted, activity per folder: min {pp.min():.2f}  median {np.median(pp):.2f}"
      f"  max {pp.max():.2f}   (max/min x{pp.max() / pp.min():.1f})")
print(f"  folders below the headset average ({hs_w[..., M63].sum(-1).mean():.2f}): "
      f"{int((pp < hs_w[..., M63].sum(-1).mean()).sum())} of {len(pp)}")

# ------------------------------------------------------------------ 4. model
print("\n" + "=" * 84)
print("4. HOW THE ACTED-ONLY MODEL READS THE OLD ACTED DATA")
print("=" * 84)
print("  Windows: 5 consecutive frames, non-overlapping, no gate (runtime-like).")
print("  CAUTION: old acted data is in the acted-only model's TRAINING set (split unknown here),")
print("  so these numbers are partly memorised, not a fair test.\n")
per_rec = []
for r in rec["ACT"]:
    au = r["au"].astype(np.float32)
    n = len(au) // 5
    w = au[: n * 5].reshape(n, 5, 70)
    pred = rn.run(None, {INP: w})[0].argmax(1)
    per_rec.append((r["p"], r["e"], float(np.mean(pred == r["e"])), au[:, M63].sum(1).mean(), pred))
print(f"  {'emotion':<10}{'window acc':>11}   most often predicted as")
for e in range(7):
    rows = [x for x in per_rec if x[1] == e]
    preds = np.concatenate([x[4] for x in rows])
    top = np.bincount(preds, minlength=7)
    order = [c for c in np.argsort(-top) if c != e][:1]
    print(f"  {E[e]:<10}{np.mean(preds == e):>11.2f}   {E[order[0]]} ({top[order[0]] / len(preds):.0%})")
allp = np.concatenate([x[4] for x in per_rec]); ally = np.concatenate([[x[1]] * len(x[4]) for x in per_rec])
print(f"  {'all':<10}{np.mean(allp == ally):>11.2f}")

fa = {}
for p, e, a, i, _ in per_rec:
    fa.setdefault(p, ([], []))[0].append(a); fa[p][1].append(i)
acc_p = np.array([np.mean(v[0]) for v in fa.values()]); int_p = np.array([np.mean(v[1]) for v in fa.values()])
print(f"\n  does the model do better on people with bigger faces?  "
      f"correlation(activity, accuracy) across {len(acc_p)} folders = {np.corrcoef(int_p, acc_p)[0, 1]:+.2f}")
lo = acc_p[int_p <= np.median(int_p)].mean(); hi = acc_p[int_p > np.median(int_p)].mean()
print(f"  smaller-face half: accuracy {lo:.2f}   bigger-face half: accuracy {hi:.2f}")
