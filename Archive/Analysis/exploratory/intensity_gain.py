"""Does amplifying small faces help? Acted-only teacher, 6 session-local-v1 sessions.

gain g multiplies every action-unit value (clipped to [0, 1]) before the model runs.
  fixed g       the same for every session: 1.0 (unchanged), 1.5, 2.0, 3.0
  auto g        per session, label-free: EmojiHero mean activity / this session's mean
                activity over its CALIBRATION block (no labels used, eval block untouched)
Each is scored raw (argmax) and with personal calibration (centroids re-fitted on the
amplified calibration block). Trial-majority macro-F1 on the evaluation block.
"""
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))

import numpy as np
import onnxruntime as ort
import calibration_study as cs
import emojihero_calibration_study as eh
import session_analysis as sa

MODEL = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
E = list(eh.EMOTIONS)
M63 = slice(0, 63)
runner = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
inp = runner.get_inputs()[0].name

# EmojiHero reference activity: validation+test people, model-input windows, 63 channels.
eh_seqs = eh.load_split(REPO / "Analysis/data", "validation")[0] + \
          eh.load_split(REPO / "Analysis/data", "test")[0]
eh_win = np.stack([eh.to_model_windows(s.au) for s in eh_seqs])
EH_ACT = float(eh_win[..., M63].sum(-1).mean())

S = []
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if meta.get("protocol_version") != "session-local-v1":
        continue
    r = cs.replay(sa.load_session(d / "predictions.csv"), MODEL)
    keep = np.isin(r.tick, list(cs.SCORED_TICKS))
    S.append({"name": d.name[-6:], "win": r.windows[keep], "y": r.prompted[keep],
              "role": r.role[keep], "trial": r.trial[keep], "tick": r.tick[keep]})


def run(win, g):
    x = np.clip(win * g, 0.0, 1.0).astype(np.float32)
    x[..., 63:] = win[..., 63:]          # tongue channels untouched
    return runner.run(None, {inp: x})[0].astype(np.float64)


def f1(s, pred_ev):
    ev = s["role"] == sa.ROLE_EVALUATION
    names = np.array(E)
    _, ty, tp = cs.trial_vote(names[s["y"][ev]], names[pred_ev], s["trial"][ev], s["tick"][ev], E)
    ty = np.array([E.index(v) for v in ty]); tp = np.array([E.index(v) for v in tp])
    return eh.fast_macro_f1(ty, tp), float(np.mean(ty == tp))


def evaluate(s, g):
    sc = run(s["win"], g)
    lf = cs.logscore(sc)
    ev, cal = s["role"] == sa.ROLE_EVALUATION, s["role"] == sa.ROLE_CALIBRATION
    K = np.stack([lf[cal & (s["y"] == c)].mean(0) for c in range(7)])
    raw = f1(s, sc[ev].argmax(1))
    pers = f1(s, cs.classify(K, list(range(7)), lf[ev], "cosine")[0])
    return raw, pers


print(f"EmojiHero mean activity per frame: {EH_ACT:.2f}\n")
print(f"{'session':<9}{'activity':>9}{'auto g':>8}")
for s in S:
    cal = s["role"] == sa.ROLE_CALIBRATION
    s["act"] = float(s["win"][cal][..., M63].sum(-1).mean())
    s["auto"] = EH_ACT / s["act"]
    print(f"{s['name']:<9}{s['act']:>9.2f}{s['auto']:>8.2f}")
print()

conds = [("g = 1.0 (unchanged)", lambda s: 1.0), ("g = 1.5", lambda s: 1.5),
         ("g = 2.0", lambda s: 2.0), ("g = 3.0", lambda s: 3.0),
         ("auto g (per session)", lambda s: s["auto"])]
res = {lab: [evaluate(s, fn(s)) for s in S] for lab, fn in conds}

for which, j in (("RAW model", 0), ("PERSONAL calibration", 1)):
    print("=" * 92)
    print(f"{which} -- trial-majority macro-F1 on the evaluation block")
    print("=" * 92)
    print(f"  {'condition':<24}" + "".join(f"{s['name']:>9}" for s in S) + f"{'mean':>9}{'vs g=1':>10}")
    base = np.array([r[j][0] for r in res["g = 1.0 (unchanged)"]])
    for lab, _ in conds:
        v = np.array([r[j][0] for r in res[lab]])
        d = v - base
        print(f"  {lab:<24}" + "".join(f"{x:>9.3f}" for x in v) + f"{v.mean():>9.3f}"
              f"{f'{int((d > 0).sum())}+/{int((d < 0).sum())}-':>10}")
    print()
