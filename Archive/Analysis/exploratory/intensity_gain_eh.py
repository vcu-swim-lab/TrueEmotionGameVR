"""Same gain on EmojiHero validation+test people (who already exaggerate). Raw model only.
auto g per person = EmojiHero mean activity / this person's mean activity (label-free)."""
import sys
from pathlib import Path
REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np, onnxruntime as ort
import emojihero_calibration_study as eh
MODEL = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
rn = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"]); inp = rn.get_inputs()[0].name
seqs = eh.load_split(REPO/"Analysis/data","validation")[0] + eh.load_split(REPO/"Analysis/data","test")[0]
W = np.stack([eh.to_model_windows(s.au) for s in seqs]); y = np.array([s.emotion for s in seqs])
P = np.array([s.participant for s in seqs]); ref = W[..., :63].sum(-1).mean()
def f1_person(g_of):
    out = []
    for p in sorted(set(P)):
        m = P == p; g = g_of(p, m)
        x = np.clip(W[m] * g, 0, 1).astype(np.float32).reshape(-1, 5, 70)
        pr = rn.run(None, {inp: x})[0].argmax(1).reshape(-1, 6)
        out.append(eh.fast_macro_f1(y[m], eh.fast_majority_vote(pr)))
    return np.array(out)
base = f1_person(lambda p, m: 1.0)
autos = {p: ref / W[P == p][..., :63].sum(-1).mean() for p in set(P)}
print(f"auto g per EmojiHero person: min {min(autos.values()):.2f}  max {max(autos.values()):.2f}")
for lab, fn in (("g = 1.0", lambda p, m: 1.0), ("g = 1.5", lambda p, m: 1.5),
                ("g = 2.0", lambda p, m: 2.0), ("auto g", lambda p, m: autos[p])):
    v = f1_person(fn); d = v - base
    print(f"{lab:<10} mean macro-F1 {v.mean():.3f}   better {int((d>0).sum())}, worse {int((d<0).sum())} of {len(v)}")
