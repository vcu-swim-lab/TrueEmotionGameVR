"""The notebook keeps an expressive window only if >=3 consecutive frames have some AU > 0.25,
and a Neutral window only if every AU <= 0.25. How many headset ticks would that gate have
thrown away, and what does the model say on them?"""
import json, sys
from pathlib import Path
REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import calibration_study as cs, session_analysis as sa
MODEL = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
E = ["Anger","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]
W, Y, P = [], [], []
for d in sorted((REPO/"Logs/quest-sessions/sessions").iterdir()):
    try: meta = json.loads((d/"session_metadata.json").read_text(encoding="utf-8"))
    except Exception: continue
    if meta.get("protocol_version") != "session-local-v1": continue
    r = cs.replay(sa.load_session(d/"predictions.csv"), MODEL)
    k = np.isin(r.tick, list(cs.SCORED_TICKS))
    W.append(r.windows[k]); Y.append(r.prompted[k]); P.append(r.scores[k].argmax(1))
W, Y, P = np.concatenate(W), np.concatenate(Y), np.concatenate(P)
active = (W > 0.25).any(-1)                       # frame has some AU > 0.25
def run3(a):
    c = 0
    for v in a:
        c = c + 1 if v else 0
        if c >= 3: return True
    return False
passes = np.array([run3(a) for a in active])
print("expressive prompts (not Neutral):")
m = Y != 4
print(f"  ticks that would FAIL the training gate (face too small): {np.mean(~passes[m]):.0%}")
print(f"  model says Neutral on those small ticks: {np.mean(P[m & ~passes] == 4):.0%}")
print(f"  model says Neutral on ticks that pass:   {np.mean(P[m & passes] == 4):.0%}")
print(f"  accuracy on small ticks {np.mean(P[m & ~passes] == Y[m & ~passes]):.0%}   "
      f"on passing ticks {np.mean(P[m & passes] == Y[m & passes]):.0%}")
print("per emotion, share of ticks too small for the gate:")
for e in range(7):
    if e == 4: continue
    s = Y == e
    print(f"  {E[e]:<10} {np.mean(~passes[s]):.0%}")
