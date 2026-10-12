"""Raw / population / personal / hybrid on every session-local-v1 recording, both teachers.

Same functions as `calibration_study.py --enrol-role calibration --test-role evaluation
--compare-all`. The only difference: a session replayed through the model it was NOT recorded
with is allowed here (the CLI refuses it), and is labelled "replayed". That is a deliberate
counterfactual: participants see no prediction, so the model running did not change the face.
"""
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
SP = Path(__file__).parent
sys.path.insert(0, str(REPO / "Analysis"))

import numpy as np
import calibration_study as cs
import session_analysis as sa

ALPHA = 0.40
MODELS = {
    "mixed": (REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model.onnx",
              REPO / "Analysis/results/emojihero_population_centroids.json",
              "sparseVR_teacher_model"),
    "acted_only": (REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx",
                   SP / "pop_acted_only.json",
                   "sparseVR_teacher_model_acted_only"),
}
METHODS = (("argmax_full", "raw"), ("population", "population"),
           ("adapter", "personal"), ("hybrid", "hybrid"))

root = REPO / "Logs/quest-sessions/sessions"
sessions = []
for d in sorted(root.iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if meta.get("protocol_version") == "session-local-v1":
        sessions.append((d, meta["models"]["face_model_asset"]))

cond = cs.Condition()
out = {}
for mname, (mpath, pop_path, asset) in MODELS.items():
    pop_M, _ = cs.load_population_profile(str(pop_path), mpath)
    for d, recorded_with in sessions:
        data = cs.replay(sa.load_session(d / "predictions.csv"), mpath)
        own = recorded_with == asset
        if own and not data.faithful:
            raise SystemExit(f"{d.name}: own-model replay does not match the log")
        enrol, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
        r = cs.evaluate(data, cond, enrol, test,
                        required_ticks=cs.TICKS_REQUIRED_PER_CENTROID,
                        test_role=sa.ROLE_EVALUATION,
                        population_M=pop_M, hybrid_alpha=ALPHA)
        out[(mname, d.name)] = (own, r)

for mname in MODELS:
    print("=" * 96)
    print(f"MODEL {mname}   (trial-majority on the 14 evaluation trials; accuracy / macro-F1)")
    print("=" * 96)
    print(f"  {'session':<10}{'source':<10}" + "".join(f"{lab:>16}" for _, lab in METHODS)
          + f"{'hyb-pers':>10}")
    accs = {lab: [] for _, lab in METHODS}
    f1s = {lab: [] for _, lab in METHODS}
    for d, _ in sessions:
        own, r = out[(mname, d.name)]
        tm = r["trial_majority"]
        line = f"  {d.name[-6:]:<10}{'recorded' if own else 'replayed':<10}"
        for key, lab in METHODS:
            c = tm[key]
            accs[lab].append(c["accuracy"]); f1s[lab].append(c["macro_f1"])
            line += f"{c['accuracy']:>9.3f}/{c['macro_f1']:.3f}"
        line += f"{tm['hybrid']['macro_f1'] - tm['adapter']['macro_f1']:>+10.3f}"
        if not r["calibration_valid"]:
            line += "  (incomplete calibration)"
        print(line)
    line = f"  {'mean':<20}"
    for _, lab in METHODS:
        line += f"{np.mean(accs[lab]):>9.3f}/{np.mean(f1s[lab]):.3f}"
    print(line)
    for a, b in (("personal", "raw"), ("hybrid", "raw"), ("hybrid", "personal"),
                 ("population", "raw")):
        dlt = np.array(f1s[a]) - np.array(f1s[b])
        print(f"    {a} - {b:<10} macro-F1 mean {dlt.mean():+.3f}   "
              f"better {int((dlt > 0).sum())}, worse {int((dlt < 0).sum())}, "
              f"tied {int((dlt == 0).sum())} of {len(dlt)}")
    print()
