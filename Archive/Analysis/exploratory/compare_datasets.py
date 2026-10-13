"""EmojiHero (validation + test people) vs headset session-local-v1 sessions. Read-only.

Three levels, kept apart:
  A. input    the 5x70 windows the model is given
  B. output   what the model says
  C. geometry the centroids the four calibration methods are built from
"""
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR")
SP = Path(__file__).parent
sys.path.insert(0, str(REPO / "Analysis"))

import numpy as np
import au_names
import calibration_study as cs
import emojihero_calibration_study as eh
import session_analysis as sa

E = eh.EMOTIONS
MODELS = {
    "mixed": (REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model.onnx",
              REPO / "Analysis/results/emojihero_population_centroids.json"),
    "acted_only": (REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx",
                   SP / "pop_acted_only.json"),
}


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ------------------------------------------------------------------ load
loaded, prov = {}, {}
for split in eh.SPLITS:
    loaded[split], prov[split] = eh.load_split(REPO / "Analysis/data", split)
eh.check_corpus(loaded, prov)
eh_seqs = loaded["validation"] + loaded["test"]          # people the models never trained on
eh_win = np.stack([eh.to_model_windows(s.au) for s in eh_seqs])   # [n, 6, 5, 70]

sessions = []
for d in sorted((REPO / "Logs/quest-sessions/sessions").iterdir()):
    try:
        meta = json.loads((d / "session_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        continue
    if meta.get("protocol_version") == "session-local-v1":
        sessions.append(sa.load_session(d / "predictions.csv"))

# Headset windows: ticks 1-9 only (tick 0 is a transition sample, dropped everywhere).
first = cs.replay(sessions[0], MODELS["mixed"][0])
hs = []
for s in sessions:
    r = cs.replay(s, MODELS["mixed"][0])
    keep = np.isin(r.tick, list(cs.SCORED_TICKS))
    hs.append({"name": s.session_id[-6:], "session": s, "keep": keep,
               "win": r.windows[keep], "y": r.prompted[keep], "role": r.role[keep]})
hs_win = np.concatenate([h["win"] for h in hs])
hs_y = np.concatenate([h["y"] for h in hs])
eh_y_seq = np.array([s.emotion for s in eh_seqs])
assert list(first.classes) == list(E)

print("=" * 88)
print(f"EmojiHero: {len(eh_seqs)} sequences, "
      f"{len({(s.participant) for s in eh_seqs})} people (validation+test)")
print(f"Headset:   {len(hs)} sessions, {len(hs_win)} scored ticks")
print("=" * 88)

# ============================================================ A. input
print("\nA. INPUT -- the 5x70 windows the model sees\n")

eh_frames = eh_win.reshape(-1, 70)
hs_frames = hs_win.reshape(-1, 70)
measured = list(range(63))            # EmojiHero has no tongue; compare on the shared 63

eh_mean, hs_mean = eh_frames[:, measured].mean(0), hs_frames[:, measured].mean(0)
print(f"  overall activity (sum of 63 channels per frame): "
      f"EmojiHero {eh_frames[:, measured].sum(1).mean():.2f}   "
      f"headset {hs_frames[:, measured].sum(1).mean():.2f}")
print(f"  channel profile similarity (corr of the 63 per-channel means): "
      f"{np.corrcoef(eh_mean, hs_mean)[0, 1]:.3f}")
eh_dead = [i for i in measured if eh_frames[:, i].max() < 1e-6]
hs_dead = [i for i in measured if hs_frames[:, i].max() < 1e-6]
print(f"  never-moving channels among the 63: EmojiHero {len(eh_dead)}, headset {len(hs_dead)}")
if hs_dead:
    print(f"    headset-only dead: {', '.join(au_names.name(i) for i in hs_dead if i not in eh_dead)}")

ratio = (hs_mean + 1e-3) / (eh_mean + 1e-3)
order = np.argsort(ratio)
print("  biggest per-channel differences (headset mean / EmojiHero mean):")
for i in list(order[:5]) + list(order[-5:]):
    print(f"    {au_names.name(i):<28} EH {eh_mean[i]:.3f}  HS {hs_mean[i]:.3f}  x{ratio[i]:.2f}")

# Smoothness inside a window: EmojiHero frames are means of 3 raw frames; headset frames are
# instantaneous samples.
def jitter(w):
    return np.abs(np.diff(w[..., measured], axis=-2)).sum(-1).mean()
print(f"  frame-to-frame change inside a window (sum |diff|): "
      f"EmojiHero {jitter(eh_win):.3f}   headset {jitter(hs_win):.3f}")

print("\n  activity by prompted emotion (sum of 63 channels, mean over frames):")
print(f"    {'emotion':<10}{'EmojiHero':>10}{'headset':>10}")
eh_act = eh_win[..., measured].sum(-1).mean((1, 2))
hs_act = hs_win[..., measured].sum(-1).mean(1)
for e in range(7):
    print(f"    {E[e]:<10}{eh_act[eh_y_seq == e].mean():>10.2f}{hs_act[hs_y == e].mean():>10.2f}")
eh_range = eh_act[eh_y_seq != 4].mean() - eh_act[eh_y_seq == 4].mean()
hs_range = hs_act[hs_y != 4].mean() - hs_act[hs_y == 4].mean()
print(f"    expressive minus Neutral: EmojiHero {eh_range:+.2f}   headset {hs_range:+.2f}")

# ============================================================ B + C per model
for mname, (mpath, pop_path) in MODELS.items():
    print("\n" + "=" * 88)
    print(f"MODEL {mname}")
    print("=" * 88)
    G, _ = cs.load_population_profile(str(pop_path), mpath)

    eh_sc = eh.run_model(eh_seqs, mpath)                      # [n, 6, 7]
    eh_lf = cs.logscore(eh_sc, cs.DEFAULT_EPSILON)
    for h in hs:
        r = cs.replay(h["session"], mpath)
        h["sc"] = r.scores[h["keep"]].astype(np.float64)
        h["lf"] = cs.logscore(h["sc"], cs.DEFAULT_EPSILON)
    hs_sc = np.concatenate([h["sc"] for h in hs])

    # ---------------------------------------------------------- B. output
    print("\nB. OUTPUT -- what the model says (window/tick level)\n")
    eh_w_sc = eh_sc.reshape(-1, 7)
    eh_w_y = np.repeat(eh_y_seq, 6)
    print(f"  {'':<24}{'EmojiHero':>10}{'headset':>10}")
    print(f"  {'raw accuracy':<24}{np.mean(eh_w_sc.argmax(1) == eh_w_y):>10.3f}"
          f"{np.mean(hs_sc.argmax(1) == hs_y):>10.3f}")
    print(f"  {'top score (confidence)':<24}{eh_w_sc.max(1).mean():>10.3f}{hs_sc.max(1).mean():>10.3f}")
    print(f"  {'p(prompted class)':<24}{eh_w_sc[np.arange(len(eh_w_y)), eh_w_y].mean():>10.3f}"
          f"{hs_sc[np.arange(len(hs_y)), hs_y].mean():>10.3f}")
    print("\n  recall by emotion, and share of all predictions going to each class:")
    print(f"    {'emotion':<10}{'recall EH':>10}{'recall HS':>10}{'pred% EH':>10}{'pred% HS':>10}")
    for e in range(7):
        print(f"    {E[e]:<10}{np.mean(eh_w_sc.argmax(1)[eh_w_y == e] == e):>10.2f}"
              f"{np.mean(hs_sc.argmax(1)[hs_y == e] == e):>10.2f}"
              f"{np.mean(eh_w_sc.argmax(1) == e):>10.2f}{np.mean(hs_sc.argmax(1) == e):>10.2f}")

    # ---------------------------------------------------------- C. geometry
    # Per person: K = the calibration material, D = the scored material, per emotion.
    #   EmojiHero: K = one sequence (primary profile), D = that person's other sequences.
    #   Headset:   K = calibration block ticks, D = evaluation block ticks.
    units = []
    by_cell = eh._sequences_by_cell(eh_seqs)
    idx = {s.sequence_id: i for i, s in enumerate(eh_seqs)}
    split_of = {s.sequence_id: sp for sp in ("validation", "test") for s in loaded[sp]}
    for p in sorted({s.participant for s in eh_seqs}):
        if any(len(by_cell.get((p, e), [])) < 2 for e in range(7)):
            continue
        sp = split_of[by_cell[(p, 0)][0].sequence_id]
        prof = eh.calibration_profile(by_cell, sp, p, 0)
        K = np.stack([eh_lf[idx[prof[e].sequence_id]].mean(0) for e in range(7)])
        D = np.stack([np.concatenate([eh_lf[idx[s.sequence_id]] for s in by_cell[(p, e)]
                                      if s is not prof[e]]).mean(0) for e in range(7)])
        units.append(("EH", K, D, 6))
    for h in hs:
        K = np.stack([h["lf"][(h["y"] == e) & (h["role"] == sa.ROLE_CALIBRATION)].mean(0)
                      for e in range(7)])
        D = np.stack([h["lf"][(h["y"] == e) & (h["role"] == sa.ROLE_EVALUATION)].mean(0)
                      for e in range(7)])
        units.append(("HS", K, D, 9))

    def summarise(src):
        rows = [u for u in units if u[0] == src]
        stay = np.mean([np.mean([cos(K[e], D[e]) for e in range(7)]) for _, K, D, _ in rows])
        # does the person's own K pick the right class for their D centroids?
        k_hit = np.mean([np.mean([np.argmax([cos(D[e], K[c]) for c in range(7)]) == e
                                  for e in range(7)]) for _, K, D, _ in rows])
        # does the population pick the right class for their D centroids?
        g_hit = np.mean([np.mean([np.argmax([cos(D[e], G[c]) for c in range(7)]) == e
                                  for e in range(7)]) for _, K, D, _ in rows])
        near_pop = np.mean([np.mean([cos(K[e], G[e]) for e in range(7)]) for _, K, D, _ in rows])
        d_pop = np.mean([np.mean([cos(D[e], G[e]) for e in range(7)]) for _, K, D, _ in rows])
        sep = np.mean([np.mean([cos(K[a], K[b]) for a in range(7) for b in range(a + 1, 7)])
                       for _, K, D, _ in rows])
        norm = np.mean([np.mean(np.linalg.norm(K, axis=1)) for _, K, D, _ in rows])
        return len(rows), stay, k_hit, g_hit, near_pop, d_pop, sep, norm

    print("\nC. GEOMETRY -- what calibration is built from (log-score centroids)\n")
    print("   K = calibration material, D = scored material, G = EmojiHero population")
    print(f"   {'':<50}{'EmojiHero':>10}{'headset':>10}")
    a, b = summarise("EH"), summarise("HS")
    labels = [
        ("units (people / sessions)", 0, "d"),
        ("K->D stability: cos(K_e, D_e), same emotion", 1, ".4f"),
        ("personal works: K picks right class for D_e", 2, ".3f"),
        ("population works: G picks right class for D_e", 3, ".3f"),
        ("K close to population: cos(K_e, G_e)", 4, ".4f"),
        ("D close to population: cos(D_e, G_e)", 5, ".4f"),
        ("K classes blur together: mean cos(K_a, K_b)", 6, ".4f"),
        ("K size (norm; the hybrid blends raw vectors)", 7, ".2f"),
    ]
    for lab, i, f in labels:
        print(f"   {lab:<50}{a[i]:>10{f}}{b[i]:>10{f}}")
    print(f"   {'population G size (norm)':<50}{np.mean(np.linalg.norm(G, axis=1)):>10.2f}")

    print("\n   per headset session:")
    print(f"   {'session':<10}{'K->D':>8}{'pers hit':>10}{'pop hit':>9}{'cos(D,G)':>10}")
    for h, u in zip(hs, [u for u in units if u[0] == "HS"]):
        _, K, D, _ = u
        print(f"   {h['name']:<10}{np.mean([cos(K[e], D[e]) for e in range(7)]):>8.4f}"
              f"{np.mean([np.argmax([cos(D[e], K[c]) for c in range(7)]) == e for e in range(7)]):>10.2f}"
              f"{np.mean([np.argmax([cos(D[e], G[c]) for c in range(7)]) == e for e in range(7)]):>9.2f}"
              f"{np.mean([cos(D[e], G[e]) for e in range(7)]):>10.4f}")
print()
