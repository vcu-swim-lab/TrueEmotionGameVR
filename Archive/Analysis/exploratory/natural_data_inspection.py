"""Can the natural (video-elicited) recordings serve as UNLABELLED data for the acted teacher?

Inspection only. No training, no labels from natural data.

Sources
  old study  Analysis/data/filtered-dataset-.../<participant>/ACT_<E>.csv (acted), weights_<E>.csv (natural)
             The notebook reads these as they are (first column = time, no downsampling).
  new study  multimodal-vr-emotion-study-main/ActingRecordings/<p>/<E>/*.csv (acted)
             multimodal-vr-emotion-study-main/VideoRecordings/<p>/<E>/*.csv (natural)
             The notebook downsamples by mean of 6 rows (60 Hz -> 10 Hz) after ffill/bfill.

Checks
  1. Columns: all 70 AU names present, extra columns (FaceValid, time), value range.
  2. Frame rate from the time column; missing values; tracking failures (all-zero rows, FaceValid).
  3. Participant split: old-study val/test participants found by EXACT matching of the npz windows
     to windows rebuilt from each participant's ACT files; new study from acted_splits.csv.
  4. Movement: per frame, the strongest expressive AU (gaze, blink, tongue left out) and face size,
     for old/new acted, old/new natural and headset ticks. Natural windows available for training
     (non-overlapping 5-frame windows inside one recording, train participants only).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
sys.path.insert(0, str(REPO / "Analysis/exploratory"))
import numpy as np
import pandas as pd

import teacher_preprocessing as tp

Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
AU = [str(a) for a in Z["au_names"]]
EXPR = np.array([not a.startswith(("EyesLook", "EyesClosed", "Tongue")) for a in AU])
OLD = REPO / "Analysis/data/filtered-dataset-20260924T162300Z-1-001/filtered-dataset"
NEW = Path(r"C:\Users\alime\multimodal-vr-emotion-study\multimodal-vr-emotion-study-main")
E = [str(c) for c in Z["class_names"]]


def read_old(path):
    df = pd.read_csv(path)
    t = df.iloc[:, 0].to_numpy(float)
    missing = [a for a in AU if a not in df.columns]
    extra = [c for c in df.columns[1:] if c not in AU]
    X = df[AU].to_numpy(np.float64) if not missing else None
    return t, X, missing, extra


def read_new(path):
    df = pd.read_csv(path)
    missing = [a for a in AU if a not in df.columns]
    extra = [c for c in df.columns if c not in AU]
    raw = df[AU].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    nan_share = float(np.isnan(raw).mean())
    valid = df["FaceValid"].to_numpy() if "FaceValid" in df.columns else None
    tcol = next((c for c in df.columns if c.lower() in ("time", "timestamp", "t", "unitytime", "elapsed")), None)
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float) if tcol else None
    X = tp.downsample(tp.clean(raw)) if hasattr(tp, "clean") else None
    return t, raw, X, missing, extra, nan_share, valid, tcol


def frame_stats(name, frames, rows):
    peak = frames[:, EXPR].max(1)
    size = frames[:, :63].sum(1)
    zero = (frames.max(1) == 0)
    rows.append({"data": name, "frames": len(frames), "all-zero frames": zero.mean(),
                 "median face size": np.median(size), "expressive AU > 0.25": (peak > 0.25).mean(),
                 "expressive AU > 0.10": (peak > 0.10).mean(), "median expressive peak": np.median(peak)})


def main():
    pd.set_option("display.width", 250)
    rows = []

    # ---------------- 1-2 old study
    print("=" * 78 + "\nOLD STUDY\n" + "=" * 78)
    parts = sorted(p for p in OLD.iterdir() if p.is_dir())
    act_frames, nat_frames, rates, bad = [], [], [], []
    old_act_windows = {}
    nat_per_part = {}
    for p in parts:
        for f in sorted(p.glob("*.csv")):
            kind = "act" if f.name.startswith("ACT_") else "nat" if f.name.startswith("weights_") else None
            if kind is None:
                continue
            t, X, missing, extra = read_old(f)
            if missing:
                bad.append((f.name, p.name, missing[:3]))
                continue
            if len(t) > 2:
                rates.append((kind, 1 / np.median(np.diff(t))))
            if np.isnan(X).any():
                bad.append((f.name, p.name, f"NaN share {np.isnan(X).mean():.3f}"))
            X = np.nan_to_num(X)
            (act_frames if kind == "act" else nat_frames).append(X)
            emo = f.stem.split("_", 1)[1].split("(")[0]
            if kind == "act":
                old_act_windows.setdefault(p.name, []).append(tp.make_windows(X.astype(np.float32), emo))
            else:
                nat_per_part.setdefault(p.name, []).append(X)
    print(f"participants {len(parts)} | files with problems: {len(bad)}")
    for b in bad[:8]:
        print("   ", b)
    r = pd.DataFrame(rates, columns=["kind", "Hz"])
    print("frame rate from the time column (median per file):", r.groupby("kind").Hz.describe()[["count", "50%", "min", "max"]].round(2).to_dict("index"))
    print("extra columns in a natural file:", read_old(next(OLD.glob("*/weights_Anger.csv")))[3][:6])
    frame_stats("old acted", np.concatenate(act_frames), rows)
    frame_stats("old natural", np.concatenate(nat_frames), rows)

    # ---------------- 3 old split by exact window matching
    key = lambda w: np.round(w, 5).tobytes()
    owner = {}
    for p, ws in old_act_windows.items():
        for w in np.concatenate([x for x in ws if len(x)]) if any(len(x) for x in ws) else []:
            owner.setdefault(key(w), set()).add(p)
    split_of = {}
    for sp in ("train", "val", "test"):
        X = Z["X_train"][Z["S_train"] == 0] if sp == "train" else Z[f"X_{sp}_old"]
        found = [owner.get(key(w), set()) for w in X]
        hit = np.mean([len(f) == 1 for f in found])
        people = sorted(set().union(*found))
        print(f"old acted {sp:<5}: {len(X)} npz windows, matched to exactly one participant: {hit:.0%} -> "
              f"{len(people)} participants")
        for q in people:
            split_of.setdefault(q, set()).add(sp)
    leaks = {p: s for p, s in split_of.items() if len(s) > 1}
    print("participants in more than one split:", leaks or "none")
    old_eval = sorted(p for p, s in split_of.items() if s & {"val", "test"})
    print("old val/test participants:", old_eval)
    no_acted = [p.name for p in parts if p.name not in split_of]
    print("old participants with no matched acted window (split unknown):", no_acted)

    # ---------------- 1-2 new study
    print("\n" + "=" * 78 + "\nNEW STUDY\n" + "=" * 78)
    splits = pd.read_csv(REPO / "Analysis/data/acted_splits.csv", comment="#")
    split_new = dict(zip(splits.participant, splits.split))
    files = sorted((NEW / "VideoRecordings").glob("*/*/*.csv"))
    t, raw, X, missing, extra, nan_share, valid, tcol = read_new(files[0])
    print("natural file columns: missing AU", missing, "| extra", extra[:12], "| time column", tcol)
    new_nat, new_nat_per_part, info = [], {}, []
    for f in files:
        t, raw, X, missing, extra, nan_share, valid, tcol = read_new(f)
        hz = 1 / np.median(np.diff(t)) if t is not None and len(t) > 2 else np.nan
        info.append({"file": f, "Hz": hz, "nan": nan_share,
                     "FaceValid=0": float((valid == 0).mean()) if valid is not None else np.nan,
                     "seconds": len(raw) / hz if hz == hz else np.nan})
        new_nat.append(X)
        new_nat_per_part.setdefault(f.parents[1].name, []).append(X)
    inf = pd.DataFrame(info)
    print("natural files:", len(inf), "| rate Hz median", round(inf.Hz.median(), 1), "| seconds per file median",
          round(inf.seconds.median(), 1), "| NaN share max", round(inf.nan.max(), 4),
          "| FaceValid=0 share mean", round(inf["FaceValid=0"].mean(), 4), "max", round(inf["FaceValid=0"].max(), 4))
    act_new = []
    for f in sorted((NEW / "ActingRecordings").glob("*/*/*.csv")):
        act_new.append(read_new(f)[2])
    frame_stats("new acted", np.concatenate(act_new), rows)
    frame_stats("new natural", np.concatenate(new_nat), rows)

    # ---------------- headset
    import calibration_study as cs
    import session_analysis as sa
    from jitter_headset_replay import sessions
    D = REPO.parent / "VREmotionDetection/Runtime/Models/sparseVR_teacher_model_acted_only.onnx"
    hs = [cs.replay(sa.load_session(d / "predictions.csv"), D).windows.reshape(-1, 70) for d in sessions()]
    frame_stats("headset (10 sessions)", np.concatenate(hs), rows)

    print("\n" + "=" * 78 + "\n4. MOVEMENT PER FRAME (expressive = gaze, blink, tongue left out)\n" + "=" * 78)
    print(pd.DataFrame(rows).set_index("data").round(3).to_string())

    # ---------------- usable natural windows, train participants only
    print("\nusable natural windows (non-overlapping 5 frames inside one recording), TRAIN participants only:")
    def count(per_part, allowed):
        n_all = n_move = 0
        for p, recs in per_part.items():
            if p not in allowed:
                continue
            for X in recs:
                k = len(X) // 5
                if k == 0:
                    continue
                W = X[:k * 5].reshape(k, 5, 70)
                n_all += k
                n_move += int((W[..., EXPR].max((1, 2)) > 0.25).sum())
        return n_all, n_move
    old_train = {p for p, s in split_of.items() if s == {"train"}}
    new_train = {p for p, s in split_new.items() if s == "train"}
    for name, per, allowed in (("old natural", nat_per_part, old_train), ("new natural", new_nat_per_part, new_train)):
        a, m = count(per, allowed)
        print(f"  {name:<12} participants {len(allowed & set(per))}: windows {a}, with an expressive AU > 0.25: {m}")
    print("  (labelled acted training windows for comparison: old", int((Z['S_train'] == 0).sum()),
          ", new", int((Z['S_train'] == 2).sum()), ")")
    print("\nnew-study val/test participants:", sorted(p for p, s in split_new.items() if s != "train"))
    print("new-study natural participants without a split entry:",
          sorted(set(new_nat_per_part) - set(split_new)))


if __name__ == "__main__":
    main()
