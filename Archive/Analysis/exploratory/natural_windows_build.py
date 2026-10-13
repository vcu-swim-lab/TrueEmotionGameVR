"""Build unlabelled window pools for the masked-AU auxiliary experiment (masked_au_aux.py).

Two pools, built by the SAME function, TRAIN participants only, no emotion labels kept:
  natural   old-study weights_<E>.csv + new-study VideoRecordings      (arm B)
  acted     old-study ACT_<E>.csv     + new-study ActingRecordings     (arm C, ungated)

Train participants
  old study: the participants whose acted windows match the npz TRAIN windows exactly
             (natural_data_inspection.py; 18 people). Validation/test participants and the 10
             folders whose split is unknown are not used.
  new study: Analysis/data/acted_splits.csv, split == train (10 people).
  The developer is not a study participant (confirmed by the user, 2026-10-08), so the headset
  sessions do not overlap with these people.

Cleaning, per recording, BEFORE any downsampling
  invalid frame: any non-finite AU value, all 70 AUs exactly 0, or FaceValid == 0 (new study).
  Invalid frames are removed and the recording is cut there. A recording is also cut where the
  time step is > 2.5x its median step (old: first column, ~0.125 s; new: RealtimeSeconds, ~0.014 s).
  new study: each clean segment is downsampled by mean of 6 rows (as the notebook does), only
  inside the segment. old study: used at its own rate (as the notebook uses old acted).
  Windows: non-overlapping 5 frames inside one segment, no gating.

Output: Analysis/data/aux_pools.npz with X_<pool>, participant_<pool>, recording_<pool>, and
a printed report of what was excluded.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Analysis"))
import numpy as np
import pandas as pd

import teacher_preprocessing as tp

Z = np.load(REPO / "notebooks/teacher_arrays_with_emojihero.npz")
AU = [str(a) for a in Z["au_names"]]
OLD = REPO / "Analysis/data/filtered-dataset-20260924T162300Z-1-001/filtered-dataset"
NEW = Path(r"C:\Users\alime\multimodal-vr-emotion-study\multimodal-vr-emotion-study-main")
GAP = 2.5


def old_train_participants():
    """Participants whose acted windows match the npz old-acted TRAIN windows (exact)."""
    key = lambda w: np.round(w, 5).tobytes()
    train_keys = {key(w) for w in Z["X_train"][Z["S_train"] == 0]}
    eval_keys = {key(w) for s in ("val", "test") for w in Z[f"X_{s}_old"]}
    train, evalp = set(), set()
    for p in sorted(d for d in OLD.iterdir() if d.is_dir()):
        for f in p.glob("ACT_*.csv"):
            df = pd.read_csv(f)
            X = np.nan_to_num(df[AU].to_numpy(np.float64)).astype(np.float32)
            emo = f.stem.split("_", 1)[1].split("(")[0]
            for w in tp.make_windows(X, emo):
                k = key(w)
                if k in train_keys:
                    train.add(p.name)
                if k in eval_keys:
                    evalp.add(p.name)
    assert not (train & evalp), train & evalp
    return train


def segments(X, t, valid):
    """Cut at invalid frames and at time gaps. Returns a list of clean frame blocks."""
    bad = ~np.isfinite(X).all(1) | (np.nan_to_num(X).max(1) == 0)
    if valid is not None:
        bad |= valid == 0
    gap = np.zeros(len(X), bool)
    if t is not None and len(t) > 2:
        dt = np.diff(t)
        gap[1:] = dt > GAP * np.median(dt)
    out, cur = [], []
    for i in range(len(X)):
        if bad[i]:
            if cur:
                out.append(np.array(cur)); cur = []
            continue
        if gap[i] and cur:
            out.append(np.array(cur)); cur = []
        cur.append(X[i])
    if cur:
        out.append(np.array(cur))
    return out, int(bad.sum()), int(gap.sum())


def build(pool):
    old_train = old_train_participants()
    new_split = pd.read_csv(REPO / "Analysis/data/acted_splits.csv", comment="#")
    new_train = set(new_split.participant[new_split.split == "train"])
    files = []
    pat_old = "weights_*.csv" if pool == "natural" else "ACT_*.csv"
    for p in sorted(old_train):
        files += [("old", p, f) for f in sorted((OLD / p).glob(pat_old))]
    sub = "VideoRecordings" if pool == "natural" else "ActingRecordings"
    for p in sorted(new_train):
        files += [("new", p, f) for f in sorted((NEW / sub / p).glob("*/*.csv"))]

    Xs, parts, recs, rep = [], [], [], []
    for study, p, f in files:
        df = pd.read_csv(f)
        X = df[AU].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        if study == "old":
            t, valid = df.iloc[:, 0].to_numpy(float), None
        else:
            t = pd.to_numeric(df["RealtimeSeconds"], errors="coerce").to_numpy(float)
            valid = df["FaceValid"].to_numpy() if "FaceValid" in df.columns else None
        segs, n_bad, n_gap = segments(X, t, valid)
        n_win = 0
        for s in segs:
            if study == "new":
                s = tp.downsample(s)
            k = len(s) // 5
            if k:
                Xs.append(s[:k * 5].reshape(k, 5, 70).astype(np.float32))
                parts += [f"{study}:{p}"] * k
                recs += [f"{study}:{p}:{f.parent.name if study == 'new' else ''}/{f.name}"] * k
                n_win += k
        rep.append({"study": study, "participant": p, "file": f.name, "frames": len(X), "invalid frames": n_bad,
                    "gap cuts": n_gap, "segments": len(segs), "windows": n_win})
    return np.concatenate(Xs), np.array(parts), np.array(recs), pd.DataFrame(rep)


def main():
    out = {}
    for pool in ("natural", "acted"):
        X, P, R, rep = build(pool)
        out[f"X_{pool}"], out[f"participant_{pool}"], out[f"recording_{pool}"] = X, P, R
        print(f"\n=== {pool} pool (train participants only)")
        g = rep.groupby("study")
        print(pd.DataFrame({"participants": g.participant.nunique(), "recordings": g.size(), "frames": g.frames.sum(),
                            "invalid frames removed": g["invalid frames"].sum(), "gap cuts": g["gap cuts"].sum(),
                            "windows": g.windows.sum()}).to_string())
        per_p = pd.Series(P).value_counts()
        per_r = pd.Series(R).value_counts()
        print(f"windows per participant: min {per_p.min()} median {int(per_p.median())} max {per_p.max()} | "
              f"per recording: min {per_r.min()} median {int(per_r.median())} max {per_r.max()} | "
              f"recordings with 0 windows: {(rep.windows == 0).sum()}")
        worst = rep.assign(share=rep["invalid frames"] / rep.frames).sort_values("share", ascending=False).head(3)
        print("highest invalid share:", worst[["participant", "file", "share"]].round(3).to_dict("records"))
    np.savez_compressed(REPO / "Analysis/data/aux_pools.npz", **out)
    print("\nsaved Analysis/data/aux_pools.npz")


if __name__ == "__main__":
    main()
