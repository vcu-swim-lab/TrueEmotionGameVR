# Jitter and volume + jitter augmentation

How the teacher model's training copies are made with **jitter alone** and with **volume + jitter**,
and what we measured. The volume rule itself is described in `VOLUME_AUGMENTATION_REPORT.md`.

- Notebook: `notebooks/teacher_jitter_volume_augmentation.ipynb` (Colab, steps 1–10)
- Data: `teacher_arrays_with_emojihero.npz` (model D's exact arrays)
- Headset replay: `Analysis/exploratory/jitter_headset_replay.py`, results in
  `Analysis/results/jitter_headset_replay.csv`

## 1. Terms

| Term | Meaning |
|---|---|
| Window | 5 frames × 70 values: 63 face action units (AUs) and 7 tongue values. One training sample. |
| Face size | Mean over the 5 frames of the sum of the 63 face AUs. Measures how strong the face is. |
| Copy | A changed version of a real window. It keeps the label and the sample weight of its original. |
| Epoch | One pass over the training data. |

## 2. How copies enter training ("add" mode, +100%)

1. At the start of every epoch, take **all** real windows (8,181 with EmojiHero, 2,397 without).
2. Make **one new copy of each** with the chosen method. The real windows are never changed.
3. Shuffle the real windows and the copies together: 16,362 windows per epoch with EmojiHero.
4. In the next epoch, make new copies with new random numbers. Copies are never saved.

The same training settings are used for every method: model D's architecture, SGD 1e-3, batch 10,
at most 30 epochs, early stopping with patience 8 on the domain-balanced validation macro-F1.

## 3. Jitter alone

**One random number per value.** Each of the 350 values of the window (5 frames × 70) is multiplied
by its own random number, drawn uniformly from 0.8 to 1.2.

Steps for one copy:
1. Draw 350 numbers `j[f][a]`, each uniform in [0.8, 1.2]. The mean is 1.0, the spread (sd) is
   about 0.115.
2. `copy[f][a] = window[f][a] × j[f][a]`
3. Clip every value to [0, 1].
4. If the window is Neutral and its largest value is now above 0.25, scale the whole copy down so
   the largest value is 0.25.

Properties:
- A value of 0 stays 0, so jitter never adds a movement that was not there.
- The face size hardly changes: the ups and downs cancel out (median ratio ×1.00, 5–95%: ×0.97–1.03).
- The shape changes a little: shape similarity (cosine) to the original is 0.994.
- The movement in time becomes slightly uneven, because each frame gets different numbers.

Example (one AU, 5 frames):

| Frame | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Original | 0.40 | 0.42 | 0.45 | 0.44 | 0.43 |
| Jitter number | ×1.12 | ×0.85 | ×1.03 | ×0.91 | ×1.18 |
| Copy | 0.448 | 0.357 | 0.464 | 0.400 | 0.507 |

The 0.8–1.2 range is close to the usual magnitude augmentation for sensor data (Um et al. 2017;
Iwana & Uchida 2021 use a scaling sd of 0.1). It was chosen before the runs and was not tuned.

## 4. Volume + jitter

**Volume first, then jitter**, on the same copy.

Steps for one copy:
1. **Face size** `s` of the original window.
2. **Volume range:** from `0.8 / s` to `5.5 / s`, each end kept inside [1/3, 3]. For a Neutral
   window the range is [0.5, 1.5] instead.
3. **Volume number** `g`: drawn log-uniformly in that range, so ×0.5 and ×2 are equally likely.
4. Multiply the **whole window** by `g`.
5. Draw the 350 **jitter numbers** (uniform 0.8–1.2) and multiply each value by its own number.
6. Clip every value to [0, 1].
7. Neutral check: if the largest value of a Neutral copy is above 0.25, scale the copy down to 0.25.

Example (one AU, 2 frames, volume ×0.5):

| | Frame 1 | Frame 2 |
|---|---|---|
| Original | 0.40 | 0.60 |
| After volume ×0.5 | 0.20 | 0.30 |
| Jitter number | ×1.10 | ×0.80 |
| Copy | 0.22 | 0.24 |

Two real training windows, to show both directions of volume:

| Window | Face size | Volume range |
|---|---|---|
| Quiet Anger (our VR data) | 1.03 | ×0.78 – ×3 (mostly louder) |
| Strong Happiness (EmojiHero) | 6.75 | ×0.33 – ×0.81 (always quieter) |

What each step contributes:

| | Strength | Shape |
|---|---|---|
| Volume | changes a lot (×1/3 – ×3) | stays the same |
| Jitter | stays about the same | changes a little |
| Volume + jitter | changes a lot | changes a little |

## 5. Code (from the notebook, step 6)

```python
def volume_gains(X, y, rng):
    # one multiplier per window, log-uniform, aimed at a face size of 0.8-5.5
    size = np.maximum(face_size(X), 1e-6)
    low = np.clip(VOLUME_TARGET_LOW / size, VOLUME_GAIN_MIN, VOLUME_GAIN_MAX)
    high = np.clip(VOLUME_TARGET_HIGH / size, VOLUME_GAIN_MIN, VOLUME_GAIN_MAX)
    neutral = y == NEUTRAL_ID
    low[neutral], high[neutral] = NEUTRAL_GAIN_LOW, NEUTRAL_GAIN_HIGH
    return np.exp(rng.uniform(np.log(low), np.log(high))).astype(np.float32)


def jitter_gains(X, rng):
    # one multiplier per AU per frame
    return rng.uniform(JITTER_LOW, JITTER_HIGH, size=X.shape).astype(np.float32)


def make_copies(X, y, method, rng):
    out = X.astype(np.float32, copy=True)
    if "volume" in method:
        out *= volume_gains(X, y, rng)[:, None, None]
    if "jitter" in method:
        out *= jitter_gains(X, rng)
    out = np.clip(out, 0.0, 1.0)

    neutral = y == NEUTRAL_ID
    if neutral.any():
        peak = out[neutral].max(axis=(1, 2))
        scale = np.minimum(1.0, NEUTRAL_MAX_AU / np.maximum(peak, 1e-6))
        out[neutral] *= scale[:, None, None]
    return out
```

Settings: `JITTER_LOW, JITTER_HIGH = 0.8, 1.2`; `VOLUME_TARGET_LOW, VOLUME_TARGET_HIGH = 0.8, 5.5`;
`VOLUME_GAIN_MIN, VOLUME_GAIN_MAX = 1/3, 3`; `NEUTRAL_GAIN_LOW, NEUTRAL_GAIN_HIGH = 0.5, 1.5`;
`NEUTRAL_MAX_AU = 0.25`.

## 6. Measured effect on the windows (all 8,181 training windows)

| Method | Shape similarity to the original | Size ratio, median | Size ratio, 5% – 95% |
|---|---|---|---|
| Volume | 0.999 | ×0.74 | ×0.36 – ×2.30 |
| Jitter | 0.994 | ×1.00 | ×0.97 – ×1.03 |
| Volume + jitter | 0.993 | ×0.74 | ×0.36 – ×2.25 |

For comparison: the same person on different days has a shape similarity of 0.88; two different
people showing the same emotion have 0.50. The median volume copy is quieter because most training
windows are strong EmojiHero faces.

## 7. Results (seed 42)

**Test set, with EmojiHero** (macro-F1 unless noted):

| Metric | None | Volume | Jitter | Volume + jitter |
|---|---|---|---|---|
| VR combined | 0.464 | 0.484 | 0.485 | **0.495** |
| New VR (quieter faces) | 0.397 | 0.440 | 0.414 | **0.452** |
| Old VR | 0.524 | 0.505 | **0.541** | 0.527 |
| EmojiHero | 0.556 | 0.552 | 0.555 | 0.554 |
| Emotional windows called Neutral (of 520) | 34 | **17** | 34 | 21 |
| Quiet-face accuracy | 0.11 | **0.27** | 0.11 | 0.25 |
| Log loss VR (lower is better) | 1.237 | 1.277 | 1.234 | **1.222** |

**Test set, without EmojiHero:** VR combined none 0.434, volume 0.443, jitter 0.437,
volume + jitter **0.452**. Jitter alone lowers EmojiHero by 0.034 and makes the log loss much worse
(1.279 → 1.519).

**Headset, 10 sessions** (tick macro-F1; the none and volume rows are the deployed ONNX models D and C):

| Model (with EmojiHero) | No calibration | Personal | Movement |
|---|---|---|---|
| None (D) | 0.584 | 0.652 | 0.737 |
| Volume (C) | 0.598 | **0.671** | 0.741 |
| Jitter | 0.576 | 0.634 | 0.738 |
| Volume + jitter | **0.616** | 0.661 | 0.740 |

Volume + jitter is better than D in 8 of 10 sessions without calibration.

## 8. Why volume + jitter does better than either alone

1. **They help different faces.** Volume helps quiet faces (false Neutrals 34 → 17, quiet accuracy
   0.11 → 0.27). Jitter does not change strength, so it does not help quiet faces; it helps the
   normal-strength old VR faces (0.524 → 0.541). Together they get most of both.
2. **Jitter repairs a side effect of volume.** Volume alone lowers old VR (0.524 → 0.505) and
   worsens log loss. A possible reason (not proven): a volume copy is an exact scaled version of a
   real window, so the model sees each exact shape twice per epoch. Jitter makes every copy a little
   different. With jitter added, old VR returns to 0.527 and log loss is the best (1.222).
3. **On the headset it is never the worst emotion by emotion.** Volume alone loses Disgust
   (0.47); jitter alone loses Surprise (0.39); volume + jitter is best or second best for every
   emotion (Anger 0.55, Disgust 0.57, Surprise 0.67, Happiness 0.86).

## 9. Limits and decision

- **One seed per model.** The gain of volume + jitter over volume alone (+0.011 on the test set,
  +0.018 on the headset) is smaller than the seed noise (about 0.02). It needs more seeds.
- **The headset sessions are mostly one person** (9 of 10).
- **With movement calibration every model is equal** (0.737–0.744), so the choice of copies matters
  only for players who are not calibrated.
- Jitter changes the shape much less than real person differences (0.993 vs 0.50), so it cannot
  teach the model how other people look.
- The jitter models are `.keras` files only; no ONNX has been exported.

**Current decision:** keep model D for sessions with movement calibration. If uncalibrated play
matters, train more seeds of volume and volume + jitter, and export the better one to ONNX.
