# Volume Augmentation: Method, Implementation and First Results

**Status (2026-10-05):**
- Models A and B (no EmojiHero) were trained and evaluated on the test sets. B was tested on the
  headset and **does not win** by the plan's rules (5.9).
- **Model C (D + augmentation, EmojiHero kept)** was trained, evaluated and recorded live on the
  headset (5.10). It is the **best model on the test sets** (VR combined macro-F1 0.490 vs D 0.455)
  and the best on weak headset ticks. On the headset it is **about equal to D** overall: slightly
  better raw, slightly worse with personal calibration. It is much better on Anger and worse on
  Disgust.
- **Fear** is about 10% on the headset for every model. Section 5.11 shows why: headset Fear
  faces are very small, and their shape is closer to Surprise than to training Fear.
- The A and B test numbers in 5.3–5.8 are preliminary (gate temperature, 5.9). The C numbers are
  final.
**Notebook:** `notebooks/teacher_notebook_only_acted_data_volume_augmentation.ipynb`, section
5.5.1 "With Self-Filtering Only-Acted Except Emoji Hero, Volume Augmentation".
**Plan it implements:** `Docs/NEXT_STEP_VOLUME_AUGMENTATION_PLAN.md` (the 2026-09-29 plan; this
report records what was changed while implementing it and what came out).

---

## 1. The problem

On the headset, the teacher calls a face "Neutral" whenever it moves only a little. Many of our
headset users move their faces much less than the people the model was trained on, and the
training notebook keeps a window only if some signal goes above 0.25, so the model has almost never
seen weak versions of an emotion.

The idea: we cannot easily record many new people, so we take the training windows we already have
and make **quieter and louder copies** of them, like turning a volume knob. The model then learns
that a small Anger face is still Anger.

### 1.1 Evidence from the headset

Six `session-local-v1` sessions, current acted-only teacher (`Analysis/exploratory/gate_check.py`):

| Expressive headset ticks | Value |
|---|---|
| Too weak to pass the notebook's 0.25 rule | **32%** |
| Accuracy on those weak ticks | **20%** |
| Accuracy on ticks that pass the rule | 71% |
| Weak ticks the model calls Neutral | **72%** |

| Share of weak ticks | Anger | Fear | Surprise | Disgust | Happiness | Sadness |
|---|---|---|---|---|---|---|
| | **60%** | **67%** | **48%** | 6% | 5% | 7% |

The emotions the model fails on are the ones people express weakly.

Two later sessions show that a quiet face is not the only problem:
- **Sep 24 (`739a55`)** is a quiet face: Anger moves about 20% as much as in the test data, Fear
  about 30–65%. This is what volume augmentation is built for.
- **Sep 30 (`26908f`)** is a normal-size face whose Disgust has a different *shape* (to the model it
  looks like Surprise). Changing the volume does not change the shape, so augmentation cannot fix
  that.

### 1.2 Evidence from the training data

**Face size** of a frame = the sum of its 63 non-tongue signals. A resting Neutral face is about
0.6–1.2; a strong expression about 4–6.

**Median face size per collection** = for each person, the face size of their recording of that
emotion (averaged over its frames); then the middle person of the collection.

| Emotion | Old acted (24) | New acted (15) | EmojiHero (36) |
|---|---|---|---|
| Anger | 2.70 | **1.88** | 5.11 |
| Disgust | 3.89 | **2.55** | 5.56 |
| Fear | 2.26 | **1.15** | 3.13 |
| Happiness | 5.42 | **3.12** | 5.21 |
| Sadness | 2.49 | **2.02** | 4.14 |
| Surprise | 2.58 | **1.49** | 3.30 |
| Neutral | 1.21 | 0.64 | 1.03 |

- The three collections differ for every expressive emotion (Kruskal–Wallis p ≤ 0.0011).
- New acted Fear (1.15) is about the size of an old acted *Neutral* face (1.21).
- From the quietest to the most expressive person is about 10×.

The median is used (not the average) because a few very expressive people would pull an average up.

---

## 2. The method, step by step

### Step 1: measure the face size of a window

A window is 5 frames × 70 signals (half a second at 10 Hz).

1. For each of the 5 frames, add up the 63 non-tongue signals.
2. Average the 5 sums. That is the window's face size.

Example: frame sums 1.70, 1.90, 2.10, 2.00, 1.80 → face size = 9.5 / 5 = **1.90**.

**Why the tongue is left out of the size:**

| Data | Tongue's share of all face movement | Windows where the tongue moves (> 0.05) |
|---|---|---|
| Old acted test | 0.13% | 2.3% |
| New acted test | 0.59% | 6.2% |
| EmojiHero test | 0% (no tongue data) | 0% |
| Headset (1,323 ticks) | 0.11% | 0.5% |

Setting all 7 tongue signals to 0 changes under 1% of the current teacher's predictions and
macro-F1 by at most 0.0014. EmojiHero has no tongue data, and the target sizes below were measured
without it.

The tongue is left out only when **measuring** the size. The multiplier is still applied to **all
70 signals**, tongue included, so the whole window gets quieter or louder together (and 0 stays 0).

### Step 2: the quietest target, 0.8

We want copies as quiet as the quiet headset users, but not pure sensor noise.

1. Take every expressive headset tick and compute its face size.
2. The quietest 10% are below about 0.6.
3. Round up to **0.8**, because at 0.6 a "face" is a resting face plus tracking noise.

This is the least certain number, and it was taken from the same sessions that will judge the
result, so a win must be confirmed on new people.

### Step 3: the loudest target, 5.5

We do not want copies louder than any real person.

1. Take all 75 people (24 old acted, 15 new acted, 36 EmojiHero) and compute each person's average
   face size.
2. 90% of people are below 5.47 → **5.5**.

### Step 4: the multiplier range of a window

The copy's size = window size × multiplier, so multiplier = target size ÷ window size:

- lowest multiplier = 0.8 ÷ window size (makes the copy size 0.8)
- highest multiplier = 5.5 ÷ window size (makes the copy size 5.5)

Both are then kept between **×1/3 and ×3**. Above ×3, sensor noise is amplified too much, and real
exaggerated faces use extra muscles that scaling cannot create; ×1/3 is the mirror. These two caps
are design choices, not measurements.

| Window size | Wanted range | After the caps | Copies land at | Effect |
|---|---|---|---|---|
| 7.0 (strong) | ×0.11 – ×0.79 | **×0.33** – ×0.79 | 2.3 – 5.5 | only quieter |
| 4.93 | ×0.16 – ×1.12 | **×0.33** – ×1.12 | 1.6 – 5.5 | mostly quieter |
| 2.2 (typical) | ×0.36 – ×2.5 | same | 0.8 – 5.5 | both ways |
| 1.0 (weak) | ×0.8 – ×5.5 | ×0.8 – **×3** | 0.8 – 3.0 | mostly louder |

### Step 5: pick one multiplier and apply it

1. Pick one number from the window's range, **log-uniformly** (halving and doubling are equally
   likely; a plain uniform pick between 0.36 and 2.5 would mostly make copies louder).
2. Multiply **all 70 signals in all 5 frames** by that one number.
3. Clip to [0, 1]: the headset never reports values outside 0–1, so the model should never train on
   them.
4. Keep the label.

Example (Anger, ×0.5): BrowLowerer 0.35 → 0.18, LidTightener 0.42 → 0.21, JawDrop 0.05 → 0.03.
Same muscles, same balance, quieter, still Anger.

**Why one number per window, not one per signal:** an emotion is the *balance* between muscles.
Random numbers per signal can turn Anger into something that looks like Surprise (for example
JawDrop ×3 while the brows go ×0.4), and the label would then be partly wrong.

**Clipping, measured:** 0.11% of copy values are clipped, and 9.5% of copies have at least one
clipped value (real training data). A copy that clips loses a little of its muscle balance; a cap
of g ≤ 1 ÷ (largest signal) would avoid this entirely and is a possible later refinement.

### Step 6: Neutral windows

A Neutral face must stay neutral.

1. Pick the multiplier log-uniformly between ×0.5 and ×1.5.
2. If it would push any signal above 0.25, lower it so the largest signal is exactly 0.25 (the
   notebook's own Neutral rule).

### Step 7: how copies are fed to training (k = 1)

Every epoch the model sees:
1. **all original windows**, unchanged, and
2. **one scaled copy of each window** (k = 1), with the same label and the same sample weight.

So 2,397 → **4,794 windows per epoch**: 50% real, 50% copies. A new copy is drawn every epoch, so
over training each window is seen at many strengths. Nothing is saved to disk; validation and test
windows are never changed.

**Why k = 1:** the scaling rule decides what a copy looks like; k only decides how much of training
is copies. Simulated on the gated old + new acted windows:

| Copies per window | Windows per epoch | Real | Quiet faces (size < 2.5) |
|---|---|---|---|
| 0 | 2,400 | 100% | 27% |
| **1** | **4,800** | **50%** | **36%** |
| 2 | 7,200 | 33% | 39% |
| 4 | 12,000 | 20% | 41% |
| 9 | 24,000 | 10% | 43% |

The first copy gives most of the effect; more copies mainly cost training time.

### What the copies look like (real training data, one epoch)

| Expressive face size | 5% | 25% | 50% | 75% | 95% |
|---|---|---|---|---|---|
| Originals | 1.08 | 2.09 | 3.10 | 4.52 | 8.53 |
| Copies | 0.97 | 1.64 | 2.57 | 3.82 | 5.10 |

The over-loud windows (95th percentile 8.53) are pulled back, and quiet faces become common.

---

## 3. How others do it

The same idea exists under different names. Nobody we found scales headset face signals to a
target face size, but each field supports part of the design.

| Field | Name | What they do | What it tells us |
|---|---|---|---|
| Wearable sensors (Um et al. 2017) | scaling | one random number per window, N(1, 0.1) | scaling did **not** help (77.54% → 77.52%) when strength is part of the label; watch Neutral |
| Time series (Iwana & Uchida 2021) | scaling | one number per series, N(1, 0.2), about ×0.6–×1.4 | supports one number per example; their range is much narrower than ours |
| Audio (audiomentations) | gain | −12 to +12 dB ≈ ×0.25–×4, chosen in dB (a log scale), fresh each time | closest match: similar range, log-uniform pick, warns about clipping |
| Micro-expressions | magnification | ×2 to ×13 | too much causes deformation; one fixed factor for all samples works badly, which supports a per-window range |

How many copies people add is usually not tested: Iwana & Uchida added four times the original
data and kept that fixed. The careful way is to try a few values on validation. Image work on
"augmentation multiplicity" (Hoffer et al. 2019; Fort et al. 2021) found several copies per
example in a batch help, on much larger datasets than ours.

Sources:
[Um et al. 2017](https://arxiv.org/html/1706.00527) ·
[Iwana & Uchida 2021](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254841) ·
[audiomentations Gain](https://iver56.github.io/audiomentations/waveform_transforms/gain/) ·
[Magnification-robust micro-expression recognition](https://pmc.ncbi.nlm.nih.gov/articles/PMC9498083/) ·
[Hoffer et al.](https://arxiv.org/abs/1901.09335) ·
[Fort et al.](https://arxiv.org/abs/2105.13343)

---

## 4. Implementation in the notebook

The new section reuses Parts 1–5.4 of the notebook unchanged. Run order:

| Step | Cell | Run? |
|---|---|---|
| 1 | original 5.5.1 "With Self-Filtering-Only-Acted" and cells 52–58 | skip (they put EmojiHero back into training) |
| 2 | **A: corpus without EmojiHero in training** (new) | run |
| 3 | **B: volume augmentation** (new) | run |
| 4 | Part 6 cell 61; Part 7 cells 64, 66, 70 | run |
| 5 | original training cell 71 | skip |
| 6 | **C: training** (new; cell 71 with the changes below) | run |
| 7 | **D: test evaluation of one run**, **E: per-emotion A vs B** (new) | run |
| 8 | **F: pooled and domain-balanced metrics**, **G: the same per emotion** (new; need E) | run |

**Cell A** builds training from old acted (1,553 windows) + new acted (844) = **2,397 windows**.
The deployed teacher was trained with 5,784 EmojiHero windows as well (71% of its 8,181). Source
weights now split the loss 50/50 between old and new acted. Validation (278) and test (607) windows
are the same as before.

**Cell B** holds the rule of section 2 (`AUG_COPIES = 1` for model B, `0` for model A):

```python
AUG_COPIES = 1                 # k; 0 = model A (no augmentation), 1 = model B
AUG_SEED = 42

FACE_SIZE_CHANNELS = slice(0, 63)   # tongue (63-69) is left out of the size only
TARGET_SIZE_LOW = 0.8
TARGET_SIZE_HIGH = 5.5
GAIN_MIN = 1.0 / 3.0
GAIN_MAX = 3.0

NEUTRAL_ID = int(le.transform([Emotion.NEUTRAL.value])[0])
NEUTRAL_GAIN_LOW = 0.5
NEUTRAL_GAIN_HIGH = 1.5
NEUTRAL_MAX_SIGNAL = 0.25           # the notebook's own Neutral rule


def window_face_size(X):
    """(N, 5, 70) -> (N,)"""
    return X[..., FACE_SIZE_CHANNELS].sum(axis=-1).mean(axis=-1)


def gain_ranges(X, y):
    """Allowed multiplier range per window."""
    size = np.maximum(window_face_size(X), 1e-6)
    low = np.clip(TARGET_SIZE_LOW / size, GAIN_MIN, GAIN_MAX)
    high = np.clip(TARGET_SIZE_HIGH / size, GAIN_MIN, GAIN_MAX)

    neutral = y == NEUTRAL_ID
    low[neutral] = NEUTRAL_GAIN_LOW
    high[neutral] = NEUTRAL_GAIN_HIGH
    return low, high


def draw_gains(X, y, low, high, rng):
    """One log-uniform multiplier per window; Neutral is capped at 0.25."""
    gains = np.exp(rng.uniform(np.log(low), np.log(high)))

    neutral = y == NEUTRAL_ID
    peak = X[neutral].max(axis=(1, 2))
    gains[neutral] = np.minimum(
        gains[neutral],
        NEUTRAL_MAX_SIGNAL / np.maximum(peak, 1e-6),
    )
    return gains.astype(np.float32)


class VolumeAugmentedWindows(tf.keras.utils.PyDataset):
    """Originals + `copies` freshly scaled copies per window, reshuffled every epoch."""

    def __init__(self, X, y, sample_weight, copies, batch_size, seed, **kwargs):
        super().__init__(**kwargs)
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.w = np.asarray(sample_weight, dtype=np.float32)
        self.copies = int(copies)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.low, self.high = gain_ranges(self.X, self.y)
        self._new_epoch()

    def _new_epoch(self):
        n = len(self.X)
        self.source = np.tile(np.arange(n), 1 + self.copies)
        self.gain = np.ones(len(self.source), dtype=np.float32)
        self.is_copy = np.arange(len(self.source)) >= n
        if self.copies:
            self.gain[n:] = np.concatenate([
                draw_gains(self.X, self.y, self.low, self.high, self.rng)
                for _ in range(self.copies)
            ])
        order = self.rng.permutation(len(self.source))
        self.source = self.source[order]
        self.gain = self.gain[order]
        self.is_copy = self.is_copy[order]

    def __len__(self):
        return math.ceil(len(self.source) / self.batch_size)

    def __getitem__(self, index):
        batch = slice(index * self.batch_size, (index + 1) * self.batch_size)
        rows = self.source[batch]
        X_batch = np.clip(self.X[rows] * self.gain[batch, None, None], 0.0, 1.0)
        return X_batch, self.y[rows], self.w[rows]

    def on_epoch_end(self):
        self._new_epoch()
```

**Cell C** is cell 71 with five changes:
1. run name `teacher-70AU-oldacted-new15acted-volaug-k{AUG_COPIES}`, so A and B never overwrite
   each other;
2. checkpoint selection on old + new acted validation only (EmojiHero removed from
   `validation_domains`);
3. pooled validation without EmojiHero;
4. the callback's print line no longer reads `'emojihero'`;
5. `model.fit(train_data, ...)` with `train_data = VolumeAugmentedWindows(...)`; `batch_size` and
   `sample_weight` come from the dataset.

Everything else is identical for A and B: seed, split, model, SGD, batch size 10, 30 epochs, early
stopping (patience 8) on the domain-balanced validation macro-F1.

**Cells D–G** rebuild the model, load a saved run's weights and score the test sets, without
retraining or tuning. D: one run's overall results, recall per emotion, confusion matrices and
accuracy by face size. E: per-emotion precision, recall and F1 for A and B on each test set. F: the
pooled and domain-balanced rows of the original final evaluation, including log loss. G: the same
two rows per emotion. Results go to `output_path/volume_augmentation_eval/`.

---

## 5. Results

### 5.1 The three models

| Model | EmojiHero in training | Volume augmentation |
|---|---|---|
| **D** (deployed, `sparseVR_teacher_model_acted_only`) | yes | no |
| **A** | no | no |
| **B** | no | yes (k = 1) |

A vs D shows the effect of removing EmojiHero; B vs A shows the effect of augmentation.

**Model B training:** best epoch 14 of 22 (early stop), validation macro-F1 old acted 0.7248, new
acted 0.5733, domain balanced 0.6490.

### 5.2 Test sets

All three models are scored on the same held-out windows:
- **old VR acted:** 4 people (P10, P11, P14, P21), 303
  windows;
- **new VR acted:** 3 people (P47, P48, P49), 304 windows;
- **VR combined:** both together, 607 windows from 7 people (the notebook's primary test);
- **EmojiHero:** its test split, 2,268 windows (reporting only; A and B never trained on EmojiHero).

### 5.3 Overall

| Test set | Windows | Accuracy D | Accuracy A | Accuracy B | Macro-F1 D | Macro-F1 A | Macro-F1 B |
|---|---|---|---|---|---|---|---|
| Old VR acted | 303 | 0.522 | **0.535** | 0.525 | 0.494 | **0.506** | 0.499 |
| New VR acted | 304 | **0.493** | 0.457 | 0.490 | **0.414** | 0.363 | 0.386 |
| **VR combined** | **607** | **0.507** | 0.496 | **0.507** | **0.455** | 0.438 | 0.452 |
| EmojiHero | 2,268 | **0.590** | 0.523 | 0.519 | **0.566** | 0.495 | 0.490 |

| Test set | Removing EmojiHero (A − D) | Augmentation (B − A) |
|---|---|---|
| Old VR acted | +0.012 | −0.006 |
| New VR acted | **−0.051** | **+0.023** |
| VR combined | −0.016 | +0.013 |
| EmojiHero | −0.070 | −0.005 |

### 5.4 F1 per emotion (D / A / B; bold = best of the three)

| Emotion | Old VR acted | New VR acted | VR combined | EmojiHero |
|---|---|---|---|---|
| Anger | 0.29 / 0.28 / **0.34** | **0.17** / 0.13 / 0.04 | **0.24** / 0.20 / 0.21 | **0.48** / 0.36 / 0.33 |
| Disgust | 0.42 / **0.54** / 0.52 | 0.50 / 0.45 / **0.54** | 0.46 / 0.49 / **0.53** | 0.39 / 0.39 / **0.41** |
| Fear | 0.34 / **0.36** / 0.34 | **0.11** / 0.00 / 0.00 | **0.27** / 0.25 / 0.24 | **0.18** / 0.07 / 0.09 |
| Happiness | 0.78 / **0.83** / 0.74 | **0.54** / 0.53 / 0.51 | 0.65 / **0.66** / 0.62 | **0.80** / 0.74 / 0.72 |
| Neutral | **0.88** / 0.79 / 0.86 | 0.77 / 0.75 / **0.84** | 0.81 / 0.77 / **0.85** | **0.68** / 0.58 / 0.57 |
| Sadness | 0.32 / 0.31 / **0.33** | **0.54** / 0.47 / 0.50 | 0.40 / 0.37 / **0.40** | **0.75** / 0.72 / 0.70 |
| Surprise | 0.43 / **0.43** / 0.37 | **0.27** / 0.22 / **0.27** | **0.35** / 0.33 / 0.32 | **0.67** / 0.61 / 0.61 |
| **Macro-F1** | 0.49 / **0.51** / 0.50 | **0.41** / 0.36 / 0.39 | **0.45** / 0.44 / **0.45** | **0.57** / 0.50 / 0.49 |

### 5.5 Recall per emotion (D / A / B)

| Emotion | Old VR acted | New VR acted | VR combined | EmojiHero |
|---|---|---|---|---|
| Anger | 0.23 / 0.20 / 0.25 | 0.13 / 0.10 / 0.03 | 0.18 / 0.15 / 0.14 | 0.40 / 0.32 / 0.26 |
| Disgust | 0.35 / 0.43 / 0.46 | 0.55 / 0.42 / 0.60 | 0.45 / 0.42 / 0.53 | 0.42 / 0.41 / 0.49 |
| Fear | 0.38 / 0.41 / 0.38 | 0.06 / 0.00 / 0.00 | 0.23 / 0.21 / 0.20 | 0.13 / 0.05 / 0.06 |
| Happiness | 0.96 / 0.94 / 0.93 | 0.82 / 0.82 / 0.80 | 0.89 / 0.88 / 0.86 | 0.86 / 0.74 / 0.73 |
| Neutral | 1.00 / 1.00 / 0.92 | 0.98 / 1.00 / 1.00 | 0.99 / 1.00 / 0.97 | 0.80 / 0.64 / 0.58 |
| Sadness | 0.33 / 0.36 / 0.36 | 0.41 / 0.41 / 0.46 | 0.37 / 0.38 / 0.40 | 0.73 / 0.74 / 0.73 |
| Surprise | 0.36 / 0.36 / 0.31 | 0.20 / 0.15 / 0.20 | 0.28 / 0.25 / 0.25 | 0.79 / 0.76 / 0.77 |

A Neutral recall near 1.00 is not good news by itself: it partly means the model calls too many
windows Neutral (5.6).

### 5.6 The "weak face called Neutral" problem (VR combined)

| | D | A | B |
|---|---|---|---|
| Expressive windows wrongly called Neutral (of 520) | 39 | 53 | **27** |
| Neutral precision | 0.69 | 0.62 | **0.76** |
| Correct windows in total (of 607) | 308 | 301 | 308 |

Correct windows per emotion, VR combined, A → B: Anger 13 → 12, Disgust 46 → **58**, Fear 14 → 13,
Happiness 91 → 89, Neutral 87 → 84, Sadness 31 → 33, Surprise 19 → 19.

### 5.7 Model B accuracy by face size (VR combined, expressive windows)

| Face size | Windows | Accuracy |
|---|---|---|
| quiet (< 1.5) | 44 | 0.18 |
| 1.5 – 2.5 | 94 | 0.18 |
| 2.5 – 4.0 | 174 | 0.37 |
| loud (> 4.0) | 208 | 0.64 |

The same table for D and A has not been computed yet. The test windows passed the 0.25 rule, so
only 44 of 520 are quiet; the headset has far more.

A's overall numbers were computed from its per-emotion results (macro-F1 is the mean of the 7
per-emotion F1 scores, and accuracy comes from the exact correct-window counts); the same
computation reproduces B's saved summary exactly.

### 5.8 All acted test sets together (pooled and domain-balanced)

The same two summary rows as the original final test evaluation of D:
- **all_acted_pooled:** old VR + new VR + EmojiHero test windows, scored together once (2,875
  windows). 79% of these windows are EmojiHero, so this row mostly measures EmojiHero, which only D
  trained on.
- **all_acted_domain_balanced:** the plain average of the three test sets' scores, so each set
  counts one third. Our two VR groups make up two thirds of it.

Computed in Colab by Cells F and G (section 4). The accuracy, F1, precision and recall values match
a reconstruction from the saved per-emotion results exactly.

**all_acted_pooled (2,875 windows)**

| Model | Accuracy | Balanced accuracy | Precision (macro) | Recall (macro) | F1 macro | F1 weighted | Log loss ↓ |
|---|---|---|---|---|---|---|---|
| D | **0.5725** | **0.5692** | **0.5480** | **0.5692** | **0.5460** | **0.5483** | **1.3252** |
| A | 0.5176 | 0.5143 | 0.4787 | 0.5143 | 0.4878 | 0.4908 | 1.3992 |
| B | 0.5162 | 0.5123 | 0.4853 | 0.5123 | 0.4859 | 0.4889 | 1.3748 |

**all_acted_domain_balanced**

| Model | Accuracy | Balanced accuracy | Precision (macro) | Recall (macro) | F1 macro | F1 weighted | Log loss ↓ |
|---|---|---|---|---|---|---|---|
| D | **0.5349** | **0.5183** | **0.5215** | **0.5183** | **0.4910** | **0.5012** | 1.3412 |
| A | 0.5051 | 0.4884 | 0.4620 | 0.4884 | 0.4545 | 0.4685 | 1.3485 |
| B | 0.5111 | 0.4909 | 0.4624 | 0.4909 | 0.4584 | 0.4729 | **1.3032** |

**F1 per emotion (D / A / B)**

| Emotion | Pooled | Domain-balanced |
|---|---|---|
| Anger | **0.43** / 0.33 / 0.30 | **0.32** / 0.26 / 0.23 |
| Disgust | 0.41 / 0.41 / **0.44** | 0.44 / 0.46 / **0.49** |
| Fear | **0.20** / 0.11 / 0.12 | **0.21** / 0.14 / 0.15 |
| Happiness | **0.76** / 0.72 / 0.69 | **0.71** / 0.70 / 0.66 |
| Neutral | **0.71** / 0.62 / 0.63 | **0.78** / 0.71 / 0.76 |
| Sadness | **0.68** / 0.65 / 0.65 | **0.53** / 0.50 / 0.51 |
| Surprise | **0.63** / 0.57 / 0.57 | **0.46** / 0.42 / 0.42 |

**Recall per emotion (D / A / B)**

| Emotion | Pooled | Domain-balanced |
|---|---|---|
| Anger | 0.36 / 0.29 / 0.24 | 0.25 / 0.21 / 0.18 |
| Disgust | 0.42 / 0.41 / 0.50 | 0.44 / 0.42 / 0.52 |
| Fear | 0.14 / 0.07 / 0.08 | 0.19 / 0.15 / 0.15 |
| Happiness | 0.87 / 0.78 / 0.76 | 0.88 / 0.83 / 0.82 |
| Neutral | 0.84 / 0.72 / 0.66 | 0.93 / 0.88 / 0.83 |
| Sadness | 0.66 / 0.67 / 0.67 | 0.49 / 0.50 / 0.52 |
| Surprise | 0.69 / 0.66 / 0.67 | 0.45 / 0.42 / 0.42 |

**Log loss.** Accuracy only checks whether the top guess is right; log loss also checks how
confident the model was, and punishes confident mistakes most. Lower is better.

| | D | A | B |
|---|---|---|---|
| Pooled | **1.325** | 1.399 | 1.375 |
| Domain-balanced | 1.341 | 1.348 | **1.303** |

- B has lower log loss than A on both rows: augmentation makes the probabilities more trustworthy,
  plausibly because B has seen faces at many strengths and is less overconfident on quiet or loud
  ones.
- On domain-balanced, B's log loss is even lower than D's, although D is better on accuracy and F1.
- Personal calibration on the headset works on the model's log-scores, not only on its top guess,
  so better probabilities might help calibration more than the small accuracy gain suggests. This
  is a hypothesis for the headset replay to test. (Section 5.9: on the headset it did not hold.)

### 5.9 Headset

**Export.** Model B was exported to ONNX with the standalone Cell H (fresh runtime, weights from
Drive). It has the same interface as the deployed file: input `seq` (batch, 5, 70), output
`probabilities` (batch, 7), opset 17. Keras and ONNX agree on 300 of 300 windows (largest
difference 7e-7). On the headset it is
`sparseVR_teacher_model_acted_only_vol-aug-k1_excluding-EmojiHero.onnx`, SHA-256 `996f62ba…7fcf76`,
the same file as the export.

**Gate temperature.** The gate's temperature tau is a plain `tf.Variable`, so it is not stored in
`.keras` files: a model loaded from a file runs with tau = 2.0. The deployed D was exported with
tau = 0.7 (its graph multiplies the gate logits by 1/0.7 = 1.4286). Cell H sets tau = 0.7 after
loading, so B on the headset is consistent with D. **The test-set numbers for A and B in sections
5.3–5.8 were computed from loaded weights, that is with tau = 2.0, and are therefore
preliminary.** Cells D–G need `model.get_layer("gate").tau.assign(0.7)` after `load_weights` and a
re-run.

**New live session `20261001_170504_e839ad` (model B on the headset).** All 210 ticks: 79.5% tick
accuracy, 81% trial accuracy, the best raw session so far. `runtime_calibration_report.py
--verify`: PARITY OK.

| Emotion (all ticks) | Correct | Most confused with |
|---|---|---|
| Disgust, Happiness, Neutral, Sadness | 100% | |
| Surprise | 80% | Fear (20%) |
| Anger | 77% | Neutral (17%) |
| Fear | **0%** | Neutral (77%), Surprise (23%) |

| Evaluation block (14 trials) | Tick accuracy | Trial macro-F1 |
|---|---|---|
| Raw | 0.754 | **0.701** |
| Personal calibration | 0.690 | 0.605 |

Personal calibration made this session worse: it flipped one Surprise trial to Fear.

The same windows replayed through D (participants see no prediction, so this is a fair
counterfactual): Anger eval-tick accuracy D 0.94 vs B 0.61, Surprise D 0.50 vs B 0.67, Fear 0 for
both, all other emotions 1.00 for both. Trial macro-F1: raw D 0.76 vs B 0.70; personal D 0.67 vs
B 0.61.

**All 8 `session-local-v1` sessions, D vs B** (replayed; evaluation block, trial-majority
macro-F1; weak tick = no signal above 0.25 anywhere in the window):

| Session | Face size | Weak ticks | D raw | B raw | D personal | B personal |
|---|---|---|---|---|---|---|
| 0916 `85c128` | 2.49 | 18% | 0.64 | 0.56 | 0.66 | 0.68 |
| 0917 `759c71` | 1.33 | 37% | 0.12 | 0.20 | 0.17 | 0.12 |
| 0922 `3f4b7c` | 2.57 | 14% | 0.76 | **0.81** | 0.81 | 0.81 |
| 0922 `b3ec46` | 1.84 | 26% | 0.52 | **0.60** | **0.76** | 0.62 |
| 0922 `0e088a` | 2.59 | 22% | 0.73 | 0.73 | 0.78 | 0.78 |
| 0924 `739a55` | 2.08 | 23% | 0.49 | **0.60** | **0.80** | 0.72 |
| 0930 `26908f` | 2.81 | 6% | 0.52 | **0.57** | 0.57 | 0.57 |
| 1001 `e839ad` (B live) | 2.18 | 19% | **0.76** | 0.70 | **0.67** | 0.61 |
| **Mean** | | | 0.568 | **0.596** | **0.651** | 0.612 |
| **B better / worse / tied** | | | | **5 / 2 / 1** | | **1 / 5 / 2** |

| Evaluation ticks | D | B |
|---|---|---|
| Tick accuracy (mean over sessions) | 0.606 | **0.631** |
| Weak-tick accuracy (178 weak ticks, pooled) | **0.157** | 0.118 |
| Expressive ticks called Neutral | 0.248 | 0.251 |
| Anger | **0.49** | 0.44 |
| Disgust | 0.56 | **0.65** |
| Fear | 0.08 | 0.07 |
| Happiness | 0.85 | 0.84 |
| Neutral | 0.95 | 0.94 |
| Sadness | 0.86 | 0.88 |
| Surprise | 0.46 | **0.60** |

Script: `Analysis/exploratory/d_vs_b_sessions.py` (uses `calibration_study.replay` and
`evaluate`, the same functions as `sessions_four_methods.py`).

**Against the plan's rules (headset):**

| Rule | Result |
|---|---|
| 1. Macro-F1 higher, raw **and** with personal calibration | raw yes (+0.028, 5 of 8 better); personal **no** (−0.039, 5 of 8 worse) |
| 2. More accurate on weak ticks | **no** (0.118 vs 0.157) |
| 3. Happiness, Sadness, Neutral not clearly worse | yes |

So **B does not win on the headset.** The comparison is D vs B, not A vs B, so it does not isolate
augmentation; 8 sessions with 14 evaluation trials each means a ±0.1 change in one session is 1–2
trials.

### 5.10 Model C: D + augmentation (EmojiHero kept)

**What C is.** D's exact training data (old acted 1,553 + EmojiHero 5,784 + new acted 844 = 8,181
windows, three sources, each source one third of the loss), D's validation (old, new and
EmojiHero), D's model and settings, plus volume augmentation with k = 1 (16,362 windows per
epoch). Augmentation is the only intended difference from D. The saved notebook was checked cell
by cell: the corpus, the sample weights (identical to D's: min 0.3436, max 5.2906), the
augmentation (k = 1) and the training cell (only the run name and the `fit` call differ from D's
cell). One remaining difference: C's batches are shuffled by the new dataset code.

**Training.** Best epoch 10 of 18 (early stop). Validation macro-F1 at epoch 10: old acted
0.6966, new acted 0.5966, EmojiHero 0.5731, domain-balanced **0.6221**.

**Test sets** (Cell I, gate tau = 0.7, the same as D):

| Test set | D macro-F1 | C macro-F1 | C − D | D log loss | C log loss ↓ |
|---|---|---|---|---|---|
| Old VR acted | 0.494 | **0.507** | +0.013 | **1.283** | 1.295 |
| New VR acted | 0.414 | **0.467** | **+0.054** | 1.423 | **1.386** |
| **VR combined** | 0.455 | **0.490** | **+0.036** | 1.353 | **1.340** |
| EmojiHero | **0.566** | 0.552 | −0.014 | 1.318 | **1.264** |
| All acted pooled | **0.546** | 0.541 | −0.005 | 1.325 | **1.280** |
| All acted domain-balanced | 0.491 | **0.509** | +0.018 | 1.341 | **1.315** |

VR combined accuracy: D 0.507 (308 correct) → C **0.522 (317 correct)** of 607.

| VR combined, per emotion | Windows | D F1 (correct) | C F1 (correct) |
|---|---|---|---|
| Anger | 84 | 0.24 (15) | **0.37 (34)** |
| Disgust | 109 | **0.46 (49)** | 0.41 (38) |
| Fear | 66 | 0.27 (15) | 0.29 (15) |
| Happiness | 103 | **0.65** (92) | 0.61 (92) |
| Neutral | 87 | 0.81 (86) | **0.90 (83)** |
| Sadness | 82 | 0.40 (30) | **0.51 (34)** |
| Surprise | 76 | 0.35 (21) | 0.35 (21) |

New VR acted only, correct windows D → C: Anger 5 → **13** (of 40), Sadness 15 → 19 (of 37),
Disgust 30 → 25 (of 55), Fear 2 → 3 (of 32).

| VR combined | D | C |
|---|---|---|
| Expressive windows called Neutral (of 520) | 39 | **15** |
| Neutral precision | 0.69 | **0.85** |

| Accuracy by face size (VR combined, expressive) | Windows | D | C |
|---|---|---|---|
| quiet (< 1.5) | 44 | 0.25 | **0.27** |
| 1.5 – 2.5 | 94 | 0.17 | **0.23** |
| 2.5 – 4.0 | 174 | 0.39 | **0.41** |
| loud (> 4.0) | 208 | 0.62 | 0.62 |

**Export.** Cell H with tau = 0.7; 300 of 300 Keras/ONNX predictions agree. On the headset:
`sparseVR_teacher_model_acted_only_vol-aug-k1_with-EmojiHero.onnx`, SHA-256 `899523af…62bd3684`.

**New live session `20261005_141047_dd4b60`** (model C, raw mode, no calibration on the headset).
All 210 ticks: 69.5% tick accuracy, 71.4% trial accuracy. The offline replay matches the headset.

| Emotion (all ticks) | Correct | Most confused with |
|---|---|---|
| Happiness, Sadness, Surprise | 100% | |
| Neutral | 87% | Sadness (10%) |
| Anger | 57% | Neutral (43%) |
| Disgust | **40%** | **Anger (60%)** |
| Fear | **3%** | Neutral (73%), Surprise (23%) |

The same face through the three models (scored ticks 1–9, all trials):

| | D | B | C (live) |
|---|---|---|---|
| All ticks | 0.70 | 0.68 | 0.70 |
| Anger | 0.41 | 0.30 | **0.63** |
| Disgust | 0.56 | **0.85** | 0.37 |
| Prompted Anger → | 11 Anger, 15 Neutral | 8 Anger, 14 Neutral | **17 Anger**, 10 Neutral |
| Prompted Disgust → | 15 Disgust, 6 Anger, 5 Fear | **23 Disgust** | 10 Disgust, **17 Anger** |
| Trial macro-F1, raw (evaluation block) | 0.75 | 0.75 | 0.73 |
| Trial macro-F1, personal (offline) | 0.85 | **0.92** | 0.78 |

**All 9 `session-local-v1` sessions, D vs B vs C** (replayed; evaluation block; script
`Analysis/exploratory/headset_model_comparison.py`):

| Session | Face size | D raw | B raw | C raw | D personal | B personal | C personal |
|---|---|---|---|---|---|---|---|
| 0916 `85c128` | 2.49 | 0.64 | 0.56 | 0.64 | 0.66 | 0.68 | **0.73** |
| 0917 `759c71` | 1.33 | 0.12 | 0.20 | **0.21** | 0.17 | 0.12 | **0.26** |
| 0922 `3f4b7c` | 2.57 | 0.76 | **0.81** | 0.76 | 0.81 | 0.81 | 0.81 |
| 0922 `b3ec46` | 1.84 | 0.52 | **0.60** | 0.42 | **0.76** | 0.62 | 0.62 |
| 0922 `0e088a` | 2.59 | 0.73 | 0.73 | **0.79** | **0.78** | **0.78** | 0.70 |
| 0924 `739a55` | 2.08 | 0.49 | **0.60** | 0.52 | **0.80** | 0.72 | 0.67 |
| 0930 `26908f` | 2.81 | 0.52 | 0.57 | 0.57 | 0.57 | 0.57 | 0.57 |
| 1001 `e839ad` (B live) | 2.18 | 0.76 | 0.70 | 0.76 | 0.67 | 0.61 | **0.76** |
| 1005 `dd4b60` (C live) | 2.48 | 0.75 | 0.75 | 0.73 | 0.85 | **0.92** | 0.78 |

| Mean over 9 sessions | D | B | C |
|---|---|---|---|
| Trial macro-F1, raw | 0.588 | **0.613** | 0.600 |
| Trial macro-F1, personal | **0.672** | 0.646 | 0.656 |
| Tick accuracy | 0.620 | **0.638** | 0.634 |
| Weak-tick accuracy (200 weak ticks) | 0.155 | 0.120 | **0.175** |
| Expressive ticks called Neutral | 0.245 | 0.249 | **0.215** |
| Anger | 0.49 | 0.43 | **0.60** |
| Disgust | 0.59 | **0.67** | 0.47 |
| Fear | 0.07 | 0.06 | **0.10** |
| Happiness | 0.86 | 0.86 | **0.90** |
| Neutral | **0.94** | 0.92 | 0.90 |
| Sadness | 0.88 | **0.90** | 0.89 |
| Surprise | 0.52 | **0.64** | 0.57 |

C − D trial macro-F1: raw **+0.012** (better 4, worse 2, tied 3); personal **−0.016** (better 3,
worse 4, tied 2).

| Rule (headset) | C |
|---|---|
| 1. Higher raw **and** personal | raw yes (+0.012); personal no (−0.016), within noise |
| 2. Better on weak ticks | **yes** (0.175 vs 0.155) |
| 3. Happiness, Sadness, Neutral not clearly worse | yes (Neutral −0.04) |

### 5.11 Why Fear fails

Script: `Analysis/exploratory/fear_analysis.py`. 243 headset Fear ticks (scored ticks, 9
sessions), compared with old acted, new acted and EmojiHero training-style windows.

**1. Headset Fear faces are almost resting faces.**

| Median face size | Headset | Old acted | New acted | EmojiHero |
|---|---|---|---|---|
| Fear | **0.73** | 3.02 | 2.02 | 3.31 |
| Surprise | 1.52 | 2.83 | 2.41 | 3.63 |
| Neutral | 0.56 | 0.67 | 0.45 | 1.12 |

Headset Fear (0.73) is 3–4× smaller than training Fear and close to Neutral (0.56). 53% of headset
Fear ticks have no signal above 0.25 (Surprise 27%, Neutral 65%).

**2. Headset Fear has a different shape: raised brows, like Surprise.**

| | Top muscles |
|---|---|
| Headset Fear | OuterBrowRaiser, InnerBrowRaiser, UpperLidRaiser, JawDrop |
| Headset Surprise | InnerBrowRaiser, OuterBrowRaiser, JawDrop (the same muscles, about 2–3× stronger) |
| Old acted Fear | **UpperLidRaiser, LidTightener**, UpperLipRaiser, CheekRaiser |
| New acted Fear | UpperLidRaiser, **LidTightener**, InnerBrowRaiser, BrowLowerer |
| EmojiHero Fear | UpperLidRaiser, InnerBrowRaiser, OuterBrowRaiser, JawDrop |

Training Fear has tense eyes (LidTightener); headset Fear does not.

**3. Headset Fear looks more like training Surprise than training Fear** (cosine similarity of
size-normalised profiles, so shape only):

| Training collection | Fear | Surprise | Neutral |
|---|---|---|---|
| Old acted | 0.62 | **0.79** | 0.46 |
| New acted | 0.74 | **0.86** | 0.47 |
| EmojiHero | 0.84 | **0.87** | 0.66 |

Per session, headset Fear is most like training Surprise in 7 of 9 sessions and most like Neutral
in 2 (two of the smallest faces). Even in training, Fear and Surprise are very alike (cosine 0.90
in every collection), so the model can barely separate them.

**4. What the models do with headset Fear.**

| | → Neutral | → Surprise | → Fear | Ticks called Fear | of which truly Fear |
|---|---|---|---|---|---|
| D | 60% | 28% | 8% | 135 | 20 (66 are prompted Surprise) |
| B | 63% | 29% | 6% | 67 | 15 |
| C | 55% | 35% | 9% | 102 | 22 (57 are prompted Surprise) |

**Conclusion.** Fear fails for two reasons together. **Volume:** headset Fear is about as small as
a resting face, so it is called Neutral. **Shape:** what our headset users do for Fear is a small
Surprise (raised brows), not the tense-eyed Fear of the training data, so the rest is called
Surprise. And when a model does say "Fear", it is usually wrong (mostly prompted Surprise). Volume
augmentation can help only the first part.

### 5.12 How many copies, and how strong (steps A and B, trained offline)

**Question.** Is model C's augmentation (one copy per window per epoch, multipliers up to ×1/3–×3)
too much?

**Method.**
- **Data:** the exact arrays of model D, exported from Colab
  (`notebooks/teacher_arrays_with_emojihero.npz`). The deployed D reproduces its notebook test
  results on these arrays to 4 digits.
- **Training:** on this computer (CPU) with the notebook's architecture cell and model D's settings:
  three validation domains, early stopping with patience 8, gate tau 0.7 for evaluation.
- **"Add" mode:** every epoch shows all originals, plus a copy of a random fraction of the
  windows.
- **Strong** is model C's rule. **Mild** is the same rule, also kept to ×0.6–×1.6.
- Script: `Analysis/exploratory/aug_amount_strength.py`. Results:
  `Analysis/results/aug_amount_strength*_seed{42,1}.csv`.

**Seed 42, all runs (test sets):**

| Run | Val domain-bal | VR combined | New VR | EmojiHero | Pooled | Domain-bal | VR correct | False Neutral | Quiet acc. |
|---|---|---|---|---|---|---|---|---|---|
| A0 (no copies) | **0.629** | 0.464 | 0.407 | 0.552 | 0.539 | 0.494 | 313 | 28 | 0.14 |
| A20 (+20%, strong) | 0.627 | 0.469 | 0.441 | 0.563 | 0.551 | 0.499 | 319 | 35 | 0.20 |
| A50 (+50%, strong) | 0.622 | 0.476 | **0.451** | 0.556 | 0.548 | 0.498 | 323 | 33 | 0.20 |
| A100 (+100%, strong) | 0.619 | **0.490** | 0.449 | 0.563 | **0.556** | **0.509** | **333** | **27** | **0.23** |
| B20 (+20%, mild) | 0.624 | 0.465 | 0.426 | 0.562 | 0.549 | 0.498 | 316 | 28 | 0.11 |

**A0 and A100 with a second seed (seed 1):**

| Run | Seed | VR combined | New VR | EmojiHero | Domain-bal | VR correct | False Neutral | Quiet acc. |
|---|---|---|---|---|---|---|---|---|
| A0 | 42 | 0.464 | 0.407 | 0.552 | 0.494 | 313 | 28 | 0.14 |
| A100 | 42 | 0.490 | 0.449 | 0.563 | 0.509 | 333 | 27 | 0.23 |
| A0 | 1 | 0.440 | 0.405 | 0.557 | 0.483 | 297 | 36 | 0.11 |
| A100 | 1 | 0.444 | 0.433 | 0.565 | 0.488 | 292 | 20 | 0.16 |

| A100 − A0 | Seed 42 | Seed 1 | Same direction? |
|---|---|---|---|
| VR combined macro-F1 | +0.026 | +0.004 | yes, but small in seed 1 |
| VR correct windows | +20 | −5 | **no** |
| New VR macro-F1 | +0.042 | +0.028 | **yes** |
| EmojiHero macro-F1 | +0.011 | +0.008 | yes |
| Domain-balanced macro-F1 | +0.016 | +0.005 | yes |
| False Neutral | −1 | −16 | **yes** |
| Quiet-window accuracy | +0.09 | +0.05 | **yes** |
| Fear / Surprise F1 | +0.14 / −0.09 | −0.06 / +0.06 | **no (reverses)** |

**Findings.**
1. **Run-to-run noise is large.** The same setting (A0) with two seeds differs by 0.024 VR
   combined macro-F1 (313 vs 297 correct windows). That is as large as the seed-42 gain of A100.
   One run per setting cannot rank A20, A50 and A100.
2. **Augmentation consistently helps where it was built to help:** in both seeds it raises new VR
   macro-F1 (+0.03 to +0.04) and quiet-window accuracy, and lowers or keeps false Neutrals. It
   does not hurt EmojiHero.
3. **The overall VR gain is small and uncertain:** +0.026 in one seed and +0.004 in the other.
4. **The Fear ↔ Surprise shift is noise.** In seed 42, A100 moved the Fear/Surprise boundary toward
   Fear; in seed 1, toward Surprise. Neither direction is an augmentation effect. In seed 42 all 10
   lost Surprise windows went to Fear; Fear + Surprise together went from 32 to 38 correct.
5. **Mild is not better than strong** (B20 vs A20, one seed): it is lower on new VR and on quiet
   windows. The concern that the strong rule is too much is not supported.
6. **Validation cannot choose the amount:** all runs are within 0.619–0.632 on validation, which
   has only 2 new acted people.
7. Unrelated to augmentation: on VR, Surprise is often called Happiness (31 of 76 windows with
   A0, seed 42).

**Decision.** Keep model C's setting (+100%, strong). It is not harmful, and it gives a consistent
gain on quiet faces. To rank amounts or to show an overall gain, average several seeds per setting
(about 3 minutes per run offline).

### 5.13 Is the +100% gain real? Five seeds (offline)

**Method.** A0 (no copies, a "D twin") and A100 (+100% strong, model C's setting) with 3 more
seeds (7, 123, 2024), with the same script, data and settings as 5.12. The seed changes only the
start weights, the window order and the copy multipliers, so A0 and A100 with the same seed are a
fair pair. Results: `Analysis/results/aug_amount_strength_seed{42,1,7,123,2024}.csv`.

**A100 − A0, test sets (paired over 5 seeds):**

| Metric | A0 mean ± sd | A100 mean ± sd | Mean change | Seeds better / worse | Paired t-test p |
|---|---|---|---|---|---|
| VR combined macro-F1 | 0.457 ± 0.012 | 0.476 ± 0.021 | +0.020 | 5 / 0 | 0.07 |
| **New VR macro-F1** | 0.417 ± 0.015 | 0.457 ± 0.018 | **+0.040** | **5 / 0** | **0.004** |
| EmojiHero macro-F1 | 0.561 ± 0.009 | 0.560 ± 0.009 | −0.001 | 3 / 2 | 0.86 |
| Pooled macro-F1 | 0.545 | 0.548 | +0.004 | 3 / 2 | 0.54 |
| Domain-balanced macro-F1 | 0.492 | 0.505 | +0.013 | 4 / 1 | 0.09 |
| VR correct windows (of 607) | 309.2 | 314.8 | +5.6 | 3 / 2 | 0.42 |
| **False Neutral (of 520)** | 33.4 | 21.6 | **−11.8** | **5 / 0 fewer** | **0.02** |
| **Quiet-window accuracy (n = 44)** | 0.18 | 0.23 | **+0.055** | **4 / 0** (1 tie) | **0.03** |
| Log loss VR | 1.268 | 1.300 | +0.032 (worse) | 2 / 3 | 0.31 |
| Validation domain-bal | 0.625 | 0.618 | −0.007 | 0 / 5 | 0.04 |

**F1 per emotion, VR combined (mean over 5 seeds):** Anger 0.23 → 0.29 (better in 4 of 5),
Sadness 0.42 → 0.46 (5 of 5), Surprise 0.21 → 0.23 (4 of 5), Happiness 0.65 → 0.63 (worse in 4
of 5); Disgust, Fear and Neutral are mixed.

**Findings.**
1. **The gain on quiet faces is real.** New VR macro-F1, false Neutrals and quiet-window accuracy
   improve in every seed (p ≤ 0.03).
2. **The overall VR gain is probably real but small:** +0.020, positive in 5 of 5 seeds, but p =
   0.07 and two seeds are near zero. The number of correct windows does not change reliably.
3. **EmojiHero is not hurt** (−0.001).
4. **Validation cannot see the gain:** A100 is slightly lower on validation in every seed. The
   validation set has only 2 new acted people and few quiet faces.
5. **The probabilities are not better** (log loss VR mixed). The earlier log-loss gain of C was
   one seed.
6. Model C's Colab result (0.490) is at the top of the A100 range (0.444–0.491).

### 5.14 Augmentation vs personal calibration (headset)

Same 9 sessions as 5.9 (8 of them the developer's face), evaluation block, trial macro-F1:

| | Raw | With personal calibration | Gain from calibration |
|---|---|---|---|
| D | 0.588 | **0.672** | **+0.084** (better in 8 of 9 sessions) |
| C | 0.600 | 0.656 | +0.056 |
| Gain from augmentation | +0.012 (4 better / 2 worse / 3 tied) | −0.016 | |

1. **Calibration helps about 7× more than augmentation** on the headset, and it helps in 8 of 9
   sessions (sign test p ≈ 0.04).
2. **The two do not add up:** D + calibration is better than C + calibration. Both seem to fix
   the same thing (the person's face differs from the training faces), and calibration fixes more
   of it.
3. **Augmentation is useful when there is no calibration** (raw +0.012, and the quiet-face gains
   of 5.13). It costs nothing in the game.
4. Limits: one person in 8 of 9 sessions; calibration cannot be measured on the test set (no
   calibration block there).

---

## 6. Interpretation

1. **Removing EmojiHero hurt.** New VR −0.051, EmojiHero −0.070, old VR about the same. EmojiHero
   training was helping our VR test people a little.
2. **Augmentation recovered most of that loss.** B is +0.013 over A on VR combined and +0.023 on new
   VR, the quietest group, and is back at D's level (0.452 vs 0.455).
3. **Augmentation did what it was built for: it halved the false Neutrals** (53 → 27 of 520
   expressive windows). The model learned that a weaker face can still be an emotion.
4. **But the freed windows mostly went to Disgust,** not to Anger, Fear or Surprise. Those three stayed
   flat; on new VR, Anger and Fear are almost never correct for any model without EmojiHero.
   Augmentation fixed "is this an emotion at all?", not "which emotion is it?" for these three.
   That is a shape problem, which volume cannot fix (as in the Sep 30 session; Fear is weak in
   every dataset).
5. **Exaggerators were not hurt by augmentation:** EmojiHero A vs B is −0.005.
6. **On all acted test sets together (5.8),** D is best on accuracy and F1 because those sets
   include EmojiHero. A and B are almost equal on the pooled row (F1 0.488 vs 0.486); B is slightly
   ahead on the domain-balanced row (0.458 vs 0.455).
7. **Augmentation improved the probabilities:** B's log loss is lower than A's on both rows, and on
   the domain-balanced row lower than D's (1.303 vs 1.341).

**Against the plan's criteria (test set only):**

| Criterion | Result |
|---|---|
| B macro-F1 > A | yes: +0.013 VR combined, +0.023 new VR |
| Happiness, Sadness, Neutral not clearly worse | Neutral +0.08 and Sadness +0.03; Happiness −0.05 F1, but only 2 fewer correct windows (91 → 89) |
| Exaggerators not hurt | yes: EmojiHero −0.005 |

**Caution:** 607 windows from 7 people is a small test. The +0.013 overall is 7 more correct
windows (301 → 308). Only the Neutral and Disgust changes are clearly visible; the rest is within
noise. The test set is not the deciding test. These test numbers also used tau = 2.0 (5.9) and
need a re-run.

**On the headset (5.9):**

8. **Raw, B is slightly better than D** (+0.028 trial macro-F1, better in 5 of 8 sessions), mainly
   from Surprise (+14 points) and Disgust (+9). Anger is slightly worse (−5).
9. **With personal calibration, D is better** (−0.039, B worse in 5 of 8). Calibration adds +0.083
   on top of D but only +0.016 on top of B. The test-set hint that B's better probabilities would
   help calibration did not hold. Why calibration gains less with B is not yet known.
10. **The weak-face goal was not met on the headset.** B is less accurate on weak ticks than D
    (0.12 vs 0.16, 178 ticks) and calls expressive ticks Neutral just as often (25%). The halving
    of false Neutrals on the test set did not carry over to the headset.
11. **Fear is about 0% on the headset for both models**, mostly called Neutral (23 of 30 ticks in
    the new session). It is the clearest remaining problem.
12. **The new live session shows the trade-off in one person:** D would have scored higher there
    (raw 0.76 vs 0.70) because of Anger (0.94 vs 0.61), while B was better on Surprise.

**Model C (5.10) and Fear (5.11):**

13. **Keeping EmojiHero and adding augmentation works better than either change alone.** On the
    test sets C is the best model (VR combined +0.036 over D, new VR +0.054), with the most
    trustworthy probabilities (lower log loss on 5 of 6 rows) and 60% fewer false Neutrals.
14. **On the headset C helps where augmentation was meant to help:** the best weak-tick accuracy
    (0.175 vs D 0.155) and the fewest Neutral calls (21.5% vs 24.5%). Anger improves strongly
    (0.49 → 0.60), as on the test set.
15. **C trades Disgust for Anger.** Disgust drops on the headset (0.59 → 0.47); in the C live
    session 17 of 27 Disgust ticks were called Anger. C pushes faces with lowered brows toward
    Anger.
16. **Overall on the headset C and D are about equal:** C − D is +0.012 raw and −0.016 with personal
    calibration, both within noise for 9 sessions. C is clearly better than B.
17. **Fear needs something other than volume.** Headset Fear is both tiny and Surprise-shaped
    (5.11). Augmentation can address the size, not the shape.

---

## 7. Limits

- **No new people.** Training is 39 people (24 old acted, 15 new acted), each acting each emotion
  once.
- **No new muscle patterns,** only new strengths of existing ones.
- **The 0.8 target came from the headset sessions** that will judge the result.
- **The 0.25 rule still removes weak real frames** before augmentation sees them.
- **The test set has few quiet faces** (44 of 520 expressive windows), so it can show only a
  small part of what augmentation is meant to fix.
- **Two things differ between D and B.** Only A separates them; A and B were trained once each,
  with one seed.
- **C was trained once, with one seed,** and its batches are shuffled by the new dataset code, not
  by D's `model.fit` on arrays. A "D twin" (same code, k = 0) and more seeds would remove both
  doubts.
- **9 headset sessions, possibly fewer people.** Two of them were recorded live with B and C, and
  the 0.8 target came from the earlier ones.

---

## 8. Next steps

1. ~~Model C: D + augmentation (EmojiHero kept).~~ Done (5.10).
2. ~~Check that C's gain is stable.~~ Done offline with 5 seeds (5.13): the quiet-face gain is
   real, the overall gain is small (+0.020). Calibration helps more on the headset (5.14).
3. **Disgust vs Anger in C.** Compare headset Disgust ticks with training Anger and Disgust, and
   find out why C moves Disgust to Anger.
4. **Fear (5.11).** Volume cannot fix the shape. Options, one at a time:
   - change the acting instruction for Fear in the study (for example "eyes wide and tense, lips
     stretched"), so that headset Fear matches what the training data calls Fear;
   - record more headset Fear from new participants;
   - merge Fear and Surprise into one class, and report the 6-class result next to the 7-class one.
5. **Re-run test Cells D–G with tau = 0.7** for A and B, and correct sections 5.3–5.8.
6. **Model E: "tone down EmojiHero"** (shrink EmojiHero windows to our face sizes before
   training), compared with C.
7. **Optional later experiments, one at a time, chosen on validation:**
   - cap the multiplier at 1 ÷ (largest signal), so no copy is ever clipped;
   - k = 2;
   - augment before the 0.25 rule, or lower it.
8. **New participants** with more than one trial per emotion, to confirm any win on people who were
   not used for any choice.
