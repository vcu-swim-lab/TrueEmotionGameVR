# Per-Window Volume Augmentation: Plan and Evidence

**Status:** plan agreed, not implemented. Last updated 2026-09-29.
**Notebook:** `notebooks/teacher_notebook_only_acted_data.ipynb`
**Supporting analysis:**
- the Part 5.7 exaggeration cell in that notebook, with output in
  `Analysis/data/exaggeration_analysis-20260929T144543Z-1-001/`
- the exploratory scripts in `Analysis/exploratory/`

This document replaces the 2026-09-24 version, which proposed one global volume range for every
window. The per-window rule below replaces it; section 6 explains why.

---

## 1. The problem in one paragraph

On the headset, the teacher calls a face "Neutral" whenever it moves only a little. Anger, Fear
and Surprise suffer most. Two findings explain it:

1. Our participants move their faces much less than the EmojiHero participants the model was
   partly trained on.
2. The training notebook only keeps a window when some signal goes above 0.25, so the model
   seldom sees weak expressions of any emotion.

The goal of this plan is to teach the model that **weak faces are still emotions**, without
recording new people, by making training copies of real windows at different strengths.

## 2. Evidence

### 2.1 Headset sessions: small faces are the failure

These numbers come from the 6 `session-local-v1` headset sessions and the current acted-only
teacher. Script: `Analysis/exploratory/gate_check.py`.

| Expressive headset ticks | Value |
|---|---|
| Share too small to pass the notebook's 0.25 rule | **32%** |
| Accuracy on those small ticks | **20%** |
| Accuracy on ticks that pass the rule | 71% |
| Small ticks the model calls Neutral | **72%** (vs 5% on ticks that pass) |

Share of ticks too small for the rule, per emotion:

| Anger | Fear | Surprise | Disgust | Happiness | Sadness |
|---|---|---|---|---|---|
| **60%** | **67%** | **48%** | 6% | 5% | 7% |

The three emotions the model fails on are the three that people express weakly.

### 2.2 How exaggerated is each data collection

These numbers come from the exaggeration cell (Part 5.7) of the teacher notebook, which covers
every recording of old acted, new acted and all EmojiHero splits.

**Face size** is the sum of the 63 non-tongue signals in a frame, averaged over a recording's
frames (a Neutral face is about 0.6–1.2). Each value in the table is the **median person**; the
median is used because a few very expressive people would pull an average up.

| Emotion | Old acted (24) | New acted (15) | EmojiHero (36) | Order |
|---|---|---|---|---|
| Anger | 2.70 | **1.88** | 5.11 | EmojiHero > old > new |
| Disgust | 3.89 | **2.55** | 5.56 | EmojiHero > old > new |
| Fear | 2.26 | **1.15** | 3.13 | EmojiHero > old > new |
| Happiness | 5.42 | **3.12** | 5.21 | old > EmojiHero > new |
| Sadness | 2.49 | **2.02** | 4.14 | EmojiHero > old > new |
| Surprise | 2.58 | **1.49** | 3.30 | EmojiHero > old > new |
| Neutral | 1.21 | 0.64 | 1.03 | old > EmojiHero > new |

- The three collections differ for every expressive emotion: Kruskal–Wallis p ≤ 0.0011.
- New acted Fear (1.15) is about the size of an old acted *Neutral* face (1.21).

**Per person** (average over the 6 expressive emotions):

| Collection | Smallest | Median | Largest | Spread between people (CV) | Signals moving (median, of 70) |
|---|---|---|---|---|---|
| Old acted | 1.71 | 3.20 | 5.52 | 0.35 | 46 |
| New acted | **0.75** | **2.20** | 3.83 | 0.39 | **41** |
| EmojiHero | 2.26 | 4.47 | 7.60 | 0.26 | 52.5 |

Other findings:
- **People differ a lot.** From the smallest to the largest person is about 10×, and each person
  uses a different set of signals (plot 6, person × signal heatmap).
- **The 0.25 rule removes weak faces.** It keeps only 46–54% of new acted Anger, Fear and
  Surprise frames, vs 82–91% for EmojiHero (plot 7).
- **The mouth carries most of the signal.** EmojiHero people also use brows and eyes much more
  (plot 5).
- **Some signals barely move for anyone:** tongue, CheekPuff, CheekSuck, LipPressor,
  MouthLeft/Right and JawSideways.
- **Values almost never reach the 1.0 ceiling** (0.07–0.27% of values), so there is room to scale up.

### 2.3 Earlier comparisons that led here (2026-09-24)

- **Calibration on the headset.**
  - Personal calibration with the acted-only teacher beat raw in 6 of 6 sessions (macro-F1
    0.544 → 0.662).
  - Hybrid calibration, which blends in the EmojiHero population, did not beat personal on the
    headset, although it did on EmojiHero.
- **Datasets.** The four datasets differ in face size: EmojiHero 4.52, old acted 3.08, headset
  2.24, new acted 1.99 (mean per frame). Across all 61 people, bigger faces meant higher accuracy
  (face size 0–2: 44%, 4 and up: 59%).
- **Auto-volume**, a test-time idea that was checked but is not part of this plan.
  - Scaling each person's input by a label-free gain helped headset sessions (0.544 → 0.621) and
    EmojiHero people (0.575 → 0.595).
  - A fixed gain helped the headset but hurt EmojiHero in 15 of 16 people.
  - The lesson: the scaling must adapt to the face, not be one number for everyone.

## 3. Decisions

1. **Train on old acted and new acted only.** EmojiHero is dropped from training, because its
   faces are already about 2× ours. It stays available as a reference only.
2. **Add per-window volume augmentation** (section 4) to the training windows.
3. **Validation and test windows are never changed.**
4. **Early stopping and checkpoint selection use old acted and new acted validation only.**
5. **Compare two models that are identical except for the augmentation** (section 7).

## 4. The augmentation rule

### 4.1 Where it happens

The notebook already:
1. reads each recording;
2. cuts it into windows of 5 frames × 70 signals;
3. keeps windows that pass the 0.25 rule.

**Augmentation is applied after step 3, to training windows only, every time a batch is made.**
The 0.25 rule itself is unchanged.

### 4.2 Face size of a window

For each of the 5 frames, add the 63 non-tongue signals, then average the 5 sums. The tongue is
left out because EmojiHero has no tongue data and the target range (4.4) was measured without it.

### 4.3 Expressive windows (every emotion except Neutral)

For each window, each time it is used:

1. `size` = face size of the window.
2. Allowed multiply range:
   - `low = 0.8 / size` and `high = 5.5 / size`
   - both clipped to `[1/3, 3]`
3. With probability 0.2, keep the window as it is.
4. Otherwise pick a multiply number `g` between `low` and `high`, **log-uniformly**, so halving
   and doubling are equally likely.
5. Multiply **all 70 signals in all 5 frames** by `g`, then clip to `[0, 1]`.
6. Keep the label.

Worked example: a real Anger window from new-acted participant `P38`, size 4.93.

| | Calculation | Result |
|---|---|---|
| low | 0.8 / 4.93 = 0.16, raised to the cap | **×0.33** |
| high | 5.5 / 4.93 | **×1.12** |
| Copies | | size 1.6 to 5.5, mostly quieter |

| Signal (frame mean) | Real | × 0.5 |
|---|---|---|
| LidTightenerL | 0.425 | 0.212 |
| BrowLowererL | 0.349 | 0.175 |
| BrowLowererR | 0.338 | 0.169 |

The same muscles in the same balance, just quieter, and still labelled Anger.

Different windows get different ranges:

| Window size | Allowed multiply | Copies land at | Effect |
|---|---|---|---|
| 7.0 (very strong) | ×0.33 – ×0.79 | 2.3 – 5.5 | only quieter |
| 4.93 (the example) | ×0.33 – ×1.12 | 1.6 – 5.5 | mostly quieter |
| 2.2 (typical) | ×0.36 – ×2.5 | 0.8 – 5.5 | quieter or louder |
| 1.0 (quiet) | ×0.8 – ×3 | 0.8 – 3.0 | mostly louder |

Edge cases: the caps override the target.
- A window below size 0.27 is multiplied by ×3 and stays below 0.8.
- A window above size 16.5 is multiplied by ×0.33 and stays above 5.5.

### 4.4 Neutral windows

Neutral means a resting face, so it must stay neutral.

1. Pick `g` log-uniformly between ×0.5 and ×1.5, again keeping 20% of windows real.
2. If `g` would push any signal above 0.25, lower `g` so that the largest signal is exactly 0.25.
   This is the notebook's own Neutral rule.

### 4.5 What stays the same

- Labels, class weights and source weights stay as they are.
- The number of windows per epoch stays the same, about 2,400. Nothing is saved, and every epoch
  sees freshly scaled copies.
- The model architecture stays the same, so ONNX export and the headset runtime don't change.
- Signals a person never moved stay at 0, since 0 × g = 0. Nobody gains muscles they didn't use.

## 5. The numbers and where they come from

| Setting | Value | Source | How solid |
|---|---|---|---|
| Quietest target size | 0.8 | 10% of expressive headset ticks are below ~0.6; rounded up so copies are not pure tracking noise | measured, then rounded |
| Loudest target size | 5.5 | 90% of all 75 people are below 5.47 (section 2.2) | measured |
| Max multiply up | ×3 | beyond this, noise is amplified; real exaggeration also recruits extra muscles that scaling can't create | judgment |
| Max multiply down | ×1/3 | the mirror of ×3 | judgment |
| Share kept real | 20% | common practice; keeps real recordings in every epoch | judgment |
| Neutral range | ×0.5 – ×1.5, max signal 0.25 | 0.25 is the notebook's rule; the range is a judgment | partly measured |

One inconsistency in how these were measured, stated openly:
- 5.5 is a per-person average;
- 0.8 comes from single headset ticks.

Augmentation acts on single windows, so a window-level lower bound is the relevant one. But 0.8 is
the least certain number.

**These values are frozen for the first experiment.** Variations (section 8) come only after
A and B have been compared.

## 6. Why per window, not per person

The first version gave each person one multiply range based on their average face size. Checking
it against a real window showed the flaw.

The example window above has size **4.93**, while its owner's average is **2.2**:
- the 0.25 rule keeps only the stronger frames;
- Anger is this person's strongest emotion.

With the per-person range (×0.36 – ×2.49), a ×2 copy reaches **9.55**, far beyond any real face.

Computing the range from the window itself guarantees every copy lands in 0.8–5.5 (unless capped).
It still keeps the person's own muscles and balance, and it needs no person ID per window, which
makes the code simpler.

## 7. The experiment

| | Model A | Model B |
|---|---|---|
| Training data | old acted + new acted | old acted + new acted |
| Volume augmentation | no | yes (section 4) |
| Seed, split, epochs, early stopping, weights | identical | identical |
| Checkpoint selection | old + new acted validation | old + new acted validation |

Steps:
1. Edit the notebook:
   - remove EmojiHero from training (cells 55 and 56);
   - select checkpoints on old + new acted validation (cell 69);
   - add the augmentation cell before Part 6;
   - feed it to `model.fit`;
   - give each run its own `NEW_TEACHER_TAG`.
2. Train A and B in Colab.
3. Export both to ONNX with the existing `tf2onnx` cells.
4. Evaluate both on the 6 headset sessions.

## 8. How the result is judged

On the 6 headset sessions (evaluation block, trial-majority), **B wins** if all three hold:

1. B's macro-F1 is higher than A's, with raw argmax and with personal calibration on top;
2. B is more accurate on **small ticks** (ticks that fail the 0.25 rule; 20% today);
3. B is not clearly worse on Happiness, Sadness or Neutral: at most one trial lower in each.

Also reported, but not part of the decision:
- Anger, Fear and Surprise recall;
- auto-volume on top;
- EmojiHero test people, as a check that exaggerators are not hurt.

If B wins, augmentation stays, and these variations can follow, one at a time, chosen on
**validation** people only:
- **B-safe:** target 1.2–5.5, cap ×2
- **B-wide:** target 0.6–6.5

If B doesn't win, find out why before changing anything.

## 9. Limits

- **No new people.** Training is still 39 people, and each did each emotion once.
- **No new muscle patterns,** only new strengths of existing ones. Real exaggeration sometimes
  recruits extra muscles, and scaled copies can't.
- **The headset test is small:** 6 sessions, possibly fewer than 6 people. Session `759c71` looks
  broken: its calibration and evaluation blocks disagree.
- **The 0.8 lower bound was chosen by looking at the same headset sessions** that are used to
  judge the result. It is kept as a coarse, rounded value, and any win must be confirmed on new
  participants.
- **The 0.25 rule still removes weak real frames before augmentation sees them.** Changing that
  rule is a separate, later experiment.

## 10. Later, not in this plan

- Record 6–8 new headset participants, each with an anonymous participant ID, and 3 trials per
  emotion.
- Try augmenting before the 0.25 rule, or lowering it.
- Fear is weak in every dataset. It may need more data, or merging with Surprise.
