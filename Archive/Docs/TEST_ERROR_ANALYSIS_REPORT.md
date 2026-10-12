# Test Error Analysis of the Acted-Only Teacher (Model D)

Date: 2026-10-08. Model: D, the deployed acted-only teacher (`sparseVR_teacher_model_acted_only.onnx`).

Two kinds of results are kept apart in this report:

- **Test**: offline test windows from `notebooks/teacher_arrays_with_emojihero.npz`
  (5 frames × 70 AUs, one score per window).
- **Headset**: offline replay of the 10 recorded headset sessions (one score per tick, one tick per
  second). Nine of the ten sessions are the developer's own face. Only `26908f` is another person.

Neither is a live result. The test and headset data have shaped this research, so they are no longer
untouched final evaluations.

---

## 1. Main error analysis (test windows)

Script: `Analysis/exploratory/test_error_analysis.py`. Output: `Analysis/results/test_error_windows.csv`.

D got **1,229 of 2,875** test windows wrong (43%).

### 1.1 Kind of mistake

| Kind | EmojiHero | New acted | Old acted | All |
|---|---|---|---|---|
| Emotion read as another emotion | 687 | 124 | 135 | 946 (77%) |
| Emotion read as Neutral | 177 | 29 | 10 | 216 (18%) |
| Neutral read as an emotion | 66 | 1 | 0 | 67 (5%) |

Biggest confusions: Fear→Surprise 158, Anger→Disgust 115, Fear→Neutral 69, Disgust→Anger 66,
Sadness→Disgust 60.

### 1.2 Likely cause (one bucket per error, first matching rule)

| # | Cause | Rule | Errors |
|---|---|---|---|
| 1 | No expression | Emotion label, but every AU above 0.25 is gaze, blink or tongue | 100 (8%) |
| 2 | Unlike training | Farther from every training window than 95% of validation windows | 54 (4%) |
| 3 | Looks like the predicted class | Most of the 15 nearest training windows carry the predicted label | 553 (45%) |
| 4 | Near miss | True class ranked 2nd, less than 0.2 behind | 99 (8%) |
| 5 | Model miss | Nearest training windows carry the true label, yet D is wrong | 167 (14%) |
| 6 | Unclear | Nearest training windows vote for a third class | 256 (21%) |

### 1.3 Per emotion

| True class | Error rate | Main confusion |
|---|---|---|
| Fear | 86% | Surprise |
| Anger | 64% | Disgust, Sadness |
| Disgust | 58% | spread over Anger, Happiness, Sadness, Fear |
| Sadness | 34% | Disgust |
| Surprise | 31% | Happiness (new acted) |
| Neutral | 16% | — |
| Happiness | 13% | — |

### 1.4 Main points

- About two-thirds of the errors (buckets 3 and 6) are not supported by the training data near that
  face. Only about 22% (buckets 4 and 5) look fixable by a better model.
- **Errors repeat within a recording.** In old and new acted data, 79% of errors have a neighbouring
  window in the same recording that is wrong in the same way. The errors come from how a person makes
  the face, not from random noise.
- **Emotion→Neutral errors are weak faces**: median face size 1.3, against 4.8 for correctly read
  emotion windows.
- A plain nearest-neighbour vote on raw AUs scores about the same as D (0.563 vs 0.546 pooled test
  macro-F1). The model has learned what the data shows.

**Limits.** The npz has no participant or recording IDs, so "repeats within a recording" is an
estimate. Buckets 3, 5 and 6 depend on a neighbour vote; the shares were stable for K = 5, 15 and 30.

---

## 2. Follow-up checks

Scripts: `error_category_checks.py`, `duplicate_windows.py`.

### 2.1 No-expression windows

| | Train | Val | Test |
|---|---|---|---|
| EmojiHero | 436 (9.5%) | 85 | 86 |
| New acted | 79 (11.3%) | 13 | 20 |
| Old acted | 70 (5.2%) | 2 | 4 |

- The AU that let them through the filter is gaze (547) or blink (105). The strongest expressive AU
  has a median of 0.16, against 0.71 in other emotion windows.
- 17 of 24 acted test cases sit in the first or last two windows of a recording.
- D predicts Neutral for 87 of 110. That is a sensible reading of the face.
- Without them, test macro-F1 rises a little: EmojiHero 0.566 → 0.584, new 0.414 → 0.426,
  old 0.494 → 0.501.
- This is a filter problem: gaze and blink count as "expression". An earlier experiment showed that
  filtering EmojiHero hurt, so removing weak windows has to be tested, not assumed.

### 2.2 Fear→Surprise

FACS cues used: Fear = brow lowerer, lid tightener, lip stretcher; Surprise = jaw drop with brows up.

| Training windows labelled Fear | Only Fear cues | Only Surprise cues | Both | Neither |
|---|---|---|---|---|
| Old acted | 45% | 9% | 9% | 36% |
| EmojiHero | 11% | 49% | 20% | 19% |
| New acted | 24% | 6% | 13% | 57% |

- 136 of the 158 Fear→Surprise test errors come from EmojiHero.
- EmojiHero acts Fear as a surprise face. Whole clips fail together (0–23% accuracy inside the
  largest groups of similar windows).
- This is a label/data problem in EmojiHero Fear. No model can recover "Fear" from AUs that show a
  surprise face.

### 2.3 Anger→Disgust

| Training windows labelled Anger | Only Anger cues | Only Disgust cues | Both | Neither |
|---|---|---|---|---|
| Old acted | 23% | 48% | 5% | 23% |
| EmojiHero | 15% | 42% | 11% | 32% |
| New acted | 20% | 30% | 9% | 41% |

- Misread Anger windows show disgust cues (nose wrinkler about 0.76, upper lip raiser about 0.86).
- Actors in every source wrinkle the nose in Anger; FACS lists this as an Anger variant.
- Not a label error: the two classes overlap in AU space.

### 2.4 Duplicates and leakage

No exact duplicates. At most one test window is a near-copy of a training window. No leakage.

---

## 3. Retrains based on the errors

Script: `targeted_retrain.py`. Results: `Analysis/results/targeted_retrain.csv`.
Same architecture and training as D; each model trained 3 times; means shown.

| Model | Old acted test | New acted test | VR test | EmojiHero test | False Neutral | Headset tick | Headset better / worse than base |
|---|---|---|---|---|---|---|---|
| Base (D twin) | .503 | .412 | .457 | .562 | 34 | .577 | — |
| Gaze-off | .502 | .441 | .472 | .559 | 42 | .582 | 7 / 3 (p .49) |
| Fear weighting | .499 | .404 | .452 | .560 | 32 | .564 | 2 / 8 (p .16) |
| Left/right mirror | .466 | .420 | .446 | .577 | 32 | .570 | 4 / 6 (p .28) |

Repeat runs of the same model differ by 0.02–0.05 (VR test) and 0.03–0.04 (headset). No change is
larger than that.

- **Gaze** (`gaze_reliance.py`): shuffling gaze on D lowers the score in 9 of 10 headset sessions.
  Looking down is higher for Disgust, Sadness and Anger in all sources. Gaze is a real cue. Keep it.
- **Fear weighting** made Fear slightly worse (VR Fear F1 .37 → .33).
- **Mirror** helps EmojiHero a little, lowers old acted, and leaves the headset unchanged.

---

## 4. Pair analyses

Scripts: `pair_separation.py`, `fear_surprise_rule_and_anger_timing.py`.

### 4.1 Anger vs Disgust (test)

| | Anger ok | Anger→Disgust | Disgust ok |
|---|---|---|---|
| Nose only (no lip press) | 10% | 87% | 82% |
| Lips pressed / tightened, no nose | 31% | 3% | 0% |
| Nose share of face movement | 0.05 | 0.28 | 0.22 |
| Face size | 5.2 | 6.8 | 6.1 |

- Only 1 of 67 features separates Anger and Disgust in all three sources (nose share).
- A rule trained on two sources barely works on the third (AUC 0.41–0.65). It rescues 10% of errors.
- The misread Anger windows look more like Disgust than correct Disgust does. The model is not
  missing a pattern.

### 4.2 Fear vs Surprise

- Jaw drop (lower in Fear) and eye share (higher in Fear) hold in direction in all three sources,
  but the amount differs.
- A tie-breaker rule (re-decide only when D picks Fear or Surprise), fitted on VR training data:

| Data | D alone | D + rule |
|---|---|---|
| Headset replay, mean of 10 sessions | 0.584 | 0.627 (8 better, 0 worse) |
| Test, new acted | 0.414 | 0.486 |
| Test, old acted | 0.494 | 0.460 |
| Test, EmojiHero | 0.566 | 0.554 |
| Validation, new acted | 0.604 | 0.547 |

- The headset gain is mostly Surprise (98 → 147 of 180 ticks) and mostly the developer's face.
  The other person's session is unchanged.
- Fitted on EmojiHero or on all sources, the rule makes the headset worse.
- Not safe to deploy without testing on other people.

### 4.3 Anger timing

- **Headset:** Anger is never read as Disgust. It is read as Neutral (about 45% of ticks) at every
  tick, with no timing effect. Face size about 1.2, nose share 0.02.
- **EmojiHero test:** Anger→Disgust is about 2.5× more common while the face is still rising (39%
  vs 13–15%).
- **Own acted test:** errors peak in the middle of the expression, not at the onset.

---

## 5. Weak Anger vs Neutral (headset)

Script: `weak_anger_vs_neutral.py`.

| Group (developer, 9 sessions) | Ticks | Face size | Brow lowerer | Lid tightener |
|---|---|---|---|---|
| Anger ok | 108 | 1.58 | 0.32 | 0.20 |
| Anger→Neutral | 117 | 0.74 | 0.07 | 0.10 |
| Neutral ok | 234 | 0.56 | 0.03 | 0.015 |
| Neutral→Anger | 0 | — | — | — |

- The other person (26908f) has no Anger→Neutral errors. Their Anger is large (size 3.5).
- Lip press and lip tightening never move on the headset, for anyone.
- At matched face size, **lid tightener** is higher in weak Anger in 98% of pairs (8 of 8 sessions).
  Brow lowerer is inconsistent.
- A rule fitted with whole sessions held out:

| Rule | Anger recognised (developer) | Neutral→Anger | Other emotions→Anger |
|---|---|---|---|
| D alone | 108 / 243 | 0 | — |
| D's own Anger-vs-Neutral score | 125 | 19 | 3 |
| AU cues (mainly lid tightener) | 201 | 14 | 44 |

- On other people it only adds errors: 26908f +3 Neutral→Anger, +7 other→Anger; old acted test
  Neutral→Anger 0 → 12 of 36.
- The cue belongs to the developer's face. Disgust and Happiness also tighten the lids.

---

## 6. Label-distribution learning check (no training)

Script: `ldl_target_diagnostic.py`. Reference: Le et al., WACV 2023 (arXiv 2209.10448).

- The paper builds targets from neighbours in a predicted valence–arousal space, using the model's
  own predictions. We have no valence–arousal and no recording IDs.
- Same-source AU neighbours agree with the label 0.71–0.84; other-source neighbours only 0.33–0.44.
  Same-source neighbours mostly echo the same people.
- Other-source neighbour distributions correlate r = 0.65 with D's validation confusions
  (Fear↔Surprise, →Neutral, Anger↔Disgust/Sadness).
- Soft targets from AU neighbours would mainly repeat the known label conflicts. Not enough
  independent evidence for a meaningful experiment.

---

## 7. Overall conclusion

| Problem | Where | Main cause | Lever |
|---|---|---|---|
| Fear→Surprise | EmojiHero, headset | Fear acted as a surprise face | Fear decision (instruction, data, or merge) |
| Anger→Disgust | Acted data | Nose wrinkle used in Anger | Acting instruction / data |
| Anger→Neutral | Headset (developer) | Weak faces | Weak real examples from other people |
| Repeated errors per person | Everywhere | How each person makes a face | More people in training and evaluation |

Model and augmentation changes are exhausted on the current training set (volume, jitter,
person-style copies, EmojiHero filtering, gaze-off, Fear weighting, mirroring). Every result is
within the run-to-run spread.

---

## 8. Next step without recording new people

An inventory of the local data shows **recordings of people we already have that the teacher never
used**:

| Data | People | Size | Used by D? |
|---|---|---|---|
| Old study, natural (`weights_*.csv`, `Analysis/data/filtered-dataset-…`) | 34 folders, 141 files | 53,238 rows (about 8 Hz) | No |
| New study, natural (`VideoRecordings/`, multimodal-vr-emotion-study) | 15 | 105 files, about 50,000 frames at 10 Hz | No |
| Acted frames removed by the 0.25 filter (old + new acted) | 49 | about half of new acted Anger and Fear | No |
| EmojiHero | unknown IDs | 5,784 train windows | Yes |

Both studies also keep **participant IDs in their folder names**, which the npz lost.

Options, in suggested order:

1. **Rebuild the VR data with participant IDs and evaluate with whole people held out**
   (cross-validation over the 49 VR people). No training change. This gives a "new people" estimate
   with far more people than the 3 test participants. It also allows the developer-specific rules
   (jaw drop, lid tightener) to be tested on other people's faces.
2. **Add the weak acted frames that the filter removes**, from the same people, with the original
   label. These are real weak expressions, which is exactly the headset failure. The EmojiHero filter
   experiment already showed that weak real windows help.
3. **Add the natural recordings** (elicited by video). Expressions are weaker and closer to the
   headset. The labels are noisier (elicited, not acted). The new study has a self-report audit that
   can remove mismatched recordings. Why natural data was left out of the teacher is not documented
   and should be checked first.
4. **Self-report as independent evidence** (new study forms) for label weighting or label
   distributions on natural data. This is the evidence the label-distribution idea was missing.

Each option is judged with whole people held out, on old acted, new acted, EmojiHero and headset
replay separately, and by both outcomes for Anger: more Anger recognised, and no more Neutral or
other emotions called Anger.
