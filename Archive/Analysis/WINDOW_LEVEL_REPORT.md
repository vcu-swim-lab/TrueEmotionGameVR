# The 5×70 Windows the Model Actually Saw

**Scope:** every live window the two players produced, compared against the acted training
windows built the way the headset builds one, in the model's input space and in its output
space, kept apart.

**Reproduce:**
`python Analysis/window_level_study.py --json results/window_level.json`
`python Analysis/window_level_study.py --condition runtime_proxy --channels expression55`
`python Analysis/test_window_level_study.py`   (44 tests)

**Status:** diagnostic, exploratory. Nothing is fitted, no calibration is proposed, no model
is re-trained. Two live people. Every ONNX number is behind a passing replay-fidelity check
(agreement 1.000, max diff ≤ 4.2e-07 on all three sessions).

**Primary condition** is `runtime_proxy` — instantaneous 10 Hz samples, no averaging, no gate
— because it is the only acted condition whose windows are built the way the device builds
one. `notebook_exact` is reported as a training-distribution control and is **not** a runtime
measurement. **Primary channel set** is all 70, because that is what the model is given;
`expression55` is a sensitivity check only.

---

## 0. The control that reframes everything

Before asking what A and B do, ask what raw 5×70 cosine can tell apart at all. Each acted
training participant's windows for one emotion, against the *other nine* participants'
windows for every emotion:

| acted | Anger | Disgust | Fear | Happiness | Sadness | Surprise | self-nearest? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Anger | **0.459** | 0.450 | 0.442 | 0.420 | *0.464* | 0.442 | **no** — Sadness |
| Disgust | 0.464 | **0.504** | 0.466 | *0.516* | 0.493 | 0.458 | **no** — Happiness |
| Fear | 0.443 | 0.441 | **0.455** | 0.405 | 0.440 | *0.466* | **no** — Surprise |
| Happiness | 0.446 | 0.519 | 0.440 | ***0.677*** | 0.461 | 0.449 | yes, by +0.158 |
| Sadness | 0.471 | 0.492 | 0.453 | 0.427 | ***0.493*** | 0.433 | yes, by +0.001 |
| Surprise | 0.410 | 0.408 | 0.442 | 0.411 | 0.411 | ***0.466*** | yes, by +0.024 |

**Only Happiness is genuinely separable in this space.** Sadness clears its runner-up by
0.001 and Surprise by 0.024, which is nothing; Anger, Disgust and Fear are each nearer some
other class than themselves. The same holds under `notebook_exact` (Happiness +0.280, Fear
nearest Sadness, Disgust nearest Happiness).

And the specific confusion: **acted Fear is already nearest acted Surprise.** Not in the live
data — in the training corpus, between participants the model was fitted on.

Everything below is read against this table. A live window landing near acted Surprise when
Fear was asked for is, in the first instance, a fact about the representation.

---

## 1. Answers

### 1. Why do A and B get different results on Happiness?

**Verdict: mixed, leaning to the decision boundary.** The inputs are in the same place; the
outputs are not.

| | input cos to acted Happiness | prompted ranked 1st | nearest neighbour | top-5 neighbours | model confidence | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.818 | 18/18 windows | Happiness 18/18 | Happiness 89, Surprise 1 | 0.534 | 0.94 |
| B | 0.731 | 9/9 windows | Happiness 6/9 | **Surprise 23**, Happiness 18, other 4 | **0.257** | 0.11 |

Happiness ranks first among the six for **every one of B's nine windows**, and this is the one
emotion the control above says the space can actually resolve. So B's Happiness input is not
in the wrong region.

What differs is the answer. B's mean output vector is almost flat — confidence 0.257 against a
seven-class floor of 0.143:

```
             Anger Disgust  Fear  Happy Neutral Sadness Surprise   conf
  live A     0.035  0.167  0.111  0.530  0.021  0.074  0.061      0.534
  live B     0.113  0.134  0.206  0.167  0.070  0.117  0.193      0.257
```

Fear wins by 0.039 over Happiness. The model has no opinion and the argmax is decided by
noise.

The qualifier: B's input is measurably nearer the *edge* of the Happiness region. Its pooled
top-5 neighbours are more Surprise than Happiness (23 vs 18) where A's are 89 Happiness to 1
Surprise, and its balanced cosine is 0.087 lower. So this is not a clean case 2 — the input
is in the right region but not deep inside it, and the output collapses to near-uniform. Both
are true and the second is the larger effect.

### 2. Why is Fear Surprise for both of them?

**Verdict: supported, and it is not about the players.** The input really does sit nearer
acted Surprise, for both people, and the model's answer follows it exactly.

| | cos to acted Fear | cos to acted **Surprise** | prompted ranked 1st | nearest = Surprise | model |
| --- | --- | --- | --- | --- | --- |
| A Fear | 0.496 | **0.574** | 1/18 | 14/18 | Surprise ×9, Neutral ×9 |
| B Fear | 0.404 | **0.526** | 0/9 | 9/9 | Surprise ×9 |

The overlap is near-total. Of every window in the study where the model answered Surprise,
all but four also had a Surprise window as their nearest acted neighbour:

| | near & said | near only | said only | neither |
| --- | --- | --- | --- | --- |
| A Fear | 9 | 5 | **0** | 4 |
| B Fear | 9 | 0 | **0** | 0 |
| B Disgust | 5 | 0 | 3 | 1 |
| B Surprise | 8 | 0 | 1 | 0 |

But §0 shows the direction was already there: acted Fear is nearer acted Surprise by 0.011
between training participants. The live data **amplifies** an existing property rather than
introducing one — the live gap is 0.078 (A) and 0.122 (B), seven to eleven times the acted
baseline.

So: Fear→Surprise is a property of this feature space that both players fall into, not a
difference between them. It is the same finding as
[SUBJECT_VARIABILITY_REPORT.md §11](SUBJECT_VARIABILITY_REPORT.md) item 1 seen from the input
side.

### 3. Why is B's Disgust Surprise?

**Verdict: supported.** The windows are not near acted Disgust at all.

| target | balanced cosine |
| --- | --- |
| **Surprise** | **0.574** |
| Fear | 0.459 |
| Sadness | 0.423 |
| *Disgust (prompted)* | *0.400* |
| Anger | 0.399 |
| Happiness | 0.374 |

Disgust ranks **4th of 6**, and in the pooled neighbourhood Disgust appears once in 45 top-5
slots against 22 for Surprise and 17 for Fear. Not one of the nine windows has a Disgust
nearest neighbour. The model answered Surprise ×8, Fear ×1, with mean scores Surprise 0.543,
Fear 0.283, Disgust 0.049.

The input explains the answer here. Note again the §0 caveat: acted Disgust is not
self-nearest either (its nearest is Happiness), so this is a class the representation handles
badly to begin with.

### 4. Why does B's Anger fail?

**Verdict: mixed, and partly outside what this data can answer.**

In the AU comparison B's Anger windows are nearest **Anger** — balanced cosine 0.465, the
highest of the six, ranked first for 5 of 9 windows, with Fear second (4 of 9). So at window
level they are not outside the acted Anger region, which qualifies the peak-profile finding in
[LIVE_VS_ACTED_REPORT.md](LIVE_VS_ACTED_REPORT.md) that B's Anger sat below all ten training
participants. What is true is that the level is low: A's Anger reaches 0.586 against the same
reference, and B's single nearest acted neighbour is only 0.526 where every other cell in the
study has one at 0.759 or above.

What the model does, stated as observed and no further: mean scores **Neutral 0.360, Fear
0.342, Anger 0.146**, confidence 0.654, answers Neutral ×5 and Fear ×4. The Fear half is
consistent with the input — Fear is the second-nearest acted class for these windows. The
Neutral half cannot be checked here at all: Neutral is excluded from the AU comparison,
because the game never prompts it and the teacher pipeline *inverts* the activity rule that
defines a usable Neutral frame. No AU similarity to Neutral is computed, and none should be
inferred.

Worth noting that A's Anger fails in the same direction — Neutral ×8, Anger ×5, Fear ×5,
accuracy 0.28 — from an input that is *closer* to acted Anger than B's. Anger is weak for both.

### 5. What did the window-level analysis explain that the peak profile could not?

**That the absolute cosine level carries almost no information, and the neighbourhood does.**

The two clearest cells in the study have nearly the same balanced cosine to their prompted
class and opposite meanings:

| | cos to prompted | nearest neighbour | top-5 neighbours | accuracy |
| --- | --- | --- | --- | --- |
| A Sadness | 0.397 | Sadness 18/18 | Sadness 90/90 | **1.00** |
| B Disgust | 0.400 | Disgust 0/9 | Disgust 1/45 | **0.00** |

0.397 and 0.400. A's Sadness is unambiguously inside the Sadness region — every one of its
ninety top-5 neighbours is a Sadness window, all from one acted participant
(`P37`) — and the model returns Sadness at 0.993 confidence. B's Disgust is
unambiguously outside the Disgust region. A peak profile collapses both to one scalar and
cannot distinguish them, which is exactly why the previous report found no correlation between
distance and accuracy (Spearman +0.007). The correlation was absent because the quantity being
correlated was the wrong one.

The window level also supplies the thing the peak profile structurally could not: a **per-window
join between the input and the answer**. That is what makes the Surprise table in §2 possible
— "the model said Surprise on exactly the windows whose input was nearest a Surprise window"
is not a statement a trial-level summary can make.

Three further things it added:

- **Confidence is a separate axis from position.** B's Happiness failure is a 0.257-confidence
  output from an input in the right region. Nothing about distance predicts that.
- **The live and acted output distributions barely overlap.** Acted `runtime_proxy` windows are
  Neutral-dominated and diffuse (Neutral mean 0.20–0.48, argmax is Neutral for the plurality
  in five of six emotions, confidence 0.47–0.62). Live windows are peaked and confident (A's
  Sadness 0.993, A's Disgust 0.754, live Neutral often below 0.01). The same network, fed the
  two corpora, behaves like two different models.
- **The representation is the ceiling.** §0 says raw-window cosine resolves one class out of
  six. That bounds how much any nearest-neighbour method built on it — including anything a
  calibration might do in this space — could deliver.

### 6. What is still unexplained?

- **Why live outputs are confident and acted outputs are not.** This is the largest unexplained
  gap in the report and it undercuts any offline estimate of live behaviour. `runtime_proxy` was
  built to be the closest offline stand-in and it does not reproduce the output distribution.
- **Why Neutral wins on B's Anger and on both players' Anger.** Not answerable from an AU
  comparison that excludes Neutral by construction. It needs acted Neutral windows brought in
  under a stated rule, which is a separate decision, not a detail.
- **Why B's outputs are low-confidence where A's are peaked.** Observed across cells, not
  explained. It could be expression intensity, headset fit, or where B's windows sit relative
  to a decision boundary; nothing here separates those.
- **Whether any of this is about B rather than about a second recording session.** N = 2.
- **The direction of the Surprise effect end to end.** Live data is biased *toward* Surprise;
  acted held-out data never emits it (0/5 recall in
  [model_performance.txt](results/model_performance.txt)). Both are now measured; neither is
  explained. The golden-window check is still the next step and this report does not replace it.

---

## 2. Method

**Live windows** are the exact `[5, 70]` blocks the logger wrote alongside the prediction they
caused — not reconstructed from raw tracking, so a window and the score beside it are the same
event. Tick 0 is excluded as a transition sample, selected on `prediction_index` rather than by
counting rows. `no_tracking` rows carry no window. A's two sessions are summarised separately
and combined with **equal weight per session**.

**Acted windows** come from `teacher_preprocessing.windows_for_condition`, the same function
Phase B uses. `runtime_proxy` is every 6th frame of the ~60 Hz capture in non-overlapping fives
with no gate: 1199 windows from the 60 training recordings, 19–20 per recording.
`notebook_exact` yields 680 from the same recordings, 0–19 each.

**Reference** is the ten `train` participants from `acted_splits.csv`. Validation and test
participants are not in it.

**Flattening** is `[5, 70] → [350]`, frame-major, so position `f*70 + c` is channel `c` at
frame `f`. Cosine on this is sensitive to *when* a channel moved: a rising expression and its
time-reverse have a cosine of 0.333 here and are identical under a peak profile. Values are
raw — the model was trained on unscaled action units.

**Participant balance.** Each acted participant contributes their single best-matching window
per emotion, and the ten are then averaged. A plain maximum over the pooled set would let a
participant who happens to contribute more windows decide the answer alone — which
`notebook_exact`, at 0 to 19 windows per recording, would.

**Score space is kept separate** and is reported after the mass table, because these are
Softmax vectors: a component every class shares lifts every cosine between them, and acted
windows carry 0.20–0.48 mean Neutral on every emotion. Raw score cosines in this report run
0.40–0.95 and should be read as weak evidence. `calibration_study.features` documents the log
transform that addresses this; this file does not apply it, because that is a modelling choice
and this round is measurement.

**Channel sets** are reported separately and never substituted. All conclusions above hold
under `expression55`: Fear→Surprise (A 0.558 vs 0.494, B 0.537 vs 0.456), B Disgust→Surprise
(0.605 vs 0.419, prompted ranked 1st in 0 of 9 windows), B Happiness ranked 1st in 8 of 9
windows at 0.738, A Sadness still low-cosine-high-accuracy (0.460, 1.00). They also hold under
`notebook_exact`.

---

## 3. Limits

- **Two live people.** Two of the three sessions are one person on two evenings. Every
  comparison is one person against one person.
- **Windows are not observations.** Nine windows from one trial are nine samples of one
  continuous performance; so are the twenty from an acted recording. No p-value is computed and
  none would be meaningful.
- **Nearest-neighbour is a diagnostic, not a design.** That a live window sits near an acted
  Surprise window does not license a calibration built on that fact — and §0 is the reason to
  be especially careful here, since the space resolves one class of six.
- **Neutral is outside the AU comparison entirely.** Where it appears it is only as an observed
  model output.
- **`runtime_proxy` is still a proxy.** The device polls against live tracking and skips frames
  the tracker rejects, so a real window can span more than 500 ms. §1.6 records that this proxy
  demonstrably fails to reproduce the live output distribution.
- **`notebook_exact` numbers are a training-distribution control** and must not be quoted as
  runtime behaviour.
- **The score-space cosines are inflated** by shared Neutral mass, as above.
- **The acted and live protocols differ** in sampling, duration, task and feedback. A live
  player is watching a score bar and adapting within the trial; an acted participant is not.
