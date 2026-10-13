# Anger — Post-Hoc Logit Adjustment Study

**Project:** TrueEmotionGameVR / `com.vcu-swim-lab.vr-emotion-detection`
**Model under study:** `Runtime/Models/sparseVR_teacher_model.onnx`
**Scope:** whether the live Anger deficit can be corrected by a per-class decision adjustment,
without retraining
**Status:** **rejected on validation.** Deployed weight remains 1.0. No runtime code was changed.
The test set was not opened. A follow-up preprocessing hypothesis was raised and then falsified
(section 8). The conclusion is that there was never good evidence for a decision-bias problem in the
first place: Anger is weak both offline and live, and the live result that motivated the study rests
on two trials from one subject.

Companion to `VR_EMOTION_DETECTION_VALIDATION_REPORT.md`. Section 9 of that report listed a
protocol re-test and an EmojiHeroVR label check as the two priorities; both are addressed here
(sections 3 and 8).

---

## 1. Executive summary

Two logged headset sessions showed Anger winning only 2 and 3 of the 10 logged ticks of its trial.
The Anger output channel scored distinctively during those trials (ROC AUC 0.940 and 0.974) while
losing the argmax to Neutral. Scaling only the Anger score by ≈2.7 before the argmax took those
counts to 7 and 8 and raised overall accuracy from 0.633 to 0.750, with no other class harmed, in
both directions between sessions.

**That evidence is much thinner than it first appears, and section 4 sets out why.** Each session
holds one trial per emotion and ten consecutive ticks per trial, so the live sample for Anger is two
trials from a single subject, and the ten ticks behind each "x/10" are correlated windows over one
continuous expression rather than independent observations.

That result motivated a hypothesis — the model under-predicts Anger because Anger is the rarest
class in a third of its training data, so its prior is depressed — and a remedy: **post-hoc logit
adjustment**, a per-class additive bias applied after training.

The hypothesis makes a testable prediction: if the cause is a training-set prior, the same deficit
and a similar coefficient must appear in the pipeline's own held-out validation data.

**It does not.** On subject-exclusive own-acted validation the best coefficient is 1.23–1.55, not
≈2.7, and no coefficient in the scanned range satisfied the pre-registered acceptance criteria. The
adjustment was rejected and the weight left at 1.0.

Three further findings came out of the same work, and all of them matter more than the rejected
adjustment:

- The precision surplus that motivated the prior hypothesis (Anger precision 0.56 vs recall 0.38)
  comes almost entirely from EmojiHeroVR, which is 79.7% of the acted-only test set. On our own
  acted data Anger precision is 0.21 and recall is 0.17 — both poor, with nothing to trade.
- Own-acted validation macro-F1 is 0.607 on the older recordings and 0.332 on the newer ones. The
  model performs roughly half as well on the newer participant recordings. This is a larger and
  less explained gap than the Anger deficit itself, and section 8 rules out preprocessing as its
  cause.
- A follow-up hypothesis — that the runtime's un-averaged sampling mismatches time-averaged training
  input — was raised and then falsified by measurement (section 8). The old recordings are natively
  10 Hz and un-averaged, and they are the slice on which the model performs best.

**The premise itself does not survive.** The study assumed a live-versus-offline anomaly, but
own-acted Anger recall (0.174) and live Anger recall (≈0.25) are comparable: Anger is weak in both
places. A decision adjustment corrects a class that is well separated and badly ranked; on our own
data Anger is misranked in both directions, which indicates a weak representation rather than a
miscalibrated prior. This agrees with section 8 of the validation report, where Anger moves only the
brow and does so less than Disgust.

---

## 2. The technique, stated precisely

The adjustment multiplies one class score by a constant before the argmax:

```python
decision_scores = raw_scores.copy()
decision_scores[ANGER] *= weight
prediction = decision_scores.argmax()
```

Because the model ends in a softmax, `p_i ∝ exp(z_i)`, and therefore

```
w · p_i  ∝  w · exp(z_i)  =  exp(z_i + log w)
```

Multiplying a probability is identical to adding a constant to the corresponding logit. `×2.7` is
`+log 2.7 = +0.993` on the Anger logit. The technique is **post-hoc logit adjustment** (Menon et
al., ICLR 2021); in older literature, prior correction or threshold moving.

**This is not the calibration ruled out in section 8.5 of the validation report.** Temperature and
Platt scaling apply a monotone transform that preserves the ranking of the outputs and therefore
cannot change any argmax decision; they correct reported confidence only. A per-class additive bias
changes the ranking, which is its purpose. The two should not be referred to by the same word. The
validation report's conclusion is correct for the first and does not apply to the second.

---

## 3. Provenance of the deployed model — resolved

`Runtime/Models/sparseVR_teacher_model.onnx` was verified against the Keras checkpoint it was
exported from, on the 2,847-window acted-only test set:

```
Keras accuracy       : 0.5838
ONNX  accuracy       : 0.5838
Prediction agreement : 1.0000
Max probability diff : 0.000001
```

Source checkpoint:

```
teacher-70AU-MoE_3experts_2gate-with_15_participants-NOT-normalized_every_teacher_dataset
-self_filtered_ONLY_NATURAL-EmojiHero_fixed.keras
```

Graph interface: input `seq` `[N, 5, 70]`, output `y` `[N, 7]`, ending in Softmax. Output sums
verified to 1.0 ± 2e-7, confirming the `appendSoftmax: false` choice in
[NewFaceAuModel.cs:77](../Runtime/Scripts/NewFaceAuModel.cs#L77).

Architecture: Mixture-of-Experts temporal CNN — three dilated causal TCN experts (natural, acted,
shared), dilation rates 1/2/4, 128 filters, top-2 gate with load-balancing and entropy
regularisation, 256-unit dense head, 7-way softmax.

No ONNX export code exists in either notebook; the conversion step remains undocumented. The
equivalence check above substitutes for that provenance, but the export script should still be
committed.

---

## 4. Live evidence that motivated the study

Sessions `20260912_213907_eec228` and `20260913_005336_07c050`.

**Sample structure, verified against the logs — this bounds everything in this section.** Each
session contains **6 trials, one per prompted emotion, and 10 consecutive logged ticks per trial**:
60 predictions per session, 120 in total. The ten ticks within a trial are successive 500 ms windows
over one continuous expression and are therefore strongly correlated; they are not ten independent
observations.

The effective live sample for Anger is **two trials — one per session, from a single subject.**
Every per-class figure below that reads "x/10" is a count of consecutive ticks inside one trial, not
a count of trials.

Per-channel separability, computed on the logged score columns:

| | Anger channel | Fear channel |
|---|---|---|
| Session 1 AUC | **0.940** | 0.724 |
| Session 2 AUC | **0.974** | **0.440** (below chance) |
| Mean score on own trials | 0.222 / 0.237 | 0.140 / 0.099 |
| Mean score on other trials | 0.034 / 0.024 | 0.096 / **0.139** |

Read with the sample structure above, these AUC figures describe **one trial against five others**
within a session. A value of 0.94 says that this subject's single anger expression scored
distinctively on the Anger channel during that trial. It is not an estimate of how separable the
Anger class is in general, and it should not be quoted as one. The Fear reading is the more robust
of the two, because a channel scoring *higher on other trials than on its own* (session 2) is a
qualitative failure that a small sample can still establish.

**Fear and Anger are different problems and the report should stop treating them as one.**

Cross-session evaluation of an Anger-only coefficient (fit on one session, evaluated on the other):

```
fit on session 1 (w = 2.59) → test on session 2:  0.650 → 0.733,  Anger 3/10 → 8/10 ticks
fit on session 2 (w = 2.86) → test on session 1:  0.617 → 0.700,  Anger 2/10 → 7/10 ticks
```

No other class lost accuracy in either direction. An earlier draft described this as cross-session
replication; that overstates it. Fitting on one anger trial and testing on one anger trial from the
same subject is a consistency check between two expressions, not a replication across samples.

**Scaling only Anger is essential.** Fitting the full seven-element weight vector does not
generalise: session 1's fitted `Fear = 3.67` drops session 2 to 0.550 and destroys Surprise
(9/10 → 0/10). The validation report's concern about calibration is justified in general and is
specifically justified for Fear.

A fixed rule of `Anger ×2.7` with Neutral suppressed during a prompted trial gives 0.750 on both
sessions (pooled 0.633 → 0.750 over 120 ticks). Neutral is never a prompted class in the game, yet
it takes 8/10 of session 1's Anger ticks and 7/10 of session 2's Fear ticks.

These numbers are from one subject and were used to generate a hypothesis, never to fit a deployed
value. Given the sample structure, they are weaker than the offline validation evidence in section
6, which rests on 46 Anger windows from four participants. **Where the two disagree, the offline
result governs.**

---

## 5. Offline study design

Pre-registered before the scan, to separate coefficient selection from final evaluation:

| Stage | Data |
|---|---|
| Select coefficient | own-acted validation (old + new) |
| Robustness check | EmojiHeroVR validation |
| Lock | one number, written down, not revisited |
| Final evaluation (once) | own-acted test (primary), EmojiHeroVR test, combined acted test |
| Device check | 120 live ticks — never for fitting |

Fitting on own-acted rather than the full validation mixture was deliberate: the acted-only test set
is 79.7% EmojiHeroVR (2,268 of 2,847 windows), so a coefficient selected on the mixture would be
tuned to EmojiHeroVR rather than to the deployment domain.

Primary metric: macro-F1. Scan `w ∈ [1.0, 4.0]`, step 0.01, Anger only. Reported alongside: Anger
precision/recall/F1, balanced accuracy, per-class recall, confusion matrices.

**Split integrity.** Own recordings are split subject-exclusively by
`StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` grouped on participant, folds
0–6 train, fold 7 validation, folds 8–9 test; disjointness is asserted in the notebook.

```
Validation participants (4): P08, P20, P41, P46
Test participants (7):       P10, P11, P14, P21,
                             P40, P48, P49
```

Validation window counts:

| Split | Total | Anger | Disgust | Fear | Happiness | Neutral | Sadness | Surprise |
|---|---|---|---|---|---|---|---|---|
| Old acted val | 167 | 25 | 26 | 12 | 28 | 25 | 26 | 25 |
| New acted val | 146 | 21 | 22 | 21 | 22 | 19 | 21 | 20 |
| Combined val | 313 | 46 | 48 | 33 | 50 | 44 | 47 | 45 |
| Combined test | 579 | 68 | 108 | 61 | 102 | 89 | 83 | 68 |

**Leakage assessment.** Own-data windows are non-overlapping (`sep_process_one` advances by
`SEQ_LENGTH` on acceptance). EmojiHeroVR windows use stride 1 within 30-frame sequences that never
cross a split boundary, so there is no cross-split leakage, but 2,310 validation windows represent
only 385 independent sequences. Whether EmojiHeroVR's own published split is participant-exclusive
is **not verified**.

**The binding limitation is sample size.** 46 Anger windows from 4 participants. A recall change of
0.17 → 0.28 is a change in 5 windows.

---

## 6. Results

### 6.1 Best coefficient per validation slice

| Slice | Best w | macro-F1 (1.0 → w) | Anger recall (1.0 → w) |
|---|---|---|---|
| Old own-acted | 1.23 | 0.607 → 0.620 | 3/25 → 5/25 |
| New own-acted | 1.55 | 0.319 → 0.332 | 5/21 → 8/21 |
| Combined own-acted | 1.55 | 0.502 → 0.518 | 8/46 → 13/46 |
| EmojiHeroVR (control, w = 1.55) | — | 0.590 → **0.580** | 0.339 → 0.403 |

The offline optimum is 1.23–1.55. The live-derived 2.59–2.86 does not appear anywhere in the
offline data.

### 6.2 Acceptance criteria and decision

A coefficient was required to satisfy all four:

1. own-acted macro-F1 must not decrease
2. own-acted Anger recall must increase
3. own-acted Anger F1 must increase
4. EmojiHeroVR macro-F1 must not decrease

Baselines at `w = 1.0`:

```
own-acted   macro-F1 0.5021   Anger P 0.2105  R 0.1739  F1 0.1905
EmojiHeroVR macro-F1 0.5898   Anger P 0.5545  R 0.3394  F1 0.4211
```

**Number of weights in [1.0, 4.0] satisfying all four criteria: 0.** `w = 1.55` improves own-acted
Anger but costs EmojiHeroVR macro-F1 (0.590 → 0.580). The largest EmojiHeroVR-safe candidate,
`w = 1.14`, leaves Anger recall unchanged at 0.174 and lowers Anger F1 to 0.172, below baseline.

Recorded decision:

```json
{
  "status": "rejected",
  "deployed_weight": 1.0,
  "tested_candidate_own_acted": 1.55,
  "tested_candidate_with_emojihero_guardrail": 1.14,
  "reason": "No candidate increased own-acted Anger recall and F1 while preserving macro-F1 on own-acted and EmojiHero validation.",
  "test_set_opened": false,
  "next_step": "Investigate offline-to-live domain and preprocessing shift.",
  "model": "sparseVR_teacher_model.onnx"
}
```

The test set was **not** opened and remains clean for a future attempt.

### 6.3 Statement of record

For citation in downstream write-ups:

> An Anger-specific decision weight applied to the own-acted validation set was able to increase
> Anger recall, but the same weight reduced performance on EmojiHeroVR validation. No single global
> weight satisfied the pre-registered acceptance criteria on both domains. Post-hoc Anger weighting
> was therefore rejected and the test set was not opened for evaluation.

This is a rejection of one global weight on the evidence available, not a finding that the Anger
channel lacks signal. What it establishes is narrower and more useful: the Anger deficit does not
arise from a single decision bias shared across all data sources.

### 6.4 Reference model performance (acted-only test, 2,847 windows)

Not part of the adjustment decision; recorded for context.

| Model | Accuracy | macro-F1 | Anger F1 | Fear F1 |
|---|---|---|---|---|
| `sparseVR_teacher_model.onnx` | 0.58 | 0.57 | 0.45 | 0.35 |
| `act_trial_92.onnx` | 0.55 | 0.52 | 0.41 | 0.20 |
| `natural_trial_96.onnx` | 0.29 | 0.25 | 0.14 | 0.02 |

Teacher confusion, dominant off-diagonals: Anger → Disgust 128 of 392; Fear → Surprise 111 and
Fear → Neutral 73 of 385. The Anger→Disgust confusion matches both the action-unit analysis in
section 8 of the validation report and the published EmojiHeroVR benchmark, where Anger is also the
worst class (F1 49.11%) with the same dominant error.

---

## 7. Interpretation

**The prior hypothesis is not supported.** It predicted a comparable deficit and coefficient in
held-out validation data. Offline the coefficient is 1.23–1.55 and fails the guardrails.

**The precision surplus was an artefact of test-set composition.** The argument for a prior
correction rested on Anger precision 0.56 against recall 0.38 — a classic under-prediction
signature. That measurement comes from a test set that is 79.7% EmojiHeroVR. On our own acted data
Anger precision is 0.21 and recall 0.17: the class is wrong in both directions, and no threshold
shift can repair that. This correction should be carried into any text that cites the 0.56/0.38
figures.

**There is no live-versus-offline anomaly to explain.** This was the study's working premise and it
does not survive the numbers. Own-acted validation Anger recall is 0.174; live Anger recall is
roughly 0.25. Anger is weak in both places to a similar degree. What differed was only that a
coefficient appeared to help live — on a sample of two trials from one subject, where ten correlated
ticks per trial made the effect look larger and more replicated than it is.

**The deficit is therefore not a decision bias.** A threshold problem is a class that the model
separates well and ranks badly. On own-acted data Anger has precision 0.21 against recall 0.17: it
is misranked in both directions, which is what a weak feature representation looks like, not a
miscalibrated prior. This converges with section 8 of the validation report, where Anger moves only
the brow and does so less than Disgust, and with the test-set confusion, where 128 of 392 Anger
windows go to Disgust.

**The largest unexplained finding is not about Anger.** Old own-acted validation macro-F1 0.607
versus new own-acted 0.332, with Disgust recall falling 0.731 → 0.227. The model is markedly worse
on the newer participant recordings. This deserves priority over the Anger question.

---

## 8. Preprocessing mismatch — hypothesis raised and falsified

**Outcome: rejected.** The measurements in 8.2 show the old recordings are natively 10 Hz and are
not time-averaged, and that the model's strongest slice is therefore un-averaged data of the same
character as the runtime's. The paired experiment in 8.3 was designed but **should not be run** — it
would test a hypothesis the cheaper check has already eliminated. The section is kept in full
because the preprocessing inventory it establishes is worth having on record.

A frame reaching the model is constructed differently in each data source. Of the 16,742 training
windows:

| Source | Windows | How one 10 Hz frame is produced |
|---|---|---|
| EmojiHeroVR | 5,784 | mean of 3 raw frames (`average_every_3_rows`, 30 → 10) |
| New participant recordings | 4,019 | mean of 6 raw frames (`downsample_60hz_to_10hz_face_au`, 60 → 10 Hz) |
| Old `filtered-dataset` natural | 5,386 | **unknown** — read as-is, no resampling in the notebook |
| Old `filtered-dataset` acted | 1,553 | **unknown** — read as-is, no resampling in the notebook |
| **Live runtime** | — | **single instantaneous read per poll, no averaging** |

58.6% of training windows are demonstrably time-averaged. The sampling of the remaining 41.4% is
not documented in either repository; the capture application that produced those CSVs is not in
`VREmotionDetection` or `TrueEmotionGameVR`.

Runtime behaviour, verified in [DeviceManager.cs:269-304](../Runtime/Scripts/DeviceManager.cs#L269-L304)
and [DeviceManager.cs:160-185](../Runtime/Scripts/DeviceManager.cs#L160-L185): `Update` checks once
per rendered frame whether `Time.time` has reached the next poll time; on a poll it performs a
single `faceExpressions.CopyTo(data)` and schedules the next for `now + 0.1`. Values between polls
are never read. The interval is therefore *at least* 100 ms, quantised to the frame boundary
(≈104–111 ms at 72 Hz), not a clean 10 Hz, and it is not a decimation of a buffered 60 Hz stream —
no such buffer exists.

Averaging suppresses transient noise and slightly reduces peak amplitude. Classes carried by one
large action unit (Happiness, Surprise) should tolerate the difference; classes carried by the
sustained co-activation of several moderate units (Anger, Fear) should not. Those are exactly the
two failing classes. That is the mechanism the hypothesis proposes.

Input scaling is *not* a mismatch: `mean_centered_max_scale` is commented out throughout the
notebook and the runtime feeds raw action-unit values. This was checked.

### 8.1 Evidence against the hypothesis

The old-versus-new gap in section 7 points the wrong way for it:

| Validation slice | Preprocessing | macro-F1 |
|---|---|---|
| Old own-acted | unknown | **0.607** |
| New own-acted | time-averaged | **0.332** |

If the model depended on seeing time-averaged input, the averaged slice should be the stronger one.
It is the weaker one by a wide margin. The old/new gap is therefore unlikely to be a preprocessing
effect and more likely reflects participants, capture application, or device firmware. It should not
be cited as support for the preprocessing hypothesis, and an earlier draft of this document that did
so was wrong.

### 8.2 The falsification check, and its result

Two measurements on the existing data, no new collection.

**Native rate.** The time column of the old CSVs (`itimes`, dropped by `data_frame`) steps by a
constant `0.099975 s` across files and participants — natively 10 Hz, and near-perfectly regular.
The notebook is right not to resample them, and there is no hidden time-scale error.

**Smoothing.** Temporal roughness, `std(frame-to-frame difference) / std(signal)` per channel,
averaged over channels and windows. Time-averaged input gives a low ratio; independently sampled
input gives a high one, tending to √2 ≈ 1.41 for white noise.

| Source | Preprocessing | mean r |
|---|---|---|
| EmojiHeroVR | mean of 3 | 0.913 |
| New acted | mean of 6 | 0.966 |
| Old acted | *was unknown* | **1.115** |
| Old natural | *was unknown* | **1.126** |
| **Live headset** | instantaneous | **1.18 – 1.25** |

The old recordings sit with the runtime, not with the averaged corpora. Combined with the exact
10 Hz spacing, they were captured by instantaneous sampling, like the runtime.

**This falsifies the hypothesis.** Old acted is the slice on which the model performs *best*
(macro-F1 0.607) and it is un-averaged. The model does not depend on seeing time-averaged input, and
the runtime's lack of averaging is therefore not the explanation for the live Anger deficit. A
residual gap remains (1.11 live-like recordings versus 1.18–1.25 live), plausibly jitter — runtime
polls are frame-quantised at ≥100 ms while the old recorder was regular to five decimal places — but
it is small next to the distance to the averaged sets and does not restore the hypothesis.

### 8.3 The paired experiment — designed, not to be run

Retained as a record of what was planned before 8.2 settled the question. **Do not execute it**; it
would spend effort testing an eliminated hypothesis.

Existing 60 Hz recordings, no new data collection and no retraining. Build two versions of the same
test windows and run both through the same ONNX graph:

- **`averaged`** — mean of every 6 frames. What the new-recording training path produces.
- **`decimated`** — every 6th frame, no averaging. A *proxy* for the runtime.

Name the two arms in code and in any results table; do not label them A and B, which invites a
silent swap between whoever runs the experiment and whoever reads it.

`decimated` approximates rather than reproduces runtime behaviour. The runtime reads whatever the
tracker last produced at a frame-quantised interval of ≥100 ms, so real polls are not phase-locked
to a 60 Hz grid, may drift, and may re-read an unchanged value. Regular decimation is the closest
construction available from recorded data and isolates the averaging question; it does not test
jitter. If the paired comparison comes out null, jitter remains untested rather than excluded.

Participant and protocol are held constant; only preprocessing varies, which the old-versus-new
comparison in section 7 cannot do.

- If Anger degrades under **`decimated`**, the hypothesis is supported and the remedy is to sample
  at 60 Hz and average 6 frames in `AUDevice` — a small runtime change, no retraining, no
  coefficient.
- If the two arms agree, preprocessing is exonerated and the remaining candidates are the subject
  (one live participant versus four in validation) and the protocol (in-game prompting versus a
  recording session).

---

## 9. What was changed

No code, model, or configuration file. This document is the only addition; every runtime script,
model asset, and test is byte-identical to the state at the start of the investigation, and HEAD is
still `923ce26`. The session logs under `TrueEmotionGameVR/Logs/quest-sessions/sessions` were read
only, and all analysis ran in a scratch directory outside the project.

The deployed decision rule is still a plain argmax over the raw model outputs
([NewFaceAuModel.cs:156-163](../Runtime/Scripts/NewFaceAuModel.cs#L156-L163)), equivalent to
`w = 1.0` for every class.

A behaviour-neutral runtime scaffold was proposed but **not** implemented, pending the outcome of
section 8.1: an Inspector-exposed per-class decision weight defaulting to 1.0, raw scores preserved
and logged unmodified, adjustment applied to a separate decision vector, and `anger_weight` /
`weight_source` / `decision_adjustment_version` recorded in session metadata.

---

## 10. Corrections to the validation report

1. **Section 8.5 and section 9 item 7** rule out "probability calibration" for both Fear and Anger.
   That conclusion is correct for temperature-style calibration, which cannot change an argmax, and
   for Fear in any form — the Fear channel is at or below chance separability. It should not be
   worded so as to exclude per-class logit adjustment, which is a different intervention. That
   intervention was tested here on its merits and rejected on evidence, not ruled out a priori.

2. **Section 9 priority 2, the EmojiHeroVR label mapping, is closed as not-a-bug.** The dataset
   documents `0=Anger, 1=Disgust, 2=Fear, 3=Happiness, 4=Neutral, 5=Sadness, 6=Surprise`, identical
   to the alphabetical `LabelEncoder` order, and the notebook asserts this order explicitly. Two
   narrower checks remain open: EmojiHeroVR records 63 channels against the runtime's 70, so the
   zero-padding of slots 63–69 in the training pipeline should be confirmed (live tongue channels
   read ~0.0001, so it is benign at runtime); and `emotion-id` is preceded by `emoji-id` in the
   filename, where a field-index shift would silently read the wrong label.

3. **Section 9 priority 1, the protocol re-test, appears to have been run and to have failed.**
   Session 2 shows Fear expression intensity falling 0.204 → 0.154, `LipStretcher` (AU20) at exactly
   0.000 in both sessions, and `LipTightener`/`LipPressor` (AU23/AU24) between 0.000 and 0.017
   across every emotion in both sessions. Anger was partly driven — `LidTightener` rose 0.133 →
   0.345 — but the lower-face units never appeared. Voluntary production of these units should be
   treated as unavailable for this subject.

---

## 11. Open items, in priority order

1. Explain the old-versus-new own-acted gap (macro-F1 0.607 vs 0.332, Disgust recall 0.731 → 0.227).
   This is now the largest unexplained effect in the pipeline, it is bigger than the Anger deficit,
   and sections 8.1–8.2 rule out preprocessing as its cause — leaving participants, capture
   application, and device firmware.
2. Collect more live trials before drawing any further conclusion from headset data. One trial per
   emotion per session cannot support a per-class claim; the logging format already records
   everything needed, so this costs session time only.
3. Commit the ONNX export script; the conversion step is currently undocumented.
4. Add a runtime activity gate. Training required 3 consecutive frames with any unit above 0.25;
   the runtime applies no gate. On gated windows live accuracy is ~0.87 against ~0.08 on rejected
   ones — this suppresses bad output rather than improving accuracy.
5. Extend the window beyond 5 frames (500 ms) and add first-difference channels. Published onset
   durations are 480–650 ms for fear and 670–890 ms for anger; an anger onset does not fit the
   current window. Comparable dynamic models use 16 frames and EmojiHeroVR ships 30-frame sequences.
   Fear/Surprise separation is reported to be kinematic — mouth velocity and acceleration — rather
   than configural, with no significant eyebrow differences, which is consistent with AU20 never
   appearing here.
6. Record per-trial self-report in the game. The training pipeline already excludes recordings whose
   intended emotion never appears in the participant's free-text report (Part 2 of the teacher
   notebook); live sessions have no equivalent, so a failed elicitation is currently indistinguishable
   from a model error.
7. Only then reconsider a decision adjustment, on a larger validation sample and with the test set
   still unopened.

---

## 12. Artefacts

Written by the teacher notebook to
`…/predictions/anger_decision_adjustment/` on Drive:

- `validation_guardrail_scan.csv` — the full weight scan with both domains' metrics
- `validation_baseline_vs_candidates.csv` — baseline versus candidate per validation slice
- `validation_decision.json` — the recorded rejection above

Analysis of the live sessions was performed on the logged `score_<Class>` columns in
`predictions.csv`; the ONNX graph was not re-run for that part.
