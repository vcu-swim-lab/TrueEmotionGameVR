# Per-Person Calibration — Recording Protocols

**Current protocol:** `session-local-v1`
**Software:** [`ProtocolRunner.cs`](../Assets/Scripts/ProtocolRunner.cs)
**Status:** recording software complete. **Data collection has not started.** See section 8.

This file holds the current protocol *and* the ones it replaced, with the measurements that
caused each replacement. Sessions recorded under different `protocol_version` values are never
pooled, so a superseded protocol is documented rather than deleted.

---

## 1. What is being tested, and what was ruled out

`sparseVR_teacher_model.onnx` is frozen: no retraining, no fine-tuning, no weight edits. The
question is whether a short per-person enrolment, used to reinterpret the model's own seven
output scores, improves recognition for that person.

Three headset sessions started this off. Against a chance level of 14.3%:

| Session | Subject | Tick accuracy |
|---|---|---|
| `20260912_213907_eec228` | A | 61.7% |
| `20260913_005336_07c050` | A | 65.0% |
| `20260914_175512_8ccb69` | **B** | **20.0%** |

Subject B is near chance, and not for want of effort: B's peak action-unit intensity was
comparable to A's and *higher* for Anger (0.403 against 0.255) and Fear (0.609 against 0.363),
the two classes B scored 0% on. Neither baseline subtraction nor gain rescaling recovered B.

A nearest-centroid adapter fitted on part of B's session recovered the rest of it, **0.375 →
0.792** against a matched baseline. (An earlier draft reported 0.167 → 0.792; that compared a
six-class adapter against a seven-class argmax and so credited it for excluding Neutral. The
fair figure is 0.375. This is the project's worked example of why the analysis has to be frozen
before the data is seen — see section 8.)

### 1.1 Superseded: the persistent cross-session profile

The original product claim was "calibrate once, reuse the profile on later days." **It was
tested against the recordings on disk and it failed.** Two subjects have two sessions each, so
four same-person cross-session directions are available. Adapter against the class-matched
argmax baseline, centroids from ticks 1–9, tested on ticks 1–9:

| direction | argmax | adapter | delta | trial-vote argmax → adapter |
|---|---|---|---|---|
| A1→A2 (6-class) | 0.7778 | 0.7407 | **−0.0370** | 0.6667 → 0.8333 |
| A2→A1 (6-class) | 0.8333 | 0.8148 | **−0.0185** | 0.8333 → 0.8333 |
| C1→C2 (7-class) | 0.1429 | 0.2063 | **+0.0635** | 0.1429 → 0.1429 |
| C2→C1 (7-class) | 0.6508 | 0.3492 | **−0.3016** | 0.7143 → 0.2857 |

Mean delta **−0.0734**; one of four positive, and that one's gain disappears at trial level.
In `C2→C1` the subject's *own* profile collapsed Happiness 9/9→0/9, Sadness 9/9→0/9 and
Neutral 9/9→0/9. C2 was deliberately recorded with low-intensity expressions, and that is the
mechanism: a profile enrolled at one expression intensity poisons the same person at another.

Notably the centroids themselves are stable — the same state's centroid across two sittings has
cosine 0.9837 (A) and 0.9202 (C) against a between-state separation of 0.8387 / 0.8463, so both
margins are positive. The representation is not meaningless. What fails is that for a few states
the drift *exceeds* the between-state separation, and nearest-centroid then picks the wrong one.

**Consequence:** the persistent-profile claim is withdrawn. Cross-session transfer stays in any
report as a stated scientific limit, not as a product feature.

### 1.2 The within-trial measurement is mostly leakage

The 0.375 → 0.792 result fitted on the early ticks of a held expression and tested on its late
ticks. Widening the gap between the two tick sets, within the same trial, shows what it was
measuring — mean adapter gain over the matched baseline across all five sessions:

| enrol \| test ticks | accuracy gain | macro-F1 gain |
|---|---|---|
| 1-5 \| 6-9 (the reported split) | **+0.1500** | +0.1746 |
| 1-4 \| 6-9 | +0.1107 | +0.1359 |
| 1-3 \| 7-9 | +0.0762 | +0.0926 |
| 1-2 \| 8-9 | +0.0714 | +0.0840 |
| 1 \| 9 (maximum separation) | **−0.0095** | +0.0129 |

Monotone decay to nothing, and subject B's headline decays fastest of all: **+0.4167 → 0.0000**.
A separate trial is *more* separated than tick 1 against tick 9 of one trial — different prompt,
a relax phase between, the expression rebuilt from scratch. So a trial-disjoint split is expected
to land at or below zero, and that is what `session-local-v1` measures.

### 1.3 A ceiling to state in advance

Raw argmax recall on ticks 1–9, every session:

| session | Anger | Fear | Disgust | Happiness | Sadness | Surprise | Neutral |
|---|---|---|---|---|---|---|---|
| A1 | 2/9 | 0/9 | 9/9 | 8/9 | 9/9 | 9/9 | — |
| A2 | 3/9 | 0/9 | 9/9 | 9/9 | 9/9 | 9/9 | — |
| B1 | 0/9 | 0/9 | 0/9 | 1/9 | 2/9 | 9/9 | — |
| C1 | 0/9 | 5/9 | 9/9 | 9/9 | 9/9 | 0/9 | 9/9 |
| C2 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 9/9 |

Anger **5/45**, Fear **5/45**.
[`VR_EMOTION_DETECTION_VALIDATION_REPORT.md`](VR_EMOTION_DETECTION_VALIDATION_REPORT.md) §8
concludes both fail on the model's *input*, not on the model. Corroborating: Anger's centroid is
more similar across different people (A1~B1 = 0.9779) than across one person's two sittings
(A1~A2 = 0.9378) — an inversion no other state shows. There is no person-specific Anger signal
to capture, and reinterpreting an output vector cannot create information the input does not
contain. **A realistic macro-F1 ceiling is set by roughly five usable classes of seven.** Judge
Anger and Fear separately, or the pilot is scored against a target it cannot reach.

---

## 2. Current protocol: `session-local-v1`

Calibration lives only in memory, for one headset sitting. The claim is narrowed to match:

> After putting the headset on and completing a short calibration, does recognition improve
> **within that same sitting**, on separate trials?

This is **untested**. It sits between the leaky within-trial evidence (positive) and the
cross-session evidence (negative). Section 1.2 is the reason to expect little.

### Blocks

Seven calibration states, Neutral included:

```
Anger   Disgust   Fear   Happiness   Neutral   Sadness   Surprise
```

| block | trials | purpose |
|---|---|---|
| **K** — calibration | 7 (one per state) | fits the centroids; frozen immediately |
| **D** — evaluation | 14 (**two** per state) | trial-disjoint test set; never enters a centroid |

**Total 21 trials, about 7 minutes.** Each block is a sequence of shuffled passes over all seven
states, and a pass never opens on the state the previous pass closed with — including across the
K→D boundary — so no state is ever acted twice in a row. The seed is recorded, so any session's
order reproduces exactly.

### Why D has two trials per state

A seven-trial evaluation cannot support a decision. Three measurements say the same thing:

- **Sign test.** With 7 paired trials only a perfect 7-0 sweep reaches p<0.05 (p=0.0156); 6-1 is
  p=0.125. Discordant pairs are always fewer than trials, so 7-0 is unreachable in practice.
- **Discordance.** In the real pooled trial outcomes (32 trials, maximum-separation split) the
  adapter and the baseline **tie on 23 of 32** — a 28% discordance rate. Seven trials yields
  roughly **two** informative trials.
- **Bootstrap** over the real outcome pool (20 000 resamples, pool delta +0.0938):

  | eval trials | SD of delta | 95% CI width | reports the wrong sign |
  |---|---|---|---|
  | 7 | 0.1977 | 0.714 | **0.194** |
  | 14 | 0.1398 | 0.500 | 0.176 |
  | 21 | 0.1132 | 0.476 | 0.146 |

  These come from the old tick-split proxy and are **exploratory justification for the block
  size, not an error rate to quote.**

A 14-trial session (K=7, D=7) is a fine plumbing rehearsal. It must not carry a kill decision.

### Neutral is a real trial

Neutral gets its own `trial_id`, `block_role`, `repeat_index`, its own prompt and countdown, and
its own ten ticks — structurally identical to every other state. It is **not** harvested from the
inter-trial relax phase. Building a Neutral centroid from tick-0/relax samples was measured
making things worse: subject B fell 0.792 → 0.625 and that subject's Sadness collapsed 4/4 → 0/4,
because a transition sample carries the tail of the previous expression.

It has no emoji. Every emoji depicts a *performed* expression, and showing one would prompt the
participant to act neutral rather than to rest. The wording is fixed and identical for everyone:

```
Neutral

Let your face relax naturally.
Keep your mouth and eyebrows in their natural position.
```

During the hold the screen reads `Stay relaxed`.

Including Neutral also closes the baseline trap of section 1: with all seven states prompted, the
adapter and the raw model range over the *same* class set, so "argmax over all classes" and
"argmax over enrolled classes" coincide and the 0.167-versus-0.375 error becomes impossible.

### Timing

6-second instruction screen, then 21 trials of 20 seconds:

| Phase | Duration | On screen |
|---|---|---|
| Prompt | 2 s | 🤢 Disgust (or the Neutral text) |
| Countdown | 3 s | 3 → 2 → 1 |
| **Hold** | **10 s** | 🤢 Hold it (or `Stay relaxed`) |
| Acknowledgement | 1 s | Recording complete. |
| Relax | 4 s | Relax your face. |

The model is polled once per second during the hold only: 10 ticks per trial, 210 per session.
The relax phase is fixed and never skipped — it is what keeps the next trial's opening frames
from carrying the tail of the previous expression.

### No feedback, in either block

No score, no predicted class, no confidence, no right/wrong signal, no advice. The only per-trial
feedback is `Recording complete.`, identical whatever the model predicted.

This matters more now than it did before. Calibration and evaluation are in the *same* session,
so a participant who saw their score could adapt their acting toward whatever the model rewards
between K and D, and "calibration helped" would be indistinguishable from "the participant
learned." The scored game (`ProgressGame`, which displays `Score: 7/10`) is disabled in the scene
for exactly this reason and must stay disabled during any measured block.

---

## 3. Frozen analysis specification

Fixed before collection, and not adjustable afterwards. Implemented in
[`calibration_study.py`](../Analysis/calibration_study.py), echoed into every result file, and
covered by [`test_calibration_study.py`](../Analysis/test_calibration_study.py).

**Transform.** `z = log(clip(score, epsilon))` with **`epsilon = 1e-12`**, a `log(0)` guard and
not a tunable. The previous default of `1e-6` sat *inside* the live dynamic range — the smallest
score in any recording is 7.16e-06 — and sweeping it moved one session's adapter accuracy from
0.7407 to 0.8333. A floor that can move the headline number by 0.09 is a hyperparameter wearing
a disguise. At 1e-12 its inertness is demonstrable, and a test asserts it.

**Centroid.** Unweighted mean of the nine per-tick `z` vectors, taken *after* the log — the
geometric mean of the probabilities. Averaging in probability space and then taking the log would
give the arithmetic mean, which is dominated by the highest-scoring ticks.

**Decision.** Cosine similarity to the nearest centroid. Recorded as a frozen choice; CLR with
Euclidean distance (the Aitchison geometry proper to compositional data) is the principled
alternative and was **not** selected on evidence — no transform/metric combination beat the
baseline on the cross-session directions.

**Class set.** All seven.

**Ticks.** Tick 0 discarded always: it is a transition sample, caught as the prompt appears.
Ticks 1–9 are the scored set.

**Primary metric.** Per-trial majority vote, then per-participant seven-class macro-F1. Tick
accuracy is secondary and always reported with its design effect.

**Tie rule for the majority vote**, applied identically to every method compared:

1. the class winning the most ticks in the trial;
2. on a tie, the prediction of the trial's **final tracked tick**;
3. if that still does not decide it, the first tied class in the fixed class order
   (`Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise`).

Step 2 is not arbitrary: a tie is usually a trial that changed its mind partway through, and the
end of a held expression is the part the participant had most settled into. Step 3 is arbitrary on
purpose — it has to be fixed, decided in advance, and not read off the scores.

**Missing tracking.** Decided now, because no recording so far has exercised it (all five tracked
at 100%):

- *Calibration (K).* All nine of ticks 1–9 must be tracked for a state. Fewer than nine and no
  centroid is fitted for it, and the session is marked invalid for primary calibration
  evaluation. A partial hold is refused rather than averaged: a centroid from two of nine ticks is
  not a weaker version of the same estimate, it is a different one. `ticks_per_centroid` records
  the count actually used, per state.
- *Evaluation (D).* Baseline and adapter are scored on the **same** tracked tick set. A trial with
  no tracked ticks is **unscored**, never folded silently into the confusion matrix or macro-F1,
  and the number excluded appears in both the console output and the result file.

**Primary macro-F1 completeness.** The macro-F1 helper leaves a class with no support out of the
mean. That is right when the class does not apply and wrong when it is missing because the
recording is incomplete — there it *raises* the figure. So the primary number is valid only when
all seven states have at least one scored evaluation trial and no centroid is incomplete;
otherwise it is marked invalid and reported as a diagnostic with its class coverage.

**Comparisons.** Raw ONNX argmax on D; the session-local adapter on D; and as diagnostic negative
controls, shuffled labels and another participant's centroids.

**Enrolment diagnostics, no thresholds.** Centroid separation (mean pairwise cosine between the
fitted centroids) and K→D drift are computed and reported. Both are derived from the calibration
block alone, so looking at them leaks nothing. **No product threshold is set.** The one
low-quality recording available (C2: separation 0.9798 against 0.8387–0.9098 elsewhere, mean AU
0.0145 against 0.0329–0.0486) is nowhere near enough to fit a gate, and a gate fitted to it would
be fitted to noise. Refusing a bad profile is a safety measure; it is not a fix for weak
expression and must not be presented as one.

**Result files.** One per session, `Analysis/results/calibration_<session_id>.json`, never
overwritten — an existing file is an error. Contents are a whitelist: the session id, protocol
version, the frozen settings above, baseline and adapter metrics, macro-F1, per-class recall, the
confusion matrix, separation and drift diagnostics, sample and trial counts, the replay-fidelity
result, and the rehearsal flag. **Centroid vectors, score vectors and AU windows are never
written.** The research CSV is the consented data record; the analysis output must not become a
second copy of the face signal.

**Replay fidelity.** Every run re-runs the ONNX graph over the logged windows and refuses to
report anything if the replayed scores disagree with the `score_<Class>` columns (exact argmax
agreement, max probability difference under 1e-4; observed drift ~4e-07). A mismatch means the
model on disk is not the one that produced the recording.

**Rehearsal.** A recording made with `fast_rehearsal` on gets three ticks per trial, so after
dropping tick 0 a centroid rests on **two** samples. The analysis prints a `non-production
rehearsal` warning and stamps it into the result file. Such a run checks that the pipeline works;
it is not evidence about calibration.

---

## 4. Effective sample size

Intraclass correlation of the log-scores across ticks within a trial, per session: A1 0.853,
A2 0.799, B1 0.619, C1 0.963, C2 0.543. Design effect ×5.3 to ×8.7, so **nine ticks are worth
roughly 1.2 to 1.8 independent observations.**

Two consequences, both to be written into any report:

- A centroid built from nine ticks is a **one-sample estimate with reduced measurement noise**,
  not a nine-sample estimate. Its standard error is near `SD/√1`, not `SD/√9`.
- D's 14 trials stabilise a participant's estimate but **are not 14 independent observations**.
  **The unit of comparison is the participant.** Run 6–8 people and apply the sign test across
  participants; a single person's session, at any block size, cannot carry a go/no-go decision.

---

## 5. What is recorded

One folder per session, named by UTC start time:

```
session_20260916_175512_8ccb69/
    predictions.csv
    session_metadata.json
```

`predictions.csv` holds one row per prediction — 210 per complete session, 373 columns:

| Column | Contents |
|---|---|
| `timestamp_utc` | when the row was written |
| `session_id` | shared by every row of this session |
| `trial_id` | 0–20, the prompt's index in the session |
| `block_role` | `calibration` or `evaluation`, lowercase |
| `repeat_index` | 1-based presentation of this state **within its block** |
| `prompted_emotion` | what the participant was asked for |
| `prediction_index` | 0–9, the tick within that trial |
| `predicted_emotion`, `correct` | the model's output, recorded but never displayed |
| `score_<Class>` | all seven model outputs |
| `status` | `ok`, or `no_tracking` when the face was lost |
| `generation` | monotonic count of valid frames, for diagnosing dropouts |
| `frame_<f>_au_<a>` | the complete 5 × 70 action-unit window fed to the model |

`block_role` and `repeat_index` are written from the plan, so they are present even on a
`no_tracking` row: a dropout must stay attributable to the block and repeat it happened in.
`repeat_index` restarts per block, so it reads as "the nth time this state was shown in this
block" and does not depend on how long the preceding block was. Together with `block_role` that is
unambiguous, which whole-session numbering would not be.

The scored game uses the same writer but leaves both columns empty, because it has neither block.
The role split then refuses such a file by name rather than treating a game recording as study
data.

Recording the whole input window is what makes offline replay possible: a session can be pushed
back through the same ONNX graph on a laptop, with no headset.

`session_metadata.json` records `protocol_version`, `randomization_seed`, `trial_count`,
`block_count`, `calibration_trials`, `evaluation_trials`, `calibration_passes`,
`evaluation_passes`, `evaluation_repeats`, `ticks_per_trial`, `fast_rehearsal`, the model asset
name, and the class order. The older `blocks` field is retained and **deprecated**; use
`block_count`. Trial counts are taken from the plan that actually ran, so they cannot disagree
with it.

---

## 6. Assumptions and confounds, stated rather than engineered around

**Headset fit.** If the headset is removed at any point after K begins, the recording is
**invalid for primary analysis**: the profile is enrolled against one fit, and a refit changes
which action units register. At this stage an **operator note is sufficient** — no HMD-event
handling is added. Two things help after the fact: `OnApplicationPause(true)` already closes the
logger when the headset comes off, so a removal truncates the CSV visibly, and `generation` steps
reveal dropped frames.

**Order.** D always follows K, so fatigue and learning are inside the effect rather than
controlled for. Mitigations: compare early against late D trials, and read the K→D centroid drift
as the direct measure. Counterbalancing is impossible, because calibration-first *is* the product
flow.

**Neutral's asymmetry.** Every trial is preceded by a relax phase, so a Neutral trial begins with
the face already in its target state while every other state requires a transition into it.
Neutral is therefore systematically advantaged. This is inherent to prompting a rest state.

**Scope.** Results describe controlled trials in which the participant was told which state to
produce. A spontaneously neutral face during ordinary use is outside what this measures.

**No participant identifier in the files.** Recordings are identified only by start timestamp; the
operator keeps the mapping outside the application. If those notes are lost, the affected
recordings cannot be matched to a person, and nothing inside the file can recover it. Write the
note at the moment the session starts.

**No self-report.** Considered and dropped: the study uses no controllers, so an answer could only
be spoken and written down by the operator. The cost is known — when a class scores badly there is
no way to separate "the participant could not produce the expression" from "the model could not
detect it," and Fear scored 0% for both early subjects in exactly this way. Revisiting it needs an
input channel; head-gaze dwell on a 1–5 scale is the most practical controller-free option.

---

## 7. Invalid trials and sessions

Defined **before** collecting, and not changed after seeing results.

- A recording made with `fast_rehearsal: true` is a rehearsal and is never analysed as evidence.
- A session whose `protocol_version` is not `session-local-v1` is not pooled with these.
- A trial whose tracking rate falls below a pre-specified threshold is invalid. **That threshold
  is not yet set** — see section 8. All five existing sessions tracked at 100%, so something
  around 90% is a reasonable starting point.
- A trial interrupted by a wrong prompt, an application fault, or the headset coming off is
  invalid, and per section 6 the headset coming off invalidates the whole recording for primary
  analysis.
- **No trial is ever removed after looking at what the model predicted for it.**

---

## 8. Before data collection can begin

**1. The first recording is a plumbing test.** One self-recorded session, to confirm the runner
produces 21 correctly tagged trials, that the Neutral trial records properly, that K and D
separate in analysis, and that the CSV supports building K centroids and comparing them against D.
It carries no go/kill decision about the method.

**2. IRB approval and a retention decision, before anyone else is recorded.** This study records
facial-movement data from identifiable participants and is not exempt by default. Note what the
CSV actually contains: 350 per-tick action-unit values plus `device_model`, written to
`Application.persistentDataPath` — on Quest that is
`/sdcard/Android/data/<package>/files/sessions`, readable over `adb` without root — and **nothing
is ever deleted**. Retention and access must be decided explicitly rather than inherited from the
logger's default. Participants are referred to by anonymous codes; any mapping to real identities
is kept separately and securely, and never in this repository.

**3. Remaining items to fix before the multi-participant pilot.**

- The tracking-rate threshold from section 7.
- The go/no-go criteria: mean per-participant macro-F1 gain of at least **+0.10**; improvement in
  at least **75%** of participants; **no participant made worse by more than 0.05 macro-F1** —
  this is the criterion the persistent design failed, and the one that matters for a library; no
  class losing more than 0.10 mean recall; Neutral recall remaining acceptable; and Anger and Fear
  judged separately per section 1.3.

**Scale.** 6–8 participants as an engineering pilot, framed as a **product development decision
rule, not a confirmatory hypothesis test**. With 8 participants "6 of 8 improved" is p ≈ 0.15 on a
sign test — enough to decide whether to keep building, not enough to publish. A confirmatory stage
would need 15–20 participants, and if only one study can be run, 15–20 participants is worth more
than more trials from fewer people, because the claim is about generalising across people.

---

## 9. Superseded protocols

Kept so the lineage is readable. None of these are pooled with each other or with the current one.

**`calibration-v1`** — 3 blocks × 7 classes = 21 trials, 20 s each, Neutral prompted. Designed for
the persistent cross-session claim: two visits 1–7 days apart, headset removed and refitted
between them, the first repeat of each class serving as the one-shot enrolment set. Withdrawn with
that claim (section 1.1). **Worth restoring if the single-trial profile proves unstable across
sittings**, since three repeats per state is the only way to get trial-disjoint centroids *and* a
non-degenerate effective sample size within one block.

**`single-pass-v1`** — 1 block × 7 states = 7 trials, Neutral prompted. The two 2026-09-16
recordings (`c626ee`, `1c58ab`) are this. Same person, about 11 hours apart; the second was
deliberately recorded with low-intensity expressions, which is what produced the `C2→C1` collapse
in section 1.1.

**`single-pass-v2`** — 1 block × 6 states = 6 trials, Neutral **not** prompted and the relax phase
not polled, so no Neutral data is captured. Never used for a recording. Its meaning is unchanged
by the current work: `session-local-v1` is a new, separate protocol rather than a redefinition,
because redefining a version in place would break the no-pooling guarantee for anything already
stamped with it.

---

## 10. Change control

Any change to the class set, block structure, timings, or feedback policy requires bumping
`ProtocolVersion` in [`ProtocolRunner.cs`](../Assets/Scripts/ProtocolRunner.cs) to a new value —
never a redefinition of an existing one. Sessions recorded under different protocol versions are
not pooled.
