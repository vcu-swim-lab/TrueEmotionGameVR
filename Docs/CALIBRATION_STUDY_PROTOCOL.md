# Per-Person Calibration — Recording Protocols

> **Retired 2026-10-19.** The study flow described here (`session-local-v1` to `v3`) was removed
> from `ProtocolRunner.cs`; the game now runs only the former demo flow (`demo-v2`, see the README,
> *The game*). This file is kept as the record of those protocols and their recordings, which are
> never pooled with `demo-v2` ones. The code is in git history (last in commit `64ae8f5`).

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

### 2.0 First trial-disjoint measurement — n = 1

`session_20260916_201632_85c128`, Quest Pro, 21 trials, 100% tracking, all seven centroids on
the full nine ticks, no unscored trials, replay fidelity 1.000 at 2.98e-07. The first honest
measurement of this question:

| | raw ONNX argmax | adapter | delta |
|---|---|---|---|
| tick accuracy | 0.5159 | 0.6746 | **+0.1587** |
| **trial-majority accuracy (primary)** | 0.6429 | 0.7143 | **+0.0714** |
| **macro-F1 (primary)** | 0.5952 | 0.6810 | **+0.0857** |

Per-class trial recall (two trials each), argmax → adapter: Anger 0.50 → **1.00**, Disgust
0.50 → **1.00**, Surprise 0.50 → 0.50, Neutral 1.00 → 1.00, Sadness 1.00 → 1.00, Fear
0.00 → 0.00, Happiness 1.00 → **0.50**.

Both baseline rows are identical (0.6429 / 0.6429), which is the seven-state design working as
intended: the adapter and the raw model range over the same class set, so the baseline cannot be
chosen after the fact.

Enrolment diagnostics: centroid separation 0.87847 (mean pairwise cosine; compare 0.8387–0.9098
on the earlier good recordings and 0.9798 on the low-intensity one). K→D drift 0.97842 mean, and
the drift is lowest exactly where the adapter fails — Fear 0.9460, Happiness 0.9612.

**What this is not.** One participant, one sitting. The unit of a decision is the participant, so
this is a single observation of the quantity that matters, not a result. The evaluation block
always follows the calibration block, so order, fatigue and learning are inside the +0.0857. And
Happiness went *down* by a full trial, which is the failure mode the go/no-go criterion in
section 8 exists to catch — with 14 trials one class moving is well inside noise, which is the
point of not deciding from one person.

**What it does establish**, and this is worth separating from the effect size: the pipeline is
real. A trial-disjoint split is now constructible, the frozen rules fire, and the number is
attributable to a recording and a settings block. Section 1.2 predicted "at or below zero" from
the decay analysis; the measured value is above zero on this participant. That is a reason to run
the 6–8 person pilot, not a reason to believe the method works.

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
Let your face rest.
```

During the hold the screen reads `Stay relaxed`.

Two lines, matching the shape of the emoji prompts, so the Neutral trial does not read as a
different kind of screen — and so it fits the instruction box rather than overflowing it.
It deliberately avoids "Relax your face", which is what the between-trial transition screen
says: reusing those words would blur the boundary between the part that is recorded and the
part that is not.

Where the mouth and eyebrows should sit is said **once, in the intro**, not on the prompt:

```
We will show you one expression at a time.
Hold it until you are told to relax.

One prompt asks you to rest your face instead. For that one,
let your mouth and eyebrows sit where they naturally fall.

You will not be scored.
```

That guidance is worth giving — asked to "be neutral", people often produce a deliberately
blank or tensed face rather than a resting one, and a tensed face is not the resting baseline
the centroid is supposed to capture. But it is too long to repeat on a screen that appears
mid-trial, and the intro has six seconds in which to read it.

Including Neutral also closes the baseline trap of section 1: with all seven states prompted, the
adapter and the raw model range over the *same* class set, so "argmax over all classes" and
"argmax over enrolled classes" coincide and the 0.167-versus-0.375 error becomes impossible.

### Timing

6-second instruction screen, then 21 trials of 20 seconds:

| Phase | Duration | On screen |
|---|---|---|
| Prompt | 2 s | `Trial 3 of 21`, then 🤢 Disgust (or the Neutral text) |
| Countdown | 3 s | 3 → 2 → 1 |
| **Hold** | **10 s** | 🤢 Hold it (or `Stay relaxed`) |
| Acknowledgement | 1 s | Recording complete. |
| Relax | 4 s | Relax your face. |

The model is polled once per second during the hold only: 10 ticks per trial, 210 per session.
The relax phase is fixed and never skipped — it is what keeps the next trial's opening frames
from carrying the tail of the previous expression.

### The trial counter, and what it deliberately does not say

The prompt screen opens with `Trial 3 of 21`. Twenty-one trials is long enough that "how much
is left" is a real question, and leaving it unanswerable is its own distraction.

The counter runs **straight through from 1 to 21**. It does not reset at the block boundary and
it never names the block. Both are deliberate:

- A counter that restarted at trial 8 would announce precisely where K ends and D begins.
- Naming the blocks is worse still. `Analysis/block_effect.py` exists to ask whether K and D
  were the same task performed the same way; a participant who knows which block they are in
  answers that question for it, and the script can no longer separate *the model read the same
  face differently* from *the participant performed the second block differently*. The blocks
  are not counterbalanced — D always follows K in one sitting — so order, practice and fatigue
  already live inside every K-to-D difference. Expectancy does not need to join them.
- The word "evaluation" would contradict the intro's `You will not be scored.`

The counter appears on the **prompt only**, never during the hold. The hold is the recorded
part, and putting fresh text in front of someone while their face is the measurement records
them reading rather than expressing.

Recordings carry `trial_counter_shown: true` in `session_metadata.json`, so a reader can tell a
counter session from an earlier one without knowing which build produced which folder.

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

## 9. EmojiHero offline analogue: `emojihero-personal-v1`

An offline study on the EmojiHeroVR corpus, run by
[`emojihero_calibration_study.py`](../Analysis/emojihero_calibration_study.py). It exists because
section 8 needs 6–8 participants that do not exist yet, and EmojiHero already holds 16 in
participant-disjoint validation and test splits. It tests the same *mechanism* at participant
scale without collecting a headset session.

**It does not replace the live pilot, and a result here is not a bound on the live effect in
either direction.** The two are not the same measurement:

| live `session-local-v1` | EmojiHero |
|---|---|
| ~10 s trial, 9 scored ticks | 30-frame sequence, 6 scored windows |
| ring buffer cleared between predictions, so windows share **zero** frames | stride-1 windows share **4 of 5** frames |
| K then D in one sitting, one headset | pre-recorded sequences across emoji and levels |
| our prompt protocol | the EmojiHero game protocol |

### Unit mapping

The live *trial* maps to an EmojiHero *sequence*. One complete sequence per emotion is enrolled;
that participant's remaining sequences of the same emotion are scored. **A sequence is enrolled or
scored, never split** — splitting it would reproduce the within-trial leak of section 1.2.

### Four methods, one evaluation set

| method | centroids |
|---|---|
| `raw` | none; the model's own argmax |
| `population` | seven centroids from the EmojiHero **training** split |
| `personal` | seven centroids from this participant's enrolled sequences |
| `hybrid` | `H_e = alpha * P_e + (1 - alpha) * G_e` |

All four are scored on identical sequences with the identical tie rule, so every difference is a
difference of decision rule.

**`hybrid − personal` is the comparison that isolates the population term.** `hybrid − raw` cannot:
the personal term alone would move it. A hybrid that beats raw while matching personal has shown
nothing about the population centroid.

### Population centroid

Built **only** from the training split — a validation or test sequence entering it would make the
population term partly personal. The estimator is hierarchical and participant-balanced:

```
window log-scores   -> sequence centroid          (mean over that sequence's 6 windows)
sequence centroids  -> participant-emotion        (mean over that person's sequences)
participant-emotion -> population centroid        (UNWEIGHTED mean over participants)
```

The last step is the point. The corpus is unbalanced — Anger ranges from 3 to 10 sequences per
person — so a flat mean over sequences would weight each participant by how prolific they were, and
the "population" profile would mostly be the few most prolific people. A training participant
missing an emotion simply does not contribute to that emotion, and the count is recorded; fewer
than five contributing participants for any emotion is a hard error.

The artifact is `Analysis/results/emojihero_population_centroids.json`. It carries the model
SHA-256, and a run refuses an artifact built against a different model — the centroids live in that
model's output space and mean nothing outside it. It is the **only** centroid written to disk,
because it is an average over twenty people and describes nobody. Personal centroids are never
written.

### Alpha

One global scalar, swept over `{0.00, 0.05, …, 1.00}`. Not per class, not per participant, not
quality-dependent: each of those is a second thing fitted on the same eight validation
participants, and the grid already contains both endpoints — `alpha = 0` is population-only and
`alpha = 1` is personal-only.

Alpha is chosen on the **validation split only**, by this objective, fixed before it was run:

> the highest participant-equal mean sequence-level seven-class macro-F1 — profiles are averaged
> within a participant first, then participants are averaged with equal weight

Averaging over all participant-profile pairs at once would weight participants by how many
evaluation sequences they happen to have, which is the same mistake the population hierarchy above
exists to avoid. Only eligible participants contribute: a participant who cannot carry a
seven-class figure cannot help choose the weight one will be read from.

Ties, exact or numerical, go to the **larger** alpha — the more personal and less
population-dependent of two equal options. **If alpha = 1 is selected, the population centroid adds
nothing over personal enrolment, and that is the result**, not a reason to pick a different alpha.

### Profile sampling and what its spread means

`MASTER_SEED = 42`, 1000 profiles per participant, selection by
`sha256(seed|split|participant|emotion|profile_index)` indexed into that cell's sequences sorted by
`(timestamp, sequence_id)`. Python's `hash()` is process-salted and is never used. The seven
emotions are chosen **jointly** — a profile is one whole set, because a prediction compares all
seven centroids at once.

Enrolling a sequence removes it from the evaluation set, so the spread across profiles mixes the
calibration choice with the evaluation composition. It is reported as **full protocol sensitivity**,
never as "calibration-sequence sensitivity". Within each profile all four methods are scored on the
same sequences, so each delta is correctly paired.

### Eligibility

A participant enters the primary seven-class analysis only if **every** emotion has at least two
sequences — one to enrol and one to score. A general rule over participant-emotion cells; no
participant ID appears in it. On the current corpus this gives 8 eligible validation participants
and 7 of the 8 test participants, the exception being the one whose Fear class holds a single
sequence. That sequence is **not** split, and the resulting missing class invalidates the
seven-class macro-F1 rather than being dropped from the mean, which would raise it.

### The primary analysis, named before the test split was opened

Exactly one comparison is primary:

| | |
|---|---|
| metric | participant-level seven-class sequence macro-F1 |
| comparison | **hybrid − personal** |
| inference unit | participant |
| test | exact two-sided paired sign test over participants |
| alpha | fixed at the frozen config's `selected_alpha` |

`hybrid − personal` is primary because it is the only comparison that isolates what the
population term contributed; `hybrid − raw` moves when the personal term alone does the work.

**The calibration profiles are not a sample size.** They are repeated re-draws of the same
participants' sequences, averaged *within* a participant to give one value per participant. `n`
is the number of eligible participants. No test is ever computed over profiles.

Secondary, and not promotable afterwards: `hybrid − raw`, `personal − raw`, `population − raw`,
every accuracy figure, window-level metrics, per-class recall, profile sensitivity, the
non-overlapping window condition, and all diagnostics.

### Frozen before test

`--split test` requires `--frozen-config` and refuses, by name, `--hybrid-alpha`,
`--profile-count`, `--windows` and `--frozen-config-output`. There is no `--epsilon`, `--metric`,
`--seed` or `--primary-profile` flag anywhere in the parser. Alpha, profile count, window
condition and primary profile index all come from the config, so a config written against
different code defaults still governs the run.

The config pins the model SHA-256, the population artifact's **semantic** hash, every
preprocessing and decision setting, the selected alpha (stated twice, and a disagreement is an
error), the primary profile index and the robustness count. Any drift between the config and the
running build is a hard error. After the test result is seen, nothing in it may change.

The population pin is a semantic hash rather than a file hash because the artifact records when
it was built: a rebuild from identical inputs is byte-different but means the same thing, and a
byte pin would reject a correct rebuild. The hash covers the schema version, class order,
epsilon, feature space, transform, estimator, preprocessing configuration, model hash, training
corpus fingerprint and the centroid values — and nothing else.

`--dry-run` performs every check that does not require scoring and then stops, running no model
over the split. It is safe against `test` before the test phase is approved.

### Provenance, which limits what each split can carry

- The teacher was trained by gradient on the EmojiHero **training** split, which is also where the
  population centroids come from. The population term therefore describes participants the model
  was fitted on.
- The EmojiHero **validation** split contributed 2,310 of the 3,279 windows in the tensor
  `model.fit` received as `validation_data`, and early stopping monitored `val_accuracy` with
  best-weight restore. That is **~70.4% of the model-selection signal**. No gradients touched it,
  but the stopping epoch and the chosen checkpoint did. It is suitable for developing the adapter
  and choosing alpha; it is not independent evidence about the base model.
- The EmojiHero **test** split entered neither gradients nor model selection, but has been opened
  before when reporting teacher performance. It is not untouched. It is confirmatory here only
  because the adapter and alpha were frozen before it was run.

### VR shadow evaluation

After the frozen EmojiHero test, the same four methods are computed **offline** on the existing
`session-local-v1` recordings: personal centroids from the 7 K trials as specified in section 3,
blended with the frozen population artifact, scored on the 14 D trials. Nothing is shown to the
participant — no prediction, no confidence, no calibration state, no correctness — and the runtime
decision is unchanged. Wiring a calibrated prediction into gameplay is a separate, separately
approved step that cannot happen before all three results have been read.

The shadow evaluation remains offline. Section 10 describes the separate runtime path added
since, which computes a calibrated prediction on the headset and records it. That path still
shows the participant nothing, and still does not reach gameplay.

---

## 10. Runtime calibration on the headset

Sections 3 and 9 describe calibration as an offline reinterpretation of recorded scores. This
section describes the same arithmetic running on the Quest during the session. It changes what the
headset computes; it changes nothing about the protocol the participant experiences, and nothing
about the model.

**The teacher is still never retrained.** Only the interpretation of its seven outputs changes,
exactly as offline.

> **Update 2026-10-12.** The personal, population and hybrid modes, `RuntimeCalibrator` and
> `RuntimeCalibrationSettings` were removed. Calibration is now movement calibration only, from
> the VREmotionDetection library (`MovementCalibrator`), and it is switched on by the **Use
> Calibration** checkbox on the **New Face Au Model** component (off by default). The debug
> logging checkbox is on `ProtocolRunner`. This section records the setup as it was.

### 10.1 Four modes, from two operator toggles

`RuntimeCalibrationSettings`, on the same GameObject as `ProtocolRunner`:

| Enable Personal Calibration | Enable EmojiHero Population Calibration | mode |
|---|---|---|
| off | off | `raw` — the untouched argmax, which is what every recording so far holds |
| on | off | `personal` |
| off | on | `population` |
| on | on | `hybrid`, at `alpha = 0.40` |

`alpha` is a compile-time constant. It was selected on the EmojiHero validation split before the
test split was opened (section 9, "Alpha"), and there is deliberately no inspector field, command
line or config file that can change it. A session recorded at another alpha would not be a session
of this study, and the selection curve is flat enough (0.40 against 0.45 differ by 0.00053) that a
settable knob would invite exactly the re-selection the preregistration rules out.

Both default to **off**, which is `raw`. A session run by someone who never touches these
checkboxes records exactly what this protocol recorded before runtime calibration existed, and the
committed scene holds them off.

Both toggles are read once, at session start. Changing one mid-session cannot affect the running
session: the values are copied into an immutable object and nothing downstream refers back to the
component.

**The component must be present in the scene, and `[RequireComponent]` does not put it there.**
That attribute fires when `ProtocolRunner` is added in the editor; it does nothing to a GameObject
already serialised in a scene. `Main.unity` therefore opened without the component, silently, until
it was added and saved deliberately. `ProtocolRunner.Start` now refuses to run when it is missing,
naming the GameObject, and `Assets/Tests/EditMode/SceneWiringTests.cs` asserts the invariant
against the scene file itself so the failure lands in a test run rather than in front of a
participant.

### 10.2 The protocol is unchanged, in every mode

The block structure, the timings, the randomisation, the prompts and the trial counter are
identical whatever the mode. **The calibration block runs and is recorded identically in all four
modes**, including the two that never use its ticks, so recordings made under different modes stay
comparable trial for trial.

Feedback policy (section 2, "No feedback, in either block") is unchanged and unconditional. The
participant is shown no prediction, no correctness, no score and no calibration state in any mode.

A third checkbox, `Enable Calibration Debug Logging`, writes each evaluation tick's decision to the
Unity console for the operator to read over `adb logcat`. **It draws nothing on the headset panel;
there is no overlay.** It is named for what it does: the field was briefly
`showCalibrationDebugOverlay`, which promised a visible readout that did not exist and would have
led a reader of the metadata to conclude the participant had seen feedback when they had not.
Development builds only, off by default, recorded as `debug_logging_enabled` alongside an explicit
`participant_feedback_shown` stating that nothing is shown in any mode.

### 10.3 The seal

On the way into the first evaluation trial, the centroid set is fitted, blended if the mode calls
for it, and frozen. Nothing reopens it. Ticks 1-9 only, on both sides, matching section 3 exactly:
a runtime vote that also counted tick 0 would disagree with a recomputation of its own recording.

### 10.4 An unfittable calibration is not a raw session

Personal and hybrid require nine tracked ticks for all seven classes. A shortfall, a missing class,
a non-finite score or a wrong-width score vector invalidates the profile. The session continues and
the raw recording stays complete — throwing away a participant's visit over a tracking dropout
would be the worse error — but no evaluation tick is calibrated, and the reason is written into the
metadata, into every evaluation row and into the log. Such a session is reported with raw figures
and an explicit absence where the calibrated ones would be. It is never silently reported as raw.

Population and hybrid additionally require the population artifact, and there the session refuses
to start at all. A run that silently fell back to raw would be filed as a calibrated session and
pooled with real ones.

`fastRehearsal` cannot reach nine ticks and is refused in combination with any calibrated mode,
before the session rather than at the seal.

### 10.5 Provenance of the artifact in the build

`Assets/StreamingAssets/emojihero_population_centroids_unity.json` is a conversion of the frozen
analysis artifact, produced by `Analysis/export_population_artifact_for_unity.py` and carrying the
same seven seven-dimensional vectors. It is aggregate model output over twenty training
participants: no action units, no windows, no individual's centroid, and no EmojiHero raw data
enters a build.

The build checks, before the first trial: schema version, source schema version, source split
(`training`), class order, epsilon, feature space, transform, metric, finiteness, a 7x7 shape, the
recorded model hash against the hash this build expects, the preregistered semantic hash
`ab1f0a1d...` against a compiled-in constant, and a conversion hash recomputed from the file's own
contents.

**Stated limitation:** the runtime cannot re-hash the ONNX graph, because Sentis re-serialises it
at import and the original bytes are not on the device. The end-to-end model pin is the export
script, which hashes the real file, plus the constant above.

### 10.6 Parity is a requirement, not an expectation

The offline analysis and the headset must produce the same number, and this is enforced in two
places rather than assumed:

1. `Analysis/fixtures/runtime_calibration_parity.json` is generated by calling the offline
   functions themselves (`calibration_study.logscore`, `.centroids`, `.classify`, `.trial_vote`) on
   synthetic sessions, and the C# tests replay it, comparing personal centroids, hybrid centroids,
   every tick prediction, every similarity and margin, and every trial-majority vote. Changing the
   offline algorithm changes this fixture and fails the C# until the runtime follows.
2. `Analysis/runtime_calibration_report.py --verify` refits a real recording offline and compares
   it against what the headset wrote. A single differing tick or trial is a parity failure and the
   tool exits non-zero.

Decisions are compared exactly. Magnitudes are compared to a tolerance, because the offline
pipeline carries float32 and the runtime uses double; on the shipped fixture the two precisions
disagree about zero of 630 tick predictions.

### 10.7 What is recorded, and what is not

Added to `predictions.csv` (appended, so no existing column moves), to a new `trials.csv`, and to a
`calibration` block in `session_metadata.json`: the mode, both toggle values, the alpha, the
feature configuration, the artifact's hashes and source split, the tick indices used, whether the
personal centroids were built, the per-tick calibrated prediction with its winner and runner-up
similarities and margin, both trial-majority votes with their tie flags, and the validity flag with
its reason.

**No centroid is written to any file.** A personal centroid is a biometric description of one
identifiable participant. The population centroid values are already in the repository under their
own hash and are not duplicated into results. Section 5's rule is unchanged: what leaves the
headset is predictions, similarities and provenance.

### 10.8 What switching this on does not establish

Nothing about whether calibration helps. That question still rests on the results in section 9 and
in `Docs/VR_SHADOW_EVALUATION_REPORT.md`, which disagree with each other in direction and neither
of which reaches significance at the available sample sizes. This section describes an instrument.
The decision to deploy a calibrated prediction into gameplay remains a separate, separately
approved step, and no such wiring exists: the runtime computes the calibrated prediction, records
it, and shows nobody.

---

## 11. Superseded protocols

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

## 12. Change control

Any change to the class set, block structure, timings, or feedback policy requires bumping
`ProtocolVersion` in [`ProtocolRunner.cs`](../Assets/Scripts/ProtocolRunner.cs) to a new value —
never a redefinition of an existing one. Sessions recorded under different protocol versions are
not pooled.
