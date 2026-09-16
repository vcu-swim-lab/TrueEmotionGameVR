# Per-Person Calibration Study — Recording Protocol

**Protocol version:** `calibration-v1`
**Software:** [`ProtocolRunner.cs`](../Assets/Scripts/ProtocolRunner.cs)
**Status:** recording software complete and ready for rehearsal. **Data collection has not
started.** Two items listed in section 8 must be settled first.

---

## 1. Why this study exists

Three headset sessions have been recorded with `sparseVR_teacher_model.onnx`. Two came from
one subject, the third from a second subject. Against a chance level of 14.3%:

| Session | Subject | Tick accuracy |
|---|---|---|
| `20260912_213907` | A | 61.7% |
| `20260913_005336` | A | 65.0% |
| `20260914_175512` | **B** | **20.0%** |

The second subject performs near chance. This is not explained by effort: subject B's peak
action-unit intensity was comparable to subject A's, and *higher* for Anger (0.403 against
0.255) and Fear (0.609 against 0.363) — the two classes B scored 0% on. Nor is it explained
by calibration of signal magnitude: neither baseline subtraction nor gain rescaling recovered
B in offline replay.

An offline pre-experiment ([`calibration_study.py`](../Analysis/calibration_study.py)) found
that fitting a per-person nearest-centroid adapter on part of B's session recovered the rest
of it, from 0.375 to 0.792 against a matched baseline, while subject A's profile applied to B
did nothing (0.208). That result is **not evidence that calibration works.** Enrolment and
test windows came from the same continuous expression, there was one trial per emotion, and
Neutral was never prompted. It is a hypothesis, and this protocol is the experiment that
tests it.

**The question this study answers:**

> Does a calibration profile built from a person's first visit improve seven-class emotion
> recognition for that same person on a **different day**?

The distinction that matters is between "calibration does not work" and "calibration works
but does not survive across sessions." These lead to different product decisions, and the
symmetric design in section 2 separates them.

---

## 2. Session structure

Every session is identical. There is no separate "calibration session" and "test session" —
both visits use the same 21-trial structure, and the analysis assigns the roles. This costs
nothing extra to record and yields several comparisons instead of one.

### Trials

Seven emotion classes, including **Neutral**:

```
Anger   Disgust   Fear   Happiness   Neutral   Sadness   Surprise
```

**Three blocks. Each block contains all seven classes exactly once**, shuffled independently
within the block. Total: 21 trials.

Blocking rather than shuffling all 21 at once is deliberate. A flat shuffle could place an
emotion three times in the first third, and "repeat 1" would stop meaning "this person's
first attempt at this emotion." Blocking guarantees one presentation of every class before
any class comes round again, which is what makes repeat 1 usable on its own as the one-shot
enrolment set — the thing a 60-second calibration would actually capture in a product.

The only cross-block constraint: a block never opens with the emotion the previous block
closed on, so no class is ever acted twice in a row.

Example (seed 12345):

```
Block 1:  Surprise  Neutral  Disgust  Fear  Anger  Sadness  Happiness
Block 2:  Surprise  Anger  Sadness  Disgust  Happiness  Neutral  Fear
Block 3:  Sadness  Anger  Neutral  Disgust  Fear  Happiness  Surprise
```

The randomisation seed is recorded in the session metadata, so any session's order can be
reproduced exactly.

### Timing

Session opens with a 6-second instruction screen, then 21 trials of 20 seconds each.

| Phase | Duration | On screen |
|---|---|---|
| Prompt | 2 s | 🤢 Disgust |
| Countdown | 3 s | 3 → 2 → 1 |
| **Hold** | **10 s** | 🤢 Hold it |
| Acknowledgement | 1 s | Recording complete. |
| Relax | 4 s | Relax your face. |

**Total: about 7 minutes.** The model is polled once per second during the hold phase only,
giving 10 recorded predictions per trial and 210 per session.

The relax phase is fixed and never skipped. It is what keeps the opening frames of the next
trial from carrying the tail of the previous expression.

### Neutral is prompted differently

Neutral has no emoji. Every emoji depicts a performed expression, and showing one would
prompt the participant to *act* neutral rather than to rest. It is prompted with text
instead, worded identically for every participant:

```
Neutral

Let your face relax.
Keep your mouth and eyebrows in their natural position
and look at the dot on the screen.
```

During the hold phase the screen reads `Stay relaxed`.

This matters beyond politeness. The game the lab used previously never prompted Neutral at
all, which is why no seven-class result can be computed from any recording made before this
protocol. Neutral is also the class the deployed model over-predicts, so leaving it out of a
study of that model would remove its most common failure.

### The participant is shown nothing about the model

No score, no predicted class, no confidence, no right/wrong signal, no performance advice.
The only per-trial feedback is `Recording complete.`, which is identical whatever the model
predicted.

This is the single most important difference from the existing game, which displayed
`Score: 7/10` after every trial. A participant who sees their score learns, over a session,
which faces this particular model rewards. They would then arrive at visit 2 as a different
subject, and "did the calibration profile survive?" would be confounded with "did the person
learn to satisfy the model?" Both visits use the same feedback policy: none.

---

## 3. Visit schedule

Each participant attends **two sessions**.

- Preferably **1–7 days apart**. A different day is more valuable than a different hour: it
  tests whether a profile survives normal day-to-day variation in face, fit and mood.
- At minimum, the same day with a 15–30 minute break.
- **The headset must be fully removed and re-fitted between visits.** Headset fit changes
  which action units register, and a profile that only survives an unbroken wearing session
  is not a useful product feature.
- No calibration is repeated or shown to the participant. Both sessions are just recordings.
- The emotion order differs between visits: each session draws its own randomisation seed.

---

## 4. Running a session

The application takes **no input of any kind**. There is no controller, no hand tracking, no
in-headset menu. It runs start to finish on its own.

1. In the Unity scene `Assets/Scenes/Main.unity`, on the `Canvas` object, confirm that
   **`ProtocolRunner` is enabled and `ProgressGame` is disabled.**
2. Confirm **`Fast Rehearsal` is unchecked.** When it is on, every screen carries the banner
   `-- REHEARSAL - NOT VALID DATA --` and the recording is marked as a rehearsal in its
   metadata.
3. Launch the application and fit the headset on the participant.
4. **Write down the start time and who is wearing the headset.** See section 6 — this note is
   the only link between a recording and a person.
5. The session runs for about 7 minutes. Nobody touches anything.
6. The screen reads `Session complete. Thank you.` when it is finished.
7. Collect the recording:

   ```bash
   python Analysis/pull_sessions.py
   ```

8. Check it immediately:

   ```bash
   python Analysis/analyze_sessions.py
   ```

   A tracking rate well below 100% means the headset lost the face; see section 7.

---

## 5. What is recorded

One folder per session, named by UTC start time:

```
session_20260914_175512_8ccb69/
    predictions.csv
    session_metadata.json
```

`predictions.csv` holds one row per prediction — 210 per complete session:

| Column | Contents |
|---|---|
| `timestamp_utc` | when the row was written |
| `trial_id` | 0–20, the prompt's index in the session |
| `prompted_emotion` | what the participant was asked for |
| `prediction_index` | 0–9, the tick within that trial |
| `predicted_emotion`, `correct` | the model's output, recorded but never displayed |
| `score_<Class>` | all seven model outputs |
| `status` | `ok`, or `no_tracking` when the face was lost |
| `generation` | monotonic count of valid frames, for diagnosing dropouts |
| `frame_<f>_au_<a>` | the complete 5 × 70 action-unit window fed to the model |

Recording the whole input window is what makes offline replay possible: a session can be
pushed back through the same ONNX graph on a laptop, with no headset, and reproduces the
logged scores exactly (largest observed difference 4e-07).

`session_metadata.json` records `protocol_version`, `randomization_seed`, `trial_count`,
`blocks`, `ticks_per_trial`, `fast_rehearsal`, the model asset name, and the class order.

**Block and repeat indices are not stored** — they are derived in analysis:
`block = trial_id / 7`, and repeat index is the *n*th occurrence of that emotion in trial
order.

---

## 6. Two things this protocol deliberately does not collect

Both were considered and dropped. They are recorded here so they read as decisions rather
than oversights, and so their cost is known in advance.

### 6.1 No participant identifier in the files

Recordings are identified **only by their start timestamp**. Nothing inside the file says who
it belongs to. The operator keeps that mapping outside the application.

*Consequence:* the link between a person and their two recordings exists only in the
operator's notes. If those notes are lost, or a row is written against the wrong time, the
affected recordings cannot be recovered or verified — there is nothing in the file to check
them against. The mapping must reach analysis in machine-readable form eventually, since the
study compares one person's visit 1 with that same person's visit 2.

*Mitigation:* write the note at the moment the session starts, not afterwards, and record the
start time to the minute.

### 6.2 No self-report

The protocol as originally drafted asked the participant, after every trial, "how well were
you able to make that expression? 1–5". This is not collected.

*Consequence:* when a class scores badly, there is no way to separate two explanations —
the participant could not produce the expression, or the model could not detect it. **Fear
scored 0% for both subjects recorded so far, and that result will remain uninterpretable in
exactly this way.** The same applies to Anger, which is weak for every subject measured.

*Why:* the study uses no controllers and no in-headset input, so an answer could only be
spoken aloud and written down by the operator — 336 hand-recorded answers across the pilot,
each of which has to be matched back to the right trial.

*If this is revisited:* it requires an input channel. Head-gaze dwell on a 1–5 scale is the
most practical controller-free option, since head tracking is always available and needs no
permission.

---

## 7. Invalid trials and sessions

Define these rules **before** collecting, and do not change them after seeing results.

- A recording made with `fast_rehearsal: true` is a rehearsal and is never analysed.
- A session whose `protocol_version` is not `calibration-v1` is not pooled with these.
- A trial whose tracking rate falls below a pre-specified threshold is invalid. **This
  threshold is not yet set** — see section 8. All three existing sessions tracked at 100%, so
  something around 90% is a reasonable starting point.
- A trial interrupted by a wrong prompt, an application fault, or the headset coming off is
  invalid.
- **No trial is ever removed after looking at what the model predicted for it.**

---

## 8. Before data collection can begin

Two items are outstanding.

**1. IRB approval.** This study records facial-movement data from identifiable participants
and is not exempt by default. Discuss with your advisor and VCU IRB before recruiting anyone.
Participants should be referred to by anonymous codes throughout; any mapping to real
identities is kept separately and securely, and never in this repository.

**2. The analysis must be written and frozen first.** This is a methodological requirement,
not a formality. An analysis written after seeing data can be adjusted — usually without any
bad intent — until it produces an agreeable result.

This project already has a concrete example. The pre-experiment initially reported subject B
improving from 0.167 to 0.792, a comparison against a seven-class baseline while the adapter
could only emit six classes. The fair comparison was 0.375 to 0.792. That baseline was chosen
after the data had been seen; choosing it beforehand would have made the error impossible.

The analysis script must therefore fix, in advance:

- the comparison baseline (seven-class, matched to the adapter's class set)
- the primary metric — per-trial majority vote, then per-participant seven-class macro-F1
- which ticks are used, excluding the transition ticks at the start of each hold
- the five comparison directions (visit 1 within, **visit 1 → visit 2** as primary, visit 2
  within, visit 2 → visit 1, and the original within-trial analysis for reference)
- the control conditions: another participant's profile, and a pooled
  leave-one-subject-out profile
- the secondary four-class macro-F1 over Disgust, Happiness, Sadness and Surprise — reported
  alongside the primary, never in place of it
- the go/no-go thresholds
- the tracking-rate threshold from section 7

Proposed go/no-go criteria, to be confirmed before collection: mean macro-F1 gain of at least
+0.10; improvement in at least 75% of participants; no class losing more than 0.10 mean
recall; Neutral recall remaining acceptable; and, if abstention is used, coverage of at least
80%.

**Scale.** Plan for 8 participants × 2 visits as a pilot, explicitly framed as a **product
development decision rule, not a confirmatory hypothesis test**. With 8 participants, "6 of 8
improved" is p ≈ 0.15 on a sign test — enough to decide whether to keep building, not enough
to publish. A confirmatory stage would need 15–20 participants. If only one can be run,
15–20 participants at 3 repetitions is worth more than 8 participants at 5, because the claim
is about generalising across people.

---

## 9. Change control

Any change to the class set, block structure, timings, or feedback policy requires bumping
`ProtocolVersion` in [`ProtocolRunner.cs`](../Assets/Scripts/ProtocolRunner.cs). Sessions
recorded under different protocol versions are not pooled.
