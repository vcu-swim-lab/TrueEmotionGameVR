# Subject Variability in the Acted Corpus — Phases A and B

**Scope:** inventory and description of the local `ActingRecordings` corpus (§2–9), a verified
reproduction of the teacher's input pipeline (§4), and per-participant model performance
(§10).
**Reproduce:**
`python Analysis/subject_variability_study.py --json results/subject_variability.json`
`python Analysis/model_performance_study.py --json results/model_performance.json`
**Status:** descriptive and exploratory. No calibration is fitted. Every model number rests on
5 held-out participants and is not a confirmatory test.

> ### Retraction
>
> An earlier version of this report applied the self-report exclusion list to the acted
> corpus and built its primary analysis on the 76 recordings that survived. **Those results
> are withdrawn.** The list is a filter for the *natural* recordings; the teacher notebook
> applies it only to `VideoRecordings` and asserts that it changes no acted recording
> (§3). Every figure that followed from filtering the acted data — in particular the claim
> that the usable held-out set was 19 recordings from 4 participants, and that no held-out
> participant retained seven classes — was an artefact of that mistake.
>
> **The acted corpus is 105 recordings from 15 participants. The held-out set is 35
> recordings from 5 participants, with full seven-class coverage.** The filtered subset is
> retained below only as a labelled sensitivity check.
>
> A second correction, in §2: this report previously gave the capture rate as 70.9 Hz and
> called the documented 60 Hz wrong. The documentation was right; the metric was not.

---

## 1. Summary

Three live sessions gave 61.7% and 65.0% for one subject and 20.0% for another against 14.3%
chance, and the failing subject was **not** under-expressing — their peak action-unit
intensity was higher than the succeeding subject's on both classes they scored 0% on. This
phase asks what the acted corpus says about why.

**Finding 1 — identity variance rivals emotion variance on nearly every channel.** Of 70
channels, only **2** have a 95% bootstrap interval entirely above 1 on the ratio of
between-emotion to between-participant variance. Both are `LipCornerPuller`, the smile. On
this corpus, smiling is the only movement that separates emotions more reliably than it
separates people.

**Finding 2 — where people agree least is where the model fails.** Cross-participant profile
similarity per emotion:

| Emotion | Primary (105) | Sensitivity (76) |
|---|---|---|
| Fear | **0.610** | 0.585 |
| Anger | **0.615** | 0.617 |
| Disgust | 0.641 | 0.618 |
| Surprise | 0.645 | 0.581 |
| Sadness | 0.714 | 0.724 |
| Happiness | **0.819** | **0.830** |
| Neutral | **0.823** | 0.801 |

Fear and Anger sit at the bottom. Those are exactly the two classes the teacher is worst at
(`ANGER_POST_HOC_LOGIT_ADJUSTMENT.md` §6.4: Fear F1 0.35, Anger F1 0.45, against 0.57 macro)
and exactly the two that scored 0% and 20% across both live subjects. Happiness and Neutral,
where people agree most, are where it performs best. Individual pairs go as low as **0.034**
cosine on Fear — two people acting the same prompted emotion with almost no channel in common.

**Finding 3 — the elicitation failed for anger far more than for anything else.** In the
natural recordings, 7 of 15 participants asked to produce anger reported feeling something
else, most often Neutral, Sadness or Disgust. That is nearly half, against 2 of 15 for
Disgust. This does not explain the model's anger deficit and is not offered as one, but it
bears on how that deficit has been interpreted.

**What this does not establish.** High between-participant variance is consistent with two
different causes — measurement (fit, morphology, dead channels) and expression style — and
this phase does not separate them. Section 7 sets out what would.

---

## 2. Inventory

| | |
|---|---|
| Source | `ActingRecordings` (`acted-local` domain) |
| Participants | 15 |
| Recordings | 105 — **one trial per participant per emotion** |
| Emotions | 7, **including Neutral** |
| Channels | 70, named, verified against `OVRPlugin.FaceExpression2` and the CSV header |
| Sample rate | median **60.23 Hz** (58.85–60.28), frames ÷ elapsed time |
| Duration | median 10.00 s |
| Face tracking | median 100%, worst recording 61.3% |
| Participant id | directory name (capture timestamp) |
| Label | directory name, cross-checked against the `Emotion` column |

**A correction to an earlier draft of this report.** It stated the capture runs at 70.9 Hz
and flagged a conflict with the documented 60 Hz. That was wrong, and the error was in the
metric, not the data. The capture is bursty: most frames arrive ~14 ms apart and a minority
arrive much later, so the median inter-frame interval implies 70.9 Hz while frames ÷ elapsed
time gives 60.2 Hz. The second is the rate. The documentation was right, and the teacher's
6-row averaging does land on 10.0 Hz as intended.

**Neutral is present here**, unlike every live session recorded through the old game. This is
the only corpus on hand that can support a seven-class analysis.

## 3. The primary set is all 105 recordings

An exclusion list exists for recordings where a participant's free-text self-report did not
name the emotion they were prompted to produce. **It is a filter for the natural recordings,
not the acted ones.** `final_teacher_notebook (4).ipynb` applies it only to `VideoRecordings`
and asserts that it touches no acted recording:

```python
is_natural = records_new["recording_type"] == "VideoRecordings"
drop_participant = is_natural & ...
assert before_act == after_act, "Acted recordings must not be filtered."
```

with the stated reason: *"Self-report validity only speaks to whether the video elicitation
worked. Acted recordings are instructed poses, so they are kept unfiltered."* Someone asked to
pose anger was not asked to feel it.

The primary analysis therefore uses all 105 recordings, matching the data the model was
trained on. Every held-out participant keeps all seven emotions: **5 participants, 35
recordings, full seven-class coverage.**

The filtered subset is still computed and reported as a **sensitivity analysis**, because it
is a fair question whether someone who did not feel an emotion also posed it badly. It is
never the headline. On that subset the corpus thins sharply — 76 recordings overall, and among
held-out participants only 19 recordings across 4 people, with `P41` losing all
seven and disappearing. No held-out participant retains seven classes there, so that subset
carries no seven-class subject-level score and must not be given one by counting an absent
class as zero.

### Split membership

| Role | Participants |
|---|---|
| train (10) | population reference / prototype construction only |
| validation (2) | `P41`, `P46` |
| test (3) | `P40`, `P48`, `P49` |

Only those five are named in `ANGER_POST_HOC_LOGIT_ADJUSTMENT.md` §5; the other ten are
marked train on the assumption that the validation and test lists are complete. That
assumption is recorded per row in `data/acted_splits.csv` and should be checked against the
notebook before anything leans on it.

## 4. The teacher's input pipeline, as verified

Transcribed into `teacher_preprocessing.py` from `final_teacher_notebook (4).ipynb` cells 9,
12, 15, 17 and 23. Four points contradict the project's own documentation, and each would
have made Phase B's numbers meaningless:

1. **Downsampling is by frame count, not by time.** `df.groupby(arange(len(df)) // 6).mean()`
   — no timestamp is ever read. It happens to give 10.0 Hz here only because the capture rate
   really is ~60.
2. **No scaling is applied.** `mean_centered_max_scale` is defined and commented out at its
   only call site. That single `#` is the entire meaning of `NOT-normalized` in the checkpoint
   name. Raw action-unit values go into the model.
3. **`FaceValid` is never consulted.** The pipeline selects the 70 AU columns by name and
   ignores the tracker's own validity flag. The descriptive analysis in this report *does*
   drop invalid frames, and should; reproducing model input must not.
4. **All 70 channels are used.** A leftover cell asserts the eye channels were removed, which
   would leave 60. It is dead code — the cell that ran asserts the opposite and its saved
   output reads `Number of AU features (including eye AUs): 70`.

**The activity gate is not a filter — it is how a window is created at all.** A window is
emitted only where three consecutive frames each have some channel above 0.25 (inverted for
Neutral, which requires every channel at or below it), and **2 recordings produce no windows
at all**.

### The three conditions Phase B must report separately

The runtime and the notebook do not sample the same way, and conflating them would hide the
difference that matters most. `AUDevice` polls on a 10 Hz timer
([`DeviceManager.cs:161`](../../VREmotionDetection/Runtime/Scripts/DeviceManager.cs#L161)) and
`AwaitWindow` clears the ring buffer and waits for five fresh readings
([`NewFaceAuModel.cs:105`](../../VREmotionDetection/Runtime/Scripts/NewFaceAuModel.cs#L105)),
so the headset feeds the model five **instantaneous** samples 100 ms apart. The notebook feeds
it five **means of six frames**.

| Condition | Sampling | Gate | Windows | What it measures |
|---|---|---|---|---|
| `notebook_exact` | mean of 6 | on | 1259 | the training distribution |
| `notebook_ungated` | mean of 6 | off | 2098 | the gate's contribution, sampling held fixed |
| `runtime_proxy` | every 6th frame | off | 2099 | the closest stand-in for deployed input |

**`notebook_ungated` is not a runtime result and must not be labelled one.** Averaging six
frames smooths; taking one sample does not. Measured on this corpus, mean within-window
temporal roughness is 0.618 for `notebook_ungated` against 0.705 for `runtime_proxy` — the
same direction `ANGER_POST_HOC_LOGIT_ADJUSTMENT.md` §8.2 found between averaged training data
and live headset data, though computed differently here and not comparable to its absolute
figures.

`runtime_proxy` is still a proxy. The device polls against live tracking, and a frame the
tracker rejects is dropped rather than sampled, so a real window can span more than 500 ms of
wall clock. Decimating a continuous 60 Hz capture cannot reproduce that.

## 5. Variance: identity against emotion

For each channel, between-emotion variance (how much it separates emotions within a person) is
compared against between-participant variance (how much the same emotion differs across
people). The bootstrap resamples **participants**, never frames — frames within a trial are
consecutive samples of one continuous expression, and resampling them would report an interval
many times narrower than the data earns.

| | Primary | Sensitivity |
|---|---|---|
| Channels with ratio > 1 | 15 of 70 | 14 of 70 |
| Channels with 95% CI entirely > 1 | **2** | **0** |

The two are `LipCornerPullerL` and `LipCornerPullerR`, ratio 1.40 and 1.44. Every other
channel's advantage is inside bootstrap noise, and several — gaze direction, jaw sideways,
mouth left/right — are clearly identity-dominated, with ratios around 0.23–0.40.

## 6. Measurement diagnostics

Collected per participant: active and dead channel counts, overall peak, mean bilateral
asymmetry over the 29 mirrored channel pairs, and face-tracking rate. The spread is real:

- active expression channels range **32 to 50**
- dead channels range **1 to 11**
- mean bilateral asymmetry ranges **0.062 to 0.301**
- one participant's tracking averages 91.2%; one individual recording falls to 61.3%

`P46` (validation) shows 11 dead channels and the second-highest asymmetry — what a
fit problem would look like. `P41` (validation) shows the *most* active channels of
anyone, 50, with low asymmetry, yet every one of its natural recordings failed the self-report
check. Expressive, well-tracked, and reporting a different emotion than prompted: that is not
a sensor story.

## 7. Measurement or style — what this phase cannot settle

With one trial per participant per emotion there is no way to ask whether a person's deviation
is **repeatable**. A measurement effect should persist across that person's recordings; a
one-off should not. Distinguishing them needs repeated trials, which this corpus does not have.

The honest statement: between-participant variance is large and comparable to between-emotion
variance on almost every channel; the emotions where people diverge most are the emotions the
model fails on; both a measurement and a style explanation remain live.

### What would decide it

- **Repeated trials per person per emotion** — the `calibration-v1` protocol's three blocks.
  A channel pattern that reproduces across a person's own repeats but differs from other
  people's is style. One that does not reproduce was noise.
- **Removing and refitting the headset between sessions** — also in `calibration-v1`. A
  deviation that survives a refit is not fit.
- **The natural corpus, as a paired diagnostic only.** `VideoRecordings` holds the same 15
  participants and the same seven emotions, locally. A measurement effect — a dead channel, a
  lopsided pair, a fit offset — should appear in *both* of a person's conditions; a style
  effect need not. That makes it a useful second angle on §7, but **only as a paired
  within-participant comparison.** It is a different elicitation with a different self-report
  rule and its own filter, so it is never pooled with the acted corpus, never used to enlarge
  a sample, and never averaged into an acted result. Separate domain, separate tables.
- **Per-participant model performance** (Phase B), to test whether high-asymmetry or
  dead-channel participants are in fact the ones the model fails on. On 5 held-out
  participants this remains exploratory.

## 8. Limits

- **One trial per participant per emotion.** No ICC and no within-person repeatability.
  Within-trial stability is measured and names one continuous expression, never repeats.
- **The sensitivity subset carries no seven-class subject-level score**, and none may be
  constructed for it.
- **Split labels are partly inferred** — see §3.
- **One corpus, one protocol.** EmojiHeroVR, the older 2025 recordings and the local natural
  corpus are deliberately not pooled; mixing devices and protocols would turn a protocol
  difference into an apparent person difference, which is the effect under study.
- **Cosine similarity between profiles is not a model metric.** It describes the input, not
  what the classifier does with it.

## 9. Where this leaves the three options

1. **Neutral/scale normalization is sufficient** — *not supported so far.* An offset or gain
   correction acts along a channel; the variance results show people differing in *which*
   channels carry an emotion, which such a correction cannot reach. Offline replay of the live
   sessions already failed to recover the weak subject by rescaling.
2. **Per-person output-centroid adaptation is needed** — *cannot be tested on this corpus.*
   It requires enrolment data disjoint from test data, and one trial per person-emotion
   provides none.
3. **Expression styles differ fundamentally** — *consistent with the evidence, not established
   by it.* The similarity ordering tracking the model's per-class weakness is the strongest
   support; the failure to rule out measurement causes is what stops it being a conclusion.

No method is recommended on this evidence.

## 10. Phase B — per-participant model performance

**Reproduce:** `python Analysis/model_performance_study.py --json results/model_performance.json`

The ONNX path is checked first against the three live sessions, where the headset logged the
scores it computed alongside the exact windows that produced them: agreement 1.000, largest
probability difference 4.2e-07. The acted corpus has no logged scores of its own, so this is
what licenses running the graph on it. The study refuses to report anything if it fails.

Held-out participants only — 5 people, 35 recordings, all seven classes.

| Condition | Windows scored | Offered | Coverage | Mean trial accuracy |
|---|---|---|---|---|
| `notebook_exact` | 422 | 422 | 100% | **0.490** |
| `fixed_windows_gated` | 416 | 700 | **59.4%** | **0.490** |
| `notebook_ungated` | 700 | 700 | 100% | 0.343 |
| `runtime_proxy` | 700 | 700 | 100% | 0.343 |

Chance is 0.143. Per participant under `notebook_exact`: 0.57, 0.17, 0.43, 0.71, 0.57.

### The gain is the activity selection, not the window placement

`notebook_exact` and `notebook_ungated` do not share window boundaries, so the difference
between them is not a measurement of the gate. Ungated windows start at 0, 5, 10, …, while the
gated scan advances one frame on rejection and therefore lands its accepted windows on active
stretches, emits a different number of votes per trial, and leaves one recording with no
prediction at all.

`fixed_windows_gated` is the control: the same 700 fixed windows as `notebook_ungated`, with
the activity rule applied as an accept-or-abstain decision on each. **It reproduces
`notebook_exact` exactly — 0.490 — while holding the boundaries fixed.** So the window-shift
explanation is ruled out and the gain is the selection itself.

**The price is explicit: it declines to answer 41% of windows.** Accuracy under this condition
is measured only on what it answered for and is not comparable to a condition that answers for
everything. Accuracy and coverage must always be quoted together.

Per class, the damage from removing the selection is concentrated:

| Class | `notebook_exact` F1 | `notebook_ungated` F1 |
|---|---|---|
| Anger | 0.50 | **0.00** |
| Fear | 0.44 | 0.33 |
| Sadness | 0.57 | 0.33 |
| Neutral | 0.57 (precision 0.40) | 0.33 (precision **0.21**) |
| Disgust | 0.67 | 0.67 |
| Surprise | 0.00 | 0.00 |

Ungated, Anger is never once predicted correctly, and Neutral precision halves — the model
spends the quiet windows saying Neutral, which is exactly the failure the live sessions showed.

**The deployed runtime has no gate**, so it is running in the ungated regime. But *"adding the
gate would raise live accuracy by 15 points"* does not follow from this, and should not be
said:

- **The runtime policy is undefined.** A gated-out tick could abstain, emit Neutral, hold the
  previous answer, wait for an active window, or be dropped from the game score. Those have
  different accuracies and very different feels to a player. This condition measures exactly
  one of them — abstain, and leave the denominator — because that is the cleanest to
  interpret, not because it is the right product choice.
- **Coverage is part of the result.** At 59.4% the system says nothing four times in ten. A
  score built on that needs a rule for the silent ticks before it means anything.
- **Some of the "errors" it removes may not be errors.** A ten-second recording carries one
  label, so a quiet window at the start or end is scored as a miss even when the participant
  genuinely had a neutral face at that moment. The gate removes those, which flatters it.

What Phase B does support is narrower and still useful: the model performs markedly better on
the active-expression distribution it was trained on than on the ungated distribution the
headset currently feeds it. That makes the activity gate the strongest candidate to try in
C# — after a runtime policy is chosen — rather than a measured 15-point win.

### Averaging versus sampling changes windows but not verdicts

`notebook_ungated` and `runtime_proxy` differ in every one of the 35 recordings, and **5.7% of
individual window predictions differ** (40 of 700). Not one trial verdict changes, because the
majority vote absorbs it.

That is a null result with a boundary. These acted recordings are ten seconds of deliberately
held expression and are smooth by construction; live headset data is measurably rougher
(`ANGER_POST_HOC_LOGIT_ADJUSTMENT.md` §8.2). The finding is that smoothing does not matter
*on held acted poses*, and it should not be carried over to live data without being measured
there.

### Two results that complicate the subject-variability story

**Surprise scores 0.00 on every held-out participant, in every condition** — yet in the three
live sessions it was among the *best* classes, at 90% in all three. Tracing every Surprise
recording across all fifteen participants:

| Outcome | Participants |
|---|---|
| Correct, window accuracy 0.82–1.00 | `P35`, `P36`, `P39`, `P44` (all train) |
| Wrong | the other 11, including all 5 held-out |

The errors scatter — Fear ×3, Neutral ×3, Disgust ×2, Happiness ×2, Anger ×1 — rather than
collapsing onto one class, which is what a class-order or label-mapping bug would produce.
Two checks confirm it is not that:

- The notebook's own saved output reads `le.classes_ = ['Anger', 'Disgust', 'Fear',
  'Happiness', 'Neutral', 'Sadness', 'Surprise']`, identical to the order used here. This is
  now pinned by a regression test, because the live-session fidelity check proves the *graph*
  is right and says nothing about how this corpus is labelled.
- `AU_NAMES` in the notebook begins `BrowLowererL, BrowLowererR, CheekPuffL`, matching the
  channel order used here, and its saved output asserts 70 channels including the eye ones.

So Surprise is not mislabelled. It is recognised near-perfectly for four people and not at all
for eleven — which is the sharpest single illustration of this report's main finding, not a
counterexample to it. Two caveats keep it from being conclusive: all four successes are
training participants, so memorisation cannot be excluded, and six of ten training participants
fail too, so it is not simply a train/held-out split.

**Still unexplained:** why Surprise works live at 90% and fails here. That is a domain
difference between acted recordings and live gameplay which nothing in this report accounts
for, and it should be resolved before Surprise results from either source are trusted.

**Training membership does not explain the spread.** `P38` scores 0.14 under
`notebook_exact` — the worst of all fifteen participants — and is training data. Held-out
participants range 0.17 to 0.71. So the between-participant spread is not simply the model
remembering the people it was fitted to; it is present inside the training set too.

### What this does and does not add to §7

It does not settle measurement against style. `P41` remains the sharpest case: the
most active channels of anyone (50), low bilateral asymmetry, full tracking — and the *worst*
held-out score at 0.17. A sensor explanation has little to work with there. But one
participant is not evidence, and with one trial per emotion there is still no way to ask
whether that deviation is repeatable.

### Limits specific to Phase B

- 5 held-out participants, 35 recordings. Exploratory throughout; no confirmatory claim.
- A subject-level seven-class macro-F1 here is built from seven single-recording classes, so
  it is granular to about 1/7 and is close to a restatement of trial accuracy. Both are
  printed for that reason.
- Under `notebook_exact` one held-out recording produced no window at all and is counted as
  unscored, never as an error — the gate refused to emit an input, which is not a model
  mistake.
- `runtime_proxy` is a proxy. The device polls against live tracking and skips frames the
  tracker rejects, so a real window can span more than 500 ms. Decimating a continuous
  capture cannot reproduce that.

## 11. Next

Three things must happen before the activity gate is worth trying in C#, in this order:

1. **Resolve Surprise end to end.** It works at 90% live and 0% on held-out acted recordings.
   Until that is explained, neither source's Surprise numbers can be trusted, and it sits
   inside every macro-F1 in §10. The remaining check is a golden-window regression test:
   extract a handful of windows, labels and predicted scores from the notebook's own saved
   evaluation and assert this code reproduces them. Class order and channel order are already
   pinned; what is not yet pinned is that the whole acted path matches the notebook's
   evaluation end to end.
2. **Choose the runtime gate policy.** Abstain, emit Neutral, hold the previous answer, wait,
   or drop the tick from the score — each is a different product and a different number. The
   cleanest starting point is abstain plus exclusion from the score denominator, which is
   what `fixed_windows_gated` measures, and it requires the game to report coverage
   alongside any score.
3. **Then try it in C#**, as the strongest available candidate rather than a measured win.

Separately, and not blocking:

4. **The natural corpus** as a paired within-participant diagnostic, never pooled.
5. **The anger elicitation finding** as its own question.
