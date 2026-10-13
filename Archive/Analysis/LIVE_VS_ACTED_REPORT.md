# Live Players Against the Acted Training Distribution

**Scope:** the three pulled live sessions — two people, not three — described against the
105-recording acted corpus, with the ten `train` participants as the reference distribution.

**Reproduce:**
`python Analysis/live_vs_acted_study.py --json results/live_vs_acted.json`
`python Analysis/live_vs_acted_study.py --sampling decimated`   (sensitivity check)
`python Analysis/test_live_vs_acted_study.py`   (41 tests)

**Status:** descriptive and exploratory. Nothing is fitted, no calibration is applied, no
model is re-run. N = 2 on the live side.

**The hypothesis tested:** *the model works on player A and fails on player B because A's
per-emotion channel profiles sit closer to the acted participants the teacher was trained on,
and B's sit further away.*

---

## 1. Answers

**1. Is A really closer to the acted training distribution than B?**

Not as a general fact about the two people. It depends entirely on which emotion is asked
about, and it goes both ways:

| | A closer | B closer | indistinguishable |
| --- | --- | --- | --- |
| | Anger, Disgust | Sadness, Surprise | Fear, Happiness |

"Indistinguishable" has a specific meaning here: the A–B gap is smaller than A's own
session-to-session variation for that emotion, measured directly from A's two evenings. On
Fear the gap is 0.032 against A's own 0.184 of drift; on Happiness, 0.021 against 0.025.

The one place the hypothesis holds cleanly is **B's Anger**, at cosine 0.366 against a
training reference that runs 0.387–0.897. It sits below all ten training participants, and B
scores 0.00 on Anger. A's Anger, by contrast, is at 0.873 — above eight of the ten.

**2. Every emotion, or only some?**

Only some, and the pattern does not favour one person. Three of the twelve subject-emotion
cells fall outside the training reference range entirely, and they do not line up with
performance at all:

| cell | cosine | reference range | position | model accuracy |
| --- | --- | --- | --- | --- |
| B Anger | 0.366 | 0.387 – 0.897 | below all 10 | **0.00** |
| A Surprise | 0.568 | 0.585 – 0.929 | below all 10 | **1.00** |
| A Disgust | 0.928 | 0.606 – 0.894 | above all 10 | **1.00** |

A's Surprise is the second-most-distant profile in the whole study and the model scores it
perfectly.

**3. Do B's failing emotions collide with some other emotion's centroid?**

Yes, and this is the clearest positive finding in the report. **Surprise is a sink in the
live profile space.** In four of the twelve cells the live profile was nearest to the acted
*Surprise* centroid when Surprise was not what was prompted — A's Fear, and B's Anger,
Disgust and Fear. In three of those four the model's own dominant misprediction was also
Surprise, and those are the only three cells in the study where it was:

| cell | nearest acted centroid | prompted rank | model's dominant miss |
| --- | --- | --- | --- |
| A Fear | Surprise | 2 of 6 | Surprise ×9 |
| B Fear | Surprise | 2 of 6 | Surprise ×9 |
| B Disgust | Surprise | 5 of 6 | Surprise ×8 |
| B Anger | Surprise | 3 of 6 | Neutral ×5 |

So where a live profile drifts toward acted Surprise, the model tends to answer Surprise. Note
also that both players score **1.00 on Surprise when it is actually prompted**.

This bears directly on the open question in
[SUBJECT_VARIABILITY_REPORT.md §11](SUBJECT_VARIABILITY_REPORT.md) — Surprise works at ~90% live
and 0/5 on held-out acted recordings. The live data does not merely score Surprise well; it
is *biased toward* Surprise. That is the mirror image of the acted held-out behaviour, where
Surprise is never emitted at all. Both facts are consistent with a single explanation — the
live and acted inputs land in different regions of the model's input space, with the
live region sitting on Surprise — and that explanation is testable by the golden-window check
already scheduled.

**4. Sensor or expression style?**

Neither is supported. The measurement diagnostics do not separate the two players, and both
sit inside the acted training spread on every one:

| | active channels | dead channels | bilateral asymmetry | tracking |
| --- | --- | --- | --- | --- |
| live A | 37 | 0 | 0.114 | 1.000 |
| live B | 38 | 3 | 0.166 | 1.000 |
| acted train (n=10) | 32 – 47 (med 38) | 1 – 8 (med 4) | 0.062 – 0.301 (med 0.162) | 1.000 |

B is not a tracking failure, is not unusually asymmetric, and is not missing channels. There
is nothing here a sensor explanation can hold onto. That is a negative finding about the
sensor, not a positive one about style: the live and acted protocols differ in sampling rate,
duration, task and feedback, and each of these numbers moves with all four.

**5. What cannot be said from this data?**

The central one: **profile distance to the training distribution does not predict whether the
model gets the trial right.** Across the twelve subject-emotion cells the rank correlation
between cosine-to-centroid and accuracy is **Spearman +0.007** (Pearson +0.138), and it is
unchanged under the alternative sampling (+0.007 / +0.141).

| subject | emotion | cosine | accuracy |
| --- | --- | --- | --- |
| A | Happiness | 0.953 | 0.94 |
| B | Happiness | 0.932 | 0.11 |
| A | Disgust | 0.928 | 1.00 |
| B | Sadness | 0.904 | 0.22 |
| A | Anger | 0.873 | 0.28 |
| A | Fear | 0.758 | 0.00 |
| B | Fear | 0.725 | 0.00 |
| B | Disgust | 0.705 | 0.00 |
| B | Surprise | 0.703 | 1.00 |
| A | Sadness | 0.698 | 1.00 |
| A | Surprise | 0.568 | 1.00 |
| B | Anger | 0.366 | 0.00 |

Twelve cells from two people are not twelve observations, and no p-value is computed or
implied. But a correlation of +0.007 is not a weak version of the hypothesis; it is the
absence of the effect the hypothesis predicts, in the one place it was most likely to appear.

Also not sayable: anything about Neutral (excluded throughout), anything about a third
player, and any causal reading of a low cosine — different muscle use, headset fit, face
shape and tracker confidence all lower it and this data cannot separate them.

---

## 2. What was compared, and how

**Two live people.** `20260912_213907_eec228` and `20260913_005336_07c050` are the same
person on two evenings; `20260914_175512_8ccb69` is a second person. A's two sessions are
profiled separately and then averaged with **equal weight per session**, not by pooling
frames — pooling would give whichever session ran longer more say in what "A's Fear" means.
The mapping lives in one place (`LIVE_SUBJECTS`) and is asserted by the tests.

**Profiles.** For each person and emotion, the per-channel maximum over the exact 5×70
windows the headset logged. Tick 0 of each trial is excluded: it is a transition sample
caught as the prompt appears, carrying whatever the face was doing on the way out of the
previous trial. Ticks are selected on the `prediction_index` column, never by counting rows,
so a dropped `no_tracking` tick cannot shift the boundary.

This is the same statistic `subject_variability_study.profile` takes over an acted recording's
valid frames, so the two sides are the same measurement.

**Similarity.** Cosine over `au_names.EXPRESSION_INDICES` — the 55 expression channels, with
the eight gaze and seven tongue channels dropped. Gaze is where someone was looking, not how
their face was set; leaving it in would let a player who looked around the scene register as
expressively different.

**"Training distribution" means the ten `train` participants.** The two validation and three
test participants are acted data the model was not fitted on, and calling them training
distribution would answer a different question. They are carried as secondary context only.

**Position in the distribution.** Each training participant is scored against a centroid of
the other nine — leave-one-out, because a participant compared against a centroid containing
themselves is compared partly against themselves. Those ten values are the reference. The live
subject is scored against *the same nine-member centroids*, once per held-out participant and
then averaged, so both sides of the comparison use centroids of the same size. A live subject
never enters any centroid it is scored against; this is asserted by the tests.

With ten reference values the percentile is granular to 10 points and should be read only as
"inside the range" or "below/above everyone".

**Model verdicts** are read, not recomputed: live predictions come from the headset's own
logs, and acted per-recording verdicts are read out of `results/model_performance.json`
(condition `notebook_exact`), which was already checked against the headset's logged scores.

### The sampling confound, and why it does not decide anything

An acted recording holds ~602 tracker-valid frames over ~10 s at ~60 Hz. A live emotion here
holds 45 frames — nine ticks of five. A per-channel *maximum* is exactly the statistic that
cares: a max over 602 draws sits higher than one over 45 on any channel with noise in it,
which inflates noisy channels relative to consistently-used ones and can rotate the profile.

`--sampling decimated` rebuilds the acted profiles from every 6th valid frame — the 10 Hz
spacing the headset itself samples at, ~100 frames. That narrows the gap from ~13× to ~2×
rather than closing it. Every conclusion above is unchanged under it; the largest movement in
any of the twelve cells is 0.005, and the only qualitative change is B's Sadness moving from
9-of-10 to above all ten. A regression test asserts that the sampling choice never flips which
subject is closer for any emotion.

---

## 3. Per-emotion detail

Cosine is to the nine-member leave-one-out training centroids. "A self" is the cosine between
A's two evenings for that emotion — one person's own variation, and the yardstick any A–B gap
has to clear.

| emotion | A cos | B cos | A−B gap | A self | reading | A acc | B acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Anger | 0.873 | 0.366 | +0.507 | 0.631 | A closer | 0.28 | 0.00 |
| Disgust | 0.928 | 0.705 | +0.223 | 0.927 | A closer | 1.00 | 0.00 |
| Fear | 0.758 | 0.725 | +0.032 | 0.816 | within A's own spread | 0.00 | 0.00 |
| Happiness | 0.953 | 0.932 | +0.021 | 0.975 | within A's own spread | 0.94 | 0.11 |
| Sadness | 0.698 | 0.904 | −0.206 | 0.909 | B closer | 1.00 | 0.22 |
| Surprise | 0.568 | 0.703 | −0.135 | 0.980 | B closer | 1.00 | 1.00 |

Overall tick accuracy, ticks 1–9: **A 0.704, B 0.222**. Over all ten ticks — the figure the
in-game score and the earlier reports in this folder count — **A 0.633, B 0.200**, consistent
with the 61.7% / 65.0% / 20.0% quoted in [README.md](README.md).

Two rows deserve attention beyond the summary.

**A's Anger is close and still fails** (0.873, above eight of ten training participants; 0.28
accuracy, dominant miss Neutral ×8). A profile can sit squarely inside the training
distribution and still not be answered correctly.

**A's Anger is also A's least repeatable emotion** — 0.631 between the two evenings, against
0.98 for Surprise and 0.975 for Happiness. One person's Anger varies more between two evenings
than their Surprise varies at all. This matters for the calibration protocol: an enrolment
captured on one visit will transfer unevenly across emotions, and Anger is the one to watch.

### Nearest training participants

The three acted training participants closest to each live profile, with whether the model got
that participant's own trial right (`ok` / `--`). **Those verdicts are on the model's own
training data and describe memorisation, not generalisation.**

```
  subject A
    Anger       P37 0.862 ok   P35 0.819 --   P38 0.810 --
    Disgust     P37 0.964 --   P35 0.889 --   P47 0.877 ok
    Fear        P36 0.928 ok   P44 0.822 ok   P37 0.678 --
    Happiness   P47 0.957 ok   P37 0.949 ok   P43 0.923 --
    Sadness     P37 0.958 ok   P42 0.877 --   P35 0.722 ok
    Surprise    P36 0.759 ok   P44 0.738 ok   P45 0.607 --

  subject B
    Anger       P45 0.526 ok   P36 0.521 --   P38 0.328 --
    Disgust     P36 0.816 --   P38 0.777 --   P47 0.637 ok
    Fear        P36 0.928 ok   P44 0.872 ok   P37 0.673 --
    Happiness   P43 0.940 --   P47 0.918 ok   P38 0.907 --
    Sadness     P43 0.892 ok   P44 0.864 --   P38 0.820 --
    Surprise    P44 0.833 ok   P37 0.783 --   P36 0.723 ok
```

Even B's *nearest* Anger neighbour is only 0.526, where every other row in the table has a
nearest neighbour at 0.759 or above. Whatever B does when asked for Anger, no acted training
participant does much of it.

Two rows also show the reverse of the hypothesis directly: A's Fear nearest neighbour is
`P36` at 0.928 — a participant the model gets Fear right on — and A still scores
0.00 on Fear. B's Fear has the same nearest neighbour at the same 0.928, and also scores 0.00.

---

## 4. Limits

- **Two live people.** Every comparison in this report is one person against one person, with
  one of them sampled twice. No p-value, effect size, or claim about a third player is
  supportable, and none is made.
- **Frames and windows are not sample size.** One live emotion is one continuous performance
  sampled 45 times, not 45 observations. The same applies to the acted side.
- **The twelve-cell correlation is exploratory** and the cells are not independent — six share
  subject A and six share subject B.
- **A low cosine does not identify its cause.** Different muscle use, headset fit, face shape
  and tracker confidence all lower it, and nothing here separates them. The sensor diagnostics
  rule out the grossest failures only.
- **The protocols differ.** Acted: continuous ~10 s capture at ~60 Hz, no feedback, prompted
  from a script. Live: ten 5-frame windows sampled at 10 Hz while the player watches a score
  bar that tells them how they are doing. The second of those is not a neutral difference — a
  live player is adapting to feedback within the trial, which is precisely what the
  calibration protocol forbids for exactly this reason.
- **Neutral is excluded throughout.** It is not merely absent from the live data: the teacher
  pipeline *inverts* the activity rule for Neutral, so an acted Neutral profile is built under
  a different definition of a usable frame. No conclusion about Neutral is available here.
- **Acted verdicts next to neighbours are on training participants** and describe
  memorisation.
- **A's two sessions are one person's two evenings**, not a repeatability study. They were
  recorded a few hours apart under the same conditions and set a floor on that person's
  variation, not a general estimate of it.

## 5. What this changes

Nothing about the model, the runtime, or the protocol — this round was measurement only. Two
things it does change about what to look at next:

1. **The "B is outside the training distribution" explanation is not supported as stated**,
   and should not be the premise of a calibration design. Distance to the training centroid
   and model accuracy are uncorrelated across these twelve cells. Any calibration justified by
   "bring outlier users closer to the training distribution" would be justified by something
   this data does not show.
2. **Surprise gets a second, independent piece of evidence.** It is now not just that Surprise
   scores ~100% live and 0/5 on held-out acted recordings — the live *profiles* also sit
   nearest the acted Surprise centroid in four of twelve cells where Surprise was not asked
   for, and the model's mispredictions follow. That strengthens the case for the golden-window
   end-to-end check being item 1 in [SUBJECT_VARIABILITY_REPORT.md §11](SUBJECT_VARIABILITY_REPORT.md)
   rather than a loose end.
