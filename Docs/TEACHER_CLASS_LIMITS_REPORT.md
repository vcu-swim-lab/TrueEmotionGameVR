# Two classes the teacher does not support, and what that costs the calibration study

**Status:** findings report, 2026-09-22. No preregistered result is changed by it. The
EmojiHero validation and test results stand as recorded; this report is about the model those
results were computed on and about five headset recordings.

**Short version.** On held-out participants from our own acted collection, the teacher
recognises Anger at 0.125 recall and Fear at 0.063. Chance is 0.143. Both are below it, and
they are below it although ten of those fifteen participants are in the training set at roughly
a third of the effective training mass. Session-local calibration cannot repair either class,
and we measured what it does instead: it manufactures predictions for the dead class by taking
them from a neighbouring one. The seven-class protocol is running on a model that supports
five classes.

---

## 1. What prompted this

Five `session-local-v1` recordings now exist. Across all of them Fear was the worst class by a
wide margin, and calibration never recovered it. The obvious explanations — participants not
producing the expression, EmojiHero domain shift, an unlucky cohort — were each plausible and
each turned out to be wrong or insufficient. This report records what the evidence actually
supports, including two hypotheses of ours that it refuted.

---

## 2. What the teacher can and cannot do, by source

The `acted_only` teacher was trained on three sources, source-weighted so that each contributes
roughly equally:

| source | training windows | effective weight |
|---|---|---|
| old acted (Tare et al.) | 1,553 | ≈ 2,833 |
| EmojiHero | 5,784 | ≈ 2,599 |
| new acted (ours, 10 of 15 participants) | 844 | ≈ 2,748 |
| **total** | **8,181** | |

"Acted-only" means no *natural* data. EmojiHero is in the training set. Our fifteen
participants were split participant-exclusively: 10 train, 2 validation, 3 test, with overlap
checks passed.

### Anger and Fear recall, by source

| source | split | Anger | Fear |
|---|---|---|---|
| old acted | train, in-sample | 0.772 | 0.480 |
| old acted | test, held-out | 0.227 | 0.382 |
| **new acted (ours)** | **test, held-out** | **0.125** | **0.063** |
| EmojiHero | test, held-out | 0.404 | 0.127 |

Chance is 1/7 = 0.143.

Two things to read here.

**Anger is memorised.** 0.772 in-sample falls to 0.227 on held-out participants of the same
study, and to 0.125 on ours. A three-fold drop inside the corpus it was trained on is not a
domain-transfer story.

**Neither class survives the transfer to our participants**, and this is the number that
matters for the headset study, because our headset participants are that population. Ten of
those fifteen participants are in training, weighted up 3.23×. The lever "train on our own
people" has already been pulled. It did not lift these two classes.

### Why the pooled figure hid this

The reported `all_acted_pooled` evaluation — accuracy 0.573, macro-F1 0.546 — is exactly the
three held-out sets combined (2,875 windows, 390 Fear; both verified against the per-source
counts). Its composition:

| source | share of windows | share of Fear |
|---|---|---|
| EmojiHero | 78.9% | 83.1% |
| old acted | 10.5% | 8.7% |
| new acted | 10.6% | 8.2% |

Four fifths of that number is EmojiHero. Read as "how the acted-only teacher does on acted
data", it is misleading in both directions: it hides that Fear still half-works on old acted
(0.382) and that it fails completely on ours (0.063).

### Where the errors go

True Fear, held-out, by destination:

| source | Surprise | Neutral | Disgust | **Fear** | Happiness |
|---|---|---|---|---|---|
| new acted | 0.344 | 0.219 | 0.188 | **0.063** | 0.125 |
| EmojiHero | 0.420 | 0.179 | 0.117 | **0.127** | 0.090 |
| old acted | 0.324 | 0.118 | 0.000 | **0.382** | 0.059 |

Fear → Surprise is the top destination in all three sources. That is a systematic confusion,
not noise.

True Anger, held-out:

| source | top destination | Anger |
|---|---|---|
| new acted | Disgust 0.500 | 0.125 |
| old acted | Sadness 0.455 | 0.227 |
| EmojiHero | (Anger 0.404) | 0.404 |

Anger's error destination changes with the corpus. Fear has a stable wrong answer; Anger does
not appear to have a stable signature at all.

### Per-class F1 on the pooled held-out set

| class | F1 | | class | F1 |
|---|---|---|---|---|
| Happiness | 0.761 | | Surprise | 0.630 |
| Neutral | 0.709 | | Anger | **0.434** |
| Sadness | 0.681 | | Disgust | **0.410** |
| | | | Fear | **0.197** |

Three of seven classes are below 0.44. Practically this is a five-class model, and Disgust
is marginal.

---

## 3. The headset recordings

Five recordings under `session-local-v1`. Participant identity across them is not known to the
analysis and is not assumed; these are five *sessions*, not five participants.

`759c71` is carried as a separately labelled data-quality diagnostic — its absolute figures are
an order of magnitude below the others — and is not pooled with the rest. It is listed for
completeness only.

Trial-level macro-F1, trial-disjoint split (fit on the K block, scored on the 14 D trials,
ticks 1–9):

| session | teacher | mode run live | raw | personal | population | hybrid |
|---|---|---|---|---|---|---|
| `85c128` | mixed | — (offline) | 0.595 | **0.681** | 0.434 | 0.629 |
| `3f4b7c` | mixed | hybrid | 0.762 | **0.810** | 0.667 | 0.762 |
| `b3ec46` | mixed | personal | 0.516 | **0.614** | 0.540 | 0.524 |
| `0e088a` | acted-only | raw | 0.733 | **0.781** | — | — |
| *`759c71`* | *mixed* | *— (diagnostic)* | *0.101* | *0.152* | *0.115* | *0.139* |

Deltas on the four pooled sessions:

```
personal − raw       +0.086   +0.048   +0.098   +0.048     4 of 4 positive
hybrid  − personal   −0.052   −0.048   −0.091      —       3 of 3 negative
```

Four sessions all favouring personal, three all disfavouring the population term. Consistent in
direction; with n = 4 a sign test reaches p = 0.125 at best, and the sessions may not be four
distinct people. The population artifact could not be scored on `0e088a` because it is pinned
to the other teacher's output space — the provenance check refused it, which is the intended
behaviour.

**Runtime parity.** `3f4b7c` and `b3ec46` were calibrated on the headset and verified against
an offline recomputation: 126 ticks and 14 trials each, every prediction and every similarity
matching. The runtime and the analysis are the same computation. `0e088a` ran raw, so it
produced no calibrated prediction to verify; its personal figures above are the offline
counterfactual, computed from the calibration block the session recorded anyway.

### Two teachers on the same recordings

Raw tick accuracy, same windows, only the model changed:

| recording | mixed teacher | acted-only teacher |
|---|---|---|
| `b3ec46` | 0.492 | **0.587** |
| `3f4b7c` | **0.770** | 0.738 |
| `0e088a` | 0.762 | **0.810** |

Better on two of three. On `0e088a` the entire gain is Surprise (9/18 → 17/18). Fear does not
move (5/18 → 4/18).

### The class ranking transfers

Per-class results on `0e088a` against the teacher's own held-out evaluation:

| class | acted recall | acted F1 | VR ticks |
|---|---|---|---|
| Fear | 0.144 | 0.197 | 4/18 |
| Anger | 0.358 | 0.434 | 10/18 |
| Disgust | 0.425 | 0.410 | 17/18 |
| Sadness | 0.658 | 0.681 | 18/18 |
| Surprise | 0.693 | 0.630 | 17/18 |
| Neutral | 0.837 | 0.709 | 18/18 |
| Happiness | 0.871 | 0.761 | 18/18 |

Spearman rank correlation: **+0.86** on recall, **+0.90** on F1. The headset results are not
anomalous. They are what this model's own evaluation predicts.

---

## 4. Why calibration cannot repair these classes

Session-local calibration reinterprets the model's seven outputs. It cannot add information the
outputs do not carry. When one output is uninformative, reinterpretation can only move
predictions *into* that class from its neighbours.

We measured this directly. `0e088a`, acted-only teacher, tick hits out of 18:

| class | raw | personal |
|---|---|---|
| Anger | 10 | **5** |
| Fear | 4 | **11** |
| Neutral | 18 | 9 |
| Surprise | 17 | 18 |
| Disgust | 17 | 16 |

Calibration produced seven new Fear predictions and gave up five Anger ones and nine Neutral
ones. Tick accuracy fell from 0.810 to 0.754. Trial-level macro-F1 rose (+0.048) because
macro-F1 rewards a class going from 0 to something, but no information was recovered — the
predictions were taken from elsewhere.

The centroid geometry says the same thing. Cosine between the closest pair of personal
centroids, per session:

| session | teacher | closest pair | cosine |
|---|---|---|---|
| `0e088a` | mixed | Anger ↔ Fear | 0.9994 |
| `0e088a` | acted-only | Anger ↔ Fear | 0.9982 |
| `b3ec46` | mixed | Fear ↔ Neutral | 0.9978 |
| `3f4b7c` | mixed | Fear ↔ Neutral | 0.9917 |

Every session has at least one pair above 0.99, and Fear is in it every time. Two centroids
that close are one centroid with two labels; nearest-centroid classification between them is
decided by noise, and the damage always lands on whichever neighbour Fear has collapsed onto.

`centroid_separation` is already computed on every session and has never had a threshold
attached, deliberately, because one low-quality recording is not enough to set one. There are
now five, and the pattern repeats in all of them.

---

## 5. Two explanations we tested and rejected

Recorded because both were ours and both were wrong, and because the second cost a measurement
worth keeping.

**"Participants are not producing Fear."** Refuted by section 2. Fear works at 0.382 on
held-out participants of the old acted corpus, and it half-works in-sample at 0.480. The
expression is learnable and the model does learn it; it does not transfer to our people. An
earlier claim of ours that this was a production problem was based on a single calibration
trial per class and did not survive the source-level breakdown.

**"Headset expressions are weaker than acted-corpus expressions."** Refuted by measurement. AU
peak intensity, ours against the acted corpus:

| class | acted | VR | ratio | | distance from Neutral (acted → VR) |
|---|---|---|---|---|---|
| Fear | 1.388 | 1.325 | 0.95× | | 0.618 → **0.857** |
| Anger | 1.484 | 1.148 | 0.77× | | 0.721 → **0.805** |

Our headset Fear is *further* from the resting face in action-unit space than the acted
corpus's Fear, and the model still calls it Neutral 63% of the time. Intensity does not explain
it.

---

## 6. What is not established

- **Why the failure destination differs by corpus.** On acted data Fear goes to Surprise
  (0.344); in our headset recordings it goes to Neutral (0.630). Same model, same class,
  different wrong answer. We have no supported explanation and are not offering an unsupported
  one.
- **Sample sizes are small.** `new_acted_test` is three participants, 40 Anger and 32 Fear
  windows; "below chance" rests on 5/40 and 2/32, with wide intervals. It is corroborated by
  the direction of every other source and by five headset sessions, not by its own precision.
- **Participant identity across the headset sessions.** Unknown to the analysis and not
  assumed. Four sessions favouring personal calibration is not four participants favouring it.
- **Disgust.** F1 0.410 pooled, but 17/18 on `0e088a`. It sits between the working classes and
  the broken ones and has not been examined per source.

---

## 7. The decision this forces

The protocol prompts seven states and the primary metric is a seven-class macro-F1. Two of
those classes are below chance on the population the study recruits from. Every participant
spends 3 of 21 trials on Anger and Fear, and 2 of 7 terms in every macro-F1 are contributed by
classes carrying no signal.

This has to be settled before a multi-participant pilot is preregistered, because all three
options change what gets collected:

1. **Keep seven classes, report a preregistered six- or five-class secondary.** Nothing is
   hidden, nothing is selected after the fact, and the primary stays comparable with the
   existing recordings. Costs nothing but leaves the noise in the primary.
2. **Keep seven prompted classes, add a centroid-separation validity gate.** A class whose
   centroid sits above a threshold cosine from another is not enrolled, and is excluded
   symmetrically from every method compared — so no method is credited for ranging over fewer
   classes, which is the inflation error this project has already made once. Applies
   retroactively to the five existing sessions. The threshold must be justified before it is
   applied, not chosen from these five.
3. **Drop Anger and Fear from the protocol.** Cleanest metric, largest loss: a new
   `ProtocolVersion`, no pooling with anything recorded so far, and a deployed system that
   cannot report two of the seven emotions it was commissioned for.

Our reading is that option 2 is the most defensible and option 1 the cheapest, and that option
3 should not be taken on the strength of one teacher — the same two classes may behave
differently under a model trained with more, and more varied, Anger and Fear data.

What the evidence does not support is running the pilot as currently specified and expecting
the calibration question to be answerable. It will not be, for 2 of 7 classes, whatever the
calibration does.

---

## 8. Reproducing the numbers

```bash
# the five headset sessions
python Analysis/pull_sessions.py

# runtime-recorded calibration, and its parity against an offline recomputation
python Analysis/runtime_calibration_report.py --verify

# four methods on one recording, trial-disjoint, with the model made explicit
python Analysis/calibration_study.py <session> \
    --model <path to the teacher that recording was made with> \
    --enrol-role calibration --test-role evaluation \
    --population-centroids Analysis/results/emojihero_population_centroids.json \
    --hybrid-alpha 0.40 --compare-all
```

Per-source teacher metrics are from the training run's own evaluation
(`anger_fear_metrics_by_source.csv`, `anger_fear_prediction_destinations_by_source.csv`) and are
not recomputed here. Everything else in this report was computed from the recordings in
`Logs/quest-sessions/sessions/` and the result files in `Analysis/results/vr_shadow/`.

The teacher a recording was made with is written into its own metadata as
`models.face_model_asset`; `replay_fidelity` fails loudly if the wrong one is supplied.

| teacher | SHA-256 |
|---|---|
| mixed (`sparseVR_teacher_model`) | `ef6cd4134f4d64ab5a7233ac719cc05de88963b33757eca4d92f1b625413343c` |
| acted-only (`sparseVR_teacher_model_acted_only`) | `63538d9d1a3cf84f7e9c0a719dd42818044cd61415548256bdc44c84a7fdee39` |
