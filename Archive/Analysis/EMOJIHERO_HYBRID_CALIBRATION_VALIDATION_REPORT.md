# EmojiHero Hybrid Calibration — Validation Phase

**Study:** `emojihero-personal-v1` (offline analogue of `session-local-v1`)
**Split reported:** EmojiHeroVR **validation** — 8 participants, 385 sequences
**Test split:** the test split's metadata and label/count information were read during the
dry-run that verifies the pre-frozen integrity and eligibility rules. **No ONNX inference,
prediction, metric or model selection was performed on the test data, and no test outcome was
seen.**
**Audience:** general machine-learning background, no prior knowledge of this project

For the protocol, the frozen rules and the live study this analogue is built from, see
[`CALIBRATION_STUDY_PROTOCOL.md`](../Docs/CALIBRATION_STUDY_PROTOCOL.md) §9 and
[`README.md`](README.md#emojihero-hybrid-calibration-study).

---

## 1. Purpose

Our emotion model works well for some people and poorly for others. Rather than retrain it, we
ask whether its seven output scores can be *reinterpreted* for a particular person. The live
study ([`SESSION_LOCAL_CALIBRATION_REPORT.md`](../Docs/SESSION_LOCAL_CALIBRATION_REPORT.md))
found a positive trial-disjoint result for exactly one participant, and this folder's own
conclusion is that a decision needs 6–8.

EmojiHeroVR already holds 16 participants in participant-disjoint validation and test splits.
This study asks the same question of them, and adds a second one:

1. Do **personal** centroids, from one labelled sequence per emotion, beat the raw model on
   that person's *other* sequences?
2. Does a **population** profile, built from 20 training participants the person never met,
   add anything on top?

The model is never retrained, fine-tuned or re-exported. Only the interpretation changes.

## 2. What was run

Four methods, scored on identical evaluation sequences with an identical tie rule:

| method | centroids |
|---|---|
| `raw` | none — the deployed argmax |
| `population` | seven centroids from the EmojiHero **training** split, participant-balanced |
| `personal` | seven centroids from one enrolled sequence per emotion, this participant |
| `hybrid` | `H_e = alpha · P_e + (1 − alpha) · G_e` |

1000 calibration profiles per participant, deterministically selected from
`sha256(seed | split | participant | emotion | profile_index)` with `MASTER_SEED = 42`. A
calibration sequence is removed from the evaluation set **whole**; no sequence contributes to
both sides, and no sequence is split across them. Between 992 and 1000 of the 1000 profiles
were distinct for each participant.

All 30 corpus integrity checks passed, including 964/385/378 sequences, 20/8/8 participants,
zero participant overlap between splits, 30 frames per sequence, `Label` agreeing with the
sequence ID's emotion field, and the action-unit columns matching the teacher's first 63 in
order. All 8 validation participants are eligible for the seven-class primary analysis.

Population support per emotion: 20 training participants each, except Fear and Sadness at 19 —
one training participant contributes no sequence of each. Those participants simply do not
enter those two centroids, and the counts are recorded in the artifact.

## 3. Alpha was selected on this split, and that matters

The objective, fixed before the run: highest **participant-equal** mean sequence-level
seven-class macro-F1 — profiles averaged within a participant first, then participants averaged
with equal weight. Ties go to the larger alpha.

| alpha | 0.00 | 0.10 | 0.20 | 0.30 | **0.40** | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| macro-F1 | .589 | .607 | .621 | .633 | **.643** | .642 | .635 | .628 | .613 | .601 | .582 |

**Selected alpha = 0.40.** The curve is single-peaked and the interior optimum beats both
endpoints, so on this split a blend genuinely beats either pure method.

## 4. Results

Participant-equal means over all 1000 profiles, eligible participants only:

| method | sequence macro-F1 | sequence accuracy |
|---|---|---|
| raw | 0.6249 | 0.6538 |
| population | 0.5888 | 0.6379 |
| personal | 0.5820 | 0.6231 |
| **hybrid (α = 0.40)** | **0.6427** | **0.6783** |

Paired deltas, per participant, with the exact two-sided sign test over the 8:

| comparison | macro-F1 | improved | p | accuracy |
|---|---|---|---|---|
| personal − raw | **−0.043** | 2/8 | 0.289 | −0.031 |
| population − raw | **−0.036** | 2/8 | 0.289 | −0.016 |
| hybrid − raw | +0.018 | 5/8 | 0.727 | +0.024 |
| hybrid − personal | **+0.061** | **8/8** | **0.008** | +0.055 |

With 8 participants the smallest reachable p is 0.008, so `hybrid − personal` is as clean as
this design can produce — and `hybrid − raw` at 5/8 is not distinguishable from chance.

### 4.1 Three findings, one of them unwelcome

**Personal-only enrolment made things worse, not better.** −0.043 macro-F1, helping only 2 of
8 participants. This is the opposite direction to the single live participant. The most likely
reason is that the enrolment here is far thinner than the live protocol's: one EmojiHero
sequence is 6 windows sharing 4 of every 5 frames, which is close to a single observation,
where a live calibration trial gives 9 ticks across ~10 seconds with no shared frames. A
centroid from one 30-frame performance is a noisy estimate of how a person expresses an
emotion. **This does not overturn the live result, and the live result does not rescue this
one** — they are different enrolment budgets on different data.

**Population-only also lost to raw**, −0.036. Neither recalibration alone beats the deployed
model on this corpus.

**The blend beat personal-only for every single participant.** That is the one unambiguous
signal in the table, and §5 is why it must not yet be read as a discovery.

### 4.2 Per class

Mean sequence-recall change over profiles and participants:

| class | personal − raw | hybrid − personal |
|---|---|---|
| Anger | **+0.103** | −0.095 |
| Disgust | −0.119 | **+0.170** |
| Fear | +0.025 | +0.031 |
| Happiness | −0.057 | +0.097 |
| Neutral | +0.017 | +0.083 |
| Sadness | −0.078 | +0.065 |
| Surprise | **−0.169** | +0.044 |

The adapter redistributes errors between classes rather than lifting them uniformly — the same
behaviour the live study reported. Anger is the one class personal enrolment helps and the
blend gives back.

## 5. The caveat that governs §4

**`hybrid − personal` is an in-sample optimum.** Alpha was chosen to maximise the hybrid
objective on these eight participants, and `hybrid − personal` is then measured on the same
eight. Some of that +0.061 is the selection itself. The number is not a lie — the ranking of
methods at a fixed alpha is real, the pairing is correct, and every method saw identical
sequences — but it is the value the search was pointed at, and an in-sample optimum is
optimistically biased by construction.

The EmojiHero **test** split exists to answer exactly this, which is why alpha, the population
artifact, the preprocessing, the seed and the eligibility rule were frozen first. The test
split's metadata and label/count information were read during the dry-run that verifies those
pre-frozen integrity and eligibility rules; no ONNX inference, prediction, metric or model
selection was performed on the test data, and no test outcome was seen.

## 6. Sensitivity

**Window overlap.** Re-run scoring only the two windows that share no frame (`[0:5]` and
`[5:10]`) at the same alpha: `hybrid − personal` = **+0.053, 8/8 improved, p = 0.008**. The
finding does not depend on the overlapping windows.

**Profile choice.** Spread across the 1000 profiles is reported as *full protocol sensitivity*,
not calibration-sequence sensitivity: enrolling a sequence also removes it from the evaluation
set, so the spread mixes both. Within each profile all four methods are scored on the same
sequences, so each delta is correctly paired. `hybrid − personal` was positive for all 8
participants on the profile-averaged figure and 7 of 8 at the single primary profile.

**Level drift.** The calibration-minus-evaluation mean level gap ranges from −0.044 to +0.099
across participants, i.e. essentially zero on a 1–4 scale. Expression-intensity drift is not
driving these numbers. Reported, never acted on.

**Centroid geometry.** Personal centroids sit at mean pairwise cosine 0.83–0.90 — close
together, so the decision is fragile. Personal-to-population cosine is 0.955–0.975, meaning a
person's profile and the population's are already very similar; the blend is a modest pull
toward a better-estimated version of nearly the same thing, which is consistent with it acting
as shrinkage of a noisy estimator rather than as new information.

## 7. Limitations

- **This is the offline analogue of `session-local-v1`, not a repeat of it.** EmojiHero
  stride-1 windows share 4 of 5 frames where live windows share none; a sequence is 30 frames
  where a live trial is ~10 seconds; the prompt protocol differs. A result here is **not** an
  upper or lower bound on the live effect in either direction.
- **The live 6–8 participant pilot is still required.** Nothing here substitutes for it.
- **Neither split is clean.** The teacher was trained by gradient on the training split, which
  is also where the population centroids come from — the population term describes participants
  the model was fitted on. The validation split supplied 2,310 of the 3,279 windows in the
  tensor the teacher early-stopped on (**~70.4% of the model-selection signal**); no gradients
  touched it, but the stopping epoch and chosen checkpoint did.
- **Eight participants cap the evidence.** The smallest reachable two-sided p is 0.008.
- **Six windows per sequence are not six observations.** The sequence is the unit of
  measurement and the participant the unit of decision. No window or sequence pooling across
  participants was used to manufacture a larger n.
- **Tongue channels are zero-filled** because the teacher was trained that way. For descriptive
  action-unit work they remain unobserved, not measured zeros.
- The optional fixed-evaluation-set analysis, which would isolate calibration-sequence
  sensitivity from evaluation composition, was **not** run. It is secondary and cannot change
  the primary result.

## 8. What is now frozen

**Preregistration snapshot: `cc8ff2d959a6dca4b98b010b4540f835dd428bc6`**
(branch `alimert_work`). The code, the tests, this report, the frozen config and the
population artifact are all in that one commit. Everything below was written before the test
split was opened, and the commit is what shows it.

### 8.1 The primary analysis

Exactly one comparison is primary. It is stated in the code (`FROZEN_SETTINGS.primary_analysis`)
and again in the config, and a disagreement between the two is a hard error.

| | |
|---|---|
| metric | participant-level seven-class sequence macro-F1 |
| comparison | **hybrid − personal** |
| inference unit | **participant** |
| n | eligible participants — 8 on validation, **7 on test** |
| test | exact two-sided paired sign test over participants |
| alpha | **fixed at 0.40**, from the config, never the command line |

**The 1000 profiles are not a sample size.** They are repeated re-draws of the same
participants' sequences. They are averaged *within* a participant to give one value per
participant before any test is computed, and no test is ever computed over profiles.

Secondary, and not promotable after the fact: `hybrid − raw`, `personal − raw`,
`population − raw`, every accuracy figure, window-level metrics, per-class recall, profile
sensitivity, the non-overlapping condition, and all diagnostics.

### 8.2 The pins

| pinned | value |
|---|---|
| alpha | 0.40 |
| ONNX model SHA-256 | `ef6cd4134f4d64ab5a7233ac719cc05de88963b33757eca4d92f1b625413343c` |
| population **semantic** hash | `ab1f0a1de18229374c104434562892ac3fc2b0993d4d925538152ffb045b5ffd` |
| seed / primary profile / robustness | 42 / 0 / 0–999 |
| window condition | `stride1` |
| plus | preprocessing, epsilon 1e-12, cosine metric, centroid definition, tie rule, eligibility rule, missing-class rule |

The population artifact is pinned by a **semantic** hash, not by its file bytes. The artifact
records when it was built, so a rebuild from identical inputs is byte-different but means the
same thing, and a byte pin would reject a correct rebuild. The hash covers the centroid values,
class order, schema version, epsilon, feature space, transform, estimator, preprocessing
configuration, ONNX model hash and training corpus fingerprint — and excludes timestamps, paths
and git commits. Verified: two builds a moment apart produce different file bytes and the same
semantic hash, and an artifact whose centroids are edited after writing is refused on load.

### 8.3 What the test command refuses

`--split test` takes nothing from the command line that could move a primary number. It refuses
`--hybrid-alpha`, `--profile-count`, `--windows` and `--frozen-config-output` by name, naming
every offending flag at once. Alpha, profile count, window condition and primary profile index
come from the config, so a config written against different code defaults still governs the run.
There is deliberately no `--epsilon`, `--metric`, `--seed` or `--primary-profile` flag anywhere
in the parser. The run also refuses a mismatched model hash, a mismatched population semantic
hash, a config missing its `primary_analysis.fixed_alpha` or any `test_plan` pin, and any drift
between the config's `frozen_settings` and the running build. Each of these is covered by a unit
test.

After the test result is seen, none of it may change.

### 8.4 Checks run without opening the test split

- **136 unit tests**, synthetic fixtures, no corpus and no ONNX file required — all pass.
- **50 existing `calibration_study.py` tests** plus the other five suites — all pass unchanged.
- **Corpus integrity gate**, 30 checks — all pass: 964/385/378 sequences, 20/8/8 participants,
  zero participant overlap, 30 frames each, 63 measured action units, `Label` agreeing with the
  sequence ID's emotion field, columns matching the teacher's first 63 in order, all values
  finite, six (5, 70) windows per sequence, tongue channels zero-filled and nothing else.
- **Population rebuild determinism** — identical centroids, identical semantic hash, different
  file bytes.
- **Validation rerun under the frozen code** — bit-identical to the first run (alpha 0.40,
  hybrid − personal +0.060678, 8/8).
- **`--dry-run` against the test split** — this **reads the test split's metadata and
  label/count information** in order to verify the pre-frozen integrity and eligibility rules.
  Corpus, config, artifact and eligibility all validate, and it confirms **n = 7** with the
  single-Fear participant excluded. **No ONNX inference, prediction, metric or model selection
  was performed on the test data, and no test outcome was seen.**

## 9. Recommendation

Proceed to the frozen EmojiHero test split, which is the only thing that can tell whether
`hybrid − personal` survives outside the participants alpha was chosen on. Expect the effect to
shrink; an in-sample optimum usually does.

Two results would each be informative:

- **It holds on test** — the population profile is genuinely shrinking a noisy one-sequence
  personal estimate, and the VR shadow evaluation is worth running.
- **It collapses on test** — alpha was fitted to eight people, and the honest conclusion is
  that one EmojiHero sequence is too thin an enrolment for this method, whatever the blend.

Either way, **the live pilot remains necessary**, personal enrolment's failure here is not
evidence against the live protocol's richer enrolment, and no calibrated prediction goes near
gameplay until the VR shadow evaluation has been read.

---

**Reproduce**

```bash
python Analysis/test_emojihero_calibration_study.py     # 109 tests, no corpus needed
python Analysis/emojihero_calibration_study.py --build-population
python Analysis/emojihero_calibration_study.py \
    --split validation \
    --population-centroids Analysis/results/emojihero_population_centroids.json \
    --profile-count 1000 \
    --output Analysis/results/emojihero_hybrid_validation.json \
    --frozen-config-output Analysis/results/emojihero_hybrid_frozen_config.json
```
