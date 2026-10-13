# Analysis

Offline analysis of the session logs the game writes while someone plays.

Every prediction the model makes during a session is written to a `predictions.csv` by
[`SessionLogger.cs`](../Assets/Scripts/SessionLogger.cs): the emoji the player was asked
to act, the emotion the model guessed, the score it gave all seven classes, and the whole
five-frame window of 70 action units that produced the guess. This folder turns a pile of
those files into readable numbers — how often the model agreed with the prompt, which
emotions it confuses, whether the headset was tracking the face at all, and which of the
70 action-unit channels ever moved.

Nothing here touches Unity, and it is not part of the build. Python scripts you run on
your laptop after a play session:

- `pull_sessions.py` — copies new recordings off the headset
- `analyze_sessions.py` — turns them into the report
- `block_effect.py` — splits a `session-local-v1` recording into its calibration and
  evaluation blocks and compares them, which says whether the calibration study can be
  read on that recording at all; see [Block effect](#block-effect)
- `calibration_study.py` — replays recordings back through the model to test whether a
  short per-person enrolment would help; see [Calibration study](#calibration-study)
- `emojihero_calibration_study.py` — the same question asked of 16 EmojiHeroVR
  participants offline, with a population profile blended in; see
  [EmojiHero hybrid calibration study](#emojihero-hybrid-calibration-study)
- `live_vs_acted_study.py` — places the live players' expression profiles against the acted
  training participants; see [Live vs acted comparison](#live-vs-acted-comparison)
- `window_level_study.py` — the same comparison at the level of the individual 5×70 window the
  model consumed; see [Window-level comparison](#window-level-comparison)
- `live_pairwise_study.py` — one person's two evenings against the two people, per emotion;
  see [Within-person against between-person](#within-person-against-between-person)

## Requirements

Python 3.10 or newer. `pull_sessions.py` and `analyze_sessions.py` need no packages — the
standard library only, so they run on a lab machine or a student laptop without setting
anything up.

`calibration_study.py` is the exception, because it re-runs the network instead of reading
what the headset already wrote:

```bash
pip install numpy onnxruntime
```

## Usage

```bash
python Analysis/analyze_sessions.py
```

With no arguments it goes looking for session logs itself, so the same command works on
anyone's machine without editing a path into the script. You can always name a path
instead — a CSV, a session folder, or a folder full of session folders:

```bash
# one session, or a whole folder of them
python analyze_sessions.py path/to/predictions.csv
python analyze_sessions.py ~/sessions

# every session, plus one combined report over all of them
python analyze_sessions.py ~/sessions --pool

# per-trial detail, and machine-readable copies for a notebook or a spreadsheet
python analyze_sessions.py --trials --json report.json --trials-csv trials.csv
```

### Where it looks

In order, stopping at the first place that has anything:

1. `$TRUEEMOTION_SESSIONS`, if you set it (several paths allowed, separated the way your
   platform separates `PATH` entries)
2. anywhere inside this checkout — `Logs/quest-sessions/sessions` is where we keep pulled
   sessions, but any folder in the project works, and `Logs/` is gitignored so recordings
   never end up in a commit
3. Unity's `persistentDataPath` for this project on this machine, which is where the
   Editor writes: `AppData/LocalLow/<company>/<product>` on Windows,
   `~/Library/Application Support/<company>/<product>` on macOS. The company and product
   names are read out of `ProjectSettings.asset`, so they stay right if they change.
4. `~/Downloads` and `~/Desktop`, shallowly — the usual landing spot for an `adb pull`

Heavy folders (`Library/`, `Temp/`, `Build/`, `Assets/`, …) are skipped, so scanning the
whole project takes a fraction of a second.

To see what your own machine resolves to:

```bash
python Analysis/analyze_sessions.py --where
```

### Getting the files off the headset

Plug the Quest in and run:

```bash
python Analysis/pull_sessions.py            # copies whatever is new
python Analysis/pull_sessions.py --analyze  # ...and reports on it straight away
```

Recordings you already have are skipped, so running it after every playtest costs nothing,
and it never deletes anything from the headset. Files land in `Logs/quest-sessions/sessions`,
which is gitignored, so a playtest cannot end up in a commit by accident.

There is nothing to configure. adb is found on your PATH, in the Android SDK, in Meta's
Developer Hub, or in the copy Unity installs with Android Build Support — whichever you have.
The package name to pull from is read out of `ProjectSettings.asset`, so renaming the app
does not break it.

| flag | what it does |
| --- | --- |
| `--list` | show what is on the headset without copying anything |
| `--analyze` | run the analysis on the sessions just pulled |
| `--all` | re-pull sessions you already have, overwriting them |
| `--dest DIR` | put them somewhere else |
| `--device SERIAL` | pick a headset, when more than one is plugged in |
| `--package NAME` | pull from a different app |
| `--adb PATH` | point at adb, if it cannot be found |

If it cannot see the headset, the usual causes in order: a charge-only USB cable, Developer
Mode not enabled for that headset in the Meta Quest phone app, or the *Allow USB debugging*
prompt still waiting unanswered inside the headset. The script names which of those it hit.

Each session is a folder holding two files:

```
session_<timestamp>_<id>/predictions.csv
session_<timestamp>_<id>/session_metadata.json
```

Keep them together — the report reads the metadata for the model names, and it records the
class order and window size the data was captured with.

#### Doing it by hand

If you would rather not use the script, or you are on a machine without Python:

```bash
adb pull /sdcard/Android/data/<application.identifier>/files/sessions Logs/quest-sessions
```

Drop the result anywhere in the project (or in Downloads) and the analysis will find it.

Useful flags:

| flag | what it does |
| --- | --- |
| `--pool` | adds a combined report across every file (trial numbers are renumbered so they don't collide) |
| `--pool-only` | prints only that combined report |
| `--trials` | adds the per-trial table to each report |
| `--top-aus N` | how many action units to list (default 12) |
| `--json FILE` | writes every metric, including the ones not printed, as JSON |
| `--trials-csv FILE` | writes one row per trial — the easiest thing to load into pandas or Excel |
| `--where` | prints the folders searched on this machine and what each holds, then exits |

## Reading the report

**session** — how many prediction ticks were logged and how many of them had a tracked
face. `no_tracking` rows are logged rather than dropped, so a low tracking rate is a real
finding: the headset lost the face, and the accuracy below is measured on whatever is left.
`frames per tick` comes from the `generation` counter, which counts valid frames pushed into
the model's ring buffer. A step much shorter than the session's own median means frames were
dropped between predictions; a repeated window means the same five frames were predicted
twice.

**accuracy** — two different things, deliberately:

- *tick accuracy* is the fraction of individual predictions that matched the prompt. This is
  the number the in-game score is built from (the game counts matching seconds out of ten).
- *per trial* rolls each prompt up to one hit or miss, three ways. `majority` is how the
  trial mostly looked, `final tick` is where the player ended up, and `ever hit` is whether
  the expression was reached at all. `ever hit` is always the most generous; the gap between
  it and `majority` tells you whether a player found the expression and held it, or only
  brushed past it.

Both are printed against *chance*, which is 14.3% for seven classes. Anything near that is
not evidence of anything.

*confidence* is the model's top score. If it is high when the model is wrong, the model is
confidently wrong, which is a different problem from an uncertain one.

The **mean score per class** table is where the model's output mass sits, regardless of what
was asked for. Read it before reading anything into a low accuracy row: if a class barely
registers here, no amount of acting will score it, and that row is telling you about the
model rather than about the player.

**confusion** — rows are the emoji the player was asked for, columns are what the model
guessed. Read along a row for the pairs it mixes up. Read *down* a column, using the
`predicted` and `precision` rows underneath, for the more useful question: how eagerly does
the model reach for this class, and how often is it right when it does? The two failures look
identical in an accuracy column and are not the same problem —

- **rarely emitted, high precision** — the model is too shy to use the class. It is right
  when it commits, it just almost never commits.
- **often emitted, low precision** — the class is a sink, absorbing expressions that belong
  to other classes.

Note that the game never prompts Neutral, so every Neutral prediction is a miss by
construction, and its column is a direct measure of how much of the session the model spent
seeing nothing in particular.

**within a trial** — accuracy and the score given to the prompted class by position within
the trial. Tick 0 is the moment the prompt appeared. It normally climbs, because the player
needs a second or two to build the expression; a flat line means the prompt never landed, and
a curve that climbs then falls means they could not hold it.

**action units** — which of the 70 channels ever move. The exported floats bottom out at
denormals (`1.4e-45`) rather than at zero, so anything under `1e-6` is counted as flat. A
large flat count is expected — the headset does not report every channel — but a channel that
goes flat *between* sessions is a tracking problem worth chasing. Measured on the newest frame
of each window only: consecutive windows overlap by four frames, so pooling all five would
count most samples several times over.

## Block effect

```bash
python Analysis/block_effect.py path/to/session_folder

# the same tick window calibration_study.py scores on, for a like-for-like number
python Analysis/block_effect.py path/to/session_folder --ticks 1-9
```

Run this **before** the calibration study on any new recording. The calibration study fits
a per-person profile on the calibration block (K) and scores it on the evaluation block (D);
that only means something if the two blocks are the same task performed the same way. This
script checks that, and it needs no model — it reads the log.

Three numbers per block: what the deployed argmax scored, the score the model gave the
prompted class, and **AU activity** — the sum of the 70 action units in the newest window
frame, averaged over the ticks of the six acted prompts. Neutral is left out of the activity
meter because sitting still is the correct answer for that prompt, so counting it would mix
*did not perform* with *performed correctly*.

Activity is what makes the accuracy difference readable:

| accuracy K→D | AU activity K→D | what it means |
| --- | --- | --- |
| holds | any | the blocks are comparable; the calibration study can be read |
| falls | holds | the same behaviour was read differently — the case calibration exists for |
| falls | falls too | the participant performed D differently; nothing about the model follows |

The third row is a session the calibration study cannot use, and it is not hypothetical.
Of the two `session-local-v1` recordings pulled so far:

```
  session                       calib     eval     delta   activity
  20260916_201632_85c128        62.9%    51.4%   -11 pts      0.91x
  20260917_133343_759c71        74.3%    23.6%   -51 pts      0.47x
```

In the second, Disgust, Happiness, Sadness and Surprise all went from 100% in K to 0–55% in
D, 70% of the evaluation block's ticks were predicted Neutral, and the face moved **53% less**
while it happened. An adapter fitted on K and scored on D there is being asked to predict a
weaker performance than it was fitted on, and will look bad for a reason that has nothing to
do with calibration. The calibration study does report that session — 0.222 deployed, 0.175
adapter — and that number should not be read as evidence against per-person calibration.

The blocks are never counterbalanced: D always follows K in one sitting, so order, fatigue
and practice sit inside every difference printed. The Fisher exact test on trial-majority
hits is over *those* trials only and is not evidence about the next participant. The unit of
any decision remains the participant.

`--ticks 1-9` reproduces `calibration_study.py`'s deployed baseline exactly, which is
asserted by a test: two pieces of code, different paths, same file, same number.

### Tests

```bash
python Analysis/test_block_effect.py
```

24 tests, stdlib `unittest`, no third-party dependency. They cover the splitting (trials land
in the block their rows name, repeats merge without double counting, untracked ticks are
excluded), the activity meter (Neutral excluded, newest frame only), the exact Fisher p
against a value worked out by hand, and each branch of the verdict paragraph. The integration
tests skip themselves when no sessions are pulled to the machine.

## Calibration study

```bash
# screen the recording first -- see Block effect, above
python Analysis/block_effect.py --ticks 1-9

# the split the study is about -- needs a session-local-v1 recording
python Analysis/calibration_study.py --check \
    --enrol-role calibration --test-role evaluation

# the within-trial reference condition, for the older recordings
python Analysis/calibration_study.py --check
```

Five logged sessions, tick accuracy against 14.3% chance: 61.7% and 65.0% (subject A), 20.0%
(subject B), 65.7% and 14.3% (subject C, whose second session was deliberately recorded with
low-intensity expressions). This script asks whether giving the model a handful of labelled
examples from a person makes it better for that person.

**There are two splits, and the difference between them is the whole point.**

`--enrol-role calibration --test-role evaluation` fits the centroids on the calibration block
and scores them on **separate trials** of the same states in the evaluation block. That is the
measurement that means something, and it needs a `session-local-v1` recording.

The default, with no roles given, fits on ticks 1–5 of each trial and tests on ticks 6–9 —
the early and late parts of the *same held expression*. It is kept because its numbers are in
the reports below, and it is optimistically biased. How biased: widening the gap between the
two tick sets decays the mean gain from **+0.150** (adjacent) to **−0.010** (tick 1 against
tick 9), and subject B's headline **+0.4167 decays to exactly 0.000**. Most of what it reported
was short-timescale correlation within one performance.

See [`Docs/CALIBRATION_STUDY_PROTOCOL.md`](../Docs/CALIBRATION_STUDY_PROTOCOL.md) for the frozen
analysis specification, the cross-session results that withdrew the persistent-profile claim,
and the Anger/Fear input-level ceiling that any target has to account for.

### VR shadow evaluation

```bash
python Analysis/calibration_study.py \
    "Logs/quest-sessions/sessions/session_<id>" \
    --check --enrol-role calibration --test-role evaluation \
    --population-centroids Analysis/results/emojihero_population_centroids.json \
    --hybrid-alpha 0.40 --compare-all \
    --results-dir Analysis/results/vr_shadow \
    --json Analysis/results/vr_shadow_<id>.json
```

Adds two further methods to the *same* test ticks: `population`, the nearest of seven centroids
estimated from the EmojiHero training participants, and `hybrid`, the nearest of
`alpha * personal + (1 - alpha) * population`. Alpha is frozen at 0.40 by the EmojiHero
validation phase and is not re-chosen here. Without both options nothing about the output
changes — the personal-only result file keeps the same key set it had before these options
existed, which a test asserts.

**Two deltas, two different questions.** `hybrid - personal` says whether the population term
earned its place. `hybrid - raw` is the deployment gate. They are not interchangeable: a hybrid
that beats the raw model may be doing it entirely with the personal term, and on the one usable
live recording that is exactly what happens.

The population centroids describe how the teacher answers **EmojiHero** input — 30-frame
captures mean-pooled three at a time. A live window is five instantaneous 10 Hz samples. The
blend is well defined, because both live in the model's seven-dimensional output space, but the
population term is applied off the distribution it was estimated on, and the run prints that
caveat every time.

Result: [`Docs/VR_SHADOW_EVALUATION_REPORT.md`](../Docs/VR_SHADOW_EVALUATION_REPORT.md) — an
**n = 1 case study**, because the session logs carry no participant identifier and only one of
the two `session-local-v1` recordings passes the block-effect screen. Personal-only beat the raw
model (+0.086 macro-F1); the hybrid was **worse than personal-only** (−0.052), the opposite
direction to EmojiHero. Both differences are within about one trial of what 14 trials can
resolve. No significance is computed and the deployment gate is not open.

Every run first checks the replayed scores against the `score_<Class>` columns the headset
logged, and **refuses to report anything if they disagree** — a mismatch means the ONNX file
on disk is not the model that produced the recordings, and every number would then describe
the wrong thing. Agreement must be exact and the largest probability difference under `1e-4`
(observed drift between Unity Inference Engine and onnxruntime is ~4e-07). `--check` only
decides whether the table is printed; the failure is enforced either way.

**This is not a validated result and the script says so on every run.** It is a pre-experiment:
it makes the measurement reproducible and forces its choices into the open. Read the validity
block it prints before the numbers.

### The knobs, and why they are knobs

Every one of these changes the headline number. Three of them were found by review to have
been doing silent work in an earlier version of this analysis, which is why they are now
command-line flags with defaults written down rather than choices buried in a script.

| flag | default | why it matters |
| --- | --- | --- |
| `--space` | `logscore` | `logscore` is `log(clip(p, epsilon))`. Cosine on raw probabilities is dominated by whichever class already holds most of the mass, so a subject whose output has collapsed onto one class has nearly the same direction for every trial. The log makes a consistent *ratio* shift — what a mis-mapped subject produces — into a consistent offset. On the third subject: **0.792 with the log, 0.583 without**. |
| `--epsilon` | `1e-6` | The floor under that log — and **not a free parameter**, though this default let it behave like one. The smallest score in any recording is 7.16e-06, so `1e-6` sits *inside* the live dynamic range, and sweeping it moved one session's adapter accuracy from 0.7407 to 0.8333. The frozen specification uses **`1e-12`**, which is provably below every observed score and so demonstrably inert, while still preventing `log(0)`. |
| `--neutral` | `none` | Under `session-local-v1` Neutral is a prompted state, so it is enrolled like any other and this flag is not needed. It exists for the older recordings: `rest` builds a Neutral centroid from tick-0 windows, which backfired (see below), and `prompted` refuses when no Neutral trial exists. |
| `--metric` | `cosine` | `cosine` or `euclidean`. Cosine wins on the within-trial condition (0.750/0.833/0.792 against 0.708/0.667/0.750), but no transform/metric combination beat the baseline on the cross-session directions, so this is a frozen choice rather than an evidenced one. |
| `--enrol-role` / `--test-role` | none | The trial-disjoint split. Both must be given, and they must differ. Refuses a recording without a `block_role` column by name rather than falling back silently. |
| `--enrol-ticks` / `--test-ticks` | `1-5` / `6-9` | The within-trial reference condition, used when no roles are given. Tick 0 is always dropped: it is a transition sample, caught as the prompt appears and before the subject has begun to act. |
| `--results-dir` | `Analysis/results` | One `calibration_<session_id>.json` per session. An existing file is an error, never an overwrite. Centroid vectors, score vectors and AU windows are never written to it. |
| `--abstain-on` | `margin` | What the adapter declines on: gap to the runner-up centroid, or absolute similarity to the winner. See [Abstention](#abstention). |

Tick positions come from the `prediction_index` column, which `SessionLogger.cs` documents
as "0-based tick within the trial". Counting them by enumeration instead is only correct
while no row has been dropped — and rows *are* dropped, since only `ok` rows carry a window.
One `no_tracking` tick mid-trial would compress the timeline and shift rows between the
enrolment and test splits. The run prints which source it used; the fallback exists only for
files written before the column did.

### The baselines are the point

The script prints three rows, and which one you compare against decides the answer. On the
within-trial reference condition, for the first three sessions:

```
  argmax, all classes (deployed)            0.667          0.792          0.167
  argmax, enrolled classes only             0.875          0.875          0.375
  per-person adapter                        0.750          0.833          0.792
```

`argmax, enrolled classes only` is the fair comparison — it restricts the ordinary model to
the same six classes the adapter was enrolled on. Against it the adapter is *worse* on both
sessions from the first subject. Comparing against the deployed seven-way row instead credits
the adapter for excluding Neutral rather than for calibrating anything, and inflates the
third subject's gain from 0.375 → 0.792 into 0.167 → 0.792.

Under `session-local-v1` all seven states are prompted, so the adapter and the raw model range
over the same class set, the two baseline rows coincide, and this particular error becomes
structurally impossible rather than merely documented.

### The primary metric and its tie rule

Accuracy over ticks is secondary. The primary figure is a **per-trial majority vote**, then
per-participant seven-class macro-F1 — because ticks within a trial are strongly correlated
(measured ICC 0.543–0.963, a design effect of ×5.3 to ×8.7), so nine ticks are worth roughly
1.2–1.8 independent observations and the trial is the unit worth reading. The unit of a
*decision* is the participant.

The tie rule is frozen and applied identically to the adapter and to both baselines: the class
winning the most ticks; on a tie, the prediction of the final tracked tick; failing that, the
first tied class in the fixed class order.

Macro-F1 leaves a class with no support out of the mean, which is right when the class does not
apply and wrong when it is missing because the recording is incomplete — there it *raises* the
figure. So the run marks the primary number invalid unless all seven states have a scored
evaluation trial and no centroid is incomplete.

A large part of what looks like cross-subject failure is Neutral over-prediction, which
affects everyone. What survives the fair comparison is narrower but real: on the third
subject, Disgust, Fear and Happiness go from 0/4 to 3–4/4, where the restricted baseline
gets 0/4 on all three. Anger goes the other way — 4/4 down to 0/4. The adapter redistributes
errors between classes per subject rather than uniformly improving.

### What it cannot tell you

The validity block prints these from the data on every run, and what it says depends on which
split is in use.

On the **within-trial reference condition** (no roles given):

- **One trial per state per session** in every recording so far, so enrolment and test come
  from the *same continuous expression*. This measures consistency within one performance, and
  section 1.2 of the protocol document shows how much of the gain that accounts for.
- **Neutral is never prompted** in the pre-`session-local-v1` recordings, so no honest
  seven-class number can be produced from them at all.

On the **trial-disjoint role split**, the within-trial leak is gone and what remains is:

- Both blocks are one sitting, so it says nothing about another day. The cross-session
  question was tested separately and the persistent-profile claim was withdrawn.
- The evaluation block always follows the calibration block, so order, fatigue and learning
  are inside the effect rather than controlled for.
- Ticks within a trial are strongly correlated either way, so the trial is the unit worth
  reading and the participant is the unit of a decision.

Common to both: **zero frames are shared between consecutive windows** —
`NewFaceAuModel.AwaitWindow` clears the ring buffer and waits for five fresh frames, and about
nine frames arrive per one-second tick — so the windows are disjoint and the usual
"overlapping windows" objection does not apply. Adjacent ticks are still correlated.

A decision needs what does not exist yet: **6–8 participants**, with the comparison made
across participants rather than across trials.

### Abstention

The adapter can decline instead of guessing, on either of two criteria. They answer
different questions and neither subsumes the other:

- `margin` — how far the winning centroid beat the runner-up. Catches a sample stranded
  between two classes. It does **not** catch a sample far from *every* centroid: something
  unlike anything enrolled can still win by a wide margin and be confidently wrong.
- `distance` — absolute similarity to the winning centroid. Catches exactly that case, and
  is what an "I don't recognise this expression" gate would have to be built on.

Thresholds are not comparable between the two, nor between `--metric` choices. In log-score
space with cosine, useful margins are around 0.01–0.1 and useful distances around 0.99–0.999.
Note that negative thresholds need `--abstain=-0.5,0` rather than a space, or argparse reads
them as flags.

On the current data `distance` looks the better gate — at 0.99 it keeps 75%/50%/50% of ticks
at 0.833/1.000/0.917 accuracy, where `margin` at 0.01 keeps ~80% at 0.842/0.800/0.800 and
makes the second session *worse*. **Do not read either as an operating point.** Both are
swept and scored on the same 24 correlated test ticks as everything else, no threshold helps
every session at once, and a gate chosen this way is fitted to the sessions it is measured on.

### A result worth not repeating

`--neutral rest` — building the Neutral centroid out of the tick-0 windows — looks like the
obvious way to get a seventh class out of the data we already have. It backfires. The third
subject drops from 0.792 to 0.625, and that subject's Sadness collapses from 4/4 to 0/4: the
resting centroid sits close enough to a held sad expression to swallow it.

Tick 0 is not a rest sample. It is a *transition* sample, caught at the instant the prompt
appears, and it carries whatever the face was doing on the way out of the previous trial.
What the protocol needs is a dedicated neutral/rest enrolment phase — a held, deliberate
resting face with its own prompt and its own settling time — not a Neutral trial slotted into
the existing rotation, and not a frame borrowed from the start of another emotion. The option
is kept so the finding stays reproducible rather than getting rediscovered.

### Tests

```bash
python Analysis/test_calibration_study.py
```

50 tests, stdlib `unittest`, needing only numpy. They cover centroid construction, both metrics
and the case where they disagree, the margin-versus-distance distinction, tick selection
including the dropped-row regression, all three Neutral policies, the epsilon floor, and that a
replay mismatch fails.

The ones guarding this protocol specifically:

- the role split is **trial-disjoint** — no trial appears on both sides — and a recording
  without a `block_role` column is refused by name rather than falling back silently;
- `block_role` is lowercased on read, so a hand-edited `Calibration` does not become a third
  block that the split quietly drops;
- the tie rule at all three levels, including a tie the final tick cannot break, and that the
  same rule reaches every method being compared;
- a state short of nine calibration ticks gets **no** centroid and invalidates the session,
  while the tick-split reference condition still enrols its five ticks — the requirement is
  opt-in, not always on;
- an entirely untracked evaluation trial is reported as unscored rather than dropped, and
  leaves the primary macro-F1 invalid;
- a six-class recording cannot produce a seven-class primary figure;
- a result file is never overwritten, and no centroid vector, score vector or AU window
  reaches it — asserted against the field whitelist, so a new field has to be named before it
  can be written;
- `epsilon = 1e-12` is inert over the observed score range while still preventing `log(0)`.

Replaying the real ONNX graph is one integration test that skips itself when the model or the
recordings are not on the machine.

## EmojiHero hybrid calibration study

```bash
# 1. build the population profile from the EmojiHero TRAINING split (once)
python Analysis/emojihero_calibration_study.py --build-population

# 2. develop the method and choose alpha, on validation only
python Analysis/emojihero_calibration_study.py \
    --split validation \
    --population-centroids Analysis/results/emojihero_population_centroids.json \
    --profile-count 1000 \
    --output Analysis/results/emojihero_hybrid_validation.json \
    --frozen-config-output Analysis/results/emojihero_hybrid_frozen_config.json

# 3. ONLY after the validation report has been read and the test phase approved
python Analysis/emojihero_calibration_study.py \
    --split test \
    --population-centroids Analysis/results/emojihero_population_centroids.json \
    --frozen-config Analysis/results/emojihero_hybrid_frozen_config.json \
    --output Analysis/results/emojihero_hybrid_test.json

python Analysis/test_emojihero_calibration_study.py     # 109 tests, no corpus needed
```

Needs `numpy`, `onnxruntime`, `pandas` and `openpyxl`, and the EmojiHeroVR workbooks in
`Analysis/data` (gitignored) or a zip passed to `--data`. The model is loaded read-only and
executed; it is never retrained, fine-tuned or re-exported.

### What it measures

`calibration_study.py`, above, has **one participant** on the trial-disjoint split, and this
folder's own conclusion is that a decision needs 6–8. EmojiHeroVR already holds 16 in
participant-disjoint validation and test splits. This study asks the same question of them:

> Does a short labelled enrolment from a person improve the model on that same person's
> *separate* performances — and does a population profile built from strangers add anything
> on top?

Four methods, scored on identical evaluation sequences with the identical tie rule:

| method | centroids |
| --- | --- |
| `raw` | none — the deployed argmax |
| `population` | seven centroids from the EmojiHero **training** participants |
| `personal` | seven centroids from one enrolled sequence per emotion, this person |
| `hybrid` | `alpha * personal + (1 - alpha) * population` |

**Read `hybrid − personal`, not `hybrid − raw`.** Only the first isolates what the population
term contributed; the second moves when the personal term alone does the work.

### What it does not measure

It is the **offline analogue** of `session-local-v1`, not a repeat of it, and a gain here is
not an upper or lower bound on the live gain in either direction:

| live `session-local-v1` | EmojiHero |
| --- | --- |
| ~10 s trial, 9 scored ticks | 30-frame sequence, 6 scored windows |
| windows share **zero** frames (`AwaitWindow` clears the ring buffer) | stride-1 windows share **4 of 5** frames |
| K then D in one sitting | pre-recorded sequences across emoji and levels |
| our prompt protocol | the EmojiHero game protocol |

**The live 6–8 participant pilot is still required.** Nothing here substitutes for it.

Neither split is clean, and the run prints why on every execution:

- the teacher was trained by gradient on the **training** split, which is also where the
  population centroids come from;
- the **validation** split supplied 2,310 of the 3,279 windows in the tensor the teacher
  early-stopped on — about **70.4% of the model-selection signal**. No gradients, but the
  stopping epoch and the chosen checkpoint came largely from these participants. It is where
  the method is developed, not evidence about the base model;
- the **test** split touched neither gradients nor model selection, but has been opened
  before when reporting teacher performance. It is confirmatory only because the adapter and
  alpha were frozen first.

### The knobs, and why they are not knobs

Everything that could move a number is in one `FROZEN_SETTINGS` block, written into every
result file and checked against the frozen config before the test split will run.

| setting | value | why |
| --- | --- | --- |
| feature space | `log(clip(p, 1e-12))` | the same transform and the same inert epsilon as `calibration_study.py`, imported from it rather than copied |
| metric | cosine | frozen, not evidenced — see the knobs table above |
| pooling | mean of 3 frames, 30 → 10 | the notebook's EmojiHero path; no normalisation, no scaling |
| windows | length 5, stride 1, six per sequence | none crosses a sequence boundary |
| tongue channels | zero-filled to 70 | **model input only.** `au_names.py` still holds for descriptive work: these channels are unobserved, not measured zeros. The teacher was trained on zeros, so replay must feed zeros |
| aggregation | majority vote → one prediction per sequence | six overlapping windows are not six observations |
| alpha | one global scalar, grid `0.00…1.00` by `0.05` | not per class, not per participant, not quality-dependent |

### Population centroid

Built only from the training split, hierarchically and **participant-balanced**:

```
window log-scores   -> sequence centroid       (mean over that sequence's 6 windows)
sequence centroids  -> participant-emotion     (mean over that person's sequences)
participant-emotion -> population centroid     (UNWEIGHTED mean over participants)
```

The last step is the whole point. Anger ranges from 3 to 10 sequences per person, so a flat
mean over sequences would make the "population" profile mostly the most prolific
participants. A training participant missing an emotion contributes nothing to it and the
count is recorded; fewer than five contributing participants for an emotion is a hard error.

The artifact carries the model SHA-256 and a run **refuses one built against a different
model** — these centroids live in that model's output space. It is the only centroid written
to disk, because it averages twenty people and describes nobody; personal centroids never
reach a file. It is one of two files under `results/` that version control keeps, the other
being the frozen config: a preregistration that is not in the history cannot show it was
written before the test split was opened.

### The semantic hash, and why the file hash is not the pin

The artifact records when it was built, so rebuilding it from identical inputs produces
identical centroids inside a byte-different file. A byte hash would therefore reject a
*correct* rebuild, which is the opposite of what a pin is for. So the frozen config pins a
**semantic hash** instead — a SHA-256 over a canonical serialisation of:

```
schema version · class order · epsilon · feature space · transform · estimator
preprocessing config · ONNX model SHA-256 · training corpus fingerprint · centroid values
```

and of nothing else. Timestamps, file paths and git commits are excluded; none of them can
change a centroid. Loading an artifact recomputes this from the file's own contents and
refuses one whose centroids were edited after it was written.

```bash
# same inputs, different file bytes, same pin
python Analysis/emojihero_calibration_study.py --build-population --population-output /tmp/rebuilt.json
```

### Alpha selection

On validation only, by an objective fixed before it ran: the highest **participant-equal**
mean sequence-level seven-class macro-F1 — profiles averaged within a participant first, then
participants averaged with equal weight. Pooling all participant-profile pairs would weight
people by how many sequences they happen to have, which is the mistake the population
hierarchy above exists to avoid.

Ties go to the **larger** alpha, the more personal option. If `alpha = 1` is selected, the
population adds nothing over personal enrolment and the report says exactly that.

### Profile sensitivity

1000 calibration profiles per participant, selected by
`sha256(seed|split|participant|emotion|profile_index)` — never Python's process-salted
`hash()`. The seven emotions are drawn **jointly**, because a prediction compares all seven
centroids at once.

Enrolling a sequence removes it from the evaluation set, so the spread across profiles mixes
the calibration choice with the evaluation composition. It is reported as **full protocol
sensitivity**, not as calibration-sequence sensitivity. Within a profile all four methods see
the same sequences, so every delta is correctly paired.

### Eligibility, and the class that cannot be scored

A participant enters the primary seven-class analysis only if every emotion has at least two
sequences — one to enrol, one to score. A general rule over participant-emotion cells, with no
participant ID in it. On the current corpus that is 8 validation participants and 7 of the 8
test participants; the exception has a single Fear sequence. **That sequence is not split**,
and the missing class invalidates the seven-class macro-F1 rather than being dropped from the
mean — dropping it would *raise* the figure.

### The frozen primary analysis

Exactly one comparison is primary, and it was named before the test split was opened:

| | |
| --- | --- |
| metric | participant-level seven-class sequence macro-F1 |
| comparison | **hybrid − personal** |
| inference unit | participant |
| test | exact two-sided paired sign test over participants |
| alpha | the frozen config's `selected_alpha`; never the command line |

**The 1000 profiles are not a sample size.** They are repeated re-draws of the same
participants' sequences. They are averaged *within* a participant to give one value per
participant, and `n` is the number of eligible participants — 8 on validation, 7 on test.
No test is ever computed over profiles.

Everything else is secondary and cannot be promoted afterwards: `hybrid − raw`,
`personal − raw`, `population − raw`, all accuracy figures, window-level metrics, per-class
recall, profile sensitivity, the non-overlapping condition, and every diagnostic below.

### Running the test split, and what it refuses

`--split test` takes **nothing** that could move a primary number from the command line.
`--hybrid-alpha`, `--profile-count`, `--windows` and `--frozen-config-output` are all
refused by name, and there is deliberately no `--epsilon`, `--metric`, `--seed` or
`--primary-profile` flag anywhere in the parser. The alpha, profile count, window condition
and primary profile index all come from the frozen config, so a config written against
different defaults still governs the run. The run also refuses a mismatched model hash, a
mismatched population semantic hash, and any drift between the config's `frozen_settings`
and the running build.

`--dry-run` does every check that does not require scoring — corpus integrity, config
validation, artifact validation, eligibility, and printing the frozen primary-analysis
definition — then stops. It runs no model over the split and produces no prediction, so it
is safe against `test` before the test phase is approved:

```bash
python Analysis/emojihero_calibration_study.py --split test \
    --population-centroids Analysis/results/emojihero_population_centroids.json \
    --frozen-config Analysis/results/emojihero_hybrid_frozen_config.json \
    --dry-run
```

### Diagnostics that never select anything

Level and emoji IDs, centroid separation for all three families, personal→population cosine,
calibration→evaluation drift, window-level metrics, and a two-window non-overlapping
sensitivity condition (`--windows non-overlapping`). A positive calibration-minus-evaluation
level gap would mean the enrolled sequences came from harder levels than the ones scored,
confounding personalisation with expression-intensity drift. Reported, never acted on.

## Runtime calibration on the headset

Everything above fits centroids *after* a session, from the recorded scores. This section is
the other thing: the same arithmetic running on the Quest, so the evaluation block is
classified as it happens rather than months later.

```bash
# 1. put the frozen population profile where a build can read it (once, and after any rebuild)
python Analysis/export_population_artifact_for_unity.py
python Analysis/export_population_artifact_for_unity.py --check   # verify, write nothing

# 2. regenerate the golden fixture the C# is held to (only if the offline algorithm changed)
python Analysis/make_runtime_parity_fixture.py

# 3. after pulling a session, report on what the headset decided
python Analysis/runtime_calibration_report.py --verify

# tests
dotnet test Analysis/CsharpTests/Calibration.Tests.csproj   # 123, includes Python parity
python Analysis/test_runtime_calibration_report.py          # 27
```

### Three checkboxes, five modes

They live on the `RuntimeCalibrationSettings` component, on the same GameObject as
`ProtocolRunner` — the `Canvas` object in `Assets/Scenes/Main.unity`.

Both default to **off**, which is `raw`: a session set up by someone who never touches them
records exactly what this protocol recorded before runtime calibration existed. That is also
what the committed scene holds, and `SceneWiringTests` fails if it ever stops being true.

| Enable Personal Calibration | Enable EmojiHero Population Calibration | mode | what the evaluation block is scored against |
|---|---|---|---|
| off | off | `raw` | nothing; the untouched model argmax, as before |
| on | off | `personal` | seven centroids fitted from this participant's calibration block |
| off | on | `population` | the frozen EmojiHero profile: twenty strangers' average |
| on | on | `hybrid` | `0.40 * personal + 0.60 * population` |

The third checkbox, **Enable Movement Calibration**, runs alone:

| Enable Movement Calibration | mode | what the evaluation block is scored against |
|---|---|---|
| on (the other two off) | `movement` | two fingerprints per emotion from the calibration block: the personal score centroid, and the player's typical face (the 63 face action units, each window averaged over its frames; the 7 tongue values left out). A tick goes to the emotion with the highest *score cosine + movement cosine* |

It cannot be combined with the other two. It already contains the personal score centroids, so
`personal + movement` would count that term twice, and a combination with the population profile
was never measured. The inspector clears the other side when one side is ticked
(`RuntimeCalibrationSettings.OnValidate`), and `CalibrationSessionConfig.Lock` refuses the
combination at session start in case the scene file was edited by hand. In movement mode the
winner, runner-up and margin columns are sums of two cosines, so they range from 0 to 2.

Offline, on the 9 headset sessions available (8 of them one person), movement calibration
reached tick macro-F1 0.730 against 0.647 for personal calibration, better in all 9 sessions:
`Analysis/exploratory/au_calibration.py`, method `fuse-sim`. `calibration_study.classify_movement`
reproduces that result exactly, and the parity fixture holds the C# to it.

The alpha is a `const` in `CalibrationMath`. There is no inspector field, no command line and
no config file that can move it: it was selected on the EmojiHero validation split before the
test split was opened, and a session recorded at some other alpha is not a session of that
study.

Both toggles are read **once**, at session start, into a `CalibrationSessionConfig`. Nothing
downstream holds a reference back to the component, so a checkbox ticked mid-session cannot
reach the running session. A recording that started as personal-only and finished as hybrid
would be a recording of neither.

### Scene wiring, which `RequireComponent` does not give you

`[RequireComponent(typeof(RuntimeCalibrationSettings))]` adds the component when
`ProtocolRunner` is *added* in the editor. It does nothing to a GameObject already serialised
in a scene, so `Main.unity` opened without the component and with no warning until it was
added deliberately and saved.

Three things now cover that, because the attribute alone plainly does not:

- **`SceneWiringTests`** parses `Main.unity` as text and asserts that every GameObject carrying
  `ProtocolRunner` also carries `RuntimeCalibrationSettings`, that its fields are serialised
  under their current names, that it appears in the host's `m_Component` list (so it actually
  shows in the inspector), and that both checkboxes are saved off. It runs under `dotnet test`
  as well as in Unity, so it is part of the ordinary sweep.
- **`ProtocolRunner.Start`** checks for null explicitly and refuses to run with a message
  naming the GameObject and what to add, instead of failing with a NullReferenceException
  partway through a session.
- **`Tools > Calibration > Wire calibration settings into the open scene`**
  ([CalibrationSceneWiring.cs](../Assets/Editor/CalibrationSceneWiring.cs)) is the fixup, kept
  in the repository rather than run once and deleted. It is idempotent and adds nothing but the
  missing component. In batch mode:
  `-executeMethod CalibrationSceneWiring.WireMainScene`.

### What the participant sees: nothing

Identical in all four modes. No prediction, no correctness, no score, no progress feedback.
The evaluation block is the measurement, and a participant who can see what the model thinks
is performing a feedback task instead.

The third checkbox, **Enable Calibration Debug Logging**, writes each evaluation tick's
decision to the Unity console, readable over `adb logcat`, and **nothing to the headset
panel**. There is no overlay. It is named for what it does: the field was briefly called
`showCalibrationDebugOverlay`, which promised a visible readout that did not exist and would
have made a reader of the metadata conclude the participant had seen feedback when they had
not. Development builds only, off by default, and recorded either way as
`debug_logging_enabled` beside `participant_feedback_shown`.

### K to D, and the seal

The calibration block runs identically in every mode, including `raw` and `population`, where
its ticks are recorded but never become centroids. That keeps recordings comparable across
modes. On the way into the first evaluation trial the profile is **sealed**: centroids are
fitted, blended if the mode asks for it, and frozen for the rest of the session. A calibration
tick arriving after the seal is an error, not a late addition.

Only ticks 1-9 are taken, on both sides. Tick 0 is a transition sample, and a runtime vote
that counted it would disagree with a recomputation of the same recording, which is the one
thing the parity check exists to make impossible.

### A calibration that cannot be fitted does not become a raw session

Personal and hybrid need nine tracked ticks for all seven classes. Anything less (a dropout, a
missing class, a non-finite score, a wrong-width score vector) and the profile is refused. The
session keeps running and the raw recording stays complete and usable, but no evaluation tick
is calibrated, and the reason is written into the metadata, into every evaluation row, and
into the log.

Population and hybrid also need the artifact, and there the session does not start at all: a
missing, tampered or wrong-model profile is fatal before the first trial. A run that quietly
scored raw would be filed as a calibrated one and pooled with real ones.

`fastRehearsal` records three ticks per trial and so can never reach nine. Combining it with
any calibrated mode is refused up front rather than discovered at the seal, after a
participant has already sat through the session.

### The artifact in the build

`Assets/StreamingAssets/emojihero_population_centroids_unity.json`, written by the export
script from `Analysis/results/emojihero_population_centroids.json`. Aggregate model output
only: seven seven-dimensional vectors averaged over twenty training participants, carrying no
action units, no windows and no individual's centroid. No EmojiHero raw data enters a build.

The export validates the source through `load_population_centroids` (schema, the ONNX file's
own hash, source split, class order, epsilon, shape, finiteness, and the semantic hash
recomputed from the artifact's contents), then reshapes it and re-reads what it wrote,
comparing IEEE-754 bit patterns rather than decimal strings.

Two hashes travel with it and both are checked at load:

- **`semantic_sha256`** is the preregistration pin, `ab1f0a1d...`, carried through unchanged
  and compared against a constant compiled into the build. It answers *which* profile this is.
- **`conversion_sha256`** is a hash over the converted file's meaning, computable in both
  languages. It answers *has this file been edited since export*.

They are separate because the first cannot be recomputed in C#: it hashes a canonical
`json.dumps`, and Python's float repr and .NET's `"R"` format agree on digits but not on
exponent spelling. The conversion hash sidesteps formatting entirely by serialising every
double as its bit pattern in hex, which `struct.pack('>d', v).hex()` and
`BitConverter.DoubleToInt64Bits(v).ToString("x16")` produce identically.

**Known limitation:** the runtime cannot re-hash the ONNX itself. Sentis re-serialises the
graph at import and the original bytes are not on the device, so what is checked is the hash
the artifact *records* against the hash this build expects. The end-to-end pin is the export
script, which hashes the real file.

### Parity, in two places

The offline analysis and the headset have to be the same number computed twice, not two
numbers that happen to agree.

`Analysis/fixtures/runtime_calibration_parity.json` is generated by calling
`calibration_study.logscore`, `.centroids`, `.classify` and `.trial_vote`, the functions the
published results came from, on synthetic sessions. `Analysis/CsharpTests` replays it through
the C# and compares personal centroids, hybrid centroids, every tick prediction, every
similarity and margin, and every trial-majority vote. Change the offline algorithm and this
fixture changes; the C# tests then fail until the runtime is changed to match.

Two claims are checked separately. Against the float64 reference, that the arithmetic is the
same arithmetic. Against the float32 pipeline the analysis actually runs, that the runtime
reaches the same *decisions*. Bit-exact agreement is neither achievable nor claimed, since
NumPy's vectorised `log` and .NET's `Math.Log` differ by about an ulp, so magnitudes are
compared to 1e-12 and decisions exactly. On the shipped fixture the two precisions disagree on
**zero** of 630 ticks.

`runtime_calibration_report.py --verify` is the second place. It refits from a real recording's
calibration block, reclassifies its evaluation block, and compares against what the headset
wrote. One differing tick or trial is a parity failure and the tool exits non-zero. It
recomputes from the *logged score columns* rather than replaying windows through ONNX, which
keeps the check on the calibration arithmetic; whether Sentis agrees with onnxruntime is the
separate `replay_fidelity` question.

### What the report says, and what it refuses to say

Evaluation block only: all fourteen D trials, none of the seven K trials. Tick accuracy, trial
accuracy, seven-class trial macro-F1, per-class precision/recall/F1, the confusion matrix, and
the trials calibration changed, split into the ones it fixed and the ones it broke.

Every figure comes from `calibrated_predicted_emotion` as the headset wrote it, not from a
recomputation. A recomputation presented as a recording would hide exactly the bug worth
finding.

A `raw` session gets raw figures and an explicit `null` where the calibrated ones would be; so
does a session whose profile failed to seal, with the reason. Sessions recorded under
different modes are reported separately and never pooled: personal-only and hybrid answer
different questions and a mean over them answers neither.

### New columns

`predictions.csv` gains twelve columns, appended after the action units so nothing already
written moves: `calibration_mode`, `personal_calibration_enabled`,
`population_calibration_enabled`, `movement_calibration_enabled` (added with movement mode),
`hybrid_alpha`, `raw_predicted_emotion`,
`calibrated_predicted_emotion`, `calibrated_winner_similarity`,
`calibrated_runner_up_similarity`, `calibrated_margin`, `calibration_valid`,
`calibration_error`. `raw_predicted_emotion` is an explicit alias for `predicted_emotion`:
once calibrated columns sit beside it, "predicted" stops being self-explanatory.

`trials.csv` is new, and is the file meant to be read by eye. One row per trial with both
majority votes, both correctness flags, the scored-tick counts, and whether either vote was
decided by the tie rule.

`session_metadata.json` gains a `calibration` block: mode, both checkbox values, alpha,
epsilon, feature space, transform, metric, estimator, class order, the tick indices the
personal centroids used, whether they were built, the tie rule, the runtime implementation
version, the debug flag, and the artifact's hashes and source split.

The scored game's recordings are untouched. `ProgressGame` constructs a `SessionLogger`
without a calibration config, and in that case no column, no `trials.csv` and no metadata
block is added.

**No centroid, personal or population, is written to any of them.** A personal centroid is a
biometric description of one identifiable participant; the population profile already lives in
the repository under its own hash. What a reader needs to interpret a result (the mode, the
alpha, the artifact identity, the predictions, the similarities and margins) is all recorded,
and none of it is a face.

### What this does not establish

That runtime calibration should be switched on. The evidence is where it was: on EmojiHero
(n = 7) personal calibration lost to raw and hybrid's +0.044 did not reach significance
(p = 0.125); on the one headset session personal *beat* raw by +0.086 while hybrid moved the
wrong way, and that +0.086 is about 1.1 trials of macro-F1 resolution. Population-only is the
one comparison that cleared its floor, and it was negative in both datasets.

This is the instrument, built and verified. The decision still needs the multi-participant
pilot.

## Live vs acted comparison

```bash
python Analysis/live_vs_acted_study.py --json results/live_vs_acted.json
python Analysis/live_vs_acted_study.py --sampling decimated   # sensitivity check
python Analysis/test_live_vs_acted_study.py                   # 41 tests
```

Needs `numpy` and the `ActingRecordings` corpus, found the same way
`subject_variability_study.py` finds it (`--data PATH` or `$ACTING_RECORDINGS`). It does not
run the model: live predictions come from the headset's own logs and acted verdicts are read
out of `results/model_performance.json`.

It asks one question — whether the player the model works on has expression profiles nearer
the ten acted **training** participants than the player it fails on — by building a per-channel
peak profile for each person and emotion and placing it in the leave-one-out distribution of
those ten. Neutral is excluded throughout; the game never prompts it and the teacher pipeline
defines a usable Neutral frame differently.

The two live sessions from the same person are averaged with equal weight per session, never
by pooling frames, and the live subject never enters a centroid it is then scored against.
Both are enforced by the tests rather than left to care.

**The answer was no**, and [LIVE_VS_ACTED_REPORT.md](LIVE_VS_ACTED_REPORT.md) is that result:
across twelve subject-emotion cells, distance to the training centroid and model accuracy
correlate at Spearman +0.007. Read the report's limits before quoting any of it — there are
two live people.

## Window-level comparison

```bash
python Analysis/window_level_study.py --json results/window_level.json
python Analysis/window_level_study.py --channels expression55   # sensitivity check
python Analysis/test_window_level_study.py                      # 44 tests
```

Needs `numpy`, `onnxruntime`, and the acted corpus. Where the comparison above reduces a
ten-second recording to one peak profile, this one keeps the unit the model actually works in:
the individual `[5, 70]` window, flattened frame-major to 350 numbers, so *when* a channel
moved still matters.

It measures the input space and the model's seven-dimensional output space **separately**,
which is what lets it tell "these are different inputs" from "these are similar inputs the
model answers differently". Nothing is run without a passing replay-fidelity check.

Read [WINDOW_LEVEL_REPORT.md](WINDOW_LEVEL_REPORT.md) starting from §0, which is the control
that bounds the rest: in raw-window cosine, only **Happiness** is nearer itself than to some
other class across the acted training participants — and acted Fear is already nearest acted
Surprise, the same confusion the live players show.

The main use of the window level is that absolute distance turns out to carry almost nothing
while the neighbourhood carries a lot. One live trial at cosine 0.397 to its prompted class
scores 1.00 and another at 0.400 scores 0.00; the first has 90 of 90 nearest neighbours in the
right class and the second has 1 of 45.

## Within-person against between-person

```bash
python Analysis/live_pairwise_study.py --json results/live_pairwise.json
python Analysis/test_live_pairwise_study.py    # 21 tests
```

Needs only `numpy` and the pulled live sessions — no acted corpus, no ONNX. It puts three
numbers side by side for each emotion:

```
  A1 <-> A2    one person, two evenings
  A1 <-> B     that person's first evening against the other person
  A2 <-> B     that person's second evening against the other person
```

If the first beats the other two, the same person twice is more alike than two different
people are, and there is something stable for a per-person profile to be fitted to. Both
representations are reported — the peak profile and the 5×70 windows — because they can
disagree, and here they agree on the verdict while differing by a factor of two on the size.

**Identity is the larger effect in five of six emotions**, by +0.242 on average in window
space. **Fear is the exception in both representations**: one person's two evenings of Fear
are less alike than either is to the other person's. See
[LIVE_PAIRWISE_REPORT.md](LIVE_PAIRWISE_REPORT.md), including §3, which is the first quantity
in this project that tracks accuracy — and the reasons not to believe it yet.

## Using it from Python

The metrics live in `session_analysis.py` and are all plain dictionaries and lists, so you
can skip the printed report and work with them directly:

```python
import session_analysis as sa

files, _ = sa.autodiscover()               # or sa.discover(["some/path"])
report = sa.analyse(sa.load_session(files[0]))

print(report["accuracy"]["tick_accuracy"])
print(report["confusion"]["Fear"])          # what Fear gets mistaken for

# or pool every session into one
combined = sa.analyse(sa.pool([sa.load_session(p) for p in files]))
```

`sa.Session.rows` gives you the raw ticks, each with its score vector and its full
`frames[frame][au]` window, if you want to push a recorded session back through the ONNX
model without a headset.

## Notes

- The class list, window length, and feature count are read from the CSV header rather than
  hardcoded, so a model change on the Unity side will not quietly produce wrong numbers here.
  Pooling files captured with different schemas is refused rather than guessed at.
- The `correct` column is recomputed from `predicted_emotion == prompted_emotion` rather than
  trusted, and the report warns if the two ever disagree.
- Files are read as `utf-8-sig`, so a CSV that has been through Excel and come back with a
  byte-order mark still loads.
