# TrueEmotionVR — Getting Started & Task List

Welcome! This document is your roadmap. Work through it top to bottom. Each task is
self-contained, small enough to finish in one sitting or two, and ends with a commit.

Everything here can be done **on your laptop, in the Unity Editor, with no VR headset.**

---

## 1. What this project actually is

Our lab built a machine-learning model that tries to recognize which emotion a person is
expressing on their face. It works from *action units* — 70 numbers that the Meta Quest Pro
headset reports about the wearer's face (how raised the left eyebrow is, how open the jaw is,
and so on). The model takes those 70 numbers over a short window of time and outputs one of
seven emotions: **Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise**.

This repository is the *game* we use to test that model. It shows the player an emoji — say 😱 —
and asks them to act out that emotion with their face. While they're acting, the model makes a
prediction once per second for ten seconds, and the game counts how many of those ten seconds
the model agreed with the emoji it asked for. That count is the score.

> **Read this part carefully:** the model is not very accurate yet. A player can make a
> genuinely great sad face and still score 2/10. **That is a known problem, it is being worked
> on separately by the rest of the team, and it is not something you are responsible for or
> expected to fix.** If you see wrong predictions while testing, that's expected — it is not a
> bug you introduced. Your job is to make the *game* around the model clearer and more
> enjoyable to play.

The game code you'll be working on is genuinely tiny: **two C# scripts and one scene.** The
enormous `Assets/Samples/` folder is Meta's SDK sample content — you can ignore all of it.

| Path | What it is |
| --- | --- |
| `Assets/Scenes/Main.unity` | The entire game. There is only one scene. |
| `Assets/Scripts/ProgressGame.cs` | The whole game loop: emoji order, countdown, timer, scoring, results text. ~250 lines. |
| `Assets/Scripts/ProgressBar.cs` | A small progress-bar helper. ~55 lines. |
| `Assets/PredictorConfig.asset` | Points at the trained model file. **Don't edit this.** |

The model itself lives in a separate package (`com.vcu-swim-lab.vr-emotion-detection`) that
Unity downloads automatically. You don't need to open it.

---

## 2. Setup

### 2.1 Install Unity

1. Install **Unity Hub** from <https://unity.com/download>.
2. In Unity Hub go to **Installs → Install Editor → Archive** and install version
   **`6000.0.33f1`**. The version must match exactly — a different Unity version will silently
   upgrade project files and produce a huge, unreviewable diff the first time you commit.
3. When it asks which modules to add, tick **Android Build Support** (including *OpenJDK* and
   *Android SDK & NDK Tools*). You won't build for Android yourself, but some of the Meta
   packages won't compile without it.

### 2.2 Get the code

```bash
git clone https://github.com/vcu-swim-lab/TrueEmotionGameVR.git
cd TrueEmotionGameVR
git checkout -b emily_work
git push -u origin emily_work
```

Use your own name — `emily_work` is just an example. **This one branch is where all of your work
lives.** You don't need a new branch for each task; just keep committing to this one. Nothing you
do here can affect `main` until someone deliberately merges it, so you can experiment freely.

### 2.3 Open it

In Unity Hub: **Add → Add project from disk**, pick the folder, and open it.

**The first open will take a long time — 10 to 30 minutes is normal.** Unity is downloading
packages from GitHub and building its `Library/` folder from scratch. Don't cancel it, and
don't panic at the progress bar sitting still. Once it finishes, open
`Assets/Scenes/Main.unity` from the Project window.

If Unity shows errors about `com.vcu-swim-lab.vr-emotion-detection` failing to resolve, that
means it couldn't reach the package repository on GitHub — ask the team for access rather than
trying to work around it.

---

## 3. Ground rules

A few of these exist because of specific traps in this repo. Please don't skip them.

- **Commit after each task**, with a message saying what the task was. Small, frequent
  commits are much easier to review — and much easier to undo when something goes wrong.
- **Push to your branch regularly.** `git push` at the end of each work session. That's the
  backup, and it's how anyone else can see where you're at.
- **Never commit the `Library/` folder.** The `.gitignore` already excludes it. If you ever see
  `Library/` in `git status`, stop and ask.
- **Always run `git status` after Unity has been open, and commit the `.meta` files it creates.**
  Unity writes a `.meta` file next to every asset; it holds the ID that scenes use to find that
  asset. If you forget to commit one, the project breaks for everyone else but keeps working
  fine for you — which makes it very confusing to debug. *(There is already one missing in this
  repo. You'll fix it in Task 2.)*
- **Watch your `Main.unity` diffs.** Unity likes to write small changes into the scene file just
  because you clicked something. Before committing, run `git diff Assets/Scenes/Main.unity` and
  make sure everything in there is a change you meant to make.
- **Don't edit `Packages/manifest.json`, `Assets/Samples/`, `Assets/Oculus/`, or
  `Assets/PredictorConfig.asset`** without asking first.
- When you finish a task, say so — with *what changed, how to test it, and anything you
  weren't sure about.* That last part is the useful one; don't skip it because it feels like
  admitting you don't know something. It's the fastest way to get unstuck, and nobody expects
  you to have figured everything out alone. Ask questions early: one that takes someone two
  minutes to answer beats a day spent guessing.

---

## 4. Task 1 — Make the game playable on a laptop

**Size:** Medium · **Do this one first — every other task depends on it.**

### Goal
Be able to press Play in the Unity Editor, with no headset attached, and play a full round of
the game from countdown to results screen.

### Why it matters
Right now you can't. The game asks the headset for face data (there is none), asks a controller
for a button press (there is none), and crashes before it gets that far anyway. Until this is
fixed you have no way to see any of your own work. This task is the key to the whole list.

### Files
`Assets/Scripts/ProgressGame.cs`, `Assets/Scenes/Main.unity`

### Steps

**a) Add a "fake predictions" switch.**

Near the top of the class, add:

```csharp
[Header("Desktop Testing")]
[SerializeField] private bool useMockPredictions = false;
[SerializeField, Range(0f, 1f)] private float mockCorrectChance = 0.5f;
```

Then find `PredictEmotionAsync()` near the bottom of the file. There's a big commented-out block
in it — someone started this and never finished. Change the method to take the emotion the game
is currently asking for:

```csharp
private async Awaitable<Dictionary<Emotion, float>> PredictEmotionAsync(Emotion target)
```

and, when `useMockPredictions` is true, return a made-up answer instead of calling the real
model. Use `mockCorrectChance` to decide whether the fake answer is the *right* emotion or a
random wrong one. (Don't just return a uniformly random emotion — a mock that's right about half
the time feels like the real game and makes the rest of your work much easier to judge.)

Update the one place that calls this method (inside `RunPredictionCoroutine`) to pass `emotion`.

**b) Make the keyboard work.**

Find the line near the results screen that waits for a controller button:

```csharp
while (!OVRInput.GetDown(OVRInput.Button.Any))
```

Without a headset this never becomes true, so the game freezes on the score screen forever. Add
the spacebar as an alternative — but **leave the `OVRInput` check in place** so it still works on
the headset.

> ⚠️ This project uses Unity's **new Input System**. The old `Input.GetKeyDown(KeyCode.Space)`
> will throw an exception, not just fail. You need `using UnityEngine.InputSystem;` at the top
> and then `Keyboard.current != null && Keyboard.current.spaceKey.wasPressedThisFrame`.

Pull this out into a small helper method — you'll need the same check again in Task 3.

**c) Delete two lines that shouldn't be there.**

- Inside `RunGame()`, the last statement in the loop calls `RunGame()` again. But `RunGame()`
  already contains a `while (true)`, so this starts a *second* copy of the loop on top of the
  first one every time you restart, forever. Delete that call — the `while` loop handles it.
- A few lines above the emoji display there's `text.text = $"(n + 1) / 6\n";`. Someone forgot the
  `{}` braces, so it literally prints the characters `(n + 1) / 6`. It's immediately overwritten
  on the next line anyway. Delete it.

**d) Stop hardcoding "10".**

There's a block that looks like this:

```csharp
#if UNITY_EDITOR
    int times = 2;
#else
    int times = 10;
#endif
```

Replace it with two serialized fields — `secondsPerEmotion` (default 10) and
`editorSecondsPerEmotion` (default 3) — so you can tune the round length from the Inspector
without recompiling.

Then find the three places that have `10` typed directly into the on-screen text — the
`"You got 10s to act..."` line, `"Score: {this_score}/10"`, and the results screen's `/10` — and
make them read from the field instead. Right now if anyone changes the timer, the text on screen
starts lying to the player.

**Keep the real device at 10 seconds.** That's the study's actual measurement window; only the
Editor value should be short.

**e) Make the canvas visible.**

Press Play and you'll probably see... nothing. The game's UI lives on a **World Space** canvas
with a component called `OVROverlayCanvas`, which hands the canvas off to the Quest's display
system. With no Quest attached, there's nothing to hand it to.

Fix for day-to-day work: select the **`Canvas`** object in the Hierarchy and **untick the
checkbox next to `OVR Overlay Canvas`** in the Inspector. Now it renders normally in the Game
view.

> ⚠️ **Re-tick that box before you commit.** If you commit `Main.unity` with it off, you break
> the game on the actual headset. Add it to your pre-commit routine along with
> `git diff Assets/Scenes/Main.unity`.

(Alternative if you're just peeking: during Play mode, click the **Scene** tab instead of the
Game tab. World-space canvases always show up there.)

### Done when
- You press Play with no headset and see: instructions → 3-2-1 countdown → all six emoji, one at
  a time → a results screen.
- Pressing **Space** on the results screen starts a new round.
- The Console has no red errors.
- `Use Mock Predictions` is committed as **unticked**, and `OVR Overlay Canvas` is committed as
  **ticked**. Both should only ever be turned on locally while you're testing.

---

## 5. Task 2 — Build the progress bar and fix its crash

**Size:** Small–Medium

### Goal
Get the progress bar actually appearing on screen and filling up as the player works through the
six emotions — smoothly, not in jumps.

### Why it matters
`ProgressBar.cs` exists in the repo but there is **no progress bar object in the scene**, so the
script never finds one. Worse, `ProgressGame` uses it before checking whether it found it, which
throws a `NullReferenceException` on the very first emotion and kills the entire game loop. The
loop is `async void`, which means C# swallows the error — the game just quietly stops instead of
showing you a crash. This is why the game appears to hang.

### Files
`Assets/Scripts/ProgressBar.cs`, `Assets/Scripts/ProgressGame.cs`, `Assets/Scenes/Main.unity`

### Steps

**a) Build the bar in the scene.** Under the `Canvas` object, create:

- an empty GameObject called `ProgressBar`, with the `ProgressBar` script on it;
- inside it, a UI **Image** called `ProgressBarBack` — this is the dark track;
- inside it, a second UI **Image** called `ProgressFill` — this is the coloured part. In the
  Inspector set **Image Type → Filled**, **Fill Method → Horizontal**, **Fill Origin → Left**.

Drag `ProgressBarBack` and `ProgressFill` into the matching slots on the `ProgressBar` component.

**b) Wire it up properly.** Select the `Canvas` object (that's where `ProgressGame` lives) and
drag your new `ProgressBar` object into `ProgressGame`'s **Progress Bar** field. This means the
script no longer has to go hunting for it at runtime, which is both faster and much harder to
break.

**c) Fix the crash.** In `ProgressGame.cs`, `progressBar.SetMaximum(max);` currently sits *above*
the `if (progressBar == null)` check that's supposed to protect it. Move the call so it happens
after — and inside — the null check.

**d) Clean up `ProgressBar.cs`:**

- `GetCurrentFill()` has a `Debug.Log` in it, and `Update()` calls that method **every single
  frame**. That's ~90 log lines per second drowning your Console. Delete the log.
- `[ExecuteInEditMode()]` is written above `Start()`. That attribute only does anything when it's
  above the *class*, so where it currently sits it does nothing at all. Delete it.

**e) Make it glide.** Right now `fillAmount` snaps straight to its new value. Store the value you
*want* and, in `Update()`, move the actual `fillAmount` toward it a little each frame using
`Mathf.Lerp` or `Mathf.MoveTowards`. This is your first bit of animation code and you'll reuse
the same idea in Tasks 4 and 7 — worth getting comfortable with here.

**f) Commit the missing `.meta` file.** Run `git status`. You should see
`Assets/Scripts/ProgressBar.cs.meta` as untracked — **it is missing from the repository right
now**, which is a real bug affecting everyone. Commit it along with the rest of this task and
mention it — it's a fix for everybody, not just you.

### Done when
The bar shows on screen, advances 1/6 → 6/6 over the course of a round, animates rather than
snaps, hides itself between rounds, and the Console is quiet.

---

## 6. Task 3 — Title screen and how-to-play

**Size:** Medium

### Goal
Start the player on a title screen that explains the game, and only begin when they say they're
ready.

### Why it matters
Today, `RunGame()` fires from `Start()`, so the moment the scene loads the player is already two
seconds into a countdown. In a study session that means the participant is being measured before
they've understood the task — and the very first emotion always gets the worst score for reasons
that have nothing to do with their face.

### Files
`Assets/Scenes/Main.unity`, `Assets/Scripts/ProgressGame.cs`

### Steps

1. In the scene, create an empty `GamePanel` object under `Canvas` and move the existing
   `Instruction` text inside it.
2. Create a sibling `TitlePanel` containing: the game's name, two or three short lines explaining
   what the player is about to be asked to do, and a prompt line —
   *"Press Space, or any controller button, to start"*.
3. In `ProgressGame`, don't call `RunGame()` from `Start()` anymore. Instead show `TitlePanel`,
   wait for the input, then hide it, show `GamePanel`, and start the game.
4. Reuse the input helper you wrote in Task 1 rather than writing a second copy of the same
   check.
5. After the results screen, go back to the title rather than straight into a new round.

Keep the wording plain and short. Participants read it once, standing up, wearing a headset.

### Done when
Play → title → press to start → full round → results → press → back to the title.

---

## 7. Task 4 — Make the ten seconds visible

**Size:** Medium

### Goal
Show the player how much of their ten seconds is left.

### Why it matters
During the acting window the screen just shows a static emoji. The player has no idea whether
they have nine seconds left or one, so they either give up early or hold a face long past the
point where it's being measured. A visible timer is the single biggest clarity win in the game.

### Files
`Assets/Scenes/Main.unity`, `Assets/Scripts/ProgressGame.cs`

### Steps

1. Add a UI **Image** ring around the emoji. Set **Image Type → Filled** and
   **Fill Method → Radial 360**. Use a donut/ring sprite, or a thick circle outline.
2. Drain it from full to empty over `secondsPerEmotion` (the field from Task 1 — don't hardcode
   the number again).
3. Shift its colour as time runs out — green → amber → red is fine, or just fade it.
4. Reset it at the start of every emotion.

**Keep it visually distinct from the Task 2 progress bar.** Those two show completely different
things — "how far through the six emotions you are" versus "how long you have on *this* one" —
and if they look alike the player will read them as the same thing. Different shape and different
colour.

**Optional polish while you're here:** the 3-2-1 countdown is currently a hard text swap. Scaling
each digit down and fading it out as it changes takes about ten lines and makes the whole opening
feel considered instead of unfinished.

### Done when
The ring empties over exactly the configured duration, resets each round, and doesn't drift out
of sync when you change `secondsPerEmotion` in the Inspector.

---

## 8. Task 5 — Sound

**Size:** Small–Medium

### Goal
Give the game audio. It currently has none at all.

### Why it matters
There is exactly one audio component in the entire project — an `AudioListener` — and zero sounds.
The game is completely silent. Sound is the cheapest way to make a game feel finished, and in VR
it also carries information: a countdown tick tells the player what's happening even while
they're concentrating on their face rather than reading the screen.

### Files
New `Assets/Audio/` folder, new `Assets/Scripts/GameAudio.cs`, `Assets/Scripts/ProgressGame.cs`

### Steps

1. Add an `AudioSource` to the scene and a small `GameAudio` script with clearly-named methods —
   `PlayTick()`, `PlayRoundStart()`, `PlayScoreReveal()`, `PlayFinale()` — each doing a
   `PlayOneShot`. Keeping the sound logic out of `ProgressGame` keeps that file readable.
2. Call them from the matching moments in the game loop.
3. Find clips that are **CC0 / public domain**. Good sources: [Kenney's UI and interface
   packs](https://kenney.nl/assets) (all CC0), or [freesound.org](https://freesound.org) with the
   licence filter set to CC0.
4. **Write down where each clip came from and its licence** — in the commit message, or in a
   short `Assets/Audio/CREDITS.md`. This is a
   research project — we have to be able to say where every asset came from.
5. Keep the files small. Short `.wav` or `.ogg` files, nothing minutes long.

### Done when
Each beat of the game has its own sound, nothing overlaps unpleasantly, and the results screen
isn't still playing ticks.

---

## 9. Task 6 — Make the feedback kinder

**Size:** Medium

### Goal
Replace the bare `Score: 2/10` with feedback that tells the player something useful and doesn't
feel like a failure.

### Why it matters
This is the task that most directly addresses why the game isn't fun right now. The score counts
how many of the ten one-second checks the model ranked your target emotion first. Because the
model isn't very accurate, a player who does everything right routinely sees `2/10` and concludes
they're bad at expressing emotion. They aren't — the model is.

### Files
`Assets/Scripts/ProgressGame.cs`

### Steps

> **Important constraint:** do **not** change how `this_score` is calculated. That number is the
> study's actual measurement and the rest of the team depends on it staying comparable. Build
> your new feedback *on top of* it.

1. **Use the confidence value.** Look at `RunPredictionCoroutine` — it pulls a `confidence` float
   out of the prediction and then never uses it. Average it across the ten ticks and show it as a
   small "expression intensity" bar. This is the first time that number will have done anything.
2. **Track what the model actually saw.** `PredictEmotionAsync` returns a dictionary whose single
   key is the model's *actual* guess — not necessarily the emotion you asked for. Tally those
   across the ten ticks so you know the model's most common guess for the round.
3. **Rewrite the per-round screen** to show:
   - a **0–3 star rating** derived from `this_score` (pick thresholds that feel fair — you'll
     need to playtest with `mockCorrectChance` set low to check the bad case);
   - a short encouraging line that varies with the rating;
   - when the score is low, a hint: *"The camera mostly read that as **Fear** — try
     exaggerating the mouth."*

That last one is the important one. It turns "you got 2/10 and we won't say why" into
"here's what the system saw" — which is honest, actionable, and much less discouraging.

### Done when
Set `mockCorrectChance` to `0.2`, play a full round, and read the screens. If they still feel
like a report card, keep tuning.

---

## 10. Task 7 — A real results screen

**Size:** Medium–Large

### Goal
Replace the wall of text at the end with a laid-out summary panel.

### Why it matters
The results screen is currently built by gluing strings together into a single text label —
`"Thanks for playing! Scores:\n" + string.Join(...)`. It's the last thing a participant sees and
it looks like console output.

### Files
`Assets/Scenes/Main.unity`, `Assets/Scripts/ProgressGame.cs`

### Steps

1. Build a `ResultsPanel` with one row per emotion: the emoji, the emotion name, a small fill bar
   for the score, and the star rating from Task 6.
2. Call out the emotion they did best at.
3. Add a real on-screen **Restart button**. The scene already has a `GraphicRaycaster` and an
   `EventSystem`, so a standard UI Button works with the mouse in the Editor and with a
   controller pointer in VR. Keep the keyboard and `OVRInput` paths working too — three ways in
   is fine, and each one covers a case the others don't.
4. Fade the rows in one after another with a short delay between them, reusing the animation
   approach from Task 2. Six rows appearing in sequence reads much better than six rows appearing
   at once.

### Done when
The results are a designed panel rather than a paragraph, and restart works by mouse click,
spacebar, and controller button.

---

## 11. Task 8 — Practice round *(stretch)*

**Size:** Medium

### Goal
An untimed, unscored warm-up before the six real emotions begin.

### Why it matters
Even with a title screen, the first real emotion is where the participant figures out the format
— what the countdown means, how long they have to hold a face, what the screen does. Giving them
a throwaway round to learn on means the six measured rounds are all measured under the same
conditions.

### Steps
Show one emotion with no timer and no score, wait for the player to say they're ready, then start
the real sequence. Put it behind a `[SerializeField] private bool showPracticeRound = true;` so
it can be switched off for participants who've done the study before.

Keep it purely additive — the six real rounds and their scoring must be untouched.

---

## 12. Task 9 — Write it down

**Size:** Small

### Goal
Update `README.md` so the next person doesn't have to rediscover what you learned.

### Steps

1. Add a **"Running without a headset"** section covering: the `Use Mock Predictions` toggle,
   unticking `OVR Overlay Canvas` (and re-ticking it before committing), Space to restart, and
   the `secondsPerEmotion` / `editorSecondsPerEmotion` fields.
2. Add anything that cost you more than twenty minutes to figure out. That's the real test of
   whether a line belongs in the README.
3. **Fix the build settings.** Open **File → Build Settings** and click **Add Open Scenes** with
   `Main.unity` open. The build scene list is currently empty, which means `File > Build and Run`
   today produces a build containing *no scene at all*. One click, and it's been broken for a
   while.

### Done when
Someone who has never seen this project can follow the README and get the game running on a
laptop.

---

## 13. Order and dependencies

```
Task 1 (laptop mode)  ─┬─►  Task 2 (progress bar)  ──►  Task 4 (countdown ring)
                       │              │
                       ├─►  Task 3 (title screen) ─────►  Task 7 (results)  ──►  Task 8 (practice)
                       │                                        ▲
                       ├─►  Task 5 (audio)                      │
                       └─►  Task 6 (feedback)  ─────────────────┘

                            Task 9 (docs) — any time after Task 1
```

Task 1 must come first. After that, Tasks 5 and 6 don't depend on 2, 3 or 4 — so if you get stuck
on the scene-editing tasks, switch to one of those and come back.

---

## 14. Troubleshooting

**Play mode shows a blank Game view.**
The `OVR Overlay Canvas` component on `Canvas` is still ticked. Untick it. (See Task 1e.)

**The game shows the first countdown and then just stops, with no error.**
That's the `ProgressBar` null reference. The game loop is `async void`, so C# swallows the
exception instead of reporting it — the loop dies silently. Task 2 fixes it.

**`NullReferenceException` pointing at `ProgressGame`.**
Something in the Inspector isn't wired up. Select the `Canvas` object and check every field on
`ProgressGame` has an object in it.

**`Input.GetKeyDown` throws an `InvalidOperationException`.**
This project uses the new Input System exclusively. Use `Keyboard.current` instead. (Task 1b.)

**The Console is flooded with "Fill amount is 0.16666".**
`ProgressBar.GetCurrentFill()` logs every frame. Delete the log — Task 2d.

**Predictions look random / always the same emotion.**
Expected without a headset: with no Quest Pro there's no face data, so the model is being fed
zeros. Turn on `Use Mock Predictions`. Also see the note in section 1 — real-headset accuracy is
someone else's problem, not yours.

**Unity changed hundreds of files after I opened the project.**
You're on the wrong Unity version. Discard the changes (`git checkout .`), install
`6000.0.33f1`, and open it with that.

**A colleague pulled my branch and the scene is broken, but it works for me.**
You forgot to commit a `.meta` file. Run `git status` and commit anything untracked under
`Assets/`.

---
