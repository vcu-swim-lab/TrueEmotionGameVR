# Jev (temporary in-game copy)

`JevClient`, `JevFaceDescriber`, `JevPrompt`, `JevRequestBuilder`, `JevResponseParser` and
`JevSettings` are byte-for-byte copies of `VREmotionDetection/Runtime/Jev` (the library's
uncommitted `jev_branch`). They are here only so the Jev backend can be tested before the library
publishes it. Do not edit them here; change the library's copy instead.

`JevScorer` is the one game-side file. It calls the copied `JevClient` on the same five-frame
window the local model scored, and returns Jev's seven scores next to the local prediction.
It also gives Jev the player's calibration, as the library does. The library keeps its
calibrator private, so `JevScorer` rebuilds the profile: from the same samples in the study
protocol (`StartProtocolCalibration` ... `FinishProtocolCalibration`), and from the file the
library saved in the game (`LoadProfile`).

## When the library publishes Jev

The library defines the same types in the same namespace, so the copies must go in the same
change that bumps the package:

1. Bump the `com.vcu-swim-lab.vr-emotion-detection` hash in `Packages/packages-lock.json`.
2. Delete the six copied `.cs` files (and their `.meta`) here. Keep `JevScorer.cs`.
3. Change `JevScorer.Score` to set `NewFaceAuModel.Backend = PredictionBackend.Jev`, call
   `PredictDetailed`, and read `JevError`. The profile code in `JevScorer` (the `*Protocol*`
   methods, `LoadProfile`, `ClearProfile`) and the `scorer.*Profile*`/`*Protocol*` calls beside the
   predictor's calibration calls go: the library passes its own profile. The other call sites in
   `ProgressGame` and `ProtocolRunner` and the logging in `SessionLogger` stay as they are.
4. In the library, make `AddCalibrationSample` score the window with the local model when the
   backend is Jev. Without that, `ProtocolRunner`'s calibration trials would build the calibration
   from Jev's scores instead of the local model's.
5. Drop the `com.unity.nuget.newtonsoft-json` line from `Packages/manifest.json`; the library
   declares it.
