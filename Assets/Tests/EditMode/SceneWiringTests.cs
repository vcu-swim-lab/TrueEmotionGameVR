using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

using NUnit.Framework;

namespace VcuSwimLab.Calibration.Tests
{
    /// <summary>
    /// Checks that the scene actually carries the components the runtime assumes.
    /// </summary>
    /// <remarks>
    /// `[RequireComponent]` is not a guarantee about a scene. It adds the required component
    /// when the requiring one is *added* in the editor, and does nothing at all to a GameObject
    /// that was serialised before the attribute existed. A scene saved earlier therefore opens
    /// with the component missing and no warning anywhere, and the first symptom is a
    /// NullReferenceException partway through a session -- in front of a participant. That is
    /// precisely what happened here, and it was caught by a person reading the scene file
    /// rather than by anything that ran.
    ///
    /// So this reads the scene YAML directly, as text, and asserts the invariant. Parsing the
    /// file rather than loading it through `EditorSceneManager` is deliberate:
    ///
    ///   - it needs no reference to Assembly-CSharp, which an assembly definition cannot have
    ///     in any case, so the test can live beside the rest of the calibration tests;
    ///   - it runs under `dotnet test` as well as in the Unity Editor, so the check is part of
    ///     the same sweep as the Python parity tests rather than something that only happens
    ///     when someone opens Unity;
    ///   - and it asserts what is *on disk*, which is what gets built and deployed, rather than
    ///     what some editor session might have reconstructed in memory.
    /// </remarks>
    public class SceneWiringTests
    {
        private const string ScenePath = "Assets/Scenes/Main.unity";
        private const string ProtocolRunnerScript = "Assets/Scripts/ProtocolRunner.cs";

        // NewFaceAuModel lives in the VREmotionDetection package, whose path differs between a
        // local checkout and a git install, so its script is identified by its fixed GUID.
        private const string NewFaceAuModelGuid = "1e701289fcb9a714db0f5f067a0c54c1";

        // Removed game-side components; nothing in the scene may still point at them.
        private const string RemovedCalibrationSettingsGuid = "9b1f664d04035ce4dabcfa384cc6eff6";
        private const string RemovedProgressGameGuid = "8289b5e9f362b36439ae8d269a35a379";

        /// <summary>One `--- !u!114 &id MonoBehaviour` block of a scene file.</summary>
        private sealed class SceneComponent
        {
            public string FileId;
            public string GameObjectId;
            public string ScriptGuid;
            public readonly Dictionary<string, string> Fields = new(StringComparer.Ordinal);
        }

        // `--- !u!114 &964688081` : a MonoBehaviour, with its file id.
        private static readonly Regex BlockHeader =
            new(@"^--- !u!(?<class>\d+) &(?<id>\d+)", RegexOptions.Compiled);

        private static readonly Regex GameObjectRef =
            new(@"^\s+m_GameObject: \{fileID: (?<id>-?\d+)\}", RegexOptions.Compiled);

        private static readonly Regex ScriptRef =
            new(@"^\s+m_Script: \{fileID: -?\d+, guid: (?<guid>[0-9a-f]+)", RegexOptions.Compiled);

        private static readonly Regex Field =
            new(@"^  (?<key>\w+): (?<value>.*)$", RegexOptions.Compiled);

        private static string Root => TestPaths.RepositoryRoot;

        /// <summary>The GUID Unity assigns a script, read from its `.meta` sidecar.</summary>
        private static string ScriptGuid(string relativePath)
        {
            string meta = Path.Combine(Root, relativePath.Replace('/', Path.DirectorySeparatorChar))
                          + ".meta";

            Assert.IsTrue(File.Exists(meta), $"no .meta beside {relativePath}");

            foreach (string line in File.ReadLines(meta))
            {
                if (line.StartsWith("guid: ", StringComparison.Ordinal))
                {
                    return line.Substring("guid: ".Length).Trim();
                }
            }

            throw new AssertionException($"{meta} has no guid line");
        }

        private static List<SceneComponent> MonoBehaviours()
        {
            string path = Path.Combine(Root, ScenePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.IsTrue(File.Exists(path), $"no scene at {path}");

            var components = new List<SceneComponent>();
            SceneComponent current = null;

            foreach (string line in File.ReadLines(path))
            {
                var header = BlockHeader.Match(line);
                if (header.Success)
                {
                    // 114 is MonoBehaviour. Every other class id starts a block this test has
                    // no interest in, and `current` going null is what stops its fields being
                    // attributed to the previous component.
                    current = header.Groups["class"].Value == "114"
                        ? new SceneComponent { FileId = header.Groups["id"].Value }
                        : null;

                    if (current != null) components.Add(current);
                    continue;
                }

                if (current == null) continue;

                var gameObject = GameObjectRef.Match(line);
                if (gameObject.Success)
                {
                    current.GameObjectId = gameObject.Groups["id"].Value;
                    continue;
                }

                var script = ScriptRef.Match(line);
                if (script.Success)
                {
                    current.ScriptGuid = script.Groups["guid"].Value;
                    continue;
                }

                var field = Field.Match(line);
                if (field.Success)
                {
                    current.Fields[field.Groups["key"].Value] = field.Groups["value"].Value.Trim();
                }
            }

            return components;
        }

        private static List<SceneComponent> With(string guid)
            => MonoBehaviours().Where(c => c.ScriptGuid == guid).ToList();

        // ------------------------------------------------------------------- the invariant

        [Test]
        public void The_scene_is_text_serialised_so_this_test_can_read_it()
        {
            string settings = Path.Combine(Root, "ProjectSettings", "EditorSettings.asset");
            Assert.IsTrue(File.Exists(settings), settings);

            // 2 is ForceText. Under ForceBinary the scene would be unreadable here and this
            // whole file would silently stop checking anything, so the mode is pinned.
            StringAssert.Contains("m_SerializationMode: 2", File.ReadAllText(settings),
                                  "the scene must stay text-serialised");
        }

        [Test]
        public void The_scene_has_a_protocol_runner()
        {
            var runners = With(ScriptGuid(ProtocolRunnerScript));

            Assert.AreEqual(1, runners.Count,
                            $"expected exactly one ProtocolRunner in {ScenePath}");
            Assert.IsNotNull(runners[0].GameObjectId, "the ProtocolRunner has no host GameObject");
        }

        [Test]
        public void The_scene_does_not_choose_the_condition()
        {
            var runner = With(ScriptGuid(ProtocolRunnerScript)).Single();

            // The condition is ConditionSetup.Selected, in code. A serialized `condition` on the
            // ProtocolRunner would be stored here, win over the code, and survive edits to
            // ConditionSetup.cs -- the next build would quietly run an older condition. If a
            // field with this name comes back, it is a second source of truth: remove it.
            Assert.IsFalse(runner.Fields.ContainsKey("condition"),
                           "the ProtocolRunner in the scene stores a 'condition'; the condition "
                           + "is chosen only by ConditionSetup.Selected, so this value would "
                           + "override it. Delete the field instead of setting it");
        }

        [Test]
        public void The_scene_s_predictor_has_calibration_off()
        {
            var predictors = With(NewFaceAuModelGuid);
            Assert.AreEqual(1, predictors.Count, $"expected exactly one NewFaceAuModel in {ScenePath}");

            // Calibration is off by default: a build made from the committed scene runs the
            // model's own predictions until someone ticks "Use Calibration".
            Assert.IsTrue(predictors[0].Fields.ContainsKey("useCalibration"),
                          "the scene does not store the Use Calibration checkbox");
            Assert.AreEqual("0", predictors[0].Fields["useCalibration"],
                            "Use Calibration is ticked in the committed scene");
        }

        [Test]
        public void No_component_points_at_the_removed_calibration_settings()
        {
            Assert.IsEmpty(With(RemovedCalibrationSettingsGuid),
                           "the scene still holds RuntimeCalibrationSettings, which no longer "
                           + "exists; Unity would show it as a missing script");
        }

        [Test]
        public void No_component_points_at_the_removed_emotion_game()
        {
            Assert.IsEmpty(With(RemovedProgressGameGuid),
                           "the scene still holds ProgressGame, the old emotion game, which no "
                           + "longer exists; Unity would show it as a missing script");
        }
    }
}
