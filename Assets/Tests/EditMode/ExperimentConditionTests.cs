using System.Linq;
using System.Text.RegularExpressions;

using NUnit.Framework;

namespace VcuSwimLab.Calibration.Tests
{
    /// <summary>
    /// The naming of the four experiment conditions, and the reading of the Jev key file.
    /// </summary>
    /// <remarks>
    /// The condition itself is chosen by calling the library in ConditionSetup.Configure, which
    /// needs Unity and the headset. What can be checked here is everything that turns the
    /// library's answer (is the active backend Jev? is calibration on?) into the labels and
    /// file names the game writes down, because a mix-up there would put a session in the wrong
    /// condition's column without anything failing.
    /// </remarks>
    public class ExperimentConditionTests
    {
        // Condition number -> (uses Jev, uses calibration), exactly as the table in ConditionSetup.
        private static readonly (int number, bool jev, bool calibration, string id)[] Conditions =
        {
            (1, false, false, "sparsevr_nocal"),
            (2, false, true, "sparsevr_cal"),
            (3, true, false, "jev_nocal"),
            (4, true, true, "jev_cal"),
        };

        [Test]
        public void The_four_conditions_have_the_agreed_numbers_and_ids()
        {
            foreach (var c in Conditions)
            {
                Assert.AreEqual(c.number, ExperimentCondition.Number(c.jev, c.calibration),
                                $"number of {c.id}");
                Assert.AreEqual(c.id, ExperimentCondition.Id(c.jev, c.calibration),
                                $"id of condition {c.number}");
            }
        }

        [Test]
        public void Ids_and_labels_are_all_different()
        {
            var ids = Conditions.Select(c => ExperimentCondition.Id(c.jev, c.calibration)).ToList();
            var labels = Conditions.Select(c => ExperimentCondition.Label(c.jev, c.calibration)).ToList();

            Assert.AreEqual(4, ids.Distinct().Count());
            Assert.AreEqual(4, labels.Distinct().Count());
        }

        [Test]
        public void Labels_say_the_number_the_backend_and_the_calibration()
        {
            Assert.AreEqual("Condition 1: SparseVR, no calibration", ExperimentCondition.Label(false, false));
            Assert.AreEqual("Condition 2: SparseVR, calibration", ExperimentCondition.Label(false, true));
            Assert.AreEqual("Condition 3: Jev, no calibration", ExperimentCondition.Label(true, false));
            Assert.AreEqual("Condition 4: Jev, calibration", ExperimentCondition.Label(true, true));
        }

        [Test]
        public void The_backend_name_is_the_one_the_metadata_documents()
        {
            Assert.AreEqual("SparseVr", ExperimentCondition.BackendName(false));
            Assert.AreEqual("Jev", ExperimentCondition.BackendName(true));
        }

        [Test]
        public void A_backend_is_Jev_only_when_the_library_says_so()
        {
            // The library names a Jev backend "jev:<model>:p<prompt version>"; the SparseVR
            // backend is named after its model asset.
            Assert.IsTrue(ExperimentCondition.IsJevBackend("jev:jev-1.13.0:p1"));
            Assert.IsFalse(ExperimentCondition.IsJevBackend("sparseVR_teacher_model_acted_only_vol-aug-k1"));
            Assert.IsFalse(ExperimentCondition.IsJevBackend("my-jev-lookalike"));
            Assert.IsFalse(ExperimentCondition.IsJevBackend(string.Empty));
            Assert.IsFalse(ExperimentCondition.IsJevBackend(null), "no backend selected is not Jev");
        }

        [Test]
        public void Each_backend_saves_a_calibration_under_its_own_id()
        {
            string sparse = ExperimentCondition.CalibrationFileId("player1", usesJev: false);
            string jev = ExperimentCondition.CalibrationFileId("player1", usesJev: true);

            Assert.AreEqual("player1-sparsevr", sparse);
            Assert.AreEqual("player1-jev", jev);
            Assert.AreNotEqual(sparse, jev, "they would overwrite each other's calibration file");
        }

        [Test]
        public void A_calibration_id_is_still_a_valid_library_user_id()
        {
            // The library's CalibrationStore accepts only letters, digits, '-' and '_'.
            var allowed = new Regex(@"^[A-Za-z0-9_-]+\z");

            foreach (string player in new[] { "player1", "P-07", "study_2026" })
            {
                foreach (bool jev in new[] { false, true })
                {
                    Assert.IsTrue(allowed.IsMatch(ExperimentCondition.CalibrationFileId(player, jev)),
                                  $"{player} / jev={jev}");
                }
            }
        }

        // ------------------------------------------------------------------- the key file

        [Test]
        public void The_placeholder_the_key_file_ships_with_is_not_a_key()
        {
            string shipped =
                "# comment\n# more comment\n#\n" + ExperimentCondition.KeyPlaceholder + "\n";

            Assert.IsNull(ExperimentCondition.KeyFromFileText(shipped));
        }

        [Test]
        public void The_key_is_the_first_line_that_is_not_blank_or_a_comment()
        {
            const string key = "tsk_0123456789abcdef";
            string text = "# Typesafe key\n\n   \n# another\n  " + key + "  \nsecond-line-ignored\n";

            Assert.AreEqual(key, ExperimentCondition.KeyFromFileText(text));
        }

        [Test]
        public void Windows_line_endings_do_not_end_up_in_the_key()
        {
            Assert.AreEqual("tsk_abc", ExperimentCondition.KeyFromFileText("# c\r\ntsk_abc\r\n"));
        }

        [Test]
        public void A_missing_or_empty_file_has_no_key()
        {
            Assert.IsNull(ExperimentCondition.KeyFromFileText(null));
            Assert.IsNull(ExperimentCondition.KeyFromFileText(string.Empty));
            Assert.IsNull(ExperimentCondition.KeyFromFileText("   \n\n# only a comment\n"));
        }
    }
}
