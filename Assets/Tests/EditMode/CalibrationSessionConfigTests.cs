using NUnit.Framework;

namespace VcuSwimLab.Calibration.Tests
{
    public class CalibrationSessionConfigTests
    {
        [Test]
        public void The_config_captures_the_four_modes()
        {
            Assert.AreEqual(CalibrationMode.Raw,
                            CalibrationSessionConfig.Lock(false, false, false).Mode);
            Assert.AreEqual(CalibrationMode.Personal,
                            CalibrationSessionConfig.Lock(true, false, false).Mode);
            Assert.AreEqual(CalibrationMode.Population,
                            CalibrationSessionConfig.Lock(false, true, false).Mode);
            Assert.AreEqual(CalibrationMode.Hybrid,
                            CalibrationSessionConfig.Lock(true, true, false).Mode);
        }

        [Test]
        public void A_locked_config_is_a_snapshot_not_a_view()
        {
            // The inspector fields this stands in for are mutable while the game is playing.
            // What matters is that the object handed to the session holds values, so a later
            // change to the source produces a *different* object rather than mutating this
            // one -- there is no setter and no reference back to the component.
            bool personal = true;
            bool population = false;

            var locked = CalibrationSessionConfig.Lock(personal, population, false);
            Assert.AreEqual(CalibrationMode.Personal, locked.Mode);

            // The operator "ticks the second checkbox" mid-session.
            population = true;
            var wouldBeNow = CalibrationSessionConfig.Lock(personal, population, false);

            Assert.AreEqual(CalibrationMode.Personal, locked.Mode,
                            "the running session's mode must not follow the inspector");
            Assert.AreEqual(CalibrationMode.Hybrid, wouldBeNow.Mode);
            Assert.IsFalse(locked.PopulationEnabled);
        }

        [Test]
        public void Only_hybrid_carries_an_alpha()
        {
            Assert.IsTrue(double.IsNaN(CalibrationSessionConfig.Lock(false, false, false).Alpha));
            Assert.IsTrue(double.IsNaN(CalibrationSessionConfig.Lock(true, false, false).Alpha));
            Assert.IsTrue(double.IsNaN(CalibrationSessionConfig.Lock(false, true, false).Alpha));
            Assert.AreEqual(0.40, CalibrationSessionConfig.Lock(true, true, false).Alpha);
        }

        [Test]
        public void The_alpha_is_written_as_a_fixed_two_decimal_string()
        {
            Assert.AreEqual("0.40", CalibrationSessionConfig.Lock(true, true, false).AlphaText);
            Assert.AreEqual(string.Empty, CalibrationSessionConfig.Lock(true, false, false).AlphaText);
            Assert.AreEqual(string.Empty, CalibrationSessionConfig.Lock(false, false, false).AlphaText);
        }

        [Test]
        public void Only_the_population_modes_ask_for_the_artifact()
        {
            Assert.IsFalse(CalibrationSessionConfig.Lock(false, false, false).NeedsPopulationArtifact);
            Assert.IsFalse(CalibrationSessionConfig.Lock(true, false, false).NeedsPopulationArtifact);
            Assert.IsTrue(CalibrationSessionConfig.Lock(false, true, false).NeedsPopulationArtifact);
            Assert.IsTrue(CalibrationSessionConfig.Lock(true, true, false).NeedsPopulationArtifact);
        }

        [Test]
        public void The_debug_logging_flag_is_carried_so_the_recording_can_state_it()
        {
            Assert.IsFalse(CalibrationSessionConfig.Lock(true, false, false).DebugLoggingEnabled);
            Assert.IsTrue(CalibrationSessionConfig.Lock(true, false, true).DebugLoggingEnabled);
        }

        [Test]
        public void Both_toggles_default_to_off_which_is_raw()
        {
            // The defaults an unconfigured component serialises into a scene. A session set up
            // by someone who has not touched these checkboxes records exactly what the
            // protocol recorded before runtime calibration existed.
            var defaults = CalibrationSessionConfig.Lock(false, false, false);

            Assert.AreEqual(CalibrationMode.Raw, defaults.Mode);
            Assert.AreEqual("raw", defaults.ModeName);
            Assert.IsFalse(defaults.PersonalEnabled);
            Assert.IsFalse(defaults.PopulationEnabled);
            Assert.IsFalse(defaults.NeedsPopulationArtifact);
            Assert.IsFalse(defaults.DebugLoggingEnabled);
            Assert.AreEqual(string.Empty, defaults.AlphaText);
        }

        [Test]
        public void The_mode_name_is_the_on_disk_spelling()
        {
            Assert.AreEqual("raw", CalibrationSessionConfig.Lock(false, false, false).ModeName);
            Assert.AreEqual("personal", CalibrationSessionConfig.Lock(true, false, false).ModeName);
            Assert.AreEqual("population", CalibrationSessionConfig.Lock(false, true, false).ModeName);
            Assert.AreEqual("hybrid", CalibrationSessionConfig.Lock(true, true, false).ModeName);
        }
    }
}
