using NUnit.Framework;

namespace VcuSwimLab.Calibration.Tests
{
    public class CalibrationSessionConfigTests
    {
        [Test]
        public void The_config_captures_raw_and_movement()
        {
            Assert.IsFalse(CalibrationSessionConfig.Lock(false, false).MovementEnabled);
            Assert.IsTrue(CalibrationSessionConfig.Lock(true, false).MovementEnabled);
        }

        [Test]
        public void The_mode_name_is_the_on_disk_spelling()
        {
            // Matched by the Python analysis; a rename here would silently split the corpus.
            Assert.AreEqual("raw", CalibrationSessionConfig.Lock(false, false).ModeName);
            Assert.AreEqual("movement", CalibrationSessionConfig.Lock(true, false).ModeName);
        }

        [Test]
        public void A_locked_config_is_a_snapshot_not_a_view()
        {
            // The inspector fields this stands in for are mutable while the game is playing.
            // What matters is that the object handed to the session holds values, so a later
            // change to the source produces a *different* object rather than mutating this one.
            bool movement = true;
            var locked = CalibrationSessionConfig.Lock(movement, false);

            movement = false;

            Assert.IsTrue(locked.MovementEnabled);
            Assert.AreEqual("movement", locked.ModeName);
        }

        [Test]
        public void The_debug_logging_flag_is_carried_so_the_recording_can_state_it()
        {
            Assert.IsTrue(CalibrationSessionConfig.Lock(false, true).DebugLoggingEnabled);
            Assert.IsFalse(CalibrationSessionConfig.Lock(false, false).DebugLoggingEnabled);
        }
    }
}
