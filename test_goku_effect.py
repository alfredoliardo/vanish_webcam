import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import numpy as np

from local_app import vanish_cam as app
from local_app.goku_effect import (
    GOKU_EFFECT_SECONDS, forehead_from_face, make_speed_lines, render_teleport,
    two_finger_salute,
)
from local_app.particles_effect import PARTICLES_SECONDS, ParticleBurst
from local_app.test_video_output import arguments


FACE = np.array([(440.0, 200.0), (520.0, 200.0), (480.0, 240.0), (480.0, 280.0)])


def salute_hand(tip=(480.0, 140.0), spread=0.0):
    """Upright hand: index and middle straight, ring and little finger folded."""
    points = np.array([
        (0, 0), (-12, -8), (-18, -18), (-20, -28), (-20, -36),
        (-15, -40), (-14, -60), (-13, -75), (-12 - spread, -88),
        (-5, -42), (-5, -62), (-5, -78), (-5 + spread, -92),
        (5, -40), (5, -55), (5, -45), (5, -35),
        (15, -36), (14, -48), (13, -40), (12, -32),
    ], dtype=float)
    return points + (np.array(tip) - points[[8, 12]].mean(axis=0))


class GokuPoseTests(unittest.TestCase):
    def setUp(self):
        self.forehead, self.size = forehead_from_face(FACE, 200.0)

    def test_forehead_is_above_the_eyes(self):
        np.testing.assert_allclose(self.forehead, (480.0, 136.0))

    def test_two_fingers_on_forehead(self):
        self.assertTrue(two_finger_salute(salute_hand(), self.forehead, self.size))

    def test_v_sign_or_far_hand_is_rejected(self):
        self.assertFalse(two_finger_salute(salute_hand(spread=30.0), self.forehead, self.size))
        self.assertFalse(two_finger_salute(salute_hand(tip=(480.0, 420.0)),
                                           self.forehead, self.size))

    def test_hold_needs_one_second_and_release_between_teleports(self):
        state = app.AppState(phase="ready")
        with patch("builtins.print"):
            self.assertFalse(app.update_goku_hold(state, True, 10.0))
            self.assertFalse(app.update_goku_hold(state, False, 10.1))  # Breve buco.
            self.assertTrue(app.update_goku_hold(state, True, 11.0))
            self.assertFalse(app.update_goku_hold(state, True, 12.5))  # Mano ancora su.
            self.assertFalse(app.update_goku_hold(state, False, 12.9))
            self.assertFalse(app.update_goku_hold(state, True, 13.0))
            self.assertTrue(app.update_goku_hold(state, True, 14.0))


class TeleportRenderTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((60, 80, 3), 30, np.uint8)
        self.frame = np.full((60, 80, 3), 200, np.uint8)
        self.mask = np.zeros((60, 80), np.float32)
        self.mask[10:50, 25:55] = 1.0
        self.lines = make_speed_lines(80)

    def test_person_vanishes_then_lines_fade(self):
        middle = render_teleport(self.frame, self.background, self.mask, 0.3, self.lines)
        self.assertTrue(np.any(middle != self.background))
        end = render_teleport(self.frame, self.background, self.mask,
                              GOKU_EFFECT_SECONDS, self.lines)
        np.testing.assert_array_equal(end, self.background)

    def test_person_reappears(self):
        start = render_teleport(self.frame, self.background, self.mask, 0.0, self.lines, True)
        np.testing.assert_array_equal(start, self.background)
        end = render_teleport(self.frame, self.background, self.mask,
                              GOKU_EFFECT_SECONDS, self.lines, True)
        np.testing.assert_array_equal(end[self.mask > 0], self.frame[self.mask > 0])

    def test_particles_leave_only_background(self):
        burst = ParticleBurst(self.frame, self.mask)
        self.assertGreater(burst.count, 0)
        np.testing.assert_array_equal(burst.render(self.background, 0.0)[30, 40], (200, 200, 200))
        self.assertTrue(np.any(burst.render(self.background, 1.0) != self.background))
        np.testing.assert_array_equal(burst.render(self.background, PARTICLES_SECONDS),
                                      self.background)


class GokuVideoLoopTests(unittest.TestCase):
    def test_teleport_out_and_back_in(self):
        frames = [np.full((40, 60, 3), 10 + 15 * index, np.uint8) for index in range(13)]
        capture = Mock()
        capture.read.side_effect = [(True, frame) for frame in frames]
        segmenter, recognizer = Mock(), Mock()
        recognizer.detect_salute.side_effect = [True, True, True, True, False, True, True]
        sound = Mock()
        virtual = Mock(width=60, height=40)
        preview = []
        times = (0, 0, 0.1, 4.1, 4.2, 4.7, 5.25, 5.6, 6.3, 6.4, 6.8, 7.0, 8.1, 8.5, 9.2)
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(mode="goku")))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, recognizer)))
            sounds = mocks.enter_context(patch.object(app, "VanishSound", return_value=sound))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=virtual))
            mocks.enter_context(patch.object(app, "person_mask", return_value=np.ones((40, 60), np.float32)))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=times))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: preview.append(frame.copy())))
            # V e R sono disattivati in modalita' Goku.
            keys = (0, 0, ord("v"), 0, 0, ord("r"), 0, 0, 0, 0, 0, 0, ord("q"))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=keys))
            mocks.enter_context(patch.object(app.cv2, "putText", side_effect=AssertionError("Overlay vietato")))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
        self.assertEqual(sounds.call_args.args, (app.TELEPORT_SOUND_PATH,))
        self.assertEqual(len(preview), 13)
        for sent, shown in zip(virtual.send.call_args_list, preview):
            np.testing.assert_array_equal(sent.args[0], shown)
        np.testing.assert_array_equal(preview[3], frames[3])  # Conto alla rovescia.
        self.assertTrue(np.any(preview[5] != frames[5]))  # Linee in corso.
        np.testing.assert_array_equal(preview[7], frames[1])  # Sparito: solo sfondo.
        self.assertTrue(np.any(preview[11] != frames[11]))  # Ricomparsa in corso.
        np.testing.assert_array_equal(preview[12], frames[12])  # Di nuovo visibile.
        self.assertEqual(recognizer.detect_salute.call_count, 7)
        self.assertEqual(sound.play.call_count, 2)
        sound.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
