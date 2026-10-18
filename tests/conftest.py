"""Shared test setup.

The game module plays audio and loads OpenCV/MediaPipe at import time, so we
force SDL's dummy audio driver (no sound card needed, e.g. on CI) and make
the project root importable before any test imports it.
"""

import os
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import OoE_CV as game  # noqa: E402


# ── Fake MediaPipe hands ─────────────────────────────────────────────────────

class LM:
    """Stand-in for a MediaPipe NormalizedLandmark."""

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


# x position of each finger column (index, middle, ring, pinky)
_FINGER_X = {5: 0.44, 9: 0.50, 13: 0.56, 17: 0.62}


def make_hand(thumb=0, index=0, middle=0, ring=0, pinky=0, dx=0.0, dy=0.0, scale=1.0):
    """Build 21 landmarks for an upright hand with the given fingers extended.

    `dx`/`dy` shift the hand around the frame and `scale` makes it bigger or
    smaller (closer to / farther from the camera).
    """
    def p(x, y):
        # scale around the wrist, then shift
        return LM(0.5 + (x - 0.5) * scale + dx, 0.8 + (y - 0.8) * scale + dy)

    lms = [p(0.5, 0.8) for _ in range(21)]
    lms[0] = p(0.50, 0.80)                                   # wrist
    lms[1], lms[2], lms[3] = p(0.45, 0.76), p(0.42, 0.72), p(0.41, 0.68)
    lms[4] = p(0.25, 0.65) if thumb else p(0.42, 0.62)      # thumb tip

    for mcp, up in ((5, index), (9, middle), (13, ring), (17, pinky)):
        x = _FINGER_X[mcp]
        lms[mcp] = p(x, 0.60)                                # knuckle
        lms[mcp + 1] = p(x, 0.50)                            # middle joint
        lms[mcp + 2] = p(x, 0.42 if up else 0.58)
        lms[mcp + 3] = p(x, 0.35 if up else 0.65)            # tip
    return lms


GESTURES = {
    1: (0, 1, 0, 0, 0), 2: (0, 1, 1, 0, 0), 3: (0, 1, 1, 1, 0),
    4: (0, 1, 1, 1, 1), 5: (1, 1, 1, 1, 1), 6: (1, 0, 0, 0, 0),
    7: (1, 1, 0, 0, 0), 8: (1, 1, 1, 0, 0), 9: (1, 1, 1, 1, 0),
}
FIST = (0, 0, 0, 0, 0)


def result(*hands):
    """Wrap hands the way HandLandmarker.detect() returns them."""
    return SimpleNamespace(hand_landmarks=list(hands))


# ── Controllable clock ───────────────────────────────────────────────────────

class Clock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(game.time, "time", c)
    return c


@pytest.fixture
def fresh_game():
    g = game.new_game()
    g.update({'mode': '1 to 6', 'run_range': range(1, 7),
              'user_batting': True, 'batting_first': 'user'})
    return g
