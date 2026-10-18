"""The CPU opponent (Brain)."""

import random

import pytest

from conftest import game

RANGES = [range(1, 7), range(1, 11)]
DIFFS = ["EASY", "MEDIUM", "HARD"]


@pytest.fixture(autouse=True)
def seeded():
    random.seed(1234)


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("rng", RANGES)
@pytest.mark.parametrize("bowling", [True, False])
def test_picks_are_always_legal(diff, rng, bowling):
    brain = game.Brain(diff)
    for i in range(500):
        choice = brain.pick(rng, bowling)
        assert choice in rng
        brain.record(random.choice(list(rng)))


@pytest.mark.parametrize("diff", DIFFS)
def test_works_with_no_history(diff):
    assert game.Brain(diff).pick(range(1, 7), True) in range(1, 7)


@pytest.mark.parametrize("diff", ["MEDIUM", "HARD"])
def test_history_from_a_wider_range_is_ignored_safely(diff):
    # Player used 8s and 9s in a 1-10 game, then a new 1-6 innings starts.
    brain = game.Brain(diff)
    for v in [8, 9, 8, 8, 9, 8]:
        brain.record(v)
    for _ in range(200):
        assert brain.pick(range(1, 7), True) in range(1, 7)


def test_history_is_capped():
    brain = game.Brain("HARD")
    for i in range(100):
        brain.record(i % 6 + 1)
    assert len(brain.hist) == 12


def test_hard_bowler_targets_a_predictable_player():
    """A player who always shows 3 should get out far more often on HARD."""
    def outs(diff, balls=2000):
        brain = game.Brain(diff)
        n = 0
        for _ in range(balls):
            if brain.pick(range(1, 7), True) == 3:
                n += 1
            brain.record(3)
        return n / balls

    easy, hard = outs("EASY"), outs("HARD")
    assert easy < 0.25          # ~1 in 6 by chance
    assert hard > easy + 0.25   # HARD exploits the pattern


def test_hard_batter_avoids_the_players_favourite_number():
    brain = game.Brain("HARD")
    for _ in range(6):
        brain.record(4)
    hits = sum(brain.pick(range(1, 7), False) == 4 for _ in range(2000))
    assert hits / 2000 < 1 / 6
