"""Scoring rules for a single ball (resolve)."""

import itertools

import pytest

from conftest import game

CLASSIC_PAIRS = list(itertools.product(range(1, 7), repeat=2))
TEN_PAIRS = list(itertools.product(range(1, 11), repeat=2))


@pytest.mark.parametrize("mode,pairs", [("1 to 6", CLASSIC_PAIRS), ("1 to 10", TEN_PAIRS)])
def test_normal_modes_out_only_on_same_number(mode, pairs):
    for bat, bowl in pairs:
        runs, out, msg = game.resolve(bat, bowl, mode)
        if bat == bowl:
            assert (runs, out) == (0, True), (bat, bowl)
            assert "OUT" in msg
        else:
            assert (runs, out) == (bat, False), (bat, bowl)
            assert msg == f"+{bat}"


@pytest.mark.parametrize("bat,bowl", [(3, 4), (4, 3), (1, 2), (9, 10), (10, 1), (1, 10)])
def test_crazy_adjacent_numbers_are_out(bat, bowl):
    runs, out, msg = game.resolve(bat, bowl, "CRAZY")
    assert (runs, out) == (0, True)
    assert "Adjacent" in msg


@pytest.mark.parametrize("n", range(1, 11))
def test_crazy_same_number_scores_its_square(n):
    runs, out, msg = game.resolve(n, n, "CRAZY")
    assert (runs, out) == (n * n, False)
    assert "SQUARED" in msg


def test_crazy_other_numbers_score_normally():
    for bat, bowl in TEN_PAIRS:
        d = abs(bat - bowl)
        if d in (0, 1, 9):
            continue
        assert game.resolve(bat, bowl, "CRAZY") == (bat, False, f"+{bat}")


def test_runs_are_never_negative_and_outs_never_score():
    for mode, pairs in (("1 to 6", CLASSIC_PAIRS), ("1 to 10", TEN_PAIRS), ("CRAZY", TEN_PAIRS)):
        for bat, bowl in pairs:
            runs, out, _ = game.resolve(bat, bowl, mode)
            assert runs >= 0
            if out:
                assert runs == 0
