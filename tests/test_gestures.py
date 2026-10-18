"""Hand landmarks -> finger states -> game numbers."""

import pytest

from conftest import FIST, GESTURES, LM, game, make_hand, result

ALL = range(1, 11)


@pytest.mark.parametrize("number,fingers", GESTURES.items())
def test_every_single_hand_gesture_is_recognised(number, fingers):
    hand = make_hand(*fingers)
    assert game.fstates(hand) == fingers
    assert game.raw_gesture(result(hand), ALL) == number


@pytest.mark.parametrize("number,fingers", GESTURES.items())
@pytest.mark.parametrize("scale", [0.4, 1.0, 1.6])
@pytest.mark.parametrize("dx,dy", [(0, 0), (-0.3, -0.2), (0.3, 0.1)])
def test_recognition_does_not_depend_on_hand_size_or_position(number, fingers, scale, dx, dy):
    hand = make_hand(*fingers, dx=dx, dy=dy, scale=scale)
    assert game.raw_gesture(result(hand), ALL) == number


def test_two_cupped_hands_is_ten():
    mpr = result(make_hand(*FIST), make_hand(*FIST, dx=0.2))
    assert game.raw_gesture(mpr, ALL) == 10


def test_cupped_hands_with_one_finger_each_still_counts_as_ten():
    mpr = result(make_hand(index=1), make_hand(thumb=1, dx=0.2))
    assert game.raw_gesture(mpr, ALL) == 10


def test_ten_is_rejected_when_not_a_valid_choice():
    mpr = result(make_hand(*FIST), make_hand(*FIST, dx=0.2))
    assert game.raw_gesture(mpr, range(1, 7)) is None


def test_two_open_hands_is_not_a_gesture():
    mpr = result(make_hand(*GESTURES[5]), make_hand(*GESTURES[5], dx=0.2))
    assert game.raw_gesture(mpr, ALL) is None


def test_no_hands_returns_none():
    assert game.raw_gesture(result(), ALL) is None


def test_single_fist_is_not_a_number():
    assert game.raw_gesture(result(make_hand(*FIST)), ALL) is None


@pytest.mark.parametrize("fingers", [
    (0, 0, 0, 0, 1),   # pinky only
    (0, 1, 0, 0, 1),   # "rock on"
    (0, 0, 1, 0, 0),   # middle finger only
    (1, 0, 0, 0, 1),   # "shaka"
])
def test_unmapped_finger_combinations_are_ignored(fingers):
    assert game.raw_gesture(result(make_hand(*fingers)), ALL) is None


@pytest.mark.parametrize("number", [7, 8, 9])
def test_numbers_outside_the_mode_range_are_ignored(number):
    hand = make_hand(*GESTURES[number])
    assert game.raw_gesture(result(hand), range(1, 7)) is None


def test_menus_only_accept_their_options():
    three = result(make_hand(*GESTURES[3]))
    assert game.raw_gesture(three, range(1, 3)) is None
    assert game.raw_gesture(three, range(1, 4)) == 3


def test_degenerate_landmarks_do_not_crash():
    # All 21 points on top of each other (bad detection): palm size is zero.
    hand = [LM(0.5, 0.5) for _ in range(21)]
    assert game.fstates(hand) == FIST
    assert game.raw_gesture(result(hand), ALL) is None


def test_hand_raised_detection():
    assert game.is_hand_raised(result(make_hand(*GESTURES[5])))
    assert not game.is_hand_raised(result(make_hand(*FIST)))
    assert not game.is_hand_raised(result())
