"""Hold-to-confirm and quit-by-gesture timing."""

from conftest import FIST, GESTURES, game, make_hand, result


# ── GestureConfirm ───────────────────────────────────────────────────────────

def test_gesture_confirms_after_hold_time(clock):
    gc = game.GestureConfirm(hold=1.0)
    assert gc.update(3) == (None, 0.0)
    clock.advance(0.5)
    conf, prog = gc.update(3)
    assert conf is None and 0.45 < prog < 0.55
    clock.advance(0.5)
    assert gc.update(3) == (3, 1.0)


def test_confirmed_gesture_must_be_held_again_to_repeat(clock):
    gc = game.GestureConfirm(hold=1.0)
    gc.update(2)
    clock.advance(1.0)
    assert gc.update(2)[0] == 2
    # Next frame, same gesture: timer starts again rather than re-firing.
    assert gc.update(2) == (None, 0.0)


def test_changing_gesture_restarts_the_timer(clock):
    gc = game.GestureConfirm(hold=1.0)
    gc.update(2)
    clock.advance(0.9)
    assert gc.update(3) == (None, 0.0)
    clock.advance(0.9)
    assert gc.update(3)[0] is None
    clock.advance(0.1)
    assert gc.update(3)[0] == 3


def test_losing_the_hand_cancels_the_hold(clock):
    gc = game.GestureConfirm(hold=1.0)
    gc.update(4)
    clock.advance(0.9)
    assert gc.update(None) == (None, 0.0)
    clock.advance(5)
    assert gc.update(None) == (None, 0.0)


def test_reset_clears_progress(clock):
    gc = game.GestureConfirm(hold=1.0)
    gc.update(5)
    clock.advance(0.8)
    gc.reset()
    assert gc.update(5) == (None, 0.0)


# ── QuitDetector ─────────────────────────────────────────────────────────────

TWO_FISTS = result(make_hand(*FIST), make_hand(*FIST, dx=0.2))


def hold_fists(qd, clock, seconds, fps=30):
    out = None
    for _ in range(int(seconds * fps)):
        out = qd.update(TWO_FISTS)
        clock.advance(1 / fps)
    return out


def test_quit_needs_both_fists_held_for_quit_hold(clock):
    qd = game.QuitDetector()
    active, prog = hold_fists(qd, clock, game.QUIT_HOLD * 0.5)
    assert active and prog < 1.0
    active, prog = hold_fists(qd, clock, game.QUIT_HOLD * 0.6)
    assert prog == 1.0


def test_one_fist_never_quits(clock):
    qd = game.QuitDetector()
    for _ in range(300):
        assert qd.update(result(make_hand(*FIST))) == (False, 0.0)
        clock.advance(0.05)


def test_one_fist_one_open_hand_never_quits(clock):
    qd = game.QuitDetector()
    mpr = result(make_hand(*FIST), make_hand(*GESTURES[5], dx=0.2))
    for _ in range(300):
        assert qd.update(mpr) == (False, 0.0)
        clock.advance(0.05)


def test_any_interruption_resets_quit_timer(clock):
    qd = game.QuitDetector()
    hold_fists(qd, clock, game.QUIT_HOLD * 0.9)
    qd.update(result())                     # hands drop for one frame
    active, prog = hold_fists(qd, clock, game.QUIT_HOLD * 0.5)
    assert prog < 1.0


def test_brief_fist_flicker_does_not_start_timer(clock):
    qd = game.QuitDetector()
    for _ in range(4):
        assert qd.update(TWO_FISTS)[1] == 0.0
        clock.advance(1.0)
