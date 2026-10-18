"""Smoke tests: every screen and effect draws without crashing.

These render onto blank frames at several window sizes (including very
small ones) to catch out-of-range drawing, missing game-state keys, etc.
"""

import numpy as np
import pytest

from conftest import GESTURES, game, make_hand, result

SIZES = [(720, 960), (480, 640), (240, 320), (1080, 1920)]


def blank(h, w):
    return np.full((h, w, 3), 40, dtype=np.uint8)


def mid_game(**kw):
    g = game.new_game()
    g.update({
        'mode': '1 to 6', 'run_range': range(1, 7), 'state': 'PLAYING',
        'toss_choice': 'ODD', 'toss_user_num': 1, 'toss_comp_num': 2,
        'toss_total': 3, 'toss_result_str': 'ODD', 'user_won_toss': True,
        'toss_comp_dec': 'BAT', 'user_batting': True, 'batting_first': 'user',
        'current_runs': 17, 'ball_msg': '+4', 'you_showed': 4, 'comp_showed': 2,
        'ball_timer': None,
    })
    g.update(kw)
    return g


GAMES = {
    'first_innings': mid_game(),
    'chasing': mid_game(innings=1, innings_scores=[20, None], user_batting=False),
    'super_over_1': mid_game(is_super_over=True, innings=0, super_balls=3),
    'super_over_2': mid_game(is_super_over=True, innings=1, innings_scores=[7, None]),
    'crazy': mid_game(mode='CRAZY', run_range=range(1, 11)),
    'lost_toss': mid_game(user_won_toss=False),
    'won': mid_game(innings_scores=[30, 12]),
    'lost': mid_game(innings_scores=[12, 30]),
    'tied': mid_game(innings_scores=[12, 12], is_super_over=True),
}


@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("name", GAMES)
def test_all_screens_render(h, w, name):
    g = dict(GAMES[name])
    for dkey in ('EASY', 'MEDIUM', 'HARD'):
        f = blank(h, w)
        game.screen_mode_select(f, None, w, h, 10)
        game.screen_difficulty(f, None, w, h, 10)
        game.screen_toss_choice(f, w, h)
        game.screen_toss_number(f, g, w, h)
        game.screen_toss_result(f, g, None, w, h)
        game.screen_play(f, g, dkey, dkey, w, h)
        game.screen_ball_result(f, g, w, h, 1.2)
        game.screen_ball_result(f, g, w, h, 0)       # zero pause must not divide by 0
        if g['innings'] == 1 or g['is_super_over']:   # only reachable then
            game.screen_innings_break(f, g, w, h)
        game.screen_game_result(f, g, dkey, w, h)
        game.topbar(f, g, w, dkey, dkey)
        assert f.shape == (h, w, 3)


@pytest.mark.parametrize("h,w", SIZES)
def test_hud_widgets_render(h, w):
    f = blank(h, w)
    for prog in (0.0, 0.3, 1.0):
        game.gesture_indicator(f, 5, prog, w)
        game.gesture_indicator(f, None, prog, w)
        game.progress_bar(f, prog, w, h)
        game.draw_quit_indicator(f, prog, w, h)


@pytest.mark.parametrize("h,w", SIZES)
def test_coin_animation_every_frame(h, w):
    for t in range(game.COIN_DUR + 5):
        game.draw_coin(blank(h, w), t, 1, 2, w, h)


@pytest.mark.parametrize("kind", game.CELEBRATION_CONFIG)
def test_celebrations_render_and_finish(kind, clock):
    h, w = 720, 960
    g = mid_game(celebration_start=clock(), celebration_acknowledged=False)
    raised = result(make_hand(*GESTURES[5]))
    done = False
    for i in range(400):
        done = game.screen_celebration(blank(h, w), g, kind, w, h, raised, i)
        clock.advance(1 / 30)
        if done:
            break
    assert done, f"{kind} celebration never finished"
    game.fx.clear()


def test_hand_skeleton_draws_and_handles_no_hands():
    f = blank(480, 640)
    game.draw_hands(f, result(), 640, 480)
    game.draw_hands(f, result(make_hand(*GESTURES[5])), 640, 480)


@pytest.mark.parametrize("h,w", SIZES)
def test_every_effect_runs_to_completion(h, w):
    fx = game.FX()
    for kind, kw in [('OUT', {}), ('BIGRUN', {'lbl': 'SIX!'}),
                     ('RUN_POP', {'runs': 6}), ('FLASH', {}), ('WIN', {})]:
        fx.trigger(kind, **kw)
    fx.confetti_on(w, h)
    for _ in range(300):
        fx.draw(blank(h, w))
    assert fx.q == []       # every timed effect expired
    fx.clear()
    assert fx.confetti is None


def test_rounded_rect_handles_zero_and_tiny_sizes():
    f = blank(50, 50)
    game.rounded_rect(f, (10, 10), (10, 10), (255, 255, 255), radius=12)
    game.rounded_rect(f, (10, 10), (12, 11), (255, 255, 255), radius=12)
    game.rounded_rect(f, (0, 0), (49, 49), (255, 255, 255), radius=100, thickness=1)
