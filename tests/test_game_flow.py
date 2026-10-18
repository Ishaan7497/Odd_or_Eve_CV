"""Innings, chases, super overs and milestones."""

import pytest

from conftest import game


def apply(g, **ball):
    """Play one ball's outcome into `g` and apply the resulting transition."""
    g.update(ball)
    if not g['ball_out']:
        g['current_runs'] += ball.get('ball_runs', 0)
    g.update(game.end_of_ball_update(g))
    return g


def test_new_game_starts_at_mode_select():
    g = game.new_game()
    assert g['state'] == 'MODE_SELECT'
    assert g['innings_scores'] == [None, None]


def test_new_games_do_not_share_state():
    a, b = game.new_game(), game.new_game()
    a['innings_scores'][0] = 42
    assert b['innings_scores'] == [None, None]


def test_end_of_ball_does_not_mutate_the_game(fresh_game):
    g = fresh_game
    g.update({'state': 'BALL_RESULT', 'is_super_over': True, 'super_balls': 6})
    before = dict(g)
    game.end_of_ball_update(g)
    assert g == before


def test_runs_keep_the_innings_going(fresh_game):
    g = apply(fresh_game, ball_out=False, ball_runs=4)
    assert g['state'] == 'PLAYING'
    assert g['current_runs'] == 4


def test_first_wicket_ends_first_innings_and_swaps_roles(fresh_game):
    g = apply(fresh_game, ball_out=False, ball_runs=5)
    g = apply(g, ball_out=True, ball_runs=0)
    assert g['state'] == 'INNINGS_BREAK'
    assert g['innings'] == 1
    assert g['innings_scores'] == [5, None]
    assert g['current_runs'] == 0
    assert g['user_batting'] is False


def test_out_first_ball_gives_target_of_one(fresh_game):
    g = apply(fresh_game, ball_out=True, ball_runs=0)
    assert g['innings_scores'] == [0, None]
    # Chaser wins with any run.
    g = apply(g, ball_out=False, ball_runs=1)
    assert g['state'] == 'GAME_RESULT'
    assert g['innings_scores'] == [0, 1]


def chase(target_score, **kw):
    g = game.new_game()
    g.update({'mode': '1 to 6', 'run_range': range(1, 7), 'innings': 1,
              'innings_scores': [target_score, None], 'user_batting': False,
              'batting_first': 'user', 'current_runs': 0})
    g.update(kw)
    return g


def test_chase_ends_as_soon_as_target_is_passed():
    g = apply(chase(10), ball_out=False, ball_runs=6)
    assert g['state'] == 'PLAYING'
    g = apply(g, ball_out=False, ball_runs=5)
    assert g['state'] == 'GAME_RESULT'
    assert g['innings_scores'] == [10, 11]


def test_equalling_the_target_does_not_win():
    g = apply(chase(10), ball_out=False, ball_runs=6)
    g = apply(g, ball_out=False, ball_runs=4)
    assert g['current_runs'] == 10
    assert g['state'] == 'PLAYING'


def test_chaser_out_short_of_target_loses():
    g = apply(chase(10), ball_out=False, ball_runs=3)
    g = apply(g, ball_out=True, ball_runs=0)
    assert g['state'] == 'GAME_RESULT'
    assert g['innings_scores'] == [10, 3]


def test_tie_starts_a_super_over_with_user_batting():
    g = apply(chase(6, user_batting=False, batting_first='computer'), ball_out=False, ball_runs=6)
    g = apply(g, ball_out=True, ball_runs=0)
    assert g['state'] == 'INNINGS_BREAK'
    assert g['is_super_over'] is True
    assert g['innings'] == 0
    assert g['innings_scores'] == [None, None]
    assert g['super_balls'] == 6
    assert g['user_batting'] is True
    assert g['batting_first'] == 'user'


def super_over(innings=0, first=None):
    g = game.new_game()
    g.update({'mode': '1 to 6', 'run_range': range(1, 7), 'is_super_over': True,
              'innings': innings, 'innings_scores': [first, None], 'super_balls': 6,
              'user_batting': innings == 0, 'batting_first': 'user', 'current_runs': 0})
    return g


def test_super_over_lasts_six_balls_and_outs_do_not_end_it():
    g = super_over()
    for ball in range(5):
        g = apply(g, ball_out=(ball % 2 == 0), ball_runs=2)
        assert g['state'] == 'PLAYING'
    g = apply(g, ball_out=False, ball_runs=2)
    assert g['state'] == 'INNINGS_BREAK'
    assert g['innings_scores'] == [6, None]   # only the 3 non-out balls scored
    assert g['super_balls'] == 6
    assert g['user_batting'] is False


def test_super_over_chase_stops_when_target_passed():
    g = apply(super_over(innings=1, first=5), ball_out=False, ball_runs=6)
    assert g['state'] == 'GAME_RESULT'
    assert g['innings_scores'] == [5, 6]


def test_super_over_tie_is_final():
    g = super_over(innings=1, first=4)
    for _ in range(5):
        g = apply(g, ball_out=True, ball_runs=0)
    g = apply(g, ball_out=False, ball_runs=4)
    assert g['state'] == 'GAME_RESULT'
    assert g['innings_scores'] == [4, 4]


# ── Milestones ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("runs,expected", [(49, None), (50, 'FIFTY'), (99, 'FIFTY'), (100, 'HUNDRED')])
def test_milestone_thresholds(fresh_game, runs, expected):
    fresh_game['current_runs'] = runs
    assert game.check_milestone(fresh_game) == expected


def test_each_milestone_fires_once(fresh_game):
    g = fresh_game
    g['current_runs'] = 55
    assert game.check_milestone(g) == 'FIFTY'
    g['current_runs'] = 60
    assert game.check_milestone(g) is None
    g['current_runs'] = 104
    assert game.check_milestone(g) == 'HUNDRED'
    g['current_runs'] = 150
    assert game.check_milestone(g) is None


def test_jumping_past_fifty_and_hundred_celebrates_hundred(fresh_game):
    fresh_game['current_runs'] = 45 + 100   # e.g. 10^2 in CRAZY mode
    assert game.check_milestone(fresh_game) == 'HUNDRED'
    assert fresh_game['fifty_shown'] is True
    assert game.check_milestone(fresh_game) is None
