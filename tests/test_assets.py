"""Locating the hand model and optional audio files."""

import os

import pytest

from conftest import game


def test_bundled_model_is_found(monkeypatch):
    monkeypatch.delenv("HAND_LANDMARKER_PATH", raising=False)
    path = game.find_model()
    assert os.path.basename(path) == "hand_landmarker.task"
    assert os.path.isfile(path)


def test_env_var_overrides_model_path(monkeypatch, tmp_path):
    custom = tmp_path / "my_model.task"
    custom.write_bytes(b"x")
    monkeypatch.setenv("HAND_LANDMARKER_PATH", str(custom))
    assert game.find_model() == str(custom)


def test_bad_env_var_falls_back_to_bundled_model(monkeypatch, capsys):
    monkeypatch.setenv("HAND_LANDMARKER_PATH", "/definitely/not/here.task")
    assert os.path.basename(game.find_model()) == "hand_landmarker.task"
    assert "does not exist" in capsys.readouterr().out


def test_missing_model_exits_with_message(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("HAND_LANDMARKER_PATH", raising=False)
    monkeypatch.setattr(game, "ASSET_DIRS", (str(tmp_path),))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        game.find_model()
    assert e.value.code == 1
    assert "hand_landmarker.task not found" in capsys.readouterr().out


def test_find_asset_prefers_first_directory_and_extension(monkeypatch, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (b / "bgm.mp3").write_bytes(b"x")
    (a / "bgm.ogg").write_bytes(b"x")
    monkeypatch.setattr(game, "ASSET_DIRS", (str(a), str(b)))
    assert game.find_asset("bgm", ("mp3", "ogg")) == str(a / "bgm.ogg")
    assert game.find_asset("nothing", ("mp3",)) is None


def test_unknown_sound_name_is_ignored():
    game.play("this_sound_does_not_exist")
