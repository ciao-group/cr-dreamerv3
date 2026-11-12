"""Share of time humans looked at object categories (embodied/core/object_gaze.py)."""

import numpy as np
import pytest

from embodied.core import object_gaze, objects, vision

GRID = object_gaze.Grid(size=(96, 96), square=(24, 24), count=(5, 5))


def test_grid_matches_cr_atari():
  assert GRID.square_px == (40, 52)
  squares = GRID.squares()
  assert len(squares) == 25
  assert squares[0][:2] == (0, 0)
  assert squares[-1][0] + 40 == 160 and squares[-1][1] + 52 == 210


def test_cells_match_vision_conversion():
  rng = np.random.default_rng(0)
  xy = rng.uniform((0, 0), (160, 210), size=(500, 2))
  expected = [vision.convert_2d_gaze_position_to_1d_vision_square_position(
      tuple(p), GRID.count, object_gaze.SCREEN, GRID.square_px) for p in xy]
  assert GRID.cells(xy).tolist() == expected


def test_frame_shares():
  enemy = objects.GameObject('enemy', 100, 100, 7, 11)
  # Two samples on the enemy, one 3 px left of it (inside the 1 degree
  # margin of ~3.6 px), one far away.
  gaze = np.array([[103, 105], [101, 101], [97, 105], [10, 30]], float)
  shares = object_gaze.frame_shares([enemy], gaze, GRID)
  assert shares['looked_at_direct/enemy'] == pytest.approx(0.75)
  assert shares['on_screen/enemy'] == 1.0
  assert shares['on_screen/player'] == 0.0
  assert shares['looked_at_direct/player'] == 0.0
  # The grid squares of the first three samples overlap the enemy.
  assert shares['looked_at_grid/enemy'] == pytest.approx(0.75)
  # Grown box: (7 + 2 * 3.59) x (11 + 2 * 7.37) px of 160 x 210.
  assert shares['chance_direct/enemy'] == pytest.approx(14 * 26 / (160 * 210), rel=0.1)
  hits = objects.square_hits([enemy], GRID.squares())
  assert shares['chance_grid/enemy'] == pytest.approx(hits['enemy'].mean())


def test_read_trial_drops_offscreen_and_missing_gaze(tmp_path):
  path = tmp_path / 'trial.txt'
  path.write_text(
      'frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions\n'
      'RZ_1_1,null,null,1358,0,0,79.0,105.0,-3.0,50.0,80.0,106.0\n'
      'RZ_1_2,null,null,50,0,0,null\n')
  ids, durations, gaze = object_gaze.read_trial(str(path))
  assert ids == ['RZ_1_1', 'RZ_1_2']
  assert durations.tolist() == [1358, 50]
  assert gaze[0].tolist() == [[79, 105], [80, 106]]
  assert len(gaze[1]) == 0


def test_trial_shares_weights_frames_by_duration(tmp_path):
  from PIL import Image
  frame = np.zeros((210, 160, 3), np.uint8)
  frame[100:111, 100:107] = (228, 111, 111)  # enemy
  (tmp_path / 'X_1').mkdir()
  for i in (1, 2):
    Image.fromarray(frame).save(tmp_path / 'X_1' / f'X_1_{i}.png')
  # Frame 1 (300 ms): gaze on the enemy; frame 2 (100 ms): gaze elsewhere.
  (tmp_path / 'X_1.txt').write_text(
      'frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions\n'
      'X_1_1,null,null,300,0,0,103,105,104,106\n'
      'X_1_2,null,null,100,0,0,10,30\n')
  result = object_gaze.trial_shares(str(tmp_path), 'X_1', GRID)
  assert result['looked_at_direct/enemy'] == pytest.approx(0.75)
  assert result['on_screen/enemy'] == pytest.approx(1.0)
  assert result['frames'] == 2
