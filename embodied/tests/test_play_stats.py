"""Avatar and gaze statistics (embodied/core/play_stats.py)."""

import importlib.util
import pathlib

import numpy as np
import pytest

from embodied.core import object_gaze, play_stats

GRID = object_gaze.Grid(size=(96, 96), square=(24, 24), count=(5, 5))
N = 25


def test_cut():
  rewards = np.array([0, 50, 50, 0])
  assert play_stats.cut(rewards) == 4
  assert play_stats.cut(rewards, max_frames=2) == 2
  assert play_stats.cut(rewards, max_score=50) == 2
  assert play_stats.cut(rewards, max_frames=1, max_score=50) == 1


def avatar_square(x, y):
  return int(GRID.cells(np.array([[x + 4, y + 5.5]]))[0])


def test_stats_gaze_on_avatar():
  # Avatar on lane 3 (y = 26 + 3 * 16), gaze always on its square.
  x, y = 76, 74
  cell = avatar_square(x, y)
  traj = play_stats.agent_trajectory([5, 5, 10], [0, 50, 0], [x] * 3, [y] * 3, [cell] * 3, N)
  s = play_stats.stats(traj, GRID)
  assert s['minutes'] == pytest.approx(1000 / 60000)
  assert s['score'] == 50
  assert s['lane/3'] == 1.0
  assert s['avatar_x_mean'] == x
  assert s['gaze_avatar_same_square'] == 1.0
  assert s['gaze_avatar_drow/0'] == 1.0
  assert s['gaze_switches_per_s'] == 0
  assert s['gaze_dwell_ms'] == 1000
  assert s['gaze_entropy'] == 0
  assert s[f'gaze_cell/{cell}'] == 1.0


def test_stats_time_weighting_and_offsets():
  x, y = 76, 74
  cell = avatar_square(x, y)
  below = cell + 5  # one row lower
  # 1 frame on the avatar, 3 frames one row below it: 25 % / 75 %.
  traj = play_stats.agent_trajectory([1, 3], [0, 0], [x, x], [y, y], [cell, below], N)
  s = play_stats.stats(traj, GRID)
  assert s['gaze_switches_per_s'] == pytest.approx(1 / 0.2)
  assert s[f'gaze_cell/{below}'] == pytest.approx(0.75)
  assert s['gaze_avatar_drow/0'] == pytest.approx(0.25)
  assert s['gaze_avatar_drow/1'] == pytest.approx(0.75)
  assert s['gaze_avatar_dcol/0'] == 1.0
  assert s['gaze_avatar_near'] == 1.0
  assert s['gaze_entropy'] == pytest.approx(-(0.25 * np.log2(0.25) + 0.75 * np.log2(0.75)))


def test_stats_lanes():
  # Lane 3 for 1 frame, lane 4 for 3 frames.
  traj = play_stats.agent_trajectory([1, 3], [0, 0], [76, 76], [74, 90], [0, 0], N)
  s = play_stats.stats(traj, GRID)
  assert s['lane/3'] == pytest.approx(0.25) and s['lane/4'] == pytest.approx(0.75)
  assert s['lane_changes_per_min'] == pytest.approx(1 / (200 / 60000))


def test_stats_missing_avatar_and_gaze():
  # Row 2 has no avatar, row 3 no gaze: they only count where they can.
  traj = play_stats.Trajectory(
      time=np.array([100.0, 100.0, 200.0]), reward=np.zeros(3),
      avatar=np.array([[76, 74], [np.nan, np.nan], [76, 74]], float),
      gaze=np.array([np.eye(N)[avatar_square(76, 74)], np.eye(N)[0], np.zeros(N)]))
  s = play_stats.stats(traj, GRID)
  assert s['avatar_found'] == pytest.approx(0.75)
  assert s[f'gaze_cell/{avatar_square(76, 74)}'] == pytest.approx(0.5)
  assert s['gaze_avatar_same_square'] == 1.0


def test_human_trajectory(tmp_path):
  from PIL import Image
  (tmp_path / 'X_1').mkdir()
  frame = np.zeros((210, 160, 3), np.uint8)
  frame[74:85, 76:84] = (187, 187, 53)  # player
  Image.fromarray(frame).save(tmp_path / 'X_1' / 'X_1_1.png')
  Image.fromarray(np.zeros_like(frame)).save(tmp_path / 'X_1' / 'X_1_2.png')
  (tmp_path / 'X_1.txt').write_text(
      'frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions\n'
      'X_1_1,null,null,300,0,0,80,80,80,80,5,5\n'
      'X_1_2,null,null,100,50,0,null\n')
  traj = play_stats.human_trajectory(str(tmp_path), 'X_1', GRID)
  assert traj.time.tolist() == [300, 100]
  assert traj.avatar[0].tolist() == [76, 74]
  assert np.isnan(traj.avatar[1]).all()
  assert traj.gaze[0, avatar_square(76, 74)] == pytest.approx(2 / 3)
  assert traj.gaze[0, 0] == pytest.approx(1 / 3)
  assert traj.gaze[1].sum() == 0


def test_eval_episodes_use_gaze_of_previous_action():
  path = pathlib.Path(__file__).resolve().parents[2] / 'scripts/vision_grid/eval_objects.py'
  spec = importlib.util.spec_from_file_location('eval_objects', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  episodes = module.Episodes(max_frames=0)

  def tran(first, last, frames, gaze):
    t = {'is_first': first, 'is_last': last, 'gaze_position': np.int32(gaze),
         'log/frames': frames, 'log/game_score': 0.0,
         'log/player_position_x_raw': 76, 'log/player_position_y_raw': 74}
    t.update({f'log/{k}/{c}': 0.0 for k in module.KINDS for c in module.objects.CATEGORIES})
    return t

  episodes(tran(True, False, 0, 7), worker=0)
  episodes(tran(False, False, 5, 8), worker=0)
  episodes(tran(False, True, 4, 9), worker=0)
  traj = episodes.done[0]['trajectory']
  assert traj['gaze'] == [7, 8]
  assert traj['frames'] == [5, 4]
