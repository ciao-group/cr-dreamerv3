import io
import os
import tarfile

import numpy as np
from PIL import Image

from embodied.core import atari_head as ah

NAME = '100_XY_123_Jan-01-00-00-00'


def _gaze(x, y, n):
  return ','.join(f'{x:.2f},{y:.2f}' for _ in range(n))


def _make_trial(tmp_path):
  # Frames 1-2 normal (50 ms), frame 3 paused for 300 ms with two fixations,
  # frame 4 without gaze samples.
  rows = [
      f'XY_1,null,null,50,0,3,{_gaze(20, 30, 50)}',
      f'XY_2,null,null,50,10,4,{_gaze(20, 30, 50)}',
      f'XY_3,null,null,300,0,1,{_gaze(100, 150, 150)},{_gaze(40, 60, 150)}',
      'XY_4,null,null,50,0,0,null',
  ]
  with open(tmp_path / f'{NAME}.txt', 'w') as f:
    f.write('frame_id,episode_id,score,duration(ms),unclipped_reward,action,'
            'gaze_positions\n')
    f.write('\n'.join(rows) + '\n')
  with tarfile.open(tmp_path / f'{NAME}.tar.bz2', 'w:bz2') as tar:
    for i in range(1, 5):
      buf = io.BytesIO()
      Image.fromarray(np.full((210, 160, 3), i, np.uint8)).save(buf, 'PNG')
      info = tarfile.TarInfo(f'{NAME}/XY_{i}.png')
      info.size = buf.tell()
      buf.seek(0)
      tar.addfile(info, buf)


def test_load_trial(tmp_path):
  _make_trial(tmp_path)
  assert ah.list_trials(tmp_path) == [NAME]
  trial = ah.load_trial(str(tmp_path), NAME)
  assert len(trial) == 4
  assert trial.subject == 'XY'
  assert trial.score == 10
  np.testing.assert_array_equal(trial.actions, [3, 4, 1, 0])
  np.testing.assert_array_equal(trial.onsets, [0, 50, 100, 400])
  assert len(trial.gaze[3]) == 0
  assert trial.frame(2)[0, 0, 0] == 3
  assert os.path.isdir(trial.frame_dir)


def test_fixations(tmp_path):
  _make_trial(tmp_path)
  fix = ah.load_trial(str(tmp_path), NAME).fixations()
  np.testing.assert_allclose(fix[:, :2], [[20, 30], [100, 150], [40, 60]])


def test_to_steps_splits_paused_frames(tmp_path):
  _make_trial(tmp_path)
  steps = ah.to_steps(ah.load_trial(str(tmp_path), NAME))
  # Frame 3 becomes a pause step plus an action step.
  assert len(steps) == 5
  np.testing.assert_array_equal(steps.frames[:, 0], [0, 1, 2, 2, 3])
  np.testing.assert_array_equal(steps.pause, [0, 0, 1, 0, 0])
  np.testing.assert_array_equal(steps.action, [3, 4, -1, 1, 0])
  np.testing.assert_allclose(steps.fovea_xy[2], [100, 150])
  np.testing.assert_allclose(steps.next_gaze_xy[2], [40, 60])
  np.testing.assert_allclose(steps.duration, [50, 50, 150, 150, 50])
  np.testing.assert_array_equal(steps.valid, [1, 1, 1, 0, 0])
  assert steps.is_first[0] and not steps.is_first[1:].any()


def test_stride_groups_frames(tmp_path):
  _make_trial(tmp_path)
  steps = ah.to_steps(ah.load_trial(str(tmp_path), NAME), stride=2, pooling=2)
  np.testing.assert_array_equal(steps.frames[0], [0, 1])
  np.testing.assert_array_equal(steps.action[0], 3)


def test_split_trials():
  train, test = ah.split_trials([f't{i}' for i in range(6)])
  assert test == ['t2', 't5']
  assert len(train) == 4
