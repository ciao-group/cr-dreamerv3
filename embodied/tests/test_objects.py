"""Colour-based object detection in Atari frames (embodied/core/objects.py).

Synthetic frames check the classification rules; real ALE frames check that
the detected player matches the player position in RAM.
"""

import numpy as np
import pytest

from embodied.core import objects

YELLOW = (187, 187, 53)
PINK, RED, DARK_RED = (228, 111, 111), (184, 50, 50), (167, 26, 26)
VIOLET, GREY, PURPLE = (127, 92, 213), (170, 170, 170), (198, 89, 179)
LANE = (84, 92, 214)


def blank():
  frame = np.zeros((210, 160, 3), np.uint8)
  frame[40:42] = LANE  # lane lines are background
  return frame


def draw(frame, x, y, w, h, *colours):
  """Fill a box with horizontal stripes of the given colours."""
  for i in range(h):
    frame[y + i, x:x + w] = colours[i % len(colours)]


def found(frame):
  return sorted((o.category, o.bbox) for o in objects.detect(frame, 'asterix'))


def test_classifies_each_object_type():
  frame = blank()
  draw(frame, 60, 74, 8, 11, YELLOW)
  draw(frame, 100, 106, 7, 11, PINK, RED)
  draw(frame, 20, 122, 7, 10, DARK_RED, RED)
  draw(frame, 120, 58, 6, 11, VIOLET, GREY, PURPLE)
  assert found(frame) == sorted([
      ('player', (60, 74, 8, 11)),
      ('enemy', (100, 106, 7, 11)),
      ('item', (20, 122, 7, 10)),
      ('reward', (120, 58, 6, 11)),
  ])


def test_player_is_separated_from_an_overlapping_item():
  frame = blank()
  draw(frame, 50, 90, 7, 10, DARK_RED, RED)
  draw(frame, 54, 90, 8, 11, YELLOW)
  cats = [c for c, _ in found(frame)]
  assert sorted(cats) == ['item', 'player']


def test_hud_outside_the_playfield_is_ignored():
  frame = blank()
  draw(frame, 70, 180, 20, 8, YELLOW)  # score
  draw(frame, 30, 165, 8, 11, YELLOW)  # lives
  assert found(frame) == []


def test_empty_frame_has_no_objects():
  assert found(blank()) == []


def test_unknown_game_raises():
  with pytest.raises(ValueError):
    objects.detect(blank(), 'pong')


def test_intersects_with_margin():
  obj = objects.GameObject('enemy', 10, 10, 5, 5)
  assert objects.intersects(obj, (14, 14, 3, 3))
  assert not objects.intersects(obj, (15, 10, 3, 3))  # touching edge only
  assert objects.intersects(obj, (16, 10, 3, 3), margin=2)
  assert objects.categories_in([obj], (0, 0, 9, 9)) == set()


def test_player_matches_ram_position_on_ale_frames():
  ale_py = pytest.importorskip('ale_py')
  from ale_py import roms
  ale = ale_py.ALEInterface()
  ale.setLoggerMode(ale_py.LoggerMode.Error)
  ale.setInt('random_seed', 0)
  ale.setFloat('repeat_action_probability', 0.0)
  ale.loadROM(roms.get_rom_path('asterix'))
  actions = ale.getMinimalActionSet()
  rng = np.random.default_rng(0)
  ram = np.zeros(128, np.uint8)
  hits = total = 0
  for step in range(3000):
    ale.act(actions[rng.integers(len(actions))])
    if ale.game_over():
      ale.reset_game()
    if step % 10:
      continue
    # getRAM() without a buffer returns zeros with ale_py 0.9 and numpy 2.
    ale.getRAM(ram)
    x, y = int(ram[41]), 16 * int(ram[39]) + 26
    players = [o for o in objects.detect(ale.getScreenRGB(), 'asterix')
               if o.category == 'player' and o.w == 8]
    if not players:
      continue
    total += 1
    hits += any((o.x, o.y) == (x, y) for o in players)
  assert total > 200
  assert hits / total > 0.98


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
