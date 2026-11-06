"""The size of the vision square and the number of gaze positions are
independent of each other.

- The number of positions (vision_square_count) only moves the squares, it
  never changes how many pixels the agent sees per fixation.
- The square size (vision_square_size) never changes how many positions
  there are; the grid stays on screen and centered for every size.

Debug in PyCharm: right-click this file -> "Debug 'pytest in test_vision_...'",
or set a breakpoint and run the file directly (see __main__ at the bottom).
"""

import numpy as np
import pytest

from embodied.core import vision

SIZE = (96, 96)
SCREEN = (160, 210)  # (W, H) of the ALE screen
COUNTS = [(3, 3), (4, 4), (5, 5), (6, 6), (5, 4)]
SQUARES = [(16, 16), (24, 24), (32, 32), (24, 16)]


def scaled(square):
  """Vision square on the ALE screen, as computed in CrAtari."""
  return (square[0] * SCREEN[0] // SIZE[0], square[1] * SCREEN[1] // SIZE[1])


def fits(count, square):
  """Same condition as the assertion in CrAtari.__init__: no gaps."""
  return all(count[i] * square[i] >= SIZE[i] for i in (0, 1))


def visible_box(position, count, square_px):
  """Bounding box (x, y, w, h) and pixel count of what the agent sees."""
  image = np.full((SCREEN[1], SCREEN[0]), 255, dtype=np.uint8)
  seen = vision.apply_vision_square(
      gaze_position=position,
      image=image,
      mode='foveated',
      vision_square_count=count,
      vision_square_size=square_px,
      screen_size=SCREEN,
  )
  ys, xs = np.nonzero(seen)
  box = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
  return box, len(xs)


@pytest.mark.parametrize('square', SQUARES)
@pytest.mark.parametrize('count', COUNTS)
def test_visible_area_does_not_depend_on_count(count, square):
  if not fits(count, square):
    pytest.skip('grid would leave gaps (rejected by CrAtari)')
  square_px = scaled(square)
  for position in range(count[0] * count[1]):
    (x, y, w, h), pixels = visible_box(position, count, square_px)
    # Exactly one full square, never cut off at the screen edge.
    assert (w, h) == square_px, (position, (x, y, w, h))
    assert pixels == square_px[0] * square_px[1], position
    assert x >= 0 and y >= 0
    assert x + w <= SCREEN[0] and y + h <= SCREEN[1], (position, (x, y))


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('square', SQUARES)
def test_number_of_positions_does_not_depend_on_square_size(count, square):
  if not fits(count, square):
    pytest.skip('grid would leave gaps (rejected by CrAtari)')
  square_px = scaled(square)
  corners = {
      vision.convert_1d_vision_square_position_to_2d(
          vision_square_position=i,
          vision_square_count=count,
          screen_size=SCREEN,
          vision_square_size=square_px,
      )
      for i in range(count[0] * count[1])
  }
  # Every position is a distinct square.
  assert len(corners) == count[0] * count[1]


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('square', SQUARES)
def test_grid_is_centered_for_every_square_size(count, square):
  if not fits(count, square):
    pytest.skip('grid would leave gaps (rejected by CrAtari)')
  square_px = scaled(square)
  for axis in (0, 1):
    offsets = [
        vision.vision_square_offset(i, count[axis], square_px[axis], SCREEN[axis])
        for i in range(count[axis])]
    assert offsets[0] == 0
    assert offsets[-1] + square_px[axis] == SCREEN[axis], offsets
    assert np.all(np.diff(offsets) > 0), offsets
    centers = np.array(offsets) + square_px[axis] / 2
    assert centers.mean() == pytest.approx(SCREEN[axis] / 2, abs=0.5), centers


@pytest.mark.parametrize('n', [3, 5, 7])
def test_center_cell_stays_at_screen_center_for_every_square_size(n):
  count = (n, n)
  for square in SQUARES:
    if not fits(count, square):
      continue
    square_px = scaled(square)
    center = vision.convert_1d_vision_square_position_to_2d_center(
        vision_square_position=vision.center_vision_square_position(count),
        vision_square_count=count,
        vision_square_size=square_px,
        screen_size=SCREEN,
    )
    assert center[0] == pytest.approx(SCREEN[0] / 2, abs=1), (square, center)
    assert center[1] == pytest.approx(SCREEN[1] / 2, abs=1), (square, center)


@pytest.mark.parametrize('count,square', [
    ((4, 4), (24, 24)),
    ((5, 5), (24, 24)),
    ((5, 5), (32, 32)),
    ((6, 6), (16, 16)),
])
def test_cr_atari_action_space_and_square_are_independent(count, square):
  from embodied.envs.cr_atari import CrAtari
  env = CrAtari(
      'asterix', size=SIZE, vision_square_size=square,
      vision_square_count=count, seed=0)
  try:
    # Number of gaze actions follows only the count ...
    assert env.act_space['gaze_position'].high == count[0] * count[1]
    # ... and the square on the screen follows only the size.
    assert env.scaled_vision_square_size == scaled(square)
  finally:
    env.close()


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
