"""Find game objects (avatar, enemies, items) in raw Atari frames by colour.

The same detector runs on the agent's frames and on the Atari-HEAD frames of
human players, so that "which objects were looked at" is measured the same way
for both. It works on the raw RGB screen (210x160x3) and needs no RAM, which
Atari-HEAD does not provide.

Detection is stateless: every frame is segmented on its own, there is no
tracking between frames. Player pixels are found first, so the player is still
separated from an item or enemy it overlaps. All other non-background pixels in
the playfield are grouped into connected components and classified by their
colours.

The colour tables follow OCAtari's vision mode (ocatari/vision/<game>.py).
OCAtari itself is not used at runtime: its Asterix detector drops all items,
because every item type is matched into the same object slots and a later type
without detections clears the earlier ones.
"""

import dataclasses

import cv2
import numpy as np

CATEGORIES = ('player', 'enemy', 'item', 'reward')


@dataclasses.dataclass(frozen=True)
class GameObject:
  category: str
  x: int
  y: int
  w: int
  h: int

  @property
  def bbox(self) -> tuple[int, int, int, int]:
    return (self.x, self.y, self.w, self.h)


@dataclasses.dataclass(frozen=True)
class GameSpec:
  """Colours and layout of one game.

  playfield: (first, last + 1) screen rows that contain game objects; the HUD
    (score, lives) outside of it is ignored.
  background: colours that never belong to an object.
  player: colours of the avatar.
  enemy: a component with any of these colours is an enemy.
  reward: a component whose colours all lie in this set is a reward display
    (points shown where an item was collected).
  Everything else in the playfield is an item.
  """
  playfield: tuple[int, int]
  background: tuple[tuple[int, int, int], ...]
  player: tuple[tuple[int, int, int], ...]
  enemy: tuple[tuple[int, int, int], ...]
  reward: tuple[tuple[int, int, int], ...]


GAMES = {
    # Level 1 checked on ALE frames: player 8x11 (16x11 when wide), lyre
    # (enemy) 7x11, cauldron (item) 7x10, reward display "50" 6x11. The items
    # of later levels (helmet, shield, lamp, ...) fall into 'item'.
    'asterix': GameSpec(
        playfield=(24, 152),
        background=((0, 0, 0), (84, 92, 214), (51, 26, 163)),
        player=((187, 187, 53),),
        enemy=((228, 111, 111),),
        reward=((127, 92, 213), (170, 170, 170), (198, 89, 179)),
    ),
}

_KERNEL = np.ones((3, 3), np.uint8)


def _pack(colours) -> np.ndarray:
  colours = np.asarray(colours, np.uint32).reshape(-1, 3)
  return (colours[:, 0] << 16) | (colours[:, 1] << 8) | colours[:, 2]


def _components(mask: np.ndarray):
  """Bounding boxes and labels of connected components after closing small
  gaps, as objects are drawn with holes and stripes."""
  closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _KERNEL)
  n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
  return labels, [tuple(int(v) for v in stats[k, :4]) for k in range(1, n)]


def detect(frame: np.ndarray, game: str) -> list[GameObject]:
  """Objects in one RGB frame (H, W, 3) in screen pixel coordinates."""
  if game not in GAMES:
    raise ValueError(f"No object colours for game '{game}', known: {sorted(GAMES)}")
  spec = GAMES[game]
  top, bottom = spec.playfield
  field = np.asarray(frame)[top:bottom].astype(np.uint32)
  packed = (field[..., 0] << 16) | (field[..., 1] << 8) | field[..., 2]

  objects = []
  player = np.isin(packed, _pack(spec.player))
  _, boxes = _components(player)
  objects += [GameObject('player', x, y + top, w, h) for x, y, w, h in boxes]

  rest = ~np.isin(packed, _pack(spec.background)) & ~player
  labels, boxes = _components(rest)
  enemy, reward = _pack(spec.enemy), _pack(spec.reward)
  for k, (x, y, w, h) in enumerate(boxes, 1):
    colours = packed[(labels == k) & rest]
    if np.isin(colours, enemy).any():
      category = 'enemy'
    elif np.isin(colours, reward).all():
      category = 'reward'
    else:
      category = 'item'
    objects.append(GameObject(category, x, y + top, w, h))
  return objects


def intersects(obj: GameObject, box, margin: float = 0.0) -> bool:
  """Whether the object's box overlaps `box` (x, y, w, h), both in screen
  pixels, after growing the object by `margin` pixels on every side."""
  x, y, w, h = box
  return (obj.x - margin < x + w and x < obj.x + obj.w + margin
          and obj.y - margin < y + h and y < obj.y + obj.h + margin)


def categories_in(objects, box, margin: float = 0.0) -> set[str]:
  """Categories of all objects that overlap `box`."""
  return {o.category for o in objects if intersects(o, box, margin)}


def square_hits(objects, squares, margin: float = 0.0) -> dict[str, np.ndarray]:
  """For every category, which of the `squares` (x, y, w, h) overlap at least
  one object of that category, as a bool array of len(squares).

  With the squares of the vision grid, entry [gaze] says whether the category
  was looked at, and the mean over all squares is the chance level of a gaze
  chosen uniformly from the grid."""
  hits = {c: np.zeros(len(squares), bool) for c in CATEGORIES}
  for obj in objects:
    row = hits[obj.category]
    for k, box in enumerate(squares):
      row[k] = row[k] or intersects(obj, box, margin)
  return hits
