# Object gaze: agent vs. Atari-HEAD humans (Asterix)

As of 2026-11-12, branch `feat/object-gaze-metrics`.

## Data

| Folder/file | Content | Created with |
|---|---|---|
| `agent/seed{0,1,2}_step*.csv` | 10 episodes per seed, model `grid5x5_square24x24_EMMA_foveated_effort`, cluster job 3310213 | `scripts/vision_grid/eval_objects.sh` |
| `human/full_trials.csv` | 15 trials (15 min each, one episode) | `scripts/human_object_gaze.py` |
| `human/max_frames_5000.csv` | first 5000 frames of each trial | `... --max-frames 5000` |
| `human/max_score_7000.csv`, `max_score_12000.csv` | up to score 7000 / 12000 | `... --max-score N` |
| `human/play_stats_*.csv` | avatar positions and gaze behaviour (same cuts) | `scripts/human_play_stats.py` |
| `comparison.csv` | object shares per group (human cut, agent seed, all seeds) and category: mean, sd, min, max, chance, ratio, on screen | `scripts/object_gaze_report.py` |
| `gaze_heatmaps.png`, `gaze_grid.csv` | gaze time per location: humans per pixel and on the grid (mean over trials), agent on the grid once `agent/*play_stats*.csv` exist | `scripts/object_gaze_report.py` |

Method: `embodied/core/object_gaze.py` (object shares) and `embodied/core/play_stats.py`
(avatar/gaze). Everything is time-weighted (agent: emulator frames of 50 ms, humans: duration(ms)
per frame) and measured on the agent's vision grid (5x5 squares of 40x52 px).
Chance = share of grid squares that overlap a category.

## 1. Share of time on object categories (grid)

In % of episode time, chance level in parentheses, × = ratio to chance
(mean of the per-episode/per-trial ratios).

| Category | Human first 5000 frames | Human full trial | Agent mean 3 seeds | Agent seed 1 (best) |
|---|---|---|---|---|
| player | 57.5 (9.4) · 6.1× | 60.1 (10.4) · 5.8× | 15.9 (9.3) · 1.7× | 20.1 (9.9) · 2.1× |
| enemy | 28.8 (13.6) · 2.1× | 32.3 (17.5) · 1.8× | 14.4 (11.2) · 1.3× | 15.9 (11.4) · 1.4× |
| item | 30.0 (7.8) · 3.9× | 37.5 (9.6) · 3.9× | 17.6 (12.3) · 1.4× | 19.3 (12.1) · 1.6× |
| reward | 45.8 (21.8) · 2.1× | 44.1 (19.5) · 2.3× | 24.7 (16.7) · 1.5× | 27.0 (15.8) · 1.7× |

- Game progress is comparable: humans have 6520 ± 880 points after 5000 frames, the agent
  ~6800 points in ~5000 frames (seed 1: 11440 in 6460 frames). The agent just dies earlier
  (all agent episodes end by game over after 1.2–7.5 min).
- Cutting at score 7000/12000 barely changes the human values (player 57.5/57.4 %).
  Human gaze behaviour is stable across levels; the gap is not a level effect.

### Best model only (seed 1, step 37.7 M)

Humans cut at score 12000 (≈ same part of the game as seed 1), range over trials
or episodes, 95 % bootstrap CI for seed 1:

| Category | Human (score ≤ 12000) | Seed 1 mean [95 % CI] | Seed 1 range |
|---|---|---|---|
| player | 57.4 [51.7–64.2] | 20.1 [18.4–21.9] | 14.7–25.9 |
| enemy | 31.7 [19.3–38.0] | 15.9 [14.6–17.1] | 11.4–18.6 |
| item | 31.4 [28.3–34.4] | 19.3 [17.3–21.0] | 12.2–24.1 |
| reward | 41.9 [36.0–46.7] | 27.0 [24.6–29.7] | 20.8–36.7 |

- For player and item the ranges do not overlap: the best agent episode is below the
  weakest human trial. 10 episodes are enough for this.
- Within the agent, looking at the avatar is strongly related to the score
  (r = 0.80 over all 30 episodes, r = 0.90 within seed 1). Enemy/item/reward: r = 0.2–0.3.
  However, the chance level for player also rises in the best episodes (up to 13.5 %).

## 2. Avatar positions and gaze behaviour (humans)

Agent values follow from the new eval run (`agent_play_stats.csv`). Local test of the script with
an almost untrained checkpoint: gaze entropy 4.5 bit (uniform), 4 switches/s.

| Measure | Human first 5000 frames | Human full trial |
|---|---|---|
| avatar found | 99.6 % | 98.3 % |
| avatar x mean ± sd (px, screen 160) | 76.3 ± 15.8 | 76.1 ± 18.0 |
| lane changes per min | 66 | 80 |
| horizontal speed (px/s) | 13.0 | 18.5 |
| time per lane 0–7 (top→bottom, %) | 5 · 6 · 11 · 28 · 27 · 10 · 8 · 5 | 6 · 8 · 12 · 23 · 24 · 12 · 9 · 6 |
| gaze entropy (bit, max 4.64) | 3.44 | 3.45 |
| square switches per s | 2.9 | 2.8 |
| mean dwell time (ms) | 353 | 365 |
| gaze in the avatar's square | 34.5 % | 34.7 % |
| gaze ≤ 1 square from the avatar | 84.6 % | 87.5 % |
| distance square centre – avatar (px) | 31.9 | 31.1 |
| row gaze − avatar (−2…+2, %) | 0.8 · 14.8 · 56.0 · 23.2 · 3.4 | 1.1 · 16.0 · 56.7 · 21.7 · 3.0 |
| column gaze − avatar (−2…+2, %) | 4.6 · 14.3 · 60.1 · 15.7 · 4.9 | 3.3 · 16.3 · 59.1 · 17.2 · 3.7 |

Gaze time per grid square, humans, first 5000 frames (%):

```
 0.1   1.6   1.1   0.5   0.1
 2.1   7.3  14.5   7.7   1.4
 2.5   8.3  26.3  10.6   2.9
 0.4   2.6   3.9   2.5   0.6
 0.0   0.2   2.5   0.4   0.1
```

Humans keep the avatar in the middle of the screen (lanes 3–4, x ≈ 76) and almost always look at it
or a neighbouring square, more often below than above it.

Notes on the comparison:
- The agent can switch only once per env step (≈ 5 frames, ≥ ~250 ms). Humans are counted per frame
  (50 ms), with the most frequent square of the frame.
- Agent avatar from RAM, human avatar from the colour detector. Both agree on random play
  (x ±1 px, y exact in 99.5 %).
- For the agent the chosen square is counted per step; the old squares during EMMA saccades
  are ignored here (unlike part 1).
