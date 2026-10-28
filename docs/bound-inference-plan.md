# CR-Dreamer: Inferring Human Bounds from Atari-HEAD

Implementation and training plan for replicating human gameplay **and** visual
sampling in Atari with a computationally rational (CR) DreamerV3 agent.

## 1. Goal and ground rules

* The agent maximises task utility (game score minus internal costs) under
  explicit human bounds θ (perception, oculomotor control, motor control,
  memory). Human-likeness must **only** come from θ and the utility, never from
  fitting the network to human behaviour.
* Human data (Atari-HEAD) is therefore used for exactly two things:
  1. **Inference of θ** ("imitation learning" on the bound level): find the θ
     under which the θ-optimal policy assigns the highest likelihood to the
     human's actions and gaze. Network weights are never trained on human data.
  2. **Evaluation**: scanpath, saliency, timing and score metrics.
* Status quo: the agent reaches roughly human scores but its sampling behaviour
  is far from human scanpaths (cf. CR-Eyes, Lorenz et al. 2026).

## 2. Approach: amortised bound inference

Training one agent per candidate θ and comparing rollouts is too expensive for a
multi-dimensional θ (one Dreamer run ≈ 48 h on 8×A30). Instead:

1. **Bound-conditioned agent.** At every episode reset the environment samples
   θ ~ p(θ) from a prior box and exposes it (normalised to [0, 1]) as the
   observation key `bounds`. Dreamer encodes and reconstructs it, so the latent
   state carries θ through imagination and the actor learns π(a, g | h, θ) —
   the optimal policy for *every* θ in the prior, in a single run
   (amortised CR, cf. Moon et al. 2023, "Amortized inference with user
   simulations").
2. **Likelihood-based inference (teacher forcing).** Human trials are replayed
   through the agent: the observation at step t is the human frame foveated at
   the human's own gaze, the previous action fed to the RSSM is the human's
   action. For a batch of candidate θ we accumulate
   `log π(a_t^human, g_t^human, pause_t^human | h_t, θ)` and maximise over θ
   (grid → CMA-ES refinement). This yields θ̂ per subject and per trial plus
   a likelihood profile (identifiability / uncertainty).
3. **Closed-loop validation.** Roll out the agent with θ fixed to θ̂ and compare
   distributional statistics against humans (rollouts never align with the
   human's game states, so pairwise scanpath metrics only exist in replay).
4. **Bound-completeness check.** Agent-side bounds that cannot be amortised
   cheaply (memory capacity / noise) are compared as separate models by
   held-out likelihood and closed-loop metrics (model comparison). A residual
   gap after inference indicates missing bounds.

## 3. Bound inventory

| Group | Parameter | Where | Prior / values | Status |
|---|---|---|---|---|
| Oculomotor | `emma_K` (encoding time scale) | env | [0.003, 0.012] | existing, now parametrised |
| Oculomotor | `emma_t_prep` (saccade preparation, s) | env | [0.10, 0.20] | existing, now parametrised |
| Oculomotor | `effort_k` (saccade effort cost) | env/reward | [0, 0.004] | existing, now parametrised |
| Perception | `periphery_sigma` (peripheral blur, px) | env | [1, 8] | new continuous form of `periphery` modes |
| Motor | `motor_delay` (s, mean) | env | [0.04, 0.12] | existing, now parametrised |
| Motor | `reaction_rate` (1/s, LATER mean rate) | env | [3.5, 6.5] | existing, now parametrised |
| Utility | `time_cost` (reward per real second) | reward | [0, 2] | new, needed for pausing |
| Memory | `mem_noise` (σ of RSSM state noise per step) | agent | {0, 0.05, 0.1, 0.2} | **new**, model comparison |
| Memory | `mem_decay` (leak of RSSM state per step) | agent | {0, 0.01, 0.05} | **new**, model comparison |
| Memory | capacity (`deter` size) | agent | {1024, 4096} | existing hyperparameter |

**Pausing.** Atari-HEAD runs semi-frame-by-frame; ~4 % of the frames are
paused (> 55 ms) and pauses carry multiple fixations. A `pause` action is added:
the game does not advance, the fovea moves (costing EMMA time) and real time
passes at `time_cost`. Consecutive pauses are capped to avoid degenerate
behaviour. Paused human frames are split into fixations, each mapped to one
pause step, so pausing behaviour enters the likelihood.

## 4. Metrics

Implemented in `embodied/core/scanpath_metrics.py` (pure numpy, unit-tested).

* **Fixation detection** for raw Atari-HEAD gaze (I-DT, 1° dispersion,
  ≥ 50 ms), producing fixations with position and duration.
* **Saliency** (location/distribution based, as in Atari-HEAD): NSS, AUC-Judd,
  CC, KLD, SIM; Gaussian blur σ = 1 visual degree.
* **Paired scanpath similarity** (replay setting, same frames):
  normalised Levenshtein distance on grid-cell strings, ScanMatch-style
  Needleman–Wunsch score, DTW distance, MultiMatch (shape, direction, length,
  position, duration).
* **Distributional statistics** (closed loop, no alignment needed):
  saccade amplitude, saccade direction, fixation duration, pause duration,
  gaze–avatar distance, refixation rate, stationary gaze entropy and gaze
  transition entropy (Krejtz et al.); compared via Wasserstein-1 and
  Jensen–Shannon divergence.
* **Likelihood metrics**: per-step NLL of human gaze/action/pause, top-1/top-3
  gaze-cell accuracy.
* **Composite human-likeness score** for model selection: mean of normalised
  distributional distances plus game-score ratio to Atari-HEAD 15-min scores.

## 5. Work packages (each a semantic commit or small series)

1. **WP1 Atari-HEAD loader** (`embodied/core/atari_head.py`): parse trials
   (`.txt` + `.tar.bz2`), per-frame gaze samples, durations, actions; split
   paused frames into fixations; map to agent steps (gaze cell, action, pause);
   trial-level train/test split. `feat:`
2. **WP2 Scanpath metrics** + tests. `feat:` / `test:`
3. **WP3 Parametrised bounds**: `Bounds` spec with priors, per-episode
   sampling, `bounds` observation; EMMA/effort/motor/periphery read from θ.
   `feat:`
4. **WP4 Pause action** in `CrAtari`. `feat:`
5. **WP5 Memory bounds** in the RSSM (`mem_noise`, `mem_decay`). `feat:`
6. **WP6 Inference script** `infer_bounds` (teacher-forced log-likelihood,
   batched θ grid, CMA-ES refinement, per-trial/per-subject output) and a
   **parameter-recovery** mode on synthetic agent data. `feat:`
7. **WP7 Closed-loop evaluation** `eval_human_likeness` (rollouts with fixed θ,
   distributional metrics vs. human). `feat:`
8. **WP8 Cluster scripts and configs** (`cr-atari-amortized` config, SLURM
   scripts per phase). `chore:` / `ci:`
9. Cleanup of noisy per-step debug prints in the env. `refactor:`

## 6. Training plan (cluster, Asterix first, then Seaquest and H.E.R.O.)

| Phase | Runs | Purpose | Est. compute |
|---|---|---|---|
| 0 Smoke | 1 CPU run, ~2k steps | pipeline end-to-end (local) | minutes |
| 1 Amortised agent | 1 run × 3 seeds, 5e7 steps, `mem_noise=0` | π(a,g\|h,θ) over the prior | 3 × 48 h × 8 GPU |
| 2 Recovery check | infer θ on agent-generated data with known θ | identifiability per parameter; drop/fix unidentifiable ones | 1 GPU, hours |
| 3 Inference | infer θ̂ on Atari-HEAD Asterix (trial-wise split 10/5) | θ̂ per subject, likelihood profiles | 1 GPU, hours |
| 4 Memory comparison | amortised runs for `mem_noise` ∈ {0.05, 0.1, 0.2}, `mem_decay` ∈ {0.01, 0.05} | held-out NLL + closed-loop metrics → best memory model | 5 × 48 h × 8 GPU |
| 5 Closed loop | rollouts at θ̂ (best memory model) | scanpath/timing/score human-likeness | 1 GPU, hours |
| 6 Optional fine-tune | fixed θ̂, continue training | sharper policy at θ̂ | 1 × 24 h |
| 7 Other games | repeat 1–5 for Seaquest, H.E.R.O. | generalisation of θ̂ across games | as above |

Decision rules:
* A parameter with recovery correlation r < 0.5 (Phase 2) is fixed to its
  literature default and removed from the inference.
* If the best memory model improves held-out NLL by < 1 % and no closed-loop
  metric improves, memory bounds are dropped (parsimony).
* If closed-loop metrics at θ̂ remain far from humans, candidates for missing
  bounds are: eccentricity-dependent acuity instead of a hard fovea grid,
  finer gaze action resolution, and visual short-term memory decay with time
  (not steps).

## 7. Risks and open questions

* **Frame/step alignment**: one Atari-HEAD frame is treated as one ALE frame at
  20 Hz (the env's `TIME_PER_FRAME = 0.05`). Replay stride = env `repeat`.
  Must be verified against the dataset recording code.
* **Gaze discretisation**: human gaze is continuous, agent gaze is a 4×4 grid;
  likelihood and string metrics are computed on grid cells, saliency on
  continuous coordinates (agent fixation = cell centre + uniform jitter).
* **θ through the latent**: θ reaches the actor only via the RSSM state; the
  decoder reconstructs `bounds`, which keeps it in the latent. If the actor
  ignores θ (flat likelihood profiles), θ is concatenated to the actor input.
* **Teacher forcing vs. EMMA timing**: in replay the human's timing is given;
  the likelihood therefore measures how θ shapes the policy, not the timing
  model. Timing is validated separately (fixation/pause duration metrics).
