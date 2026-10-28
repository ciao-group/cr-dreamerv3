import elements
import numpy as np

import embodied.core.vision as vision
from embodied.core.bounds import Bounds
from embodied.envs.atari import Atari

class CrAtari(Atari):
    """Atari with human bounds: foveated vision, EMMA saccade timing, eye
    effort, motor and reaction delays, and optional pausing as in the
    semi-frame-by-frame mode of Atari-HEAD.

    The numeric bound parameters live in `bounds` (see embodied.core.bounds).
    Bounds named in `bounds_sample` are resampled at every episode start and
    exposed to the agent as the observation `bounds`.
    """

    def __init__(
        self,
        name,
        repeat=4,
        size=(84, 84),
        gray=True,
        noops=0,
        lives="unused",
        sticky=True,
        actions="all",
        length=108000,
        pooling=2,
        aggregate="max",
        resize="pillow",
        autostart=False,
        clip_reward=False,
        seed=None,
        vision_square_size=(12, 12),
        vision_square_count: tuple[int, int] | None = None,
        vision_mode: vision.Vision_Mode_Type = "foveated",
        vision_model: str | None = None,
        motor_action_delay: bool = False,
        reaction_time_delay: bool = False,
        eye_movement_effort: bool = False,
        bounds: dict | None = None,
        bounds_sample: str = "",
        pause: bool = False,
        max_pauses: int = 10,
    ):
        super().__init__(
            name,
            repeat,
            size,
            gray,
            noops,
            lives,
            sticky,
            actions,
            length,
            pooling,
            aggregate,
            resize,
            autostart,
            clip_reward,
            seed,
        )

        self.vision_square_size = vision_square_size
        self.vision_mode: vision.Vision_Mode_Type = vision_mode
        self.vision_model = vision_model

        self.vision_square_count = vision_square_count if vision_square_count is not None else vision.calc_vision_square_count(
            self.size, self.vision_square_size
        )

        self.H, self.W = self.ale.getScreenDims()

        assert self.vision_square_count[0] * self.vision_square_size[0] >= self.size[0], f"vision_square_count {vision_square_count[0]} does not match dimensions self.vision_square_size: {self.vision_square_size[0]}, H: {self.H}, W: {self.W}"

        self.scaled_vision_square_size = (
                    self.vision_square_size[0] * self.W // self.size[0],
                    self.vision_square_size[1] * self.H // self.size[1]
                )

        self.TIME_PER_FRAME = 0.05 # Represents 20 Hz
        VISUAL_DEGREE_SCREEN_SIZE = (44.6, 28.5) # (W,H)
        # Gaze geometry is computed on the native ALE screen (W, H).
        self.VISUAL_DEGREES_PER_PIXEL = np.array(VISUAL_DEGREE_SCREEN_SIZE) / np.array((self.W, self.H))
        self.passed_time = 0
        self.prev_gaze_position = None
        self.motor_action_delay = motor_action_delay
        self.reaction_time_delay = reaction_time_delay
        self._game_score = 0
        self._eye_movement_effort = eye_movement_effort
        self.bounds = Bounds.from_config(dict(bounds or {}), bounds_sample, seed)
        self.pause = pause
        self.max_pauses = max_pauses
        self._pauses = 0
        self._paused = False
        self._step_time = 0.0
        # Unfoveated copies of the pooled screens, to replay recorded episodes.
        self.raw_buffers = [np.zeros_like(b) for b in self.buffers]

    @property
    def obs_space(self):
        spaces = {
            'image': elements.Space(np.uint8, (*self.size, 1 if self.gray else 3)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
            'log/player_position_x': elements.Space(np.uint8),
            'log/player_position_y': elements.Space(np.uint8),
            'log/player_position_x_raw': elements.Space(np.uint8),
            'log/player_position_y_raw': elements.Space(np.uint8),
            'log/player_bb_w': elements.Space(np.uint8),
            'log/player_bb_h': elements.Space(np.uint8),
            'log/avatar_distance_vision_square': elements.Space(np.float32),
            'log/avatar_distance_vision_square_scaled': elements.Space(np.float32),
            'log/avatar_is_in_vision_square_scaled': elements.Space(bool),
            'log/game_score': elements.Space(np.float32),
            'log/gaze_position': elements.Space(np.int32),
            'log/paused': elements.Space(bool),
            'log/step_time': elements.Space(np.float32),
        }
        if self.bounds.names:
            spaces['bounds'] = elements.Space(
                np.float32, (len(self.bounds.names),), 0.0, 1.0)
            for name in self.bounds.names:
                spaces[f'log/bound_{name}'] = elements.Space(np.float32)
        return spaces

    @property
    def act_space(self):
        spaces = {
            "action": elements.Space(np.int32, (), 0, len(self.actionset)),
            "reset": elements.Space(bool),
            "gaze_position": elements.Space(
                np.int32,
                (),
                0,
                self.vision_square_count[0] * self.vision_square_count[1],
            ),
        }
        if self.pause:
            spaces["pause"] = elements.Space(np.int32, (), 0, 2)
        return spaces

    def distance_avatar_vision_square_scaled(self) -> float:
        return vision.distance_to_vision_square_scaled(gaze_position=self.prev_gaze_position,
                                                vision_square_count=self.vision_square_count,
                                                vision_square_size=self.scaled_vision_square_size, bbox_to_check=(
            self.character_position[0], self.character_position[1], 8, 11), x_scale=self.size[0] / self.W, y_scale=self.size[1] / self.H, screen_size=(self.W, self.H))

    def distance_avatar_vision_square(self) -> float:
        return vision.distance_to_vision_square(gaze_position=self.prev_gaze_position, vision_square_count=self.vision_square_count, vision_square_size=self.scaled_vision_square_size, bbox_to_check=(self.character_position[0], self.character_position[1], 8, 11), screen_size=(self.W, self.H))

    def avatar_is_observed(self) -> bool:
        if self.prev_gaze_position:
            return vision.check_if_bbox_intersects_vision_square(gaze_position=self.prev_gaze_position, vision_square_count=self.vision_square_count, vision_square_size=self.scaled_vision_square_size, bbox_to_check=(self.character_position[0], self.character_position[1], 8, 11), screen_size=(self.W, self.H))
        return False

    def avatar_is_observed_scaled(self) -> bool:
        if self.prev_gaze_position:
            return vision.check_if_bbox_intersects_vision_square_scaled(gaze_position=self.prev_gaze_position,vision_square_count=self.vision_square_count, vision_square_size=self.scaled_vision_square_size, bbox_to_check=(self.character_position[0], self.character_position[1], 8, 11), scale_x=self.size[0] / self.W, scale_y=self.size[1] / self.H, screen_size=(self.W, self.H))

    def _render(self, reset=False):
        super()._render(reset)
        self.raw_buffers.insert(0, self.raw_buffers.pop())
        np.copyto(self.raw_buffers[0], self.buffers[0])

    def foveate(self, image: np.ndarray, gaze_position: int) -> np.ndarray:
        return vision.apply_vision_square(
            gaze_position=gaze_position,
            image=image,
            mode=self.vision_mode,
            vision_square_count=self.vision_square_count,
            vision_square_size=self.scaled_vision_square_size,
            screen_size=(self.W, self.H),
            periphery_sigma=self.bounds["periphery_sigma"],
        )

    def step(self, action):
        gaze_position = action["gaze_position"]
        wants_pause = self.pause and bool(action.get("pause", 0))

        action = {"action": action["action"], "reset": action["reset"]}

        if action['reset'] or self.done:
            # Fovea starts at the screen center, see _reset().
            self.bounds.resample()
            self._reset()
            self.prevlives = self.ale.lives()
            self.duration = 0
            self.passed_time = 0
            self.done = False
            self._pauses = 0
            self._paused = False
            self._step_time = 0.0
            self._game_score = 0
            return self._obs(0.0, is_first=True)

        all_delay_times = {"default": 0.0}
        emma_time = 0.0
        emma_frames = 0
        effort_penalty = 0
        if self.vision_model == "EMMA":
            emma_time = vision.calc_EMMA_time_from_1d_vision_square_positions(
                prev_position=self.prev_gaze_position,
                next_position=gaze_position,
                vision_square_count=self.vision_square_count,
                vision_square_size=self.scaled_vision_square_size,
                screen_size=(self.W, self.H),
                visual_degrees_per_pixel=self.VISUAL_DEGREES_PER_PIXEL,
                K=self.bounds["emma_K"],
                k=self.bounds["emma_k"],
                t_prep=self.bounds["emma_t_prep"],
            )
            emma_frames = emma_time // self.TIME_PER_FRAME
            all_delay_times["EMMA"] = emma_time

            effort_penalty = vision.calc_effort(
                position=gaze_position,
                vision_square_count=self.vision_square_count,
                vision_square_size=self.scaled_vision_square_size,
                screen_size=(self.W, self.H),
                visual_degrees_per_pixel=self.VISUAL_DEGREES_PER_PIXEL,
                kstatic=self.bounds["effort_k"],
            )

        if wants_pause and self._pauses < self.max_pauses:
            return self._pause_step(gaze_position, emma_time, effort_penalty)
        self._pauses = 0
        self._paused = False

        motor_action_frames = 0
        if self.motor_action_delay:
            motor_action_time = np.round(np.random.normal(1000 * self.bounds["motor_delay"], 12.8398, None) / 1000, 2)
            total_delay_time = motor_action_time

            if self.reaction_time_delay:
                # Calculate reaction time based on a reciprocal normal distribution (LATER model).
                # Mean rate of 5.0 corresponds to ~200ms peak latency.
                reciprocal_rate = np.random.normal(loc=self.bounds["reaction_rate"], scale=0.8)

                # Clip the rate to avoid division by zero or negative values.
                # A minimum rate of 0.5 caps the maximum latency at 2.0 seconds.
                reciprocal_rate = max(reciprocal_rate, 0.5)
                reaction_time = np.round(1.0 / reciprocal_rate, 2)

                # Add reaction time and motor delay to get the total delay
                total_delay_time = motor_action_time + reaction_time

            motor_action_frames = total_delay_time // self.TIME_PER_FRAME
            all_delay_times["motor_action"] = total_delay_time

        repeating = self.repeat
        max_delay_time = max(list(all_delay_times.values()))
        if max_delay_time > self.repeat * self.TIME_PER_FRAME:
            repeating = int(max_delay_time // self.TIME_PER_FRAME)

        if max_delay_time % self.TIME_PER_FRAME + self.passed_time % self.TIME_PER_FRAME >= self.TIME_PER_FRAME:
            repeating += 1

        self.passed_time += max_delay_time % self.TIME_PER_FRAME

        reward = 0.0
        terminal = False
        last = False
        assert 0 <= action['action'] < len(self.actionset), action['action']
        act = self.actionset[action['action']]
        no_act = self.ACTION_MEANING.index('NOOP')
        frames = 0
        for repeat in range(repeating):

            if (motor_action_frames <= repeat < self.repeat + motor_action_frames):
                reward += self.ale.act(act)
            else:
                reward += self.ale.act(no_act)
            frames += 1
            self.duration += 1
            self.passed_time += self.TIME_PER_FRAME
            if repeat >= repeating - self.pooling:
                self._render()
                self.buffers[0] = self.foveate(
                    self.buffers[0],
                    gaze_position if repeat >= (emma_frames - 1) or self.prev_gaze_position is None else self.prev_gaze_position,
                )

            if self.ale.game_over():
                terminal = True
                last = True
            if self.duration >= self.length:
                last = True
            lives = self.ale.lives()
            if self.lives == 'discount' and 0 < lives < self.prevlives:
                terminal = True
            if self.lives == 'reset' and 0 < lives < self.prevlives:
                terminal = True
                last = True
            self.prevlives = lives
            if terminal or last:
                break
        self.done = last
        self.prev_gaze_position = gaze_position

        # save for logging raw ALE reward
        self._game_score = reward
        self._step_time = frames * self.TIME_PER_FRAME

        # apply effort
        if self._eye_movement_effort:
            reward -= effort_penalty
        reward -= self.bounds["time_cost"] * self._step_time
        obs = self._obs(reward, is_last=last, is_terminal=terminal)
        return obs

    def _pause_step(self, gaze_position, emma_time, effort_penalty):
        """The game stays frozen while the eyes move to `gaze_position`.

        The step lasts as long as the gaze shift (at least one frame) and
        costs real time, like pausing in Atari-HEAD.
        """
        self._pauses += 1
        self._paused = True
        self._step_time = max(float(emma_time), self.TIME_PER_FRAME)
        self.passed_time += self._step_time
        screen = np.zeros_like(self.buffers[0])
        self.ale.getScreenRGB(screen)
        for buffer, raw in zip(self.buffers, self.raw_buffers):
            np.copyto(raw, screen)
            np.copyto(buffer, self.foveate(screen, gaze_position))
        self.prev_gaze_position = gaze_position
        self._game_score = 0.0
        reward = -self.bounds["time_cost"] * self._step_time
        if self._eye_movement_effort:
            reward -= effort_penalty
        return self._obs(reward)

    def _reset(self):
        with self.LOCK:
            self.ale.reset_game()
        for _ in range(self.rng.integers(self.noops + 1)):
            self.ale.act(self.ACTION_MEANING.index('NOOP'))
            if self.ale.game_over():
                with self.LOCK:
                    self.ale.reset_game()
        if self.autostart and self.ACTION_MEANING.index('FIRE') in self.actionset:
            self.ale.act(self.ACTION_MEANING.index('FIRE'))
            if self.ale.game_over():
                with self.LOCK:
                    self.ale.reset_game()
            self.ale.act(self.ACTION_MEANING.index('UP'))
            if self.ale.game_over():
                with self.LOCK:
                    self.ale.reset_game()
        self._render()
        initial_gaze_position = self.center_gaze_position()
        self.buffers[0] = self.foveate(self.buffers[0], initial_gaze_position)
        self.prev_gaze_position = initial_gaze_position
        for i, dst in enumerate(self.buffers):
            if i > 0:
                np.copyto(dst, self.buffers[0])
                np.copyto(self.raw_buffers[i], self.raw_buffers[0])

    def center_gaze_position(self) -> int:
        return vision.convert_2d_gaze_position_to_1d_vision_square_position(
                    (
                        self.vision_square_count[0] * self.vision_square_size[0] // 2,
                        self.vision_square_count[1] * self.vision_square_size[1] // 2
                    ),
                    vision_square_count= self.vision_square_count,
                    screen_size=self.size,
        )

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        character_x_raw = self.character_position[0]
        character_y_raw = self.character_position[1]
        char_x, char_y, char_w, char_h = self._scale_bounding_box(character_x_raw, character_y_raw)

        obs = super()._obs(reward=reward, is_first=is_first, is_last=is_last, is_terminal=is_terminal)
        obs["log/player_position_x"] = char_x
        obs["log/player_position_y"] = char_y
        obs["log/player_position_x_raw"] = character_x_raw
        obs["log/player_position_y_raw"] = character_y_raw
        obs["log/player_bb_w"]= char_w
        obs["log/player_bb_h"]= char_h
        obs["log/avatar_distance_vision_square"]= self.distance_avatar_vision_square()
        obs["log/avatar_distance_vision_square_scaled"] = self.distance_avatar_vision_square_scaled()
        obs["log/avatar_is_in_vision_square_scaled"] = self.avatar_is_observed_scaled()
        obs["log/game_score"] = self._game_score
        obs["log/gaze_position"] = np.int32(self.prev_gaze_position)
        obs["log/paused"] = self._paused
        obs["log/step_time"] = np.float32(self._step_time)
        if self.bounds.names:
            obs["bounds"] = self.bounds.vector()
            for name in self.bounds.names:
                obs[f"log/bound_{name}"] = np.float32(self.bounds[name])

        return obs
