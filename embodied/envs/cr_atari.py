import elements
import numpy as np

import embodied.core.objects as objects
import embodied.core.vision as vision
from embodied.envs.atari import Atari

class CrAtari(Atari):
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
        effort_kstatic: float = 0.001,
        saccade_cost: float = 0.0,
        frames_obs: bool = False,
    ):
        """eye_movement_effort: subtract the static effort of holding the gaze
        away from the screen centre every step, effort_kstatic * eccentricity^2
        (degrees). saccade_cost: subtract saccade_cost * amplitude (degrees)
        for every gaze shift, so that long saccades cost more; 0 disables it.
        Both are logged as log/effort_penalty and log/saccade_penalty.

        frames_obs: add the observation 'frames', the number of emulator
        frames the step lasted (0 on reset). With EMMA a step lasts longer
        when the gaze moves; the agent's frame_discount uses it to discount
        per frame instead of per step."""
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
        self.VISUAL_DEGREES_PER_PIXEL = np.array(VISUAL_DEGREE_SCREEN_SIZE) / np.array((self.W, self.H))
        self.passed_time = 0
        self.prev_gaze_position = None
        self.motor_action_delay = motor_action_delay
        self.reaction_time_delay = reaction_time_delay
        self._game_score = 0
        self._eye_movement_effort = eye_movement_effort
        self._effort_kstatic = effort_kstatic
        self._saccade_cost = saccade_cost
        self._effort_penalty = 0.0
        self._saccade_penalty = 0.0
        self._saccade_amplitude = 0.0

        # Vision squares on the ALE screen, indexed by gaze position, for the
        # looked-at metrics of game objects (only for games with an object
        # detector).
        self._track_objects = self.name in objects.GAMES
        self._object_frames = self._empty_object_frames()
        self._vision_squares = [
            (*vision.convert_1d_vision_square_position_to_2d(
                k, self.vision_square_count, (self.W, self.H), self.scaled_vision_square_size),
             *self.scaled_vision_square_size)
            for k in range(self.vision_square_count[0] * self.vision_square_count[1])
        ]
        self._frames_obs = frames_obs
        self._step_frames = 0

    @property
    def obs_space(self):
        frames = {'frames': elements.Space(np.float32)} if self._frames_obs else {}
        return {
            **frames,
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
            'log/effort_penalty': elements.Space(np.float32),
            'log/saccade_penalty': elements.Space(np.float32),
            'log/saccade_amplitude': elements.Space(np.float32),
            **self._object_log_space(),
        }

    def _object_log_space(self):
        """Time spent on each object category (player, enemy, item, reward),
        in emulator frames of 50 ms, summed over the frames of one step:

        frames: frames emulated in this step.
        looked_at/<c>: frames in which the shown vision square overlapped an
          object of the category.
        on_screen/<c>: frames in which such an object was on screen.
        chance/<c>: frames weighted by the share of all vision squares that
          overlapped one, i.e. looked_at of a gaze drawn uniformly from the grid.

        Share of time in an episode: looked_at/<c>/sum / frames/sum (likewise
        chance); given the category was visible: looked_at/<c>/sum /
        on_screen/<c>/sum.
        """
        if not self._track_objects:
            return {}
        space = {'log/frames': elements.Space(np.float32)}
        for key in self._object_frames:
            if key != 'frames':
                space[f'log/{key}'] = elements.Space(np.float32)
        return space

    @staticmethod
    def _empty_object_frames():
        counts = {'frames': 0.0}
        for category in objects.CATEGORIES:
            for kind in ('looked_at', 'on_screen', 'chance'):
                counts[f'{kind}/{category}'] = 0.0
        return counts

    def _count_object_frame(self, gaze):
        """Adds the current emulator frame, seen with the vision square at
        `gaze`, to the object time of this step."""
        if not self._track_objects:
            return
        found = objects.detect(self.ale.getScreenRGB(), self.name)
        hits = objects.square_hits(found, self._vision_squares)
        counts = self._object_frames
        counts['frames'] += 1
        for category, row in hits.items():
            counts[f'looked_at/{category}'] += float(gaze is not None and row[gaze])
            counts[f'on_screen/{category}'] += float(any(o.category == category for o in found))
            counts[f'chance/{category}'] += float(row.mean())

    def _object_logs(self):
        if not self._track_objects:
            return {}
        return {f'log/{k}': np.float32(v) for k, v in self._object_frames.items()}

    @property
    def act_space(self):

        return {
            "action": elements.Space(np.int32, (), 0, len(self.actionset)),
            "reset": elements.Space(bool),
            # "pause": elements.Space(np.int32, (), 0, 2),  # 1 to pause
            "gaze_position": elements.Space(
                np.int32,
                (),
                0,
                self.vision_square_count[0] * self.vision_square_count[1],
            ),
        }

    def distance_avatar_vision_square_scaled(self) -> float:
        return vision.distance_to_vision_square_scaled(gaze_position=self.prev_gaze_position,
                                                vision_square_count=self.vision_square_count,
                                                vision_square_size=self.scaled_vision_square_size, bbox_to_check=(
            self.character_position[0], self.character_position[1], 8, 11), x_scale=self.size[0] / self.W, y_scale=self.size[1] / self.H, screen_size=(self.W, self.H))

    def distance_avatar_vision_square(self) -> float:
        return vision.distance_to_vision_square(gaze_position=self.prev_gaze_position, vision_square_count=self.vision_square_count, vision_square_size=self.scaled_vision_square_size, bbox_to_check=(self.character_position[0], self.character_position[1], 8, 11), screen_size=(self.W, self.H))

    def avatar_is_observed(self) -> bool:
        if self.prev_gaze_position is not None:
            return vision.check_if_bbox_intersects_vision_square(gaze_position=self.prev_gaze_position, vision_square_count=self.vision_square_count, vision_square_size=self.scaled_vision_square_size, bbox_to_check=(self.character_position[0], self.character_position[1], 8, 11), screen_size=(self.W, self.H))
        return False

    def avatar_is_observed_scaled(self) -> bool:
        if self.prev_gaze_position is not None:
            return vision.check_if_bbox_intersects_vision_square_scaled(gaze_position=self.prev_gaze_position,vision_square_count=self.vision_square_count, vision_square_size=self.scaled_vision_square_size, bbox_to_check=(self.character_position[0], self.character_position[1], 8, 11), scale_x=self.size[0] / self.W, scale_y=self.size[1] / self.H, screen_size=(self.W, self.H))
        return False

    def step(self, action):

        # TODO implement pause
        # if action["pause"] == 1:
        #     pass
        gaze_position = action["gaze_position"]


        action = {"action": action["action"], "reset": action["reset"]}

        if action['reset'] or self.done:

            self._reset()
            self.buffers[0] = vision.apply_vision_square(
                            gaze_position=gaze_position,
                            image=self.buffers[0],
                            mode=self.vision_mode,
                            vision_square_count=self.vision_square_count,
                            vision_square_size=self.scaled_vision_square_size,
                            screen_size=(self.W, self.H),
                        )

            self.prevlives = self.ale.lives()
            self.duration = 0
            self.passed_time = 0
            self.done = False
            self._object_frames = self._empty_object_frames()
            self._effort_penalty = self._saccade_penalty = self._saccade_amplitude = 0.0
            self._step_frames = 0
            return self._obs(0.0, is_first=True)

        all_delay_times = {"default": 0.0}
        emma_frames = 0
        effort_penalty = 0
        if self.vision_model == "EMMA":
            emma_time = vision.calc_EMMA_time_from_1d_vision_square_positions(
                prev_position=self.prev_gaze_position,
                next_position=gaze_position,
                vision_square_count=self.vision_square_count,
                vision_square_size=self.scaled_vision_square_size,
                screen_size=(self.W, self.H),
                visual_degrees_per_pixel=self.VISUAL_DEGREES_PER_PIXEL
            )
            emma_frames = emma_time // self.TIME_PER_FRAME
            all_delay_times["EMMA"] = emma_time

        if self._eye_movement_effort:
            effort_penalty = vision.calc_effort(position=gaze_position,vision_square_count=self.vision_square_count, vision_square_size=self.scaled_vision_square_size, screen_size=(self.W, self.H), visual_degrees_per_pixel=self.VISUAL_DEGREES_PER_PIXEL, kstatic=self._effort_kstatic)
        amplitude = 0.0
        if self.prev_gaze_position is not None and self.prev_gaze_position != gaze_position:
            amplitude = vision.calc_gaze_shift_eccentricity_from_1d_vision_square_positions(
                prev_position=self.prev_gaze_position, next_position=gaze_position,
                vision_square_count=self.vision_square_count,
                vision_square_size=self.scaled_vision_square_size, screen_size=(self.W, self.H),
                visual_degrees_per_pixel=self.VISUAL_DEGREES_PER_PIXEL)
        saccade_penalty = self._saccade_cost * amplitude

        motor_action_frames = 0
        if self.motor_action_delay:
            motor_action_time = np.round(np.random.normal(70, 12.8398, None) / 1000, 2)
            total_delay_time = motor_action_time

            if self.reaction_time_delay:
                # Calculate reaction time based on a reciprocal normal distribution (LATER model).
                # Mean rate of 5.0 corresponds to ~200ms peak latency.
                reciprocal_rate = np.random.normal(loc=5.0, scale=0.8)

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

        # ensure that the motor action is executed, when repeating < motor_action_frames
        repeating = max(repeating, int(motor_action_frames) + self.repeat)
        reward = 0.0
        terminal = False
        last = False
        assert 0 <= action['action'] < len(self.actionset), action['action']
        act = self.actionset[action['action']]
        no_act = self.ACTION_MEANING.index('NOOP')
        self._object_frames = self._empty_object_frames()
        start = self.duration
        for repeat in range(repeating):

            if (motor_action_frames <= repeat):
                reward += self.ale.act(act)
            else:
                reward += self.ale.act(no_act)
            self.duration += 1
            self.passed_time += self.TIME_PER_FRAME
            # Until the saccade has landed, the old vision square is shown.
            self._count_object_frame(
                gaze_position if repeat >= (emma_frames - 1) or self.prev_gaze_position is None else self.prev_gaze_position)
            if repeat >= repeating - self.pooling:
                self._render()
                self.buffers[0] = vision.apply_vision_square(
                    gaze_position=gaze_position if repeat >= (emma_frames - 1) or self.prev_gaze_position is None else self.prev_gaze_position,
                            image=self.buffers[0],
                            mode=self.vision_mode,
                            vision_square_count=self.vision_square_count,
                            vision_square_size=self.scaled_vision_square_size,
                            screen_size=(self.W, self.H),
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
        self._step_frames = self.duration - start
        # TODO: Does swapping these two lines breaks anything?
        # TODO: We need the current gaze position for the vision square metric calculations
        self.prev_gaze_position = gaze_position

        # save for logging raw ALE reward
        self._game_score = reward

        # apply effort and saccade cost
        self._effort_penalty = float(effort_penalty)
        self._saccade_penalty = float(saccade_penalty)
        self._saccade_amplitude = float(amplitude)
        reward -= effort_penalty + saccade_penalty
        obs = self._obs(reward, is_last=last, is_terminal=terminal)
        return obs

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
        initial_gaze_position = vision.center_vision_square_position(self.vision_square_count)
        self.buffers[0] = vision.apply_vision_square(
            gaze_position=initial_gaze_position,
            image=self.buffers[0],
            mode=self.vision_mode,
            vision_square_count=self.vision_square_count,
            vision_square_size=self.scaled_vision_square_size,
            screen_size=(self.W, self.H),
        )
        self.prev_gaze_position = initial_gaze_position
        for i, dst in enumerate(self.buffers):
            if i > 0:
                np.copyto(dst, self.buffers[0])

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
        obs["log/effort_penalty"] = np.float32(self._effort_penalty)
        obs["log/saccade_penalty"] = np.float32(self._saccade_penalty)
        obs["log/saccade_amplitude"] = np.float32(self._saccade_amplitude)
        obs.update(self._object_logs())
        if self._frames_obs:
            obs["frames"] = np.float32(self._step_frames)

        return obs
