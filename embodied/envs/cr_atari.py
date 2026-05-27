import elements
import numpy as np
from scipy.ndimage import gaussian_filter

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
        vision_mode: vision.Vision_Mode_Type = "foveated",
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

        # Number of all possible horizontal and vertical vision squares
        self.vision_square_count = vision.calc_vision_square_count(
            self.size, self.vision_square_size
        )
        self.H, self.W = self.ale.getScreenDims()

        self.scaled_vision_square_size = (
                    self.vision_square_size[0] * self.W // self.size[0],
                    self.vision_square_size[1] * self.H // self.size[1]
                )
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
                            size=(self.W, self.H),
                        )

            self.prevlives = self.ale.lives()
            self.duration = 0
            self.done = False
            return self._obs(0.0, is_first=True)
        reward = 0.0
        terminal = False
        last = False
        assert 0 <= action['action'] < len(self.actionset), action['action']
        act = self.actionset[action['action']]
        for repeat in range(self.repeat):
            reward += self.ale.act(act)
            self.duration += 1
            if repeat >= self.repeat - self.pooling:
                self._render()
                self.buffers[0] = vision.apply_vision_square(
                            gaze_position=gaze_position,
                            image=self.buffers[0],
                            mode=self.vision_mode,
                            vision_square_count=self.vision_square_count,
                            vision_square_size=self.scaled_vision_square_size,
                            size=(self.W, self.H),
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
        self.buffers[0] = vision.apply_vision_square(
            gaze_position=vision.convert_2d_gaze_position_to_1d_vision_square_position((
                self.vision_square_count[0] * self.vision_square_size[0] // 2,
                self.vision_square_count[1] * self.vision_square_size[1] // 2
            ), self.vision_square_count, self.vision_square_size),
            image=self.buffers[0],
            mode=self.vision_mode,
            vision_square_count=self.vision_square_count,
            vision_square_size=self.scaled_vision_square_size,
            size=(self.W, self.H),
        )

        for i, dst in enumerate(self.buffers):
            if i > 0:
                np.copyto(dst, self.buffers[0])
