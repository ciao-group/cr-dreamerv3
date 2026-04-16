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

        atari_action = {"action": action["action"], "reset": action["reset"]}

        obs = super().step(atari_action)

        obs["image"] = vision.apply_vision_square(
            gaze_position=gaze_position,
            image=obs["image"],
            mode=self.vision_mode,
            vision_square_count=self.vision_square_count,
            vision_square_size=self.vision_square_size,
            size=self.size,
        )

        return obs

    def _reset(self):
        return super()._reset()
