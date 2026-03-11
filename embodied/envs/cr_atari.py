from typing import Literal
import elements
import numpy as np
from scipy.ndimage import gaussian_filter 

from embodied.envs.atari import Atari

Vision_Mode_Type = Literal["foveated", "periphery", "exponential-fovea"]

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
        vision_mode: Vision_Mode_Type="foveated"
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
        self.vision_mode: Vision_Mode_Type = vision_mode

        # Number of all possible horizontal and vertical vision squares
        self.vision_squares = (
            self.size[0] // self.vision_square_size[0],
            self.size[1] // self.vision_square_size[1],
        )

    @property
    def act_space(self):

        return {
            "action": elements.Space(np.int32, (), 0, len(self.actionset)),
            "reset": elements.Space(bool),
            # "pause": elements.Space(np.int32, (), 0, 2),  # 1 to pause
            "gaze_position": elements.Space(
                np.int32, (), 0, self.vision_squares[0] * self.vision_squares[1]
            ),
        }

    def step(self, action):

        # TODO implement pause
        # if action["pause"] == 1:
        #     pass
        gaze_position = action["gaze_position"]

        atari_action = {"action": action["action"], "reset": action["reset"]}

        obs = super().step(atari_action)

        obs["image"] = self._apply_vision_square(gaze_position, obs["image"], self.vision_mode)

        return obs

    def _reset(self):
        return super()._reset()

    def _apply_vision_square(self, gaze_position: int, image: np.ndarray, mode: Vision_Mode_Type):

        x = gaze_position % self.vision_squares[0]
        y = gaze_position // self.vision_squares[0]
        
        if mode == "foveated":
        
            new_image = np.zeros(image.shape)
        
        elif mode == "periphery":
            
            new_image = gaussian_filter(image, 2)
        
        elif mode == "exponential-fovea":
            raise NotImplementedError
        
        new_image[
            y * self.vision_square_size[1] : (y + 1) * self.vision_square_size[1],
            x * self.vision_square_size[0] : (x + 1) * self.vision_square_size[0],
        ] = image[
            y * self.vision_square_size[1] : (y + 1) * self.vision_square_size[1],
            x * self.vision_square_size[0] : (x + 1) * self.vision_square_size[0],
        ]

        return new_image
