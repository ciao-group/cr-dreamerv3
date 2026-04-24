from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

Vision_Mode_Type = Literal[
    "foveated", "periphery", "periphery-cutoff", "exponential-fovea"
]


def apply_vision_square(
    gaze_position: int,
    image: np.ndarray,
    mode: Vision_Mode_Type,
    vision_square_count: tuple[int, int],
    vision_square_size: tuple[int, int],
    size: tuple[int, int],
) -> np.ndarray:
    x = gaze_position % vision_square_count[0] * vision_square_size[0]
    y = gaze_position // vision_square_count[0] * vision_square_size[1]

    if mode == "foveated":
        new_image = np.zeros(image.shape)

    elif mode == "periphery":
        new_image = gaussian_filter(image, 2)

    elif mode == "periphery-cutoff":
        new_image = np.zeros(image.shape)
        blur_image = gaussian_filter(image, 2)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x_blur = x + dx * vision_square_size[0]
                y_blur = y + dy * vision_square_size[1]

                if (
                    x_blur + vision_square_size[0] > size[0]
                    or y_blur + vision_square_size[1] > size[1]
                ):
                    continue

                new_image[
                    y_blur : y_blur + vision_square_size[1],
                    x_blur : x_blur + vision_square_size[0],
                ] = blur_image[
                    y_blur : y_blur + vision_square_size[1],
                    x_blur : x_blur + vision_square_size[0],
                ]

    elif mode == "exponential-fovea":
        raise NotImplementedError

    new_image[
        y : y + vision_square_size[1],
        x : x + vision_square_size[0],
    ] = image[
        y : y + vision_square_size[1],
        x : x + vision_square_size[0],
    ]

    return new_image


def calc_vision_square_count(
    size: tuple[int, int], vision_square_size: tuple[int, int]
) -> tuple[int, int]:
    return (
        size[0] // vision_square_size[0],
        size[1] // vision_square_size[1],
    )


def convert_1d_vision_square_position_to_2d(
    vision_square_position: int,
    vision_square_count: tuple[int, int],
    vision_square_size: tuple[int, int],
) -> tuple[int, int]:

    x = vision_square_position % vision_square_count[0] * vision_square_size[0]
    y = vision_square_position // vision_square_count[0] * vision_square_size[1]

    return x, y


def convert_1d_vision_square_position_to_2d_center(
    vision_square_position: int,
    vision_square_count: tuple[int, int],
    vision_square_size: tuple[int, int],
) -> tuple[int, int]:
    x, y = convert_1d_vision_square_position_to_2d(
        vision_square_position=vision_square_position,
        vision_square_count=vision_square_count,
        vision_square_size=vision_square_size,
    )

    return (x + vision_square_size[0] // 2, y + vision_square_size[1] // 2)


def convert_1d_vision_square_position_to_2d_random(
    vision_square_position: int,
    vision_square_count: tuple[int, int],
    vision_square_size: tuple[int, int],
) -> tuple[int, int]:
    x, y = convert_1d_vision_square_position_to_2d(
            vision_square_position=vision_square_position,
            vision_square_count=vision_square_count,
            vision_square_size=vision_square_size,
        )
    x += np.round(np.random.rand() * (vision_square_size[0] - 1)).astype("int32")
    y += np.round(np.random.rand() * (vision_square_size[1] - 1)).astype("int32")

    return x, y

def convert_2d_gaze_position_to_1d_vision_square_position(
    gaze_position: tuple[int, int],
    vision_square_count: tuple[int, int],
    vision_square_size: tuple[int, int],
) -> int:
    vision_square_position = int(
        (gaze_position[0] // vision_square_size[0])
        + (gaze_position[1] // vision_square_size[1] * vision_square_count[0])
    )
    return vision_square_position

def add_scanpath_to_image(
    x1: int | None,
    y1: int | None,
    x2: int,
    y2: int,
    number: int,
    gaze_scanpath_image: np.ndarray,
    scale_gaze_scanpath_image: int,

):
    x2_scaled = int(x2 * scale_gaze_scanpath_image)
    y2_scaled = int(y2 * scale_gaze_scanpath_image)

    if x1 is not None and y1 is not None:
        x1_scaled = int(x1 * scale_gaze_scanpath_image)
        y1_scaled = int(y1 * scale_gaze_scanpath_image)
        cv2.line(gaze_scanpath_image, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (255,), 1)

    cv2.putText(
        gaze_scanpath_image,
        str(number),
        (x2_scaled + 3, y2_scaled + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (125,),
        1,
        cv2.LINE_AA
    )

def normalize_heatmap_image_0_to_255(gaze_heatmap_image: np.ndarray) -> np.ndarray:
    return ((gaze_heatmap_image - gaze_heatmap_image.min()) * (1/(gaze_heatmap_image.max() - gaze_heatmap_image.min()) * 255)).astype('uint8')
