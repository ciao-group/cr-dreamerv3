import math
import os
from typing import Literal

import cv2
import elements
import numpy as np
import numpy.typing as npt
import polars as pl
from polars.dataframe import DataFrame

import embodied.core.vision as vision
from embodied.envs.cr_atari import CrAtari

WEIGHTS = CrAtari.WEIGHTS
DATASET_SCREEN_SIZE = (160, 210)
SCALE_GAZE_SCANPATH_IMAGE = 100

def eval_gaze(make_agent, make_logger, args, **kwargs):

    assert args.from_checkpoint
    assert args["task"].startswith("cr-atari")
    agent = make_agent()
    logger = make_logger()

    should_log = elements.when.Clock(args.log_every)
    policy = lambda *args: agent.policy(*args, mode="eval")
    size = args["cr-atari.size"]
    vision_square_size = args["cr-atari.vision_square_size"]
    vision_square_count = (
        size[0] // vision_square_size[0],
        size[1] // vision_square_size[1],
    )
    vision_mode = args["cr-atari.vision_mode"]
    gray = args["cr-atari.gray"]
    aggregate = args["cr-atari.aggregate"]
    pooling = args["cr-atari.pooling"]
    dataset = _load_dataset(
        "data/Atari-HEAD/asterix"
    )

    for entry in dataset:
        carry = agent.init_policy(1)
        obs = {
            "image": None,
            "is_first": np.array([True], dtype=bool),
            "is_last": np.array([False], dtype=bool),
            "is_terminal": np.array([False], dtype=bool),
            "reward": np.array([0], dtype=np.float32),
        }
        img_batch = [
            np.zeros(
                [DATASET_SCREEN_SIZE[1], DATASET_SCREEN_SIZE[0], 3], dtype=np.uint8
            )
            for _ in range(pooling)
        ]

        trial_data = entry["trial_data"]
        gaze_positions = np.empty((0, 2), int)
        gaze_heatmap_image = np.zeros(size + (1,), dtype=np.int32)
        gaze_scanpath_image = np.zeros((size[0]*SCALE_GAZE_SCANPATH_IMAGE, size[1]*SCALE_GAZE_SCANPATH_IMAGE, 1), dtype=np.uint8)
        distances_to_vision_square_center = np.empty((0,), dtype=np.int32)
        distances_to_vision_square_bound = np.empty((0,), dtype=np.int32)
        last_x = None
        last_y = None

        for i in range(len(trial_data)):
            human_gaze_position = (
                trial_data[i]["gazes"]
                .str.json_decode(pl.List(pl.List(pl.Float64)))
                .list.first()
            )[0]
            if human_gaze_position is None:
                continue
            human_gaze_position = (
                human_gaze_position[0] / DATASET_SCREEN_SIZE[0] * size[0],
                human_gaze_position[1] / DATASET_SCREEN_SIZE[1] * size[1],
            )

            img = entry["trial_frames"][i]
            assert isinstance(img, np.ndarray)
            assert img.dtype == np.uint8

            # Rotate batch to the right
            img_batch[1:] = img_batch[:-1]
            img_batch[0] = img

            # cv2.imshow("TEST", img)
            # cv2.waitKey(0)

            img = _preprocess_image_batch(
                img_batch=img_batch, size=size, gray=gray, aggregate=aggregate
            )

            human_gaze_position_1d = vision.convert_2d_gaze_position_to_1d_vision_square_position(
                gaze_position=human_gaze_position,
                vision_square_count=vision_square_count,
                vision_square_size=vision_square_size
            )

            img = vision.apply_vision_square(
                gaze_position=human_gaze_position_1d,
                image=img,
                mode=vision_mode,
                vision_square_count=vision_square_count,
                vision_square_size=vision_square_size,
                size=size,
            )

            cv2.imshow("TEST2", img)
            cv2.waitKey(0)
            obs["image"] = np.expand_dims(img, 0)
            ca, acts, out = policy(carry, obs, **kwargs)
            vision_square_position = acts["gaze_position"]

            x, y = vision.convert_1d_vision_square_position_to_2d_random(
                vision_square_position=vision_square_position,
                vision_square_count=vision_square_count,
                vision_square_size=vision_square_size,
            )
            gaze_positions = np.append(gaze_positions, [[x, y]])

            gaze_heatmap_image[y,x] += 1
            vision.add_scanpath_to_image(
                          x1 = last_x,
                          y1 = last_y,
                          x2= x,
                          y2= y,
                          number = i+1,
                          gaze_scanpath_image=gaze_scanpath_image,
                          scale_gaze_scanpath_image=SCALE_GAZE_SCANPATH_IMAGE
                      )
            last_x = x
            last_y = y

            distance_to_vision_square_center = (
                _calc_gaze_distance_to_vision_square_center(
                    vision_square_position=vision_square_position,
                    vision_square_size=vision_square_size,
                    gaze_position=human_gaze_position,
                    vision_square_count=vision_square_count,
                )
            )

            distance_to_vision_square_bound = (
                _calc_gaze_distance_to_vision_square_bounds(
                    vision_square_position=vision_square_position,
                    vision_square_size=vision_square_size,
                    gaze_position=human_gaze_position,
                    vision_square_count=vision_square_count,
                )
            )

            distances_to_vision_square_center = np.append(distances_to_vision_square_center, distance_to_vision_square_center)
            distances_to_vision_square_bound = np.append(distances_to_vision_square_bound, distance_to_vision_square_bound)
            logger.add(
                {
                    "distance_to_vision_square_center": float(
                        distance_to_vision_square_center
                    ),
                    "distance_to_vision_square_bound": float(
                        distance_to_vision_square_bound
                    )
                }
            )

            logger.step += 1

            if should_log(i):
                logger.write()

        logger.image("gaze_heatmap", vision.normalize_heatmap_image_0_to_255(gaze_heatmap_image=gaze_heatmap_image))
        logger.image("gaze_scanpath", gaze_scanpath_image)
        logger.add(
            {
                'gaze_positions': np.array2string(gaze_positions,threshold=len(gaze_positions), separator=", ").replace('\n', ''),
                "distances_to_vision_square_center": float(
                    np.mean(distances_to_vision_square_center)
                ),
                "distances_to_vision_square_bound": float(
                    np.mean(distances_to_vision_square_bound)
                )
            }
        )

    logger.close()


def _load_dataset(path: str):

    dataset: list[dict[Literal["trial_data", "trial_frames"], np.ndarray | DataFrame]] = []

    csv_file_names = list(
        filter(
            lambda filename: filename.endswith(".csv"),
            os.listdir(path),
        )
    )

    for csv_file_name in csv_file_names:
        csv_path = os.path.join(path, csv_file_name)
        subdir_name = os.path.splitext(csv_file_name)[0]

        # Load images and gaze positions
        trial_data = (
            pl.scan_csv(csv_path, null_values=["null"])
            .select(
                [
                    pl.col("frame_id"),
                    pl.col("gaze_positions").alias("gazes"),
                    pl.col("duration(ms)"),
                    pl.col("unclipped_reward").cum_sum().alias("cum_unclipped_reward"),
                ]
            )
            .with_row_index("trial_frame_id")
            .collect()
        )

        trial_frames = np.array(
            [
                cv2.imread(os.path.join(path, subdir_name, id + ".png"))
                for id in trial_data["frame_id"]
            ]
        )
        assert len(trial_frames) == len(trial_data)

        dataset.append({"trial_data": trial_data, "trial_frames": trial_frames})

    return dataset


def _preprocess_image_batch(
    img_batch: list[np.ndarray],
    size: list[int],
    gray: bool,
    aggregate: Literal["max", "mean"] = "max",
) -> np.ndarray:
    if aggregate == "max":
        img = np.amax(img_batch, 0)
    elif aggregate == "mean":
        img = np.mean(img_batch, 0).astype(np.uint8)

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    if gray:
        img = (img * WEIGHTS).sum(-1).astype(img.dtype)[:, :, None]

    return img


def _calc_gaze_distance_to_vision_square_center(
    vision_square_position: int,
    vision_square_size: tuple[int, int],
    gaze_position: tuple[int, int],
    vision_square_count: tuple[int, int],
) -> float:
    x, y = vision.convert_1d_vision_square_position_to_2d_center(
       vision_square_position=vision_square_position,
       vision_square_count=vision_square_count,
       vision_square_size=vision_square_size
    )

    return np.sqrt((x - gaze_position[0]) ** 2 + (y - gaze_position[1]) ** 2)


def _calc_gaze_distance_to_vision_square_bounds(
    vision_square_position: int,
    vision_square_size: tuple[int, int],
    gaze_position: tuple[int, int],
    vision_square_count: tuple[int, int],
) -> float:
    x, y = vision.convert_1d_vision_square_position_to_2d_center(
           vision_square_position=vision_square_position,
           vision_square_count=vision_square_count,
           vision_square_size=vision_square_size
        )

    dx = abs(gaze_position[0] - x)
    dy = abs(gaze_position[1] - y)

    if dx <= vision_square_size[0] // 2 and dy <= vision_square_size[1] // 2:
        return 0

    else:
        return math.hypot(
            max(dx - vision_square_size[0] // 2, 0),
            max(dy - vision_square_size[1] // 2, 0),
        )
