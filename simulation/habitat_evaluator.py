"""Self-contained Habitat setup shared by LookStep online evaluators."""

import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import habitat
from habitat import Env
from habitat.config.default import get_agent_config, get_config as get_habitat_config

if __package__:
    from . import habitat_measures as _habitat_measures  # noqa: F401
else:
    from simulation import habitat_measures as _habitat_measures  # type: ignore # noqa: F401


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_under_data_root(raw_path: str, data_root: Path) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    parts = path.parts[1:] if path.parts and path.parts[0] == "data" else path.parts
    return str(data_root.joinpath(*parts))


class VLNEvaluator:
    """Load a Habitat config and provide common image preprocessing helpers."""

    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 1,
        output_path: str = None,
        model: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.epoch = epoch
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.config = get_habitat_config(self.config_path)
        self.save_video_ratio = args.save_video_ratio

        data_root = Path(args.data_root).expanduser().resolve()
        with habitat.config.read_write(self.config):
            self.config.habitat.seed = args.seed
            self.config.habitat.dataset.split = self.split
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                args.local_rank if args.sim_gpu_id is None else args.sim_gpu_id
            )
            current_scenes_dir = str(self.config.habitat.dataset.scenes_dir)
            current_data_path = str(self.config.habitat.dataset.data_path)
            self.config.habitat.dataset.scenes_dir = (
                str(Path(args.scenes_dir).expanduser().resolve())
                if args.scenes_dir
                else _resolve_under_data_root(current_scenes_dir, data_root)
            )
            self.config.habitat.dataset.data_path = (
                args.dataset_path
                if args.dataset_path
                else _resolve_under_data_root(current_data_path, data_root)
            )

            if self.save_video:
                from habitat.config.default_structured_configs import (
                    CollisionsMeasurementConfig,
                    FogOfWarConfig,
                    TopDownMapMeasurementConfig,
                )

                self.config.habitat.task.measurements.update(
                    {
                        "top_down_map": TopDownMapMeasurementConfig(
                            map_padding=3,
                            map_resolution=1024,
                            draw_source=True,
                            draw_border=True,
                            draw_shortest_path=True,
                            draw_view_points=True,
                            draw_goal_positions=True,
                            draw_goal_aabbs=True,
                            fog_of_war=FogOfWarConfig(
                                draw=True,
                                visibility_dist=5.0,
                                fov=90,
                            ),
                        ),
                        "collisions": CollisionsMeasurementConfig(),
                    }
                )

        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = (
            self.config.habitat.simulator.agents.main_agent.sim_sensors
        )
        self.model = model
        self.image_processor = model.processor
        self.tokenizer = model.tokenizer
        self.actions2idx = OrderedDict(
            {
                "STOP": [0],
                "MOVE_FORWARD": [1],
                "TURN_LEFT": [2],
                "TURN_RIGHT": [3],
            }
        )
        requested_device = str(args.device)
        if requested_device == "cuda":
            requested_device = f"cuda:{args.local_rank}"
        self.device = torch.device(requested_device)

    def config_env(self) -> Env:
        return Env(config=self.config)
