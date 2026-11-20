# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distill a multi-step DDIM checkpoint into a native one-step Diffusion Policy."""

import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.utils.data import default_collate

from lerobot.configs import parser
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.import_utils import _av_available, _datasets_available, require_package
from lerobot.utils.utils import init_logging

from .offline_update import optimize
from .policy import RL100Policy

if TYPE_CHECKING:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

if TYPE_CHECKING or (_av_available and _datasets_available):
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


@dataclass
class DistillConfig:
    policy_path: Path
    output_dir: Path
    dataset_repo_id: str
    dataset_root: Path | None = None
    dataset_revision: str | None = None
    video_backend: str | None = None
    device: str = "cpu"
    seed: int = 42
    batch_size: int = 16
    steps: int = 500
    learning_rate: float = 1e-4

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size) < 1 or not 0 < self.learning_rate < float("inf"):
            raise ValueError("Expected positive steps, batch size, and a finite positive learning rate.")


def one_step_scheduler(teacher: "DDIMScheduler") -> "DDIMScheduler":
    """Make stock DDIM return the student's clean-sample prediction in one step."""
    scheduler = type(teacher).from_config(teacher.config, prediction_type="sample")
    scheduler.set_timesteps(1)
    if scheduler.timesteps.tolist() != [0] or not scheduler.config.set_alpha_to_one:
        raise ValueError(
            "One-step distillation requires a DDIM schedule ending at zero with alpha set to one."
        )
    return scheduler


def distillation_loss(student: nn.Module, teacher: RL100Policy, features: Tensor, noise: Tensor) -> Tensor:
    """Regress the teacher's executed actions using its exact initial noise and observation."""
    with torch.no_grad():
        target = teacher.policy.diffusion.conditional_sample(len(features), global_cond=features, noise=noise)
    timestep = torch.zeros(len(features), device=features.device, dtype=torch.long)
    predict = student(noise, timestep, global_cond=features)
    executed = teacher.ppo.action_slice
    return mse_loss(predict[:, executed], target[:, executed])


@parser.wrap()
def train(cfg: DistillConfig) -> None:
    """Freeze the teacher and RGB encoder while fitting a single-step action predictor."""
    init_logging()
    if cfg.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {cfg.output_dir}")
    device = torch.device(cfg.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("The RL-100 example supports CPU and CUDA.")
    require_package("datasets", extra="dataset")
    require_package("av", extra="dataset")
    torch.manual_seed(cfg.seed)
    batch_generator = torch.Generator().manual_seed(cfg.seed)
    noise_generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    teacher = RL100Policy.load(cfg.policy_path, device, eta=0.5, clip_epsilon=0.2)
    teacher.denoiser.requires_grad_(False)
    policy = teacher.policy
    scheduler = one_step_scheduler(policy.diffusion.noise_scheduler)
    student = deepcopy(teacher.denoiser).requires_grad_(True)
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.learning_rate)

    meta = LeRobotDatasetMetadata(
        repo_id=cfg.dataset_repo_id, root=cfg.dataset_root, revision=cfg.dataset_revision
    )
    features = dataset_to_policy_features(meta.features)
    for key, feature in policy.config.input_features.items():
        if features.get(key) != feature:
            raise ValueError(f"Checkpoint feature {key!r} does not match the dataset schema.")
    history = [offset / meta.fps for offset in policy.config.observation_delta_indices]
    keys = tuple(policy.config.input_features)
    dataset = LeRobotDataset(
        repo_id=cfg.dataset_repo_id,
        root=cfg.dataset_root,
        revision=cfg.dataset_revision,
        delta_timestamps=dict.fromkeys(keys, history),
        video_backend=cfg.video_backend,
    )

    for step in range(cfg.steps):
        indices = torch.randint(len(dataset), (cfg.batch_size,), generator=batch_generator)
        frames = [dataset[index] for index in indices.tolist()]
        batch = default_collate([{key: frame[key] for key in keys} for frame in frames])
        conditioning = teacher.encode(teacher.preprocessor(batch))
        noise = teacher.initial_noise(conditioning, noise_generator)
        loss = distillation_loss(student, teacher, conditioning, noise)
        optimize(loss, student, optimizer)
        if step % 100 == 0 or step + 1 == cfg.steps:
            logging.info("Distillation step %d | action MSE %.5g", step + 1, loss.item())

    policy.diffusion.unet = student.eval()
    policy.diffusion.noise_scheduler = scheduler
    policy.diffusion.num_inference_steps = policy.config.num_inference_steps = 1
    policy.config.prediction_type = "sample"
    cfg.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cfg.output_dir.parent) as directory:
        staged = Path(directory) / "checkpoint"
        teacher.export(staged / "pretrained_model")
        staged.rename(cfg.output_dir)


if __name__ == "__main__":
    train()
