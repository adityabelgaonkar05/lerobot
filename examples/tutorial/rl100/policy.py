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

"""Shared checkpoint and conditioning helpers for the RL-100 examples."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import OBS_IMAGES

from .offline_update import DenoisingPPO, DenoisingTransition


@dataclass
class RL100Policy:
    policy: DiffusionPolicy
    denoiser: nn.Module
    ppo: DenoisingPPO
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction]

    @classmethod
    def load(cls, path: Path, device: torch.device, eta: float, clip_epsilon: float) -> "RL100Policy":
        if not path.is_dir():
            raise ValueError(f"Expected a local LeRobot policy directory: {path}")
        config = PreTrainedConfig.from_pretrained(path, cli_overrides=[f"--device={device}"])
        if not isinstance(config, DiffusionConfig):
            raise ValueError("RL-100 expects a LeRobot Diffusion Policy checkpoint.")
        if config.compile_model:
            raise ValueError("Use a checkpoint with compile_model disabled for RL-100 post-training.")
        images = config.image_features
        if len(images) != 1 or any(feature.shape[:1] != (3,) for feature in images.values()):
            raise ValueError("This recipe expects exactly one RGB camera.")
        # The checkpoint already contains every vision weight. Avoid downloading the initialization.
        config.pretrained_backbone_weights = None
        policy = DiffusionPolicy.from_pretrained(path, config=config, strict=True)
        policy.requires_grad_(False)
        denoiser = policy.diffusion.unet.requires_grad_(True)
        start = config.n_obs_steps - 1
        action_slice = slice(start, start + config.n_action_steps)
        if action_slice.stop > config.horizon:
            raise ValueError("The executed action positions extend past the diffusion horizon.")
        ppo = DenoisingPPO(
            policy.diffusion.noise_scheduler,
            policy.diffusion.num_inference_steps,
            action_slice,
            eta,
            clip_epsilon,
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        return cls(policy, denoiser, ppo, preprocessor, postprocessor)

    @torch.no_grad()
    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        batch = {
            **batch,
            OBS_IMAGES: torch.stack([batch[key] for key in self.policy.config.image_features], dim=-4),
        }
        return self.policy.diffusion._prepare_global_conditioning(batch)

    def sample(
        self, reference: nn.Module, conditioning: Tensor, generator: torch.Generator
    ) -> tuple[Tensor, list[DenoisingTransition]]:
        config = self.policy.config
        noise = torch.randn(
            (conditioning.shape[0], config.horizon, config.action_feature.shape[0]),
            device=conditioning.device,
            dtype=conditioning.dtype,
            generator=generator,
        )
        return self.ppo.sample(reference, conditioning, noise, generator)

    def export(self, path: Path) -> None:
        self.policy.save_pretrained(path)
        self.preprocessor.save_pretrained(path)
        self.postprocessor.save_pretrained(path)
