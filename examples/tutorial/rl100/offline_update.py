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

"""Chunk IQL and denoising policy gradients for the RL-100 offline recipe."""

import math
from dataclasses import dataclass

import torch
from diffusers import DDIMScheduler
from torch import Tensor, nn
from torch.nn.functional import mse_loss


@dataclass(frozen=True)
class DenoisingTransition:
    timestep: int
    sample: Tensor
    next_sample: Tensor
    old_log_prob: Tensor


class DenoisingPPO:
    """Score action positions independently, summing their coordinate log-probabilities.

    The reference generates a full horizon. Only the executed positions contribute to the ratio.
    The final, zero-variance transition determines the sampled action but has no PPO term.
    """

    def __init__(
        self,
        scheduler: DDIMScheduler,
        inference_steps: int,
        action_slice: slice,
        eta: float,
        clip_epsilon: float,
    ):
        if not isinstance(scheduler, DDIMScheduler) or scheduler.config.prediction_type != "epsilon":
            raise ValueError("RL-100 requires an epsilon-prediction DDIM scheduler.")
        if not 2 <= inference_steps <= scheduler.config.num_train_timesteps:
            raise ValueError(
                "RL-100 requires at least two DDIM inference steps within the training schedule."
            )
        if not 0 < eta <= 1 or not 0 < clip_epsilon < 1:
            raise ValueError("Expected 0 < eta <= 1 and 0 < clip_epsilon < 1.")

        self.scheduler = DDIMScheduler.from_config(scheduler.config)
        self.scheduler.set_timesteps(inference_steps)
        self.action_slice = action_slice
        self.eta = eta
        self.clip_epsilon = clip_epsilon
        step_size = self.scheduler.config.num_train_timesteps // inference_steps
        self.std_by_timestep = {
            t: eta * math.sqrt(float(self.scheduler._get_variance(t, t - step_size)))
            for t in self.scheduler.timesteps.tolist()
        }
        if not any(self.std_by_timestep.values()):
            raise ValueError("The DDIM schedule must include a stochastic transition.")

    def _transition_mean(
        self, denoiser: nn.Module, sample: Tensor, timestep: int, conditioning: Tensor
    ) -> Tensor:
        time_batch = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
        prediction = denoiser(sample, time_batch, global_cond=conditioning)
        # Use the same scheduler path as stock inference, including predicted-sample clipping.
        return self.scheduler.step(
            prediction, timestep, sample, eta=self.eta, variance_noise=torch.zeros_like(sample)
        ).prev_sample

    def _log_prob(self, sample: Tensor, mean: Tensor, std: float) -> Tensor:
        residual = (sample[:, self.action_slice] - mean[:, self.action_slice]) / std
        return (-0.5 * residual.square() - math.log(std) - 0.5 * math.log(2 * math.pi)).sum(dim=-1)

    @torch.no_grad()
    def sample(
        self,
        reference: nn.Module,
        conditioning: Tensor,
        noise: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, list[DenoisingTransition]]:
        """Sample an executed chunk and record the stochastic transitions that produced it."""
        sample = noise
        transitions = []
        for timestep, std in self.std_by_timestep.items():
            mean = self._transition_mean(reference, sample, timestep, conditioning)
            if std == 0:
                sample = mean
                continue
            next_sample = mean + std * torch.randn(
                sample.shape, dtype=sample.dtype, device=sample.device, generator=generator
            )
            log_prob = self._log_prob(next_sample, mean, std)
            transitions.append(DenoisingTransition(timestep, sample, next_sample, log_prob))
            sample = next_sample
        return sample[:, self.action_slice], transitions

    def loss(
        self,
        denoiser: nn.Module,
        conditioning: Tensor,
        transitions: list[DenoisingTransition],
        advantage: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply the clipped surrogate to a fixed reference chain and detached chunk advantages."""
        log_ratios = []
        for transition in transitions:
            mean = self._transition_mean(denoiser, transition.sample, transition.timestep, conditioning)
            log_prob = self._log_prob(transition.next_sample, mean, self.std_by_timestep[transition.timestep])
            log_ratios.append(log_prob - transition.old_log_prob)
        log_ratio = torch.stack(log_ratios)
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon)
        advantage = advantage.detach().reshape(1, -1, 1)
        loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
        metrics = {
            "approx_kl": ((ratio - 1) - log_ratio).detach().mean(),
            "clip_fraction": ((ratio - 1).abs() > self.clip_epsilon).float().mean(),
        }
        return loss, metrics


class ChunkCritic(nn.Module):
    """IQL with twin Q heads, target Q for expectile regression, and a live V bootstrap."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, expectile: float):
        super().__init__()
        if not 0 < expectile < 1:
            raise ValueError("Expected 0 < expectile < 1.")

        def mlp(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q = nn.ModuleList(mlp(observation_dim + action_dim) for _ in range(2))
        self.target_q = nn.ModuleList(mlp(observation_dim + action_dim) for _ in range(2))
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.requires_grad_(False)
        self.value = mlp(observation_dim)
        self.expectile = expectile

    @staticmethod
    def _q_values(heads: nn.ModuleList, features: Tensor, action: Tensor) -> Tensor:
        inputs = torch.cat((features, action.flatten(start_dim=1)), dim=-1)
        return torch.cat([head(inputs) for head in heads], dim=-1)

    def value_loss(self, features: Tensor, action: Tensor) -> Tensor:
        with torch.no_grad():
            target = self._q_values(self.target_q, features, action).amin(dim=-1, keepdim=True)
        residual = target - self.value(features)
        weight = torch.where(residual > 0, self.expectile, 1 - self.expectile)
        return (weight * residual.square()).mean()

    def q_loss(
        self, features: Tensor, action: Tensor, rewards: Tensor, next_features: Tensor, discount: Tensor
    ) -> Tensor:
        with torch.no_grad():
            target = rewards + discount * self.value(next_features)
        values = self._q_values(self.q, features, action)
        return mse_loss(values, target.expand_as(values))

    @torch.no_grad()
    def advantage(self, features: Tensor, action: Tensor) -> Tensor:
        """Evaluate the same action that will receive the actor gradient."""
        return self._q_values(self.q, features, action).amin(dim=-1, keepdim=True) - self.value(features)

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        for target, current in zip(self.target_q.parameters(), self.q.parameters(), strict=True):
            target.lerp_(current, tau)
