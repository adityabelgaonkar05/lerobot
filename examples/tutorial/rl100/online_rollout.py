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

"""Collect and score synchronized, no-autoreset LeRobot simulator rollouts."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.envs.utils import prepare_canonical_observation
from lerobot.sim.local_vector_env import LocalVectorEnv

from .offline_update import DenoisingTransition
from .policy import RL100Policy


@dataclass(frozen=True)
class OnlineDecision:
    features: Tensor
    chain: list[DenoisingTransition]
    reward: Tensor
    value: Tensor
    next_value: Tensor
    executed_steps: Tensor
    terminated: Tensor
    truncated: Tensor


@dataclass(frozen=True)
class OnlineRollout:
    policy_version: int
    decisions: list[OnlineDecision]
    episode_returns: np.ndarray
    episode_steps: np.ndarray
    successes: np.ndarray


def chunk_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    executed_steps: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE over time-major decision tensors with one column per environment.

    Discount by the number of executed simulator steps. Lambda decays once per decision.
    Truncation bootstraps from the final observation but ends the advantage trace.
    """
    discount = gamma**executed_steps
    deltas = rewards + discount * (~terminated) * next_values - values
    trace = discount * gae_lambda * ~(terminated | truncated)
    valid = executed_steps > 0
    advantages = torch.zeros_like(values)
    carry = torch.zeros_like(values[0])
    for index in reversed(range(len(values))):
        carry = torch.where(valid[index], deltas[index] + trace[index] * carry, 0.0)
        advantages[index] = carry
    return advantages, advantages + values


@torch.no_grad()
def collect_rollout(
    env: LocalVectorEnv,
    actor: RL100Policy,
    value: nn.Module,
    generator: torch.Generator,
    policy_version: int,
    seed: int,
    gamma: float,
) -> OnlineRollout:
    """Run one episode per slot using a fixed actor, retaining only actual executed steps."""
    config = actor.policy.config
    actor.preprocessor.reset()
    actor.postprocessor.reset()
    observation, _ = env.reset(seed=seed)
    keys = tuple(config.input_features)

    def process(raw: dict[str, np.ndarray]) -> dict[str, Tensor]:
        batch = prepare_canonical_observation({key: raw[key] for key in keys})
        return actor.preprocessor(batch)

    first = process(observation)
    history = {key: torch.stack([first[key]] * config.n_obs_steps, dim=1) for key in keys}
    features = actor.encode(history)
    device = features.device
    finished = np.zeros(env.num_envs, dtype=bool)
    episode_returns = np.zeros(env.num_envs, dtype=np.float64)
    decisions = []
    while not finished.all():
        actions, chain = actor.sample(actor.denoiser, features, generator)
        rewards = np.zeros(env.num_envs, dtype=np.float32)
        executed_steps = np.zeros(env.num_envs, dtype=np.int64)
        for offset, action in enumerate(actions.unbind(dim=1)):
            active = ~finished
            action = actor.postprocessor(action).cpu().numpy()
            observation, reward, terminated, truncated, _ = env.step(action)
            frame = process(observation)
            active_rows = torch.from_numpy(active).to(device)
            for key, frames in history.items():
                appended = torch.cat((frames[:, 1:], frame[key].unsqueeze(1)), dim=1)
                mask = active_rows.reshape(-1, *([1] * (frames.ndim - 1)))
                history[key] = torch.where(mask, appended, frames)
            episode_returns += np.where(active, reward, 0)
            rewards += (gamma**offset) * np.where(active, reward, 0)
            executed_steps += active
            finished = terminated | truncated
            if finished.all():
                break

        # A timed-out slot bootstraps from the history at its exact final step.
        next_features = actor.encode(history)
        decisions.append(
            OnlineDecision(
                features=features,
                chain=chain,
                reward=torch.as_tensor(rewards, device=device, dtype=features.dtype),
                value=value(features).squeeze(-1),
                next_value=value(next_features).squeeze(-1),
                executed_steps=torch.as_tensor(executed_steps, device=device),
                terminated=torch.as_tensor(terminated, device=device),
                truncated=torch.as_tensor(truncated, device=device),
            )
        )
        features = next_features

    final = env.latest
    return OnlineRollout(policy_version, decisions, episode_returns, final.elapsed_steps, final.is_success)
