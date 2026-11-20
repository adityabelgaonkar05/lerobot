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

"""Run synchronous online RL-100 post-training in the local PushT simulator."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from lerobot.configs import parser
from lerobot.envs.configs import PushtEnvConfig
from lerobot.envs.factory import make_task_env
from lerobot.envs.utils import validate_checkpoint
from lerobot.sim.local_vector_env import LocalVectorEnv
from lerobot.utils.utils import init_logging

from .offline_update import DenoisingTransition, make_critic_head, optimize
from .online_rollout import OnlineRollout, chunk_gae, collect_rollout
from .policy import RL100Policy


@dataclass
class OnlineRL100Config:
    policy_path: Path
    output_dir: Path
    device: str = "cpu"
    seed: int = 42
    env_seed: int = 20_000
    num_envs: int = 4
    episode_length: int = 300
    rounds: int = 5
    epochs: int = 2
    batch_size: int = 16
    value_hidden_dim: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_lr: float = 3e-4
    actor_lr: float = 3e-6
    ddim_eta: float = 0.5
    clip_epsilon: float = 0.2

    def __post_init__(self):
        if (
            min(
                self.num_envs,
                self.episode_length,
                self.rounds,
                self.epochs,
                self.batch_size,
                self.value_hidden_dim,
            )
            < 1
        ):
            raise ValueError("Expected positive environment sizes, round counts, and training sizes.")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Expected 0 < gamma <= 1 and 0 <= gae_lambda <= 1.")
        if not 0 < self.actor_lr < float("inf") or not 0 < self.value_lr < float("inf"):
            raise ValueError("Learning rates must be finite and positive.")


def update_actor_and_value(
    cfg: OnlineRL100Config,
    actor: RL100Policy,
    value: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    rollout: OnlineRollout,
    policy_version: int,
    generator: torch.Generator,
) -> dict[str, float]:
    """Fit one rollout with cached old likelihoods, then discard it before refreshing the actor."""
    if rollout.policy_version != policy_version:
        raise ValueError(f"Expected rollout version {policy_version}, got {rollout.policy_version}.")
    decisions = rollout.decisions

    def stack(key: str) -> Tensor:
        return torch.stack([getattr(step, key) for step in decisions])

    executed_steps = stack("executed_steps")
    valid = executed_steps > 0
    advantages, returns = chunk_gae(
        stack("reward"),
        stack("value"),
        stack("next_value"),
        executed_steps,
        stack("terminated"),
        stack("truncated"),
        cfg.gamma,
        cfg.gae_lambda,
    )
    features = stack("features")[valid]
    advantages = advantages[valid]
    returns = returns[valid]
    lengths = executed_steps[valid]
    chain = [
        DenoisingTransition(
            transitions[0].timestep,
            torch.stack([step.sample for step in transitions])[valid],
            torch.stack([step.next_sample for step in transitions])[valid],
            torch.stack([step.old_log_prob for step in transitions])[valid],
        )
        for transitions in zip(*(step.chain for step in decisions), strict=True)
    ]

    totals = {"actor_loss": 0.0, "value_loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
    positions = 0
    for _ in range(cfg.epochs):
        for indices in torch.randperm(len(features), generator=generator).split(cfg.batch_size):
            indices = indices.to(features.device)
            batch_features = features[indices]
            batch_chain = [
                DenoisingTransition(
                    step.timestep, step.sample[indices], step.next_sample[indices], step.old_log_prob[indices]
                )
                for step in chain
            ]
            batch_lengths = lengths[indices]
            policy_loss, metrics = actor.ppo.loss(
                actor.denoiser, batch_features, batch_chain, advantages[indices], batch_lengths
            )
            optimize(policy_loss, actor.denoiser, actor_optimizer)
            value_loss = mse_loss(value(batch_features).squeeze(-1), returns[indices])
            optimize(value_loss, value, value_optimizer)
            weight = int(batch_lengths.sum())
            positions += weight
            totals["actor_loss"] += weight * policy_loss.item()
            totals["value_loss"] += len(indices) * value_loss.item()
            for name, metric in metrics.items():
                totals[name] += weight * metric.item()
    totals["value_loss"] /= len(features) * cfg.epochs
    return {name: total / positions if name != "value_loss" else total for name, total in totals.items()}


@parser.wrap()
def train(cfg: OnlineRL100Config) -> None:
    """Collect a fresh PushT episode batch between each short on-policy update phase."""
    init_logging()
    if cfg.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {cfg.output_dir}")
    device = torch.device(cfg.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("The RL-100 example supports CPU and CUDA.")
    torch.manual_seed(cfg.seed)
    noise_generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    batch_generator = torch.Generator().manual_seed(cfg.seed)
    actor = RL100Policy.load(cfg.policy_path, device, cfg.ddim_eta, cfg.clip_epsilon)
    value = make_critic_head(None, cfg.value_hidden_dim).to(device)
    # Start the critic at zero without assuming the scale of the demonstration rewards.
    nn.init.zeros_(value[-1].weight)
    nn.init.zeros_(value[-1].bias)
    actor_optimizer = torch.optim.Adam(actor.denoiser.parameters(), lr=cfg.actor_lr)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=cfg.value_lr)

    env_cfg = PushtEnvConfig(episode_length=cfg.episode_length)
    env = make_task_env(env_cfg, task_group="PushT-v0", task_id=0, n_envs=cfg.num_envs)
    metrics = []
    try:
        if not isinstance(env, LocalVectorEnv):
            raise ValueError("The online recipe requires a local no-autoreset simulator.")
        validate_checkpoint(env.descriptor, actor.policy.config, rename_map={})
        for version in range(cfg.rounds):
            rollout = collect_rollout(
                env, actor, value, noise_generator, version, cfg.env_seed + version * cfg.num_envs, cfg.gamma
            )
            summary = update_actor_and_value(
                cfg, actor, value, actor_optimizer, value_optimizer, rollout, version, batch_generator
            )
            summary = {
                **summary,
                "policy_version": version,
                "episode_returns": rollout.episode_returns.tolist(),
                "episode_steps": rollout.episode_steps.tolist(),
                "successes": rollout.successes.tolist(),
            }
            metrics.append(summary)
            logging.info("Online round %d | %s", version + 1, summary)
    finally:
        env.close()

    cfg.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cfg.output_dir.parent) as directory:
        staged = Path(directory) / "checkpoint"
        actor.export(staged / "pretrained_model")
        (staged / "rollout_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        staged.rename(cfg.output_dir)


if __name__ == "__main__":
    train()
