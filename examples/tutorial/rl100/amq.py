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

"""Compare paired model-rollout Q scores with real PushT simulator returns."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.utils.data import default_collate

from lerobot.configs import parser
from lerobot.envs.configs import PushtEnvConfig
from lerobot.envs.factory import make_task_env
from lerobot.envs.utils import prepare_canonical_observation, validate_checkpoint
from lerobot.sim.local_vector_env import LocalVectorEnv
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging

from .decision_dataset import DecisionDataset
from .offline_update import ChunkCritic, make_critic_head, optimize
from .online_rollout import collect_rollout
from .policy import RL100Policy


@dataclass
class AMQConfig:
    reference_path: Path
    candidate_path: Path
    output_file: Path
    dataset_repo_id: str
    dataset_root: Path | None = None
    dataset_revision: str | None = None
    trailing_observation: bool = False
    device: str = "cpu"
    seed: int = 42
    env_seed: int = 30_000
    sim_episodes: int = 16
    fit_steps: int = 200
    batch_size: int = 32
    hidden_dim: int = 256
    ensemble_size: int = 3
    horizon: int = 3
    gamma: float = 0.99
    ddim_eta: float = 0.5

    def __post_init__(self) -> None:
        if min(self.fit_steps, self.batch_size, self.hidden_dim, self.horizon, self.sim_episodes) < 1:
            raise ValueError("Expected positive training and evaluation sizes.")
        if self.ensemble_size < 2 or not 0 < self.gamma <= 1:
            raise ValueError("Expected at least two ensemble members and 0 < gamma <= 1.")


class FeatureDynamics(nn.Module):
    """Bootstrap an ensemble of regressors for the change in the full observation history."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int, ensemble_size: int):
        super().__init__()
        self.members = nn.ModuleList(
            make_critic_head(feature_dim + action_dim, hidden_dim, feature_dim) for _ in range(ensemble_size)
        )

    def forward(self, features: Tensor, action: Tensor) -> Tensor:
        inputs = torch.cat((features, action.flatten(start_dim=1)), dim=-1)
        return torch.stack([features + member(inputs) for member in self.members])


@torch.no_grad()
def amq_score(
    features: Tensor,
    chunk: Callable[[Tensor, torch.Generator], Tensor],
    q: Callable[[Tensor, Tensor], Tensor],
    dynamics: Callable[[Tensor, Tensor], Tensor],
    member_indices: Tensor,
    noise: torch.Generator,
) -> Tensor:
    """Average min-Q along each imagined trajectory under fixed ensemble choices."""
    scores = []
    rows = torch.arange(len(features), device=features.device)
    for offset, member in enumerate(member_indices):
        actions = chunk(features, noise)
        scores.append(q(features, actions).squeeze(-1))
        if offset + 1 < len(member_indices):
            features = dynamics(features, actions)[member, rows]
    return torch.stack(scores).mean(dim=0)


def check_shared_representation(reference: RL100Policy, candidate: RL100Policy) -> None:
    # OPE shares one trained feature model and one normalized action representation.
    config_keys = (
        "input_features",
        "output_features",
        "normalization_mapping",
        "horizon",
        "n_obs_steps",
        "n_action_steps",
        "resize_shape",
        "crop_shape",
        "num_inference_steps",
    )
    for key in config_keys:
        if getattr(reference.policy.config, key) != getattr(candidate.policy.config, key):
            raise ValueError(f"AM-Q actor configs differ at {key!r}.")
    if reference.ppo.scheduler.config != candidate.ppo.scheduler.config:
        raise ValueError("AM-Q requires matching DDIM schedules.")
    ref_state = reference.policy.state_dict()
    candidate_state = candidate.policy.state_dict()
    frozen = {key for key in ref_state if ".unet." not in key}
    if frozen != {key for key in candidate_state if ".unet." not in key} or any(
        not torch.equal(ref_state[key], candidate_state[key]) for key in frozen
    ):
        raise ValueError("AM-Q requires a shared frozen observation encoder.")
    for first, second in (
        (reference.preprocessor, candidate.preprocessor),
        (reference.postprocessor, candidate.postprocessor),
    ):
        processor_specs = [(type(step), step.get_config()) for step in first.steps]
        if processor_specs != [(type(step), step.get_config()) for step in second.steps]:
            raise ValueError("AM-Q requires matching processor configurations.")
        first_state = first.state_dict()
        second_state = second.state_dict()
        if first_state.keys() != second_state.keys() or any(
            values.keys() != second_state[step].keys()
            or any(not torch.equal(value, second_state[step][key]) for key, value in values.items())
            for step, values in first_state.items()
        ):
            raise ValueError("AM-Q requires matching processor statistics.")


@parser.wrap()
def evaluate(cfg: AMQConfig) -> None:
    """Fit IQL and feature dynamics on one episode split and report paired AM-Q and returns."""
    init_logging()
    if cfg.output_file.exists():
        raise FileExistsError(f"Output file already exists: {cfg.output_file}")
    device = torch.device(cfg.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("The RL-100 example supports CPU and CUDA.")
    torch.manual_seed(cfg.seed)
    rng = torch.Generator().manual_seed(cfg.seed)
    reference = RL100Policy.load(cfg.reference_path, device, cfg.ddim_eta, 0.2)
    candidate = RL100Policy.load(cfg.candidate_path, device, cfg.ddim_eta, 0.2)
    check_shared_representation(reference, candidate)

    dataset = DecisionDataset(
        cfg.dataset_repo_id,
        cfg.dataset_root,
        cfg.dataset_revision,
        reference.policy.config,
        cfg.gamma,
        False,
        cfg.trailing_observation,
    )
    episodes = dataset.episode_ids.unique()
    if len(episodes) < 2:
        raise ValueError("AM-Q needs at least two episodes for an independent dynamics holdout.")
    n_holdout = max(1, len(episodes) // 5)
    holdout_ids = episodes[torch.randperm(len(episodes), generator=rng)[:n_holdout]]
    holdout_mask = torch.isin(dataset.episode_ids, holdout_ids)
    # Terminal chunks can lack a recorded successor. Fit and validate dynamics on live chunks.
    train_pool = (~holdout_mask & ~dataset.terminal).nonzero().flatten()
    holdout_pool = (holdout_mask & ~dataset.terminal).nonzero().flatten()
    if len(train_pool) == 0 or len(holdout_pool) == 0:
        raise ValueError("AM-Q requires live chunks in both episode splits.")

    def batch(pool: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indices = pool[torch.randint(len(pool), (cfg.batch_size,), generator=rng)]
        frame = default_collate([dataset[index] for index in indices.tolist()])
        current = reference.preprocessor({**frame["observation"], ACTION: frame["action"]})
        successor = reference.preprocessor(frame["next_observation"])
        return (
            reference.encode(current),
            current[ACTION],
            reference.encode(successor),
            frame["reward"].to(device),
            frame["discount"].to(device),
        )

    features, actions, *_ = batch(train_pool)
    feature_dim, action_dim = features.shape[-1], actions[0].numel()
    critic = ChunkCritic(feature_dim, action_dim, cfg.hidden_dim, expectile=0.7).to(device)
    dynamics = FeatureDynamics(feature_dim, action_dim, cfg.hidden_dim, cfg.ensemble_size).to(device)
    value_opt = torch.optim.Adam(critic.value.parameters(), lr=3e-4)
    q_opt = torch.optim.Adam(critic.q.parameters(), lr=3e-4)
    dynamics_opt = torch.optim.Adam(dynamics.parameters(), lr=3e-4)
    for _ in range(cfg.fit_steps):
        features, actions, successors, rewards, discount = batch(train_pool)
        optimize(critic.value_loss(features, actions), critic.value, value_opt)
        optimize(critic.q_loss(features, actions, rewards, successors, discount), critic.q, q_opt)
        critic.update_target(0.005)
        samples = torch.randint(len(features), (cfg.ensemble_size, len(features)), generator=rng).to(device)
        rows = torch.arange(cfg.ensemble_size, device=device)[:, None]
        estimates = dynamics(features, actions)[rows, samples]
        optimize(mse_loss(estimates, successors[samples]), dynamics, dynamics_opt)

    with torch.no_grad():
        dynamics_errors, persistence_errors = [], []
        for _ in range(8):
            features, actions, successors, *_ = batch(holdout_pool)
            dynamics_errors.append((dynamics(features, actions) - successors).square().mean().item())
            persistence_errors.append(mse_loss(features, successors).item())
        heldout_mse = sum(dynamics_errors) / len(dynamics_errors)
        persistence_mse = sum(persistence_errors) / len(persistence_errors)

    env = make_task_env(PushtEnvConfig(), task_group="PushT-v0", task_id=0, n_envs=cfg.sim_episodes)
    try:
        if not isinstance(env, LocalVectorEnv):
            raise ValueError("AM-Q calibration requires local PushT.")
        validate_checkpoint(env.descriptor, reference.policy.config, rename_map={})
        observation, _ = env.reset(seed=cfg.env_seed)
        keys = reference.policy.config.input_features
        frames = reference.preprocessor(
            prepare_canonical_observation({key: observation[key] for key in keys})
        )
        obs_steps = reference.policy.config.n_obs_steps
        initial_features = reference.encode(
            {key: torch.stack([frames[key]] * obs_steps, dim=1) for key in keys}
        )
        members = torch.randint(cfg.ensemble_size, (cfg.horizon, cfg.sim_episodes), generator=rng).to(device)
        scored = {}

        def noise() -> torch.Generator:
            return torch.Generator(device=device).manual_seed(cfg.seed + 1)

        for name, actor in (("reference", reference), ("candidate", candidate)):

            def sample(
                conditioning: Tensor, generator: torch.Generator, actor: RL100Policy = actor
            ) -> Tensor:
                return actor.sample(actor.denoiser, conditioning, generator)[0]

            score = amq_score(initial_features, sample, critic.min_q, dynamics, members, noise())
            rollout = collect_rollout(env, actor, critic.value, noise(), 0, cfg.env_seed, cfg.gamma)
            scored[name] = {
                "amq": score.tolist(),
                "returns": rollout.episode_returns.tolist(),
                "successes": rollout.successes.tolist(),
            }
    finally:
        env.close()

    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_file.open("x") as output:
        json.dump(
            {
                "seed": cfg.seed,
                "env_seed": cfg.env_seed,
                "holdout_ids": holdout_ids.tolist(),
                "dynamics_mse": heldout_mse,
                "persistence_mse": persistence_mse,
                **scored,
            },
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    evaluate()
