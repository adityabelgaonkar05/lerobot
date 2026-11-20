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

"""Post-train a LeRobot Diffusion Policy using the offline RL-100 recipe."""

import hashlib
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, default_collate

from lerobot.configs import parser
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging

from .decision_dataset import DecisionDataset, DecisionTransition
from .offline_update import ChunkCritic, optimize
from .policy import RL100Policy

STATE_VERSION = 1


@dataclass
class OfflineRL100Config:
    dataset_repo_id: str
    output_dir: Path
    policy_path: Path | None = None
    resume_from: Path | None = None
    dataset_root: Path | None = None
    dataset_revision: str | None = None
    video_backend: str | None = None
    done_is_termination: bool = False
    trailing_observation: bool = False
    rollout_roots: list[Path] = field(default_factory=list)
    rollout_fraction: float | None = None
    device: str = "cpu"
    seed: int = 42
    batch_size: int = 16
    critic_steps: int = 1000
    actor_steps: int = 200
    critic_hidden_dim: int = 256
    critic_lr: float = 3e-4
    actor_lr: float = 3e-6
    gamma: float = 0.99
    expectile: float = 0.7
    target_tau: float = 0.005
    ddim_eta: float = 0.5
    clip_epsilon: float = 0.2

    def __post_init__(self):
        if (self.policy_path is None) == (self.resume_from is None):
            raise ValueError("Supply exactly one of policy_path and resume_from.")
        if min(self.batch_size, self.critic_steps, self.actor_steps, self.critic_hidden_dim) < 1:
            raise ValueError("Batch size, hidden dimension, and step counts must be positive.")
        if not 0 < self.gamma <= 1 or not 0 < self.target_tau <= 1:
            raise ValueError("Expected 0 < gamma <= 1 and 0 < target_tau <= 1.")
        if not 0 < self.actor_lr < float("inf") or not 0 < self.critic_lr < float("inf"):
            raise ValueError("Learning rates must be finite and positive.")
        if self.rollout_fraction is not None and (
            not self.rollout_roots or not 0 <= self.rollout_fraction <= 1
        ):
            raise ValueError("A rollout_fraction in [0, 1] requires at least one rollout root.")


def load_decision_sources(cfg: OfflineRL100Config, policy: DiffusionConfig) -> list[DecisionDataset]:
    primary = DecisionDataset(
        cfg.dataset_repo_id,
        cfg.dataset_root,
        cfg.dataset_revision,
        policy,
        cfg.gamma,
        cfg.done_is_termination,
        cfg.trailing_observation,
        video_backend=cfg.video_backend,
    )
    roots = [primary.dataset.root.resolve(), *(root.resolve() for root in cfg.rollout_roots)]
    if len(set(roots)) != len(roots):
        raise ValueError("Each demonstration or rollout source must have a distinct root.")
    # Native eval recordings end on the final executed action and combine episode end flags.
    return [
        primary,
        *(
            DecisionDataset(
                "local/rl100-rollout", root, None, policy, cfg.gamma, False, video_backend=cfg.video_backend
            )
            for root in roots[1:]
        ),
    ]


def checkpoint_settings(cfg: OfflineRL100Config, sources: list[DecisionDataset]) -> dict:
    settings = asdict(cfg)
    for key in ("output_dir", "policy_path", "resume_from", "critic_steps", "actor_steps", "rollout_roots"):
        del settings[key]

    def dataset_identity(source: DecisionDataset) -> dict[str, str]:
        root = source.dataset.root.resolve()
        return {
            "dataset_root": str(root),
            "dataset_info_hash": hashlib.sha256((root / "meta" / "info.json").read_bytes()).hexdigest(),
        }

    settings.update(dataset_identity(sources[0]))
    if len(sources) > 1:
        settings["rollout_datasets"] = [dataset_identity(source) for source in sources[1:]]
    if cfg.rollout_fraction is None:
        del settings["rollout_fraction"]
    if cfg.video_backend is None:
        del settings["video_backend"]
    # Preserve resume compatibility for checkpoints created before trailing rows were supported.
    if not cfg.trailing_observation:
        del settings["trailing_observation"]
    return settings


def sample_indices(
    num_chunks: int,
    num_demo: int,
    batch_size: int,
    rollout_fraction: float | None,
    generator: torch.Generator,
) -> Tensor:
    if rollout_fraction is None:
        return torch.randint(num_chunks, (batch_size,), generator=generator)
    rollout_rows = torch.rand(batch_size, generator=generator) < rollout_fraction
    num_rollout = int(rollout_rows.sum())
    indices = torch.empty(batch_size, dtype=torch.long)
    indices[~rollout_rows] = torch.randint(num_demo, (batch_size - num_rollout,), generator=generator)
    indices[rollout_rows] = torch.randint(num_demo, num_chunks, (num_rollout,), generator=generator)
    return indices


@parser.wrap()
def train(cfg: OfflineRL100Config) -> None:
    """Fit the offline critics, update a fixed-reference actor, and export a standard policy."""
    init_logging()
    if cfg.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {cfg.output_dir}")
    device = torch.device(cfg.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("The RL-100 example supports CPU and CUDA.")
    policy_path = cfg.resume_from / "pretrained_model" if cfg.resume_from is not None else cfg.policy_path
    if policy_path is None or not policy_path.is_dir():
        raise ValueError(f"Expected a local LeRobot policy directory: {policy_path}")

    torch.manual_seed(cfg.seed)
    batch_generator = torch.Generator().manual_seed(cfg.seed)
    noise_generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    actor = RL100Policy.load(policy_path, device, cfg.ddim_eta, cfg.clip_epsilon)
    denoiser = actor.denoiser
    reference = deepcopy(denoiser).requires_grad_(False)
    sources = load_decision_sources(cfg, actor.policy.config)
    dataset: Dataset[DecisionTransition] = ConcatDataset(sources) if cfg.rollout_roots else sources[0]
    settings = checkpoint_settings(cfg, sources)
    if cfg.rollout_roots:
        logging.info("Sampling %d valid chunks from %d independent sources.", len(dataset), len(sources))

    def sample_batch() -> dict:
        indices = sample_indices(
            len(dataset), len(sources[0]), cfg.batch_size, cfg.rollout_fraction, batch_generator
        )
        return default_collate([dataset[index] for index in indices.tolist()])

    def critic_batch(batch: dict) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        current = actor.preprocessor({**batch["observation"], ACTION: batch["action"]})
        successor = actor.preprocessor(batch["next_observation"])
        return (
            actor.encode(current),
            current[ACTION],
            batch["reward"].to(device),
            actor.encode(successor),
            batch["discount"].to(device),
        )

    first = default_collate([dataset[0]])
    features, actions, *_ = critic_batch(first)
    critic = ChunkCritic(features.shape[-1], actions[0].numel(), cfg.critic_hidden_dim, cfg.expectile).to(
        device
    )
    optimizers = {
        "value": torch.optim.Adam(critic.value.parameters(), lr=cfg.critic_lr),
        "q": torch.optim.Adam(critic.q.parameters(), lr=cfg.critic_lr),
        "actor": torch.optim.Adam(denoiser.parameters(), lr=cfg.actor_lr),
    }
    critic_start, actor_start = 0, 0
    if cfg.resume_from is not None:
        state = torch.load(cfg.resume_from / "training_state.pt", map_location="cpu", weights_only=True)
        if state["version"] != STATE_VERSION or state["settings"] != settings:
            raise ValueError("Resume checkpoint version or training settings differ.")
        critic_start, actor_start = state["critic_steps"], state["actor_steps"]
        if cfg.critic_steps != critic_start or cfg.actor_steps <= actor_start:
            raise ValueError("Resume with the same critic_steps and a larger total actor_steps.")
        critic.load_state_dict(state["critic"])
        reference.load_state_dict(state["reference"])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        batch_generator.set_state(state["batch_rng"])
        noise_generator.set_state(state["noise_rng"])

    for step in range(critic_start, cfg.critic_steps):
        features, actions, rewards, successors, discounts = critic_batch(sample_batch())
        value_loss = critic.value_loss(features, actions)
        optimize(value_loss, critic.value, optimizers["value"])
        q_loss = critic.q_loss(features, actions, rewards, successors, discounts)
        optimize(q_loss, critic.q, optimizers["q"])
        critic.update_target(cfg.target_tau)
        if step % 100 == 0 or step + 1 == cfg.critic_steps:
            logging.info(
                "Critic step %d | value loss %.5g | Q loss %.5g", step + 1, value_loss.item(), q_loss.item()
            )

    for step in range(actor_start, cfg.actor_steps):
        batch = sample_batch()
        features = actor.encode(actor.preprocessor(batch["observation"]))
        sampled_actions, chain = actor.sample(reference, features, noise_generator)
        advantage = critic.advantage(features, sampled_actions)
        loss, metrics = actor.ppo.loss(denoiser, features, chain, advantage)
        optimize(loss, denoiser, optimizers["actor"])
        if step % 100 == 0 or step + 1 == cfg.actor_steps:
            logging.info(
                "Actor step %d | loss %.5g | approx KL %.5g | clip fraction %.5g",
                step + 1,
                loss.item(),
                metrics["approx_kl"],
                metrics["clip_fraction"],
            )

    cfg.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cfg.output_dir.parent) as directory:
        staged = Path(directory) / "checkpoint"
        actor.export(staged / "pretrained_model")
        torch.save(
            {
                "version": STATE_VERSION,
                "settings": settings,
                "critic_steps": cfg.critic_steps,
                "actor_steps": cfg.actor_steps,
                "critic": critic.state_dict(),
                "reference": reference.state_dict(),
                "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                "batch_rng": batch_generator.get_state(),
                "noise_rng": noise_generator.get_state(),
            },
            staged / "training_state.pt",
        )
        staged.rename(cfg.output_dir)
    logging.info("Saved LeRobot policy and processors to %s", cfg.output_dir / "pretrained_model")


if __name__ == "__main__":
    train()
