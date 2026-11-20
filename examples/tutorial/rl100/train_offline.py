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
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import Tensor, nn
from torch.utils.data import default_collate

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.utils import init_logging

from .decision_dataset import DecisionDataset
from .offline_update import ChunkCritic, DenoisingPPO

STATE_VERSION = 1


@dataclass
class OfflineRL100Config:
    dataset_repo_id: str
    output_dir: Path
    policy_path: Path | None = None
    resume_from: Path | None = None
    dataset_root: Path | None = None
    dataset_revision: str | None = None
    done_is_termination: bool = False
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


def checkpoint_settings(cfg: OfflineRL100Config, dataset: DecisionDataset) -> dict:
    settings = asdict(cfg)
    for key in ("output_dir", "policy_path", "resume_from", "critic_steps", "actor_steps"):
        del settings[key]
    settings["dataset_root"] = str(dataset.dataset.root.resolve())
    settings["dataset_info_hash"] = hashlib.sha256(
        (dataset.dataset.root / "meta" / "info.json").read_bytes()
    ).hexdigest()
    return settings


def optimize(loss: Tensor, module: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError("Encountered a non-finite RL-100 loss.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0, error_if_nonfinite=True)
    optimizer.step()


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
    policy_config = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=[f"--device={device}"])
    if not isinstance(policy_config, DiffusionConfig):
        raise ValueError("RL-100 expects a LeRobot Diffusion Policy checkpoint.")
    if policy_config.compile_model:
        raise ValueError("Use a checkpoint with compile_model disabled for RL-100 post-training.")
    images = policy_config.image_features
    if len(images) != 1 or any(feature.shape[:1] != (3,) for feature in images.values()):
        raise ValueError("This recipe expects exactly one RGB camera.")
    # The checkpoint already contains every vision weight. Avoid downloading the initialization.
    policy_config.pretrained_backbone_weights = None
    policy = DiffusionPolicy.from_pretrained(policy_path, config=policy_config, strict=True)
    policy.requires_grad_(False)
    denoiser = policy.diffusion.unet.requires_grad_(True)
    reference = deepcopy(denoiser).requires_grad_(False)
    start = policy_config.n_obs_steps - 1
    action_steps = policy_config.n_action_steps
    action_slice = slice(start, start + action_steps)
    if action_slice.stop > policy_config.horizon:
        raise ValueError("The executed action positions extend past the diffusion horizon.")
    ppo = DenoisingPPO(
        policy.diffusion.noise_scheduler,
        policy.diffusion.num_inference_steps,
        action_slice,
        cfg.ddim_eta,
        cfg.clip_epsilon,
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    dataset = DecisionDataset(
        cfg.dataset_repo_id,
        cfg.dataset_root,
        cfg.dataset_revision,
        policy_config,
        cfg.gamma,
        cfg.done_is_termination,
    )
    settings = checkpoint_settings(cfg, dataset)

    def sample_batch() -> dict:
        indices = torch.randint(len(dataset), (cfg.batch_size,), generator=batch_generator)
        return default_collate([dataset[index] for index in indices.tolist()])

    @torch.no_grad()
    def encode(batch: dict[str, Tensor]) -> Tensor:
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in policy_config.image_features], dim=-4)
        return policy.diffusion._prepare_global_conditioning(batch)

    def critic_batch(batch: dict) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        current = preprocessor({**batch["observation"], ACTION: batch["action"]})
        successor = preprocessor(batch["next_observation"])
        return (
            encode(current),
            current[ACTION],
            batch["reward"].to(device),
            encode(successor),
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

    action_dim = actions.shape[-1]
    for step in range(actor_start, cfg.actor_steps):
        batch = sample_batch()
        features = encode(preprocessor(batch["observation"]))
        noise = torch.randn(
            (features.shape[0], policy_config.horizon, action_dim),
            device=device,
            dtype=features.dtype,
            generator=noise_generator,
        )
        sampled_actions, chain = ppo.sample(reference, features, noise, noise_generator)
        advantage = critic.advantage(features, sampled_actions)
        loss, metrics = ppo.loss(denoiser, features, chain, advantage)
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
        export = staged / "pretrained_model"
        policy.save_pretrained(export)
        preprocessor.save_pretrained(export)
        postprocessor.save_pretrained(export)
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
