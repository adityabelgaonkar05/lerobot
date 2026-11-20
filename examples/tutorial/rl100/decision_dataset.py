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

"""Read full action-chunk transitions from a Parquet-backed LeRobot dataset."""

import logging
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.utils.constants import ACTION, DONE, REWARD, TRUNCATED
from lerobot.utils.feature_utils import dataset_to_policy_features


class DecisionTransition(TypedDict):
    observation: dict[str, Tensor]
    next_observation: dict[str, Tensor]
    action: Tensor
    reward: Tensor
    discount: Tensor


def full_chunk_indices(
    episodes: Tensor, done: Tensor, truncated: Tensor | None, chunk_size: int, done_is_termination: bool
) -> tuple[Tensor, Tensor]:
    """Return full-chunk anchors and their terminal flags.

    A boundary can end a chunk only when it is a known true termination. A live chunk must
    have a successor row in the same episode. Unknown done semantics are conservative.
    """
    if len(episodes) < chunk_size:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool)

    boundary = done if truncated is None else done | truncated
    if truncated is not None:
        termination = done & ~truncated
    else:
        termination = done if done_is_termination else torch.zeros_like(done)

    episode_windows = episodes.unfold(0, chunk_size, 1)
    same_episode = (episode_windows == episode_windows[:, :1]).all(dim=1)
    has_successor = torch.cat((episodes[chunk_size:] == episodes[:-chunk_size], episodes.new_zeros(1).bool()))
    boundary_windows = boundary.unfold(0, chunk_size, 1)
    no_early_boundary = ~boundary_windows[:, :-1].any(dim=1)
    terminal = termination[chunk_size - 1 :]
    usable_end = terminal | (~boundary_windows[:, -1] & has_successor)
    anchors = (same_episode & no_early_boundary & usable_end).nonzero().flatten()
    return anchors, terminal[anchors]


class DecisionDataset(Dataset[DecisionTransition]):
    """Use LeRobot temporal queries for histories, actions and aligned per-step rewards."""

    def __init__(
        self,
        repo_id: str,
        root: Path | None,
        revision: str | None,
        config: DiffusionConfig,
        gamma: float,
        done_is_termination: bool,
    ):
        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root, revision=revision)
        missing = {REWARD, DONE} - meta.features.keys()
        if missing:
            raise ValueError(f"RL-100 requires explicit reward and done fields. Missing: {sorted(missing)}")
        for key in (REWARD, DONE, TRUNCATED):
            if key in meta.features and tuple(meta.features[key]["shape"]) != (1,):
                raise ValueError(f"Expected a scalar feature for {key!r}.")
        features = dataset_to_policy_features(meta.features)
        policy_features = {**config.input_features, **config.output_features}
        for key, feature in policy_features.items():
            if features.get(key) != feature:
                raise ValueError(f"Checkpoint feature {key!r} does not match the dataset schema.")

        self.obs_keys = tuple(config.input_features)
        chunk_size = config.n_action_steps
        history = list(config.observation_delta_indices)
        successor = [i + chunk_size for i in history]
        obs_offsets = sorted(set(history + successor))
        self.history_positions = [obs_offsets.index(i) for i in history]
        self.successor_positions = [obs_offsets.index(i) for i in successor]
        delta_timestamps = {key: [i / meta.fps for i in obs_offsets] for key in self.obs_keys}
        for key in (ACTION, REWARD):
            delta_timestamps[key] = [i / meta.fps for i in range(chunk_size)]

        self.dataset = LeRobotDataset(
            repo_id=repo_id, root=root, revision=revision, delta_timestamps=delta_timestamps
        )
        columns = ["episode_index", DONE]
        if TRUNCATED in meta.features:
            columns.append(TRUNCATED)
        rows = self.dataset.hf_dataset.select_columns(columns).with_format("torch")[:]
        episodes = rows["episode_index"].flatten()
        done = rows[DONE].flatten().bool()
        truncated = rows[TRUNCATED].flatten().bool() if TRUNCATED in rows else None
        boundary = done if truncated is None else done | truncated
        if (boundary[:-1] & (episodes[:-1] == episodes[1:])).any():
            raise ValueError("Done or truncated flags must appear only at the end of an episode.")
        self.anchors, self.terminal = full_chunk_indices(
            episodes,
            done,
            truncated,
            chunk_size,
            done_is_termination,
        )
        if len(self.anchors) == 0:
            raise ValueError("The dataset contains no usable full action chunks.")
        logging.info(
            "Using %d full chunks (%d terminal); excluded %d anchors.",
            len(self.anchors),
            int(self.terminal.sum()),
            len(self.dataset) - len(self.anchors),
        )
        self.reward_weights = gamma ** torch.arange(chunk_size, dtype=torch.float32)
        self.bootstrap_discount = gamma**chunk_size

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> DecisionTransition:
        row = self.dataset[int(self.anchors[index])]
        observation = {key: row[key][self.history_positions] for key in self.obs_keys}
        terminal = bool(self.terminal[index])
        # The last stored observation can precede the terminal successor. Its value is masked out.
        next_observation = (
            observation if terminal else {key: row[key][self.successor_positions] for key in self.obs_keys}
        )
        return {
            "observation": observation,
            "next_observation": next_observation,
            "action": row[ACTION],
            "reward": (row[REWARD].flatten().float() * self.reward_weights).sum().reshape(1),
            "discount": torch.tensor([0.0 if terminal else self.bootstrap_discount]),
        }
