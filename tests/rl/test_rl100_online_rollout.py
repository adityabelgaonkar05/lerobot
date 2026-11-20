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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.utils.constants import OBS_STATE

pytest.importorskip("diffusers", reason="install lerobot[diffusion] to run RL-100 rollout tests")

from examples.tutorial.rl100.online_rollout import chunk_gae, collect_rollout
from examples.tutorial.rl100.train_online import OnlineRL100Config, update_actor_and_value


def test_chunk_gae_handles_partial_steps_and_separate_episode_flags():
    advantages, returns = chunk_gae(
        rewards=torch.tensor([[3.0, 1.0], [0.0, 2.0], [0.0, 10.0]]),
        values=torch.tensor([[1.0, 2.0], [5.0, 4.0], [5.0, 1.0]]),
        next_values=torch.tensor([[100.0, 4.0], [5.0, 6.0], [5.0, 12.0]]),
        executed_steps=torch.tensor([[1, 2], [0, 1], [0, 1]]),
        terminated=torch.tensor([[True, False], [True, False], [True, True]]),
        truncated=torch.tensor([[False, False], [False, True], [False, False]]),
        gamma=0.5,
        gae_lambda=0.8,
    )
    torch.testing.assert_close(advantages, torch.tensor([[2.0, 0.2], [0.0, 1.0], [0.0, 9.0]]))
    assert returns[1, 1].item() == 5.0


class TwoEpisodeEnv:
    num_envs = 2

    def reset(self, seed):
        self.elapsed = np.zeros(2, dtype=np.int64)
        return self.observation(), {}

    def observation(self):
        return {OBS_STATE: self.elapsed[:, None].astype(np.float32)}

    def step(self, action):
        active = self.elapsed < [1, 3]
        self.elapsed += active
        terminated, truncated = self.elapsed >= [1, 3]
        return (
            self.observation(),
            np.where(active, 1.0, 999.0),
            np.array([False, truncated]),
            np.array([terminated, False]),
            {},
        )

    @property
    def latest(self):
        return SimpleNamespace(elapsed_steps=self.elapsed, is_success=np.array([False, True]))


class IdentityProcessor:
    def __call__(self, value):
        return value

    def reset(self):
        pass


class HistoryActor:
    preprocessor = postprocessor = IdentityProcessor()
    denoiser = nn.Identity()
    policy = SimpleNamespace(config=SimpleNamespace(input_features={OBS_STATE: None}, n_obs_steps=2))

    def encode(self, batch):
        return batch[OBS_STATE].flatten(start_dim=1)

    def sample(self, reference, conditioning, generator):
        return torch.zeros((len(conditioning), 2, 1)), []


def test_rollout_preserves_history_at_timeout_and_skips_finished_rows(tmp_path):
    actor = HistoryActor()
    value = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        value.weight.fill_(1)
    rollout = collect_rollout(TwoEpisodeEnv(), actor, value, torch.Generator(), 7, 50, 0.5)
    first, final = rollout.decisions
    assert first.executed_steps.tolist() == [1, 2]
    assert first.truncated.tolist() == [True, False]
    assert first.next_value.tolist() == [1, 3]
    assert first.reward.tolist() == [1, 1.5]
    assert final.executed_steps.tolist() == [0, 1]
    assert final.terminated.tolist() == [False, True]
    assert rollout.episode_returns.tolist() == [1, 3]

    cfg = OnlineRL100Config(tmp_path, tmp_path / "out")
    optimizer = torch.optim.Adam(value.parameters())
    with pytest.raises(ValueError, match="Expected rollout version 8, got 7"):
        update_actor_and_value(cfg, actor, value, optimizer, optimizer, rollout, 8, torch.Generator())
