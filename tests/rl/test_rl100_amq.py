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

import pytest
import torch

pytestmark = pytest.mark.dataset
pytest.importorskip("datasets", reason="install lerobot[dataset] to run RL-100 AM-Q tests")
pytest.importorskip("diffusers", reason="install lerobot[diffusion] to run RL-100 AM-Q tests")

from examples.tutorial.rl100.amq import amq_score  # noqa: E402


def test_amq_scores_shared_dynamics_member_choices_per_environment():
    def chunk(features, generator):
        return torch.ones_like(features)

    def q(features, actions):
        return features + actions

    def dynamics(features, actions):
        return torch.stack((features + actions, features + 2 * actions))

    features = torch.tensor([[0.0], [10.0]])
    members = torch.tensor([[0, 1], [1, 0], [0, 0]])
    score = amq_score(features, chunk, q, dynamics, members, torch.Generator())
    torch.testing.assert_close(score, torch.tensor([7 / 3, 38 / 3]))


def test_amq_pairs_noise_over_the_entire_imagined_trajectory():
    def run(shift):
        noises = []

        def chunk(features, generator):
            draws = torch.randn(2, *features.shape, generator=generator)
            noises.append(draws)
            return draws.sum(dim=0) + shift

        def q(features, actions):
            return features + actions

        def dynamics(features, actions):
            return torch.stack((features + actions, features + 2 * actions))

        score = amq_score(
            torch.zeros(1, 1),
            chunk,
            q,
            dynamics,
            torch.tensor([[0], [1], [0]]),
            torch.Generator().manual_seed(100),
        )
        return score, torch.stack(noises)

    first, first_noise = run(0.0)
    second, second_noise = run(1.0)
    torch.testing.assert_close(first_noise, second_noise)
    torch.testing.assert_close(second - first, torch.tensor([7 / 3]))
