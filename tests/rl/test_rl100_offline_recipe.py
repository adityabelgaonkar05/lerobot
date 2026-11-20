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

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.dataset
pytest.importorskip("datasets", reason="install lerobot[dataset] to run RL-100 recipe tests")

from examples.tutorial.rl100.decision_dataset import DecisionDataset, full_chunk_indices  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGE, OBS_STATE, REWARD, TRUNCATED  # noqa: E402
from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: E402


@pytest.mark.parametrize(
    ("with_truncation", "trust_done", "indices", "terminal"),
    [
        (True, False, [0, 1, 2, 4], [False, False, True, False]),
        (False, False, [0, 1, 4], [False, False, False]),
        (False, True, [0, 1, 2, 4, 5], [False, False, True, False, True]),
    ],
)
def test_full_chunks_respect_episode_and_done_boundaries(with_truncation, trust_done, indices, terminal):
    episodes = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2])
    done = torch.tensor([False, False, False, True, False, False, True, False])
    truncated = torch.tensor([False, False, False, False, False, False, True, False])
    anchors, terminals = full_chunk_indices(
        episodes, done, truncated if with_truncation else None, 2, trust_done
    )
    assert anchors.tolist() == indices
    assert terminals.tolist() == terminal


@pytest.fixture
def tiny_dataset(tmp_path):
    repo_id = "test/rl100-fixture"
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=10,
        use_videos=False,
        features={
            OBS_IMAGE: {"dtype": "image", "shape": (32, 32, 3), "names": ["height", "width", "channel"]},
            OBS_STATE: {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            ACTION: {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
            TRUNCATED: {"dtype": "bool", "shape": (1,), "names": None},
        },
    )
    for episode, length in enumerate((5, 4)):
        for frame in range(length):
            value = episode * 10 + frame
            dataset.add_frame(
                {
                    OBS_IMAGE: np.full((32, 32, 3), value, dtype=np.uint8),
                    OBS_STATE: np.array([value, -value], dtype=np.float32),
                    ACTION: np.array([value, value + 1], dtype=np.float32),
                    REWARD: np.array([frame + 1], dtype=np.float32),
                    DONE: np.array([frame == length - 1]),
                    TRUNCATED: np.array([episode == 1 and frame == length - 1]),
                    "task": "Test a reward-labeled offline trajectory",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    features = dataset_to_policy_features(dataset.features)
    config = DiffusionConfig(
        input_features={key: features[key] for key in (OBS_IMAGE, OBS_STATE)},
        output_features={ACTION: features[ACTION]},
        device="cpu",
        n_obs_steps=2,
        n_action_steps=2,
        horizon=4,
        down_dims=(8,),
        n_groups=4,
        diffusion_step_embed_dim=16,
        spatial_softmax_num_keypoints=2,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        noise_scheduler_type="DDIM",
        num_train_timesteps=12,
        num_inference_steps=3,
    )
    return dataset, config


def test_decision_dataset_aligns_observations_actions_and_rewards(tiny_dataset):
    dataset, config = tiny_dataset
    decisions = DecisionDataset(
        dataset.repo_id, dataset.root, None, config, gamma=0.5, done_is_termination=False
    )
    assert len(decisions) == 6
    first = decisions[0]
    torch.testing.assert_close(first["observation"][OBS_STATE][:, 0], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(first["action"][:, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(first["next_observation"][OBS_STATE][:, 0], torch.tensor([1.0, 2.0]))
    assert first["reward"].item() == pytest.approx(2.0)
    assert first["discount"].item() == pytest.approx(0.25)

    terminal = decisions[3]
    torch.testing.assert_close(terminal["action"][:, 0], torch.tensor([3.0, 4.0]))
    assert terminal["reward"].item() == pytest.approx(6.5)
    assert terminal["discount"].item() == 0.0
    assert all(sample["action"][0, 0] != 12 for sample in decisions)
