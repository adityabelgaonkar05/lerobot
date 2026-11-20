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

import subprocess
import sys
from pathlib import Path
from shutil import copytree

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
def tiny_dataset(tmp_path, request):
    trailing_observation = getattr(request, "param", False)
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
            last_action = length - 2 if trailing_observation else length - 1
            boundary = frame >= last_action
            dataset.add_frame(
                {
                    OBS_IMAGE: np.full((32, 32, 3), value, dtype=np.uint8),
                    OBS_STATE: np.array([value, -value], dtype=np.float32),
                    ACTION: np.array([value, value + 1], dtype=np.float32),
                    REWARD: np.array([min(frame, last_action) + 1], dtype=np.float32),
                    DONE: np.array([boundary]),
                    TRUNCATED: np.array([episode == 1 and boundary]),
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


@pytest.mark.parametrize("tiny_dataset", [True], indirect=True)
def test_trailing_observation_is_never_used_as_an_action(tiny_dataset):
    dataset, config = tiny_dataset
    with pytest.raises(ValueError, match="only at the end"):
        DecisionDataset(dataset.repo_id, dataset.root, None, config, 0.5, False)
    decisions = DecisionDataset(dataset.repo_id, dataset.root, None, config, 0.5, False, True)
    assert decisions.anchors.tolist() == [0, 1, 2, 5]
    terminal = decisions[2]
    torch.testing.assert_close(terminal["action"][:, 0], torch.tensor([2.0, 3.0]))
    assert terminal["reward"].item() == pytest.approx(5.0)
    assert terminal["discount"].item() == 0.0


def test_offline_samples_accumulated_sources_without_reusing_roots(tmp_path, tiny_dataset):
    pytest.importorskip("diffusers", reason="install lerobot[diffusion] to run RL-100 training")
    from examples.tutorial.rl100.train_offline import (
        OfflineRL100Config,
        checkpoint_settings,
        load_decision_sources,
        sample_indices,
    )

    dataset, policy = tiny_dataset
    new_root = copytree(dataset.root, tmp_path / "rollout")
    cfg = OfflineRL100Config(
        dataset.repo_id,
        tmp_path / "out",
        policy_path=tmp_path / "initial",
        dataset_root=dataset.root,
        rollout_roots=[new_root],
        rollout_fraction=1.0,
    )
    demo, rollout = load_decision_sources(cfg, policy)
    indices = sample_indices(len(demo) + len(rollout), len(demo), 4, 1.0, torch.Generator().manual_seed(1))
    assert (indices >= len(demo)).all()
    assert (indices < len(demo) + len(rollout)).all()
    assert checkpoint_settings(cfg, [demo, rollout])["rollout_datasets"][0]["dataset_root"] == str(
        new_root.resolve()
    )
    cfg.rollout_roots = [dataset.root]
    with pytest.raises(ValueError, match="distinct root"):
        load_decision_sources(cfg, policy)


def test_offline_cli_exports_policy_and_resumes_exactly(tmp_path, tiny_dataset):
    pytest.importorskip("diffusers", reason="install lerobot[diffusion] to run RL-100 training")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    dataset, config = tiny_dataset
    decisions = DecisionDataset(
        dataset.repo_id, dataset.root, None, config, gamma=0.99, done_is_termination=False
    )
    policy = DiffusionPolicy(config)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    batch = decisions[1]
    bc_batch = {
        **{key: value.unsqueeze(0) for key, value in batch["observation"].items()},
        ACTION: torch.cat((batch["action"][:1], batch["action"], batch["action"][-1:])).unsqueeze(0),
        "action_is_pad": torch.zeros(1, config.horizon, dtype=torch.bool),
    }
    loss, _ = policy.eval()(preprocessor(bc_batch))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    loss.backward()
    optimizer.step()
    initial = tmp_path / "bc"
    policy.save_pretrained(initial)
    preprocessor.save_pretrained(initial)
    postprocessor.save_pretrained(initial)

    def run(output: Path, actor_steps: int, source_option: str, source: Path):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.tutorial.rl100.train_offline",
                f"--dataset_repo_id={dataset.repo_id}",
                f"--dataset_root={dataset.root}",
                f"--output_dir={output}",
                f"--{source_option}={source}",
                f"--actor_steps={actor_steps}",
                "--critic_steps=2",
                "--batch_size=2",
                "--critic_hidden_dim=8",
                "--actor_lr=0.0001",
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    partial, resumed, uninterrupted = (tmp_path / name for name in ("partial", "resumed", "uninterrupted"))
    run(partial, 1, "policy_path", initial)
    run(resumed, 2, "resume_from", partial)
    run(uninterrupted, 2, "policy_path", initial)

    policy_class = get_policy_class("diffusion")
    resumed_policy = policy_class.from_pretrained(resumed / "pretrained_model", strict=True)
    uninterrupted_policy = policy_class.from_pretrained(uninterrupted / "pretrained_model", strict=True)
    before = policy.state_dict()
    after = resumed_policy.state_dict()
    assert any(not torch.equal(value, before[key]) for key, value in after.items() if ".unet." in key)
    training_state = torch.load(resumed / "training_state.pt", weights_only=True)
    for key, value in policy.diffusion.unet.state_dict().items():
        torch.testing.assert_close(training_state["reference"][key], value, rtol=0, atol=0)
    for key, value in uninterrupted_policy.state_dict().items():
        torch.testing.assert_close(after[key], value, rtol=0, atol=0)
        if ".rgb_encoder." in key:
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)

    export = resumed / "pretrained_model"
    loaded_pre, loaded_post = make_pre_post_processors(resumed_policy.config, pretrained_path=export)
    observation = {key: value.unsqueeze(0) for key, value in batch["observation"].items()}
    noise = torch.zeros(1, config.horizon, 2)
    actual = loaded_post(resumed_policy.predict_action_chunk(loaded_pre(observation), noise=noise))
    expected = postprocessor(
        uninterrupted_policy.predict_action_chunk(preprocessor(observation), noise=noise)
    )
    assert actual.shape == (1, config.n_action_steps, 2)
    torch.testing.assert_close(actual, expected)


def test_distillation_exports_a_native_one_step_checkpoint(tmp_path, tiny_dataset):
    pytest.importorskip("diffusers", reason="install lerobot[diffusion] to run RL-100 distillation")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    dataset, config = tiny_dataset
    policy = DiffusionPolicy(config).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    initial = tmp_path / "teacher"
    policy.save_pretrained(initial)
    preprocessor.save_pretrained(initial)
    postprocessor.save_pretrained(initial)
    output = tmp_path / "student"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.tutorial.rl100.distill",
            f"--dataset_repo_id={dataset.repo_id}",
            f"--dataset_root={dataset.root}",
            f"--policy_path={initial}",
            f"--output_dir={output}",
            "--steps=2",
            "--batch_size=2",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    student = DiffusionPolicy.from_pretrained(output / "pretrained_model", strict=True)
    assert student.config.prediction_type == "sample"
    assert student.config.num_inference_steps == 1
    assert any(
        not torch.equal(value, policy.diffusion.unet.state_dict()[name])
        for name, value in student.diffusion.unet.state_dict().items()
    )
    for name, value in student.diffusion.rgb_encoder.state_dict().items():
        torch.testing.assert_close(value, policy.diffusion.rgb_encoder.state_dict()[name])
    decisions = DecisionDataset(dataset.repo_id, dataset.root, None, config, 0.99, False)
    observations = {key: value.unsqueeze(0) for key, value in decisions[0]["observation"].items()}
    pre, post = make_pre_post_processors(student.config, pretrained_path=output / "pretrained_model")
    predictions = []
    hook = student.diffusion.unet.register_forward_hook(lambda module, inputs, out: predictions.append(out))
    with torch.no_grad():
        action = student.predict_action_chunk(pre(observations), noise=torch.zeros(1, config.horizon, 2))
    hook.remove()
    assert len(predictions) == 1
    start = config.n_obs_steps - 1
    expected = predictions[0][:, start : start + config.n_action_steps].clamp(-1, 1)
    torch.testing.assert_close(post(action), postprocessor(expected))
