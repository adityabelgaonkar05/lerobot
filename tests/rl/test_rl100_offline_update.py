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

import math
from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("diffusers", reason="install lerobot[diffusion] to run RL-100 update tests")

from diffusers import DDIMScheduler

from examples.tutorial.rl100.offline_update import ChunkCritic, DenoisingPPO


class FakeDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(1, 4, 1))

    def forward(self, sample: Tensor, timestep: Tensor, *, global_cond: Tensor) -> Tensor:
        return torch.zeros_like(sample) + self.offset


@pytest.mark.parametrize("clip_sample", [False, True])
def test_chain_matches_ddim_and_unchanged_policy_has_unit_ratios(clip_sample):
    scheduler = DDIMScheduler(num_train_timesteps=12, clip_sample=clip_sample)
    scheduler.set_timesteps(4)
    action_slice = slice(1, 3)
    ppo = DenoisingPPO(scheduler, 4, action_slice, eta=0.5, clip_epsilon=0.2)
    denoiser = FakeDenoiser()
    with torch.no_grad():
        denoiser.offset.copy_(torch.tensor([0.05, -0.1, 0.2, 0.12]).reshape(1, 4, 1))
    noise = torch.linspace(-2, 2, 16).reshape(4, 4, 1)
    conditioning = torch.zeros(4, 2)
    epsilon = denoiser.offset.detach().expand_as(noise)

    chunk, chain = ppo.sample(denoiser, conditioning, noise, torch.Generator().manual_seed(42))
    generator = torch.Generator().manual_seed(42)
    expected = noise
    for timestep in scheduler.timesteps.tolist():
        step_noise = torch.randn(noise.shape, generator=generator)
        expected = scheduler.step(epsilon, timestep, expected, eta=0.5, variance_noise=step_noise).prev_sample
    torch.testing.assert_close(chunk, expected[:, action_slice])
    assert len(chain) == 3
    first = chain[0]
    # Evaluate the Gaussian coefficients independently of the scheduler's step and variance helpers.
    alpha = scheduler.alphas_cumprod[9]
    alpha_prev = scheduler.alphas_cumprod[6]
    std = 0.5 * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)).sqrt()
    clean = (first.sample - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
    if clip_sample:
        clean = clean.clamp(-1, 1)
    mean = alpha_prev.sqrt() * clean + (1 - alpha_prev - std.square()).sqrt() * epsilon
    distribution = torch.distributions.Normal(mean[:, action_slice], std)
    torch.testing.assert_close(
        first.old_log_prob,
        distribution.log_prob(first.next_sample[:, action_slice]).sum(dim=-1),
    )

    advantage = torch.tensor([1.0, -0.25, 0.5, 2.0], requires_grad=True)
    loss, metrics = ppo.loss(denoiser, conditioning, chain, advantage)
    torch.testing.assert_close(loss, -advantage.mean())
    assert metrics["approx_kl"] == pytest.approx(0.0)
    loss.backward()
    assert advantage.grad is None
    assert denoiser.offset.grad[0, 1:3].abs().sum() > 0
    assert denoiser.offset.grad[0, [0, 3]].count_nonzero() == 0


@pytest.mark.parametrize("advantage", [1.0, -1.0])
def test_advantage_controls_sample_likelihood(advantage):
    scheduler = DDIMScheduler(num_train_timesteps=8, clip_sample=False)
    ppo = DenoisingPPO(scheduler, 2, slice(1, 3), eta=0.5, clip_epsilon=0.2)
    denoiser = FakeDenoiser()
    conditioning = torch.zeros(1, 2)
    _, chain = ppo.sample(denoiser, conditioning, torch.zeros(1, 4, 1), torch.Generator().manual_seed(7))
    loss, _ = ppo.loss(denoiser, conditioning, chain, torch.tensor([advantage]))
    loss.backward()
    with torch.no_grad():
        denoiser.offset.add_(denoiser.offset.grad, alpha=-1e-4)
    improved_loss, _ = ppo.loss(denoiser, conditioning, chain, torch.tensor([advantage]))
    assert improved_loss.item() < loss.item()


def test_clipping_applies_on_the_advantage_dependent_side():
    ppo = DenoisingPPO(DDIMScheduler(num_train_timesteps=8), 2, slice(0, 2), eta=0.5, clip_epsilon=0.2)
    denoiser = FakeDenoiser()
    conditioning = torch.zeros(2, 2)
    _, chain = ppo.sample(denoiser, conditioning, torch.zeros(2, 4, 1))
    transition = chain[0]
    log_ratio = torch.tensor([[math.log(2.0), math.log(0.5)]]).expand_as(transition.old_log_prob)
    chain = [replace(transition, old_log_prob=transition.old_log_prob - log_ratio)]
    loss, metrics = ppo.loss(denoiser, conditioning, chain, torch.tensor([1.0, -1.0]))
    assert loss.item() == pytest.approx((-1.2 - 0.5 + 2.0 + 0.8) / 4)
    assert metrics["clip_fraction"] == pytest.approx(1.0)


def test_one_step_sampler_rejected():
    with pytest.raises(ValueError, match="at least two"):
        DenoisingPPO(DDIMScheduler(), 1, slice(0, 1), eta=0.5, clip_epsilon=0.2)


def test_partial_chunk_does_not_optimize_unexecuted_actions():
    ppo = DenoisingPPO(DDIMScheduler(num_train_timesteps=8), 2, slice(1, 3), 0.5, 0.2)
    denoiser = FakeDenoiser()
    conditioning = torch.zeros(1, 2)
    _, chain = ppo.sample(denoiser, conditioning, torch.zeros(1, 4, 1), torch.Generator().manual_seed(7))
    transition = chain[0]
    chain = [replace(transition, old_log_prob=transition.old_log_prob - torch.tensor([[0.0, 10.0]]))]
    loss, metrics = ppo.loss(denoiser, conditioning, chain, torch.ones(1), torch.tensor([1]))
    assert loss.item() == pytest.approx(-1.0)
    assert metrics["clip_fraction"].item() == 0
    loss.backward()
    assert denoiser.offset.grad[0, 1] != 0
    assert denoiser.offset.grad[0, 2] == 0


def test_chunk_critic_targets_and_gradient_isolation():
    critic = ChunkCritic(observation_dim=2, action_dim=2, hidden_dim=4, expectile=0.75)
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        critic.q[0][-1].bias.fill_(1)
        critic.q[1][-1].bias.fill_(3)
        critic.target_q[0][-1].bias.fill_(2)
        critic.target_q[1][-1].bias.fill_(4)
        critic.value[-1].bias.fill_(1)
    features = torch.zeros(2, 2)
    actions = torch.zeros(2, 2, 1)

    value_loss = critic.value_loss(features, actions)
    assert value_loss.item() == pytest.approx(0.75)
    value_loss.backward()
    assert critic.value[-1].bias.grad.item() == pytest.approx(-1.5)
    assert all(parameter.grad is None for parameter in critic.q.parameters())
    critic.zero_grad(set_to_none=True)

    reward = torch.tensor([[2.0], [1.0]])
    discount = torch.tensor([[0.0], [0.25]])
    q_loss = critic.q_loss(features, actions, reward, features, discount)
    assert q_loss.item() == pytest.approx((1 + 1 + 0.25**2 + 1.75**2) / 4)
    q_loss.backward()
    assert all(parameter.grad is None for parameter in critic.value.parameters())
    assert all(parameter.grad is None for parameter in critic.target_q.parameters())
    critic.update_target(0.25)
    assert critic.target_q[0][-1].bias.item() == pytest.approx(1.75)
