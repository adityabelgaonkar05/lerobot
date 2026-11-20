# Offline RL-100 with LeRobot Diffusion Policy

This example fits implicit Q-learning critics on logged action chunks, then runs a
fixed-reference denoising policy-gradient update on a standard LeRobot Diffusion
Policy. It follows the scalar IQL advantage and per-action-step likelihood ratios
of the [official 2D chunk recipe](https://github.com/Lei-Kun/RL-100/blob/6b6afceb7eb12525398207b5d6e66908863dc385/scripts/Diffusion/Offline/2D/train_policy_image_unet_chunk_two_stage.sh).
It uses stochastic DDIM transitions during post-training and stock DDIM
inference (`eta=0`) for the exported policy. The recipe covers offline
post-training and includes a separate online PushT experiment.

## Dataset and policy

Use a LeRobot v3 Parquet dataset with one RGB camera, robot state, actions, and
scalar `next.reward` and `next.done` fields aligned to each executed action.
Use `next.truncated` to distinguish time limits when available. When the field is
absent, the recipe drops chunks ending on an ambiguous done unless you set
`--done_is_termination=true` for a dataset whose done flags unambiguously mean
true termination. Short chunks, truncated chunks and live chunks without a
recorded successor are excluded. Observation history at the beginning of an
episode uses LeRobot's standard padding.

Use `--trailing_observation=true` for the pinned PushT dataset. The
[PushT converter](https://github.com/huggingface/lerobot/blob/3c0a209/lerobot/common/datasets/push_dataset_to_hub/pusht_zarr_format.py#L153-L204)
shifts rewards and flags to the preceding action and repeats the last reward
and done flag on the final observation. The recipe validates this suffix and
excludes the final row as an action. PushT's done flags mark the recording end,
so leave `done_is_termination` at its conservative default.

Start with a local LeRobot Diffusion Policy checkpoint that includes both
processors. Its input/output features and action units must match the logged
dataset. Use epsilon prediction, DDIM and at least two reverse steps.

## Train and evaluate

From the LeRobot checkout, install the training, diffusion and PushT extras.
Use CUDA for a full image-policy run, or CPU for a small smoke run. The
following is a reproducible starting configuration for an experiment. The
example does not establish a PushT success rate or an improvement over BC.

```bash
uv sync --locked --extra training --extra diffusion --extra pusht

uv run lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --dataset.revision=7628202a2180972f291ba1bc6723834921e72c19 \
  --policy.type=diffusion --policy.noise_scheduler_type=DDIM \
  --policy.n_obs_steps=2 --policy.horizon=16 --policy.n_action_steps=8 \
  --policy.num_inference_steps=10 --policy.push_to_hub=false \
  --policy.device=cuda --steps=10000 --output_dir=outputs/pusht_bc

uv run python -m examples.tutorial.rl100.train_offline \
  --dataset_repo_id=lerobot/pusht \
  --dataset_revision=7628202a2180972f291ba1bc6723834921e72c19 \
  --policy_path=outputs/pusht_bc/checkpoints/last/pretrained_model \
  --output_dir=outputs/pusht_rl100 --device=cuda \
  --trailing_observation=true \
  --critic_steps=1000 --actor_steps=200

uv run lerobot-eval --env.type=pusht --eval.n_episodes=50 \
  --policy.path=outputs/pusht_rl100/pretrained_model --policy.device=cuda
```

Use the same evaluation settings and seeds for the original BC checkpoint.
For a local dataset, pass its v3 directory using `--dataset_root`.
Each successful run saves a native checkpoint under `pretrained_model/` and a
separate local `training_state.pt`. To continue the same fixed-reference actor
phase, pass `--resume_from` with the run directory in place of `--policy_path`,
keep the dataset and training settings fixed, choose a new output directory,
and increase the total `--actor_steps`. Use the same software stack and the same
unchanged dataset for an exact continuation.

```bash
uv sync --locked --extra dataset --extra diffusion --extra test
uv run pytest tests/rl/test_rl100_offline_update.py tests/rl/test_rl100_offline_recipe.py
```

## Online PushT experiment

The online recipe runs one episode per simulator slot per round. Each round
holds the actor fixed during collection, fits a value head and updates the
denoiser against the stored chains, then collects with the updated actor.
Time limits bootstrap from the last observation history while ending the GAE
trace. Short final chunks contribute only their executed action positions.

```bash
uv run python -m examples.tutorial.rl100.train_online \
  --policy_path=outputs/pusht_rl100/pretrained_model \
  --output_dir=outputs/pusht_rl100_online --device=cuda \
  --num_envs=4 --rounds=5 --epochs=2 --env_seed=20000
```

The output includes the native policy checkpoint and per-round rewards, episode
lengths, success flags, and update metrics in `rollout_metrics.json`. Use a
separate `lerobot-eval` run with held-out seeds to assess the exported policy
under the stock DDIM sampler.

## Accumulate simulator data for another offline round

Use the native evaluator to record a policy under new training seeds. Recordings
are independent LeRobot v3 episode logs. `--rollout_roots` adds their chunks to
the next offline round without modifying the original data. The default samples
uniformly over all usable chunks. Set `--rollout_fraction` to control the expected
fraction of sampled chunks from the recordings.

```bash
uv run lerobot-eval --env.type=pusht --eval.recording=true \
  --eval.batch_size=1 --eval.n_episodes=12 --seed=21000 \
  --policy.path=outputs/pusht_rl100/pretrained_model --output_dir=outputs/collect_round1

uv run python -m examples.tutorial.rl100.train_offline \
  --dataset_repo_id=lerobot/pusht \
  --dataset_revision=7628202a2180972f291ba1bc6723834921e72c19 \
  --trailing_observation=true \
  --rollout_roots='["outputs/collect_round1/recordings/PushT-v0_0"]' \
  --rollout_fraction=0.25 --policy_path=outputs/pusht_rl100/pretrained_model \
  --output_dir=outputs/pusht_rl100_round2 --device=cuda
```

For multi-slot collection, pass each `env_*` dataset as its own root. The
evaluator stores `next.done` for any episode boundary. The offline recipe
uses live chunks from these recordings and excludes chunks ending on ambiguous
done. Save changes to training data for a new round, with the last actor as
initialization and a newly fitted critic.

## One-step distillation experiment

Fit a student on the teacher's executed action positions using the same RGB
history and initial Gaussian noise. The student predicts the clean action
sample, which makes it deployable in one model call using stock DDIM. This
experiment uses direct teacher regression. The paper's consistency-model
objective and target-network schedule are outside this recipe.

```bash
uv run python -m examples.tutorial.rl100.distill \
  --dataset_repo_id=lerobot/pusht \
  --dataset_revision=7628202a2180972f291ba1bc6723834921e72c19 \
  --policy_path=outputs/pusht_rl100/pretrained_model \
  --output_dir=outputs/pusht_rl100_student --device=cuda
```

Evaluate the student and teacher on identical simulator seeds. The student
checkpoint has one inference step and uses sample prediction. Start any
additional PPO training from the multi-step teacher checkpoint.

## AM-Q calibration experiment

Compare the post-trained actor with the BC actor it started from. Both actors
use the same frozen encoder, processors, DDIM schedule, starting observations,
initial noise, reverse-step noise, and sampled dynamics ensemble members.
The experiment fits a chunk critic and residual feature dynamics on episode
splits, then reports average imagined min-Q alongside returns from matched
stochastic PushT rollouts. It also reports held-out dynamics error and the
error of predicting that features stay unchanged.

```bash
uv run python -m examples.tutorial.rl100.amq \
  --dataset_repo_id=lerobot/pusht \
  --dataset_revision=7628202a2180972f291ba1bc6723834921e72c19 \
  --trailing_observation=true \
  --reference_path=outputs/pusht_bc/checkpoints/last/pretrained_model \
  --candidate_path=outputs/pusht_rl100/pretrained_model \
  --output_file=outputs/pusht_rl100/amq.json
```

The score averages estimated values along a fixed-length imagined rollout.
Only the simulator comparison measures returns. Review paired agreement before
using a score to promote a reference actor.

See [RL-100: Performant Robotic Manipulation with Real-World Reinforcement
Learning](https://arxiv.org/abs/2510.14830v4) for the original method and benchmarks.
