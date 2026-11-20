# RL-100 on PushT

## Reference settings

The offline objective selects `per_step:scalar_iql` from the official 2D
implementation at `6b6afceb7eb12525398207b5d6e66908863dc385`. The launcher
accepts that override via `CHUNK_LOSS_MODE_COMBOS` and defaults to a sweep with
scalar chunk ratios. In the selected `final_reward=True` mode, the official
actor samples a full denoising chain, computes the IQL advantage for the final
sampled action chunk, and broadcasts the advantage over executed action
positions. It sums coordinate log-probabilities before exponentiating and
clips one ratio per position. It takes an optimizer step at every denoising
timestep. [Official launcher](https://github.com/Lei-Kun/RL-100/blob/6b6afceb7eb12525398207b5d6e66908863dc385/scripts/Diffusion/Offline/2D/train_policy_image_unet_chunk_two_stage.sh)
· [Official actor update](https://github.com/Lei-Kun/RL-100/blob/6b6afceb7eb12525398207b5d6e66908863dc385/RL-100/rl_100/unidpg/uni_ppo.py#L744-L1040)

| Setting | Official configured path | LeRobot example |
| --- | --- | --- |
| Observation and chunk | Global conditioning with the RGB encoder. Launcher defaults to three observations and sixteen executed positions. The config removes leading observation-aligned action positions. | Stock Diffusion Policy conditioning and the native action slice starting at `n_obs_steps - 1`. The pilot uses two observations and eight executed actions from a horizon of sixteen. |
| Sampling | Epsilon DDIM with 100 cosine training timesteps, 10 reverse steps, offset 1, and a sampling standard-deviation floor of 0.0067. The launcher allows a standard-deviation cap. | Stock Diffusers epsilon DDIM with its default offset, scheduler variance scaled by `eta=0.5`, and zero-variance steps excluded from PPO. Evaluation uses deterministic DDIM transitions given the initial noise. |
| Advantage and optimizer | Twin-Q IQL with normalized advantages, configured clipping and schedules, and one optimizer step per denoising transition. | Twin-Q IQL with unnormalized detached advantages, fixed clipping, and one optimizer step on the mean surrogate across stochastic denoising transitions and executed positions. |
| Offline reference | Periodic learned-dynamics evaluation can refresh the reference. | The offline reference stays fixed for the entire update block. Online collection refreshes after each completed rollout batch. AM-Q is a separate experiment. |

The sampler test checks the Gaussian coefficients against a direct calculation
and the complete chain against Diffusers. The tests also cover scalar
advantage sign, clipping and omission of unexecuted positions. This is an
implementation of a selected objective with the listed deviations. [Official
2D epsilon config](https://github.com/Lei-Kun/RL-100/blob/6b6afceb7eb12525398207b5d6e66908863dc385/RL-100/rl_100/config/rl100_2d_epsilon.yaml)
· [Official likelihood scheduler](https://github.com/Lei-Kun/RL-100/blob/6b6afceb7eb12525398207b5d6e66908863dc385/RL-100/rl_100/unidpg/diffusion_policy/diffusers_patch/ddim_with_logprob_dpok.py)

## Three-seed CPU pilot

The experiment compares three branches from the same initial Diffusion Policy
checkpoint for each training seed. BC takes no further optimizer steps. Extra BC
updates the whole policy for 100 further steps. RL-100 fits Q/V for 300 steps
and takes 100 denoiser updates using the same batch size and actor learning
rate as extra BC with its RGB encoder frozen. All branches use the same
training data and environment seeds. Training uses zero simulator steps.
The pilot used every PushT demonstration and fixed hyperparameters before
evaluation. The simulator seeds were reserved for evaluation.

From a LeRobot checkout, install the extras and run the commands below with a
new output root. The original runs used Python 3.13, PyTorch 2.11, Diffusers
0.39, PyAV 15.1 and `gym-pusht` 0.1.8 on an Apple M4. The logged dataset is the
MIT-licensed [PushT v3 snapshot](https://huggingface.co/datasets/lerobot/pusht/tree/7628202a2180972f291ba1bc6723834921e72c19).
All 206 episodes in this snapshot repeat the final transition's done flag and
reward on the trailing observation. The converter marks demonstration ends
with done and every recorded success flag is false. The offline
recipe accepted 23,796 of 25,650 row anchors for eight-action chunks and used
zero terminal targets. [PushT converter](https://github.com/huggingface/lerobot/blob/3c0a209/lerobot/common/datasets/push_dataset_to_hub/pusht_zarr_format.py#L153-L204)

```bash
uv sync --locked --extra training --extra diffusion --extra pusht

revision=7628202a2180972f291ba1bc6723834921e72c19
root=outputs/rl100_pusht_benchmark
export OMP_NUM_THREADS=3
export SDL_VIDEODRIVER=dummy

for seed in 42 43 44; do
  uv run lerobot-train \
    --dataset.repo_id=lerobot/pusht --dataset.revision="$revision" --dataset.video_backend=pyav \
    --policy.type=diffusion --policy.noise_scheduler_type=DDIM --policy.device=cpu \
    --policy.n_obs_steps=2 --policy.horizon=16 --policy.n_action_steps=8 --policy.num_inference_steps=5 \
    --policy.down_dims='[32,64]' --policy.diffusion_step_embed_dim=32 \
    --policy.spatial_softmax_num_keypoints=8 --policy.resize_shape='[32,32]' \
    --policy.pretrained_backbone_weights=null --policy.use_group_norm=true \
    --policy.scheduler_warmup_steps=50 --policy.push_to_hub=false \
    --batch_size=16 --num_workers=0 --steps=800 --seed="$seed" --output_dir="$root/bc-$seed"

  bc="$root/bc-$seed/checkpoints/last/pretrained_model"
  uv run lerobot-train \
    --dataset.repo_id=lerobot/pusht --dataset.revision="$revision" --dataset.video_backend=pyav \
    --policy.path="$bc" --policy.device=cpu --policy.push_to_hub=false --policy.optimizer_lr=0.000003 \
    --policy.scheduler_name=constant --policy.scheduler_warmup_steps=0 \
    --batch_size=16 --num_workers=0 --steps=100 --seed="$seed" --output_dir="$root/bc-extra-$seed"

  uv run python -m examples.tutorial.rl100.train_offline \
    --dataset_repo_id=lerobot/pusht --dataset_revision="$revision" --trailing_observation=true \
    --video_backend=pyav \
    --policy_path="$bc" --output_dir="$root/rl-$seed" \
    --device=cpu --seed="$seed" --batch_size=16 --critic_steps=300 --actor_steps=100

  for kind in bc bc-extra rl; do
    case "$kind" in
      bc) policy_path="$bc" ;;
      bc-extra) policy_path="$root/bc-extra-$seed/checkpoints/last/pretrained_model" ;;
      rl) policy_path="$root/rl-$seed/pretrained_model" ;;
    esac
    uv run lerobot-eval --env.type=pusht --env.episode_length=300 \
      --eval.n_episodes=50 --eval.batch_size=5 --eval.use_async_envs=false --seed=10000 \
      --policy.path="$policy_path" --policy.device=cpu --output_dir="$root/eval-$kind-$seed"
  done
done
```

### Pilot results

| Seed | Checkpoint | Successes | Mean return | Mean peak reward | Training stage (s) | Peak RSS (MiB) | Evaluation (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | BC | 0/50 | 6.358 | 0.095 | 212.0 | — | 60.0 |
| 42 | Extra BC | 0/50 | 6.984 | 0.098 | 69.5 | 966 | 57.3 |
| 42 | Offline RL-100 | 0/50 | 6.918 | 0.094 | 237.5 | 777 | 73.0 |
| 43 | BC | 0/50 | 2.913 | 0.076 | 256.6 | 957 | 57.9 |
| 43 | Extra BC | 0/50 | 3.010 | 0.073 | 80.5 | 961 | 67.2 |
| 43 | Offline RL-100 | 0/50 | 2.971 | 0.072 | 251.8 | 792 | 67.5 |
| 44 | BC | 0/50 | 4.736 | 0.091 | 507.8 | 883 | 97.0 |
| 44 | Extra BC | 0/50 | 5.538 | 0.099 | 121.3 | 950 | 99.5 |
| 44 | Offline RL-100 | 0/50 | 4.880 | 0.091 | 278.6 | 783 | 70.5 |

Success was at the floor across all 450 held-out evaluations. Each 0/50
measurement has a 95% Wilson interval of 0–7.13%. Steps to success were
unavailable. Seed-averaged returns were 4.669 for BC, 5.177 for extra BC
and 4.923 for offline RL-100. The compact pilot establishes that native
training, export and closed-loop evaluation work. It leaves learning
benefit unresolved.

For seed 42, 28 of 50 episodes in each arm stayed below 0.01 peak reward.
The highest peak reward from the offline checkpoint was 0.744. The paired
mean return differences for offline RL-100 versus extra BC were −0.066,
−0.039, and −0.658 for seeds 42–44. Episode-pair bootstrap intervals were
[−0.962, 0.888], [−0.453, 0.347], and [−1.819, 0.240], respectively.
The intervals condition on each trained checkpoint and use 20,000 resamples
of the common evaluation seeds.

The training times describe each stage in isolation from its upstream
checkpoint. Extra BC and RL-100 both start after the BC time in their
seed's first row. Timing and peak resident memory reflect concurrent
jobs on the same host. The seed-42 BC log measured the training loop
duration only and lacked an RSS measurement.

## Larger-policy follow-up

The single-seed follow-up tested whether additional model capacity and initial
BC training raised success above the pilot's floor. It used the same PushT
data, action horizon, sampler and post-training budgets, with training seed
45 and simulator seeds 10100–10149. Use the pilot commands with these changes
to the initial BC run:

```text
--policy.device=mps
--policy.down_dims=[128,256,512]
--policy.diffusion_step_embed_dim=64
--policy.spatial_softmax_num_keypoints=16
--policy.resize_shape=[64,64]
--policy.scheduler_warmup_steps=150
--steps=2000 --save_freq=1000 --seed=45
```

The model had 27.4 million parameters. The 100-step extra BC control,
offline post-training and evaluation used CPU. Select seed 45 for both
post-training arms and change the evaluation seed to 10100.

| Checkpoint | Successes | Mean return | Mean peak reward | Training stage (s) | Peak host RSS (MiB) | Evaluation (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BC | 0/50 | 3.637 | 0.122 | 670.3 | 1117 | 63.6 |
| Extra BC | 0/50 | 3.801 | 0.131 | 170.5 | 1215 | 102.3 |
| Offline RL-100 | 0/50 | 3.124 | 0.117 | 608.2 | 1071 | 50.0 |

The offline checkpoint's paired mean return difference versus extra BC was
−0.677, with a 95% episode-pair bootstrap interval of [−1.898, 0.464].
The follow-up still had a success floor. A comparison from a reliably
successful BC checkpoint remains open.

## Online and distillation ablations

The seed-42 online run used `train_online` with the offline checkpoint,
three rounds, two epochs per round, four simulator slots, and batch size 16.
It collected 3,600 simulator steps on seeds 20000–20011, starting a new
episode in every slot and refreshing the actor after each round. A separate
13-step run with two slots and two rounds exercised the partial-chunk path.
The dataset-accumulation smoke test recorded a 24-step episode under seed
21000. It yielded 16 usable chunks. A second offline run trained on those
chunks with `--rollout_fraction=1 --critic_steps=2 --actor_steps=2 --batch_size=2`.

The one-step student used the seed-42 offline teacher, 250 distillation
steps, batch size 16, and learning rate 0.0001. It was evaluated through
the same stock `lerobot-eval` command as the teacher.

| Seed-42 policy | Successes | Mean episode return | Mean peak reward |
| --- | ---: | ---: | ---: |
| Offline teacher | 0/50 | 6.918 | 0.094 |
| Online, three rounds | 0/50 | 6.793 | 0.093 |
| One-step student | 0/50 | 3.757 | 0.073 |

For the latency probe, the policy received preprocessed PushT reset
observations with the first frame repeated as history. Teacher and student
received identical Gaussian noise for each observation. The probe timed 30
calls of `predict_action_chunk` on CPU with two Torch threads, discarded the
first five timings, and used seeds starting at 10050 for resets and noise.
The elapsed time includes observation encoding and action generation.
These timings were taken while other benchmark processes were active.

| Batch size | Teacher median / p95 (ms) | Student median / p95 (ms) | Normalized action MSE |
| ---: | ---: | ---: | ---: |
| 1 | 53.5 / 85.7 | 24.5 / 36.5 | 0.0370 |
| 5 | 103.1 / 139.9 | 63.0 / 116.4 | 0.0428 |

The student reduced model latency in this probe. Its closed-loop return
was lower than the teacher's.

## AM-Q calibration

For each training seed, run the README's AM-Q command with the corresponding
BC and offline checkpoints and add `--fit_steps=120 --batch_size=16 --seed=SEED
--video_backend=pyav`. The fit uses the demonstration train episodes only.
It measures feature-prediction MSE on held-out episodes and starts paired
stochastic rollouts on 16 fresh PushT seeds starting at 30000. Scores
average min-Q across three imagined decisions. Simulator returns below are
ordinary undiscounted returns from the same stochastic actors and initial
states.

| Seed | BC AM-Q | Offline AM-Q | BC return | Offline return | Dynamics MSE | Persistence MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 2.4411 | 2.4395 | 2.556 | 2.135 | 0.000736 | 0.006470 |
| 43 | 2.5152 | 2.5241 | 2.067 | 1.860 | 0.000833 | 0.006378 |
| 44 | 2.5508 | 2.5484 | 4.372 | 4.264 | 0.000642 | 0.005939 |

The dynamics fit reduced one-decision feature error relative to persistence
for all three seeds. The AM-Q direction disagreed with actual returns in
seed 43. The example leaves promotion to a future experiment with stronger
policies, additional seeds and a validated calibration criterion.

## Tests

The focused regression run passed 51 tests across the RL-100 examples,
native simulator, Diffusion processors, and plugin discovery. Four existing
processor tests required CUDA. In the base tier with the optional dataset and
Diffusers packages hidden, the pure IQL, GAE, rollout and AM-Q tests ran
without those dependencies.
