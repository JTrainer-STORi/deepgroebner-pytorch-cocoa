# Hyperparameter Sweeps

A sweep runs training several times with different settings and compares the
results. This package supports Weights & Biases sweep agents.
## Choose A Consistent Evaluation Protocol

The examples use a small initial search budget:

| Setting | Value |
| --- | --- |
| Ideal distribution | `3-5-3-weighted` |
| Training environment seed | `12` |
| Training random seed | `24` |
| Environments / steps per update per environment | `4` / `256` |
| Total environment steps per trial | `65536` |
| Greedy-evaluation interval | `16384` environment steps |
| Evaluation examples / base seed | `20` / `10000` |
| Evaluation target | Training environment only |

The batch size is `num_envs * num_steps_per_update`, so these trials perform 64
PPO updates and four evaluation rounds, including one at the final training step.
Training rounds its budget down to complete batches. Evaluation happens after
updates; there is no extra evaluation forced at the end of training. Keep the
same evaluation target, corpus, and budget across candidates.

Prepare the shared evaluation corpus once before starting trials:

```bash
python generate_corpus.py \
  --ideal-distribution 3-5-3-weighted \
  --examples 20 \
  --seed 10000 \
  --output-path eval_corpora/sweep_shared.jsonl
```

Each trial below selects this file explicitly with `--training.greedy-eval-corpus`,
so all trials use every row in the same order. Relative corpus paths use the
caller's working directory, independently of `training.output_directory`; use an
absolute path for trials launched elsewhere. Hold the corpus and environment
settings fixed while tuning the policy or optimizer. For an individual training
run, the trainer can prepare its evaluation corpus automatically.


## Run A W&B Sweep

Install tracking support and log in using the same environment:

```bash
python -m pip install -e ".[tracking]"
wandb login
```

Prepare the shared evaluation corpus as above, then save the following as `sweep.yaml` in
the repository root. It defines a random search capped at 12 trials:

```yaml
program: run_training.py
method: random
run_cap: 12
metric:
  name: evaluation_greedy/selection_metric
  goal: maximize
parameters:
  training.learning-rate:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.001
  training.entropy-coefficient:
    values: [0.0, 0.001, 0.01]
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - --env.ideal-dist=3-5-3-weighted
  - --env.env-seed=12
  - --training.training-seed=24
  - --training.num-envs=4
  - --training.vectorization-mode=sync
  - --training.pytorch-num-threads=1
  - --training.num-steps-per-update=256
  - --training.total-timesteps=65536
  - --training.greedy-eval-interval-steps=16384
  - --training.greedy-eval-corpus=eval_corpora/sweep_shared.jsonl
  - --training.greedy-eval-on-train-env=True
  - --training.greedy-eval-on-nopq-env=False
  - --training.track=True
  - --training.wandb-project=deepgroebner-sweeps
  - --training.group=sweep-001
  - --training.wandb-offline-on-timeout=False
  - --training.output-directory=runs/sweeps/wandb-001
  - ${args}
```

`${args}` passes sampled parameters as `--training.learning-rate=...` and similar
CLI arguments. The trainer builds `Config` from these arguments; it does not read
settings back from `wandb.config`. Use flat parameter names matching the CLI.
See [weights & biases tutorial](https://docs.coreweave.com/products/wandb/sweeps) for more information on sweeps and .yaml files.


Create the sweep and start an agent:

```bash
wandb sweep --project deepgroebner-sweeps sweep.yaml
wandb agent --count 12 YOUR_ENTITY/deepgroebner-sweeps/SWEEP_ID
```

Use the entity and sweep ID printed by the first command. The agent's `--count`
limits that agent; `run_cap` limits the whole search. Keep the project in the YAML
training arguments consistent with the project used to create the sweep.

Each trial runs in a separate process and logs
`evaluation_greedy/selection_metric` at scheduled, valid evaluation rounds.

The shared output root is safe for local logs and checkpoints because their names
include a generated run ID. With the group above, outputs share the root
`runs/sweeps/wandb-001/sweep-001/`: local logs go in `runs/`, checkpoints in
`models/`, and W&B run files in `wandb/`. The external `wandb agent` may also
create its own launcher files in its working directory; those are separate from
the trainer's outputs.


## Expand The Search

You can vary other CLI/configuration fields such as `training.clip-coef`,
`training.num-epochs-per-update`, or actor head widths. Check supported values
with `python run_training.py --help`. When changing rollout length, environment
count, or minibatch count, account for the resulting batch size and actual number
of updates. Keep a comparable evaluation schedule across trials.

For parallel searches, start additional agents or independent trial processes
with their own allocated CPU/GPU resources. 


After the initial search, repeat promising configurations with several training
seeds while keeping the evaluation corpus fixed. Use a separate test corpus for
the final comparison, generated with a different seed and an explicit
`--output-path test_corpora/3_5_3_seed20000.jsonl`. Keep this test corpus out of
hyperparameter and checkpoint selection. See the
[evaluation and test corpora tutorial](eval-corpora-and-greedy-eval.md).
