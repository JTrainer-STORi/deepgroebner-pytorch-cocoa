# Hyperparameter Sweeps

A sweep runs training several times with different settings and compares the
results. This package supports shell loops around `run_training.py`, Python loops
around `train(cfg)`, and Weights & Biases sweep agents. The examples below vary
learning rate and entropy coefficient while keeping the training budget and
evaluation protocol fixed.

Install the package as described in [INSTALL.md](../../INSTALL.md). Run the shell
examples from the repository root with that Python environment active. Python
examples also work from notebooks or other projects using the installed package.

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

Prepare the shared corpus once before starting trials:

```bash
python generate_corpus.py \
  --ideal-distribution 3-5-3-weighted \
  --examples 20 \
  --seed 10000
```

Omitting `--output-path` puts the corpus in the default cache that training
reuses. Run all trials from this same working directory. The cache is under
`eval_corpora/greedy_eval/`, independently of `training.output_directory`.
Prebuilding it is especially useful before parallel trials, since cache creation
does not use a file lock. Hold the corpus-generation and environment settings
fixed while tuning the policy or optimizer.

Greedy-evaluation returns are negative costs: **maximize** the selection score
(`-100` is better than `-200`). The standalone `evaluate_model.py` reports positive
work costs, for which lower is better. See the
[corpus and evaluation tutorial](eval-corpora-and-greedy-eval.md) for both workflows.

## Run A Local Grid Without W&B

This Bash loop runs six trials sequentially. Each trial has its own output
directory and final JSON summary. Choose a fresh `sweep_root` for a new search.

```bash
set -e
sweep_root="runs/sweeps/grid-001"

for learning_rate in 0.0001 0.0003 0.001; do
  for entropy in 0.0 0.01; do
    trial_dir="${sweep_root}/lr_${learning_rate}__entropy_${entropy}"
    python run_training.py \
      --env.ideal-dist 3-5-3-weighted \
      --env.env-seed 12 \
      --training.training-seed 24 \
      --training.num-envs 4 \
      --training.vectorization-mode sync \
      --training.pytorch-num-threads 1 \
      --training.num-steps-per-update 256 \
      --training.total-timesteps 65536 \
      --training.hparam-eval-interval-steps 16384 \
      --training.hparam-eval-num-examples 20 \
      --training.hparam-eval-seed 10000 \
      --training.hparam-eval-on-train-env True \
      --training.hparam-eval-on-nopq-env False \
      --training.learning-rate "${learning_rate}" \
      --training.entropy-coefficient "${entropy}" \
      --training.track False \
      --training.console-verbosity quiet \
      --training.output-directory "${trial_dir}" \
      --training.hparam-metrics-out "${trial_dir}/summary.json"
  done
done
```

Synchronous environments and one PyTorch CPU thread keep this example's CPU
allocation small. Adjust those settings for your machine, keeping them consistent
across candidates. Other settings retain their configuration defaults; select
actor and critic heads explicitly if you also want to compare architectures.

These examples retain the `training.cuda` default, so training uses a GPU when
available. Add `--training.cuda False` or set `cfg.training.cuda = False` to run
on the CPU.

For each trial, the paths above produce:

| Output | Location |
| --- | --- |
| Final summary | `<trial_dir>/summary.json` |
| Scalar logs and diagnostics | `<trial_dir>/runs/<run_name>/` |
| Actor checkpoints and config snapshots | `<trial_dir>/models/actors/` |
| Critic checkpoints, if enabled | `<trial_dir>/models/critics/` |

`hparam_metrics_out` is an exact file path, not a path relative to
`output_directory`; the example deliberately sets both. Its JSON is written once
when training finishes, including a controlled runtime-budget stop. Reusing that
file path replaces its previous summary. An interrupted or failed trial may not
write a summary. If you set `group`, local logs, checkpoints, and W&B run
files share the root `<output_directory>/<group>/`; the explicit summary path
stays as configured. See [grouped outputs](training-runs.md#grouped-outputs).

### Compare The Completed Trials

Save this code as a Python script or run it in a notebook from the same working
directory. It sorts trials by their latest evaluation score, highest first:

```python
import json
from pathlib import Path

sweep_root = Path("runs/sweeps/grid-001")
ranked = []
for summary_path in sweep_root.glob("*/summary.json"):
    summary = json.loads(summary_path.read_text())
    score = summary["greedy_eval_last_selection_metric"]
    if score is None:
        print(f"Skipping {summary_path}: no valid latest evaluation score")
        continue
    ranked.append((score, str(summary_path)))

for score, path in sorted(ranked, reverse=True):
    print(f"{score:.6f}\t{path}")
```

The relevant summary fields are:

| Field | Meaning |
| --- | --- |
| `greedy_eval_last_selection_metric` | Score from the latest evaluation round; `null` when unavailable or invalid |
| `greedy_eval_best_selection_metric` | Highest valid evaluation score seen during the trial |
| `greedy_eval_best_step` | Environment step at which that best score was recorded |
| `greedy_eval_best_actor_path` | Saved checkpoint associated with that best score, when actor saving is enabled |
| `selection_metric` | Latest evaluation score when available, otherwise the average return over the last 100 completed training episodes; `null` if neither exists |
| `global_step`, `timed_out` | Actual training progress and whether the configured runtime budget stopped training |

Use the explicit evaluation field when ranking trials on the same evaluation
protocol. `selection_metric` can fall back to training returns, which should not
be mixed with evaluation scores. Check completion/truncation statistics and
actual step counts before comparing runs. If NoPQ evaluation is enabled, its
score becomes the selection metric; an evaluation with truncated episodes is
invalid for checkpoint selection.

To compare the best checkpoint from each trial instead, sort by
`greedy_eval_best_selection_metric` and use `greedy_eval_best_actor_path`. The
latest-score ranking above does not necessarily select that checkpoint. Final
checkpoint paths are returned as `TrainingResult.checkpoint_paths`; they are
separate from the summary's best-evaluation paths. Complete settings are saved in
the checkpoints' companion `__config.json` files.

## Run Trials From Python Or A Notebook

Use the installed package to run a smaller four-trial grid directly. This example
also prepares the evaluation cache in the caller's working directory:

```python
from itertools import product
from pathlib import Path

from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.corpus_generation import CorpusConfig, generate_corpus
from deepgroebner_pytorch_cocoa.ppo_training import train

generate_corpus(CorpusConfig(
    ideal_distribution="3-5-3-weighted", examples=20, seed=10000,
))

cfg = Config()
cfg.env.ideal_dist = "3-5-3-weighted"
cfg.env.env_seed = 12
cfg.training.training_seed = 24
cfg.training.num_envs = 4
cfg.training.vectorization_mode = "sync"
cfg.training.pytorch_num_threads = 1
cfg.training.num_steps_per_update = 256
cfg.training.total_timesteps = 65536
cfg.training.hparam_eval_interval_steps = 16384
cfg.training.hparam_eval_num_examples = 20
cfg.training.hparam_eval_seed = 10000
cfg.training.hparam_eval_on_train_env = True
cfg.training.hparam_eval_on_nopq_env = False
cfg.training.track = False

sweep_root = Path("runs/sweeps/python-001").resolve()
for learning_rate, entropy in product([0.0001, 0.0003], [0.0, 0.01]):
    trial_dir = sweep_root / f"lr_{learning_rate}__entropy_{entropy}"
    cfg.training.learning_rate = learning_rate
    cfg.training.entropy_coefficient = entropy
    cfg.training.output_directory = str(trial_dir)
    cfg.training.hparam_metrics_out = str(trial_dir / "summary.json")
    result = train(cfg)
    print(learning_rate, entropy, result.summary["greedy_eval_last_selection_metric"])
```

Each `train(cfg)` call copies the configuration internally, so changing `cfg`
between trials is supported. Synchronous vectorization works within a notebook
without starting environment worker processes. Each call blocks until its trial
finishes, and returns the same summary that `hparam_metrics_out` saves.

## Run A W&B Sweep

Install tracking support and log in using the same environment:

```bash
python -m pip install -e ".[tracking]"
wandb login
```

Prepare the shared corpus as above, then save the following as `sweep.yaml` in
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
  - --training.hparam-eval-interval-steps=16384
  - --training.hparam-eval-num-examples=20
  - --training.hparam-eval-seed=10000
  - --training.hparam-eval-on-train-env=True
  - --training.hparam-eval-on-nopq-env=False
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
Keep `${args}` when sweeping booleans too: this package expects explicit
`True`/`False` values, while `${args_no_boolean_flags}` changes their form.
`log_uniform_values` samples learning rates between the stated numeric bounds.
See W&B's [sweep configuration reference](https://github.com/wandb/docs/blob/main/models/sweeps/sweep-config-keys.mdx)
for command macros, distributions, and run caps.

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
Online tracking is needed to supply these scores to the sweep. The example
disables the package's timeout-to-offline fallback so a failed connection does
not silently produce an offline trial. Evaluation must be enabled and reached
within the trial budget for the objective to be logged.

This objective name replaces `charts/greedy_eval_selection_metric`. Update older
sweep configurations when starting a new sweep. See the
[metric sections](training-runs.md#metric-sections) for the full logging layout;
final-summary JSON fields such as `greedy_eval_last_selection_metric` retain
their existing names.

The shared output root is safe for local logs and checkpoints because their names
include a generated run ID. With the group above, outputs share the root
`runs/sweeps/wandb-001/sweep-001/`: local logs go in `runs/`, checkpoints in
`models/`, and W&B run files in `wandb/`. The external `wandb agent` may also
create its own launcher files in its working directory; those are separate from
the trainer's outputs.

This example leaves `hparam_metrics_out` unset so parallel trials do not overwrite
one shared JSON file. If you add it, give every trial a distinct file path.
During a W&B sweep the package preserves the agent's sampled configuration;
complete resolved settings remain available in local checkpoint config snapshots.

## Expand The Search

You can vary other CLI/configuration fields such as `training.clip-coef`,
`training.num-epochs-per-update`, or actor head widths. Check supported values
with `python run_training.py --help`. When changing rollout length, environment
count, or minibatch count, account for the resulting batch size and actual number
of updates. Keep a comparable evaluation schedule across trials.

For parallel searches, start additional agents or independent trial processes
with their own allocated CPU/GPU resources. The number of environments and
PyTorch thread settings apply to every trial. `training.max_runtime_seconds` is
checked between updates; it does not interrupt a rollout, evaluation, or hung
native call. Scheduler termination can also prevent the final JSON from being
written.

After the initial search, repeat promising configurations with several training
seeds while keeping evaluation fixed. Use a separate corpus with a different
evaluation seed for the final comparison, following the
[corpus and evaluation tutorial](eval-corpora-and-greedy-eval.md).
