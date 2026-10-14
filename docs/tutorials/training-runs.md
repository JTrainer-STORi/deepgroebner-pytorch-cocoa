# Training Runs

This tutorial shows how to launch training runs for a chosen ideal distribution from the command line.

## Before You Start

Install the package following [INSTALL.md](../../INSTALL.md).

## Choose Heads, Then Override Settings

After installing the package, run commands from the repository root using the Python environment where you installed it. The training script uses the configuration dataclasses directly:

```bash
python run_training.py
```

The default actor uses a `symmetric-per-entity-mlp` (SPEM) pair head, with both
`actor.state_embedding_head` and `actor.state_features_embedding_head` set to
`None`. A pair head is required. The critic has independent defaults: a Deep Set
state head and an MLP state-feature head.

Choose components independently:

| Option | Choices | Actor default |
| --- | --- | --- |
| `--actor.pair-embedding-head` | `per-entity-mlp`, `symmetric-per-entity-mlp`, `alphastar-buchberger-pair-transformer` | `symmetric-per-entity-mlp` |
| `--actor.state-embedding-head` | `deep-set`, `None` | `None` |
| `--actor.state-features-embedding-head` | `mlp`, `None` | `None` |
| `--actor.pair-combiner` | `film`, `sqa`, `concat` | `film` |
| `--actor.state-combiner` | `film`, `concat` | `film` |

Without either actor state head, the model automatically scores pair embeddings
directly. With either state head, `pair-combiner` controls how state conditions
pair scores. When both state heads are enabled, `state-combiner` combines their
outputs first.

All user-configurable fields in `config_dataclasses.py` are exposed under
`--actor.*`, `--critic.*`, `--env.*`, and `--training.*`. Unspecified settings retain
their dataclass defaults; selecting a head does not overwrite other choices.
Input dimensions and batch sizes marked `init=False` are calculated at runtime.
Use `True`/`False` for boolean options and `None` to disable optional values.
Hyphenated names below correspond to underscores in the Python dataclasses.

```bash
# Inspect all options and defaults.
python run_training.py --help

# PEM with an MLP state-feature head and custom widths / learning rate.
python run_training.py \
  --actor.pair-embedding-head per-entity-mlp \
  --actor.state-features-embedding-head mlp \
  --actor.pair-embedding-head-out-dim 128 \
  --training.learning-rate 0.0001

# ASB with Deep Set state conditioning through SQA.
python run_training.py \
  --actor.pair-embedding-head alphastar-buchberger-pair-transformer \
  --actor.state-embedding-head deep-set \
  --actor.pair-combiner sqa
```

Tracking is disabled by default, CUDA is used when available, and optional
counterfactual/gradient/ranking diagnostics are disabled. Greedy evaluation runs
on the training environment by default; enable NoPQ evaluation explicitly with
`--training.hparam-eval-on-nopq-env True`. These are starting defaults, not a claim
that one configuration performs best for every ideal distribution.

## Observations and Diagnostic Info

Both environments return observations containing `P`, `G`, `P_mask`, `G_mask`,
and `state_features`. All pair heads encode endpoint features from `P`, which
retains both endpoints even when the basis context in `G` is truncated by the priority queue.

The priority-queue environment returns `P_idx_raw`, `P_sugar`, and
`P_normal_rank` in the `info` dictionary for policy-versus-heuristic ranking
diagnostics. `P_idx_raw` contains the original basis indices of each visible
pair. These arrays describe the returned observation and are independent
snapshots. Training stores them separately on the CPU only when
`training.ranking_diagnostic_frequency` is positive.

Training environments reset in the same step when an episode ends. The main
observation and info describe the new episode; the completed episode's data
are available under `info["final_obs"]` and `info["final_info"]`.

## Common Training Runs

Train SPEM with a Deep Set state head on a 3-20-10-weighted binomial distribution for 1,000,000 timesteps:

```bash
python run_training.py \
  --actor.state-embedding-head deep-set \
  --env.ideal-dist 3-20-10-weighted \
  --training.total-timesteps 1000000 \
  --training.num-envs 12 \
  --training.track False \
  --training.console-verbosity normal
```

Train the default SPEM actor without either state head:

```bash
python run_training.py \
  --actor.pair-embedding-head symmetric-per-entity-mlp \
  --env.ideal-dist 3-20-10-weighted \
  --training.total-timesteps 1000000 \
  --training.num-envs 12 \
  --training.track False
```

Enable both the Deep Set and MLP state-feature heads:

```bash
python run_training.py \
  --actor.state-embedding-head deep-set \
  --actor.state-features-embedding-head mlp \
  --env.ideal-dist 3-20-10-weighted \
  --training.total-timesteps 1000000 \
  --training.num-envs 12 \
  --training.track False
```

Train on a geometric distribution:

```bash
python run_training.py \
  --actor.state-embedding-head deep-set \
  --env.ideal-dist 'geom-(3-5)' \
  --training.total-timesteps 1000000 \
  --training.num-envs 12
```

Train on a benchmark family:

```bash
python run_training.py \
  --actor.state-embedding-head deep-set \
  --env.ideal-dist 'cyclic-(3-6)' \
  --training.total-timesteps 1000000 \
  --training.num-envs 12
```

Train on a mixed benchmark family:

```bash
python run_training.py \
  --actor.state-embedding-head deep-set \
  --env.ideal-dist '(cyclic+eco)-(3-6)' \
  --training.total-timesteps 1000000 \
  --training.num-envs 12
```

Change the visible pair/basis budget and queue strategy:

```bash
python run_training.py \
  --actor.state-embedding-head deep-set \
  --env.ideal-dist 3-20-10-weighted \
  --env.N 150 \
  --env.priority-queue-strategy sugar \
  --env.k 3
```

Write outputs to a dedicated experiment directory:

```bash
python run_training.py \
  --actor.state-embedding-head deep-set \
  --env.ideal-dist 3-20-10-weighted \
  --training.output-directory runs/public_release/spem_deep_set_weighted
```

## Local Tracking

When `--training.track False`, the trainer still records local artifacts under:

```text
runs/<run_name>/
```

or under:

```text
<output_directory>/runs/<run_name>/
```

The main local files are:

- `scalars.jsonl`
- `scalars.csv`
- `system_metrics.jsonl`
- `text/hyperparameters.md`

`quiet` mode is useful for unattended runs. `normal` prints one concise summary per PPO update. `verbose` additionally prints the resolved config, model structure, and detailed greedy-eval progress.

## Weights & Biases Tracking

Enable W&B tracking by setting `--training.track True`:

```bash
python run_training.py \
  --actor.state-embedding-head deep-set \
  --env.ideal-dist 3-20-10-weighted \
  --training.total-timesteps 1000000 \
  --training.num-envs 12 \
  --training.track True
```

The training script also supports W&B-specific settings. For example:

```bash
python run_training.py \
  --env.gym_id=BuchbergerExponents-v1 \
  --env.ideal_dist=3-20-10-weighted \
  --training.total_timesteps=1000000 \
  --training.num_envs=12 \
  --training.track=True \
  --training.wandb_project=deepgroebner-public \
  --training.wandb_entity=YOUR_WANDB_ENTITY \
  --training.wandb_group=spem-deep-set-release \
  --training.output_directory=wandb_runs \
  --actor.state_embedding_head=deep-set \
  --actor.state_features_embedding_head=None \
  --actor.pair_embedding_head=symmetric-per-entity-mlp \
  --actor.pair_combiner=film
```

Use `python run_training.py --help` to inspect options and their defaults.

## Training on a Fixed Ideal

Supply a fixed ideal using explicit basis data:

```bash
python run_training.py \
  --env.gym_id=BuchbergerExponents-v1 \
  --env.coeff_ring=QQ \
  --env.symbols=x,y \
  --env.basis_elements='x^2-y,y^2-1' \
  --training.total_timesteps=250000 \
  --training.num_envs=1 \
  --training.track=False \
  --actor.state_embedding_head=deep-set \
  --actor.state_features_embedding_head=None \
  --actor.pair_embedding_head=symmetric-per-entity-mlp \
  --actor.pair_combiner=film
```

In that mode the environment no longer samples from `ideal_dist`; it reuses the basis you supplied.

## Training from Python

Use `train` to run the same trainer directly, without launching a subprocess:

```python
from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.ppo_training import train

cfg = Config()
cfg.training.output_directory = "runs/my_experiment"
result = train(cfg)

print(result.summary["global_step"])
print(result.checkpoint_paths)
print(result.run_directory)
```

Each call resolves an independent copy of `cfg`. Environments and log files close
when training finishes or raises an exception.

For command-line training, use `python run_training.py`. It parses the same
configuration and calls `train` directly in the current process.
