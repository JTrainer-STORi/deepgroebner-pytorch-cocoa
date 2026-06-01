# Training Runs

This tutorial shows how to launch training runs for a chosen ideal distribution using the public CLI.

## Before You Start

Install the package:

```bash
pip install -e .
```

If you want Weights & Biases integration too, install the optional tracking dependency and log in:

```bash
pip install -e ".[tracking]"
wandb login
```

## The Main Public Command

The public training entrypoint is:

```bash
deepgroebner-pytorch-cocoa-train
```

The most important options are:

- `--architecture`: curated network preset. Supported values are `symmetric-per-entity-mlp`, `spem-deep-set`, and `spem-deep-set-state-features`.
- `--ideal-distribution`: the ideal distribution to train on.
- `--total-timesteps`: total environment steps.
- `--num-envs`: number of parallel environments.
- `--rollout-steps`: rollout length per PPO update.
- `--priority-queue-strategy`: visible S-pair ordering strategy, such as `degree`, `normal`, `sugar`, or `first`.
- `--observation-budget`: observation budget `N` for visible pairs and basis elements.
- `--top-monomials`: number of leading monomials shown per polynomial.
- `--greedy-eval-interval-steps`: how often fixed-corpus greedy evaluation runs during training. Set `0` or a negative value to disable it. If greedy-eval is causing hangups early in training, it is best to either increase this so that the policy has more time to improve or disable it altogether and perform eval on the final model.
- `--greedy-eval-examples`: how many fixed eval examples to use when greedy evaluation is enabled.
- `--track`: set to `True` to enable Weights & Biases.
- `--verbosity`: terminal output level. Use `quiet`, `normal`, or `verbose`.
- `--output-directory`: root directory for local logs and saved models.

## Choosing An Architecture Preset

Use `symmetric-per-entity-mlp` when you want the smallest supported policy setup:

- symmetric per-entity MLP pair encoder
- pair-only scoring
- no deep-set state encoder
- no separate state-feature encoder

Use `spem-deep-set` when you want the default public setup:

- symmetric per-entity MLP pair encoder
- deep-set state encoder
- FiLM pair combiner

Use `spem-deep-set-state-features` when you want the richest curated setup:

- symmetric per-entity MLP pair encoder
- deep-set state encoder
- MLP state-feature encoder
- FiLM state and pair combiners

To inspect the exact low-level trainer command before launching anything:

```bash
deepgroebner-pytorch-cocoa-train --dry-run True
```

## Common Training Runs

Train the default public `spem-deep-set` preset on the default weighted distribution:

```bash
deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set \
  --ideal-distribution 3-20-10-weighted \
  --total-timesteps 1000000 \
  --num-envs 12 \
  --track False \
  --verbosity normal
```

Train the lean pair-only preset instead:

```bash
deepgroebner-pytorch-cocoa-train \
  --architecture symmetric-per-entity-mlp \
  --ideal-distribution 3-20-10-weighted \
  --total-timesteps 1000000 \
  --num-envs 12 \
  --track False
```

Train the fullest curated preset with both deep-set and state-feature encoders:

```bash
deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set-state-features \
  --ideal-distribution 3-20-10-weighted \
  --total-timesteps 1000000 \
  --num-envs 12 \
  --track False
```

Train on a geometric distribution:

```bash
deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set \
  --ideal-distribution 'geom-(3-5)' \
  --total-timesteps 1000000 \
  --num-envs 12
```

Train on a benchmark family:

```bash
deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set \
  --ideal-distribution 'cyclic-(3-6)' \
  --total-timesteps 1000000 \
  --num-envs 12
```

Train on a mixed benchmark family:

```bash
deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set \
  --ideal-distribution '(cyclic+eco)-(3-6)' \
  --total-timesteps 1000000 \
  --num-envs 12
```

Change the visible pair/basis budget and queue strategy:

```bash
deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set \
  --ideal-distribution 3-20-10-weighted \
  --observation-budget 150 \
  --priority-queue-strategy sugar \
  --top-monomials 3
```

Write outputs to a dedicated experiment directory:

```bash
deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set \
  --ideal-distribution 3-20-10-weighted \
  --output-directory runs/public_release/spem_deep_set_weighted
```

## Local Tracking

When `--track False`, the trainer still records local artifacts under:

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

Enable W&B tracking by setting `--track True`:

```bash
deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set \
  --ideal-distribution 3-20-10-weighted \
  --total-timesteps 1000000 \
  --num-envs 12 \
  --track True
```

The underlying trainer also supports W&B-specific settings. For example:

```bash
python -m deepgroebner_pytorch_cocoa.ppo_masked \
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

Use the public wrapper when you want the smaller supported interface. Use `python -m deepgroebner_pytorch_cocoa.ppo_masked` directly when you need advanced trainer-only flags such as W&B grouping or richer evaluation controls.

## Training on a Fixed Ideal

The small public wrapper does not expose fixed-ideal training directly. For that workflow, call the trainer with explicit basis data:

```bash
python -m deepgroebner_pytorch_cocoa.ppo_masked \
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
