# Eval Corpora And Greedy Evaluation

This tutorial explains how fixed greedy-eval corpora work in `deepgroebner-pytorch-cocoa`, how to generate them for different ideal distributions, and how to run training-time evaluations against them.

The public repo now supports both:

- generating fixed eval corpora
- evaluating saved actor checkpoints against those corpora

## What A Greedy-Eval Corpus Is

A greedy-eval corpus is a fixed list of ideals stored as JSONL. The trainer uses it so evaluation runs are repeatable across checkpoints and comparable across experiments.

Each row records:

- the episode seed
- the source distribution
- the coefficient ring
- the symbol list
- the fixed basis elements for that example

The trainer caches corpora under:

```text
eval_corpora/greedy_eval/
```

When a run asks for the same distribution, base seed, and number of examples again, the cached corpus is reused automatically.

## Generate A Corpus

The public helper command is:

```bash
deepgroebner-pytorch-cocoa-make-corpus
```

If your shell does not expose that script on `PATH`, the equivalent fallback is:

```bash
python -m deepgroebner_pytorch_cocoa.corpus_cli
```

The most important options are:

- `--ideal-distribution`: the distribution to sample from
- `--examples`: number of fixed ideals to include
- `--seed`: base seed for corpus generation
- `--priority-queue-strategy`: used when sampling from the environment
- `--observation-budget`: observation budget `N`
- `--top-monomials`: number of leading monomials per polynomial
- `--output-path`: optional custom JSONL path
- `--overwrite`: whether an existing custom output file may be replaced

Generate a cached corpus for the default weighted distribution:

```bash
deepgroebner-pytorch-cocoa-make-corpus \
  --ideal-distribution 3-20-10-weighted \
  --examples 100 \
  --seed 10000
```

Generate a cached corpus for a geometric distribution:

```bash
deepgroebner-pytorch-cocoa-make-corpus \
  --ideal-distribution 'geom-(3-5)' \
  --examples 50 \
  --seed 10000
```

Generate a cached corpus for a benchmark family:

```bash
deepgroebner-pytorch-cocoa-make-corpus \
  --ideal-distribution 'cyclic-(3-6)' \
  --examples 50 \
  --seed 10000
```

Generate a cached corpus for a mixed benchmark family:

```bash
deepgroebner-pytorch-cocoa-make-corpus \
  --ideal-distribution '(cyclic+eco)-(3-6)' \
  --examples 50 \
  --seed 10000
```

Write the corpus to a specific file instead of the default cache location:

```bash
deepgroebner-pytorch-cocoa-make-corpus \
  --ideal-distribution 'geom-(3-5)' \
  --examples 30 \
  --seed 10000 \
  --output-path corpora/geom_3_5_seed10000.jsonl
```

## Enable Greedy Evaluation During Training

The supported public evaluation workflow today is training-time greedy evaluation. The trainer either loads a matching cached corpus or generates one automatically before the first evaluation round.

In the examples below, `spem-deep-set` is used as the default public training preset. You can swap in `symmetric-per-entity-mlp` for the lean pair-only setup or `spem-deep-set-state-features` for the fullest curated state-conditioned setup.

### Train-Environment Evaluation

Enable fixed-corpus evaluation on the same environment used for training:

```bash
deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set \
  --ideal-distribution 'cyclic-(3-6)' \
  --total-timesteps 1000000 \
  --greedy-eval-interval-steps 100000 \
  --greedy-eval-examples 50 \
  --track False \
  --verbosity normal
```

That will periodically report metrics such as:

- mean return
- median return
- completion rate
- number of truncated episodes
- elapsed eval time

and it will write the same values into the local scalar logs.

### Reuse A Prebuilt Corpus

If you generated the corpus ahead of time with the same distribution, seed, and example count, the training run will reuse it automatically:

```bash
deepgroebner-pytorch-cocoa-make-corpus \
  --ideal-distribution 'cyclic-(3-6)' \
  --examples 50 \
  --seed 10000

deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set \
  --ideal-distribution 'cyclic-(3-6)' \
  --greedy-eval-interval-steps 100000 \
  --greedy-eval-examples 50
```

### Evaluate On The NoPQ Environment Too

The small public wrapper currently keeps evaluation simple and only enables train-environment greedy eval by default. To evaluate against the NoPQ environment too, call the trainer directly:

```bash
python -m deepgroebner_pytorch_cocoa.ppo_masked \
  --env.gym_id=BuchbergerExponents-v1 \
  --env.ideal_dist='cyclic-(3-6)' \
  --training.total_timesteps=1000000 \
  --training.num_envs=12 \
  --training.hparam_eval_interval_steps=100000 \
  --training.hparam_eval_num_examples=50 \
  --training.hparam_eval_seed=10000 \
  --training.hparam_eval_on_train_env=True \
  --training.hparam_eval_on_nopq_env=True \
  --training.hparam_eval_nopq_gym_id=BuchbergerExponentsNoPQ-v1 \
  --training.track=False \
  --actor.state_embedding_head=deep-set \
  --actor.state_features_embedding_head=None \
  --actor.pair_embedding_head=symmetric-per-entity-mlp \
  --actor.pair_combiner=film
```

In that mode the trainer computes:

- train-environment greedy eval metrics
- NoPQ greedy eval metrics
- comparison metrics such as NoPQ-minus-train mean return

## Tracking Evaluation Results

With `--track False`, greedy-eval metrics are written locally into:

- `scalars.jsonl`
- `scalars.csv`

When `--track True` and `wandb` is installed, the same metrics are mirrored into Weights & Biases.

Example:

```bash
deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set \
  --ideal-distribution 'geom-(3-5)' \
  --greedy-eval-interval-steps 100000 \
  --greedy-eval-examples 50 \
  --track True
```

If you want explicit W&B project or group settings, use the trainer directly:

```bash
python -m deepgroebner_pytorch_cocoa.ppo_masked \
  --env.gym_id=BuchbergerExponents-v1 \
  --env.ideal_dist='geom-(3-5)' \
  --training.total_timesteps=1000000 \
  --training.num_envs=12 \
  --training.track=True \
  --training.wandb_project=deepgroebner-public \
  --training.wandb_group=geom_eval \
  --training.hparam_eval_interval_steps=100000 \
  --training.hparam_eval_num_examples=50 \
  --actor.state_embedding_head=deep-set \
  --actor.state_features_embedding_head=None \
  --actor.pair_embedding_head=symmetric-per-entity-mlp \
  --actor.pair_combiner=film
```

## Evaluate A Saved Checkpoint Against A Corpus

The standalone command for offline corpus evaluation is:

```bash
deepgroebner-pytorch-cocoa-eval-corpus
```

The most important options are:

- `--corpus`: path to the JSONL corpus
- `--model-name`: full checkpoint path or a substring to resolve inside `--models-dir`
- `--checkpoint-type`: `auto`, `best_greedy_eval`, or `final`
- `--gym-id`: evaluation environment, usually `BuchbergerExponentsNoPQ-v1` or `BuchbergerExponents-v1`
- `--config-path`: optional explicit config snapshot path; if omitted, the CLI first looks for the saved `__config.json` next to the checkpoint
- `--output-dir` or `--output-root`: where evaluation artifacts should be written

Evaluate a saved checkpoint on a fixed corpus in the NoPQ environment:

```bash
deepgroebner-pytorch-cocoa-eval-corpus \
  --corpus eval_corpora/greedy_eval/BuchbergerExponents-v1__cyclic-(3-6)__seed10000__n50.jsonl \
  --model-name models/actors/my_run.pt \
  --gym-id BuchbergerExponentsNoPQ-v1
```

Evaluate the best greedy-eval checkpoint resolved by partial name:

```bash
deepgroebner-pytorch-cocoa-eval-corpus \
  --corpus eval_corpora/greedy_eval/BuchbergerExponents-v1__cyclic-(3-6)__seed10000__n50.jsonl \
  --model-name j77pf3uu \
  --models-dir models/actors \
  --checkpoint-type best_greedy_eval \
  --gym-id BuchbergerExponents-v1
```

The command writes:

- `episode_results.csv`
- `episode_results.jsonl`
- `summary.json`
- `metadata.json`

## Current Scope

The public repo now exposes:

- corpus generation
- training-time greedy evaluation
- standalone checkpoint evaluation against a saved corpus
