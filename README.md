# deepgroebner-pytorch-cocoa

This repository is the standalone public release of `deepgroebner-pytorch` with a CoCoALib-backed runtime and a simplified training interface.

## License and Attribution

This project should be released as a derivative work of [`dylanpeifer/deepgroebner`](https://github.com/dylanpeifer/deepgroebner), which GitHub identifies as licensed under `GPL-3.0`.

For the public release, keep all three of these in the repository root:

- [LICENSE](LICENSE)
- [NOTICE.md](NOTICE.md)
- this README

The short version is:

- credit the upstream DeepGroebner project and authors
- state clearly that this repository is a modified derivative
- distribute the project under GPL-3.0 as well
- keep the license text with the source release

The goal is to preserve the core workflow:

- sample Buchberger environments
- train a PPO policy
- save actor checkpoints

while removing or hiding most of the experimental surface area:

- large architecture search spaces
- sweep-specific arguments
- imitation-learning and MCTS-based evaluation features
- diagnostic toggles
- HPC submission scripts

## What is included

- a single `deepgroebner_pytorch_cocoa` package containing both the public CLI layer and the supported runtime code
- a simplified top-level trainer entrypoint in `ppo_masked.py`
- packaging metadata for this standalone repository

## Current status

The repository now contains its own trainer and runtime package layout. The main remaining packaging limitation is the native environment build: the wrapped CoCoALib extension depends on a separately installed CoCoALib, so installation is not yet as frictionless as a pure-Python package on a fresh machine. This repo does support a documented system-install flow for CoCoALib, and the GitHub Actions workflow verifies that install path on Linux and macOS.

## Suggested supported presets

- `symmetric-per-entity-mlp`: symmetric per-entity MLP pair encoder with pair-only scoring
- `spem-deep-set`: symmetric per-entity MLP pair encoder + deep-set state encoder + FiLM pair combiner
- `spem-deep-set-state-features`: symmetric per-entity MLP pair encoder + deep-set state encoder + MLP state-feature encoder + FiLM combiners

These names are intentionally short. The internal mapping is documented in `src/deepgroebner_pytorch_cocoa/presets.py`.

## Example

```bash
pip install -e .

deepgroebner-pytorch-cocoa-train \
  --architecture spem-deep-set \
  --ideal-distribution 3-20-10-weighted \
  --total-timesteps 1000000 \
  --num-envs 12 \
  --track False
```

Use `--dry-run True` to print the exact `ppo_masked.py` command without starting training.

## Tutorials

- [Training Runs](docs/tutorials/training-runs.md)
- [Eval Corpora And Greedy Evaluation](docs/tutorials/eval-corpora-and-greedy-eval.md)

## Local Tracking Without W&B

When `--track False` is used, runs still produce local artifacts under `runs/<run_name>/`:

- `scalars.jsonl`: append-only scalar metrics in JSONL format
- `scalars.csv`: the same scalar metrics in simple CSV form
- `system_metrics.jsonl`: periodic CPU/GPU/process memory and rollout-size snapshots
- `text/hyperparameters.md`: resolved run settings

The public wrapper also exposes a simple terminal verbosity control:

```bash
deepgroebner-pytorch-cocoa-train --verbosity quiet
deepgroebner-pytorch-cocoa-train --verbosity normal
deepgroebner-pytorch-cocoa-train --verbosity verbose
```

`quiet` minimizes terminal output, `normal` prints update summaries, and `verbose` adds config/model dumps plus greedy-eval progress details.

## Installation note

The major remaining install constraint is the native CoCoALib/Cython environment layer used by the Buchberger environments.

This repository assumes option 3 for native dependencies:

- CoCoALib is installed separately on the system
- this package links against that system installation
- installation instructions live in [INSTALL.md](INSTALL.md)

For a genuinely easy public install, the next repo step should be one of:

1. publish wheels for the supported platforms
2. provide a Docker image
3. move the native build into a more automated CI-produced wheel pipeline

The public wrapper already removes one avoidable pain point: `wandb` is not required when tracking is disabled.

## Public Release Checklist

Before making this repository public, make sure the final root contains:

1. `LICENSE` with the full GPL-3.0 text.
2. `NOTICE.md` with upstream attribution and derivative-work language.
3. A README section explaining that this project is derived from `dylanpeifer/deepgroebner`.
4. Prominent notes for major modifications when copying or heavily adapting upstream files.
