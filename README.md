# deepgroebner-pytorch-cocoa

This repository is the standalone public release of `deepgroebner-pytorch` with a CoCoALib-backed runtime and a simplified training interface.

## Installation

Installation instructions are in INSTALL.md

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
