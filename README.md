# deepgroebner-pytorch-cocoa

This repository is a public release of code I've worked on through my PhD combining reinforcement learning with a CoCoALib-backed Buchberger's algorithm environment in C++. The purpose of publishing this seperate repo is so that I can maintain my messy research code in its original repo whilst providing this repo with all of the useful functionality still intact for others to use.

## Installation

Installation instructions are in INSTALL.md

After installation, run these scripts from the repository root using the same
Python environment:

```bash
python run_training.py --help
python generate_corpus.py --help
python evaluate_model.py --help
```

See [INSTALL.md](INSTALL.md) for installation checks using pytest.

## Tutorials

- [Training Runs](docs/tutorials/training-runs.md)
- [Eval Corpora And Greedy Evaluation](docs/tutorials/eval-corpora-and-greedy-eval.md)
- [Hyperparameter Sweeps](docs/tutorials/hyperparameter-sweeps.md)

Training, corpus generation, and checkpoint evaluation can also be called from
notebooks or other Python projects using the installed package. See the corpus
and evaluation tutorial for the `generate_corpus(CorpusConfig(...))` and
`evaluate(EvalCorpusConfig(...))` APIs.

Geometric ideal distributions use the bundled Laman graph tables for 3 to 10
vertices. These files are installed with the package, so geometric environments
can be created from any working directory.

## Tracking with W&B

If you want to be able to use your Weights & Biases account to track training runs, make sure to install the optional tracking dependency and log in to Weights and Biases.

## Local Tracking Without W&B

When `--training.track False` is used, runs still produce local artifacts under `runs/<run_name>/`:

- `scalars.jsonl`: append-only scalar metrics in JSONL format
- `scalars.csv`: the same scalar metrics in simple CSV form
- `system_metrics.jsonl`: periodic CPU/GPU/process memory and rollout-size snapshots
- `text/hyperparameters.md`: resolved run settings

The training script also exposes a simple terminal verbosity control:

```bash
python run_training.py --training.console-verbosity quiet
python run_training.py --training.console-verbosity normal
python run_training.py --training.console-verbosity verbose
```

`quiet` minimizes terminal output, `normal` prints update summaries, and `verbose` adds config/model information plus greedy-eval progress details.

## Configure Training

The default actor uses SPEM with no state or state-feature head. Select heads
independently, then override any setting from `config_dataclasses.py`:

```bash
python run_training.py \
  --actor.pair-embedding-head symmetric-per-entity-mlp \
  --actor.state-embedding-head deep-set \
  --actor.state-features-embedding-head mlp \
  --actor.pair-combiner film \
  --training.learning-rate 0.0001
```

Omitted options use the shared dataclass defaults. Use `--help` for the complete
list of options and defaults. The critic is configured separately with
`--critic.*`. See the [training tutorial](docs/tutorials/training-runs.md) for
examples.

## Policy architectures included in this library

When creating the policy, there must be a pair embedding head. The pair embedding head can be one of three choices:
- `per-entity-mlp`: A multilayer perceptron neural network that is applied independently to each pair in the pair set. This network is included for comparison purposes. In all tests I've performed, the symmetric per entity mlp is a straightforward upgrade.
- `symmetric-per-entity-mlp`: A multilayer perceptron neural network that is applied independently to each pair in the pair set. This network respects the symmetry of the pairs ((i,j) will give the same output as (j,i)). Each polynomial in the pair is passed seperately to the same MLP and then the embeddings are combined using symmmetric operations.
- `alphastar-buchberger-pair-transformer`: A shared MLP encodes both endpoint vectors from `P`. Symmetric combinations of these embeddings, plus optional encoded LCM/GCD features, form pair tokens. These pass through residual blocks and masked self-attention without positional encodings. Every valid pair participates, including pairs with endpoints outside the visible basis budget.
Then, a state embedding head can be optionally included based on the deep-set network architecture. By default, the state embedding head will be disabled. To add a state embedding head to the model, pass `--actor.state_embedding_head deep-set`. If this state embedding head is included then there is a choice to be made about how to combine the information from the state embedding head with the pair embeddings. By default the `film` option is used. The output from the state embedding head will be a single vector and the output of the pair embedding head is a matrix equal to the size of the priority queue x the pair embedding dimension :
- `film`: Based on the FiLM - Feature Wise Linear Modulation method of conditioning in neural networks, this method uses the state embedding to create a scale and shift vector which is then applied to each pair embedding.
- `sqa`: This method of conditioning the pair embeddings on the state embedding works by using scaled dot product attention. The state embedding is the single query input and the pairs are the keys. The attention weights calculated between the single query and the keys directly become the probabilities of selecting each of the pairs in the policy.
- `concat`: This straightforwardly combines the pair embeddings with the state embedding by concatening the state embedding to the end of each of the pair embeddings and then passing the concatenated vector through an MLP.
A state feature embedding head can be optionally included and is a simple MLP. By defualt it is disabled, to add a state feature embedding head to the model, pass `--actor.state_features_embedding_head mlp`. If this state feature head is included without a state embedding head then the pair embeddings will be combined with the state feature embedding using one of the methods given above for the state embedding. If both a state embedding head and a state feature embedding head are provided then the state embedding and state feature embedding will be combined using either the `film` or `concat` method. The `sqa` method is not used as their is only one state feature embedding and one state embedding. This combined state and state feature embedding is then used to condition the pair embeddings using one of the above methods.
