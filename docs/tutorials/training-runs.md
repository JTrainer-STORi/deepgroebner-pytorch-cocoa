# Training Runs

This tutorial shows how to launch training runs for a chosen ideal distribution from the command line.

For information about specifying input ideals for training, see
[Input Ideals](input-ideals.md).

## Choose Heads, Then Override Settings

After installing the package, run commands from the repository root using the Python environment where you installed it. The training script uses the configuration dataclasses directly:

```bash
python run_training.py
```

Policy architectures included in this library

When creating the policy, there must be a pair embedding head. The pair embedding head can be one of three choices:
- `per-entity-mlp`: A multilayer perceptron neural network that is applied independently to each pair in the pair set. This network is included for comparison purposes. In all tests I've performed, the symmetric per entity mlp is a straightforward upgrade.
- `symmetric-per-entity-mlp`: A multilayer perceptron neural network that is applied independently to each pair in the pair set. This network respects the symmetry of the pairs ((i,j) will give the same output as (j,i)). Each polynomial in the pair is passed seperately to the same MLP and then the embeddings are combined using symmmetric operations.
- `alphastar-buchberger-pair-transformer`: A shared MLP encodes both endpoint vectors from `P`. Symmetric combinations of these embeddings, plus optional encoded LCM/GCD features, form pair tokens. These pass through residual blocks and masked self-attention without positional encodings. Every valid pair participates, including pairs with endpoints outside the visible basis budget.

Then, a state embedding head can be optionally included based on the deep-set network architecture. By default, the state embedding head will be disabled. To add a state embedding head to the model, pass `--actor.state_embedding_head deep-set`. If this state embedding head is included then there is a choice to be made about how to combine the information from the state embedding head with the pair embeddings. By default the `film` option is used. The output from the state embedding head will be a single vector and the output of the pair embedding head is a matrix equal to the size of the priority queue x the pair embedding dimension :
- `film`: Based on the FiLM - Feature Wise Linear Modulation method of conditioning in neural networks, this method uses the state embedding to create a scale and shift vector which is then applied to each pair embedding.
- `sqa`: This method of conditioning the pair embeddings on the state embedding works by using scaled dot product attention. The state embedding is the single query input and the pairs are the keys. The attention weights calculated between the single query and the keys directly become the probabilities of selecting each of the pairs in the policy.
- `concat`: This straightforwardly combines the pair embeddings with the state embedding by concatening the state embedding to the end of each of the pair embeddings and then passing the concatenated vector through an MLP.

A state feature embedding head can be optionally included and is a simple MLP. By defualt it is disabled, to add a state feature embedding head to the model, pass `--actor.state_features_embedding_head mlp`. If this state feature head is included without a state embedding head then the pair embeddings will be combined with the state feature embedding using one of the methods given above for the state embedding. If both a state embedding head and a state feature embedding head are provided then the state embedding and state feature embedding will be combined using either the `film` or `concat` method. The `sqa` method is not used as their is only one state feature embedding and one state embedding. This combined state and state feature embedding is then used to condition the pair embeddings using one of the above methods.

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
counterfactual/attention/gradient/ranking diagnostics are disabled.

## Greedy Evaluation during Training

Greedy evaluation periodically chooses the policy's highest-probability action
on a fixed evaluation corpus. When enabled, the trainer prepares this corpus
automatically from the training inputs. See [Input Ideals](input-ideals.md#how-input-choices-affect-greedy-evaluation)
for how generated distributions, fixed ideals, and JSONL training inputs are handled.

Evaluation uses the training environment by default; enable NoPQ evaluation with
`--training.greedy-eval-on-nopq-env True`.

Use `--training.greedy-eval-first-step` to choose when evaluation starts and
`--training.greedy-eval-interval-steps` for subsequent rounds. A first step of
`0` evaluates before training; omitting it uses the interval for the first round.
Set `--training.greedy-eval-interval-steps` to a nonpositive value to disable
greedy evaluation during training.

Use `--training.greedy-eval-num-examples` to control the automatic corpus size
and `--training.greedy-eval-seed` to control corpus selection. To use an existing
JSONL corpus instead, set `--training.greedy-eval-corpus path/to/corpus.jsonl`;
every row is evaluated in file order. The example count must remain positive to
keep evaluation enabled, but it does not limit an explicit corpus.

Reserve a separate **test corpus** for comparing saved policies after training,
using a different generation seed and a path outside the training evaluation
cache. See [Evaluation and Test Corpora](eval-corpora-and-greedy-eval.md).

Use `--training.greedy-eval-max-episode-seconds` to limit the time spent on each
ideal during greedy evaluation.

## Observations and Diagnostic Info

Both environments return observations containing `P`, `G`, `P_mask`, `G_mask`,
and `state_features`. All pair heads encode endpoint features from `P`, which
retains both endpoints even when the basis context in `G` is truncated by the priority queue.

### Heuristic Ranking Diagnostics

The priority-queue environment returns `P_idx_raw`, `P_sugar`, and
`P_normal_rank` in the `info` dictionary for policy-versus-heuristic ranking
diagnostics. `P_idx_raw` contains the original basis indices of each visible
pair. These arrays describe the returned observation and are independent
snapshots which are only used when
`--training.ranking_diagnostic_frequency` is positive.
 This diagnostic perdiodically (every `--training.ranking_diagnostic_frequency` policy updates) compares the choice of S-pair under the policy with what would have been selected by each of the classic heuristics.

### Counterfactual Diagnostics
 Set `--training.counterfactial_diagnostic_frequency` to `n> 0` to run the counterfactual diagnostic every `n` policy updates. If using a state feature embedding head and/or state embedding head, this diagnostic looks at how the policy changes when the inputs to these heads are replaced by a zeroed input or another random input from the training batch.

### Gradient Sensitivity Diagnostics

Set `--training.gradient_sensitivity_diagnostic_frequency` to `n > 0`
to run a diagnostic that calculates the sensitivity of the policy gradient to the different input heads.

### Attention Diagnostics

When using the `alphastar-buchberger-pair-transformer` pair embedding head, set `--training.attention_diagnostic_frequency` to `n>0` to run a diagnostic that will look at the attention weight entropy within attention layers and the agreement between attention heads.

## Priority Queue

The priority queue mechanism contrils the size of observations during training which prevents runaway memory usage for training runs. Both the pair set observation and the basis set observation are subject to the priority queue. Use the config option `--env.gym-id="BuchbergerExponentsNoPQ-v1"`to disable the priority queue when training. With the warning that this will make OOM errors much more likely.

The size of the priority queue is controlled using `--env.N` and is `100` by default. This will apply to both the pair set and basis set observations. To set the size of the priority queues seperately, use `--env.N_P` and `--env.N_G`.

Use `--env.priority_queue_strategy`to decide which `--env.N`pairs will be put in the observation if the number of pairs exceeds `--env.N`. Basis elements are always prioritized based on which basis elements are currently present in the pair set observation starting with the highest priority pairs in the pair set observation. The order of the basis set is used after all basis set elements present in pairs are included. The options `["first" "degree", "normal", "sugar"]` 
rank the pairs in the pair set according to the corresponding heuristic and the top N become the observation. `"random"` samples `--env.N`pairs randomly without replacement.

## Critic Options

There are three options for the critic during training.

- `"heuristic-only"`- With this setting, the value of states is determined by running Buchberger's algorithm using the basis and pair set in the state and performing S-pair selection using the `--critic.value-function-heuristic`heuristic.
- `"learned-only"`- The value of states is given by a value neural network which is trained alongside the policy. This network uses a `deep-set`architecture and takes the basis set observation as input.
- `"heuristic-plus-learned-residual"`- This critic strategy used the heuristic to estimate the value but applies a residual correction to the heuristic which is given by a network with the same architecture as the one used with the learned-only critic.

## Grouped Outputs

Set `training.group` to collect related runs beneath one directory:

```bash
python run_training.py \
  --env.ideal-dist 3-20-10-weighted \
  --training.output-directory experiments \
  --training.group sweep-001
```

The resulting layout is:

```text
experiments/sweep-001/
  models/actors/       # Final and best actor checkpoints and config snapshots
  models/critics/      # Critic checkpoints and config snapshots, when enabled
  runs/<run_name>/     # Scalar logs, diagnostics, and system metrics
  wandb/              # W&B run files, when tracking is enabled
```

Grouping applies to local outputs even with `training.track=False`. Without
`output_directory`, the group directory is created in the current working
directory. Without a group, `models/`, `runs/`, and `wandb/` sit directly under
the output directory, which also defaults to the current working directory.

`hparam_metrics_out` remains an exact file path; set it to, for example,
`experiments/sweep-001/trial-001-summary.json` to place that summary in the group.
The shared evaluation corpus cache remains under `eval_corpora/greedy_eval/`
in the current working directory so different runs can reuse it.

Model weights are saved in the models directory shown above. Policy weights are saved by default. Set `--training.save-critic True` to save critic weights. The model weights are saved as `.pt` files and are used along with the model config saved in a `.jsonl` model to load models for evaluation. If greedy evaluation was enabled during training then the best performing model during greedy evaluation wll be saved in additon to the final model.

## Local Tracking

When `--training.track False`, the trainer still records local artifacts. With
neither an output directory nor a group specified, these are under:

```text
runs/<run_name>/
```

With an output directory and no group, these are under:

```text
<output_directory>/runs/<run_name>/
```

With a group, these are under `<output_directory>/<group>/runs/<run_name>/`, as
shown above.

The main local files are:

- `scalars.jsonl`
- `scalars.csv`
- `system_metrics.jsonl`
- `text/hyperparameters.md`

`quiet` mode is useful for unattended runs. `normal` prints one concise summary per PPO update. `verbose` additionally prints the resolved config, model structure, and detailed greedy-eval progress.

## Weights & Biases Tracking

Enable W&B tracking by setting `--training.track True`:



### Metric Sections

Trainer scalar names use lowercase section prefixes with one slash. The same
names appear in W&B, `scalars.jsonl`, and `scalars.csv`:

| Section | Contents and examples |
| --- | --- |
| `training_rollouts/` | Sampled training episode returns, lengths, and completion counts, plus rollout state counts, basis sizes, and steps since a basis update. |
| `evaluation_greedy/` | Fixed-corpus greedy evaluation, with `train_env_*` and `nopq_*` results, completion/truncation counts, and `selection_metric`. |
| `policy_function/` | Policy `loss`, entropy, KL estimates, `clip_fraction`, `grad_norm_before_clip`, optional `state_features_running_*` normalization statistics, and `diagnostic_*` ranking, counterfactual, gradient, and attention metrics. |
| `value_function/` | Value `loss`, `weighted_loss`, `explained_variance`, clipping, gradient norms, advantage mean/std, and optional `running_target_*` normalization statistics. |
| `algorithm_general/` | `policy_updates`, optional reward normalization statistics, and `learning_rate` when learning-rate annealing is enabled. |
| `priority_queue/` | Pending-pair counts, processed-pair counts, pair budgets, and overflow/binding statistics such as `rollout_pair_count_mean` and `rollout_pq_binding_step_fraction`. |
| `performance/` | Training `steps_per_second` and per-round `greedy_train_env_seconds` / `greedy_nopq_seconds`. |
| `system_cpu/` | Process user/system/total CPU seconds, `process_utilization_percent`, and native W&B CPU readings when tracking. |
| `system_gpu/` | `cuda_enabled`, PyTorch allocated/reserved/peak GPU memory in GiB, and available native W&B GPU utilization, memory, temperature, and power readings. |
| `system_memory/` | Process resident, peak resident, virtual, and swap memory in GiB, plus native W&B host-memory readings when tracking. |
| `system_general/` | `elapsed_seconds` and native W&B disk, network, and other system readings when tracking. |



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

