# Evaluation and Test Corpora

An **evaluation corpus** is a fixed set of ideals used for periodic greedy
evaluation during training. A **test corpus** is reserved for comparing saved
policies and heuristic baselines after training. Both use the same JSONL format.


## Evaluation Corpora and Test Corpora

Evaluation corpora support checkpoint selection and hyperparameter tuning. Keep a
separate test corpus for comparisons after training, using a different seed and
an explicit `test_corpora/` path. A different seed does not guarantee different
ideals for finite benchmark families or a single fixed ideal; those tests measure
performance on the specified inputs rather than unseen ideals.

Each JSONL row records an episode seed, source distribution, coefficient ring,
ordered symbols, and fixed basis elements. The same format can supply training
inputs through `--env.ideal-corpus`; see [training from a file of ideals](input-ideals.md#train-from-a-jsonl-file).

## Generate A Corpus

Use `generate_corpus.py` to prepare a separate test corpus after choosing your
training configuration. For example, for a policy trained on `geom-(3-5)` with
evaluation seed `10000`:

```bash
python generate_corpus.py \
  --ideal-distribution 'geom-(3-5)' \
  --examples 30 \
  --seed 20000 \
  --output-path test_corpora/geom_3_5_seed20000.jsonl
```

The main options are `--ideal-distribution`, `--examples`, `--seed`, and
`--output-path`. An existing explicit output file is protected unless
`--overwrite True` is set. Omitting `--output-path` uses the training evaluation cache;
use an explicit path when creating a test corpus.

### Generate From Python Or A Notebook

The installed package exposes the same generation workflow outside the repository:

```python
from deepgroebner_pytorch_cocoa.corpus_generation import CorpusConfig, generate_corpus

result = generate_corpus(
    CorpusConfig(
        ideal_distribution="geom-(3-5)",
        examples=30,
        seed=20000,
        output_path="/path/to/test_corpora/geom_3_5_seed20000.jsonl",
    )
)

print(result.corpus_path)
print(len(result.corpus_rows))
print(result.generated)
```

`generate_corpus()` returns a `CorpusGenerationResult` containing the rows, the
absolute JSONL path, and whether it generated a new corpus. Set `overwrite=True`
to replace an existing explicit output. Relative paths use the caller's working
directory. Omitting `output_path` uses `eval_corpora/greedy_eval/` there; a cache
hit sets `result.generated` to `False`.

## Generating an Evaluation Corpus to use in Training

By default, the training script prepares the evaluation corpus automatically when greedy evaluation
is enabled. For generated distributions, it generates or reuses a cached corpus
under `eval_corpora/greedy_eval/`; no separate corpus-generation command is needed.
The same examples are used at every evaluation round.
 |


To generate and use a existing evaluation corpus, first generate it using the `generate_corpus.py` script and then pass its JSONL path with
`--training.greedy-eval-corpus`. This skips automatic corpus generation and cache
selection. 
```bash
python generate_corpus.py \
  --ideal-distribution '3-20-10-weighted' \
  --examples 50 \
  --seed 10000 \
  --output-path eval_corpora/shared.jsonl

python run_training.py \
  --env.ideal-dist '3-20-10-weighted' \
  --training.greedy-eval-interval-steps 100000 \
  --training.greedy-eval-corpus eval_corpora/shared.jsonl
```






## Evaluate A Saved Checkpoint Against A Test Corpus

The standalone command for evaluating a saved policy on a test corpus is:

```bash
python evaluate_model.py
```

The most important options are:

- `--corpus`: path to the JSONL test corpus
- `--model-name`: full checkpoint path or a substring to resolve inside `--models-dir`
- `--checkpoint-type`: `auto`, `best_greedy_eval`, or `final`
- `--gym-id`: evaluation environment, usually `BuchbergerExponentsNoPQ-v1` or `BuchbergerExponents-v1`
- `--config-path`: optional explicit config snapshot path; if omitted, the CLI first looks for the saved `__config.json` next to the checkpoint
- `--output-dir` or `--output-root`: where evaluation artifacts should be written

Use the test corpus generated above to evaluate a checkpoint trained on
`geom-(3-5)` in the NoPQ environment:

```bash
python evaluate_model.py \
  --corpus test_corpora/geom_3_5_seed20000.jsonl \
  --model-name models/actors/my_run.pt \
  --gym-id BuchbergerExponentsNoPQ-v1
```

Evaluate the best greedy-eval checkpoint resolved by partial name:

```bash
python evaluate_model.py \
  --corpus test_corpora/geom_3_5_seed20000.jsonl \
  --model-name j77pf3uu \
  --models-dir models/actors \
  --checkpoint-type best_greedy_eval \
  --gym-id BuchbergerExponents-v1
```

By default, the command writes a new directory under
`eval_results/test_corpus__.../` containing:

- `episode_results.csv`
- `episode_results.jsonl`
- `summary.json`
- `metadata.json`

### Evaluate From Python Or A Notebook

The installed package exposes the same evaluation workflow directly. Import it
from any project using the Python environment where the package is installed:

```python
from deepgroebner_pytorch_cocoa.model_evaluation import TestCorpusConfig, evaluate

result = evaluate(
    TestCorpusConfig(
        corpus="/path/to/test_corpora/geom_3_5_seed20000.jsonl",
        model_name="/path/to/models/actors/my_run.pt",
        gym_id="BuchbergerExponentsNoPQ-v1",
        device="cpu",
        output_dir="/path/to/evaluation",
    )
)

print(result.summary["return_mean"])
print(result.episode_results)
print(result.output_directory)
print(result.artifact_paths["summary"])
```

`evaluate()` returns an `EvaluationResult` containing summary metrics, per-episode
results, the output directory, and paths to all four reports. It also writes the
same files as `python evaluate_model.py`. The checkpoint's companion
`__config.json` is loaded automatically; set `config_path` to use another snapshot.
Relative input and output paths use the caller's current working directory.

`EvalCorpusConfig` remains a compatibility alias for `TestCorpusConfig`; use
`TestCorpusConfig` in new code to distinguish testing from greedy evaluation
during training.

## Evaluate Heuristic Baselines

Use `evaluate_heuristics.py` to run the package's native Buchberger algorithm
with each pair-selection heuristic on the same test corpus inputs. When
comparing a saved policy with these heuristics, give both commands the same
test corpus. A model checkpoint is not required:

```bash
python generate_corpus.py \
  --ideal-distribution 3-6-4-weighted \
  --examples 20 \
  --seed 20000 \
  --output-path test_corpora/ideals_seed20000.jsonl

python evaluate_heuristics.py --corpus test_corpora/ideals_seed20000.jsonl
```

The default strategies are `first`, `degree`, `normal`, `sugar`, and `random`.
Select a subset or change the elimination and sorting settings explicitly:

```bash
python evaluate_heuristics.py \
  --corpus test_corpora/ideals_seed20000.jsonl \
  --strategies degree normal sugar \
  --elimination gebauermoeller \
  --sort-input False \
  --sort-reducers True \
  --rewards additions \
  --output-dir eval_results/heuristic_baselines
```

Each strategy starts from the same initialized input and selects from the
**full pending S-pair set**. There is no priority-queue observation budget.
Use matching elimination, reducer sorting, input sorting, and reward settings
when comparing these costs with a trained policy.

### Random Runs and Seeds

Repeat each strategy on each ideal with `--repetitions`. This is particularly
useful for the random strategy:

```bash
python evaluate_heuristics.py \
  --corpus test_corpora/ideals_seed20000.jsonl \
  --strategies random \
  --repetitions 10 \
  --seed 123
```

With the default `seed=None`, an entry's `episode_seed` is the base seed; if it
is missing or null, the base is `corpus_index * repetitions`. An explicit
`--seed` overrides entry seeds, using `seed + corpus_index * repetitions` as
the base. Repetition `r`, numbered from zero, adds `r` to that base. Seeds must
fit in a nonnegative signed 32-bit integer.

All strategies receive the same seed for a given ideal and repetition.
Changing strategy order does not change the random trajectory. Repetitions
also rerun deterministic strategies when they are selected. Operation counts
are repeatable; wall-clock measurements naturally vary.

### Reports and Cost Conventions

The script writes a new directory under `eval_results/heuristics__...` by
default. `--output-dir` selects an explicit directory. Existing report files
are protected unless `--overwrite True` is supplied.

| Report | Contents |
| --- | --- |
| `episode_results.csv` | One row per corpus entry, strategy, and repetition. |
| `episode_results.jsonl` | The same per-run results as JSON objects. |
| `summary.json` | Per-strategy means, population standard deviations, minima, medians, and maxima. |
| `metadata.json` | Corpus path, strategies, elimination, sorting, reward settings, repetitions, seed policy, and timing scope. |

Per-run results include the corpus index, optional ideal ID and distribution,
stored episode seed, actual evaluation seed, ring, strategy, repetition, and:

- `polynomial_additions`: additions counted by the package's Buchberger reward
  convention, including one operation for each processed S-pair.
- `reductions`: processed S-pairs, split into `zero_reductions` and
  `nonzero_reductions`.
- `cost`: a nonnegative operation count, selected by `--rewards additions` or
  `--rewards reductions`. Lower is better.
- `reward_return`: the corresponding signed, undiscounted reward, equal to
  `-cost`.
- `elapsed_seconds`: native solve time, including final basis interreduction,
  excluding input parsing and environment initialization. Operation counts
  cover pair processing and do not include final basis interreduction.

The checkpoint evaluator's `return` column records positive accumulated cost;
compare it with this evaluator's `cost` column using matching reward settings.

### Evaluate Heuristics From Python

```python
from deepgroebner_pytorch_cocoa.heuristic_evaluation import (
    HeuristicEvalConfig,
    evaluate_heuristics,
)

result = evaluate_heuristics(HeuristicEvalConfig(
    corpus="/path/to/test_corpora/ideals_seed20000.jsonl",
    strategies=("degree", "sugar", "random"),
    repetitions=5,
    seed=123,
    output_dir="/path/to/eval_results/heuristics",
))

print(result.summary["strategies"]["sugar"]["cost_mean"])
print(result.artifact_paths["episode_results_csv"])
```

`evaluate_heuristics()` returns a `HeuristicEvaluationResult` with the summary,
episode results, output directory, and report paths. Relative paths resolve
against the caller's working directory, so it can be used from notebooks or
other projects with the package installed.

## Runtime and Memory Statistics

Both evaluation commands report runtime and memory automatically, with a concise
console summary in seconds and MiB. The CSV, JSONL, and JSON reports store raw
seconds and bytes. Unavailable memory measurements are `null` in JSON and empty
in CSV. No extra flags are needed. `--verbosity quiet` suppresses progress and
the console summary.

Each episode records `elapsed_seconds` and `cpu_seconds`, process RSS at its
start and end (`rss_start_bytes`, `rss_end_bytes`), their signed difference
(`rss_delta_bytes`), and `process_peak_rss_bytes`. Model evaluation also records
CUDA allocated and reserved memory at the start and end, their signed
differences, and `cuda_peak_allocated_bytes` and `cuda_peak_reserved_bytes` when
running on CUDA.

The timing scopes differ:

- Model episode time covers the policy rollout after environment reset,
  including inference, native steps, and observation conversion. It excludes
  input parsing and reset. CUDA work is synchronized for wall-clock timing.
- Heuristic episode time covers the native solve and final basis interreduction,
  excluding parsing and initialization.
- `summary.json` contains a `resources` object for the complete evaluation,
  including loading and setup, and excluding report writing. This total can
  exceed the sum of episode times.

Model summaries include runtime statistics across episodes; heuristic summaries
include them per strategy. Compare both operation counts and timings: model
rollouts include inference and observation costs that the native heuristic
solver does not incur.

RSS measures the whole process, including Python, PyTorch, and CoCoA; it does not
isolate solver allocations. `process_peak_rss_bytes` is the process lifetime
high-water mark, so earlier work in a notebook can affect it. CUDA peak values
use PyTorch's allocator counters since their last reset; the evaluators do not
reset those global counters. Memory deltas can be negative, and retained
allocations can affect later episodes.
