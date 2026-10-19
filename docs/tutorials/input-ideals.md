# Input Ideals

This tutorial explains how to choose the ideals used for training: sample from a
distribution, reuse one generated ideal, supply an exact generating set, or load
a fixed collection from a file. After [installing the package](../../INSTALL.md),
run the commands below from the repository root.

## Choose an Input Mode

| Input mode | Configuration | What happens on reset |
| --- | --- | --- |
| Generated ideals | `--env.ideal-dist ...` | Sample another ideal from the distribution. |
| One generated ideal | `--env.ideal-dist ... --env.fix-ideal True` | Reuse the first sampled ideal. |
| One explicit ideal | `--env.coeff-ring ... --env.symbols ... --env.basis-elements ...` | Reuse the supplied generators in every worker. |
| A file of fixed ideals | `--env.ideal-corpus ideals.jsonl` | Sample an entry uniformly, with replacement. |

The default is generated ideals from `3-20-10-weighted`. These input modes work
with both `BuchbergerExponents-v1` and `BuchbergerExponentsNoPQ-v1`.

## Generate Random Binomial Ideals

This ideal distribution was first used to train S-pair selection in the paper ["Learning selection strategies in Buchberger's algorithm"](https://arxiv.org/abs/2005.01917).  The random-binomial distribution format is:

```text
n-d-s-type
```

| Field | Meaning | Value in `3-20-10-weighted` |
| --- | --- | --- |
| `n` | Number of variables. | 3 |
| `d` | Maximum total degree of either monomial in a generator. | 20 |
| `s` | Number of binomial generators sampled. | 10 |
| `type` | How monomial degrees are sampled. | `weighted` |

For example:

```bash
python run_training.py \
  --env.ideal-dist 3-20-10-weighted \
  --env.env-seed 12
```

The degree choices are:

| Type | Sampling rule |
| --- | --- |
| `weighted` | Choose a degree uniformly from `1` through `d`, then choose a monomial uniformly at that degree. |
| `uniform` | Weight each degree by its number of monomials, giving equal probability to individual monomials of degrees `1` through `d`. |
| `maximum` | Both monomials have degree `d`. |

Each binomial uses two distinct, nonconstant monomials, with the second
coefficient sampled modulo `32003`. The distribution string has exactly four
fields; optional suffixes are not supported.

### Vary the Distribution Parameters

Replace any of `n`, `d`, or `s` with an inclusive range `(minimum-maximum)`.
The generator independently samples each ranged parameter uniformly on each
new ideal:

```bash
# Keep three variables; vary the degree bound and generator count.
python run_training.py --env.ideal-dist '3-(5-10)-(4-8)-weighted'

# Also vary the number of variables.
python run_training.py --env.ideal-dist '(2-4)-(5-10)-(4-8)-weighted'
```

Quote distribution strings containing parentheses so the shell passes them as
one argument. With a variable-count range, observations reserve exponent
coordinates for the maximum variable count and pad unused coordinates.

The built-in random-binomial generators use coefficients modulo `32003` and
variables named `x[0]`, `x[1]`, and so on. To choose a different ring or your own
polynomials, use explicit inputs or a JSONL file as described below.

## Generate Benchmark Ideals

These ideals are not a distribution in the same sense as for the Binomial ideals as there is only one ideal per benchmark family for a given number of variables. They are specified by the family name followed by the number of variables:

```bash
python run_training.py --env.ideal-dist cyclic-3
```

The supported families are:

| Family | Example |
| --- | --- |
| Cyclic | `cyclic-3` |
| Acyclic | `acyclic-3` |
| Eco | `eco-3` |
| Katsura | `katsura-3` |
| Noon | `noon-3` |
| Reimer | `reimer-3` |

A particular family and variable count produce a deterministic generating set.
For example, `cyclic-3` repeatedly supplies the same ideal even when
`fix_ideal=False`.

Use `+` to combine families, and a parenthesized range to vary the variable
count:

```bash
python run_training.py --env.ideal-dist '(cyclic+eco)-(3-6)'
```

This samples uniformly among the valid `(family, variable count)` combinations.
Benchmark generators also use coefficients modulo `32003` and indexed variables
`x[0]`, `x[1]`, and so on. As with random-binomial ranges, observations allow for
the largest variable count.

## Generate Geometric Ideals

Geometric distributions construct distance-constraint ideals from the bundled
Laman graph tables. Use `geom-n`, where `n` is the number of graph vertices, or
`geom-(minimum-maximum)` to sample a vertex count:

```bash
python run_training.py --env.ideal-dist geom-4
python run_training.py --env.ideal-dist 'geom-(3-5)'
```

The bundled tables support **3 through 10 vertices**. A generated ideal samples
a graph and integer coordinates, then builds its polynomial distance constraints
over coefficients modulo `32003`.

Here `n` counts vertices, rather than polynomial variables. After fixing the
translation and rotation freedoms, there are `2*n - 3` variables:
`x2,...,xn,y3,...,yn`. For example, `geom-3` uses `x2,x3,y3`. The graph files are
installed with the package; no separate data path is needed.

## Reuse One Generated Ideal

Set `fix_ideal=True` to sample once and reuse that ideal for subsequent episodes:

```bash
python run_training.py \
  --env.ideal-dist 3-8-6-weighted \
  --env.fix-ideal True \
  --env.env-seed 12 \
  --training.num-envs 1
```

This cache belongs to each environment. In this mode, the trainer gives every
worker the same `env.env_seed`, so workers reuse the same generated ideal.

Greedy evaluation uses this same ideal, sampled with `env.env_seed`, including
its coefficient ring and ordered variables. `training.greedy_eval_seed` controls
the evaluation episode seed; it does not select a different fixed ideal.

## Supply One Explicit Ideal

Provide all three fields together: `coeff_ring`, `symbols`, and `basis_elements`.
For example, train on the ideal generated by `x^2-y` and `x*y-1` in `QQ[x,y]`:

```bash
python run_training.py \
  --env.coeff-ring QQ \
  --env.symbols x,y \
  --env.basis-elements 'x^2-y,x*y-1'
```

The supplied polynomials are an **input generating set**; they do not need to
be a Groebner basis. Every worker reuses this set. The trainer automatically
disables distribution sampling, so you do not need to set `ideal_dist=None` or
`fix_ideal=True` yourself.

When greedy evaluation is enabled, it also uses the supplied ideal.

On the command line, `basis_elements` is a comma-separated string of
polynomials. Use `^` for powers and `*` for multiplication, and quote expressions
to protect them from shell expansion. Every polynomial must use the declared
variables. `symbols` gives their order, so `x,y` and `y,x` describe different
ordered rings.

The fixed-input parser recognizes `QQ`, `ZZ`, and `ZZmod(p)`. To use coefficients
modulo the prime `7` instead of rational coefficients:

```bash
python run_training.py \
  --env.coeff-ring 'ZZmod(7)' \
  --env.symbols x,y \
  --env.basis-elements 'x^2-y,x*y-1'
```

The `coeff_ring` and `symbols` arguments configure explicit inputs; they do not
override the built-in distribution generators. Supplying either one without
the complete explicit input requires a matching JSONL corpus instead.

## Train From a JSONL File

Use a UTF-8 file with one JSON object per line. Each entry must contain
`coeff_ring`, `symbols`, and a non-empty `basis_elements` list:

```jsonl
{"coeff_ring":"QQ","symbols":"x,y","basis_elements":["x^2-y","x*y-1"]}
{"coeff_ring":"ZZmod(7)","symbols":"y,x","basis_elements":["x^3-y","x*y-1"]}
{"coeff_ring":"QQ","symbols":"u,v,w","basis_elements":["u^2-v","u*v-w","v*w-1"]}
```

Save this as `training_ideals.jsonl`, then run:

```bash
python run_training.py \
  --env.ideal-corpus training_ideals.jsonl \
  --env.env-seed 12
```

### Mix Rings and Variable Counts

Each entry supplies its own coefficient ring and ordered variable names.
You can mix `QQ` with `ZZmod(7)`, use different names or orders such as `x,y`
and `y,x`, and vary variable counts, generator counts, and polynomial degrees.
Every ideal is parsed and computed in its declared ring. An ideal can also
leave some of its declared variables unused.

The policy's input widths use the **largest declared variable count in the
entire file**, even if the first ideal or the greedy evaluation subset has
fewer variables. Variable names are not combined across entries. Exponent
coordinates follow each entry's own variable order, and each monomial is
zero-padded to the maximum count. For example, with a maximum of three variables,
`x^2*y` in an `x,y` row is encoded as `(2, 1, 0)`, and `x^2*y` in a `y,x` row
as `(1, 2, 0)`. With `k=2`, two monomials occupy two separate three-coordinate
slots; the two endpoints of an S-pair each keep their own slots.

The file determines all rings. Optional `--env.coeff-ring` and `--env.symbols`
must match the first entry with the largest variable count; these values also
appear in the resolved configuration and checkpoint metadata. They do not
override the other entries. The package uses its native monomial-order defaults;
the file does not provide a per-entry monomial-order setting.

Ring and JSON-schema errors identify the source file and line number. Polynomial
parsing and initial-action errors identify the corpus index during environment
construction. Each generator must be nonzero, and each ideal must have at least
one pending S-pair after the selected elimination strategy initializes it.
Single-generator ideals and ideals whose pairs are all eliminated at reset
cannot be training entries.

### Sampling and Other Options

Every reset samples a row **uniformly, with replacement**. A file is a fixed
collection of available inputs, rather than a sequence traversed in file order.
Duplicate rows therefore increase an ideal's sampling frequency.

`env.env_seed` controls the sampling sequence, and parallel workers receive
distinct seeds. Ideals are parsed once per worker and reused across episodes.
Reset and step info include `ideal_corpus_index`, the zero-based index among
nonblank JSONL entries. Additional fields, such as evaluation `episode_seed`,
do not control training sampling.

`ideal_corpus` overrides `ideal_dist`. It cannot be combined with explicit
`basis_elements` or `fix_ideal=True`.

Files from [`generate_corpus.py`](eval-corpora-and-greedy-eval.md#generate-a-corpus)
can also be reused when their entries meet the training requirements. The
file's stored `coeff_ring` and `symbols` determine the ring used when replaying
it. Generated corpora can give variable-count examples a shared
variable list; unused variables are allowed, but inspect the exported ring and
generators before using a file for training.

## Configure Inputs From Python

Use the same configuration fields when calling the trainer from Python:

```python
from deepgroebner_pytorch_cocoa.config_dataclasses import Config
from deepgroebner_pytorch_cocoa.ppo_training import train

cfg = Config()
cfg.env.ideal_corpus = "/path/to/training_ideals.jsonl"
result = train(cfg)
```

For one explicit ideal instead, start with a new `Config` and set
`cfg.env.coeff_ring = "QQ"`, `cfg.env.symbols = "x,y"`, and
`cfg.env.basis_elements = "x^2-y,x*y-1"` before calling `train`. For generated
inputs, set `cfg.env.ideal_dist` to the desired distribution string.

When constructing a Gym environment directly, pass explicit generators as a
Python list:

```python
import gymnasium as gym
import deepgroebner_pytorch_cocoa  # noqa: F401: registers the Gym environments

env = gym.make(
    "BuchbergerExponents-v1",
    coeff_ring="QQ",
    symbols="x,y",
    basis_elements=["x^2-y", "x*y-1"],
)
try:
    observation, info = env.reset(seed=12)
finally:
    env.close()
```

For a corpus, use `gym.make("BuchbergerExponents-v1",
ideal_corpus="training_ideals.jsonl")` instead. For generated inputs, pass
`ideal_dist="3-8-6-weighted"` and optionally `fix_ideal=True`.

## How Input Choices Affect Greedy Evaluation

Greedy evaluation is disabled until `training.greedy_eval_interval_steps` and
`training.greedy_eval_num_examples` are positive. When enabled, the trainer
automatically prepares fixed examples for all evaluation rounds:

| Training input | Greedy evaluation examples |
| --- | --- |
| A generated distribution | A separately generated, cached evaluation corpus controlled by `greedy_eval_seed`; finite-support distributions like with the benchmark ideals will cap the number of examples and will use ideals that were seen during training. |
| A generated distribution with `fix_ideal=True` | The same ideal cached by the training workers, evaluated once per round. |
| Explicit generators | One episode using the supplied ideal, even if a larger example count is requested. |
| A JSONL training corpus | A fixed subset of corpus entries selected without replacement using `greedy_eval_seed`, capped by `greedy_eval_num_examples` and the corpus size. |

Greedy evaluation preserves each selected entry's ring and variable order, while
using the full training corpus's maximum variable count for policy inputs.

To use a pre-generated evaluation corpus instead of the automatic selection,
set `--training.greedy-eval-corpus path/to/corpus.jsonl`. Every row is used, in
file order, even if the file is larger than `greedy_eval_num_examples`. Entries
may use fewer variables than the policy supports, with zero padding, but cannot
use more. With a single fixed training ideal, an explicit evaluation corpus must
contain only that same ideal.

These are **evaluation corpora** used during training. Use a separate **test
corpus** to compare saved policies and heuristic baselines after training; see
[Evaluation and Test Corpora](eval-corpora-and-greedy-eval.md).
