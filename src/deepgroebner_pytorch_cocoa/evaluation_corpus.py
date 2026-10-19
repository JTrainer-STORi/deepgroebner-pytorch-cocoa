"""Validation and replay of an explicitly supplied training evaluation corpus."""

from __future__ import annotations

import json
from pathlib import Path

from .buchberger_env import parse_nvars_from_ideal_dist
from .ideal_corpus import normalize_coefficient_ring, split_ring_symbols
from .wrapped import CGLeadMonomialsEnv


_MAX_EPISODE_SEED = 2**31 - 1


def training_policy_nvars(cfg) -> int:
    """Return the exponent width selected by the training input configuration."""
    if cfg.env.symbols is not None:
        return len(split_ring_symbols(cfg.env.symbols))
    return parse_nvars_from_ideal_dist(cfg.env.ideal_dist)


def _fixed_training_ideal(cfg) -> dict:
    basis = cfg.env.basis_elements
    if isinstance(basis, str):
        basis = basis.split(",")
    env = CGLeadMonomialsEnv(
        ideal_dist=cfg.env.ideal_dist,
        elimination=cfg.env.elimination,
        rewards=cfg.env.rewards,
        pq_strategy=cfg.env.priority_queue_strategy,
        sort_input=cfg.env.sort_input,
        sort_reducers=cfg.env.sort_reducers,
        fix_ideal=cfg.env.fix_ideal,
        N=cfg.env.N,
        k=cfg.env.k,
        N_P=cfg.env.N_P,
        N_G=cfg.env.N_G,
        coeff_ring=cfg.env.coeff_ring,
        symbols=cfg.env.symbols,
        basis=basis,
    )
    env.seed(cfg.env.env_seed)
    env.reset()
    return env.initial_ideal()


def _canonical_generators(row: dict) -> tuple[str, ...]:
    # Parse through the same native ring and polynomial parser used for training.
    # initial_ideal() retains the original generators before Buchberger reduction.
    env = CGLeadMonomialsEnv(
        ideal_dist=None,
        coeff_ring=row["coeff_ring"],
        symbols=row["symbols"],
        basis=row["basis_elements"],
        elimination="none",
        N=1,
        k=1,
    )
    env.reset()
    return tuple(sorted(env.initial_ideal()["basis_elements"]))


def _validate_fixed_training_ideal(cfg, rows: list[dict], path: Path, line: int) -> None:
    fixed_training = cfg.env.ideal_corpus is None and (
        bool(cfg.env.fix_ideal)
        or cfg.env.basis_elements is not None
        or cfg.env.ideal_dist is None
    )
    if not fixed_training:
        return
    if len(rows) != 1:
        raise ValueError(
            f"Greedy eval corpus {path} must contain exactly one ideal when training "
            "on a single fixed ideal."
        )

    training_ideal = _fixed_training_ideal(cfg)
    row = rows[0]
    same_ring = (
        row["coeff_ring"] == normalize_coefficient_ring(training_ideal["coeff_ring"])
        and split_ring_symbols(row["symbols"]) == split_ring_symbols(training_ideal["symbols"])
    )
    try:
        same_generators = same_ring and (
            _canonical_generators(row) == tuple(sorted(training_ideal["basis_elements"]))
        )
    except (ValueError, RuntimeError) as error:
        raise ValueError(
            f"Invalid greedy eval corpus {path}, line {line}: basis_elements could not "
            f"be parsed in the specified ring: {error}"
        ) from error
    if not same_generators:
        raise ValueError(
            f"Invalid greedy eval corpus {path}, line {line}: the evaluation ideal "
            "must match the single fixed training ideal, including coeff_ring, "
            "ordered symbols, and polynomial generators."
        )


def load_explicit_greedy_eval_corpus(cfg) -> list[dict]:
    """Read all JSONL rows for greedy evaluation without generating or caching ideals."""
    path = Path(cfg.training.greedy_eval_corpus)
    policy_nvars = training_policy_nvars(cfg)
    rows = []
    row_lines = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each ideal must be a JSON object.")
                coeff_ring = normalize_coefficient_ring(row.get("coeff_ring"))
                symbols = split_ring_symbols(row.get("symbols"))
                basis = row.get("basis_elements")
                if not isinstance(basis, list) or not basis or not all(
                    isinstance(polynomial, str) and polynomial.strip() for polynomial in basis
                ):
                    raise ValueError("basis_elements must be a non-empty list of polynomial strings.")
                if len(symbols) > policy_nvars:
                    raise ValueError(
                        f"symbols contains {len(symbols)} variables, but the training policy "
                        f"supports at most {policy_nvars} variables."
                    )
                seed = row.get("episode_seed")
                if seed is None:
                    seed = cfg.training.greedy_eval_seed + len(rows)
                if (
                    isinstance(seed, bool)
                    or not isinstance(seed, int)
                    or not 0 <= seed <= _MAX_EPISODE_SEED
                ):
                    raise ValueError(
                        f"episode_seed must be an integer between 0 and {_MAX_EPISODE_SEED}; "
                        "missing or null seeds use greedy_eval_seed + row index."
                    )
            except ValueError as error:
                raise ValueError(
                    f"Invalid greedy eval corpus {path}, line {line_number}: {error}"
                ) from error
            rows.append(
                {
                    **row,
                    "coeff_ring": coeff_ring,
                    "symbols": ",".join(symbols),
                    "basis_elements": [polynomial.strip() for polynomial in basis],
                    "episode_seed": seed,
                }
            )
            row_lines.append(line_number)
    if not rows:
        raise ValueError(f"Greedy eval corpus {path} must contain at least one ideal.")
    _validate_fixed_training_ideal(cfg, rows, path, row_lines[0])
    return rows
