"""Load training ideals while preserving each ideal's polynomial ring."""

from dataclasses import dataclass
import json
from pathlib import Path
import re


def normalize_coefficient_ring(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("coeff_ring must be a non-empty string.")
    value = value.strip()
    if value in {"QQ", "ZZ"}:
        return value
    match = re.fullmatch(r"ZZmod\(\s*(\d+)\s*\)", value)
    if match and int(match[1]) > 1:
        return f"ZZmod({int(match[1])})"
    raise ValueError("coeff_ring must be QQ, ZZ, or ZZmod(p) with p > 1.")


def split_ring_symbols(value: str) -> tuple[str, ...]:
    """Split ordered indeterminates, preserving commas inside indexed names."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("symbols must be a non-empty comma-separated string.")
    tokens = []
    start = 0
    depth = 0
    for position, char in enumerate(value):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("symbols contains unmatched brackets.")
        elif char == "," and depth == 0:
            tokens.append(value[start:position].strip())
            start = position + 1
    tokens.append(value[start:].strip())
    if depth or not all(tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("symbols must contain distinct variable names with balanced brackets.")
    return tuple(tokens)


@dataclass(frozen=True)
class IdealCorpus:
    # The first widest ideal supplies representative ring metadata for snapshots.
    coeff_ring: str
    symbols: str
    bases: tuple[tuple[str, ...], ...]
    coeff_rings: tuple[str, ...]
    ring_symbols: tuple[str, ...]

    @property
    def n_vars(self) -> int:
        return len(split_ring_symbols(self.symbols))

    def rows(self) -> list[dict]:
        return [
            {
                "coeff_ring": coeff_ring,
                "symbols": symbols,
                "basis_elements": list(basis),
            }
            for coeff_ring, symbols, basis in zip(self.coeff_rings, self.ring_symbols, self.bases)
        ]


def load_ideal_corpus(path: str | Path) -> IdealCorpus:
    """Read JSONL and size observations for the ideal with the most variables."""
    bases = []
    coeff_rings = []
    ring_symbols = []
    widest_ring = None
    max_n_vars = 0
    with Path(path).open(encoding="utf-8") as handle:
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
            except ValueError as error:
                raise ValueError(f"Invalid ideal corpus {path}, line {line_number}: {error}") from error
            ring = (coeff_ring, ",".join(symbols))
            if len(symbols) > max_n_vars:
                widest_ring = ring
                max_n_vars = len(symbols)
            coeff_rings.append(ring[0])
            ring_symbols.append(ring[1])
            bases.append(tuple(polynomial.strip() for polynomial in basis))
    if widest_ring is None:
        raise ValueError(f"Ideal corpus {path} must contain at least one ideal.")
    return IdealCorpus(*widest_ring, tuple(bases), tuple(coeff_rings), tuple(ring_symbols))


def validate_corpus_options(corpus, *, coeff_ring, symbols, basis_elements, fix_ideal):
    if basis_elements is not None or fix_ideal:
        raise ValueError("ideal_corpus cannot be combined with basis_elements or fix_ideal=True.")
    if coeff_ring is not None and normalize_coefficient_ring(coeff_ring) != corpus.coeff_ring:
        raise ValueError("coeff_ring must match the first widest ideal in the ideal corpus.")
    if symbols is not None and ",".join(split_ring_symbols(symbols)) != corpus.symbols:
        raise ValueError("symbols must match the ordered variables of the first widest ideal in the ideal corpus.")
