"""Locations of bundled data used by the Python and native environments."""

from pathlib import Path


def laman_graph_directory() -> Path:
    """Return the installed Laman graph directory, independent of the working directory."""
    return Path(__file__).resolve().parent / "data" / "LamanGraphs3-10"


def laman_graph_path(n: int) -> Path:
    """Return the bundled graph table for a supported number of vertices."""
    if not 3 <= n <= 10:
        raise ValueError("Laman graph tables support vertex counts from 3 to 10.")
    return laman_graph_directory() / f"LamanGraphs{n}.txt"
