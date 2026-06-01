from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitecturePreset:
    """Curated mapping from a small public name to trainer CLI overrides."""

    name: str
    summary: str
    trainer_overrides: tuple[tuple[str, str], ...]


ARCHITECTURE_PRESETS: dict[str, ArchitecturePreset] = {
    "symmetric-per-entity-mlp": ArchitecturePreset(
        name="symmetric-per-entity-mlp",
        summary=(
            "Symmetric per-entity MLP pair encoder with pair-only scoring. "
            "This is the leanest public preset."
        ),
        trainer_overrides=(
            ("actor.state_embedding_head", "None"),
            ("actor.state_features_embedding_head", "None"),
            ("actor.pair_embedding_head", "symmetric-per-entity-mlp"),
            ("actor.pair_combiner", "pair-only"),
        ),
    ),
    "spem-deep-set": ArchitecturePreset(
        name="spem-deep-set",
        summary=(
            "Symmetric per-entity MLP pair encoder plus a deep-set state "
            "encoder with a FiLM pair combiner. This is the default public preset."
        ),
        trainer_overrides=(
            ("actor.state_embedding_head", "deep-set"),
            ("actor.state_features_embedding_head", "None"),
            ("actor.pair_embedding_head", "symmetric-per-entity-mlp"),
            ("actor.pair_combiner", "film"),
        ),
    ),
    "spem-deep-set-state-features": ArchitecturePreset(
        name="spem-deep-set-state-features",
        summary=(
            "Symmetric per-entity MLP pair encoder plus a deep-set state "
            "encoder and an MLP state-feature encoder, using FiLM combiners."
        ),
        trainer_overrides=(
            ("actor.state_embedding_head", "deep-set"),
            ("actor.state_features_embedding_head", "mlp"),
            ("actor.state_combiner", "film"),
            ("actor.pair_embedding_head", "symmetric-per-entity-mlp"),
            ("actor.pair_combiner", "film"),
        ),
    ),
}


def get_architecture_preset(name: str) -> ArchitecturePreset:
    try:
        return ARCHITECTURE_PRESETS[name]
    except KeyError as exc:  # pragma: no cover - tyro constrains values at CLI time
        available = ", ".join(sorted(ARCHITECTURE_PRESETS))
        raise ValueError(f"Unknown architecture preset '{name}'. Available: {available}") from exc
