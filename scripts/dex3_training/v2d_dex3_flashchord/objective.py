"""Fail closed when unknown human contacts might become robot contact labels."""

from dataclasses import dataclass

from flash_chord.objectives.config import ObjectiveConfig


@dataclass(frozen=True)
class Dex3ObjectiveConfig(ObjectiveConfig):
    def __post_init__(self):
        super().__post_init__()
        for name in (
            "contact_wrench_support",
            "missed_contact",
            "unintended_contact",
            "force_closure",
        ):
            term = getattr(self, name)
            if term.enabled or term.weight != 0:
                raise ValueError(
                    f"{name} requires measured Dex3 contact labels; disable it"
                )
