"""Fail closed when unknown human contacts might become robot contact labels."""

from dataclasses import dataclass

from flash_chord.objectives.config import ObjectiveConfig


@dataclass(frozen=True)
class Dex3ObjectiveConfig(ObjectiveConfig):
    estimated_contact_guidance: bool = False

    def __post_init__(self):
        super().__post_init__()
        if type(self.estimated_contact_guidance) is not bool:
            raise ValueError("Estimated guidance needs an explicit boolean")
        for name in (
            "contact_wrench_support",
            "missed_contact",
            "unintended_contact",
            "force_closure",
        ):
            term = getattr(self, name)
            if name == "contact_wrench_support" and self.estimated_contact_guidance:
                if not term.enabled or term.weight <= 0:
                    raise ValueError("Estimated guidance must explicitly enable positive CWS reward")
                continue
            if term.enabled or term.weight != 0:
                raise ValueError(
                    f"{name} requires a validated contact reference; unknown labels remain disabled"
                )

    def build(self, model, embodiment, robot_reference, reference, *args, **kwargs):
        if self.estimated_contact_guidance:
            from .estimated_reference import SCHEMA
            if reference.metadata.schema_version != SCHEMA or reference.metadata.source_dataset != "local_development_estimated_surface":
                raise ValueError("Estimated CWS requires a hash-bound estimated surface reference")
        return super().build(model, embodiment, robot_reference, reference, *args, **kwargs)
