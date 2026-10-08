from typing import Self

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator


class Instance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_items: int
    n_elements: int
    n_layers: int
    w: list[PositiveInt]
    h: list[PositiveInt]
    W: PositiveInt
    H: PositiveInt
    p: list[int]
    s: list[int]
    s_s: list[int]
    c: list[int]
    Q: PositiveInt
    E_il: list[list[set[int]]]
    n_batches_org: int | None = None

    @model_validator(mode="after")
    def _check_layers(self) -> Self:
        for i, layers in enumerate(self.E_il):
            if len(layers) != self.n_layers:
                raise ValueError(
                    f"Item {i} has {len(layers)} layers, expected {self.n_layers}."
                )
        return self

    @property
    def n_batches(self) -> int:
        return self.n_batches_org if self.n_batches_org is not None else self.n_items


@dataclass(frozen=True)
class Placement:
    item_index: int
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Batch:
    index: int
    items: tuple[int, ...]
    elements: tuple[tuple[int, ...], ...]
    processtime: float
    placements: tuple[Placement, ...]


@dataclass(frozen=True)
class Solution:
    objective_value: float
    objective_bound: float
    gap: float
    status: str
    runtime: float
    batches: tuple[Batch, ...]

    @property
    def is_optimal(self) -> bool:
        return round(self.gap, 5) == 0.0
