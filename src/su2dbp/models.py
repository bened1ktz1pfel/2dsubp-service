from dataclasses import dataclass
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)


class Instance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_items: PositiveInt
    n_elements: PositiveInt
    n_layers: PositiveInt
    w: list[PositiveInt]
    h: list[PositiveInt]
    W: PositiveInt
    H: PositiveInt
    p: list[NonNegativeInt]
    s: list[NonNegativeInt]
    s_s: list[NonNegativeInt]
    c: list[NonNegativeInt]
    Q: PositiveInt
    E_il: list[list[set[NonNegativeInt]]]
    n_batches_org: PositiveInt | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        for name, values, expected_length in [
            ("w", self.w, self.n_items),
            ("h", self.h, self.n_items),
            ("p", self.p, self.n_items),
            ("E_il", self.E_il, self.n_items),
            ("s", self.s, self.n_elements),
            ("s_s", self.s_s, self.n_elements),
            ("c", self.c, self.n_elements),
        ]:
            if len(values) != expected_length:
                raise ValueError(
                    f"Length of {name} is {len(values)}, expected {expected_length}."
                )

        for i, layers in enumerate(self.E_il):
            if len(layers) != self.n_layers:
                raise ValueError(
                    f"Item {i} has {len(layers)} layers, expected {self.n_layers}."
                )

        unknown = sorted(
            (i, layer, e)
            for i, layers in enumerate(self.E_il)
            for layer, elements in enumerate(layers)
            for e in elements
            if not 0 <= e < self.n_elements
        )
        if unknown:
            raise ValueError(
                f"E_il verweist auf unbekannte Elemente (Item, Layer, Element): {unknown}"
            )

        too_big = [
            i for i in range(self.n_items) if self.w[i] > self.W or self.h[i] > self.H
        ]
        if too_big:
            raise ValueError(
                f"Item(s) {too_big} has/have dimensions larger than the container."
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
