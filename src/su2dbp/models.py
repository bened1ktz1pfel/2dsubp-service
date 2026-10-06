from dataclasses import dataclass


@dataclass(frozen=True)
class Instance:
    n_items: int
    n_elements: int
    n_layers: int
    w: list[float]
    h: list[float]
    W: float
    H: float
    p: list[float]
    s: list[float]
    s_s: list[float]
    c: list[float]
    Q: float
    E_il: list[list[set[int]]]
    n_batches_org: int | None = None

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
