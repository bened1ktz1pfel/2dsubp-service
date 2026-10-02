from pathlib import Path

from su2dbp.inout import load_instance_from_json
from su2dbp.solver import solve


def test_ein_item_ergibt_genau_einen_batch() -> None:
    inst = load_instance_from_json(Path("tests/fixtures/single.json"))
    result = solve(inst)

    assert result.is_optimal
    assert len(result.batches) == 1
    assert result.batches[0].items == (0,)


def test_jedes_item_landet_genau_einmal_in_einem_batch() -> None:
    inst = load_instance_from_json(Path("tests/fixtures/mini.json"))
    result = solve(inst)

    zugewiesen = sorted(i for b in result.batches for i in b.items)
    assert zugewiesen == list(range(inst.n_items))
