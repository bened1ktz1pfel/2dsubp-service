from su2dbp.models import Instance
from su2dbp.solver import solve


def test_one_item_creates_one_batch(single: Instance) -> None:
    result = solve(single, time_limit=10)

    assert result.is_optimal
    assert len(result.batches) == 1
    assert result.batches[0].items == (0,)


def test_each_item_is_assigned_exactly_once(mini: Instance) -> None:
    result = solve(mini, time_limit=10)

    zugewiesen = sorted(i for b in result.batches for i in b.items)
    assert zugewiesen == list(range(mini.n_items))
