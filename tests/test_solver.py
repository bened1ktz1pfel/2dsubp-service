import pytest

from collections.abc import Callable

from su2dbp.models import Instance
from su2dbp.solver import solve


def test_one_item_creates_one_batch(single: Instance) -> None:
    result = solve(single, time_limit=10)

    assert result.is_optimal
    assert len(result.batches) == 1
    assert result.batches[0].items == (0,)


def test_each_item_is_assigned_exactly_once(mini: Instance) -> None:
    result = solve(mini, time_limit=10)

    assigned = sorted(i for b in result.batches for i in b.items)
    assert assigned == list(range(mini.n_items))


@pytest.mark.xfail(
    strict=True,
    reason="Validation of negative widths and heights is not implemented yet.",
)
def test_negative_width_height(make_instance: Callable, mini: Instance) -> None:
    with pytest.raises(ValueError):
        make_instance(w=[-1, *mini.w[1:]])
