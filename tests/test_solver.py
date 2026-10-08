import pytest

from collections.abc import Callable
from pydantic import ValidationError

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


# @pytest.mark.xfail(
#     strict=True,
#     reason="Validation of negative widths and heights is not implemented yet.",
# )
def test_negative_width(make_instance: Callable, mini: Instance) -> None:
    with pytest.raises(ValueError):
        make_instance(w=[-1, *mini.w[1:]])


def test_negative_height(make_instance: Callable, mini: Instance) -> None:
    with pytest.raises(ValueError):
        make_instance(h=[-1, *mini.h[1:]])


def test_mini_reaches_optimal_solution(mini: Instance) -> None:
    result = solve(mini, time_limit=10)

    assert result.is_optimal
    assert result.objective_value == pytest.approx(131, rel=1e-5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_items", -1),
        ("n_elements", -1),
        ("n_layers", -1),
        ("w", [-1, 2, 3]),
        ("h", [1, -2, 3]),
        ("W", -10),
        ("H", -10),
        ("p", [-1, 2, 3]),
        ("s", [1, -2, 3]),
        ("s_s", [1, 2, -3]),
        ("c", [1, 2, -3]),
        ("Q", -5),
    ],
)
def test_invalid_instance_fields(
    field: str, value: object, make_instance: Callable[..., Instance]
) -> None:
    kwargs = {field: value}
    with pytest.raises(ValidationError):
        make_instance(**kwargs)
