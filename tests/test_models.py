import pytest

from collections.abc import Callable

from su2dbp.models import Instance


def test_item_too_big(make_instance: Callable[..., Instance], mini: Instance) -> None:
    with pytest.raises(ValueError, match=r"Item\(s\) \[0\] do not fit in the bin"):
        make_instance(w=[mini.W + 5, *mini.w[1:]])


def test_unknown_element(
    make_instance: Callable[..., Instance], mini: Instance
) -> None:
    damaged_E_il = [[set(elements) for elements in row] for row in mini.E_il]
    damaged_E_il[0][0].add(mini.n_elements)  # Add an unknown element
    with pytest.raises(
        ValueError, match=r"Unknown element \(Item, Layer, Element\): *\[\(0, 0, 2\)\]"
    ):
        make_instance(E_il=damaged_E_il)


def test_inconsistent_lengths(
    make_instance: Callable[..., Instance], mini: Instance
) -> None:
    with pytest.raises(ValueError, match=r"Length of w is 1, expected 3"):
        make_instance(w=[mini.w[0]])
