from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from su2dbp.inout import load_instance_from_json
from su2dbp.models import Instance

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def single() -> Instance:
    """Load the single item instance from JSON."""
    return load_instance_from_json(FIXTURES_DIR / "single.json")


@pytest.fixture
def mini() -> Instance:
    """Load the mini instance from JSON."""
    return load_instance_from_json(FIXTURES_DIR / "mini.json")


@pytest.fixture
def sample_instance() -> Instance:
    """Load the sample instance from JSON."""
    return load_instance_from_json(FIXTURES_DIR / "sample_instance.json")


@pytest.fixture
def make_instance(mini: Instance) -> Callable[..., Instance]:
    """Return a function that creates an instance from a dictionary."""

    def _make_instance(**overrides: Any) -> Instance:
        return Instance.model_validate({**mini.model_dump(), **overrides})

    return _make_instance
