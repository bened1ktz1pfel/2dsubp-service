from collections.abc import Callable

from su2dbp.models import Instance, Solution
from su2dbp.solver import solve


def not_worse(solution1: Solution, solution2: Solution, rel_tol: float = 1e-5) -> bool:
    return solution1.objective_value <= solution2.objective_value * (1 + rel_tol)


def test_more_capacity_is_not_worse(mini: Instance, make_instance: Callable) -> None:
    original_solution = solve(mini, time_limit=10)

    increased_capacity_instance = make_instance(
        W=int(mini.W * 1.5), H=int(mini.H * 1.5)
    )

    increased_capacity_solution = solve(increased_capacity_instance, time_limit=10)

    assert not_worse(increased_capacity_solution, original_solution), (
        f"Solution with increased capacity is worse: "
        f"{increased_capacity_solution.objective_value} > {original_solution.objective_value}"
    )

    increased_element_capacity_instance = make_instance(Q=mini.Q + 2)

    increased_element_capacity_solution = solve(
        increased_element_capacity_instance, time_limit=10
    )

    assert not_worse(increased_element_capacity_solution, original_solution), (
        f"Solution with increased element capacity is worse: "
        f"{increased_element_capacity_solution.objective_value} > {original_solution.objective_value}"
    )


def test_more_batches_is_not_worse(mini: Instance, make_instance: Callable) -> None:
    original_solution = solve(mini, time_limit=10)

    increased_batches_instance = make_instance(n_batches_org=mini.n_batches_org * 2)

    increased_batches_solution = solve(increased_batches_instance, time_limit=10)

    assert not_worse(increased_batches_solution, original_solution), (
        f"Solution with increased number of batches is worse: "
        f"{increased_batches_solution.objective_value} > {original_solution.objective_value}"
    )
