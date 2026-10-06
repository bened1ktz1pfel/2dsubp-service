from su2dbp.models import Instance, Placement
from su2dbp.solver import solve

EPS = 1e-6


def overlap(p1: Placement, p2: Placement) -> bool:
    return not (
        p1.x + p1.width <= p2.x + EPS
        or p2.x + p2.width <= p1.x + EPS
        or p1.y + p1.height <= p2.y + EPS
        or p2.y + p2.height <= p1.y + EPS
    )


def test_no_overlap_in_batches(mini: Instance) -> None:
    result = solve(mini, time_limit=10)

    assert any(
        len(b.placements) >= 2 for b in result.batches
    ), "No batch has more than one placement."

    for batch in result.batches:
        placements = batch.placements
        for i in range(len(placements)):
            assert (
                -EPS <= placements[i].x
                and placements[i].x + placements[i].width <= mini.W + EPS
            ), f"Item {placements[i].item_index} in batch {batch.index} is out of bounds in x-direction."
            assert (
                -EPS <= placements[i].y
                and placements[i].y + placements[i].height <= mini.H + EPS
            ), f"Item {placements[i].item_index} in batch {batch.index} is out of bounds in y-direction."
            for j in range(i + 1, len(placements)):
                assert not overlap(
                    placements[i], placements[j]
                ), f"Overlap detected in batch {batch.index} between items {placements[i].item_index} and {placements[j].item_index}."
