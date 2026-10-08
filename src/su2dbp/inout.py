import json

from .models import Instance, Solution


def load_instance_from_json(json_file: str) -> Instance:
    with open(json_file, "r") as f:
        data = json.load(f)

    return Instance(
        n_items=data["n_items"],
        n_elements=data["n_elements"],
        n_layers=data["n_layers"],
        w=data["w"],
        h=data["h"],
        W=data["W"],
        H=data["H"],
        p=data["p"],
        s=data["s"],
        s_s=data["s_s"],
        c=data["c"],
        Q=data["Q"],
        E_il=[[set(layer) for layer in item_layers] for item_layers in data["E_il"]],
        n_batches_org=data.get("n_batches", None),
    )


def write_solution_to_json(solution: Solution, filename: str) -> None:
    """Write the solution to a JSON file."""
    solution_dict = {
        "objective_value": solution.objective_value,
        "objective_bound": solution.objective_bound,
        "gap": solution.gap,
        "status": solution.status,
        "runtime": solution.runtime,
        "batches": [
            {
                "batch": batch.index,
                "items": batch.items,
                "elements": batch.elements,
                "processing_time": batch.processtime,
                "placement": {i: (xi, yi) for i, xi, yi in batch.placements},
            }
            for batch in solution.batches
        ],
    }
    with open(filename, "w") as f:
        json.dump(solution_dict, f, indent=4)
