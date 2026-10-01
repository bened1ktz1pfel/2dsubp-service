import json

import numpy as np


def create_instance(
    n: int, e: int, l: int, W: int, H: int, rng: np.random.Generator
) -> dict:
    """Create a sample instance for testing."""
    instance_dict = {
        "n_items": n,
        "n_elements": e,
        "n_layers": l,
        "W": W,
        "H": H,
        "w": rng.integers(W // 10, W // 2, size=n).tolist(),
        "h": rng.integers(H // 10, H // 2, size=n).tolist(),
        "p": rng.integers(20, 120, size=n).tolist(),
        "s_b": rng.integers(1, 5, size=e).tolist(),
        "s_s": rng.integers(1, 2, size=e).tolist(),
        "c": [1 for _ in range(e)],
        "Q": rng.integers(3, e + 1, size=1).tolist()[0],
        "n_batches": n,
    }
    e_i = []
    e_il = []
    for i in range(n):
        item_layers = [int, ...]
        l_curr = rng.integers(
            int(0.5 * l), l + 1
        )  # Randomly select number of layers for this item
        num_elements = rng.integers(1, instance_dict["Q"] + 1)
        elements = set(rng.choice(range(e), size=num_elements, replace=False).tolist())
        e_i.append(list(elements))
        for j in range(l):
            if j >= l_curr:
                item_layers.append([])  # No elements required for this layer
                continue
            layer_elements = set(
                rng.choice(
                    list(elements),
                    size=rng.integers(1, len(elements) + 1),
                    replace=False,
                ).tolist()
            )
            item_layers.append(list(layer_elements))
        e_il.append(item_layers)

    instance_dict["E_il"] = e_il

    return instance_dict


def write_instance_to_json(instance: dict, filename: str) -> None:
    """Write the instance dictionary to a JSON file."""
    with open(filename, "w") as f:
        json.dump(instance, f, indent=4)


if __name__ == "__main__":
    # Example usage
    rng = np.random.default_rng(seed=42)
    instance = create_instance(n=5, e=4, l=50, W=25, H=25, rng=rng)
    write_instance_to_json(instance, "tests/sample_instance.json")
