"""Compute empirical/circuit TV bounds and print CSV."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from sparc import CategoricalInputNode, CircuitNode, ProductNode, SumNode


DATASET_TYPE = "test"
DELTA = 0.1
DEBD_DIR = Path("DEBD/datasets")
PC_DIR = Path("likelihood_learned_pcs/hclt")

DATASETS = sorted({
    "accidents", "nltcs", "plants", "baudio", "bbc", "bnetflix",
    "book", "c20ng", "connect4", "cr52", "cwebkb", "dna", "jester",
    "kdd", "kosarek", "moviereview", "msnbc", "mushrooms", "nips",
    "ocr_letters", "pumsb_star", "rcv1", "tmovie", "tretail",
    "voting", "ad",
})

CSV_COLUMNS = [
    "Dataset",
    "number of variables",
    "number of samples",
    "number of PC parameters",
    "empirical upper bound",
    "DTV(empirical est,Circuit)",
    "final upper bound",
]


def load_raw_dataset(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        data = np.asarray(
            [[int(value) for value in row] for row in csv.reader(f)],
            dtype=np.int32,
        )
    if data.ndim != 2 or not all(data.shape):
        raise ValueError(f"Expected a nonempty 2D dataset: {path}")
    return data


def count_pc_parameters(circuit: CircuitNode) -> int:
    """Count stored parameters once per node, using SparC node IDs."""
    total = 0
    visited_ids = set()
    stack = [circuit]

    while stack:
        node = stack.pop()
        if node.id in visited_ids:
            continue
        visited_ids.add(node.id)

        if isinstance(node, SumNode):
            total += len(node.parameters_list())
            stack.extend(node.children())
        elif isinstance(node, ProductNode):
            stack.extend(node.children())
        elif isinstance(node, CategoricalInputNode):
            total += len(node.probabilities_list())
        else:
            raise TypeError(
                f"Unsupported node type for parameter counting: "
                f"{type(node).__name__}"
            )

    return total


def compute_upper_bounds(counts, pc_vals, delta=DELTA):
    """Return (empirical upper bound, empirical/circuit TV, final upper bound).
    """
    counts = np.asarray(counts)

    # One likelihood per unique state, in the same order as counts.
    pc_vals = np.asarray(pc_vals).reshape(-1)
    if pc_vals.shape != counts.shape:
        raise ValueError("Expected one circuit probability per unique state.")
    if not np.all(np.isfinite(pc_vals)):
        raise ValueError("Circuit probabilities contain NaN or infinity.")

    tolerance = 1e-8
    if np.any(pc_vals < -tolerance) or np.any(pc_vals > 1 + tolerance):
        raise ValueError("Circuit likelihoods must be probabilities.")
    circuit_probs = np.clip(pc_vals.astype(np.float64), 0.0, 1.0)
    pc_seen_mass = math.fsum(circuit_probs.tolist())
    if pc_seen_mass > 1 + tolerance:
        raise ValueError("Circuit mass on unique observed states exceeds 1.")

    m = int(counts.sum())
    empirical_probs = counts.astype(np.float64) / m
    phi = float(np.sqrt(counts.astype(np.float64)).sum() / m)
    confidence = 3.0 * math.sqrt(math.log(2.0 / delta) / (2.0 * m))
    empirical_upper_bound = phi + confidence

    tv_seen = float(np.abs(empirical_probs - circuit_probs).sum() / 2.0)
    unseen_pc = max(0.0, 1.0 - pc_seen_mass) / 2.0
    dtv_empirical_circuit = tv_seen + unseen_pc
    final_upper_bound = empirical_upper_bound + dtv_empirical_circuit
    return empirical_upper_bound, dtv_empirical_circuit, final_upper_bound

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("train", "valid", "test"), default=DATASET_TYPE,
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--output", type=Path, help="CSV output path.")
    parser.add_argument(
        "--tex-output", type=Path,
        help="LaTeX output path; defaults to the CSV path with a .tex extension.",
    )
    args = parser.parse_args()

    results_csv = args.output or Path(f"upper_bounds_{args.split}.csv")
    results_tex = args.tex_output or results_csv.with_suffix(".tex")
    if results_csv.resolve() == results_tex.resolve():
        parser.error("CSV and LaTeX output paths must be different.")

    rows = []
    print(" | ".join(CSV_COLUMNS))

    for name in args.datasets:
        data_path = DEBD_DIR / name / f"{name}.{args.split}.data"
        circuit_path = (
            PC_DIR / name / "4" / f"hclt_{name}_blocksize4_seed0.json"
        )
        if not data_path.exists() or not circuit_path.exists():
            print(f"Skipping {name}: missing dataset or circuit.")
            continue

        data = load_raw_dataset(data_path)
        num_samples, num_variables = data.shape
        circuit = CircuitNode.load(str(circuit_path))
        if set(circuit.scope_as_list()) != set(range(num_variables)):
            raise ValueError(f"{name}: circuit scope does not match dataset.")
        num_parameters = count_pc_parameters(circuit)

        states, counts = np.unique(data, axis=0, return_counts=True)
        pc_vals = circuit.likelihood(states)
        empirical_bound, dtv, final_bound = compute_upper_bounds(
            counts, pc_vals, args.delta,
        )

        row = dict(zip(CSV_COLUMNS, [
            name,
            num_variables,
            num_samples,
            num_parameters,
            f"{empirical_bound:.6f}",
            f"{dtv:.6f}",
            f"{final_bound:.6f}",
        ]))
        rows.append(row)
        print(" | ".join(str(row[column]) for column in CSV_COLUMNS))

    if not rows:
        raise FileNotFoundError(
            "No matching dataset/circuit pairs found. Check DEBD_DIR and PC_DIR."
        )

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to {results_csv.resolve()}")
    print(f"Saved LaTeX table to {results_tex.resolve()}")


if __name__ == "__main__":
    main()
