import csv
import os
import random


def get_result_path(conf, method="random", base_dir="results_csv"):
    scenario = conf.get("scenario")
    dataset = conf.get("result_dataset", "main")

    dataset_dir = os.path.join(
        base_dir,
        dataset,
        method,
    )

    # ------------------------------------------------------------
    # scenario1 / scenario2
    # ------------------------------------------------------------
    if scenario in ("scenario1", "scenario2"):
        os.makedirs(dataset_dir, exist_ok=True)

        if method == "random" and conf.get("random_seed") is not None:
            seed = conf["random_seed"]
            filename = f"{scenario}_seed_{seed}.csv"
        else:
            filename = f"{scenario}.csv"

        return os.path.join(
            dataset_dir,
            filename,
        )

    # ------------------------------------------------------------
    # GvH / HvG mismatch experiments
    # ------------------------------------------------------------
    if scenario in (
        "total_mismatch_donor_to_patient",
        "total_mismatch_patient_to_donor",
    ):
        mismatch_limit = conf.get("total_mismatch_limit", 0)
        direction = conf.get("mismatch_direction")

        direction_names = {
            "donor_to_patient": "GvH",
            "patient_to_donor": "HvG",
        }

        if direction not in direction_names:
            raise ValueError(
                f"Unknown mismatch_direction={direction!r}"
            )

        direction_name = direction_names[direction]

        out_dir = os.path.join(
            dataset_dir,
            direction_name,
        )

        os.makedirs(
            out_dir,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # RANDOM: preserve different seeds
        # --------------------------------------------------------
        if method == "random" and conf.get("random_seed") is not None:
            seed = conf["random_seed"]

            filename = (
                f"mismatch_{mismatch_limit}"
                f"_seed_{seed}.csv"
            )

        else:
            filename = (
                f"mismatch_{mismatch_limit}.csv"
            )

        return os.path.join(
            out_dir,
            filename,
        )

    raise ValueError(
        f"Unknown scenario={scenario!r}"
    )

def call_random_from_graph(
    graph_data, g_m, coverage_percentage, conf, verbose=False
):
    """
    Fast RANDOM selection.

    Saves the full trajectory in the same CSV format as greedy:

        step
        donor
        genotype
        covered_patients
        covered_weight
        percentage
    """

    # ------------------------------------------------------------
    # Output file
    # ------------------------------------------------------------
    log_file = conf.get("result_csv_file")
    if not log_file:
        log_file = get_result_path(
            conf,
            method="random",
            base_dir="results_csv",
        )

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # ------------------------------------------------------------
    # Local variables
    # ------------------------------------------------------------
    patients = graph_data.patients
    num_occ = graph_data.num_occ
    get_neighbors = graph_data.get_neighbors

    total_weight = sum(num_occ[p] for p in patients)

    covered = set()
    covered_weight = 0

    donors = list(graph_data.donors)
    donor_count = len(donors)

    steps = []
    step_logs = []
    step = 0

    rnd = random
    seed = conf.get("random_seed")

    if seed is not None:
        rnd.seed(seed)

    # ------------------------------------------------------------
    # Random selection
    # ------------------------------------------------------------
    while donor_count > 0 and covered_weight < total_weight:
        step += 1

        idx = rnd.randrange(donor_count)
        d = donors[idx]

        # Remove selected donor in O(1)
        donors[idx] = donors[donor_count - 1]
        donor_count -= 1

        neigh = get_neighbors(d)

        gain = 0

        for n in neigh:
            if n not in covered:
                covered.add(n)
                gain += num_occ[n]

        covered_weight += gain

        pct = 100.0 * covered_weight / total_weight

        # --------------------------------------------------------
        # Donor genotype
        # Same logic as in greedy
        # --------------------------------------------------------
        if (
            hasattr(graph_data, "id_to_geno")
            and graph_data.id_to_geno is not None
        ):
            genotype = graph_data.id_to_geno[d]

        elif (
            hasattr(graph_data, "donor_genotypes")
            and graph_data.donor_genotypes is not None
        ):
            genotype = graph_data.donor_genotypes[d]

        else:
            genotype = d

        # --------------------------------------------------------
        # Existing output
        # --------------------------------------------------------
        steps.append({
            "step": step,
            "percentage": pct,
        })

        # --------------------------------------------------------
        # Detailed CSV log
        # --------------------------------------------------------
        step_logs.append({
            "step": step,
            "donor": d,
            "genotype": genotype,
            "covered_patients": len(covered),
            "covered_weight": covered_weight,
            "percentage": pct,
        })

        if verbose:
            print(
                f"[FAST RANDOM] "
                f"step={step}, donor={d}, gain={gain}, pct={pct:.4f}"
            )

        if pct >= coverage_percentage:
            break

    # ------------------------------------------------------------
    # Save trajectory
    # ------------------------------------------------------------
    with open(log_file, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow([
            "step",
            "donor",
            "genotype",
            "covered_patients",
            "covered_weight",
            "percentage",
        ])

        for s in step_logs:
            w.writerow([
                s["step"],
                s["donor"],
                s["genotype"],
                s["covered_patients"],
                s["covered_weight"],
                s["percentage"],
            ])

    if verbose:
        print(
            f"[FAST RANDOM] saved {step} steps to {log_file}; "
            f"final coverage={pct if step else 0.0:.4f}%"
        )

    return {
        "steps": steps,
        "final_step": step,
        "final_percentage": pct if step else 0.0,
        "log": step_logs,
    }