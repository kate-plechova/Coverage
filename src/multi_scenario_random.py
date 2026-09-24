import csv
import os
import random

from .multi_scenario_greedy import donor_key


def call_multi_scenario_random(
    graphs,
    coverage_percentage=99.0,
    max_donors=None,
    seed=12345,
    progress_every=100,
    log_file="logs/multi_scenario_random.csv",
    verbose=False,
):
    """Select one shared random donor sequence and evaluate it in every scenario."""
    if not graphs:
        raise ValueError("graphs must contain at least one scenario graph")

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    scenario_states = {}
    candidates = set()

    for item in graphs:
        name = item["name"]
        graph = item["graph"]
        patients = {int(p) for p in graph.patients}
        donor_by_key = {}

        for donor in graph.donors:
            donor = int(donor)
            try:
                if not bool(graph.active[donor]):
                    continue
            except (KeyError, IndexError, TypeError):
                continue

            key = donor_key(graph, donor)
            donor_by_key[key] = donor
            candidates.add(key)

        scenario_states[name] = {
            "graph": graph,
            "patients": patients,
            "donor_by_key": donor_by_key,
            "covered": set(),
            "covered_weight": 0,
            "total_weight": sum(int(graph.num_occ[p]) for p in patients),
        }

    candidate_order = sorted(candidates, key=lambda key: (key[0], str(key[1])))
    random.Random(seed).shuffle(candidate_order)
    rows = []

    fields = [
        "method", "scenario", "step", "donor_key", "genotype",
        "new_vertices", "new_weight", "covered_vertices",
        "covered_weight", "total_weight", "percentage",
    ]

    with open(log_file, "w", newline="") as log_handle:
        log_writer = csv.DictWriter(log_handle, fieldnames=fields)
        log_writer.writeheader()

        for name, state in scenario_states.items():
            log_writer.writerow({
                "method": "multi_random",
                "scenario": name,
                "step": 0,
                "donor_key": "",
                "genotype": "",
                "new_vertices": 0,
                "new_weight": 0,
                "covered_vertices": 0,
                "covered_weight": 0,
                "total_weight": state["total_weight"],
                "percentage": 0.0,
            })
        log_handle.flush()

        for key in candidate_order:
            if max_donors is not None and len(rows) >= max_donors:
                break

            if all(
                state["total_weight"] == 0
                or 100.0 * state["covered_weight"] / state["total_weight"] >= coverage_percentage
                for state in scenario_states.values()
            ):
                break

            updates = {}
            has_gain = False

            for name, state in scenario_states.items():
                donor = state["donor_by_key"].get(key)
                newly_covered = set()

                if donor is not None:
                    newly_covered = {
                        int(p)
                        for p in state["graph"].get_neighbors(donor)
                        if int(p) in state["patients"] and int(p) not in state["covered"]
                    }

                new_weight = sum(int(state["graph"].num_occ[p]) for p in newly_covered)
                has_gain = has_gain or new_weight > 0
                updates[name] = (newly_covered, new_weight)

            # Match the single-scenario random baseline: donors with zero gain
            # in every scenario are not counted as selected steps.
            if not has_gain:
                continue

            step = len(rows) + 1
            scenario_row = {}

            for name, state in scenario_states.items():
                newly_covered, new_weight = updates[name]
                state["covered"].update(newly_covered)
                state["covered_weight"] += new_weight
                total_weight = state["total_weight"]
                percentage = (
                    100.0 * state["covered_weight"] / total_weight
                    if total_weight > 0 else 100.0
                )
                scenario_row[name] = {
                    "new_vertices": len(newly_covered),
                    "new_weight": new_weight,
                    "covered_vertices": len(state["covered"]),
                    "covered_weight": state["covered_weight"],
                    "total_weight": total_weight,
                    "percentage": percentage,
                }

            row = {
                "step": step,
                "donor_key": f"{key[0]}:{key[1]}",
                "genotype": key[1],
                "scenarios": scenario_row,
            }
            rows.append(row)

            for name, state in scenario_row.items():
                log_writer.writerow({
                    "method": "multi_random",
                    "scenario": name,
                    "step": step,
                    "donor_key": row["donor_key"],
                    "genotype": row["genotype"],
                    "new_vertices": state["new_vertices"],
                    "new_weight": state["new_weight"],
                    "covered_vertices": state["covered_vertices"],
                    "covered_weight": state["covered_weight"],
                    "total_weight": state["total_weight"],
                    "percentage": state["percentage"],
                })
            log_handle.flush()

            if verbose:
                coverage_text = " | ".join(
                    f"{name}={scenario_row[name]['percentage']:.4f}%"
                    for name in scenario_states
                )
                print(f"[MULTI RANDOM] step={step:,} | {coverage_text}", flush=True)
            elif progress_every and step % progress_every == 0:
                coverage_text = " | ".join(
                    f"{name}={scenario_row[name]['percentage']:.4f}%"
                    for name in scenario_states
                )
                print(f"[MULTI RANDOM] step={step:,} | {coverage_text}", flush=True)

    final_states = {}
    for name, state in scenario_states.items():
        total_weight = state["total_weight"]
        final_states[name] = {
            "covered_vertices": len(state["covered"]),
            "covered_weight": state["covered_weight"],
            "total_weight": total_weight,
            "percentage": (
                100.0 * state["covered_weight"] / total_weight
                if total_weight > 0 else 100.0
            ),
        }

    return {
        "scenario_names": list(scenario_states),
        "num_selected_donors": len(rows),
        "final_step": len(rows),
        "scenarios": final_states,
        "log": rows,
        "random_seed": seed,
        "log_file": log_file,
        "csv_file": log_file,
    }
