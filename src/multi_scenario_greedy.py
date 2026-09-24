import csv
import heapq
import json
import os
import time
from copy import deepcopy


def load_scenarios(scenarios_file="scenarios.json"):
    with open(scenarios_file, "r") as f:
        scenarios = json.load(f)

    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError(
            f"Scenarios file {scenarios_file!r} must contain a non-empty JSON object"
        )

    return scenarios


def get_donor_genotype(graph_data, donor):
    if hasattr(graph_data, "id_to_geno") and graph_data.id_to_geno is not None:
        return graph_data.id_to_geno[int(donor)]

    if (
        hasattr(graph_data, "donor_genotypes")
        and graph_data.donor_genotypes is not None
    ):
        return graph_data.donor_genotypes[int(donor)]

    return str(donor)


def get_donor_num_occ(graph_data, donor):
    try:
        return int(graph_data.num_occ[int(donor)])
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def get_donor_is_knockout(graph_data, donor):
    """
    Для lol_unified donor_is_knockout индексируется локальным donor id.
    В старых графах этого массива может не быть -> считаем full donor.
    """
    try:
        return bool(graph_data.donor_is_knockout[int(donor)])
    except (KeyError, IndexError, TypeError, AttributeError):
        return False


def get_donor_cost(graph_data, donor):
    """
    Стоимость donor из lol_unified.
    full donor -> 1
    knockout   -> knockout_cost из конфигурации (сейчас 10)
    """
    try:
        cost = float(graph_data.donor_cost[int(donor)])
        if cost <= 0:
            return 1.0
        return cost
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        return 1.0


def donor_key(graph_data, donor):
    """
    Глобальный ключ кандидата между разными scenario-графами.

    НЕЛЬЗЯ использовать числовой donor id:
    scenario2 фильтрует вход, а scenario3 добавляет knockout donors,
    поэтому локальные id между графами не совпадают.

    Один и тот же full donor сопоставляется по genotype.
    Knockout donor является отдельным кандидатом.
    """
    genotype = get_donor_genotype(graph_data, donor)
    is_knockout = get_donor_is_knockout(graph_data, donor)
    return ("knockout" if is_knockout else "full", genotype)


def _is_active_donor(graph_data, donor):
    try:
        return bool(graph_data.active[int(donor)])
    except (KeyError, IndexError, TypeError):
        return False


def denominator(z):
    """
    score contribution = effective_gain / (covered_weight + 1)

    +1 нужен для начального шага, когда covered_weight == 0.
    """
    return float(z) + 1.0


def get_gain_for_local_donor(graph_data, donor, patients, covered):
    """
    Реальный marginal coverage для локального donor id в одном scenario.

    Returns:
        raw_gain       = sum(num_occ[p]) новых patient vertices
        uncovered_pats = set новых patient vertex ids
    """
    if donor is None:
        return 0, set()

    if not _is_active_donor(graph_data, donor):
        return 0, set()

    neigh = graph_data.get_neighbors(int(donor))

    uncovered_pats = {
        int(p)
        for p in neigh
        if int(p) in patients and int(p) not in covered
    }

    raw_gain = sum(
        int(graph_data.num_occ[p])
        for p in uncovered_pats
    )

    return raw_gain, uncovered_pats



def donor_concentration_from_details(details):
    """
    H_i = sum_j (x_ij / sum_k x_ik)^2

    Here x_ij is the weighted marginal coverage (raw_gain)
    of one donor in scenario j.

    Returns:
        (H, total_x)
    If total_x == 0, H is None.
    """
    xs = [float(d["raw_gain"]) for d in details]
    total_x = sum(xs)

    if total_x <= 0:
        return None, 0.0

    h = sum((x / total_x) ** 2 for x in xs)
    return h, total_x



def compute_initial_multi_score(candidate_key, scenario_states):
    """
    Fast score for t=0 only.

    At t=0 nothing is covered yet, so every graph neighbor is new.
    Do not build uncovered sets and do not check membership in covered.
    The same raw gains are also returned for concentration H.
    """
    total_score = 0.0
    raw_gains = []

    for state in scenario_states:
        local_donor = state["donor_by_key"].get(candidate_key)

        if local_donor is None:
            raw_gains.append(0)
            continue

        graph_data = state["graph"]

        raw_gain = sum(
            int(graph_data.num_occ[p])
            for p in graph_data.get_neighbors(int(local_donor))
        )

        cost = get_donor_cost(graph_data, local_donor)
        effective_gain = raw_gain / cost

        # At t=0 covered_weight == 0, so denominator == 1.
        total_score += effective_gain
        raw_gains.append(raw_gain)

    return total_score, raw_gains


def donor_concentration_from_raw_gains(raw_gains):
    """
    H_i = sum_j (x_ij / sum_k x_ik)^2
    from already-computed t=0 raw gains.
    """
    total_x = float(sum(raw_gains))

    if total_x <= 0:
        return None, 0.0

    h = sum(
        (float(x) / total_x) ** 2
        for x in raw_gains
    )
    return h, total_x


def scenario_reached_target(state, coverage_percentage):
    """Return True when this scenario no longer needs to drive selection."""
    total_weight = state["total_weight"]
    if total_weight <= 0:
        return True
    return (
        100.0 * state["covered_weight"] / total_weight
        >= coverage_percentage
    )


def compute_multi_score(
    candidate_key,
    scenario_states,
    coverage_percentage,
):
    """
    Общий score глобального кандидата.

    Для каждого scenario:
        raw_gain = сколько новых реальных пациентов он покрывает
        cost     = donor_cost локальной donor-вершины
        effective_gain = raw_gain / cost
        contribution   = effective_gain / (covered_weight + 1)

    Once a scenario reaches coverage_percentage, its contribution to the
    selection score is zero. Its real marginal gain is still returned in
    details, because the globally selected donor is applied and logged in
    every scenario.

    Для full donor cost=1, поэтому старый score не меняется.
    Для knockout scenario3 cost=10, поэтому его вклад штрафуется в 10 раз.

    Returns:
        total_score
        details[j] = {
            donor_id,
            raw_gain,
            cost,
            effective_gain,
            new_patients
        }
    """
    total_score = 0.0
    details = []

    for state in scenario_states:
        local_donor = state["donor_by_key"].get(candidate_key)

        if local_donor is None:
            details.append(
                {
                    "donor_id": None,
                    "raw_gain": 0,
                    "cost": None,
                    "effective_gain": 0.0,
                    "new_patients": set(),
                }
            )
            continue

        graph_data = state["graph"]

        raw_gain, uncovered = get_gain_for_local_donor(
            graph_data=graph_data,
            donor=local_donor,
            patients=state["patients"],
            covered=state["covered"],
        )

        cost = get_donor_cost(graph_data, local_donor)
        effective_gain = raw_gain / cost

        if not scenario_reached_target(state, coverage_percentage):
            z = state["covered_weight"]
            total_score += effective_gain / denominator(z)

        details.append(
            {
                "donor_id": int(local_donor),
                "raw_gain": raw_gain,
                "cost": cost,
                "effective_gain": effective_gain,
                "new_patients": uncovered,
            }
        )

    return total_score, details


def multi_log_fieldnames(scenario_states):
    fields = [
        "step",
        "donor_key",
        "genotype",
        "is_knockout",
        "num_occ",
        "donor_cost",
        "score",
        "concentration_H",
        "sum_x",
    ]

    for state in scenario_states:
        name = state["name"]
        fields.extend(
            [
                f"{name}_donor_id",
                f"{name}_new_vertices",
                f"{name}_new_weight",
                f"{name}_donor_cost",
                f"{name}_effective_gain",
                f"{name}_covered_vertices",
                f"{name}_covered_weight",
                f"{name}_percentage",
            ]
        )

    return fields


def flatten_multi_log_row(row, scenario_states):
    flat = {
        "step": row["step"],
        "donor_key": row["donor_key"],
        "genotype": row["genotype"],
        "is_knockout": row["is_knockout"],
        "num_occ": row["num_occ"],
        "donor_cost": row["donor_cost"],
        "score": row["score"],
        "concentration_H": row["concentration_H"],
        "sum_x": row["sum_x"],
    }

    for state in scenario_states:
        name = state["name"]
        scenario = row["scenarios"][name]
        flat[f"{name}_donor_id"] = scenario["local_donor_id"]
        flat[f"{name}_new_vertices"] = scenario["new_vertices"]
        flat[f"{name}_new_weight"] = scenario["new_weight"]
        flat[f"{name}_donor_cost"] = scenario["donor_cost"]
        flat[f"{name}_effective_gain"] = scenario["effective_gain"]
        flat[f"{name}_covered_vertices"] = scenario["covered_vertices"]
        flat[f"{name}_covered_weight"] = scenario["covered_weight"]
        flat[f"{name}_percentage"] = scenario["percentage"]

    return flat


def build_scenario_graphs(
    base_conf,
    build_graph_fn,
    scenarios_file="scenarios.json",
    verbose=True,
):
    scenarios = load_scenarios(scenarios_file)

    graphs = []

    for j, (scenario_name, scenario_overrides) in enumerate(
        scenarios.items(), start=1
    ):
        conf = deepcopy(base_conf)
        conf.update(scenario_overrides)

        if verbose:
            print()
            print("=" * 78)
            print(
                f"[MULTI] building scenario {j}/{len(scenarios)}: "
                f"{scenario_name}"
            )
            print("=" * 78)

        t0 = time.time()
        graph_data = build_graph_fn(conf)
        elapsed = time.time() - t0

        graphs.append(
            {
                "name": scenario_name,
                "conf": conf,
                "graph": graph_data,
            }
        )

        if verbose:
            print(
                f"[MULTI] {scenario_name} built in {elapsed:.3f}s | "
                f"donors={len(graph_data.donors):,} | "
                f"patients={len(graph_data.patients):,}"
            )

    return graphs


def call_multi_scenario_greedy(
    graphs,
    coverage_percentage=99.0,
    max_donors=None,
    log_file="logs/multi_scenario_greedy.csv",
    verbose=False,
    progress_every=100,
):
    """
    Multi-scenario lazy greedy.

    Критическое отличие от ранней версии:
    donor IDs НЕ считаются общими между графами.

    Один глобальный кандидат определяется как:
        (full/knockout, genotype)

    Для каждого scenario хранится отображение:
        candidate_key -> local donor_id
    """
    if not graphs:
        raise ValueError("graphs must contain at least one scenario graph")

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    scenario_states = []

    for item in graphs:
        graph_data = item["graph"]
        patients = {int(p) for p in graph_data.patients}

        total_weight = sum(
            int(graph_data.num_occ[p])
            for p in patients
        )

        donor_by_key = {}

        for d in graph_data.donors:
            d = int(d)
            if not _is_active_donor(graph_data, d):
                continue

            key = donor_key(graph_data, d)

            if key in donor_by_key:
                raise ValueError(
                    f"Duplicate donor key in scenario {item['name']}: {key}"
                )

            donor_by_key[key] = d

        scenario_states.append(
            {
                "name": item["name"],
                "graph": graph_data,
                "patients": patients,
                "total_weight": total_weight,
                "covered": set(),
                "covered_weight": 0,
                "donor_by_key": donor_by_key,
            }
        )

    # Union по ГЛОБАЛЬНЫМ ключам, а не по локальным числовым id.
    all_candidates = set()

    for state in scenario_states:
        all_candidates.update(state["donor_by_key"].keys())

    print()
    print("=" * 78)
    print("MULTI-SCENARIO LAZY GREEDY")
    print("=" * 78)
    print(f"scenarios        : {len(scenario_states)}")
    print(f"unique candidates: {len(all_candidates):,}")

    for state in scenario_states:
        graph_data = state["graph"]

        n_ko = 0
        if hasattr(graph_data, "donor_is_knockout"):
            try:
                n_ko = sum(
                    bool(graph_data.donor_is_knockout[int(d)])
                    for d in graph_data.donors
                )
            except Exception:
                n_ko = 0

        print(
            f"  {state['name']}: "
            f"donors={len(graph_data.donors):,}, "
            f"knockout={n_ko:,}, "
            f"patients={len(state['patients']):,}, "
            f"total_weight={state['total_weight']:,}"
        )

    # Initial lazy heap.
    # t=0 has a dedicated fast path: all neighbors are uncovered,
    # so there is no reason to build sets or run the general lazy scorer.
    pq = []
    t0 = time.time()

    total_candidates = len(all_candidates)
    init_report_every = max(
        (progress_every or 100) * 100,
        10000,
    )

    print(
        f"[MULTI] initializing heap for "
        f"{total_candidates:,} candidates...",
        flush=True,
    )

    # Donor concentration at t=0.
    # Uses exactly the same x_ij values already computed for the heap score.
    concentration_t0 = {}

    # heap tuple:
    # (-score, kind, genotype, candidate_key)
    # Build a list first and heapify once at the end.
    for init_idx, key in enumerate(
        all_candidates,
        start=1,
    ):
        score, raw_gains = compute_initial_multi_score(
            candidate_key=key,
            scenario_states=scenario_states,
        )

        h0, sum_x0 = donor_concentration_from_raw_gains(
            raw_gains
        )

        if h0 is not None:
            concentration_t0[key] = {
                "H": h0,
                "sum_x": sum_x0,
                "x_by_scenario": {
                    state["name"]: raw_gains[j]
                    for j, state in enumerate(scenario_states)
                },
            }

        if score > 0:
            pq.append(
                (-score, key[0], key[1], key)
            )

        if (
            init_idx % init_report_every == 0
            or init_idx == total_candidates
        ):
            elapsed = time.time() - t0
            rate = (
                init_idx / elapsed
                if elapsed > 0
                else 0.0
            )
            remaining = total_candidates - init_idx
            eta = (
                remaining / rate
                if rate > 0
                else 0.0
            )

            print(
                f"[MULTI] heap init: "
                f"{init_idx:,}/{total_candidates:,} candidates "
                f"({100.0 * init_idx / total_candidates:.1f}%), "
                f"candidates={len(pq):,}, "
                f"elapsed={elapsed:.1f}s, "
                f"eta={eta:.1f}s",
                flush=True,
            )

    heapq.heapify(pq)

    print(
        f"[MULTI] heap ready: "
        f"{len(pq):,} active candidates, "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    selected_keys = set()
    step_logs = []
    step = 0

    # Create the checkpoint CSV before the first greedy step. Each accepted
    # donor is written and flushed immediately, so a long run remains usable
    # even if it is interrupted before the requested coverage is reached.
    log_handle = open(log_file, "w", newline="")
    log_writer = csv.DictWriter(
        log_handle,
        fieldnames=multi_log_fieldnames(scenario_states),
    )
    log_writer.writeheader()
    log_handle.flush()

    while pq:
        if max_donors is not None and step >= max_donors:
            break

        all_reached = True

        for state in scenario_states:
            if state["total_weight"] == 0:
                continue

            pct = (
                100.0
                * state["covered_weight"]
                / state["total_weight"]
            )

            if pct < coverage_percentage:
                all_reached = False
                break

        if all_reached:
            break

        neg_old_score, _, _, key = heapq.heappop(pq)

        if key in selected_keys:
            continue

        actual_score, details = compute_multi_score(
            candidate_key=key,
            scenario_states=scenario_states,
            coverage_percentage=coverage_percentage,
        )

        if actual_score <= 0:
            continue

        if pq and actual_score < -pq[0][0]:
            heapq.heappush(
                pq,
                (-actual_score, key[0], key[1], key),
            )
            continue

        step += 1
        selected_keys.add(key)

        kind, genotype = key
        is_knockout = kind == "knockout"

        # num_occ/cost берем из первого scenario, где этот candidate реально есть.
        donor_num_occ = None
        donor_cost = None

        for j, state in enumerate(scenario_states):
            local_id = details[j]["donor_id"]
            if local_id is not None:
                donor_num_occ = get_donor_num_occ(
                    state["graph"], local_id
                )
                donor_cost = get_donor_cost(
                    state["graph"], local_id
                )
                break

        scenario_log = {}

        for j, state in enumerate(scenario_states):
            graph_data = state["graph"]
            detail = details[j]

            local_id = detail["donor_id"]
            newly_covered = detail["new_patients"]
            raw_gain = detail["raw_gain"]
            cost = detail["cost"]
            effective_gain = detail["effective_gain"]

            state["covered"].update(newly_covered)
            state["covered_weight"] += raw_gain

            # Deactivate the local donor selected for this global key.
            if local_id is not None and _is_active_donor(
                graph_data, local_id
            ):
                graph_data.deactivate(local_id)

            if state["total_weight"] > 0:
                percentage = (
                    100.0
                    * state["covered_weight"]
                    / state["total_weight"]
                )
            else:
                percentage = 100.0

            scenario_log[state["name"]] = {
                "local_donor_id": local_id,
                "new_vertices": len(newly_covered),
                "new_weight": raw_gain,
                "donor_cost": cost,
                "effective_gain": effective_gain,
                "covered_vertices": len(state["covered"]),
                "covered_weight": state["covered_weight"],
                "percentage": percentage,
            }

        h_selected, sum_x_selected = donor_concentration_from_details(details)

        row = {
            "step": step,
            "donor_key": f"{kind}:{genotype}",
            "genotype": genotype,
            "is_knockout": is_knockout,
            "num_occ": donor_num_occ,
            "donor_cost": donor_cost,
            "score": actual_score,
            "concentration_H": h_selected,
            "sum_x": sum_x_selected,
            "scenarios": scenario_log,
        }

        step_logs.append(row)
        log_writer.writerow(
            flatten_multi_log_row(row, scenario_states)
        )
        log_handle.flush()

        if verbose:
            print(
                f"[MULTI GREEDY] step={step}, "
                f"type={kind}, num_occ={donor_num_occ}, "
                f"cost={donor_cost}, score={actual_score:.10f}"
            )
            for state in scenario_states:
                s = scenario_log[state["name"]]
                print(
                    f"    {state['name']}: "
                    f"id={s['local_donor_id']}, "
                    f"new_vertices={s['new_vertices']:,}, "
                    f"new_weight={s['new_weight']:,}, "
                    f"effective_gain={s['effective_gain']:.3f}, "
                    f"pct={s['percentage']:.4f}%"
                )
        elif progress_every and step % progress_every == 0:
            parts = []
            for state in scenario_states:
                s = scenario_log[state["name"]]
                parts.append(
                    f"{state['name']}={s['percentage']:.4f}%"
                )
            print(
                f"[MULTI] step={step:,} | " + " | ".join(parts)
            )

    log_handle.close()

    final_scenarios = {}

    for state in scenario_states:
        total_weight = state["total_weight"]

        percentage = (
            100.0 * state["covered_weight"] / total_weight
            if total_weight > 0
            else 100.0
        )

        final_scenarios[state["name"]] = {
            "covered_weight": state["covered_weight"],
            "total_weight": total_weight,
            "covered_vertices": len(state["covered"]),
            "percentage": percentage,
        }

    return {
        "num_selected_donors": step,
        "selected_donors": [
            row["donor_key"]
            for row in step_logs
        ],
        "steps": [
            {
                "step": row["step"],
                "donor": row["donor_key"],
                "score": row["score"],
            }
            for row in step_logs
        ],
        "final_step": step,
        "scenarios": final_scenarios,
        "log": step_logs,
        "concentration_t0": concentration_t0,
        "scenario_names": [
            state["name"]
            for state in scenario_states
        ],
        "log_file": log_file,
    }


def build_and_run_multi_greedy(
    base_conf,
    build_graph_fn,
    scenarios_file="scenarios.json",
    coverage_percentage=99.0,
    max_donors=None,
    log_file="logs/multi_scenario_greedy.csv",
    verbose=True,
):
    graphs = build_scenario_graphs(
        base_conf=base_conf,
        build_graph_fn=build_graph_fn,
        scenarios_file=scenarios_file,
        verbose=verbose,
    )

    return call_multi_scenario_greedy(
        graphs=graphs,
        coverage_percentage=coverage_percentage,
        max_donors=max_donors,
        log_file=log_file,
        verbose=verbose,
    )
