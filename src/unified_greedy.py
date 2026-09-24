import csv
import heapq
import os

def get_result_path(
    conf,
    method="greedy",
    base_dir="results_csv",
    coverage_percentage=None,
):
    scenario = conf.get("scenario")
    dataset = conf.get("result_dataset", "main")

    dataset_dir = os.path.join(
        base_dir,
        dataset,
        method,
    )

    cov_suffix = ""
    if coverage_percentage is not None:
        cov_suffix = f"_cov{coverage_percentage:g}"

    if scenario in ("scenario1", "scenario2"):
        os.makedirs(dataset_dir, exist_ok=True)

        return os.path.join(
            dataset_dir,
            f"{scenario}{cov_suffix}.csv",
        )

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

        os.makedirs(out_dir, exist_ok=True)

        return os.path.join(
            out_dir,
            f"mismatch_{mismatch_limit}{cov_suffix}.csv",
        )

    raise ValueError(
        f"Unknown scenario={scenario!r}"
    )
    
    
def call_greedy_from_graph(
    graph_data, g_m, coverage_percentage, conf, verbose=False
):
    mismatch_limit = conf.get("total_mismatch_limit", 0)
    direction = conf.get("mismatch_direction", "donor_to_patient")

    donors_file = conf.get("donors_input_file", "unknown")
    donors_filename = os.path.splitext(os.path.basename(donors_file))[0]

    log_file = conf.get("result_csv_file")

    if not log_file:
        log_file = get_result_path(
            conf,
            method="greedy",
            base_dir="results_csv",
            coverage_percentage=coverage_percentage,
        )

    os.makedirs(
        os.path.dirname(log_file) or ".",
        exist_ok=True,
    )

    patients = set(graph_data.patients)
    total_weight = sum(graph_data.num_occ[p] for p in patients)

    covered = set()
    covered_weight = 0

    steps = []
    step_logs = []
    step = 0

    # 1. Первоначальный подсчет прироста для всех активных доноров
    pq = []
    for d in graph_data.donors:
        if graph_data.active[d]:
            neigh = graph_data.get_neighbors(d)
            valid_pats = {p for p in neigh if p in patients}
            w = sum(graph_data.num_occ[p] for p in valid_pats)
            if w > 0:
                pq.append((-w, d, valid_pats))

    heapq.heapify(pq)

    # 2. Ленивый цикл
    while pq and (covered_weight / total_weight * 100.0 < coverage_percentage):
        neg_w, d, valid_pats = heapq.heappop(pq)

        # Вычисляем НАСТОЯЩИЙ прирост только для текущего кандидата
        uncovered_pats = valid_pats - covered
        actual_w = sum(graph_data.num_occ[p] for p in uncovered_pats)

        if actual_w == 0:
            continue

        # Проверяем: стал ли он хуже, чем следующий лучший элемент в куче?
        if pq and actual_w < -pq[0][0]:
            heapq.heappush(pq, (-actual_w, d, uncovered_pats))
            continue

        # Фиксируем выбор
        covered_weight += actual_w
        covered.update(uncovered_pats)
        graph_data.deactivate(d)

        step += 1
        percentage = 100.0 * covered_weight / total_weight

        # Получаем генотип донора (из id_to_geno, donor_genotypes или берем сам d)
        # Безопасное получение генотипа (работает и со списками, и со словарями)
        if hasattr(graph_data, "id_to_geno") and graph_data.id_to_geno is not None:
            genotype = graph_data.id_to_geno[d]
        elif hasattr(graph_data, "donor_genotypes") and graph_data.donor_genotypes is not None:
            genotype = graph_data.donor_genotypes[d]
        else:
            genotype = d

        steps.append({"step": step, "percentage": percentage})

        step_logs.append({
            "step": step,
            "donor": d,
            "genotype": genotype,
            "covered_patients": len(covered),
            "covered_weight": covered_weight,
            "percentage": percentage,
        })

        if verbose:
            print(
                f"[LAZY GREEDY] step={step}, donor={d}, gain={actual_w}, pct={percentage:.4f}%"
            )

    # Сохранение лога без колонок direction и mismatch_limit
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
            
    
    return {
        "num_selected_donors": step,
        "steps": steps,
        "final_step": step,
        "final_percentage": percentage if "percentage" in locals() else 0.0,
        "log": step_logs,
        "log_file": log_file,
    }