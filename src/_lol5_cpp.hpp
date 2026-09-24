#pragma once
#include <vector>
#include <cstdint>
#include <omp.h>

struct EdgeResult {
    std::vector<int> donors;
    std::vector<int> patients;
};

inline EdgeResult match_subgraphs_cpp(
    const int* c1_donors, const int* c2_donors, int D,
    const int* c1_patients, const int* c2_patients, int P,
    int total_alleles,
    int limit
) {
    // 1. Индексация доноров по ВСЕМ локусам: A, B, C, DQB1, DRB1
    std::vector<std::vector<int>> hash_A(total_alleles);
    std::vector<std::vector<int>> hash_B(total_alleles);
    std::vector<std::vector<int>> hash_C(total_alleles);
    std::vector<std::vector<int>> hash_DQ(total_alleles);
    std::vector<std::vector<int>> hash_DR(total_alleles);

    for (int d_id = 0; d_id < D; ++d_id) {
        int idx_c1 = d_id * 6;
        int idx_c2 = d_id * 4;

        int a0 = c1_donors[idx_c1];
        int a1 = c1_donors[idx_c1 + 1];
        int b0 = c1_donors[idx_c1 + 2];
        int b1 = c1_donors[idx_c1 + 3];
        int c0 = c1_donors[idx_c1 + 4];
        int c1 = c1_donors[idx_c1 + 5];

        int dq0 = c2_donors[idx_c2];
        int dq1 = c2_donors[idx_c2 + 1];
        int dr0 = c2_donors[idx_c2 + 2];
        int dr1 = c2_donors[idx_c2 + 3];

        // A
        hash_A[a0].push_back(d_id);
        if (a1 != a0) hash_A[a1].push_back(d_id);
        // B
        hash_B[b0].push_back(d_id);
        if (b1 != b0) hash_B[b1].push_back(d_id);
        // C
        hash_C[c0].push_back(d_id);
        if (c1 != c0) hash_C[c1].push_back(d_id);
        // DQB1
        hash_DQ[dq0].push_back(d_id);
        if (dq1 != dq0) hash_DQ[dq1].push_back(d_id);
        // DRB1
        hash_DR[dr0].push_back(d_id);
        if (dr1 != dr0) hash_DR[dr1].push_back(d_id);
    }

    int max_threads = omp_get_max_threads();
    std::vector<EdgeResult> thread_results(max_threads);

    for (int t = 0; t < max_threads; ++t) {
        thread_results[t].donors.reserve(15000000 / max_threads);
        thread_results[t].patients.reserve(15000000 / max_threads);
    }

    // 2. Параллельный проход по пациентам
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& local_res = thread_results[thread_id];

        // epoch-метка, чтобы безопасно отслеживать "уже проверенных" доноров
        std::vector<uint64_t> local_seen_donor(D, 0);
        uint64_t base_epoch = ((uint64_t)thread_id + 1) << 32; // +1 чтобы не было нуля

        #pragma omp for schedule(dynamic, 64)
        for (int p_id = 0; p_id < P; ++p_id) {
            uint64_t epoch = base_epoch | (uint64_t)p_id;

            int idx_c1 = p_id * 6;
            int idx_c2 = p_id * 4;

            int pa0 = c1_patients[idx_c1];
            int pa1 = c1_patients[idx_c1 + 1];
            int pb0 = c1_patients[idx_c1 + 2];
            int pb1 = c1_patients[idx_c1 + 3];
            int pc0 = c1_patients[idx_c1 + 4];
            int pc1 = c1_patients[idx_c1 + 5];

            int pq0 = c2_patients[idx_c2];
            int pq1 = c2_patients[idx_c2 + 1];
            int pr0 = c2_patients[idx_c2 + 2];
            int pr1 = c2_patients[idx_c2 + 3];

            auto check_and_add_donor = [&](int d_id) {
                if (local_seen_donor[d_id] == epoch) return;
                local_seen_donor[d_id] = epoch;

                int d_idx_c1 = d_id * 6;
                int d_idx_c2 = d_id * 4;
                int mismatch = 0;
                int d0, d1;

                // A
                d0 = c1_donors[d_idx_c1];
                d1 = c1_donors[d_idx_c1 + 1];
                if (d0 != pa0 && d0 != pa1) mismatch++;
                if (d1 != d0 && d1 != pa0 && d1 != pa1) mismatch++;

                // B
                if (mismatch <= limit) {
                    d0 = c1_donors[d_idx_c1 + 2];
                    d1 = c1_donors[d_idx_c1 + 3];
                    if (d0 != pb0 && d0 != pb1) mismatch++;
                    if (d1 != d0 && d1 != pb0 && d1 != pb1) mismatch++;
                }

                // C
                if (mismatch <= limit) {
                    d0 = c1_donors[d_idx_c1 + 4];
                    d1 = c1_donors[d_idx_c1 + 5];
                    if (d0 != pc0 && d0 != pc1) mismatch++;
                    if (d1 != d0 && d1 != pc0 && d1 != pc1) mismatch++;
                }

                // DQB1
                if (mismatch <= limit) {
                    d0 = c2_donors[d_idx_c2];
                    d1 = c2_donors[d_idx_c2 + 1];
                    if (d0 != pq0 && d0 != pq1) mismatch++;
                    if (d1 != d0 && d1 != pq0 && d1 != pq1) mismatch++;
                }

                // DRB1
                if (mismatch <= limit) {
                    d0 = c2_donors[d_idx_c2 + 2];
                    d1 = c2_donors[d_idx_c2 + 3];
                    if (d0 != pr0 && d0 != pr1) mismatch++;
                    if (d1 != d0 && d1 != pr0 && d1 != pr1) mismatch++;
                }

                if (mismatch <= limit) {
                    local_res.donors.push_back(d_id);
                    local_res.patients.push_back(p_id);
                }
            };

            // 3. Собираем кандидатов по ВСЕМ локусам

            // A
            for (int d_id : hash_A[pa0]) check_and_add_donor(d_id);
            if (pa1 != pa0)
                for (int d_id : hash_A[pa1]) check_and_add_donor(d_id);

            // B
            for (int d_id : hash_B[pb0]) check_and_add_donor(d_id);
            if (pb1 != pb0)
                for (int d_id : hash_B[pb1]) check_and_add_donor(d_id);

            // C
            for (int d_id : hash_C[pc0]) check_and_add_donor(d_id);
            if (pc1 != pc0)
                for (int d_id : hash_C[pc1]) check_and_add_donor(d_id);

            // DQB1
            for (int d_id : hash_DQ[pq0]) check_and_add_donor(d_id);
            if (pq1 != pq0)
                for (int d_id : hash_DQ[pq1]) check_and_add_donor(d_id);

            // DRB1
            for (int d_id : hash_DR[pr0]) check_and_add_donor(d_id);
            if (pr1 != pr0)
                for (int d_id : hash_DR[pr1]) check_and_add_donor(d_id);
        }
    }

    // 4. Сборка финального результата
    EdgeResult final_result;
    size_t total_edges = 0;
    for (int t = 0; t < max_threads; ++t) {
        total_edges += thread_results[t].donors.size();
    }
    final_result.donors.reserve(total_edges);
    final_result.patients.reserve(total_edges);

    for (int t = 0; t < max_threads; ++t) {
        final_result.donors.insert(final_result.donors.end(),
                                   thread_results[t].donors.begin(),
                                   thread_results[t].donors.end());
        final_result.patients.insert(final_result.patients.end(),
                                     thread_results[t].patients.begin(),
                                     thread_results[t].patients.end());
    }

    return final_result;
}
