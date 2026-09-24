#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include <omp.h>

struct UnifiedEdgeResult {
    std::vector<int> donors;
    std::vector<int> patients;
};

inline int overlap_sorted(
    const int* a, int a_len,
    const int* b, int b_len
) {
    int i = 0;
    int j = 0;
    int overlap = 0;

    while (i < a_len && j < b_len) {
        const int av = a[i];
        const int bv = b[j];
        if (av == bv) {
            ++overlap;
            ++i;
            ++j;
        } else if (av < bv) {
            ++i;
        } else {
            ++j;
        }
    }
    return overlap;
}

inline bool class_matches_sorted(
    const int* donor, int donor_len,
    const int* patient, int patient_len,
    int donor_miss, int patient_miss
) {
    if (donor_len == 0 && patient_len == 0) return true;

    const int donor_required = std::max(0, donor_len - donor_miss);
    const int patient_required = std::max(0, patient_len - patient_miss);
    const int required = std::max(donor_required, patient_required);

    if (required == 0) return true;
    if (required > donor_len || required > patient_len) return false;

    return overlap_sorted(donor, donor_len, patient, patient_len) >= required;
}

inline bool contains_sorted(const int* values, int n, int target) {
    return std::binary_search(values, values + n, target);
}

inline bool homozygote_ok_sorted(
    const int* donor_full, int donor_len,
    const int* patient_full, int patient_len
) {
    const int even_len = donor_len - (donor_len % 2);
    for (int i = 0; i < even_len; i += 2) {
        if (donor_full[i] == donor_full[i + 1] &&
            contains_sorted(patient_full, patient_len, donor_full[i])) {
            return false;
        }
    }
    return true;
}

inline int max_allele_id_side(
    const int* values, const int* lengths,
    int n, int stride
) {
    int mx = -1;
    for (int i = 0; i < n; ++i) {
        const int* row = values + i * stride;
        for (int j = 0; j < lengths[i]; ++j) {
            if (row[j] > mx) mx = row[j];
        }
    }
    return mx;
}

inline void build_donor_allele_index(
    const int* values, const int* lengths,
    int D, int stride,
    std::vector<std::vector<int>>& index
) {
    for (int d_id = 0; d_id < D; ++d_id) {
        const int* row = values + d_id * stride;
        int prev = std::numeric_limits<int>::min();
        for (int j = 0; j < lengths[d_id]; ++j) {
            const int allele = row[j];
            if (j > 0 && allele == prev) continue;  // homozygous duplicate
            prev = allele;
            if (allele >= 0 && allele < static_cast<int>(index.size())) {
                index[allele].push_back(d_id);
            }
        }
    }
}

inline int min_required_from_donors(
    const int* donor_len,
    int D,
    int donor_miss
) {
    if (D == 0) return 0;
    int min_required = std::numeric_limits<int>::max();
    for (int d_id = 0; d_id < D; ++d_id) {
        const int required = std::max(0, donor_len[d_id] - donor_miss);
        if (required < min_required) min_required = required;
    }
    return min_required == std::numeric_limits<int>::max() ? 0 : min_required;
}

inline size_t estimate_candidate_volume(
    const int* patient, int patient_len,
    const std::vector<std::vector<int>>& index
) {
    size_t total = 0;
    int prev = std::numeric_limits<int>::min();
    for (int j = 0; j < patient_len; ++j) {
        const int allele = patient[j];
        if (j > 0 && allele == prev) continue;
        prev = allele;
        if (allele >= 0 && allele < static_cast<int>(index.size())) {
            total += index[allele].size();
        }
    }
    return total;
}

inline void gather_candidates(
    const int* patient, int patient_len,
    const std::vector<std::vector<int>>& index,
    std::vector<int>& candidates,
    std::vector<uint32_t>& marks,
    uint32_t token
) {
    int prev = std::numeric_limits<int>::min();
    for (int j = 0; j < patient_len; ++j) {
        const int allele = patient[j];
        if (j > 0 && allele == prev) continue;
        prev = allele;
        if (allele < 0 || allele >= static_cast<int>(index.size())) continue;

        const auto& bucket = index[allele];
        for (int d_id : bucket) {
            if (marks[d_id] != token) {
                marks[d_id] = token;
                candidates.push_back(d_id);
            }
        }
    }
}

inline UnifiedEdgeResult match_per_class_cpp(
    const int* donor_c1, const int* donor_c1_len,
    const int* donor_c2, const int* donor_c2_len,
    const int* donor_full, const int* donor_full_len,
    const int* donor_kir,
    int D,
    int donor_c1_stride, int donor_c2_stride, int donor_full_stride,

    const int* patient_c1, const int* patient_c1_len,
    const int* patient_c2, const int* patient_c2_len,
    const int* patient_full, const int* patient_full_len,
    const int* patient_kir,
    int P,
    int patient_c1_stride, int patient_c2_stride, int patient_full_stride,

    int use_class1,
    int use_class2,
    int mismatch_c1,
    int mismatch_c2,
    int mismatch_pat_c1,
    int mismatch_pat_c2,
    int check_homozygot,
    int use_kir
) {
    // Same candidate-generation idea as lol5 total:
    // allele -> donor IDs, then exact per-class validation.
    int max_allele = -1;
    if (use_class1) {
        max_allele = std::max(max_allele,
            max_allele_id_side(donor_c1, donor_c1_len, D, donor_c1_stride));
        max_allele = std::max(max_allele,
            max_allele_id_side(patient_c1, patient_c1_len, P, patient_c1_stride));
    }
    if (use_class2) {
        max_allele = std::max(max_allele,
            max_allele_id_side(donor_c2, donor_c2_len, D, donor_c2_stride));
        max_allele = std::max(max_allele,
            max_allele_id_side(patient_c2, patient_c2_len, P, patient_c2_stride));
    }

    std::vector<std::vector<int>> index_c1;
    std::vector<std::vector<int>> index_c2;
    if (use_class1 && max_allele >= 0) {
        index_c1.resize(static_cast<size_t>(max_allele) + 1);
        build_donor_allele_index(
            donor_c1, donor_c1_len, D, donor_c1_stride, index_c1
        );
    }
    if (use_class2 && max_allele >= 0) {
        index_c2.resize(static_cast<size_t>(max_allele) + 1);
        build_donor_allele_index(
            donor_c2, donor_c2_len, D, donor_c2_stride, index_c2
        );
    }

    const int min_donor_req_c1 = use_class1
        ? min_required_from_donors(donor_c1_len, D, mismatch_c1)
        : 0;
    const int min_donor_req_c2 = use_class2
        ? min_required_from_donors(donor_c2_len, D, mismatch_c2)
        : 0;

    const int max_threads = omp_get_max_threads();
    std::vector<UnifiedEdgeResult> thread_results(max_threads);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& local = thread_results[tid];
        std::vector<int> candidates;
        std::vector<uint32_t> marks(static_cast<size_t>(D), 0);
        uint32_t token = 0;

        #pragma omp for schedule(dynamic, 16)
        for (int p_id = 0; p_id < P; ++p_id) {
            const int* p_c1 = patient_c1 + p_id * patient_c1_stride;
            const int* p_c2 = patient_c2 + p_id * patient_c2_stride;
            const int* p_full = patient_full + p_id * patient_full_stride;
            const int p_c1_n = patient_c1_len[p_id];
            const int p_c2_n = patient_c2_len[p_id];
            const int p_full_n = patient_full_len[p_id];

            const int pat_req_c1 = use_class1
                ? std::max(0, p_c1_n - mismatch_pat_c1)
                : 0;
            const int pat_req_c2 = use_class2
                ? std::max(0, p_c2_n - mismatch_pat_c2)
                : 0;

            // A class is safe for candidate generation iff every valid pair
            // must have at least one shared allele in that class.
            const bool c1_positive = use_class1 &&
                std::max(min_donor_req_c1, pat_req_c1) > 0;
            const bool c2_positive = use_class2 &&
                std::max(min_donor_req_c2, pat_req_c2) > 0;

            int candidate_class = 0;  // 0 => full scan, 1 => C1 index, 2 => C2 index
            if (c1_positive && c2_positive) {
                const size_t est1 = estimate_candidate_volume(p_c1, p_c1_n, index_c1);
                const size_t est2 = estimate_candidate_volume(p_c2, p_c2_n, index_c2);
                candidate_class = (est1 <= est2) ? 1 : 2;
            } else if (c1_positive) {
                candidate_class = 1;
            } else if (c2_positive) {
                candidate_class = 2;
            }

            auto pair_matches = [&](int d_id) -> bool {
                const int* d_c1 = donor_c1 + d_id * donor_c1_stride;
                const int* d_c2 = donor_c2 + d_id * donor_c2_stride;
                const int* d_full = donor_full + d_id * donor_full_stride;

                if (use_class1 && !class_matches_sorted(
                    d_c1, donor_c1_len[d_id],
                    p_c1, p_c1_n,
                    mismatch_c1, mismatch_pat_c1
                )) return false;

                if (use_class2 && !class_matches_sorted(
                    d_c2, donor_c2_len[d_id],
                    p_c2, p_c2_n,
                    mismatch_c2, mismatch_pat_c2
                )) return false;

                if (check_homozygot && !homozygote_ok_sorted(
                    d_full, donor_full_len[d_id],
                    p_full, p_full_n
                )) return false;

                // KIR is enforced before the edge is materialized.
                if (use_kir && donor_kir[d_id] == patient_kir[p_id]) {
                    return false;
                }

                return true;
            };

            if (candidate_class == 0) {
                // Correctness fallback for configurations where zero-overlap
                // matches are legal in every active class.
                for (int d_id = 0; d_id < D; ++d_id) {
                    if (pair_matches(d_id)) {
                        local.donors.push_back(d_id);
                        local.patients.push_back(p_id);
                    }
                }
                continue;
            }

            ++token;
            if (token == 0) {
                std::fill(marks.begin(), marks.end(), 0);
                token = 1;
            }
            candidates.clear();

            if (candidate_class == 1) {
                gather_candidates(p_c1, p_c1_n, index_c1, candidates, marks, token);
            } else {
                gather_candidates(p_c2, p_c2_n, index_c2, candidates, marks, token);
            }

            for (int d_id : candidates) {
                if (pair_matches(d_id)) {
                    local.donors.push_back(d_id);
                    local.patients.push_back(p_id);
                }
            }
        }
    }

    UnifiedEdgeResult result;
    size_t total_edges = 0;
    for (const auto& r : thread_results) total_edges += r.donors.size();
    result.donors.reserve(total_edges);
    result.patients.reserve(total_edges);

    for (auto& r : thread_results) {
        result.donors.insert(result.donors.end(), r.donors.begin(), r.donors.end());
        result.patients.insert(result.patients.end(), r.patients.begin(), r.patients.end());
    }

    return result;
}
