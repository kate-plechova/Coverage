# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdint cimport int64_t

cdef extern from "_lol_unified_cpp.hpp":
    struct UnifiedEdgeResult:
        vector[int] donors
        vector[int] patients

    UnifiedEdgeResult match_per_class_cpp(
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
    ) except +


def match_per_class_cython(
    int[:, ::1] donor_c1,
    int[::1] donor_c1_len,
    int[:, ::1] donor_c2,
    int[::1] donor_c2_len,
    int[:, ::1] donor_full,
    int[::1] donor_full_len,
    int[::1] donor_kir,

    int[:, ::1] patient_c1,
    int[::1] patient_c1_len,
    int[:, ::1] patient_c2,
    int[::1] patient_c2_len,
    int[:, ::1] patient_full,
    int[::1] patient_full_len,
    int[::1] patient_kir,

    int use_class1,
    int use_class2,
    int mismatch_c1,
    int mismatch_c2,
    int mismatch_pat_c1,
    int mismatch_pat_c2,
    int check_homozygot,
    int use_kir,
):
    cdef int D = donor_c1.shape[0]
    cdef int P = patient_c1.shape[0]

    if D == 0 or P == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    cdef UnifiedEdgeResult cpp_res = match_per_class_cpp(
        &donor_c1[0, 0], &donor_c1_len[0],
        &donor_c2[0, 0], &donor_c2_len[0],
        &donor_full[0, 0], &donor_full_len[0],
        &donor_kir[0],
        D,
        donor_c1.shape[1], donor_c2.shape[1], donor_full.shape[1],

        &patient_c1[0, 0], &patient_c1_len[0],
        &patient_c2[0, 0], &patient_c2_len[0],
        &patient_full[0, 0], &patient_full_len[0],
        &patient_kir[0],
        P,
        patient_c1.shape[1], patient_c2.shape[1], patient_full.shape[1],

        use_class1,
        use_class2,
        mismatch_c1,
        mismatch_c2,
        mismatch_pat_c1,
        mismatch_pat_c2,
        check_homozygot,
        use_kir,
    )

    cdef size_t n_edges = cpp_res.donors.size()
    res_d = np.empty(n_edges, dtype=np.int64)
    res_p = np.empty(n_edges, dtype=np.int64)

    cdef size_t i
    cdef int64_t[::1] view_d = res_d
    cdef int64_t[::1] view_p = res_p

    for i in range(n_edges):
        view_d[i] = cpp_res.donors[i]
        view_p[i] = cpp_res.patients[i]

    return res_d, res_p
