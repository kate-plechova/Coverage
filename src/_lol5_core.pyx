# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdint cimport int64_t

cdef extern from "_lol5_cpp.hpp":
    struct EdgeResult:
        vector[int] donors
        vector[int] patients
    
    EdgeResult match_subgraphs_cpp(
        const int* c1_donors, const int* c2_donors, int D,
        const int* c1_patients, const int* c2_patients, int P,
        int total_alleles,
        int limit
    ) except +

def match_subgraphs_cython(
    int[:, :] c1_donors, int[:, :] c2_donors,
    int[:, :] c1_patients, int[:, :] c2_patients,
    int total_alleles, int limit
):
    cdef int D = c1_donors.shape[0]
    cdef int P = c1_patients.shape[0]
    
    cdef EdgeResult cpp_res = match_subgraphs_cpp(
        &c1_donors[0, 0], &c2_donors[0, 0], D,
        &c1_patients[0, 0], &c2_patients[0, 0], P,
        total_alleles,
        limit
    )
    
    # ВАЖНО: эта строка должна быть ПОСЛЕ объявления cpp_res
    cdef size_t n_edges = cpp_res.donors.size()

    res_d = np.empty(n_edges, dtype=np.int64)
    res_p = np.empty(n_edges, dtype=np.int64)

    cdef size_t i
    cdef int64_t[:] view_d = res_d
    cdef int64_t[:] view_p = res_p

    for i in range(n_edges):
        view_d[i] = cpp_res.donors[i]
        view_p[i] = cpp_res.patients[i]
    return res_d, res_p
