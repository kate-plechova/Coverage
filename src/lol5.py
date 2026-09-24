import numpy as np
from collections import Counter


def parse_genotype(geno_str):
    geno_str = (
        geno_str
        .replace("\ufeff", "")
        .replace(" ", "")
        .replace("\t", "")
        .strip()
    )

    loci_raw = geno_str.split("^")
    loci = []

    for locus in loci_raw:
        a = locus.split("+")
        if len(a) != 2:
            raise ValueError(f"Expected 2 alleles, got: {locus}")
        loci.append((a[0], a[1]))

    return tuple(loci)


def flatten(geno):
    return [allele for locus in geno for allele in locus]


class LolGraph:
    def __init__(
        self,
        index,
        neighbors,
        id_to_geno,
        donors,
        patients,
        num_occ
    ):
        self.index = index
        self.neighbors = neighbors
        self.id_to_geno = id_to_geno

        self.donors = np.array(list(donors), dtype=np.int64)
        self.patients = np.array(list(patients), dtype=np.int64)

        N = len(index) - 1
        self.node_name = np.array(
            [f"node-{i}" for i in range(N)],
            dtype=object
        )

        self.num_occ = np.array(num_occ, dtype=np.int32)
        self.active = np.ones(N, dtype=bool)

    def get_neighbors(self, node):
        return self.neighbors[
            self.index[node]:self.index[node + 1]
        ]

    def num_nodes(self):
        return len(self.node_name)

    # ============================================================
    # Совместимость со старым greedy/random API
    # ============================================================

    def num_occurrence(self, node):
        return self.num_occ[node]

    def is_active(self, node):
        return self.active[node]

    def deactivate(self, node):
        self.active[node] = False

    def remove_node_csr(self, node):
        """
        Старый greedy ожидает, что удаление узла просто делает его
        неактивным.

        В CSR-графе мы НЕ удаляем рёбра физически — это дорого.
        Мы просто помечаем узел как неактивный.
        """
        self.active[node] = False


class LolGraphBuilderV5_GraphMatch:
    def __init__(
        self,
        mismatch_c1=0,
        mismatch_c2=0,
        total_mismatch_limit=0,
        mismatch_direction="donor_to_patient"
    ):
        self.limit = total_mismatch_limit
        self.mismatch_direction = mismatch_direction

    def build_graph(self, donors_input, patients_input):
        print(
            ">>> USING CYTHON SUBGRAPH CORE V5 "
            "(AGGREGATED + NUM_OCC) <<<"
        )

        donors_raw = [
            line.strip().split(",")[1]
            for line in open(donors_input)
        ]

        patients_raw = [
            line.strip().split(",")[1]
            for line in open(patients_input)
        ]

        donors_geno = [
            parse_genotype(g)
            for g in donors_raw
        ]

        patients_geno = [
            parse_genotype(g)
            for g in patients_raw
        ]

        donor_counts = Counter(donors_geno)
        patient_counts = Counter(patients_geno)

        unique_donors = list(donor_counts.keys())
        unique_patients = list(patient_counts.keys())

        D = len(unique_donors)
        P = len(unique_patients)

        # ============================================================
        # Allele IDs
        # ============================================================

        all_alleles = set()

        for g in unique_donors + unique_patients:
            all_alleles.update(flatten(g))

        allele_to_id = {
            a: i
            for i, a in enumerate(sorted(all_alleles))
        }

        # Allele IDs остаются int32.
        # Это НЕ global node IDs.
        c1_donors = np.zeros((D, 6), dtype=np.int32)
        c2_donors = np.zeros((D, 4), dtype=np.int32)

        c1_patients = np.zeros((P, 6), dtype=np.int32)
        c2_patients = np.zeros((P, 4), dtype=np.int32)

        # ============================================================
        # Encode donors
        # ============================================================

        for i, g in enumerate(unique_donors):
            c1_donors[i] = [
                allele_to_id[g[0][0]],
                allele_to_id[g[0][1]],
                allele_to_id[g[1][0]],
                allele_to_id[g[1][1]],
                allele_to_id[g[2][0]],
                allele_to_id[g[2][1]],
            ]

            c2_donors[i] = [
                allele_to_id[g[3][0]],
                allele_to_id[g[3][1]],
                allele_to_id[g[4][0]],
                allele_to_id[g[4][1]],
            ]

        # ============================================================
        # Encode patients
        # ============================================================

        for i, g in enumerate(unique_patients):
            c1_patients[i] = [
                allele_to_id[g[0][0]],
                allele_to_id[g[0][1]],
                allele_to_id[g[1][0]],
                allele_to_id[g[1][1]],
                allele_to_id[g[2][0]],
                allele_to_id[g[2][1]],
            ]

            c2_patients[i] = [
                allele_to_id[g[3][0]],
                allele_to_id[g[3][1]],
                allele_to_id[g[4][0]],
                allele_to_id[g[4][1]],
            ]

        # ============================================================
        # Cython matching
        # ============================================================

        from . import _lol5_core

        if self.mismatch_direction == "donor_to_patient":
            print("[V5] Using direction: donor → patient")

            edges_d, edges_p = _lol5_core.match_subgraphs_cython(
                c1_donors,
                c2_donors,
                c1_patients,
                c2_patients,
                len(allele_to_id),
                self.limit
            )

        else:
            print("[V5] Using direction: patient → donor")

            edges_p, edges_d = _lol5_core.match_subgraphs_cython(
                c1_patients,
                c2_patients,
                c1_donors,
                c2_donors,
                len(allele_to_id),
                self.limit
            )

        # ============================================================
        # DIAGNOSTIC: types returned by Cython
        # ============================================================

        print("edges_d dtype:", edges_d.dtype)
        print("edges_p dtype:", edges_p.dtype)

        # ============================================================
        # SAFETY CHECK:
        # Cython must return LOCAL donor/patient indices.
        # ============================================================

        print("D =", D, "P =", P, "N =", D + P)

        if len(edges_d) > 0:
            print(
                "CHECK: edges_d range:",
                edges_d.min(),
                edges_d.max()
            )

            print(
                "CHECK: edges_p range:",
                edges_p.min(),
                edges_p.max()
            )

            if edges_d.max() >= D or edges_d.min() < 0:
                raise ValueError(
                    f"Cython returned invalid donor indices: "
                    f"min={edges_d.min()}, "
                    f"max={edges_d.max()}, "
                    f"expected < {D}"
                )

            if edges_p.max() >= P or edges_p.min() < 0:
                raise ValueError(
                    f"Cython returned invalid patient indices: "
                    f"min={edges_p.min()}, "
                    f"max={edges_p.max()}, "
                    f"expected < {P}"
                )

        else:
            print("CHECK: no edges returned")

        # ============================================================
        # Convert LOCAL patient IDs -> GLOBAL graph node IDs
        #
        # donors:
        #     0 ... D-1
        #
        # patients:
        #     D ... D+P-1
        #
        # edges_p is int64, so large global IDs are preserved.
        # ============================================================

        edges_p = edges_p + D

        print(
            "global edges_p dtype:",
            edges_p.dtype
        )

        if len(edges_p) > 0:
            print(
                "GLOBAL edges_p range:",
                edges_p.min(),
                edges_p.max()
            )

        # ============================================================
        # Edge array
        # ============================================================

        edges_arr = np.column_stack(
            (edges_d, edges_p)
        )

        if len(edges_arr) > 0:
            edges_arr = edges_arr[
                np.argsort(edges_arr[:, 0])
            ]

        # ============================================================
        # CSR graph
        # ============================================================

        N = D + P

        # CSR offsets can be large -> int64
        index = np.zeros(
            N + 1,
            dtype=np.int64
        )

        # IMPORTANT:
        # neighbors contains GLOBAL NODE IDs.
        # Therefore it MUST be int64, not int32.
        neighbors = np.zeros(
            len(edges_arr),
            dtype=np.int64
        )

        print(
            "neighbors dtype:",
            neighbors.dtype
        )

        if len(edges_arr) > 0:
            neighbors[:] = edges_arr[:, 1]

            counts = np.bincount(
                edges_arr[:, 0],
                minlength=N
            )

            index[1:] = np.cumsum(
                counts,
                dtype=np.int64
            )

        if len(neighbors) > 0:
            print(
                "FINAL neighbors range:",
                neighbors.min(),
                neighbors.max()
            )

            if neighbors.min() < 0:
                raise ValueError(
                    "Negative global node ID detected in neighbors: "
                    f"min={neighbors.min()}"
                )

            if neighbors.max() >= N:
                raise ValueError(
                    "Global node ID outside graph range: "
                    f"max={neighbors.max()}, N={N}"
                )

        # ============================================================
        # ID -> genotype mapping
        #
        # IMPORTANT:
        # Position in this list IS the global node ID.
        # Therefore:
        #
        #     id_to_geno[node_id]
        #
        # still restores the genotype.
        # ============================================================

        id_to_geno = (
            [
                ",".join(flatten(g))
                for g in unique_donors
            ]
            +
            [
                ",".join(flatten(g))
                for g in unique_patients
            ]
        )

        # ============================================================
        # Donor / patient node IDs
        # ============================================================

        donors_set = set(
            range(D)
        )

        patients_set = set(
            range(D, D + P)
        )

        # ============================================================
        # Number of occurrences
        # ============================================================

        num_occ = np.zeros(
            N,
            dtype=np.int32
        )

        for i, g in enumerate(unique_donors):
            num_occ[i] = donor_counts[g]

        for j, g in enumerate(unique_patients):
            num_occ[D + j] = patient_counts[g]

        # ============================================================
        # Final diagnostics
        # ============================================================

        print(
            "FINAL GRAPH:",
            f"N={N}",
            f"D={D}",
            f"P={P}",
            f"edges={len(neighbors)}"
        )

        print(
            "FINAL DTYPES:",
            f"index={index.dtype}",
            f"neighbors={neighbors.dtype}"
        )

        # ============================================================
        # Return graph
        # ============================================================

        return LolGraph(
            index,
            neighbors,
            id_to_geno,
            donors_set,
            patients_set,
            num_occ
        )