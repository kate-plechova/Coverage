import numpy as np
from collections import Counter, defaultdict
from time import perf_counter

LOL_UNIFIED_REV = "2026-09-03-cython-indexed-v2"


def parse_genotype(geno_str):
    """
    Parse the current 5-locus format:
        A1+A2^B1+B2^C1+C2^DQB1_1+DQB1_2^DRB1_1+DRB1_2
    """
    geno_str = (
        geno_str
        .replace("\ufeff", "")
        .replace(" ", "")
        .replace("\t", "")
        .strip()
    )

    loci = []
    for locus in geno_str.split("^"):
        alleles = locus.split("+")
        if len(alleles) != 2:
            raise ValueError(f"Expected 2 alleles in locus, got: {locus}")
        loci.append((alleles[0], alleles[1]))

    return tuple(loci)


def flatten(geno):
    return [allele for locus in geno for allele in locus]


class LolUnifiedGraph:
    """
    CSR graph compatible with the current lol5 greedy/random API.

    Only final donor -> patient edges are stored.
    Temporary NetworkX-style sub-genotype nodes are NOT materialized.
    """

    def __init__(
        self,
        index,
        neighbors,
        id_to_geno,
        donors,
        patients,
        num_occ,
        donor_is_knockout=None,
        donor_cost=None,
    ):
        self.index = np.asarray(index, dtype=np.int64)
        self.neighbors = np.asarray(neighbors, dtype=np.int64)
        self.id_to_geno = list(id_to_geno)

        self.donors = np.asarray(list(donors), dtype=np.int64)
        self.patients = np.asarray(list(patients), dtype=np.int64)

        n = len(self.index) - 1
        self.node_name = np.array([f"node-{i}" for i in range(n)], dtype=object)
        self.num_occ = np.asarray(num_occ, dtype=np.int32)
        self.active = np.ones(n, dtype=bool)

        d = len(self.donors)
        if donor_is_knockout is None:
            donor_is_knockout = np.zeros(d, dtype=bool)
        if donor_cost is None:
            donor_cost = np.ones(d, dtype=np.float64)

        self.donor_is_knockout = np.asarray(donor_is_knockout, dtype=bool)
        self.donor_cost = np.asarray(donor_cost, dtype=np.float64)

    def get_neighbors(self, node):
        return self.neighbors[self.index[node]:self.index[node + 1]]

    def num_nodes(self):
        return len(self.node_name)

    def num_occurrence(self, node):
        return self.num_occ[node]

    def is_active(self, node):
        return self.active[node]

    def deactivate(self, node):
        self.active[node] = False

    def remove_node_csr(self, node):
        self.active[node] = False


class LolUnifiedGraphBuilder:
    """
    Universal CSR builder.

    Matching modes
    --------------
    per_class:
        Generic mode for the old S1/S2/S3 family.
        Class I and Class II are checked independently.

    total:
        Keeps the current lol5 total-mismatch algorithm as a separate path.
        It delegates matching to the existing _lol5_core so the working
        total implementation is not silently rewritten.

    Scenario-specific behavior is NOT hard-coded here. Scenarios are presets
    made from generic configuration flags.
    """

    def __init__(self, conf):
        self.conf = dict(conf)

        self.mismatch_mode = self.conf.get("mismatch_mode", "per_class")

        self.use_class1 = self.conf.get("use_class1", True)
        self.use_class2 = self.conf.get("use_class2", True)

        self.class1_size = int(self.conf.get("class1_size", 6))
        self.class2_size = int(self.conf.get("class2_size", 4))

        self.mismatch_c1 = int(self.conf.get("missmatch_c1", 0))
        self.mismatch_c2 = int(self.conf.get("missmatch_c2", 0))

        pat_c1 = self.conf.get("missmatch_pat_c1")
        pat_c2 = self.conf.get("missmatch_pat_c2")
        self.mismatch_pat_c1 = self.mismatch_c1 if pat_c1 is None else int(pat_c1)
        self.mismatch_pat_c2 = self.mismatch_c2 if pat_c2 is None else int(pat_c2)

        self.total_mismatch_limit = int(self.conf.get("total_mismatch_limit", 0))
        self.mismatch_direction = self.conf.get(
            "mismatch_direction", "donor_to_patient"
        )

        self.use_kir = bool(self.conf.get("use_kir", False))
        self.use_knockout = bool(self.conf.get("use_knockout", False))
        self.knockout_cost = float(self.conf.get("knockout_cost", 10))

        self.check_homozygot = bool(self.conf.get("check_homozygot", False))

        self.alleles_in_donor = self.conf.get("alleles_in_donor")
        self.alleles_in_patient = self.conf.get("alleles_in_patient")

        self.per_class_backend = self.conf.get("per_class_backend", "cython")

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _read_rows(self, path, required_allele=None):
        """
        KIR is read ONLY when use_kir=True.

        This intentionally allows ordinary 2-column files:
            id,genotype

        When use_kir=True a third column is required:
            id,genotype,kir
        """
        rows = []

        with open(path, encoding="utf-8-sig") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 2:
                    raise ValueError(
                        f"{path}:{line_no}: expected at least id,genotype"
                    )

                geno_text = parts[1].strip()

                # Preserve the historical allele-filter idea: filter input
                # before graph construction.
                if required_allele and required_allele not in geno_text:
                    continue

                geno = parse_genotype(geno_text)

                kir = None
                if self.use_kir:
                    if len(parts) < 3:
                        raise ValueError(
                            f"{path}:{line_no}: use_kir=True but KIR column "
                            "is missing"
                        )
                    kir = parts[2].strip()

                rows.append((geno, kir))

        return rows

    def _aggregate(self, rows):
        """
        Historical graph code aggregates by genotype.

        For comparison with the original behavior we keep genotype as the
        aggregation key. If KIR is enabled, the first KIR seen for a genotype
        is retained. This mirrors the old assumption that one genotype does
        not meaningfully split into multiple KIR values.
        """
        counts = Counter(geno for geno, _ in rows)
        kir_by_geno = {}

        if self.use_kir:
            for geno, kir in rows:
                kir_by_geno.setdefault(geno, kir)

        return counts, kir_by_geno

    # ------------------------------------------------------------------
    # Generic class representation
    # ------------------------------------------------------------------

    def _normal_classes(self, geno):
        # Old S1/S2/S3 normalize the whole genotype by sorting all alleles
        # before splitting it into Class I / Class II.
        flat = sorted(flatten(geno))
        c1 = tuple(flat[:self.class1_size])
        c2 = tuple(flat[self.class1_size:self.class1_size + self.class2_size])
        return c1, c2

    def _make_donor_records(self, unique_donors, donor_kir):
        full_records = []
        knockout_by_geno = {}

        for geno in unique_donors:
            full_sorted = tuple(sorted(flatten(geno)))
            c1 = tuple(full_sorted[:self.class1_size])
            c2 = tuple(full_sorted[self.class1_size:self.class1_size + self.class2_size])

            full_records.append({
                "geno": full_sorted,
                "source_geno": geno,
                "full": full_sorted,
                "c1": c1,
                "c2": c2,
                "kir": donor_kir.get(geno),
                "is_knockout": False,
                "cost": 1.0,
            })

            if not self.use_knockout:
                continue

            # Original S3 first globally sorts the 10-allele genotype and only
            # then deletes one position at a time.
            flat = list(full_sorted)
            if len(flat) != self.class1_size + self.class2_size:
                continue

            for removed_i in range(len(flat)):
                ko = tuple(flat[:removed_i] + flat[removed_i + 1:])

                # Original S3: if the same 9-allele knockout already exists,
                # it only increments num_of_occurrence and DOES NOT rebuild
                # its assistance/class nodes. Therefore the first parent that
                # creates a knockout determines its class split.
                if ko in knockout_by_geno:
                    knockout_by_geno[ko]["generated_occ"] += 1
                    continue

                if self.class1_size == 6 and self.class2_size == 4 and len(flat) == 10:
                    c1_len = 5 + int(removed_i / 5)
                else:
                    c1_len = self.class1_size - (
                        1 if removed_i < self.class1_size else 0
                    )

                knockout_by_geno[ko] = {
                    "geno": ko,
                    "source_geno": geno,
                    "full": ko,
                    "c1": tuple(ko[:c1_len]),
                    "c2": tuple(ko[c1_len:]),
                    "kir": donor_kir.get(geno),
                    "is_knockout": True,
                    "cost": self.knockout_cost,
                    "generated_occ": 1,
                }

        # Old graph indexes all full donors as they are encountered, while
        # knockout nodes are interleaved. For CSR edge-set equivalence the
        # order is irrelevant; keeping full first makes metadata simpler.
        return full_records + list(knockout_by_geno.values())

    def _make_patient_records(self, unique_patients, patient_kir):
        records = []
        for geno in unique_patients:
            full_sorted = tuple(sorted(flatten(geno)))
            c1 = tuple(full_sorted[:self.class1_size])
            c2 = tuple(full_sorted[self.class1_size:self.class1_size + self.class2_size])
            records.append({
                "geno": full_sorted,
                "source_geno": geno,
                "full": full_sorted,
                "c1": c1,
                "c2": c2,
                "kir": patient_kir.get(geno),
            })
        return records

    # ------------------------------------------------------------------
    # per_class matching
    # ------------------------------------------------------------------

    @staticmethod
    def _subgraph_overlap_count(a, b):
        """
        Size of the largest common allele multiset.

        The old NetworkX implementation generated sub-genotypes by deleting
        alleles. Two sides meet when they can generate the same sub-genotype.
        Counter intersection is the direct non-materialized equivalent for
        the common-subset size.
        """
        ca = Counter(a)
        cb = Counter(b)
        return sum((ca & cb).values())

    def _class_matches(self, donor_class, patient_class, donor_miss, patient_miss):
        if not donor_class and not patient_class:
            return True

        overlap = self._subgraph_overlap_count(donor_class, patient_class)

        donor_required = max(0, len(donor_class) - donor_miss)
        patient_required = max(0, len(patient_class) - patient_miss)

        # For an explicit common sub-genotype to exist, both sides must be
        # able to reduce to a common size. With equal historical thresholds
        # this is simply overlap >= class_size - mismatch.
        required = max(donor_required, patient_required)
        return overlap >= required

    def _homozygote_ok(self, donor_full, patient_full):
        """
        Reproduce original Scenario 2 greedy_s2.py exactly enough for
        edge-set comparison.

        It scans the FULL sorted donor genotype in pairs. If the donor is
        homozygous for an allele, any patient whose full node-name contains
        that allele is excluded.
        """
        if not self.check_homozygot:
            return True

        n = len(donor_full) - (len(donor_full) % 2)
        patient_name = "~".join(patient_full)

        for i in range(0, n, 2):
            d0, d1 = donor_full[i], donor_full[i + 1]
            if d0 == d1 and d0 in patient_name:
                return False

        return True

    def _pair_matches_per_class(self, donor, patient):
        if self.use_class1:
            if not self._class_matches(
                donor["c1"],
                patient["c1"],
                self.mismatch_c1,
                self.mismatch_pat_c1,
            ):
                return False

        if self.use_class2:
            if not self._class_matches(
                donor["c2"],
                patient["c2"],
                self.mismatch_c2,
                self.mismatch_pat_c2,
            ):
                return False

        if not self._homozygote_ok(donor["full"], patient["full"]):
            return False

        # Original S1 applies KIR after Class-II candidate matching.
        # CSR stores only final compatible donor -> patient edges.
        if self.use_kir and donor["kir"] == patient["kir"]:
            return False

        return True

    def _candidate_donors(self, donor_records, patient):
        """
        Inverted allele index for a first functional implementation.

        It avoids a full D*P scan for ordinary cases. If a configured
        mismatch can legally allow zero overlap, all donors must be candidates
        to preserve correctness.
        """
        # Built lazily/cached by _match_per_class.
        raise RuntimeError("internal helper should not be called directly")

    def _match_per_class_python(self, donor_records, patient_records):
        allele_index = defaultdict(set)

        for d_id, donor in enumerate(donor_records):
            if self.use_class1:
                for allele in set(donor["c1"]):
                    allele_index[("c1", allele)].add(d_id)
            if self.use_class2:
                for allele in set(donor["c2"]):
                    allele_index[("c2", allele)].add(d_id)

        all_donors = set(range(len(donor_records)))
        edges_d = []
        edges_p = []

        for p_id, patient in enumerate(patient_records):
            candidates = set()

            if self.use_class1:
                for allele in set(patient["c1"]):
                    candidates.update(allele_index.get(("c1", allele), ()))

            if self.use_class2:
                for allele in set(patient["c2"]):
                    candidates.update(allele_index.get(("c2", allele), ()))

            # If an enabled class can accept zero overlap, inverted indexing
            # alone would miss valid zero-shared-allele pairs.
            zero_overlap_possible = True
            if self.use_class1:
                req_d = max(0, len(patient["c1"]) - self.mismatch_c1)
                req_p = max(0, len(patient["c1"]) - self.mismatch_pat_c1)
                if max(req_d, req_p) > 0:
                    zero_overlap_possible = False

            if self.use_class2:
                req_d = max(0, len(patient["c2"]) - self.mismatch_c2)
                req_p = max(0, len(patient["c2"]) - self.mismatch_pat_c2)
                if max(req_d, req_p) > 0:
                    zero_overlap_possible = False

            if not self.use_class1 and not self.use_class2:
                candidates = all_donors
            elif zero_overlap_possible:
                candidates = all_donors

            for d_id in candidates:
                if self._pair_matches_per_class(donor_records[d_id], patient):
                    edges_d.append(d_id)
                    edges_p.append(p_id)

        return (
            np.asarray(edges_d, dtype=np.int64),
            np.asarray(edges_p, dtype=np.int64),
        )

    def _pack_per_class_records(self, donor_records, patient_records):
        all_alleles = set()
        for r in donor_records:
            all_alleles.update(r["full"])
        for r in patient_records:
            all_alleles.update(r["full"])

        allele_to_id = {a: i for i, a in enumerate(sorted(all_alleles))}

        def pack_side(records):
            n = len(records)
            c1_stride = max(1, max((len(r["c1"]) for r in records), default=0))
            c2_stride = max(1, max((len(r["c2"]) for r in records), default=0))
            full_stride = max(1, max((len(r["full"]) for r in records), default=0))

            c1 = np.full((n, c1_stride), -1, dtype=np.int32)
            c2 = np.full((n, c2_stride), -1, dtype=np.int32)
            full = np.full((n, full_stride), -1, dtype=np.int32)
            c1_len = np.zeros(n, dtype=np.int32)
            c2_len = np.zeros(n, dtype=np.int32)
            full_len = np.zeros(n, dtype=np.int32)

            for i, r in enumerate(records):
                c1_vals = [allele_to_id[a] for a in r["c1"]]
                c2_vals = [allele_to_id[a] for a in r["c2"]]
                full_vals = [allele_to_id[a] for a in r["full"]]
                c1_len[i] = len(c1_vals)
                c2_len[i] = len(c2_vals)
                full_len[i] = len(full_vals)
                if c1_vals:
                    c1[i, :len(c1_vals)] = c1_vals
                if c2_vals:
                    c2[i, :len(c2_vals)] = c2_vals
                if full_vals:
                    full[i, :len(full_vals)] = full_vals

            return c1, c1_len, c2, c2_len, full, full_len

        d_arrays = pack_side(donor_records)
        p_arrays = pack_side(patient_records)

        # Shared KIR id-space. KIR is checked inside C++ before an edge is added.
        if self.use_kir:
            kir_values = sorted(
                {r["kir"] for r in donor_records} |
                {r["kir"] for r in patient_records}
            )
            kir_to_id = {k: i for i, k in enumerate(kir_values)}
            donor_kir = np.asarray(
                [kir_to_id[r["kir"]] for r in donor_records], dtype=np.int32
            )
            patient_kir = np.asarray(
                [kir_to_id[r["kir"]] for r in patient_records], dtype=np.int32
            )
        else:
            donor_kir = np.full(len(donor_records), -1, dtype=np.int32)
            patient_kir = np.full(len(patient_records), -1, dtype=np.int32)

        return d_arrays, p_arrays, donor_kir, patient_kir

    def _match_per_class_cython(self, donor_records, patient_records):
        from . import _lol_unified_core

        t0 = perf_counter()
        d, p, donor_kir, patient_kir = self._pack_per_class_records(
            donor_records, patient_records
        )
        t1 = perf_counter()

        result = _lol_unified_core.match_per_class_cython(
            d[0], d[1], d[2], d[3], d[4], d[5], donor_kir,
            p[0], p[1], p[2], p[3], p[4], p[5], patient_kir,
            int(self.use_class1),
            int(self.use_class2),
            self.mismatch_c1,
            self.mismatch_c2,
            self.mismatch_pat_c1,
            self.mismatch_pat_c2,
            int(self.check_homozygot),
            int(self.use_kir),
        )
        t2 = perf_counter()

        edges_d, edges_p = result

        packed_bytes = (
            sum(a.nbytes for a in d)
            + sum(a.nbytes for a in p)
            + donor_kir.nbytes
            + patient_kir.nbytes
        )
        edge_bytes = edges_d.nbytes + edges_p.nbytes

        print("\n" + "=" * 78)
        print("LOL_UNIFIED INTERNAL: PER_CLASS CYTHON")
        print("=" * 78)
        print(f"pack records -> NumPy          : {t1 - t0:.3f}s")
        print(f"C++ match + return to NumPy    : {t2 - t1:.3f}s")
        print(f"packed input arrays            : {packed_bytes / (1024**2):.1f} MiB")
        print(f"returned edge arrays           : {edge_bytes / (1024**2):.1f} MiB")
        print(f"returned edges                 : {len(edges_d):,}")
        print(f"per_class matching total       : {t2 - t0:.3f}s")

        return edges_d, edges_p

    def _match_per_class(self, donor_records, patient_records):
        if self.per_class_backend == "python":
            print("[LOL_UNIFIED] per_class backend=python")
            return self._match_per_class_python(donor_records, patient_records)

        if self.per_class_backend != "cython":
            raise ValueError(
                "per_class_backend must be 'cython' or 'python'"
            )

        try:
            print("[LOL_UNIFIED] per_class backend=cython")
            return self._match_per_class_cython(donor_records, patient_records)
        except ImportError as exc:
            raise RuntimeError(
                "\n"
                "============================================================\n"
                "ERROR: _lol_unified_core is not available.\n"
                "The slow Python fallback is disabled.\n"
                "\n"
                "Most likely you are using the wrong Python environment.\n"
                "Activate py312 and try again:\n"
                "\n"
                "    conda activate py312\n"
                "\n"
                "If needed, rebuild the extension:\n"
                "\n"
                "    python setup_lol_unified.py build_ext --inplace\n"
                "============================================================"
            ) from exc

    # ------------------------------------------------------------------
    # total matching: preserve current lol5 core
    # ------------------------------------------------------------------

    def _match_total(self, donor_records, patient_records):
        if self.use_knockout:
            raise NotImplementedError(
                "total + use_knockout is intentionally not mixed yet. "
                "The current working lol5 total algorithm is preserved "
                "as its own path."
            )

        # Current _lol5_core is fixed to 6 + 4.
        if self.class1_size != 6 or self.class2_size != 4:
            raise ValueError(
                "Current total core requires class1_size=6 and class2_size=4"
            )

        all_alleles = set()
        for r in donor_records:
            all_alleles.update(r["c1"])
            all_alleles.update(r["c2"])
        for r in patient_records:
            all_alleles.update(r["c1"])
            all_alleles.update(r["c2"])

        allele_to_id = {a: i for i, a in enumerate(sorted(all_alleles))}

        D = len(donor_records)
        P = len(patient_records)

        c1_d = np.zeros((D, 6), dtype=np.int32)
        c2_d = np.zeros((D, 4), dtype=np.int32)
        c1_p = np.zeros((P, 6), dtype=np.int32)
        c2_p = np.zeros((P, 4), dtype=np.int32)

        for i, r in enumerate(donor_records):
            c1_d[i] = [allele_to_id[a] for a in r["c1"]]
            c2_d[i] = [allele_to_id[a] for a in r["c2"]]

        for i, r in enumerate(patient_records):
            c1_p[i] = [allele_to_id[a] for a in r["c1"]]
            c2_p[i] = [allele_to_id[a] for a in r["c2"]]

        from . import _lol5_core

        if self.mismatch_direction == "donor_to_patient":
            edges_d, edges_p = _lol5_core.match_subgraphs_cython(
                c1_d, c2_d, c1_p, c2_p,
                len(allele_to_id),
                self.total_mismatch_limit,
            )
        elif self.mismatch_direction == "patient_to_donor":
            edges_p, edges_d = _lol5_core.match_subgraphs_cython(
                c1_p, c2_p, c1_d, c2_d,
                len(allele_to_id),
                self.total_mismatch_limit,
            )
        elif self.mismatch_direction == "both":
            # donor -> patient
            d2p_d, d2p_p = _lol5_core.match_subgraphs_cython(
                c1_d, c2_d, c1_p, c2_p,
                len(allele_to_id),
                self.total_mismatch_limit,
            )

            # patient -> donor
            p2d_p, p2d_d = _lol5_core.match_subgraphs_cython(
                c1_p, c2_p, c1_d, c2_d,
                len(allele_to_id),
                self.total_mismatch_limit,
            )

            # Оставляем только пары, которые проходят лимит в ОБОИХ направлениях.
            d2p_pairs = set(zip(d2p_d.tolist(), d2p_p.tolist()))
            p2d_pairs = set(zip(p2d_d.tolist(), p2d_p.tolist()))

            common_pairs = d2p_pairs & p2d_pairs

            if common_pairs:
                edges_d, edges_p = map(
                    np.asarray,
                    zip(*common_pairs),
                )
            else:
                edges_d = np.empty(0, dtype=np.int64)
                edges_p = np.empty(0, dtype=np.int64)

        else:
            raise ValueError(
                "mismatch_direction must be "
                "donor_to_patient, patient_to_donor, or both"
            )    
        
        # KIR is optional and only touched if explicitly enabled.
        if self.use_kir and len(edges_d):
            keep = np.fromiter(
                (
                    donor_records[int(d)]["kir"] != patient_records[int(p)]["kir"]
                    for d, p in zip(edges_d, edges_p)
                ),
                dtype=bool,
                count=len(edges_d),
            )
            edges_d = edges_d[keep]
            edges_p = edges_p[keep]

        return (
            np.asarray(edges_d, dtype=np.int64),
            np.asarray(edges_p, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    # CSR
    # ------------------------------------------------------------------

    @staticmethod
    def _build_csr(edges_d, edges_p_local, D, P):
        edges_p_global = edges_p_local + D

        if len(edges_d):
            order = np.argsort(edges_d, kind="stable")
            edges_d = edges_d[order]
            edges_p_global = edges_p_global[order]

        N = D + P
        index = np.zeros(N + 1, dtype=np.int64)
        neighbors = np.asarray(edges_p_global, dtype=np.int64)

        if len(edges_d):
            counts = np.bincount(edges_d, minlength=N)
            index[1:] = np.cumsum(counts, dtype=np.int64)

        return index, neighbors

    # ------------------------------------------------------------------
    # Public build
    # ------------------------------------------------------------------

    def build_graph(self, donors_input, patients_input):
        t_all0 = perf_counter()

        print(
            f">>> LOL_UNIFIED[{LOL_UNIFIED_REV}-TIMING]: mode={self.mismatch_mode}, "
            f"class1={self.use_class1}, class2={self.use_class2}, "
            f"kir={self.use_kir}, knockout={self.use_knockout} <<<"
        )

        t0 = perf_counter()
        donor_rows = self._read_rows(
            donors_input,
            required_allele=self.alleles_in_donor,
        )
        t1 = perf_counter()

        patient_rows = self._read_rows(
            patients_input,
            required_allele=self.alleles_in_patient,
        )
        t2 = perf_counter()

        donor_counts, donor_kir = self._aggregate(donor_rows)
        t3 = perf_counter()
        patient_counts, patient_kir = self._aggregate(patient_rows)
        t4 = perf_counter()

        unique_donors = list(donor_counts.keys())
        unique_patients = list(patient_counts.keys())
        t5 = perf_counter()

        donor_records = self._make_donor_records(unique_donors, donor_kir)
        t6 = perf_counter()
        patient_records = self._make_patient_records(unique_patients, patient_kir)
        t7 = perf_counter()

        if self.mismatch_mode == "per_class":
            edges_d, edges_p = self._match_per_class(
                donor_records,
                patient_records,
            )
        elif self.mismatch_mode == "total":
            edges_d, edges_p = self._match_total(
                donor_records,
                patient_records,
            )
        else:
            raise ValueError(
                f"Unknown mismatch_mode={self.mismatch_mode!r}; "
                "expected 'per_class' or 'total'"
            )
        t8 = perf_counter()

        D = len(donor_records)
        P = len(patient_records)
        N = D + P

        index, neighbors = self._build_csr(edges_d, edges_p, D, P)
        t9 = perf_counter()

        id_to_geno = [
            ",".join(r["geno"]) for r in donor_records
        ]
        id_to_geno.extend(
            ",".join(r["geno"]) for r in patient_records
        )
        t10 = perf_counter()

        num_occ = np.zeros(N, dtype=np.int32)

        for d_id, r in enumerate(donor_records):
            if r["is_knockout"]:
                num_occ[d_id] = int(r.get("generated_occ", 1))
            else:
                num_occ[d_id] = donor_counts[r["source_geno"]]

        for p_id, geno in enumerate(unique_patients):
            num_occ[D + p_id] = patient_counts[geno]

        donor_is_knockout = np.array(
            [r["is_knockout"] for r in donor_records],
            dtype=bool,
        )
        donor_cost = np.array(
            [r["cost"] for r in donor_records],
            dtype=np.float64,
        )
        t11 = perf_counter()

        graph = LolUnifiedGraph(
            index=index,
            neighbors=neighbors,
            id_to_geno=id_to_geno,
            donors=range(D),
            patients=range(D, D + P),
            num_occ=num_occ,
            donor_is_knockout=donor_is_knockout,
            donor_cost=donor_cost,
        )
        t12 = perf_counter()

        print(
            "[LOL_UNIFIED] FINAL GRAPH:",
            f"N={N}",
            f"D={D}",
            f"P={P}",
            f"edges={len(neighbors)}",
        )

#        print("\n" + "=" * 78)
#        print("LOL_UNIFIED DETAILED BUILD TIMING")
#        print("=" * 78)
#        print(f"read donors                    : {t1 - t0:.3f}s")
#        print(f"read patients                  : {t2 - t1:.3f}s")
#        print(f"aggregate donors               : {t3 - t2:.3f}s")
#        print(f"aggregate patients             : {t4 - t3:.3f}s")
#        print(f"unique lists                   : {t5 - t4:.3f}s")
#        print(f"make donor records             : {t6 - t5:.3f}s")
#        print(f"make patient records           : {t7 - t6:.3f}s")
#        print(f"matching incl. pack/core       : {t8 - t7:.3f}s")
#        print(f"build CSR (sort+bincount)      : {t9 - t8:.3f}s")
#        print(f"build id_to_geno strings       : {t10 - t9:.3f}s")
#        print(f"occ/cost/knockout metadata     : {t11 - t10:.3f}s")
#        print(f"LolUnifiedGraph object         : {t12 - t11:.3f}s")
#        print("-" * 78)
#        print(f"TOTAL build_graph              : {t12 - t_all0:.3f}s")
#        print(f"CSR index bytes                : {index.nbytes / (1024**2):.1f} MiB")
#        print(f"CSR neighbors bytes            : {neighbors.nbytes / (1024**2):.1f} MiB")
#        print(f"CSR total numeric bytes        : {(index.nbytes + neighbors.nbytes) / (1024**2):.1f} MiB")
        print("=" * 78)

        return graph

