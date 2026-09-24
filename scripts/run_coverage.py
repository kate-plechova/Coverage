#!/usr/bin/env python3
"""Command-line runner for directional HLA coverage experiments.

Two modes are intentionally kept separate:

    single  - one scenario, one mismatch setting, greedy/random/both
    multi   - one common greedy donor sequence across selected scenarios

The matching/greedy implementations are imported from the existing project.
This file only handles configuration, command-line arguments, outputs and
compact progress reporting.
"""

import argparse
import gc
import json
import os
import sys
import time
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy

from src.graph_factory import GraphFactory
from src.unified_greedy import call_greedy_from_graph
from src.unified_random import call_random_from_graph
from src.multi_scenario_greedy import call_multi_scenario_greedy
from src.multi_scenario_random import call_multi_scenario_random


DEFAULT_CONFIG = "config/base.json"
DEFAULT_SCENARIOS = "config/scenarios_paper.json"
DEFAULT_PROGRESS_EVERY = 100


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def dataset_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def safe_name(value):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


class RunLogger:
    """Write compact runner messages to terminal and run.log."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "w", buffering=1)

    def close(self):
        self._fh.close()

    def write(self, message=""):
        print(message, flush=True)
        self._fh.write(message + "\n")


class EngineOutputFilter:
    """Filter stdout produced inside existing engine code.

    Default mode:
      - hide detailed builder/debug output
      - hide repeated '* heap init:' lines
      - keep multi-greedy initialization start/end and progress lines

    Verbose mode:
      - echo everything and store everything in run.log

    This changes only presentation, never algorithmic behavior.
    """

    def __init__(self, terminal, log_fh, verbose=False, show_progress=False):
        self.terminal = terminal
        self.log_fh = log_fh
        self.verbose = verbose
        self.show_progress = show_progress
        self._buffer = ""

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line)
        return len(text)

    def flush(self):
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log_fh.flush()
        except Exception:
            pass

    def _handle_line(self, line):
        if self.verbose:
            self.terminal.write(line + "\n")
            self.log_fh.write(line + "\n")
            return

        lower = line.lower()

        # Intermediate heap initialization progress is intentionally omitted.
        if "heap init:" in lower:
            return

        # Keep messages that tell the user where result/log files were written.
        # Single-scenario engine output is otherwise intentionally compact.
        file_message = (
            ".csv" in lower
            or "saved to" in lower
            or "saved:" in lower
            or "written to" in lower
            or "output file" in lower
            or "log file" in lower
        )
        if file_message:
            self.terminal.write(line + "\n")
            self.log_fh.write(line + "\n")
            return

        if not self.show_progress:
            return

        keep = (
            "initializing heap" in lower
            or "heap ready" in lower
            or "[multi] step=" in lower
            or "[multi] selected=" in lower
            or "[multi] progress" in lower
        )

        if keep:
            self.terminal.write(line + "\n")
            self.log_fh.write(line + "\n")


@contextmanager
def engine_output(logger, verbose=False, show_progress=False):
    stream = EngineOutputFilter(
        terminal=sys.stdout,
        log_fh=logger._fh,
        verbose=verbose,
        show_progress=show_progress,
    )
    with redirect_stdout(stream):
        yield
    stream.flush()


def validate_coverage(value):
    if not 0 < value <= 100:
        raise ValueError("--coverage must be > 0 and <= 100")


def get_scenario(scenarios, name):
    if name not in scenarios:
        available = ", ".join(scenarios.keys())
        raise KeyError(f"Unknown scenario '{name}'. Available: {available}")
    return deepcopy(scenarios[name])


def make_conf(base_conf, scenario_name, scenario_conf, donors, patients, mismatch_limit=None):
    conf = deepcopy(base_conf)
    conf.update(deepcopy(scenario_conf))

    # Keep the public scenario label separate from the internal matcher label.
    conf["scenario_label"] = scenario_name
    conf["scenario"] = scenario_conf.get("scenario", scenario_name)

    conf["donors_input_file"] = donors
    conf["recipients_input_file"] = patients
    conf["result_dataset"] = dataset_stem(donors)

    if mismatch_limit is not None:
        if conf.get("mismatch_mode") != "total":
            raise ValueError(
                "--mismatch-limit can only be used with a total-mismatch "
                "scenario (HvG/GvH)."
            )
        conf["total_mismatch_limit"] = mismatch_limit

    return conf


def coverage_tag(value):
    """Compact coverage target for output directory names (80.0 -> 80)."""
    return f"{value:g}"


def default_single_run_dir(
    base_outdir, donors, scenario, mismatch_limit, coverage, method
):
    suffix = scenario
    if mismatch_limit is not None:
        suffix += f"_mm{mismatch_limit}"
    suffix += f"_cov{coverage_tag(coverage)}_{method}"
    return os.path.join(base_outdir, dataset_stem(donors), safe_name(suffix))


def default_multi_run_dir(base_outdir, donors, names, mismatch_limit, coverage, method):
    suffix = "multi_" + "-".join(names)
    if mismatch_limit is not None:
        suffix += f"_mm{mismatch_limit}"
    suffix += f"_cov{coverage_tag(coverage)}_{method}"
    return os.path.join(base_outdir, dataset_stem(donors), safe_name(suffix))


# ---------------------------------------------------------------------------
# Single-scenario run
# ---------------------------------------------------------------------------


def run_single(args):
    validate_coverage(args.coverage)

    base_conf = load_json(args.config)
    scenarios = load_json(args.scenarios_file)
    scenario_conf = get_scenario(scenarios, args.scenario)

    conf = make_conf(
        base_conf=base_conf,
        scenario_name=args.scenario,
        scenario_conf=scenario_conf,
        donors=args.donors,
        patients=args.patients,
        mismatch_limit=args.mismatch_limit,
    )

    effective_mm = conf.get("total_mismatch_limit") if conf.get("mismatch_mode") == "total" else None

    run_dir = args.output or default_single_run_dir(
        args.outdir,
        args.donors,
        args.scenario,
        effective_mm,
        args.coverage,
        args.method,
    )
    os.makedirs(run_dir, exist_ok=True)

    logger = RunLogger(os.path.join(run_dir, "run.log"))
    try:
        logger.write("SINGLE-SCENARIO COVERAGE")
        logger.write(f"scenario : {args.scenario}")
        if effective_mm is not None:
            logger.write(f"mismatch : <= {effective_mm}")
        logger.write(f"method   : {args.method}")
        logger.write(f"donors   : {args.donors}")
        logger.write(f"patients : {args.patients}")
        logger.write(f"coverage : {args.coverage}%")
        logger.write(f"output   : {run_dir}")
        logger.write("")

        save_json(
            {
                "mode": "single",
                "scenario_name": args.scenario,
                "method": args.method,
                "coverage_percentage": args.coverage,
                "donors": args.donors,
                "patients": args.patients,
                "resolved_configuration": conf,
            },
            os.path.join(run_dir, "run_config.json"),
        )

        logger.write("Building graph...")
        build_t0 = time.time()
        with engine_output(logger, verbose=args.verbose, show_progress=False):
            graph_data = GraphFactory.build(conf)
        build_time = time.time() - build_t0
        logger.write(
            f"  done ({build_time:.1f} s; donors={len(graph_data.donors):,}, "
            f"patients={len(graph_data.patients):,})"
        )

        results = {}
        g_m = None

        # Random must precede greedy because greedy may deactivate donors.
        methods = [args.method]
        if args.method == "both":
            methods = ["random", "greedy"]

        for method in methods:
            logger.write("")
            logger.write(f"Running {method}...")
            t0 = time.time()

            # Keep the detailed CSV for this run inside the same run directory.
            # Existing engine code may ignore this key until it is updated to
            # honor conf["result_csv_file"], so we only report the file if it
            # actually exists after the method finishes.
            method_conf = deepcopy(conf)
            requested_csv = os.path.join(run_dir, f"{method}.csv")
            method_conf["result_csv_file"] = requested_csv

            if method == "random":
                with engine_output(logger, verbose=args.verbose, show_progress=False):
                    result = call_random_from_graph(
                        graph_data=graph_data,
                        g_m=g_m,
                        coverage_percentage=args.coverage,
                        conf=method_conf,
                        verbose=args.verbose,
                    )
            else:
                with engine_output(logger, verbose=args.verbose, show_progress=False):
                    result = call_greedy_from_graph(
                        graph_data=graph_data,
                        g_m=g_m,
                        coverage_percentage=args.coverage,
                        conf=method_conf,
                        verbose=args.verbose,
                    )

            elapsed = time.time() - t0
            csv_path = result.get("log_file")
            if not csv_path and os.path.exists(requested_csv):
                csv_path = requested_csv

            summary = {
                "final_step": result.get("final_step"),
                "final_percentage": result.get("final_percentage"),
                "runtime_seconds": elapsed,
                "csv_file": csv_path,
            }
            results[method] = summary

            step = summary["final_step"]
            pct = summary["final_percentage"]
            step_text = f"{step:,}" if isinstance(step, int) else str(step)
            pct_text = f"{pct:.4f}%" if isinstance(pct, (int, float)) else str(pct)
            logger.write(f"Finished {method} ({elapsed:.1f} s)")
            logger.write(f"  selected donors: {step_text}")
            logger.write(f"  coverage: {pct_text}")

        save_json(
            {
                "graph_build_time_seconds": build_time,
                "results": results,
            },
            os.path.join(run_dir, "summary.json"),
        )

        del graph_data
        gc.collect()

        logger.write("")
        logger.write("Finished.")
        logger.write("")
        logger.write("Comparison:")
        method_width = max(6, max(len(method) for method in results))
        donor_width = max(
            6,
            max(
                len(f"{info['final_step']:,}")
                if isinstance(info.get("final_step"), int)
                else len(str(info.get("final_step")))
                for info in results.values()
            ),
        )
        header = f"{'method':<{method_width}}  {'donors':>{donor_width}}  {'coverage':>10}"
        logger.write(header)
        logger.write("-" * len(header))
        for method, info in results.items():
            step = info.get("final_step")
            pct = info.get("final_percentage")
            step_text = f"{step:,}" if isinstance(step, int) else str(step)
            pct_text = f"{pct:.4f}%" if isinstance(pct, (int, float)) else str(pct)
            logger.write(
                f"{method:<{method_width}}  {step_text:>{donor_width}}  {pct_text:>10}"
            )
        logger.write("")
        for method, info in results.items():
            if info.get("csv_file"):
                logger.write(f"{method} csv : {info['csv_file']}")
        logger.write(f"run config : {os.path.join(run_dir, 'run_config.json')}")
        logger.write(f"summary    : {os.path.join(run_dir, 'summary.json')}")
        logger.write(f"run log    : {os.path.join(run_dir, 'run.log')}")
    finally:
        logger.close()


# ---------------------------------------------------------------------------
# Multi-scenario run
# ---------------------------------------------------------------------------


def build_multi_graphs(base_conf, scenarios, names, donors, patients, mismatch_limit, logger, verbose):
    graphs = []

    for index, name in enumerate(names, start=1):
        scenario_conf = get_scenario(scenarios, name)
        conf = make_conf(
            base_conf=base_conf,
            scenario_name=name,
            scenario_conf=scenario_conf,
            donors=donors,
            patients=patients,
            mismatch_limit=(
                mismatch_limit
                if scenario_conf.get("mismatch_mode") == "total"
                else None
            ),
        )

        logger.write(f"  [{index}/{len(names)}] {name}: building...")
        t0 = time.time()
        with engine_output(logger, verbose=verbose, show_progress=False):
            graph_data = GraphFactory.build(conf)
        elapsed = time.time() - t0
        logger.write(f"      done ({elapsed:.1f} s)")

        graphs.append(
            {
                "name": name,
                "graph": graph_data,
                "conf": conf,
                "build_time": elapsed,
            }
        )

    return graphs


def run_multi(args):
    validate_coverage(args.coverage)

    base_conf = load_json(args.config)
    all_scenarios = load_json(args.scenarios_file)
    names = args.include or list(all_scenarios.keys())
    if not names:
        raise ValueError("No scenarios selected")
    if len(set(names)) != len(names):
        raise ValueError("Each scenario may be included only once")
    for name in names:
        get_scenario(all_scenarios, name)

    run_dir = args.output or default_multi_run_dir(
        args.outdir, args.donors, names, args.mismatch_limit, args.coverage, args.method
    )
    os.makedirs(run_dir, exist_ok=True)

    logger = RunLogger(os.path.join(run_dir, "run.log"))
    try:
        logger.write("MULTI-SCENARIO COVERAGE")
        logger.write(f"scenarios : {', '.join(names)}")
        logger.write(f"method    : {args.method}")
        if args.mismatch_limit is not None:
            logger.write(f"total mismatch override : <= {args.mismatch_limit}")
        logger.write(f"donors    : {args.donors}")
        logger.write(f"patients  : {args.patients}")
        logger.write(f"coverage  : {args.coverage}%")
        logger.write(f"output    : {run_dir}")
        logger.write("")

        logger.write("Building graphs...")
        build_t0 = time.time()
        graphs = build_multi_graphs(
            base_conf=base_conf, scenarios=all_scenarios, names=names,
            donors=args.donors, patients=args.patients,
            mismatch_limit=args.mismatch_limit, logger=logger, verbose=args.verbose,
        )
        total_build_time = time.time() - build_t0
        resolved = {item["name"]: item["conf"] for item in graphs}
        save_json({
            "mode": "multi", "scenario_names": names, "method": args.method,
            "coverage_percentage": args.coverage, "max_donors": args.max_donors,
            "progress_every": args.progress_every, "random_seed": args.random_seed,
            "donors": args.donors, "patients": args.patients,
            "resolved_configurations": resolved,
        }, os.path.join(run_dir, "run_config.json"))

        methods = [args.method] if args.method != "both" else ["random", "greedy"]
        results = {}

        for method in methods:
            logger.write("")
            logger.write(f"Running common multi-scenario {method}...")
            csv_path = os.path.join(run_dir, f"multi_scenario_{method}.csv")
            t0 = time.time()

            with engine_output(logger, verbose=args.verbose, show_progress=True):
                if method == "random":
                    result = call_multi_scenario_random(
                        graphs=graphs, coverage_percentage=args.coverage,
                        max_donors=args.max_donors, seed=args.random_seed,
                        progress_every=args.progress_every, log_file=csv_path,
                        verbose=args.verbose,
                    )
                else:
                    result = call_multi_scenario_greedy(
                        graphs=graphs, coverage_percentage=args.coverage,
                        max_donors=args.max_donors, log_file=csv_path,
                        verbose=args.verbose, progress_every=args.progress_every,
                    )

            elapsed = time.time() - t0
            scenario_summary = {name: result["scenarios"][name] for name in names}
            results[method] = {
                "runtime_seconds": elapsed,
                "num_selected_donors": result["num_selected_donors"],
                "csv_file": csv_path,
                "scenarios": scenario_summary,
            }

            logger.write(f"Finished multi {method} ({elapsed:.1f} s)")
            logger.write(f"  selected donors: {result['num_selected_donors']:,}")
            for name in names:
                pct = scenario_summary[name].get("percentage")
                logger.write(f"  {name}: {pct:.4f}%" if isinstance(pct, (int, float)) else f"  {name}: {pct}")

        save_json({
            "graph_build_time_seconds": total_build_time,
            "results": results,
        }, os.path.join(run_dir, "summary.json"))

        del graphs
        gc.collect()

        logger.write("")
        logger.write("Finished.")
        logger.write("")
        logger.write("Comparison:")
        method_width = max(6, max(len(method) for method in results))
        donor_width = max(6, max(len(f"{info['num_selected_donors']:,}") for info in results.values()))
        scenario_widths = {
            name: max(len(name), 9)
            for name in names
        }
        header = (
            f"{'method':<{method_width}}  "
            f"{'donors':>{donor_width}}  "
            + "  ".join(f"{name:>{scenario_widths[name]}}" for name in names)
        )
        logger.write(header)
        logger.write("-" * len(header))
        for method, info in results.items():
            cells = []
            for name in names:
                pct = info["scenarios"][name].get("percentage")
                cells.append(
                    f"{pct:.4f}%" if isinstance(pct, (int, float)) else str(pct)
                )
            row = (
                f"{method:<{method_width}}  "
                f"{info['num_selected_donors']:>{donor_width},}  "
                + "  ".join(
                    f"{cells[i]:>{scenario_widths[name]}}"
                    for i, name in enumerate(names)
                )
            )
            logger.write(row)

        logger.write("")
        for method, info in results.items():
            logger.write(f"{method} csv : {info['csv_file']}")
        logger.write(f"run config : {os.path.join(run_dir, 'run_config.json')}")
        logger.write(f"summary    : {os.path.join(run_dir, 'summary.json')}")
        logger.write(f"run log    : {os.path.join(run_dir, 'run.log')}")
    finally:
        logger.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_arguments(parser):
    parser.add_argument("--donors", required=True, help="Donor input CSV")
    parser.add_argument("--patients", required=True, help="Patient input CSV")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Base configuration JSON")
    parser.add_argument(
        "--scenarios-file",
        default=DEFAULT_SCENARIOS,
        help="Scenario definitions JSON",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=100.0,
        help="Coverage target in percent (default: 100)",
    )
    parser.add_argument(
        "--mismatch-limit",
        "--mismatches",
        dest="mismatch_limit",
        type=int,
        default=None,
        help="Override total mismatch limit for HvG/GvH",
    )
    parser.add_argument(
        "--outdir",
        default="runs",
        help="Base directory for generated runs (default: runs)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Exact output directory; overrides automatic run directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full internal builder/algorithm output",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run directional HLA coverage optimization experiments."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single", help="Run one scenario")
    add_common_arguments(single)
    single.add_argument(
        "--scenario",
        required=True,
        help="Scenario name from scenarios JSON (e.g. scenario1, scenario2, HvG, GvH)",
    )
    single.add_argument(
        "--method",
        choices=("greedy", "random", "both"),
        default="greedy",
        help="Selection method (default: greedy)",
    )

    multi = subparsers.add_parser("multi", help="Run common multi-scenario selection")
    add_common_arguments(multi)
    multi.add_argument("--method", choices=("greedy", "random", "both"), default="greedy", help="Selection method (default: greedy)")
    multi.add_argument("--random-seed", type=int, default=12345, help="Random seed for multi random (default: 12345)")
    multi.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Scenarios to include. Default: all scenarios in the JSON file.",
    )
    multi.add_argument(
        "--max-donors",
        type=int,
        default=None,
        help="Optional maximum number of selected donors",
    )
    multi.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print multi-greedy progress every N selected donors; 0 disables it (default: 100)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mismatch_limit is not None and args.mismatch_limit < 0:
        raise SystemExit("--mismatch-limit must be >= 0")

    if getattr(args, "progress_every", DEFAULT_PROGRESS_EVERY) < 0:
        raise SystemExit("--progress-every must be >= 0")

    try:
        if args.mode == "single":
            run_single(args)
        else:
            run_multi(args)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
