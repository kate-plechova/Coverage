# Coverage

Code for directional HLA donor coverage optimization.

## Repository structure

- `src/` - graph construction, matching logic, greedy/random optimization, and Cython cores
- `scripts/` - command-line runner
- `config/` - base and scenario configuration files

Raw donor and patient datasets are not included in this repository. Users should provide compatible input files locally.

## Installation

Install Python dependencies:

    pip install -r requirements.txt

Build the Cython extensions from the repository root:

    python src/setup_lol_unified.py build_ext --inplace

## Available scenarios

The runner currently supports:

- `scenario1`
- `scenario2`
- `HvG`
- `GvH`

`scenario1` and `scenario2` use their predefined scenario-specific matching rules.

`HvG` and `GvH` use total HLA mismatch limits. For these scenarios, the mismatch threshold can be set with:

    --mismatch-limit N

## Available methods

The following optimization modes are available:

- `greedy`
- `random`
- `both`

## Single-scenario runs

### Scenario 1 - greedy

    python -m scripts.run_coverage single \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --scenario scenario1 \
      --coverage 80 \
      --method greedy

### Scenario 1 - random

    python -m scripts.run_coverage single \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --scenario scenario1 \
      --coverage 80 \
      --method random

### Scenario 1 - greedy and random

    python -m scripts.run_coverage single \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --scenario scenario1 \
      --coverage 80 \
      --method both

The same syntax can be used for `scenario2` by replacing:

    --scenario scenario1

with:

    --scenario scenario2

## HvG runs

Example with a total mismatch limit of 3:

    python -m scripts.run_coverage single \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --scenario HvG \
      --mismatch-limit 3 \
      --coverage 80 \
      --method greedy

Random:

    python -m scripts.run_coverage single \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --scenario HvG \
      --mismatch-limit 3 \
      --coverage 80 \
      --method random

Both:

    python -m scripts.run_coverage single \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --scenario HvG \
      --mismatch-limit 3 \
      --coverage 80 \
      --method both

## GvH runs

Example with a total mismatch limit of 3:

    python -m scripts.run_coverage single \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --scenario GvH \
      --mismatch-limit 3 \
      --coverage 80 \
      --method greedy

Random:

    python -m scripts.run_coverage single \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --scenario GvH \
      --mismatch-limit 3 \
      --coverage 80 \
      --method random

Both:

    python -m scripts.run_coverage single \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --scenario GvH \
      --mismatch-limit 3 \
      --coverage 80 \
      --method both

## Multi-scenario runs

A common donor set can be optimized across several scenarios.

### Greedy

    python -m scripts.run_coverage multi \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --include scenario1 scenario2 HvG GvH \
      --mismatch-limit 3 \
      --coverage 80 \
      --method greedy

### Random

    python -m scripts.run_coverage multi \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --include scenario1 scenario2 HvG GvH \
      --mismatch-limit 3 \
      --coverage 80 \
      --method random

### Greedy and random

    python -m scripts.run_coverage multi \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --include scenario1 scenario2 HvG GvH \
      --mismatch-limit 3 \
      --coverage 80 \
      --method both

The list passed to `--include` can contain any subset of:

    scenario1 scenario2 HvG GvH

For example:

    python -m scripts.run_coverage multi \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --include scenario1 HvG GvH \
      --mismatch-limit 2 \
      --coverage 90 \
      --method greedy

## Coverage target

The desired target coverage is specified with:

    --coverage N

For example:

    --coverage 80

requests an 80% coverage target.

If the target is not reachable under the selected matching rules, the algorithm may terminate below the requested coverage after exhausting useful donor candidates.

## Output

Generated outputs are written under:

    runs/

Each run stores its configuration and summary together with method-specific output files.

Typical files include:

    run_config.json
    summary.json
    run.log
    greedy.csv
    random.csv

The exact output directory depends on the input dataset, scenario, coverage target, mismatch limit, and selected method.

## Data availability

The donor and patient genotype datasets used in the study are not distributed with this repository because they are subject to data-access restrictions.

Users should provide their own locally available datasets in the expected input format.
