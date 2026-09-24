# Coverage

Code for directional HLA donor coverage optimization.

## Repository structure

- `src/` - graph construction, greedy/random optimization, and Cython cores
- `scripts/` - command-line runner
- `config/` - base and scenario configuration files

Raw donor and patient datasets are not included in this repository. Users should provide compatible input files locally.

## Installation

Install Python dependencies:

    pip install -r requirements.txt

Build the Cython extensions from the repository root:

    python src/setup_lol_unified.py build_ext --inplace

## Running the analysis

Single-scenario example:

    python -m scripts.run_coverage single \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --scenario scenario1 \
      --coverage 80 \
      --method both

Multi-scenario example:

    python -m scripts.run_coverage multi \
      --donors path/to/donors.csv \
      --patients path/to/patients.csv \
      --include scenario1 scenario2 HvG GvH \
      --mismatch-limit 3 \
      --coverage 80 \
      --method both

Generated outputs are written to `runs/`.

## Data availability

The donor and patient genotype datasets used in the study are not distributed with this repository because they are subject to data-access restrictions.
