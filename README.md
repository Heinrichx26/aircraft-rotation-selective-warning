# Selective Warning of Downstream Flight Delays

Public reproducibility materials for the manuscript “Selective warning of downstream flight delays in aircraft rotations.” This repository contains code and reference outputs for an openly licensed United States flight archive and a fully synthetic model-training example.

## Contents

- `src/` contains the flight-connection reconstruction, warning-policy functions, assignment-history example, and the neural retrieval and tree training example.
- `data/` contains constructed flight events and model configurations. The public archive remains at Harvard Dataverse.
- `results/` contains aggregate reference counts from the public archive and expected outputs from constructed examples.
- `article/data_availability.md` gives the availability statement used in the manuscript.

## Public flight archive

The independent connection analysis uses the January 2008 Airline On-Time Data from Data Expo 2009, Harvard Dataverse, version 1.0, licensed CC0 1.0: <https://doi.org/10.7910/DVN/HG7NV7>. Run `python src/fetch_public_data.py` to download it to `data/2008.csv.bz2` and verify the published MD5 digest before analysis. The archive remains at Harvard Dataverse; this repository provides aggregate reference counts.

Using Python 3.11, install `requirements.txt`, then run the two-day check:

```bash
python -m pip install -r requirements.txt
python src/public_linkage.py data/2008.csv.bz2 --days 2 --output results/public_02.json
```

Compare the generated file with `results/expected_02.json`. After the check passes, run the complete January reconstruction:

```bash
python src/public_linkage.py data/2008.csv.bz2 --days 31 --output results/public_31.json
```

Compare that output with `results/expected_31.json`. The analysis covers January 1--31 and includes February 1 for successor lookup. The summaries report cohort coverage, agreement between scheduled-order and actual-order successors, timing-rule counts, and separate cancellation and diversion totals. `expected_14.json` preserves the earlier January 1--14 reference analysis.

## Constructed examples

Install the model-example dependencies. Run the reduced expert example first:

```bash
python -m pip install -r expert_requirements.txt
python src/expert_example.py --size 320 --output results/expert_smoke.json
```

After this check completes with finite model parameters, run the reference examples:

```bash
python src/policy_example.py --output results/policy_example.json
python src/assignment_followup.py data/constructed_events.csv --output results/assignment_example.json
python src/gate_example.py --output results/gate_example.json
python src/expert_example.py --output results/expert_example.json
```

Compare each output with its corresponding `results/expected_*.json` file. The expert example generates six independent samples of constructed flight connections, builds historical summaries, trains a neural retrieval model and a gradient-boosted tree, fits the routing gate, selects and screens policies, then evaluates all 21 alert budgets. These synthetic outputs illustrate the calculation.

## License

The code in this repository is released under the MIT License. The Harvard Dataverse input archive is distributed separately under CC0 1.0. No flight-level source records are redistributed here.
