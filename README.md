# Selective Warning of Downstream Flight Delays

Public reproducibility materials for the manuscript “Selective warning of downstream flight delays in aircraft rotations.” The supplementary archive contains code and reference outputs for an openly licensed United States flight archive and constructed examples.

## Contents

- `src/` contains connection reconstruction, warning-policy functions, assignment-history, matched-policy and daily-workload examples, and model training on the open archive and constructed connections.
- `data/` contains fictional event, policy-pair and daily-ranking examples and the model configurations. The open archive remains at Harvard Dataverse.
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

Compare that output with `results/expected_31.json`. The analysis covers January 1--31 and includes February 1 for successor lookup. The summaries report cohort coverage, agreement between scheduled-order and actual-order successors, timing-rule counts, and separate cancellation and diversion totals. `expected_14.json` gives the corresponding January 1--14 counts.

The model replay uses completed same-aircraft connections with 1 to 720 minutes between gate arrival and departure. It assigns each connection to the inbound arrival date in Coordinated Universal Time. The six windows cover 1–5, 6–10, 11–14, 15–19, 20–24 and 25–31 January for history, expert fitting, gate fitting, selection, screening and testing, respectively. Fitting, selection and screening use outcomes observed by the end of their respective windows. The expert and gate fits use fixed random samples of 32,768 and 8,192 connections.

```bash
python -m pip install -r expert_requirements.txt
python src/public_model_replay.py data/2008.csv.bz2 --smoke --output results/model_smoke.json
python src/public_model_replay.py data/2008.csv.bz2 --output results/model_replay.json
```

`results/expected_model_replay.json` gives the screening and alert counts for the two experts and the routed model across all 21 budgets. It reports realised false alerts, observed minutes selected by each channel, and paired daily rankings at identical alert counts for eligible screened points. The trace covers 120,058 test connections. The routed candidates at 15% and 18% failed their screening limits, so the timing rule supplied the test alerts at those budgets; the independently screened tree supplies their scored model comparison.

Calculate pointwise and Bonferroni-adjusted one-sided limits for the 21 screened budget policies within each method and across all reported methods:

```bash
python src/gridwise_bounds.py results/model_replay.json --output results/gridwise_bounds.json
```

The calculation uses the false-alert counts on the independent screening partition. Its Bernoulli reference limits and the separate block-resampling results describe different sampling assumptions.

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

## Paired workload comparison

`src/paired_comparison.py` calculates the difference in qualifying delay minutes and delay capture in percentage points for two alert policies on the same connections. It reports paired 95% percentile intervals for operating-day, circular three-day-block, and aircraft-cluster resampling, with seed 42 and 1,000 replicates. The optional `airport_pair` field adds departure-arrival airport-pair resampling with seed 314159. All units are sampled with replacement. Each replicate sums minutes before calculating its share. Run the six-day constructed example:

```bash
python src/paired_comparison.py data/constructed_pairs.csv --days 6 --output results/paired_example.json
```

Compare the result with `results/expected_pairs.json`. The example uses fictional aircraft identifiers. To apply this calculation to authorised scored records, provide one row per connection with `day` (integer, from 1 to the number of calendar days), `aircraft` (aircraft group), `delay_minutes` (observed minutes for delays meeting the endpoint and zero elsewhere), and `routed_alert` and `tree_alert` (binary decisions on the same connection).

`src/daily_workload.py` ranks the tree and severity scores to each routed policy's alert count separately within each operating day. It selects alerts from the scores alone and then compares observed delay minutes, false alerts and paired three-day intervals. The supplement includes the fictional six-day input file. Run the example and compare its output with `results/expected_daily_workload.json`:

```bash
python src/daily_workload_example.py data/constructed_daily_workload.csv --output results/daily_workload.json
```

`src/matched_policies.py` uses repeated constructed flight profiles to demonstrate separate selection and screening, exact comparison at the 15% and 18% targets, and evaluation on the subset with at least 15 minutes until scheduled departure. The routed policy's selection-period false-alert count is a ceiling for each comparator. The tree uses the widest eligible threshold; empirical severity maximises observed delayed minutes among the eligible thresholds. The output gives alert count, false-alert count and captured minutes, in that order, and paired intervals for each comparison. Run the 240-connection example before reproducing the reference results:

```bash
python src/matched_policies.py --size 240 --output results/matched_small.json
python src/matched_policies.py --output results/matched_example.json
```

Compare the complete example with `results/expected_matched.json`. These profiles illustrate workload and uncertainty calculations. The article reports the domestic cohort results.

## Integrity

`SHA256SUMS.txt` lists SHA-256 digests for the files in the supplementary release. Verify the extracted files before running the examples. Recomputed outputs are saved separately from the reference outputs.

## License

The code in the supplementary archive is released under the MIT License. The Harvard Dataverse input archive is distributed separately under CC0 1.0.
