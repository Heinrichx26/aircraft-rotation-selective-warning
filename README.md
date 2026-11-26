# Selective Warning of Downstream Flight Delays

Public reproducibility materials for the manuscript “Selective warning of downstream flight delays in aircraft rotations.” The supplementary archive contains code and reference outputs for an openly licensed United States flight archive and constructed examples.

## Contents

- `src/` contains connection reconstruction, warning-policy functions, assignment-history, matched-policy and daily-workload examples, model training on the open archive and constructed connections, and the article pipeline modules.
- `config/` contains the article calendar partition, selection, screening and monitoring rules, the model settings and the field schema.
- `data/` contains fictional event, policy-pair and daily-ranking examples and the model configurations of the open-archive replay. The open archive remains at Harvard Dataverse.
- `results/` contains aggregate reference counts from the public archive, expected outputs from constructed examples, the model card and selection grid of the article, and the article tables.
- `article/data_availability.md` gives the availability statement used in the manuscript.

Release reproducibility-v1.3 adds the article pipeline to release reproducibility-v1.2. Files carried over from reproducibility-v1.2 are unchanged apart from the README files and the availability statement.

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

`results/expected_model_replay.json` gives the screening and alert counts for the two experts and the routed model across all 21 budgets. It reports realised false alerts, observed minutes selected by each channel, and paired daily rankings at identical alert counts for eligible screened points. The trace covers 120,058 test connections.

Calculate pointwise and Bonferroni-adjusted one-sided limits for the 21 screened budget policies within each method and across all reported methods:

```bash
python src/gridwise_bounds.py results/model_replay.json --output results/gridwise_bounds.json
```

The calculation uses the false-alert counts on the independent screening partition. Its Bernoulli reference limits and the separate block-resampling results describe different sampling assumptions.

## Article pipeline

The article pipeline follows the calendar in `config/article_protocol.json`: January outcomes form the historical memory, a fixed sample of 131,072 February–March connections fits the experts, 8,192 April connections fit the probability calibration and the routing gate, 1–15 May selects policies, 16–31 May screens them, June is the main comparison and June–December is the closed-loop replay. Periods are given as month-day ranges; each range includes its first day and excludes its last day.

- `src/article_features.py` adds the decision-time features: ten operating-state measures observed in the 60 minutes before landing (180 minutes for the route) and the 28-day flight-number memory.
- `src/article_experts.py` fits the retrieval expert with seven historical families, the tree expert (XGBoost, LightGBM and CatBoost probabilities with a logistic calibration and two quantile severity models) and the routing gate.
- `src/policy_screen.py` evaluates the candidate grids of the routed hybrid, the seven comparison methods and the tree expert without routing, screens the frontier with the fixed-sequence test at tail probability 0.05 per budget or 0.05/21 jointly, admits operating-ceiling points with one-, three- and five-day block percentiles, replays weekly checks with 30-day replacement, and screens the review-window policies with tail probability 0.05/63.
- `config/article_models.json` lists every model setting, and `config/scored_record_schema.json` defines the archive fields, the connection fields, the decision-time features, the scored-record fields and the policy fields.
- `results/article_model_card.json` gives the fitting periods, sample sizes, selected epoch, calibration and gate coefficients, gate standardisation and fitting times of the article model.
- `results/article_selection_grid.csv.gz` lists all 298,222 candidate policies of the routed hybrid, the seven comparison methods and the tree expert without routing with their alerts, false alerts, delay minutes and one-sided upper limits on 1–15 May; 295,902 rows belong to the routed hybrid.
- `results/article_tables/` contains every table of the article and the Supplementary Material as a comma-separated value file; `index.csv` maps each file to its table number and caption.

Install the article dependencies and run the reduced check, which stops after screening:

```bash
python -m pip install -r article_requirements.txt
python src/article_example.py --smoke --output results/article_example_smoke.json
```

Then run the complete constructed example and compare the output with `results/expected_article_example.json`:

```bash
python src/article_example.py --output results/article_example.json
```

The constructed calendar has 500 fictional aircraft flying five legs a day between ten fictional airports from 1 January to 31 May, with network seed 7. Each period of the article calendar has a constructed counterpart, and April and May serve as the replay months. The example uses three training epochs and 60 trees per tree model so that it finishes in a few minutes on a desktop computer. Its output records the fitted model card, a label-flip check of the scoring functions, the selection and screening of all 21 budgets, April outcomes at the 15% and 18% budgets, the closed-loop replay with weekly checks and 30-day replacement under both operating ceilings, and the review-window policies with their weekly checks in April.

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
