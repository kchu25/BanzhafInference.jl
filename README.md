# BanzhafInference

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kchu25.github.io/BanzhafInference.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/BanzhafInference.jl/dev/)
[![Build Status](https://github.com/kchu25/BanzhafInference.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/BanzhafInference.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/BanzhafInference.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/BanzhafInference.jl)

**BanzhafInference** is a foundational library that other packages build on. It uses the [Banzhaf power index](https://en.wikipedia.org/wiki/Banzhaf_power_index) from cooperative game theory to measure and rank the importance of sequence motifs (e.g. transcription-factor binding sites) learned by a neural network, and tests which motifs — and which combinations of motifs — contribute significantly to the model's predictions.

It is meant to be consumed as a dependency: downstream packages supply their own data and trained models, and call into this library's contribution, Banzhaf, significance, and interaction-testing routines to drive their own analysis pipelines.

Given a trained model and a dataset, the package answers three questions:

1. **Which individual motifs matter?** Each motif's Banzhaf index estimates its average marginal contribution to the model output across many random "coalitions" of the other features present in a sequence.
2. **Which motifs are statistically significant?** Real-motif Banzhaf values are compared against a null distribution of random coalitions using a Mann–Whitney U test with Benjamini–Hochberg FDR control.
3. **Do motifs interact?** Co-occurring motif pairs and triplets are tested for **synergy** or **antagonism** beyond their additive contributions via a no-intercept linear interaction model.

The heavy numerical steps (coalition sampling, subset sums, and batched Mann–Whitney tests) are GPU-accelerated with CUDA.

## How it works

The Banzhaf index treats each feature (a motif occurrence at a position) as a *player* in a cooperative game whose *payoff* is the model's prediction. A player's Banzhaf value is its average marginal contribution — how much the prediction changes when the player joins a random subset (coalition) of the other players. Motifs with large positive Banzhaf values consistently push the prediction up; large negative values push it down.

The pipeline:

```
contributions  →  magnitude filter  →  single-motif Banzhaf  →  random-coalition null
      ↓                                                                ↓
 (gradient×input)                                          Mann–Whitney U + BH-FDR
                                                                       ↓
                                            significant single motifs / multi-motifs
                                                                       ↓
                                                    k-way interaction testing (synergy)
```

1. **Contributions** — per-feature contributions are computed as gradient × input from the model, yielding `(data_pt_index, filter_index, position, contribution)` records.
2. **Filtering** — contributions are kept above a magnitude percentile, and data points with too few features are dropped.
3. **Single-motif Banzhaf** — each motif's Banzhaf index is estimated by sampling random coalitions, computing subset sums on the GPU, applying the model's final nonlinearity, and averaging marginal contributions.
4. **Null distribution** — random coalitions of features (not specific motifs) form the background.
5. **Significance** — a (GPU-batched) Mann–Whitney U test compares real motifs against the null, followed by Benjamini–Hochberg FDR correction.
6. **Multi-motif analysis** — co-occurring pairs and triplets are enumerated ([EpicHyperSketch](https://github.com/kchu25/EpicHyperSketch.jl)), scored, and tested the same way.
7. **Interaction testing** — a no-intercept regression on motif counts plus an interaction indicator detects synergistic (β > 0) vs. antagonistic (β < 0) combinations.

See [`flow.md`](flow.md) for the detailed step-by-step pipeline and [`interactions.md`](interactions.md) for the interaction-testing methodology.

## Installation

This package targets **Julia 1.12+** and requires a **CUDA-capable GPU**.

Add it as a dependency of your package (or to an environment):

```julia
using Pkg
Pkg.add(url="https://github.com/kchu25/BanzhafInference.jl")
```

Then bring its routines into your own pipeline:

```julia
using BanzhafInference
```

## Usage

The exported entry points are:

| Function | Purpose |
|----------|---------|
| `compute_and_filter_contributions` | Compute and filter per-feature contributions (gradient × input) from a trained model |
| `compute_random_coalition_banzhafs_per_datapoint` | Build the random-coalition null distribution for a single data point |
| `compute_random_coalition_banzhafs_all_datapoints` | Build the random-coalition null distribution across all data points |

A downstream package wires these together with the pipeline helpers to build its own analysis. A typical run looks like:

```julia
using BanzhafInference

# 1. Compute per-feature contributions from the trained model `m`
contributions, data_load = compute_and_filter_contributions(
    data, m, processor; train_stats=train_stats, operate_on_gpu=true)

# 2. Set up configs and filtered contributions
contribs_filtered, contributions_df_filtered, ec, ac, mdc, bc =
    obtain_contribs_filtered_and_configs(data, m, processor, train_stats)

# 3. Build the random-coalition null distribution
random_coalitions =
    compute_random_coalition_banzhafs_all_datapoints(contributions_df_filtered, ac, bc)

# 4. Single-motif Banzhaf + significance filtering
single_motifs_and_significance_filtering!(
    ac, ec, contribs_filtered, contributions_df_filtered, random_coalitions)

# 5. Multi-motif (pairs, triplets) Banzhaf + significance
dfs = obtain_multi_motifs_and_banzhafs(
    contributions_df_filtered, mdc, ec, ac, random_coalitions; motif_sizes=[2, 3])

# 6. Test for synergistic / antagonistic interactions
interactions = obtain_interaction_results(contributions_df_filtered, dfs)
```

### Output

- **Single motifs** — a DataFrame of significant motifs with per-motif statistics (mean / std / median Banzhaf, p-value, q-value), plus a filtered contributions DataFrame restricted to significant motifs.
- **Multi-motifs** — one DataFrame per motif size with motif configurations and their Banzhaf indices, plus the corresponding significance results.
- **Interactions** — per-order summaries flagging synergistic and antagonistic motif combinations.

## Key parameters

Defaults live in [`src/const.jl`](src/const.jl):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAG_PERCENTILE` | 0.95 | Magnitude threshold for filtering contributions |
| `N_ROWS_THRESHOLD` | 2 | Minimum coalition size per data point |
| `MAX_INTERACTION_ORDER` | 3 | Maximum motif size (singleton, pair, triplet) |
| `N_COALITION_PER_PT` | 20 | Random coalitions per data point (adjusted dynamically) |
| `MIN_COALITION_SIZE` | 2 | Minimum random coalition size |
| `NUM_SAMPLES_PER_COALITION` | 100 | Samples per coalition for Banzhaf estimation |
| `Q_THRESHOLD` | 1e-5 | FDR significance threshold |
| `MAX_BG_DATA_PTs` | 10000 | Maximum background coalitions for the null |
| `MAX_BANZHAF_PER_GROUP` | 5000 | Maximum samples per motif group |

## Empty motif sizes are a no-op, not an error

A given motif size (pairs, triplets, …) may yield **no** motifs — e.g. a subsample has no
above-threshold activations, or the non-overlap filter in the counter (`EpicHyperSketch`, which
skips filter combinations whose footprints overlap, gap `< filter_len`) leaves nothing for that
size. Two things make this robust rather than fatal:

- **`extract_motifs_from_sample`** ([`src/motifs.jl`](src/motifs.jl)) short-circuits an empty
  activation dictionary, and normalizes the counter's column-less "nothing found" return into a
  well-formed empty table via **`empty_motif_result`** (correct `m*`/`d*`/`start`/`end`/
  `contribution` columns, zero rows). This matches the schema `EpicHyperSketch` emits for a
  non-empty result, so the assertion and every downstream consumer see a proper empty frame.
- **`obtain_multi_motifs_and_banzhafs`** ([`src/motifs_helpers.jl`](src/motifs_helpers.jl)) detects
  an empty size and pushes the aligned empty result while **omitting** that size's per-size
  analysis (significance testing, Banzhaf/summary stats, final filtering) — those steps assume at
  least one row. Keeping the empty frame in `dfs` preserves the index-alignment with `motif_sizes`
  that `obtain_interaction_results` relies on (`k = idx + 1`); an empty frame there yields an empty
  interaction summary, and the renderer skips empty sizes.

Non-empty results are untouched, so the convolution and mutagenesis paths behave exactly as before
whenever motifs are found; a missing k-motif is simply omitted from the rest of the pipeline.

## Project status

This package is under active development (`v1.0.0-DEV`). As a library consumed by other packages, its exported API is the stable surface; the internal pipeline helpers may change between versions.

## Author

Shane Kuei-Hsien Chu (skchu@wustl.edu)
