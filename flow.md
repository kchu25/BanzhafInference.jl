# BanzhafInference Computation Flow

## Overview
This document describes the end-to-end flow of computing Banzhaf indices for motifs and testing their significance.

## Main Pipeline

### 1. Data Preparation & Contribution Computation
```
obtain_contributions_df()
  ↓
  • Compute contributions for each data point using the model
  • Filter contributions based on train_stats and threshold_stats
  • Return DataFrame with columns: [data_pt_index, filter_index, position, contribution, ...]
```

### 2. Setup Configurations
```
banzhaf_setups(m, contributions_df)
  ↓
  • MotifEnumerationConfig (ec): motif size, filter length, seed
  • BanzhafAlgorithmConfig (ac): nonlinearity, scale_back functions, num_samples
  • MotifDataCache (mdc): grouped contributions, lookup indices
  • BackgroundConfig (bc): coalition parameters for random baseline
  
  Note: Automatically adjusts n_coalitions_per_datapoint to cap total at MAX_BG_DATA_PTs (10k)
```

### 3. Filter Contributions
```
prepare_banzhaf_data(contributions_df)
  ↓
  • Ensure each data point has minimum coalition size (N_ROWS_THRESHOLD=2)
  
filter_via_magnitude(contributions_df, contribs)
  ↓
  • Threshold by magnitude (MAG_PERCENTILE=0.95)
  • Returns: contribs_filtered (view), contributions_df_filtered (DataFrame)
```

## Single Motif Analysis

### 4. Compute Single Motif Banzhaf Indices
```
single_motifs_banzhaf!(ac, ec, contribs_filtered, contributions_df_filtered)
  ↓
  obtain_banzhafs_enhanced()
    ↓
    • For each contribution vector (leave-one-out configuration):
      - Sample random coalitions (num_samples_per_vec times)
      - Compute subset sums on GPU
      - Compute union terms: target_val + subset_sum
      - Apply transformations: scale_back(final_nonlinearity(x))
      - Calculate marginal contribution: union - subset
      - Average across all samples
    ↓
  • Add banzhaf column to contributions_df_filtered
  • Report statistics (max, min, mean)
```

### 5. Generate Random Coalitions (Background Distribution)
```
generate_random_coalitions_from_contributions_df(contributions_df_filtered, bc, ac, ec)
  ↓
  • For each data point group:
    - Sample n_coalitions_per_datapoint random coalitions
    - Each coalition: random subset of size ∈ [min_coalition_size, group_size-1]
    - Compute Banzhaf index for each random coalition
  ↓
  • Subsample to MAX_BG_DATA_PTs (10k) if exceeds limit
  • Return: DataFrame with background Banzhaf distribution
```

### 6. Significance Testing
```
single_motifs_and_significance_filtering!(ac, ec, contribs_filtered, 
                                          contributions_df_filtered, random_coalitions)
  ↓
  filter_and_test_significance(contributions_df_filtered, columns_of_interest, random_coalitions)
    ↓
    • Group motifs by filter_index (or filter_index + position for mutegenesis)
    • Filter groups with count > COUNT_THRESHOLD (25)
    ↓
    get_significant_motifs_gpu(grouped_motifs_dfs, random_coalitions)
      ↓
      • For each motif group:
        - Subsample to MAX_BANZHAF_PER_GROUP (5k) if exceeds limit
        - Perform Mann-Whitney U test against random_coalitions on GPU
        - Compute p-values using normal approximation
      • Apply Benjamini-Hochberg FDR correction
      • Filter to q-value < Q_THRESHOLD (1e-5)
  ↓
  apply_final_filters!(contributions_df_filtered, df_significant, columns_of_interest)
    ↓
    • Keep top and bottom motifs by median_banzhaf
    • Filter by positive distances (for multi-motifs)
    • Return: filtered singletons DataFrame
```

## Multi-Motif Analysis

### 7. Extract Multi-Motif Configurations
```
obtain_multi_motifs(ec, seed, m_syms, d_syms, contributions_df_filtered; motif_size=2)
  ↓
  • For each contribution sample (ec.num_contrib_samples times):
    - Subsample contributions (ec.subsample_rows per group)
    - Load activation dictionary
    ↓
    extract_motifs_from_sample()
      ↓
      • Enumerate all co-occurring motif pairs/triplets
      • Aggregate contribution statistics
  ↓
  • Deduplicate motif configurations
  • Add position columns for each motif in the configuration
```

### 8. Compute Multi-Motif Banzhaf Indices
```
compute_motif_banzhafs(contributions_df_filtered, ec, ac, mdc, seed, motif_size)
  ↓
  obtain_contribution_views_all(df_motifs, mdc)
    ↓
    • For each multi-motif configuration:
      - Find all data points containing this configuration
      - Create view of contributions for leave-one-out analysis
  ↓
  obtain_banzhafs_enhanced(views, df_motifs.contribution)
    ↓
    • Same Banzhaf computation as single motifs
    • Each view = coalition of features for that configuration
  ↓
  • Add banzhaf column to df_motifs
```

### 9. Multi-Motif Significance Testing
```
obtain_multi_motifs_and_banzhafs(contributions_df_filtered, mdc, ec, ac, random_coalitions)
  ↓
  • For each motif_size in [2, 3, ...]:
    - Compute multi-motif Banzhaf indices
    - Group by motif configuration columns
    - Test significance against random_coalitions (same as single motifs)
    - Apply final filters
  ↓
  • Return: list of DataFrames (one per motif size) and significance results
```

## GPU Acceleration Details

### Banzhaf Computation (compute_banzhaf_single_batch)
```
1. Generate random masks for coalition sampling
2. Compute subset sums on GPU (parallel across all samples)
3. Compute union terms on GPU (add target values)
4. Transfer to CPU for transformation (scale_back, nonlinearity)
5. Transfer back to GPU for accumulation
6. Average marginal contributions per vector on GPU
```

### Significance Testing (get_significant_motifs_gpu)
```
1. Collect all motif group data and random coalition data
2. Transfer to GPU
3. Launch kernel: one block per motif group
   - Each thread processes subset of group samples
   - Compute Mann-Whitney U statistic via rank-sum
   - Use shared memory for parallel reduction
4. Transfer results to CPU
5. Compute p-values and FDR correction on CPU
```

## Memory Management

### Adaptive Batching
- Estimates memory usage based on vector sizes and num_samples
- Caps batch size at BATCH_PROCESSING_THRESHOLD (10k)
- Automatic OOM recovery: halves batch size and retries on failure

### Background Data Subsampling
- Random coalitions capped at MAX_BG_DATA_PTs (10k total)
- Adjusts n_coalitions_per_datapoint dynamically based on num_data_pts
- Ensures significance testing remains fast regardless of dataset size

### Motif Group Subsampling
- Each motif group capped at MAX_BANZHAF_PER_GROUP (5k)
- Maintains statistical power while speeding up Mann-Whitney tests
- Random sampling without replacement preserves distribution

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| MAG_PERCENTILE | 0.95 | Magnitude threshold for filtering contributions |
| N_ROWS_THRESHOLD | 2 | Min coalition size per data point |
| MAX_INTERACTION_ORDER | 3 | Max motif size |
| N_COALITION_PER_PT | 20 | Coalitions per data point (adjusted dynamically) |
| MIN_COALITION_SIZE | 2 | Min random coalition size |
| NUM_SAMPLES_PER_COALITION | 100 | Samples for Banzhaf estimation |
| Q_THRESHOLD | 1e-5 | FDR significance threshold |
| MAX_BG_DATA_PTs | 10000 | Max background coalitions |
| MAX_BANZHAF_PER_GROUP | 5000 | Max samples per motif group |
| BATCH_PROCESSING_THRESHOLD | 10000 | Max batch size for GPU |

## Output

### Single Motifs
- `df_significant`: Significant motifs with statistics (mean/std/median banzhaf, p-value, q-value)
- `contributions_df_filtered_singletons`: Filtered contributions for significant motifs only

### Multi-Motifs
- `dfs`: List of DataFrames with multi-motif configurations and their Banzhaf indices
- `df_significants`: List of DataFrames with significance test results per motif size
