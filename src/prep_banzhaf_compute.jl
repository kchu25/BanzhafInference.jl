
"""
Because power index is based on the influence across different coalition (set) of players (features),
we exclude the player set that has too few (n=1) players.
"""
function filtering_data_pts(thresholded_contributions_df::DataFrame; n_rows_threshold=1)
    @vinfo "The dataframe has $(nrow(thresholded_contributions_df)) rows before filtering"
    thresholded_contributions_gdf_temp = groupby(thresholded_contributions_df, :data_pt_index)
    # Keep only groups with more than 1 row
    thresholded_contributions_gdf_filtered = filter(
        g -> nrow(g) > n_rows_threshold, thresholded_contributions_gdf_temp)
    # Convert back to DataFrame
    thresholded_contributions_df = vcat(thresholded_contributions_gdf_filtered...)
    @vinfo "The dataframe has $(nrow(thresholded_contributions_df)) rows after filtering"
    return thresholded_contributions_df
end

"""
    extract_leave_one_out_vectors(df::DataFrame) -> Generator

Extract leave-one-out contribution vectors grouped by data point index.
For each data point, creates vectors with each contribution excluded in turn.
Returns a generator to avoid materializing millions of SubArrays.
The generator is consumed by compute_banzhaf_optimized which processes in batches.
"""
function extract_leave_one_out_vectors(df::DataFrame)
    grouped_df = groupby(df, :data_pt_index)
    return (
        view(group[!, :contribution], [1:i-1; i+1:nrow(group)])
        for group in grouped_df
        for i in 1:nrow(group)
    )
end

"""
    sort_by_vector_length(vectors, df::DataFrame) -> (Generator/Vector, DataFrame)

Sort contribution vectors and corresponding DataFrame rows by vector length.
This improves GPU batching efficiency by grouping similar-sized vectors together.
Note: For generators, this collects them into a vector for sorting.
The batching in compute_banzhaf_optimized will handle memory efficiently.
"""
function sort_by_vector_length(vectors, df::DataFrame)
    # Materialize generator only if needed for sorting
    if !isa(vectors, AbstractVector)
        @vinfo "Collecting generator for sorting (will be processed in batches by compute_banzhaf)"
        vectors = collect(vectors)
    end
    lengths = length.(vectors)
    sorted_indices = sortperm(lengths)
    return vectors[sorted_indices], df[sorted_indices, :] # TODO return a reference instead?
end

"""
    prepare_banzhaf_data(df::DataFrame; n_rows_threshold=1) -> (generator, df)

Complete data preparation pipeline for Banzhaf computation:
1. Filter out data points with insufficient observations
2. Sort by data_pt_index for proper grouping
3. Extract leave-one-out contribution vectors as a generator

Returns a generator of contribution vectors and the filtered DataFrame.
For large datasets (millions of vectors), sorting is skipped to avoid
materializing all vectors. The compute_banzhaf_optimized function will
process the generator in batches automatically.
"""
function prepare_banzhaf_data(df::DataFrame,; n_rows_threshold=N_ROWS_THRESHOLD)
    # Filter and sort by data_pt_index
    df_filtered = filtering_data_pts(df; n_rows_threshold=n_rows_threshold)
    sort!(df_filtered, :data_pt_index)
    
    # Extract leave-one-out vectors as generator (no materialization)
    contribution_vectors = extract_leave_one_out_vectors(df_filtered)
    
    # Return generator directly - compute_banzhaf_optimized handles batching
    # Skipping sort_by_vector_length to avoid collecting millions of vectors
    return contribution_vectors, df_filtered
end
