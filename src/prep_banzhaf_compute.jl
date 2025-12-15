
"""
Because power index is based on the influence across different coalition (set) of players (features),
we exclude the player set that has too few (n=1) players.
"""
function filtering_data_pts(thresholded_contributions_df::DataFrame; n_rows_threshold=1)
    @info "The dataframe has $(nrow(thresholded_contributions_df)) rows before filtering"
    thresholded_contributions_gdf_temp = groupby(thresholded_contributions_df, :data_pt_index)
    # Keep only groups with more than 1 row
    thresholded_contributions_gdf_filtered = filter(
        g -> nrow(g) > n_rows_threshold, thresholded_contributions_gdf_temp)
    # Convert back to DataFrame
    thresholded_contributions_df = vcat(thresholded_contributions_gdf_filtered...)
    @info "The dataframe has $(nrow(thresholded_contributions_df)) rows after filtering"
    return thresholded_contributions_df
end

exclude_one_grouped(gdf::GroupedDataFrame, col::Symbol) = [
    view(group[!, col], [1:i-1; i+1:nrow(group)])
    for group in gdf
    for i in 1:nrow(group)
]

"""
    extract_leave_one_out_vectors(df::DataFrame) -> Vector{SubArray}

Extract leave-one-out contribution vectors grouped by data point index.
For each data point, creates vectors with each contribution excluded in turn.
Returns a collection suitable for Banzhaf power index computation.
"""
function extract_leave_one_out_vectors(df::DataFrame)
    grouped_df = groupby(df, :data_pt_index)
    return exclude_one_grouped(grouped_df, :contribution)
end

"""
    sort_by_vector_length(vectors, df::DataFrame) -> (Vector, DataFrame)

Sort contribution vectors and corresponding DataFrame rows by vector length.
This improves GPU batching efficiency by grouping similar-sized vectors together.
"""
function sort_by_vector_length(vectors, df::DataFrame)
    lengths = length.(vectors)
    sorted_indices = sortperm(lengths)
    return vectors[sorted_indices], df[sorted_indices, :] # TODO return a reference instead?
end

"""
    prepare_banzhaf_data(df::DataFrame; n_rows_threshold=1) -> (vectors, sorted_df)

Complete data preparation pipeline for Banzhaf computation:
1. Filter out data points with insufficient observations
2. Sort by data_pt_index for proper grouping
3. Extract leave-one-out contribution vectors
4. Sort by vector length for efficient batching

Returns the prepared contribution vectors and aligned DataFrame.
"""
function prepare_banzhaf_data(df::DataFrame; n_rows_threshold=1)
    # Filter and sort
    df_filtered = filtering_data_pts(df; n_rows_threshold=n_rows_threshold)
    sort!(df_filtered, :data_pt_index)
    
    # Extract leave-one-out vectors
    contribution_vectors = extract_leave_one_out_vectors(df_filtered)
    
    # Sort by length for batching efficiency
    contribs, thresholded_contributions_df = 
        sort_by_vector_length(contribution_vectors, df_filtered)
    return contribs, thresholded_contributions_df

    """
    # Type: Vector{SubArray{Float32, 1, ...}}
        # Example structure:
        contribs[1] = SubArray([0.23, 0.45, 0.12])        # Leave-one-out for feature 1 of data point 1
        contribs[2] = SubArray([0.23, 0.12])              # Leave-one-out for feature 2 of data point 1  
        contribs[3] = SubArray([0.45, 0.12])              # Leave-one-out for feature 3 of data point 1
        contribs[4] = SubArray([0.67, 0.34, 0.21, 0.89])  # Leave-one-out for feature 1 of data point 2
        # ... and so on
    """
end
