Base.@kwdef struct MotifEnumerationConfig
    max_interaction_order::Int = 2
    filter_len::Int = 7
    n_top_k::Int = 25 
    threshold_count::Int = 1
    sort_by::Symbol = :median_contribution
    seed::Union{Nothing, Int} = nothing
    mutagenesis::Bool = false # may not need
    subsample_rows::Int = 5
    num_contrib_samples::Int = 90
    cache_folder_path::String = "./cache"
end

Base.@kwdef struct BanzhafAlgorithmConfig
    num_samples_per_vec::Int = 100
    final_nonlinearity = FunctorWrapper(x->x)
    scale_back_function = FunctorWrapper(x->x)
    normalization_method::Union{Symbol, Nothing} = nothing
end

"""
    BackgroundConfig
    Randomly sampled coalitions used for statistical tests to take only the significant functional motifs
        n_coalitions_per_datapoint: how many random coalitions to sample per data point
        min_coalition_size: minimum size of coalitions
        num_samples_per_coalition: number of random coalitions to approximate banzhaf
"""

Base.@kwdef struct BackgroundConfig
    n_coalitions_per_datapoint::Int = 20
    min_coalition_size::Int = 2
    num_samples_per_coalition::Int = 100
end

"""
    MotifDataCache

Holds precomputed structures for efficient motif analysis: grouped contributions
by data point, lookup dicts mapping keys ↔ indices, and activation dictionaries
split by positive/negative contributions. Build once, reuse across motif sizes.
"""
Base.@kwdef struct MotifDataCache
    contributions_by_data_pt::GroupedDataFrame
    data_pt_to_group_idx::Dict
    unique_data_pt_indices::Vector
end


"""
    prepare_motif_context(contributions_df)

Build a reusable `MotifContext` from a contributions DataFrame. Groups by data
point index, extracts keys, and splits contributions into positive/negative
activation dictionaries — sets you up for fast multi-motif analysis.
"""
# function prepare_motif_context(contributions_df::DataFrame)
#     activation_dict = load_activation_dict(contributions_df)
#     gdf_by_data_pt_idx = groupby(contributions_df, :data_pt_index)
#     data_pt_keys = [k.data_pt_index for k in keys(gdf_by_data_pt_idx)]
#     idx_to_key = Dict(k => i for (i, k) in enumerate(data_pt_keys))
#     return MotifDataCache(gdf_by_data_pt_idx, idx_to_key, data_pt_keys, activation_dict)
# end