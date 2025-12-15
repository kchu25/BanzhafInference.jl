"""
Random Coalition Banzhaf Index Computation

For each data point, we:
1. Group contributions by data_pt_index
2. Generate random coalitions for the contributions within each group
3. Compute Banzhaf indices for each coalition using obtain_banzhafs_enhanced

The key insight: For each data point, we randomly select k contributions to form a 
"coalition", then compute the Banzhaf value measuring how much that coalition 
contributes relative to all other contributions for that data point.
"""

using Random
using Statistics
using DataFrames
using ProgressMeter

"""
    generate_random_coalitions_cpu(n_items::Int, n_coalitions::Int, min_size::Int, max_size::Int; seed=nothing)

Generate random coalitions on CPU using index vectors (no size limit).

# Arguments
- `n_items`: Total number of items (contributions) available
- `n_coalitions`: Number of random coalitions to generate
- `min_size`: Minimum coalition size
- `max_size`: Maximum coalition size
- `seed`: Random seed for reproducibility

# Returns
Vector of Vector{Int}, where each inner vector contains the indices of items in that coalition
"""
function generate_random_coalitions_cpu(n_items::Int, n_coalitions::Int, 
                                       min_size::Int=1, max_size::Int=n_items; 
                                       seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    
    coalitions = Vector{Vector{Int}}(undef, n_coalitions)
    
    for i in 1:n_coalitions
        # Randomly choose coalition size
        k = rand(min_size:max_size)
        
        # Randomly select k positions (indices)
        coalitions[i] = sort(randperm(n_items)[1:k])
    end
    
    return coalitions
end


"""
    extract_coalition_and_remainder(contributions::Vector{Float32}, coalition_indices::Vector{Int})

Given a contribution vector and coalition member indices, extract:
1. The coalition members (selected contributions)
2. The remainder (unselected contributions)

# Returns
- `coalition_sum`: Sum of contributions in the coalition
- `remainder_view`: View of the remaining contributions
- `selected`: Indices of coalition members
- `unselected`: Indices of non-members
"""
function extract_coalition_and_remainder(contributions::AbstractVector, coalition_indices::Vector{Int})
    n = length(contributions)
    
    # Create a set for fast lookup
    coalition_set = Set(coalition_indices)
    
    # Find unselected positions
    unselected = [i for i in 1:n if i ∉ coalition_set]
    
    # Compute coalition sum
    coalition_sum = sum(contributions[coalition_indices])
    
    # Create view of remainder
    remainder_view = @view contributions[unselected]
    
    return coalition_sum, remainder_view, coalition_indices, unselected
end


"""
    compute_random_coalition_banzhafs_per_datapoint(group_df::SubDataFrame, 
                                                    n_coalitions::Int;
                                                    min_coalition_size::Int=2,
                                                    max_coalition_size::Union{Int,Nothing}=nothing,
                                                    num_samples_per_coalition::Int=100,
                                                    seed=nothing,
                                                    final_nonlinearity=FunctorWrapper(x->x),
                                                    scale_back_function=FunctorWrapper(x->x),
                                                    save_coalition_members::Bool=false)

Compute Banzhaf indices for random coalitions within a single data point's contributions.

# Arguments
- `group_df`: SubDataFrame containing contributions for one data point
- `n_coalitions`: Number of random coalitions to generate
- `min_coalition_size`: Minimum size of each coalition
- `max_coalition_size`: Maximum size (defaults to n_items - 1)
- `num_samples_per_coalition`: Number of samples for Banzhaf computation per coalition
- `seed`: Random seed
- `final_nonlinearity`: Nonlinearity function to apply
- `scale_back_function`: Scale back function
- `save_coalition_members`: Whether to save coalition member indices (default: false)

# Returns
DataFrame with coalition information and Banzhaf indices
"""
function compute_random_coalition_banzhafs_per_datapoint(
    group_df::AbstractDataFrame, 
    n_coalitions::Int;
    min_coalition_size::Int=2,
    max_coalition_size::Union{Int,Nothing}=nothing,
    num_samples_per_coalition::Int=100,
    seed=nothing,
    final_nonlinearity=FunctorWrapper(x->x),
    scale_back_function=FunctorWrapper(x->x),
    save_coalition_members::Bool=false
    )
    
    n_items = nrow(group_df)
    max_size = isnothing(max_coalition_size) ? n_items - 1 : max_coalition_size
    
    # Skip if not enough contributions
    if n_items < min_coalition_size + 1  # Need at least min_size + 1 for remainder
        return DataFrame()
    end
    
    # Get contribution values
    contributions = Float32.(group_df.contribution)
    
    # Generate random coalitions (as index vectors)
    coalitions = generate_random_coalitions_cpu(n_items, n_coalitions, 
                                                min_coalition_size, max_size; 
                                                seed=seed)
    
    # Prepare data structures for batch Banzhaf computation
    remainder_views = Vector{AbstractVector{Float32}}(undef, n_coalitions)
    coalition_sums = Vector{Float32}(undef, n_coalitions)
    coalition_members = Vector{Vector{Int}}(undef, n_coalitions)
    
    for i in 1:n_coalitions
        coalition_sum, remainder_view, selected, unselected = 
            extract_coalition_and_remainder(contributions, coalitions[i])
        
        remainder_views[i] = remainder_view
        coalition_sums[i] = coalition_sum
        coalition_members[i] = selected
    end
    
    # Compute Banzhaf indices for all coalitions
    banzhafs = obtain_banzhafs_enhanced(
        remainder_views,
        coalition_sums;
        num_samples_per_vec=num_samples_per_coalition,
        seed=seed,
        final_nonlinearity=final_nonlinearity,
        scale_back_function=scale_back_function
    )
    
    # Create results DataFrame
    results = DataFrame(
        coalition_id = 1:n_coalitions,
        coalition_size = [length(members) for members in coalition_members],
        coalition_sum = coalition_sums,
        banzhaf = banzhafs
    )
    
    # Optionally add coalition members
    if save_coalition_members
        results.coalition_members = coalition_members
    end
    
    return results
end


"""
    compute_random_coalition_banzhafs_all_datapoints(contributions_df::DataFrame;
                                                     n_coalitions_per_datapoint::Int=10,
                                                     min_coalition_size::Int=2,
                                                     max_coalition_size::Union{Int,Nothing}=nothing,
                                                     num_samples_per_coalition::Int=100,
                                                     seed=nothing,
                                                     final_nonlinearity=FunctorWrapper(x->x),
                                                     scale_back_function=FunctorWrapper(x->x),
                                                     max_datapoints::Union{Int,Nothing}=nothing,
                                                     save_coalition_members::Bool=false,
                                                     verbose::Bool=true)

Compute random coalition Banzhaf indices for all data points.

# Arguments
- `contributions_df`: DataFrame with contributions (must have :data_pt_index and :contribution columns)
- `n_coalitions_per_datapoint`: Number of random coalitions per data point
- `min_coalition_size`: Minimum coalition size
- `max_coalition_size`: Maximum coalition size
- `num_samples_per_coalition`: Samples for Banzhaf computation per coalition
- `seed`: Random seed
- `final_nonlinearity`: Nonlinearity function
- `scale_back_function`: Scale back function
- `max_datapoints`: Limit processing to first N data points (for testing)
- `save_coalition_members`: Whether to save coalition member indices (default: false)
- `verbose`: Show progress information

# Returns
DataFrame with all coalition Banzhaf results across all data points
"""
function compute_random_coalition_banzhafs_all_datapoints(
    contributions_df::Union{DataFrame, SubDataFrame}, 
    ac, bc;
    # n_coalitions_per_datapoint::Int=10,
    # min_coalition_size::Int=2,
    max_coalition_size::Union{Int,Nothing}=nothing,
    # num_samples_per_coalition::Int=100,
    seed=nothing,
    # final_nonlinearity=FunctorWrapper(x->x),
    # scale_back_function=FunctorWrapper(x->x),
    max_datapoints::Union{Int,Nothing}=nothing,
    save_coalition_members::Bool=false,
    verbose::Bool=true
    )
    
    # Group by data point
    gdf = groupby(contributions_df, :data_pt_index)
    
    # Limit data points if requested
    n_groups = isnothing(max_datapoints) ? length(gdf) : min(max_datapoints, length(gdf))
    
    verbose && @info "Processing $n_groups data points..."
    
    all_results = DataFrame[]
    
    progress = verbose ? Progress(n_groups, desc="Computing coalitions: ") : nothing
    
    for (group_idx, group) in enumerate(gdf[1:n_groups])
        data_pt_idx = group.data_pt_index[1]
        
        # Compute coalitions for this data point
        group_seed = isnothing(seed) ? nothing : seed + group_idx
        
        results = compute_random_coalition_banzhafs_per_datapoint(
            group,
            bc.n_coalitions_per_datapoint;
            min_coalition_size=bc.min_coalition_size,
            max_coalition_size=max_coalition_size,
            num_samples_per_coalition=bc.num_samples_per_coalition,
            seed=group_seed,
            final_nonlinearity=ac.final_nonlinearity,
            scale_back_function=ac.scale_back_function,
            save_coalition_members=save_coalition_members
        )
        
        # Add data point index to results
        if nrow(results) > 0
            results.data_pt_index .= data_pt_idx
            push!(all_results, results)
        end
        
        verbose && next!(progress)
    end
    
    # Combine all results
    if isempty(all_results)
        return DataFrame()
    end
    
    combined_results = vcat(all_results...)
    
    @info "Info about combined_results.banzhaf:"
    @info "Maximum: $(maximum(combined_results.banzhaf))"
    @info "Minimum: $(minimum(combined_results.banzhaf))"
    @info "Mean: $(mean(combined_results.banzhaf))"
    
    verbose && @info "Generated $(nrow(combined_results)) coalitions across $n_groups data points"

    return combined_results
end

