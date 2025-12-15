
"""
load_activation_dict(contributions_df)

Build two activation dictionaries (positive and negative) mapping data point
indices to vectors of `ConvolutionFeature` objects extracted from the contributions in contributions_df. 
Used downstream by enrichment routines.

Returns: (activation_dict_positive, activation_dict_negative)
"""
function load_activation_dict(contributions_df::AbstractDataFrame)
    ad = Dict{EpicHyperSketch.IntType, Vector{EpicHyperSketch.ConvolutionFeature}}() # activation_dict_positive

    @info "Constructing activation dictionary..."

    data_pt_indices = contributions_df.data_pt_index
    filter_indices = contributions_df.filter_index
    positions = contributions_df.position
    contributions = contributions_df.contribution
    n = nrow(contributions_df)

    @inbounds for i in 1:n
        push!(get!(ad, data_pt_indices[i], Vector{EpicHyperSketch.ConvolutionFeature}()),
                EpicHyperSketch.make_conv_feature(filter_indices[i], contributions[i], positions[i]))
    end

    return ad
end


"""
    m_symbols(n::Integer) -> Vector{Symbol}

Return [:m1, :m2, ..., :mn]. Returns empty vector for n <= 0.

# Example
```julia
m_symbols(3)  # Returns [:m1, :m2, :m3]
```
"""
function m_symbols(n::Integer)
    @assert n > 0 "n must be a positive integer"
    return [Symbol("m$i") for i in 1:n]
end

function m_position_symbols(n::Integer)
    @assert n > 0 "n must be a positive integer"
    return [Symbol("m$(i)_position") for i in 1:n]
end

"""
    d_symbols(n::Integer) -> Vector{Symbol}

Return distance column symbols for n motifs:
- For n==1: [:d12] 
- For n==2: [:d12, :d23]
- For n>=1: [:d12, :d23, ..., :d(n)(n+1)]

Note: For n motifs, there are (n-1) distance columns.

# Example
```julia
d_symbols(3)  # Returns [:d12, :d23]
```
"""
function d_symbols(n::Integer)
    @assert n ≥ 1 "n must be >= 1"
    return [Symbol("d$i$(i+1)") for i in 1:(n-1)]
end

"""
Get motif position symbols for a given set of motif columns.
"""
get_motif_position_symbols(motif_cols) = Symbol.(string.(motif_cols, "_position"))

"""
Check for missing required columns in the DataFrame.
"""
function missing_columns_check(df, motif_cols)
    missing_cols = [col for col in motif_cols if !(col in propertynames(df))]
    @assert isempty(missing_cols) "Missing required columns: $missing_cols"
    @assert :contribution in propertynames(df) "Missing :contribution column"
end

"""
    convert_all_except!(df::DataFrame, target_type::Type, except_cols) -> DataFrame

Convert all DataFrame columns to target_type except those specified in except_cols.
Modifies the DataFrame in-place.

# Arguments
- `df::DataFrame`: DataFrame to modify
- `target_type::Type`: Target type to convert columns to  
- `except_cols`: Column names to exclude (String, Vector{String}, Symbol, or Vector{Symbol})

# Examples
```julia
# Convert all to Float64 except "category"
convert_all_except!(df, Float64, "category")

# Convert all to String except multiple columns
convert_all_except!(df, String, ["id", "value"])
```
"""
function convert_all_except!(df::DataFrame, target_type::Type, except_cols)
    # Normalize except_cols to Vector{String}
    excluded = if except_cols isa String
        [except_cols]
    elseif except_cols isa Symbol
        [string(except_cols)]
    elseif except_cols isa Vector{String}
        except_cols
    elseif except_cols isa Vector{Symbol}
        string.(except_cols)
    else
        string.(except_cols)  # fallback for other iterables
    end
    
    # Convert all columns not in excluded list
    for col_name in names(df)
        if col_name ∉ excluded
            try
                df[!, col_name] = convert.(target_type, df[!, col_name])
            catch e
                @warn "Failed to convert column '$col_name' to $target_type: $e"
            end
        end
    end
    return df
end

"""
    extract_top_k_motifs(df_motifs, grouping_columns; n_top_k=10, threshold_count=30, 
                         median_contribution=:median_contribution, 
                         std_contribution=:std_contribution,
                         sort_by=median_contribution)

Extract top-k motifs separately for positive and negative contributors.

# Arguments
- `df_motifs`: DataFrame containing motif occurrences with contributions
- `grouping_columns`: Columns to group by (e.g., motif patterns and distances)
- `n_top_k`: Number of top motifs to extract from each direction (default: 10)
- `threshold_count`: Minimum occurrences required for a motif to be considered (default: 30)
- `median_contribution`: Column name for median contribution statistic
- `std_contribution`: Column name for standard deviation statistic
- `sort_by`: Column to sort by when ranking motifs (default: median_contribution)

# Returns
- DataFrame with all occurrences of the top-k positive and top-k negative motifs

# Details
Extracts top-k motifs from positive contributors (activating motifs) and top-k from 
negative contributors (suppressing motifs) separately, ensuring balanced representation
of both directions without overlap.
"""
function extract_top_k_motifs(df_motifs, grouping_columns; 
        n_top_k=10, 
        threshold_count=30,
        median_contribution=:median_contribution,
        std_contribution=:std_contribution,
        sort_by=median_contribution
        )
    # Group motifs by their configuration (pattern + distances)
    motif_groups = groupby(df_motifs, grouping_columns)
    
    # Filter out rare motifs (below occurrence threshold)
    frequent_motif_groups = filter(group -> nrow(group) > threshold_count, motif_groups)
    
    # Compute aggregate statistics for each motif configuration
    motif_statistics = combine(frequent_motif_groups, 
        :contribution => median => median_contribution,
        :contribution => std => std_contribution,
        nrow => :count
    )
    
    # Separate positive (activating) and negative (suppressing) motifs
    activating_motifs = filter(row -> row[sort_by] > 0, motif_statistics)
    suppressing_motifs = filter(row -> row[sort_by] < 0, motif_statistics)
    
    # Sort activating motifs by strength (most positive first)
    sort!(activating_motifs, sort_by, rev=true)
    
    # Sort suppressing motifs by strength (most negative first)
    sort!(suppressing_motifs, sort_by, rev=false)
    
    # Determine actual number available (may be less than requested)
    num_activating_selected = min(n_top_k, nrow(activating_motifs))
    num_suppressing_selected = min(n_top_k, nrow(suppressing_motifs))
    
    # Extract top-k from each direction
    top_activating = first(activating_motifs, num_activating_selected)
    top_suppressing = first(suppressing_motifs, num_suppressing_selected)
    
    # Combine both groups (guaranteed no overlap since signs differ)
    top_motifs_combined = vcat(top_activating, top_suppressing)
    
    # Log selection summary
    total_motifs_selected = nrow(top_motifs_combined)
    @info "Motif selection: $(num_activating_selected) activating + $(num_suppressing_selected) suppressing = $(total_motifs_selected) total"
    
    # Extract motif configuration keys and retrieve all occurrences
    motif_keys_to_keep = select(top_motifs_combined, grouping_columns)
    all_top_motifs = innerjoin(df_motifs, motif_keys_to_keep; on=grouping_columns)

    return all_top_motifs
end

"""
add_motif_positions_columns!(df_topk, motif_cols, distance_cols, filter_len)

Populate per-motif position columns in-place on `df_topk`.

This function generates position column names by appending "_position" to each
symbol in `motif_cols`, writes the first motif position from the existing
`:start` column, and then computes subsequent motif positions as
previous_position + filter_len + corresponding distance column. Computed
position columns are coerced to the same element type as the `:start` column
to ensure type-compatibility with existing `:start` and `:end` columns. A
final consistency check asserts that the last motif's end
(position + filter_len - 1) matches the existing `:end` column for every row.

Arguments
- df_topk::DataFrame: must contain `:start` and `:end` columns and the given
  distance columns.
- motif_cols::AbstractVector{<:Symbol}: motif column symbols (e.g. `:m1, :m2, ...`).
- distance_cols::AbstractVector{<:Symbol}: distance column symbols of length
  `length(motif_cols)-1` that contain distances between successive motifs.
- filter_len::Integer: length of the motif/filter used in the arithmetic.

The function mutates `df_topk` by adding columns named like `:m1_position`,
`:m2_position`, ... and returns `nothing`.
"""
function add_motif_positions_columns!(df_topk, motif_cols, distance_cols, filter_len)
    pos_cols = get_motif_position_symbols(motif_cols)
    @assert !isempty(pos_cols) "No position columns generated" # guard for motif_size == 0
    @assert length(distance_cols) == length(pos_cols) - 1 "distance_cols must be one shorter than motif_cols"

    # target element type is taken from the existing :start column to ensure
    # compatibility with :start and :end
    start_eltype = eltype(df_topk[!, :start])

    # assign first motif position (coerced to start_eltype)
    df_topk[!, pos_cols[1]] = convert.(start_eltype, df_topk[!, :start])

    # compute subsequent motif positions and coerce to start_eltype
    for (prev_col, cur_col, dist_col) in zip(pos_cols[1:end-1], pos_cols[2:end], distance_cols)
        computed = df_topk[!, prev_col] .+ filter_len .+ df_topk[!, dist_col]
        df_topk[!, cur_col] = convert.(start_eltype, computed)
    end

    # coerce constants to the same element type for a robust equality check
    filter_len_t = convert(start_eltype, filter_len)
    one_t = convert(start_eltype, 1)

    # final sanity check: last motif end should match recorded :end
    @assert all(df_topk[!, pos_cols[end]] .+ filter_len_t .- one_t .== df_topk[!, :end]) "End position mismatch"
    return nothing
end

"""Build per-group lookup mapping (filter_index, position) -> row index.
Returns a Vector of Dicts aligned with `gdf_by_data_pt_idx`."""
function build_group_lookups(gdf_by_data_pt_idx)
       # infer element types from the first non-empty group
    T = Any
    U = Any
    for g in gdf_by_data_pt_idx
        if nrow(g) > 0
            T = eltype(g.filter_index)
            U = eltype(g.position)
            break
        end
    end

    lookups = Vector{Dict{Tuple{T,U},Int}}(undef, length(gdf_by_data_pt_idx))
    for (gi, g) in enumerate(gdf_by_data_pt_idx)
        d = Dict{Tuple{T,U},Int}()
        for r in 1:nrow(g)
            d[(g.filter_index[r], g.position[r])] = r
        end
        lookups[gi] = d
    end
    return lookups
end

"""
obtain_contribution_views_all(df_topk, gdf_by_data_pt_idx, idx_to_key, motif_size)

For each row t in df_topk returns a view into the corresponding group's
contribution column for the "remaining" rows (i.e. rows not matched by the
motif entries in df_topk[t, ...]). No data copy is performed; result is a
Vector of views (AbstractVector) aligned with rows of df_topk.
"""
function obtain_contribution_views_all(df_topk, gdf_by_data_pt_idx, idx_to_key, motif_size)
    nrows = nrow(df_topk)
    views = Vector{AbstractVector{eltype(df_topk.contribution)}}(undef, nrows)

    motif_cols = m_symbols(motif_size)
    pos_cols = get_motif_position_symbols(motif_cols)

    lookups = build_group_lookups(gdf_by_data_pt_idx)

    for t in 1:nrows
        data_pt_index = df_topk[t, :data_pt_index]
        access_idx = idx_to_key[data_pt_index]
        g = gdf_by_data_pt_idx[access_idx]
        lookup = lookups[access_idx]

        # collect matched indices and sum contributions
        row_indices = Vector{Int}(undef, motif_size)
        contrib_sum = zero(eltype(df_topk.contribution))

        for i in 1:motif_size
            mi = df_topk[t, motif_cols[i]]
            pos = df_topk[t, pos_cols[i]]
            idx = get(lookup, (mi, pos), nothing)
            @assert idx !== nothing "No matching entry for motif $i in row $t"
            row_indices[i] = idx
            contrib_sum += g[idx, :contribution]
        end

        @assert contrib_sum == df_topk[t, :contribution] "Contribution sum mismatch for row $t"

        # build BitVector mask (memory-efficient) and obtain remaining indices
        n = nrow(g)
        mask = trues(n)           # BitVector
        mask[row_indices] .= false
        remaining = findall(mask)

        # store a view into the group's contribution column (no copy)
        views[t] = @view g.contribution[remaining]
    end

    return views
end

