
function subsample_contributions(contributions_df; max_rows_per_group=10, verbose=true, seed=nothing)
    """
    Subsample contributions dataframe to maximum rows per data_pt_index.
    Returns a view (memory efficient) of the sampled data.
    """
    !isnothing(seed) && Random.seed!(seed)
    
    gdf = groupby(contributions_df, :data_pt_index)
    
    sampled_indices = Int[]
    sizehint!(sampled_indices, min(nrow(contributions_df), length(gdf) * max_rows_per_group))

    for group in gdf
        n = nrow(group)
        parent_indices = parentindices(group)[1]
        
        if n <= max_rows_per_group
            append!(sampled_indices, parent_indices)
        else
            # Most efficient: randperm only generates what we need
            append!(sampled_indices, @view parent_indices[randperm(n)[1:max_rows_per_group]])
        end
    end

    contributions_df_sampled = @view contributions_df[sampled_indices, :]
    
    if verbose
        println("\n=== SAMPLING SANITY CHECK ===")
        println("Original size: ", nrow(contributions_df))
        println("Sampled size: ", length(sampled_indices))
        println("Max rows per group: ", max_rows_per_group)
        println("Number of groups: ", length(gdf))

        gdf_sampled = groupby(contributions_df_sampled, :data_pt_index)
        group_sizes = [nrow(g) for g in gdf_sampled]
        println("\nSampled group sizes:")
        println("  Min: ", minimum(group_sizes))
        println("  Max: ", maximum(group_sizes))
        println("  Mean: ", round(mean(group_sizes), digits=2))
        println("  All ≤ $max_rows_per_group? ", all(group_sizes .<= max_rows_per_group))

        println("\nData integrity check:")
        println("  All sampled indices valid? ", all(1 .<= sampled_indices .<= nrow(contributions_df)))
        println("  No duplicate indices? ", length(unique(sampled_indices)) == length(sampled_indices))
        println("=========================\n")
    end
    
    return contributions_df_sampled
end


function extract_motifs_from_sample(activation_dict, ec, motif_size, m_syms)
    df_motifs = EpicHyperSketch.obtain_enriched_configurations_partitioned(
        activation_dict; motif_size, ec.filter_len) # TODO need a seed as well
    BanzhafInference.missing_columns_check(df_motifs, m_syms) 
    BanzhafInference.convert_all_except!(df_motifs, IntType, :contribution)
    return df_motifs
end
