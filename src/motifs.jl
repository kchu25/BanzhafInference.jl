function single_motifs_banzhaf!(ac, ec, contribs_filtered, contributions_df_filtered)
    target_vals = contributions_df_filtered.contribution;
    banzhafs = BanzhafInference.obtain_banzhafs_enhanced(
        contribs_filtered, 
        target_vals; 
        num_samples_per_vec = ac.num_samples_per_vec,
        seed = ec.seed,
        final_nonlinearity = ac.final_nonlinearity,
        # scale_back_function = ac.scale_back_function,
    ) # setup for non-linearity later
    contributions_df_filtered.banzhaf = banzhafs;
    # report stats
    println("Banzhaf stats:")
    println("Max: ", maximum(banzhafs))
    println("Min: ", minimum(banzhafs))
    println("Mean: ", mean(banzhafs))    
end

function extract_motifs_from_sample(activation_dict, ec, motif_size, m_syms)
    df_motifs = EpicHyperSketch.obtain_enriched_configurations_partitioned(
        activation_dict; motif_size, ec.filter_len) # TODO need a seed as well
    BanzhafInference.missing_columns_check(df_motifs, m_syms) 
    BanzhafInference.convert_all_except!(
        df_motifs, BanzhafInference.IntType, :contribution)
    return df_motifs
end

function obtain_multi_motifs(ec, seed, contributions_df_filtered; motif_size=2, )

    df_motifs = DataFrame();
    m_syms = BanzhafInference.m_symbols(motif_size);
    d_syms = BanzhafInference.d_symbols(motif_size);

    # starts sampling
    for offset = 0:(ec.num_contrib_samples-1)
        @info "Obtaining motifs from contribution sample $(offset + 1) / $(ec.num_contrib_samples)..."
        # println("Generating contribution sample with seed $(cur_seed + offset)...")
        contributions_df_sampled = BanzhafInference.subsample_contributions(
            obtain_contribution_views_all; max_rows_per_group=ec.subsample_rows, 
            verbose=false, seed=seed+offset)
        ad = BanzhafInference.load_activation_dict(contributions_df_sampled);
        df = extract_motifs_from_sample(ad, ec, motif_size, m_syms);
        append!(df_motifs, df);
    end

    n_before = nrow(df_motifs)
    @info "Extracted $n_before motifs before deduplication."
    unique!(df_motifs, Not(:contribution))
    n_after = nrow(df_motifs)
    @info "Retained $n_after unique motifs after deduplication."
    @info "The unique motifs is $(round(n_after / n_before * 100, digits=2))% of the original."

    BanzhafInference.add_motif_positions_columns!(df_motifs, m_syms, d_syms, ec.filter_len)
    return df_motifs, m_syms
end
