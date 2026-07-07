
"""
    process_activation_dict(ctx, motif_size)

Core pipeline for one activation dictionary (positive or negative contributions).
Steps: extract enriched motif configurations → validate and normalize types →
select top-k groups by contribution → compute motif positions → obtain views of
remaining contributions (no copies) → compute Banzhaf values. Returns a DataFrame
with motif info and Banzhaf scores.
"""
function process_activation_dict(ctx::MotifContext, motif_size::Int, 
                                 bc_ctx::BanzhafConvAssignContext)                                 

    m_syms = m_symbols(motif_size)
    d_syms = d_symbols(motif_size)
    
    sep_by = bc_ctx.mutagenesis ? vcat(m_syms, d_syms, [:start]) : m_syms

    df_motifs = EpicHyperSketch.obtain_enriched_configurations_partitioned(ctx.ad; motif_size, bc_ctx.filter_len)

    assert_required_motif_columns(df_motifs, m_syms)
    convert_all_except!(df_motifs, eltype(ctx.data_pt_keys), :contribution)
    
    df_topk = extract_top_k_motifs(df_motifs, sep_by; 
                                    bc_ctx.n_top_k, bc_ctx.threshold_count, bc_ctx.sort_by)

    add_motif_positions_columns!(df_topk, m_syms, d_syms, bc_ctx.filter_len)
    
    views = obtain_contribution_views_all(df_topk, ctx.gdf_by_data_pt_idx, 
                                          ctx.idx_to_key, motif_size)
                                          
    banzhafs = BanzhafInference.BanzhafCompute.obtain_banzhafs_enhanced(
        views, df_topk.contribution; 
            num_samples_per_vec=bc_ctx.num_samples_per_vec, 
            seed=bc_ctx.seed,
            final_nonlinearity = bc_ctx.final_nonlinearity,
            scale_back_function = bc_ctx.scale_back_function            
            )
        # TODO: take care of non-linearity here
    df_topk.banzhaf = banzhafs
    return df_topk
end