function get_significant_motifs(grouped_motifs_dfs, random_coalitions; q_thresh = 1e-5)
    p_values = Float64[]
    for group in grouped_motifs_dfs
        mtest = HypothesisTests.MannWhitneyUTest(group.banzhaf, random_coalitions.banzhaf)
        push!(p_values, pvalue(mtest))
    end

    fdr_results = adjust(p_values, BenjaminiHochberg())
    # Combine with motif statistics
    df_significant = DataFrames.combine(grouped_motifs_dfs, 
        :banzhaf => mean => :mean_banzhaf,
        :banzhaf => std => :std_banzhaf,
        :banzhaf => median => :median_banzhaf,
        :contribution => mean => :mean_contribution,
        nrow => :count
    )

    df_significant.pvalue = p_values
    df_significant.qvalue = fdr_results
    df_significant.significant = df_significant.qvalue .< q_thresh
    # Filter to keep only significant motifs
    filter!(row -> row.significant, df_significant)
    @info "Found $(nrow(df_significant)) significant motifs out of $(nrow(df_significant)) total (FDR < $(q_thresh))."
    sort!(df_significant, :mean_banzhaf, rev=true)
    return df_significant
end