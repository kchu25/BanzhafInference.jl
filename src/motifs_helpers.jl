
function obtain_contribs_filtered_and_configs(data, m, processor, train_stats)
 # obtain the configurations for each data point
    contributions_df = BanzhafInference.obtain_contributions_df(
        data, m, processor, train_stats);

    # setup the global config
    ec, ac, mdc, bc = BanzhafInference.banzhaf_setups(
        m, contributions_df; train_stats=train_stats);

    # ensure that each data point has at least a coalition of size N_ROWS_THRESHOLD (2 by default)
    contribs, contributions_df = 
        BanzhafInference.prepare_banzhaf_data(contributions_df);

    # get the contributions to be considered as targets via magnitude thresholding
    # contribs_filtered is the view; leave one out where the one is the corresponding row in contributions_df_filtered
    contribs_filtered, contributions_df_filtered = 
        BanzhafInference.filter_via_magnitude(
            contributions_df, contribs);
    return contribs_filtered, contributions_df_filtered, ec, ac, mdc, bc
end


"""
    Compute motifs and their Banzhaf indices for a given motif size
"""
function compute_motif_banzhafs(
    contributions_df_filtered, ec, ac, mdc, seed, motif_size)
    
    m_syms = BanzhafInference.m_symbols(motif_size);
    d_syms = BanzhafInference.d_symbols(motif_size);
    
    df_motifs = BanzhafInference.obtain_multi_motifs(
        ec, seed, m_syms, d_syms, contributions_df_filtered; 
        motif_size=motif_size);
    
    @time views = BanzhafInference.obtain_contribution_views_all(
        df_motifs, mdc; motif_size=motif_size);
    
    banzhafs = BanzhafInference.obtain_banzhafs_enhanced(
        views, df_motifs.contribution; 
        num_samples_per_vec=ac.num_samples_per_vec, 
        seed=seed,
        final_nonlinearity=ac.final_nonlinearity);
    
    df_motifs.banzhaf = banzhafs;
    return df_motifs, m_syms, d_syms
end

"""
    Filter motifs by count threshold and test for significance
"""
function filter_and_test_significance(
    df_motifs, columns_of_interest, random_coalitions; 
    COUNT_THRESHOLD=25, Q_THRESHOLD=1e-5)
    
    grouped_motifs_dfs = groupby(df_motifs, columns_of_interest);
    grouped_motifs_dfs = filter(g -> nrow(g) > COUNT_THRESHOLD, grouped_motifs_dfs);
    
    df_significant = BanzhafInference.get_significant_motifs_gpu(
        grouped_motifs_dfs, random_coalitions; q_thresh=Q_THRESHOLD);
    
    return df_significant
end


"""
    get_top_bottom(df, col::Symbol; n=10, merged=false)

Get the top and bottom `n` rows from a DataFrame sorted by `col` in descending order.
Returns a tuple `(top, bottom)` by default, or a single merged DataFrame if `merged=true`.
"""
function get_top_bottom(df, col::Symbol; n=3, merged=false)
    perm = sortperm(df[!, col], rev=true)
    nr = length(perm)
    nr ≤ 2n && return view(df, perm[1:min(n, nr)], :), view(df, perm[max(1, nr-n+1):nr], :)
    
    # Check if top and bottom would overlap (i.e., not enough rows to separate them)
    overlap = nr - n < n
    # If overlapping, expand top slightly and reduce bottom to avoid duplicates
    n_top = overlap ? min(n+5, nr) : n
    n_bot = overlap ? max(0, 2n - n_top) : n
    
    top = view(df, perm[1:n_top], :)
    bot = view(df, perm[max(1, nr-n_bot+1):nr], :)
    
    # If merged, combine top and bottom into a single DataFrame
    if merged
        return vcat(top, bot)
    else
        return top, bot
    end
end


"""
    Contruct a function that checks whether a given row is in the motif set
"""
function construct_is_in_motif_set(df_significant, columns_of_interest)
    motif_set = Set(Tuple(row) for row in 
        eachrow(select(df_significant, columns_of_interest)));
    is_in_motif_set(row) = Tuple(row[columns_of_interest]) ∈ motif_set
    return is_in_motif_set
end

"""
    Apply final filtering based on significance and distance constraints
"""
function apply_final_filters!(
    df_motifs, df_significant, columns_of_interest; d_syms=nothing, mutegenesis=false)
    
    if mutegenesis
        is_in_motif_set = construct_is_in_motif_set(df_significant, columns_of_interest);
        filter!(row -> is_in_motif_set(row), df_motifs)
    else
        df_significant = get_top_bottom(df_significant, :median_banzhaf; merged=true);
        is_in_motif_set = construct_is_in_motif_set(df_significant, columns_of_interest);
        if !isnothing(d_syms)
            has_positive_distances(row) = all(≥(0), (row[d] for d in d_syms))
            filter!(row -> is_in_motif_set(row) && has_positive_distances(row), df_motifs)
        else
            filter!(row -> is_in_motif_set(row), df_motifs)
        end
    end
end

"""
    Main function to obtain multi-motifs and their Banzhaf indices
"""
function obtain_multi_motifs_and_banzhafs(
    contributions_df_filtered, mdc, ec, ac, random_coalitions; 
    seed=1, motif_sizes=[2,3], mutegenesis=false, 
    COUNT_THRESHOLD=25, Q_THRESHOLD=1e-5)

    dfs = DataFrame[];
    df_significants = DataFrame[]
    for motif_size in motif_sizes
        # Compute motifs and Banzhaf indices
        df_motifs, m_syms, d_syms = compute_motif_banzhafs(
            contributions_df_filtered, ec, ac, mdc, seed, motif_size);
        
        # Determine columns for filtering
        mp_syms = BanzhafInference.m_position_symbols(motif_size);
        columns_of_interest = mutegenesis ? [m_syms..., d_syms..., mp_syms...] : m_syms;
        
        # Filter by significance
        df_significant = filter_and_test_significance(
            df_motifs, columns_of_interest, random_coalitions;
            COUNT_THRESHOLD=COUNT_THRESHOLD, Q_THRESHOLD=Q_THRESHOLD);
        
        # Apply final filters
        apply_final_filters!(df_motifs, df_significant, columns_of_interest; 
            d_syms=d_syms, mutegenesis=mutegenesis);
        
        push!(dfs, df_motifs)
        push!(df_significants, df_significant)
    end
    return dfs, df_significants
end
