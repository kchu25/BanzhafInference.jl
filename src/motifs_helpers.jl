
function obtain_contribs_filtered_and_configs(
    data, m, processor, train_stats; 
    scale_back=false, activation_thresh=0.8, predict_position=1, 
    cache_folder_parent="./")
 # obtain the configurations for each data point
    contributions_df = BanzhafInference.obtain_contributions_df(
        data, m, processor, train_stats; predict_position=predict_position);

    # setup the global config
    multi_output = length(data.raw_data.feature_names) > 1

    ec, ac, mdc, bc = BanzhafInference.banzhaf_setups(
        m, contributions_df; train_stats=train_stats, scale_back=scale_back, 
            predict_position=predict_position, multi_output=multi_output,
            cache_folder_parent=cache_folder_parent);

    # ensure that each data point has at least a coalition of size N_ROWS_THRESHOLD (2 by default)
    contribs, contributions_df = 
        BanzhafInference.prepare_banzhaf_data(contributions_df);

    # get the contributions to be considered as targets via magnitude thresholding
    # contribs_filtered is the view; leave one out where the one is the corresponding row in contributions_df_filtered
    contribs_filtered, contributions_df_filtered = 
        BanzhafInference.filter_via_magnitude(
            contributions_df, contribs; mag_percentile=activation_thresh);
    return contribs_filtered, contributions_df_filtered, ec, ac, mdc, bc
end

is_identity(f) = isa(f, BanzhafInference.Functor1Arg) && f.functor === identity

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
    
    # if is_identity(ac.final_nonlinearity)
    #     df_motifs.banzhaf = df_motifs.contribution;
    # else
        @time views = BanzhafInference.obtain_contribution_views_all(
        df_motifs, mdc; motif_size=motif_size);
        banzhafs = BanzhafInference.obtain_banzhafs_enhanced(
            views, df_motifs.contribution; 
            num_samples_per_vec=ac.num_samples_per_vec, 
            seed=seed,
            final_nonlinearity=ac.final_nonlinearity,
            scale_back_function=ac.scale_back_function
            );
        df_motifs.banzhaf = banzhafs;    
    # end

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
    if nr ≤ 2n 
        df = df[perm, :]
        if merged 
            return df
        else
            return view(df, 1:min(n, nr), :), view(df, max(1, nr-n+1):nr, :)
        end
        # return view(df, perm[1:min(n, nr)], :), view(df, perm[max(1, nr-n+1):nr], :)
    end
    
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
    df_motifs, df_significant, columns_of_interest; 
    d_syms=nothing, mutegenesis=false, top_and_bot_counts=8
    )
    
    if mutegenesis
        is_in_motif_set = construct_is_in_motif_set(df_significant, columns_of_interest);
        df_motifs_filtered = filter(row -> is_in_motif_set(row), df_motifs)
        return df_motifs_filtered
    else
        df_significant = get_top_bottom(df_significant, :median_banzhaf; merged=true, n=top_and_bot_counts);
        is_in_motif_set = construct_is_in_motif_set(df_significant, columns_of_interest);
        if !isnothing(d_syms)
            has_positive_distances(row) = all(≥(0), (row[d] for d in d_syms))
            df_motifs_filtered = filter(row -> is_in_motif_set(row) && has_positive_distances(row), df_motifs)
        else
            df_motifs_filtered = filter(row -> is_in_motif_set(row), df_motifs)
        end
        return df_motifs_filtered
    end
end

"""
    Main function to obtain multi-motifs and their Banzhaf indices
"""
function obtain_multi_motifs_and_banzhafs(
    contributions_df_filtered, mdc, ec, ac, random_coalitions; 
    seed=1, motif_sizes=[2,3], mutegenesis=false, 
    COUNT_THRESHOLD=25, Q_THRESHOLD=1e-5, top_and_bot_counts=8
    )

    dfs = DataFrame[];
    # df_significants = DataFrame[]
    for motif_size in motif_sizes
        # Compute motifs and Banzhaf indices
        df_motifs, m_syms, d_syms = compute_motif_banzhafs(
            contributions_df_filtered, ec, ac, mdc, seed, motif_size);
        
        # Determine columns for filtering
        mp_syms = BanzhafInference.m_position_symbols(motif_size);
        columns_of_interest = mutegenesis ? [m_syms..., d_syms..., mp_syms...] : m_syms;
        
        # Debug: check data before significance testing
        @info "motif_size=$motif_size: nrow=$(nrow(df_motifs)), ncol=$(ncol(df_motifs)), columns=$(names(df_motifs))"
        @info "  Banzhaf: min=$(minimum(df_motifs.banzhaf)) max=$(maximum(df_motifs.banzhaf)) mean=$(mean(df_motifs.banzhaf)) eltype=$(eltype(df_motifs.banzhaf))"
        @info "  Contribution: min=$(minimum(df_motifs.contribution)) max=$(maximum(df_motifs.contribution)) mean=$(mean(df_motifs.contribution)) eltype=$(eltype(df_motifs.contribution))"
        @info "  columns_of_interest=$columns_of_interest eltypes=$([eltype(df_motifs[!, c]) for c in columns_of_interest])"
        @info "  Q_THRESHOLD=$Q_THRESHOLD, COUNT_THRESHOLD=$COUNT_THRESHOLD"
        
        # Filter by significance
        df_significant = filter_and_test_significance(
            df_motifs, columns_of_interest, random_coalitions;
            COUNT_THRESHOLD=COUNT_THRESHOLD, Q_THRESHOLD=Q_THRESHOLD);
        
        # Apply final filters
        df_motifs_filtered = apply_final_filters!(
            df_motifs, df_significant, columns_of_interest; 
                d_syms=d_syms, mutegenesis=mutegenesis, 
                    top_and_bot_counts=top_and_bot_counts);
        
        push!(dfs, df_motifs_filtered)
        # push!(df_significants, df_significant)
    end
    return dfs
end
