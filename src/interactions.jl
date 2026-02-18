
"""
    test_kway_interaction(m_sym, ind, gdf_single, gdf_combo; response_col=:banzhaf)

Test for k-way interaction among `k` filters using linear regression (no intercept).

For k motifs `(m_sym[1], ..., m_sym[k])` with `u` unique motif types, the model is:

    y ~ 0 + x_1 + x_2 + ... + x_u + x_interaction

where:
- Each `x_i` encodes the **count** of the i-th unique motif (1 for its singleton, count in k-tuple for combo)
- `x_interaction` is an indicator (0 for singletons, 1 for the k-tuple)

The interaction coefficient captures deviation from additivity:
    Banzhaf(combo) ≠ Σ count_i × Banzhaf(singleton_i)
"""
function test_kway_interaction(m_sym, ind, gdf_single, gdf_combo; response_col=:banzhaf)
    k = length(m_sym)
    motifs = [m_sym[i] for i in 1:k]
    
    # Count occurrences of each unique motif in the k-tuple
    unique_motifs = unique(motifs)
    n_unique = length(unique_motifs)
    motif_counts = Dict(m => count(==(m), motifs) for m in unique_motifs)
    
    # Collect singleton DataFrames (one per unique motif)
    singleton_dfs = [gdf_single[(filter_index = m,)] for m in unique_motifs]
    singleton_sizes = [nrow(df) for df in singleton_dfs]
    
    # Combo DataFrame
    df_combo = gdf_combo[ind]
    n_combo = nrow(df_combo)
    n_total = sum(singleton_sizes) + n_combo
    
    # Build y vector
    y = Vector{Float64}(undef, n_total)
    offset = 0
    for (i, df) in enumerate(singleton_dfs)
        n_i = singleton_sizes[i]
        y[offset+1:offset+n_i] .= getproperty(df, response_col)
        offset += n_i
    end
    y[offset+1:end] .= getproperty(df_combo, response_col)
    
    # Build design matrix: n_unique count columns + 1 interaction indicator
    X = zeros(Float64, n_total, n_unique + 1)
    
    offset = 0
    for (i, df) in enumerate(singleton_dfs)
        n_i = singleton_sizes[i]
        # Singleton rows: count = 1 for own column, 0 for others
        X[offset+1:offset+n_i, i] .= 1.0
        offset += n_i
    end
    # Combo rows: each unique motif gets its count, interaction = 1
    for (i, m) in enumerate(unique_motifs)
        X[offset+1:end, i] .= Float64(motif_counts[m])
    end
    X[offset+1:end, n_unique+1] .= 1.0  # interaction indicator
    
    # Build DataFrame with column names x1, x2, ..., x_u, x_interaction
    col_names = [Symbol("x$i") for i in 1:n_unique]
    push!(col_names, :x_interaction)
    
    data = DataFrame(X, col_names)
    data.y = y
    
    # Build formula programmatically: y ~ 0 + x1 + x2 + ... + x_interaction
    rhs_terms = term.(col_names)
    f = term(:y) ~ ConstantTerm(0) + foldl(+, rhs_terms)
    
    model = lm(f, data)
    
    # Interaction is the last coefficient
    n_coefs = length(coef(model))
    ct = coeftable(model)
    stats = (
        β = coef(model)[n_coefs], 
        se = stderror(model)[n_coefs],
        t = ct.cols[3][n_coefs], 
        p = ct.cols[4][n_coefs]
    )
    
    # Build result NamedTuple with m1, m2, ..., mk fields
    m_keys = Tuple(Symbol("m$i") for i in 1:k)
    m_vals = Tuple(m_sym[i] for i in 1:k)
    stat_keys = (:β_interaction, :se, :t_stat, :p_value, :n_obs)
    stat_vals = (stats.β, stats.se, stats.t, stats.p, n_total)
    
    return NamedTuple{(m_keys..., stat_keys...)}((m_vals..., stat_vals...))
end

"""
    apply_fdr_correction(interaction_results; method=BenjaminiHochberg(), alpha=1e-10)

Apply FDR correction to interaction p-values and return only significant results.
Returns DataFrame with adjusted p-values, sorted by significance.
"""
function apply_fdr_correction(interaction_results; method=BenjaminiHochberg(), alpha=1e-10)
    # Handle empty results
    if nrow(interaction_results) == 0
        return interaction_results
    end
    
    # Remove NaN p-values
    valid_results = filter(row -> !isnan(row.p_value), interaction_results)
    
    # Handle case where all p-values are NaN
    if nrow(valid_results) == 0
        return valid_results
    end
    
    # Apply FDR correction
    adjusted_pvals = adjust(valid_results.p_value, method)
    
    # Add to results
    valid_results[!, :p_adjusted] = adjusted_pvals
    
    # Filter to only significant results
    significant_results = filter(row -> row.p_adjusted < alpha, valid_results)
    
    return sort(significant_results, :p_adjusted)
end

"""
    create_summary_dicts(interaction_results, k::Int; use_adjusted_p=true)

Create dictionaries mapping filter k-tuples to formatted summary strings and quantitative values.
Returns (summary_dict_str, summary_dict).

Key type is a NamedTuple like (m1=Int, m2=Int, ..., mk=Int).
"""
function create_summary_dicts(interaction_results, k::Int; use_adjusted_p=true)
    p_col = use_adjusted_p && hasproperty(interaction_results, :p_adjusted) ? :p_adjusted : :p_value
    
    m_fields = Tuple(Symbol("m$i") for i in 1:k)
    KeyType = NamedTuple{m_fields, NTuple{k, Int}}
    
    summary_dict_str = Dict{KeyType, String}()
    summary_dict = Dict{KeyType, Dict{String, Float64}}()
    
    for row in eachrow(interaction_results)
        key_vals = Tuple(getproperty(row, Symbol("m$i")) for i in 1:k)
        key = NamedTuple{m_fields}(key_vals)
        p_val = getproperty(row, p_col)
        
        summary_dict_str[key] = @sprintf("β_interaction: %+.2f, se: %.4f, p-value: %.2e", 
                         row.β_interaction, row.se, p_val)
        
        summary_dict[key] = Dict(
            "beta_interaction" => row.β_interaction,
            "se" => row.se,
            "p_value" => p_val
        )
    end
    
    return summary_dict_str, summary_dict
end

"""
    obtain_interaction_results(contributions_df_filtered, dfs; alpha=1e-10)

Test interactions for all k-tuple sizes (k=2, 3, ..., length(dfs)+1).
`dfs` is a vector where `dfs[i]` contains the DataFrame of (i+1)-tuples.

Returns `(summary_strs, summary_quants)` where each is a vector indexed by 
position in `dfs` (i.e., `summary_strs[1]` is pairs, `summary_strs[2]` is triplets, etc.).
"""
function obtain_interaction_results(contributions_df_filtered, dfs; alpha=1e-10)
    gdf_single = groupby(contributions_df_filtered, :filter_index)
    
    summary_strs = []
    summary_quants = []
    
    for (idx, df_k) in enumerate(dfs)
        k = idx + 1  # dfs[1] = pairs (k=2), dfs[2] = triplets (k=3), etc.
        m_syms = BanzhafInference.m_symbols(k)
        gdf_combo = groupby(df_k, m_syms)
        
        results = [test_kway_interaction(m_sym, ind, gdf_single, gdf_combo) 
                   for (m_sym, ind) in gdf_combo.keymap]
        
        interaction_results = DataFrame(results)
        interaction_results_fdr = apply_fdr_correction(interaction_results; alpha=alpha)
        dict_str, dict_quant = create_summary_dicts(interaction_results_fdr, k)
        
        push!(summary_strs, dict_str)
        push!(summary_quants, dict_quant)
    end
    
    return summary_strs, summary_quants
end
