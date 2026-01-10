
"""
    prepare_regression_data(df_m1, df_m2, df_m1m2, response_col::Symbol)

Efficiently construct regression arrays from singleton and pair dataframes.
Returns (y, x1, x2) vectors for interaction testing.
"""
function prepare_regression_data(df_m1, df_m2, df_m1m2, response_col::Symbol)
    n1, n2, n12 = nrow(df_m1), nrow(df_m2), nrow(df_m1m2)
    n_total = n1 + n2 + n12
    
    y = Vector{Float64}(undef, n_total)
    x1 = Vector{Float64}(undef, n_total)
    x2 = Vector{Float64}(undef, n_total)
    
    # Fill response variable
    y[1:n1] .= getproperty(df_m1, response_col)
    y[n1+1:n1+n2] .= getproperty(df_m2, response_col)
    y[n1+n2+1:end] .= getproperty(df_m1m2, response_col)
    
    # Fill design matrix: (1,0), (0,1), (1,1)
    x1[1:n1] .= 1.0
    x1[n1+1:n1+n2] .= 0.0
    x1[n1+n2+1:end] .= 1.0
    
    x2[1:n1] .= 0.0
    x2[n1+1:n1+n2] .= 1.0
    x2[n1+n2+1:end] .= 1.0
    
    return y, x1, x2, n_total
end

"""
    extract_interaction_stats(model)

Extract interaction term statistics from fitted linear model.
Returns named tuple with coefficient, standard error, t-statistic, and p-value.
"""
function extract_interaction_stats(model)
    ct = coeftable(model)
    return (
        β = coef(model)[4],
        se = stderror(model)[4],
        t = ct.cols[3][4],
        p = ct.cols[4][4]
    )
end

"""
    test_interaction(m_sym, ind, gdf_single, gdf_pair; response_col=:banzhaf)

Test for epistatic interaction between two filters using linear regression.
Fits model: y ~ x1 + x2 + x1&x2, where interaction term captures non-additivity.
"""
function test_interaction(m_sym, ind, gdf_single, gdf_pair; response_col=:banzhaf)
    # Extract data subsets
    df_m1 = gdf_single[(filter_index = m_sym[1],)]
    df_m2 = gdf_single[(filter_index = m_sym[2],)]
    df_m1m2 = gdf_pair[ind]
    
    # Prepare regression arrays
    y, x1, x2, n_total = prepare_regression_data(df_m1, df_m2, df_m1m2, response_col)
    
    # Fit model
    data = DataFrame(; y, x1, x2)
    model = lm(@formula(y ~ x1 + x2 + x1&x2), data)
    
    # Extract statistics
    stats = extract_interaction_stats(model)
    
    return (
        m1 = m_sym[1],
        m2 = m_sym[2],
        β_interaction = stats.β,
        se = stats.se,
        t_stat = stats.t,
        p_value = stats.p,
        n_obs = n_total
    )
end

"""
    apply_fdr_correction(interaction_results; method=BenjaminiHochberg(), alpha=0.05)

Apply FDR correction to interaction p-values.
Returns DataFrame with adjusted p-values and significance flags.
"""
function apply_fdr_correction(interaction_results; method=BenjaminiHochberg(), alpha=0.05)
    # Remove NaN p-values
    valid_results = filter(row -> !isnan(row.p_value), interaction_results)
    
    # Apply FDR correction
    adjusted_pvals = adjust(valid_results.p_value, method)
    
    # Add to results
    valid_results[!, :p_adjusted] = adjusted_pvals
    valid_results[!, :significant] = adjusted_pvals .< alpha
    
    return sort(valid_results, :p_adjusted)
end

"""
    create_interaction_summary_dict(interaction_results; use_adjusted_p=true)

Create a dictionary mapping filter pairs to formatted summary strings.
Returns Dict with key (m1, m2) => "β_interaction: X.XXX, se: X.XXX, p-value: X.XXX"
"""
function create_interaction_summary_dict(interaction_results; use_adjusted_p=true)
    p_col = use_adjusted_p && hasproperty(interaction_results, :p_adjusted) ? :p_adjusted : :p_value
    
    summary_dict = Dict{NamedTuple{(:m1, :m2), Tuple{Int, Int}}, String}()
    
    for row in eachrow(interaction_results)
        key = (m1=row.m1, m2=row.m2)
        p_val = getproperty(row, p_col)
        
        value = @sprintf("β_interaction: %+.2f, se: %.4f, p-value: %.2e", 
                         row.β_interaction, row.se, p_val)
        
        summary_dict[key] = value
    end
    
    return summary_dict
end

function obtain_interaction_results(contributions_df_filtered, dfs)
    m_syms = BanzhafInference.m_symbols(2);
    gdf_single = groupby(contributions_df_filtered, :filter_index)
    gdf_pair = groupby(dfs[1], m_syms)

    # Run for all pairs
    results = [test_interaction(m_sym, ind, gdf_single, gdf_pair) 
            for (m_sym, ind) in gdf_pair.keymap]

    interaction_results = DataFrame(results)

    # Apply FDR correction
    interaction_results_fdr = apply_fdr_correction(interaction_results, alpha=0.05)

    # Create summary dictionary: (m1, m2) => summary string
    interaction_summary = create_interaction_summary_dict(interaction_results_fdr)

    return interaction_summary
end
