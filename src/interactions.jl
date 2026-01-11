
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
    prepare_regression_data_triplet(df_m1, df_m2, df_m3, df_m1m2m3, response_col::Symbol)

Construct regression arrays from singleton and triplet dataframes.
Returns (y, x1, x2, x3) vectors for triplet interaction testing.
"""
function prepare_regression_data_triplet(df_m1, df_m2, df_m3, df_m1m2m3, response_col::Symbol)
    n1, n2, n3, n123 = nrow(df_m1), nrow(df_m2), nrow(df_m3), nrow(df_m1m2m3)
    n_total = n1 + n2 + n3 + n123
    
    y = Vector{Float64}(undef, n_total)
    x1 = Vector{Float64}(undef, n_total)
    x2 = Vector{Float64}(undef, n_total)
    x3 = Vector{Float64}(undef, n_total)
    
    # Fill response variable
    offset = 0
    y[offset+1:offset+n1] .= getproperty(df_m1, response_col)
    offset += n1
    y[offset+1:offset+n2] .= getproperty(df_m2, response_col)
    offset += n2
    y[offset+1:offset+n3] .= getproperty(df_m3, response_col)
    offset += n3
    y[offset+1:end] .= getproperty(df_m1m2m3, response_col)
    
    # Fill design matrix: (1,0,0), (0,1,0), (0,0,1), (1,1,1)
    x1[1:n1] .= 1.0; x2[1:n1] .= 0.0; x3[1:n1] .= 0.0
    x1[n1+1:n1+n2] .= 0.0; x2[n1+1:n1+n2] .= 1.0; x3[n1+1:n1+n2] .= 0.0
    x1[n1+n2+1:n1+n2+n3] .= 0.0; x2[n1+n2+1:n1+n2+n3] .= 0.0; x3[n1+n2+1:n1+n2+n3] .= 1.0
    x1[n1+n2+n3+1:end] .= 1.0; x2[n1+n2+n3+1:end] .= 1.0; x3[n1+n2+n3+1:end] .= 1.0
    
    return y, x1, x2, x3, n_total
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
    extract_triplet_interaction_stats(model)

Extract 3-way interaction term statistics from fitted linear model.
Returns named tuple with coefficient, standard error, t-statistic, and p-value.
"""
function extract_triplet_interaction_stats(model)
    ct = coeftable(model)
    # 3-way interaction is the 5th coefficient: intercept, x1, x2, x3, x1&x2&x3
    return (
        β = coef(model)[5],
        se = stderror(model)[5],
        t = ct.cols[3][5],
        p = ct.cols[4][5]
    )
end

"""
    test_interaction(m_sym, ind, gdf_single, gdf_pair; response_col=:banzhaf)

Test for epistatic interaction between two filters using linear regression.
Fits model: y ~ x1 + x2 + x1&x2 (or y ~ x1 + x1&x2 if m1==m2).
"""
function test_interaction(m_sym, ind, gdf_single, gdf_pair; response_col=:banzhaf)
    # Extract data subsets
    df_m1 = gdf_single[(filter_index = m_sym[1],)]
    df_m2 = gdf_single[(filter_index = m_sym[2],)]
    df_m1m2 = gdf_pair[ind]
    
    # Check for duplicate motifs
    if m_sym[1] == m_sym[2]
        # Same motif type: only use one singleton term
        n1, n12 = nrow(df_m1), nrow(df_m1m2)
        n_total = n1 + n12
        
        y = Vector{Float64}(undef, n_total)
        x1 = Vector{Float64}(undef, n_total)
        x2 = Vector{Float64}(undef, n_total)
        
        y[1:n1] .= getproperty(df_m1, response_col)
        y[n1+1:end] .= getproperty(df_m1m2, response_col)
        
        x1[1:n1] .= 1.0; x1[n1+1:end] .= 1.0
        x2[1:n1] .= 0.0; x2[n1+1:end] .= 1.0  # x2 only distinguishes pair
        
        data = DataFrame(; y, x1, x2)
        model = lm(@formula(y ~ x1 + x1&x2), data)
        
        # Interaction is 3rd coefficient: intercept, x1, x1&x2
        ct = coeftable(model)
        stats = (β = coef(model)[3], se = stderror(model)[3], 
                 t = ct.cols[3][3], p = ct.cols[4][3])
    else
        # Different motifs: use both singleton terms
        y, x1, x2, n_total = prepare_regression_data(df_m1, df_m2, df_m1m2, response_col)
        
        data = DataFrame(; y, x1, x2)
        model = lm(@formula(y ~ x1 + x2 + x1&x2), data)
        stats = extract_interaction_stats(model)
    end
    
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
    test_triplet_interaction(m_sym, ind, gdf_single, gdf_triplet; response_col=:banzhaf)

Test for 3-way interaction among three filters using linear regression.
Adjusts formula to remove duplicate singleton terms when motifs are identical.
"""
function test_triplet_interaction(m_sym, ind, gdf_single, gdf_triplet; response_col=:banzhaf)
    # Extract data subsets
    df_m1 = gdf_single[(filter_index = m_sym[1],)]
    df_m2 = gdf_single[(filter_index = m_sym[2],)]
    df_m3 = gdf_single[(filter_index = m_sym[3],)]
    df_m1m2m3 = gdf_triplet[ind]
    
    # Determine unique motifs
    unique_motifs = unique([m_sym[1], m_sym[2], m_sym[3]])
    n_unique = length(unique_motifs)
    
    if n_unique == 1
        # All same motif: y ~ x1 + x1&x2&x3
        n1, n123 = nrow(df_m1), nrow(df_m1m2m3)
        n_total = n1 + n123
        
        y = Float64.(vcat(getproperty(df_m1, response_col), getproperty(df_m1m2m3, response_col)))
        x1 = vcat(ones(n1), ones(n123))
        x2 = vcat(zeros(n1), ones(n123))
        x3 = vcat(zeros(n1), ones(n123))
        
        data = DataFrame(; y, x1, x2, x3)
        model = lm(@formula(y ~ x1 + x1&x2&x3), data)
        
        ct = coeftable(model)
        stats = (β = coef(model)[3], se = stderror(model)[3],
                 t = ct.cols[3][3], p = ct.cols[4][3])
                 
    elseif n_unique == 2
        # Two unique motifs: include only 2 singleton terms
        # Find which motif is duplicated
        if m_sym[1] == m_sym[2]
            dfs_unique = [df_m1, df_m3]
        elseif m_sym[1] == m_sym[3]
            dfs_unique = [df_m1, df_m2]
        else  # m_sym[2] == m_sym[3]
            dfs_unique = [df_m1, df_m2]
        end
        
        n1, n2, n123 = nrow(dfs_unique[1]), nrow(dfs_unique[2]), nrow(df_m1m2m3)
        n_total = n1 + n2 + n123
        
        y = Float64.(vcat(getproperty(dfs_unique[1], response_col), 
                 getproperty(dfs_unique[2], response_col),
                 getproperty(df_m1m2m3, response_col)))
        x1 = vcat(ones(n1), zeros(n2), ones(n123))
        x2 = vcat(zeros(n1), ones(n2), ones(n123))
        x3 = vcat(zeros(n1), zeros(n2), ones(n123))
        
        data = DataFrame(; y, x1, x2, x3)
        model = lm(@formula(y ~ x1 + x2 + x1&x2&x3), data)
        
        ct = coeftable(model)
        stats = (β = coef(model)[4], se = stderror(model)[4],
                 t = ct.cols[3][4], p = ct.cols[4][4])
    else
        # All different motifs: full model
        y, x1, x2, x3, n_total = prepare_regression_data_triplet(
            df_m1, df_m2, df_m3, df_m1m2m3, response_col)
        
        data = DataFrame(; y, x1, x2, x3)
        model = lm(@formula(y ~ x1 + x2 + x3 + x1&x2&x3), data)
        stats = extract_triplet_interaction_stats(model)
    end
    
    return (
        m1 = m_sym[1],
        m2 = m_sym[2],
        m3 = m_sym[3],
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

"""
    create_triplet_interaction_summary_dict(interaction_results; use_adjusted_p=true)

Create a dictionary mapping filter triplets to formatted summary strings.
Returns Dict with key (m1, m2, m3) => "β_interaction: X.XXX, se: X.XXX, p-value: X.XXX"
"""
function create_triplet_interaction_summary_dict(interaction_results; use_adjusted_p=true)
    p_col = use_adjusted_p && hasproperty(interaction_results, :p_adjusted) ? :p_adjusted : :p_value
    
    summary_dict = Dict{NamedTuple{(:m1, :m2, :m3), Tuple{Int, Int, Int}}, String}()
    
    for row in eachrow(interaction_results)
        key = (m1=row.m1, m2=row.m2, m3=row.m3)
        p_val = getproperty(row, p_col)
        
        value = @sprintf("β_interaction: %+.2f, se: %.4f, p-value: %.2e", 
                         row.β_interaction, row.se, p_val)
        
        summary_dict[key] = value
    end
    
    return summary_dict
end

function obtain_interaction_results(contributions_df_filtered, dfs)
    # Pair interactions
    m_syms_pair = BanzhafInference.m_symbols(2)
    gdf_single = groupby(contributions_df_filtered, :filter_index)
    gdf_pair = groupby(dfs[1], m_syms_pair)

    results_pair = [test_interaction(m_sym, ind, gdf_single, gdf_pair) 
                    for (m_sym, ind) in gdf_pair.keymap]
    interaction_results_pair = DataFrame(results_pair)
    interaction_results_pair_fdr = apply_fdr_correction(interaction_results_pair, alpha=0.05)
    interaction_summary_pair = create_interaction_summary_dict(interaction_results_pair_fdr)

    # Triplet interactions
    m_syms_triplet = BanzhafInference.m_symbols(3)
    gdf_triplet = groupby(dfs[2], m_syms_triplet)

    results_triplet = [test_triplet_interaction(m_sym, ind, gdf_single, gdf_triplet) 
                       for (m_sym, ind) in gdf_triplet.keymap]
    interaction_results_triplet = DataFrame(results_triplet)
    interaction_results_triplet_fdr = apply_fdr_correction(interaction_results_triplet, alpha=0.05)
    interaction_summary_triplet = create_triplet_interaction_summary_dict(interaction_results_triplet_fdr)

    return [interaction_summary_pair, interaction_summary_triplet]
end
