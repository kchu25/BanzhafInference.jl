
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
        # Same motif type: use count (1 for singleton, 2 for pair)
        n1, n12 = nrow(df_m1), nrow(df_m1m2)
        n_total = n1 + n12
        
        y = Vector{Float64}(undef, n_total)
        x_count = Vector{Float64}(undef, n_total)  # Count of motif occurrences
        x_pair = Vector{Float64}(undef, n_total)   # Indicator for pair
        
        y[1:n1] .= getproperty(df_m1, response_col)
        y[n1+1:end] .= getproperty(df_m1m2, response_col)
        
        x_count[1:n1] .= 1.0; x_count[n1+1:end] .= 2.0  # Count: 1 for singleton, 2 for pair
        x_pair[1:n1] .= 0.0; x_pair[n1+1:end] .= 1.0    # Indicator for pair
        
        data = DataFrame(; y, x_count, x_pair)
        model = lm(@formula(y ~ 0 + x_count + x_pair), data)
        
        # Interaction is 2nd coefficient (no intercept): x_count, x_pair
        ct = coeftable(model)
        stats = (β = coef(model)[2], se = stderror(model)[2], 
                 t = ct.cols[3][2], p = ct.cols[4][2])
    else
        # Different motifs: use both singleton terms
        y, x1, x2, n_total = prepare_regression_data(df_m1, df_m2, df_m1m2, response_col)
        
        data = DataFrame(; y, x1, x2)
        model = lm(@formula(y ~ 0 + x1 + x2 + x1&x2), data)
        
        # Interaction is 3rd coefficient (no intercept): x1, x2, x1&x2
        ct = coeftable(model)
        stats = (β = coef(model)[3], se = stderror(model)[3],
                 t = ct.cols[3][3], p = ct.cols[4][3])
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
        # All same motif: use count (1 for singleton, 3 for triplet)
        n1, n123 = nrow(df_m1), nrow(df_m1m2m3)
        n_total = n1 + n123
        
        y = Float64.(vcat(getproperty(df_m1, response_col), getproperty(df_m1m2m3, response_col)))
        x_count = vcat(ones(n1), fill(3.0, n123))     # Count: 1 for singleton, 3 for triplet
        x_triplet = vcat(zeros(n1), ones(n123))       # Indicator for triplet (interaction term)
        
        data = DataFrame(; y, x_count, x_triplet)
        model = lm(@formula(y ~ 0 + x_count + x_triplet), data)
        
        ct = coeftable(model)
        stats = (β = coef(model)[2], se = stderror(model)[2],
                 t = ct.cols[3][2], p = ct.cols[4][2])
                 
    elseif n_unique == 2
        # Two unique motifs: one appears twice, one appears once
        # Find which motif is duplicated and which is unique
        if m_sym[1] == m_sym[2]
            df_dup, df_uniq = df_m1, df_m3
            dup_count_in_triplet = 2
        elseif m_sym[1] == m_sym[3]
            df_dup, df_uniq = df_m1, df_m2
            dup_count_in_triplet = 2
        else  # m_sym[2] == m_sym[3]
            df_dup, df_uniq = df_m2, df_m1
            dup_count_in_triplet = 2
        end
        
        n_dup, n_uniq, n123 = nrow(df_dup), nrow(df_uniq), nrow(df_m1m2m3)
        n_total = n_dup + n_uniq + n123
        
        y = Float64.(vcat(getproperty(df_dup, response_col), 
                 getproperty(df_uniq, response_col),
                 getproperty(df_m1m2m3, response_col)))
        # x_dup: count of duplicated motif (1 for singleton, 2 for triplet)
        # x_uniq: count of unique motif (1 for singleton, 1 for triplet)
        x_dup = vcat(ones(n_dup), zeros(n_uniq), fill(Float64(dup_count_in_triplet), n123))
        x_uniq = vcat(zeros(n_dup), ones(n_uniq), ones(n123))
        x_triplet = vcat(zeros(n_dup), zeros(n_uniq), ones(n123))  # Indicator for triplet
        
        data = DataFrame(; y, x_dup, x_uniq, x_triplet)
        model = lm(@formula(y ~ 0 + x_dup + x_uniq + x_triplet), data)
        
        ct = coeftable(model)
        stats = (β = coef(model)[3], se = stderror(model)[3],
                 t = ct.cols[3][3], p = ct.cols[4][3])
    else
        # All different motifs: full model
        y, x1, x2, x3, n_total = prepare_regression_data_triplet(
            df_m1, df_m2, df_m3, df_m1m2m3, response_col)
        
        data = DataFrame(; y, x1, x2, x3)
        model = lm(@formula(y ~ 0 + x1 + x2 + x3 + x1&x2&x3), data)
        
        # Interaction is 4th coefficient (no intercept): x1, x2, x3, x1&x2&x3
        ct = coeftable(model)
        stats = (β = coef(model)[4], se = stderror(model)[4],
                 t = ct.cols[3][4], p = ct.cols[4][4])
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

Create dictionaries mapping filter pairs to formatted summary strings and quantitative values.
Returns (summary_dict_str, summary_dict) where:
- summary_dict_str: Dict with key (m1, m2) => "β_interaction: X.XXX, se: X.XXX, p-value: X.XXX"
- summary_dict: Dict with key (m1, m2) => Dict("beta_interaction" => X, "se" => X, "p_value" => X)
"""
function create_interaction_summary_dict(interaction_results; use_adjusted_p=true)
    p_col = use_adjusted_p && hasproperty(interaction_results, :p_adjusted) ? :p_adjusted : :p_value
    
    summary_dict_str = Dict{NamedTuple{(:m1, :m2), Tuple{Int, Int}}, String}()
    summary_dict = Dict{NamedTuple{(:m1, :m2), Tuple{Int, Int}}, Dict{String, Float64}}()
    
    for row in eachrow(interaction_results)
        key = (m1=row.m1, m2=row.m2)
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
    create_triplet_interaction_summary_dict(interaction_results; use_adjusted_p=true)

Create dictionaries mapping filter triplets to formatted summary strings and quantitative values.
Returns (summary_dict_str, summary_dict) where:
- summary_dict_str: Dict with key (m1, m2, m3) => "β_interaction: X.XXX, se: X.XXX, p-value: X.XXX"
- summary_dict: Dict with key (m1, m2, m3) => Dict("beta_interaction" => X, "se" => X, "p_value" => X)
"""
function create_triplet_interaction_summary_dict(interaction_results; use_adjusted_p=true)
    p_col = use_adjusted_p && hasproperty(interaction_results, :p_adjusted) ? :p_adjusted : :p_value
    
    summary_dict_str = Dict{NamedTuple{(:m1, :m2, :m3), Tuple{Int, Int, Int}}, String}()
    summary_dict = Dict{NamedTuple{(:m1, :m2, :m3), Tuple{Int, Int, Int}}, Dict{String, Float64}}()
    
    for row in eachrow(interaction_results)
        key = (m1=row.m1, m2=row.m2, m3=row.m3)
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

function obtain_interaction_results(contributions_df_filtered, dfs)
    # Pair interactions
    m_syms_pair = BanzhafInference.m_symbols(2)
    gdf_single = groupby(contributions_df_filtered, :filter_index)
    gdf_pair = groupby(dfs[1], m_syms_pair)

    results_pair = [test_interaction(m_sym, ind, gdf_single, gdf_pair) 
                    for (m_sym, ind) in gdf_pair.keymap]
    interaction_results_pair = DataFrame(results_pair)
    interaction_results_pair_fdr = apply_fdr_correction(interaction_results_pair, alpha=0.05)
    interaction_summary_pair_str, interaction_summary_pair = create_interaction_summary_dict(interaction_results_pair_fdr)

    # Triplet interactions
    m_syms_triplet = BanzhafInference.m_symbols(3)
    gdf_triplet = groupby(dfs[2], m_syms_triplet)

    results_triplet = [test_triplet_interaction(m_sym, ind, gdf_single, gdf_triplet) 
                       for (m_sym, ind) in gdf_triplet.keymap]
    interaction_results_triplet = DataFrame(results_triplet)
    interaction_results_triplet_fdr = apply_fdr_correction(interaction_results_triplet, alpha=0.05)
    interaction_summary_triplet_str, interaction_summary_triplet = create_triplet_interaction_summary_dict(interaction_results_triplet_fdr)

    return [interaction_summary_pair_str, interaction_summary_triplet_str], 
           [interaction_summary_pair, interaction_summary_triplet]
end
