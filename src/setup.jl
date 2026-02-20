function obtain_contributions_df(data, m, processor, train_stats; threshold_stats=nothing, predict_position=1)
    contributions, _ = 
        BanzhafInference.compute_and_filter_contributions(data, m, processor; 
            train_stats=train_stats, 
            threshold_stats=threshold_stats, 
            predict_position=predict_position,
            operate_on_gpu=true
            );
    contributions_df = DataFrame(contributions);
    return contributions_df
end


function _make_scale_back_function(scale_back, train_stats, multi_output, predict_position)
    if scale_back && !isnothing(train_stats)
        # TODO extend this to multi-output case
        if multi_output
            @assert !isnothing(predict_position) "predict_position must be specified for multi-output models when scale_back=true"
            BanzhafInference.FunctorWrapper(train_stats.scale_back_functor.functors[predict_position])
        else
            BanzhafInference.FunctorWrapper(train_stats.scale_back_functor)
        end
    else
        BanzhafInference.FunctorWrapper(x->x)
    end
end

function banzhaf_setups(
    m, 
    contributions_df; 
    seed=42,
    max_interaction_order=MAX_INTERACTION_ORDER, 
    train_stats=nothing, 
    n_coalitions_per_datapoint=N_COALITION_PER_PT,
    min_coalition_size=MIN_COALITION_SIZE,
    num_samples_per_coalition=NUM_SAMPLES_PER_COALITION, 
    scale_back=false,
    predict_position=nothing,
    multi_output=false
)
    # Adjust n_coalitions_per_datapoint to keep total random coalitions ≤ 10000
    num_data_pts = maximum(contributions_df.data_pt_index)
    n_coalitions_per_datapoint = min(
        n_coalitions_per_datapoint, 
        max(1, div(MAX_BG_DATA_PTs, num_data_pts))
    )
    
    # Setup enumeration config
    ec = BanzhafInference.MotifEnumerationConfig(
        max_interaction_order=max_interaction_order, 
        filter_len=m.hp.pfm_len, 
        seed=seed
    )

    # Setup Banzhaf algorithm config
    final_nonlinearity = BanzhafInference.FunctorWrapper(m.final_nonlinearity)

    scale_back_function = _make_scale_back_function(scale_back, train_stats, multi_output, predict_position)
        
    ac = BanzhafInference.BanzhafAlgorithmConfig(
        final_nonlinearity=final_nonlinearity,
        scale_back_function=scale_back_function
    )

    # Setup MotifDataCache
    # Group contributions by data point (each group = all contributions for one sequence)
    contributions_by_data_pt = groupby(contributions_df, :data_pt_index)

    # Extract unique data point indices
    unique_data_pt_indices = [
        group_key.data_pt_index for group_key in keys(contributions_by_data_pt)
    ]

    # Create reverse lookup: data_pt_index -> group position
    data_pt_to_group_idx = Dict(
        data_pt_idx => group_num 
        for (group_num, data_pt_idx) in enumerate(unique_data_pt_indices)
    )

    mdc = BanzhafInference.MotifDataCache(
        contributions_by_data_pt=contributions_by_data_pt,
        unique_data_pt_indices=unique_data_pt_indices,
        data_pt_to_group_idx=data_pt_to_group_idx
    )

    # Setup background config
    bc = BanzhafInference.BackgroundConfig(
        n_coalitions_per_datapoint=n_coalitions_per_datapoint, 
        min_coalition_size=min_coalition_size,
        num_samples_per_coalition=num_samples_per_coalition
    )                             

    return ec, ac, mdc, bc
end
