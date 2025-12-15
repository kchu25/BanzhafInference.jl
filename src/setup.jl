function banzhaf_setups(
    m, 
    contributions_df; 
    seed=42,
    max_interaction_order=3, 
    train_stats=nothing, 
    n_coalitions_per_datapoint=20,
    min_coalition_size=2,
    num_samples_per_coalition=100, 
    scale_back=false
)
    # Setup enumeration config
    ec = BanzhafInference.MotifEnumerationConfig(
        max_interaction_order=max_interaction_order, 
        filter_len=m.hp.pfm_len, 
        seed=seed
    )

    # Setup Banzhaf algorithm config
    final_nonlinearity = BanzhafInference.FunctorWrapper(m.final_nonlinearity)
    scale_back_function = (scale_back && !isnothing(train_stats)) ?
        BanzhafInference.FunctorWrapper(train_stats.scale_back_functor) :
        BanzhafInference.FunctorWrapper(x->x)
        
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
