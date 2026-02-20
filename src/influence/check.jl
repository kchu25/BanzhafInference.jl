
"""
    obtain_predictions(model, contributions, data_load; predict_position=nothing, operate_on_gpu=true)

Compute predictions from both contributions and the model.

# Arguments
- `model`: The model to evaluate
- `contributions`: Vector of contributions
- `data_load`: Data loader
- `predict_position`: Position to predict (optional)
- `operate_on_gpu`: Whether to use GPU for model inference

# Returns
- `(model_predictions, contribs2predictions)`: Tuple of prediction vectors
"""
function obtain_predictions(model, contributions, data_load; 
    predict_position=nothing, operate_on_gpu=true
    )

    @assert isa(model.hp.batch_size, Integer) "model must have property `batchsize`"

    N = contributions[end].data_pt_index
    contribs2predictions = zeros(FloatType, (N,))
    model_predictions = zeros(FloatType, (N,))
    
    # Compute predictions from contributions
    @info "Obtaining predictions from contributions ..."
    @showprogress for c in contributions
        contribs2predictions[c.data_pt_index] += c.contribution
    end
    try
        contribs2predictions = model.final_nonlinearity.(contribs2predictions)
    catch e
        @warn "model type $(typeof(model)) does not support final_nonlinearity. Use identity function instead."
        contribs2predictions = contribs2predictions
    end

    # Run the model to get the predictions
    @assert data_load.shuffle == false "dataload must have shuffle = false"
    @assert model.hp.batch_size == data_load.batchsize "model.batchsize must equal data_load.batchsize"
    data_pt_offset = 0
    @info "Obtaining predictions from model ..."
    @showprogress for (X, _) in data_load
        X = operate_on_gpu ? X |> gpu : X |> cpu
        predictions = model(X; predict_position=predict_position)
        predictions = predictions |> cpu
        start_index = data_pt_offset + 1
        end_index = min(N, data_pt_offset + data_load.batchsize)
        # Handle case where predictions might be larger than the range (last batch)
        batch_size_actual = end_index - start_index + 1
        model_predictions[start_index:end_index] .= view(predictions, 1:batch_size_actual)
        data_pt_offset += data_load.batchsize
    end

    return model_predictions, contribs2predictions
end

"""Compute R² coefficient"""
function _compute_r2(y_true::AbstractVector{T}, y_pred::AbstractVector{T}) where T<:AbstractFloat
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return T(1) - ss_res / ss_tot
end

"""
    validate_predictions(model_predictions, contribs2predictions)

Validate that predictions from contributions match model predictions.

# Arguments
- `model_predictions`: Predictions directly from the model
- `contribs2predictions`: Predictions computed from contributions

"""
function validate_predictions(model_predictions, contribs2predictions)
    r2 = _compute_r2(model_predictions, contribs2predictions)
    @info "R² (contributions vs. model): $(round(r2, digits=3))"
end

"""
    sanity_check(model, contributions, data_load; predict_position=nothing, operate_on_gpu=true)

Perform a sanity check by comparing predictions from contributions with direct model predictions.

This is a wrapper function that combines `obtain_predictions` and `validate_predictions`.

# Arguments
- `model`: The model to evaluate
- `contributions`: Vector of contributions
- `data_load`: Data loader
- `predict_position`: Position to predict (optional)
- `operate_on_gpu`: Whether to use GPU for model inference

# Returns
- `pass_condition`: Boolean indicating if validation passed (>99.5% match)
"""
function sanity_check(model, contributions, data_load; 
    predict_position=1, operate_on_gpu=true, train_stats=nothing
    )
    
    model_predictions, contribs2predictions = obtain_predictions(
        model, contributions, data_load;
        predict_position=predict_position,
        operate_on_gpu=operate_on_gpu
    )
    validate_predictions(model_predictions, contribs2predictions)

    if !isnothing(train_stats)
        labels = apply_normalization(data_load.data[2], train_stats);

        if isa(labels, Matrix)
            labels = @view labels[predict_position, :]
        end

        r2_orig = BanzhafInference.r2_score(labels, model_predictions)
        r2 = BanzhafInference.r2_score(labels, contribs2predictions)
        @info "R² with true labels vs model predictions: $(round(r2_orig, digits=3))"
        @info "R² with true labels vs contributions predictions: $(round(r2, digits=3))"
        @info "Improvement over model R²: $(round(r2 - r2_orig, digits=3)*100)%"
    else
        @warn "train_stats not provided; skipping R² with true labels."
    end
    
end
