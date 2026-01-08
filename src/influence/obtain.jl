
function obtain_grad_from_model(linear_sum_func, X; predict_position=nothing)
    try
        grad = gradient(x -> sum(linear_sum_func(x; predict_position=predict_position)), X)
        return grad[1]
    catch
        error("Error when calling linear_sum_func. Make sure your model supports " *
              "linear_sum=true and predict_position as keyword arguments.")
    end
end

"""
    _push_contributions_impl!(tuple_factory, X, grad_product, contributions)

Helper function to push nonzero contributions from `grad_product` into `contributions` 
using a user-provided tuple factory.
"""
function _push_contributions_impl!(tuple_factory, X, grad_product, contributions)
    @assert isa(grad_product, Array{eltype(grad_product)}) "grad_product must be on CPU"
    X_cpu = X |> cpu
    
    for c in findall(!iszero, grad_product)
        push!(contributions, tuple_factory(c, grad_product[c], X_cpu[c]))
    end
    return contributions
end

"""
push_to_contributions!(grad_product, data_pt_offset, contributions)

Multiple dispatchers for different contribution types. For each type, see the corresponding tuple structure in types.jl
"""
push_to_contributions!(grad_product, X, data_pt_offset, contributions::Vector{BioSequenceContributionMonomer}) =
    _push_contributions_impl!(X, grad_product, contributions) do c, contrib, mag
        # grad_product have shape (alphabet, position, 1, data_pt_index)
        (data_pt_index=c[4] + data_pt_offset, alphabet_index=c[1], position=c[2], contribution=contrib)
    end

function push_to_contributions!(grad_product, code, data_pt_offset, contributions::Vector{BioSequenceContributionCode})
    code_reshaped = size(code, 1) == 1 && size(grad_product, 1) == 1 ? 
        tail_reshape(code) : dropdims(code; dims=3)
    grad_reshaped = size(code, 1) == 1 && size(grad_product, 1) == 1 ? 
        tail_reshape(grad_product) : dropdims(grad_product; dims=3)
    
    _push_contributions_impl!(code_reshaped, grad_reshaped, contributions) do c, contrib, mag
        (data_pt_index=c[3] + data_pt_offset, filter_index=c[2], position=c[1], 
         contribution=contrib, mag=mag)
    end
end

push_to_contributions!(grad_product, X, data_pt_offset, contributions::Vector{OrdinaryFeature}) =
    _push_contributions_impl!(X, grad_product, contributions) do c, contrib, mag
        (data_pt_index=c[2] + data_pt_offset, feature_index=c[1], contribution=contrib)
    end
    
function compute_contributions(
    model,
    data_load;
    pseudo_model=nothing,
    contribution_type=BioSequenceContributionCode,
    predict_position=1,
    operate_on_gpu=true,
    threshold_stats=nothing
)
    @info "Obtaining contributions ..."
    @assert !data_load.shuffle "dataload must have shuffle = false"
    @assert isa(model.hp.batch_size, Integer) "model must have property `batchsize`"
    @assert model.hp.batch_size == data_load.batchsize "model.batchsize must equal data_load.batchsize"

    contributions = Vector{contribution_type}()
    data_pt_offset = 0
    
    @showprogress for (X, _) in data_load
        X = operate_on_gpu ? (X |> gpu) : (X |> cpu)

        input, linear_sum_func = if contribution_type == BioSequenceContributionCode
            model.code(X), model.predict_up_to_final_nonlinearity
        else
            X, model.linear_sum
        end

        if !isnothing(pseudo_model)
            pseudo_model.training[] = false
        end
        grad = obtain_grad_from_model(linear_sum_func, input; predict_position=predict_position)
        grad_product = (isnothing(pseudo_model) ? grad : pseudo_model(input, grad)) .* input |> cpu

        push_to_contributions!(grad_product, input, data_pt_offset, contributions)
        data_pt_offset += data_load.batchsize
    end
    
    return contributions
end

function compute_and_filter_contributions(data, m, processor; 
        train_stats=nothing, 
        threshold_stats=nothing, 
        predict_position=1,
        operate_on_gpu=true
        )

    data_load_complete = Flux.DataLoader(
        (data.X, data.Y), # Y is unnormalized, but doesn't matter here unless X is normalized
        batchsize = m.hp.batch_size,
        shuffle = false,
        partial = true,
    )
    
    contributions = compute_contributions(m, data_load_complete; 
        pseudo_model=processor, predict_position=predict_position,);
    @info "Total contributions obtained: $(length(contributions))"
    !isnothing(threshold_stats) && begin
        filter!(x->abs(x.contribution) ≥ threshold_stats.threshold, contributions);
    end
    @info "Contributions after filtering: $(length(contributions))"
    BanzhafInference.sanity_check(m, contributions, data_load_complete; 
        predict_position=predict_position, operate_on_gpu=operate_on_gpu,
        train_stats=train_stats
        );

    return contributions, data_load_complete
end
