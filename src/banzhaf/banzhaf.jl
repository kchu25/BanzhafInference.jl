
include("config.jl")
include("gpu_utils.jl") 
include("kernels.jl")
include("compute.jl")

include("structs.jl")
include("multi_motifs/core.jl")
# ============================================================================
# Main API Functions
# ============================================================================

"""
    compute_banzhaf(vectors, target_vals; options...)

Compute Banzhaf values for given vectors and target values.

# Arguments
- `vectors`: Collection of numerical vectors (can be SubArrays, Views, etc.)
- `target_vals`: Target values corresponding to each vector
- `options...`: Keyword arguments (see BanzhafOptions for details)

# Returns
- `Vector{Float32}`: Banzhaf values for each vector

# Examples
```julia
# Basic usage
vectors = [[1.0, 2.0], [3.0, 4.0]]
target_vals = [5.0, 7.0]
banzhafs = compute_banzhaf(vectors, target_vals)

# With custom options
banzhafs = compute_banzhaf(vectors, target_vals; 
                          num_samples_per_vec=5000,
                          seed=42)
```
"""
function compute_banzhaf(vectors, target_vals; kwargs...)
    options = BanzhafOptions(; kwargs...)
    return compute_banzhaf(vectors, target_vals, options)
end

"""
    compute_banzhaf(vectors, target_vals, options::BanzhafOptions)

Compute Banzhaf values with explicit options structure.
"""
function compute_banzhaf(vectors, target_vals, options::BanzhafOptions)
    return compute_banzhaf_optimized(vectors, target_vals;
        num_samples_per_vec = options.num_samples_per_vec,
        seed = options.seed,
        final_nonlinearity = options.final_nonlinearity,
        scale_back_function = options.scale_back_function,
        use_batching = options.use_batching
    )
end

"""
Enhanced version of obtain_banzhafs with automatic batch sizing and memory management.

# Arguments
- `vectors`: Collection or generator of vectors to compute Banzhaf values for
- `target_vals`: Target values for each vector
- `num_samples_per_vec`: Number of random samples per vector (default: 10000)
- `seed`: Random seed for reproducibility (default: 42)
- `nonlinear_fcn`: Nonlinear function to apply (default: identity)
"""
function obtain_banzhafs_enhanced(vectors, target_vals; 
                                num_samples_per_vec::Int=1000, 
                                seed=nothing, 
                                final_nonlinearity=BanzhafInference.FunctorWrapper(x->x),
                                scale_back_function=BanzhafInference.FunctorWrapper(x->x)
                                )
    
    # Create options for computation
    options = BanzhafOptions(
        num_samples_per_vec = num_samples_per_vec,
        use_batching = true,
        seed = seed === nothing ? 42 : seed,
        final_nonlinearity = final_nonlinearity,
        scale_back_function = scale_back_function
    )
    
    # Compute Banzhaf values with automatic memory management
    return compute_banzhaf(vectors, target_vals, options)
end
