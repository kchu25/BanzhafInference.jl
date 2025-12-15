"""
Configuration and constants for Banzhaf computation
Centralized configuration management
"""

# ============================================================================
# GPU Configuration
# ============================================================================

"""Default GPU thread configuration"""
const DEFAULT_THREADS_PER_BLOCK = 512

"""Maximum vector length for exact enumeration (without replacement)"""
const MAX_EXACT_ENUMERATION_LENGTH = 5

"""Maximum vector length for random sampling-based subset sum computation"""
const MAX_SAFE_VECTOR_LENGTH = 1000

# ============================================================================
# Algorithm Parameters
# ============================================================================

"""Default number of samples per vector for approximation"""
const DEFAULT_SAMPLES_PER_VECTOR = 10000

"""Default random seed for reproducible results"""
const DEFAULT_RANDOM_SEED = 42

# ============================================================================
# Memory Management
# ============================================================================

"""Threshold for switching to batch processing (number of vectors)"""
const BATCH_PROCESSING_THRESHOLD = 1000

"""Maximum GPU memory usage fraction before batching (increase for larger batches)"""
const MAX_GPU_MEMORY_FRACTION = 0.85  # Was 0.75 - now more aggressive

"""Minimum batch size for processing"""
const MIN_BATCH_SIZE = 1

"""Memory safety buffer fraction (reduce for larger batches)"""
const MEMORY_SAFETY_BUFFER = 0.05  # Was 0.1 - now less conservative


# ============================================================================
# Configuration Structure
# ============================================================================


"""
Configuration options for Banzhaf computation

# Fields
- `num_samples_per_vec::Int`: Number of samples per vector (default: 10000)
- `seed::Union{Int,Nothing}`: Random seed for reproducibility (default: nothing)
- `nonlinear_fcn::Function`: Nonlinear function to apply (default: identity)
- `use_optimized::Bool`: Use optimized implementation (default: true)
- `use_batching::Bool`: Enable automatic batching for large datasets (default: true)
- `max_batch_size::Union{Int,Nothing}`: Maximum batch size (default: auto)
- `threads_per_block::Int`: GPU threads per block (default: 512)
"""
struct BanzhafOptions
    num_samples_per_vec::Int
    seed::Union{Int,Nothing}
    final_nonlinearity::BanzhafInference.Functor1Arg
    scale_back_function::Union{BanzhafInference.Functor1Arg, BanzhafInference.Functor2Arg}
    use_optimized::Bool
    use_batching::Bool
    max_batch_size::Union{Int,Nothing}
    threads_per_block::Int
    
    function BanzhafOptions(;
        num_samples_per_vec::Int = DEFAULT_SAMPLES_PER_VECTOR,
        seed::Union{Int,Nothing} = nothing,
        final_nonlinearity = BanzhafInference.FunctorWrapper(x->x),
        scale_back_function = BanzhafInference.FunctorWrapper(x->x),
        use_optimized::Bool = true,
        use_batching::Bool = true,
        max_batch_size::Union{Int,Nothing} = nothing,
        threads_per_block::Int = DEFAULT_THREADS_PER_BLOCK
    )
        new(num_samples_per_vec, seed, final_nonlinearity, scale_back_function, use_optimized, 
            use_batching, max_batch_size, threads_per_block)
    end
end



# ============================================================================
# Utility Functions
# ============================================================================

"""
Estimate GPU memory requirements for given parameters (more comprehensive)
"""
function estimate_memory_usage(num_vectors::Int, num_samples_per_vec::Int, vec_length::Int)
    # Primary data structures
    mask_memory = num_vectors * num_samples_per_vec * sizeof(Int32)  # Random masks
    vec_id_memory = num_vectors * num_samples_per_vec * sizeof(Int32)  # Vector IDs
    sum_memory = num_vectors * num_samples_per_vec * sizeof(Float32)  # Subset sums
    vector_memory = num_vectors * vec_length * sizeof(Float32)  # Input vectors
    
    # Intermediate computations (Banzhaf calculation)
    union_terms_memory = sum_memory  # Same size as subset sums
    difference_terms_memory = sum_memory  # Same size as subset sums
    results_memory = num_vectors * sizeof(Float32)  # Final results
    counts_memory = num_vectors * sizeof(Int32)  # Counts for averaging
    target_vals_memory = num_vectors * sizeof(Float32)  # Target values
    
    # Total with safety buffer
    base_memory = (mask_memory + vec_id_memory + sum_memory + vector_memory + 
                   union_terms_memory + difference_terms_memory + 
                   results_memory + counts_memory + target_vals_memory)
    
    # Add safety buffer for GPU operations overhead
    total_memory = Int(ceil(base_memory * (1.0 + MEMORY_SAFETY_BUFFER)))
    
    return total_memory
end