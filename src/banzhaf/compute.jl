
# ============================================================================
# Optimized Mask Generation
# ============================================================================

"""
Generate random masks with automatic memory management and optimal batching
"""
function generate_masks_optimized(lengths::Vector{Int32}, num_samples_per_vec::Int; 
                                 seed=nothing, wo_replacement_len=MAX_EXACT_ENUMERATION_LENGTH)
    
    if seed !== nothing
        Random.seed!(seed)
    end
    
    M = length(lengths)
    # Only use exact enumeration (2^n) for very small vectors (n <= MAX_EXACT_ENUMERATION_LENGTH)
    # For all others, use random sampling with num_samples_per_vec
    masks_needed = [n <= MAX_EXACT_ENUMERATION_LENGTH ? min(2^n, num_samples_per_vec) : 
                   num_samples_per_vec for n in lengths]
    total_samples = sum(masks_needed)
    
    # Pre-allocate with better memory layout
    masks = Vector{Int32}(undef, total_samples)
    vec_ids = Vector{Int32}(undef, total_samples)
    
    # Optimized mask generation
    current_pos = 1
    for (vec_id, vec_length) in enumerate(lengths)
        n_masks = masks_needed[vec_id]
        range = current_pos:(current_pos + n_masks - 1)
        
        if vec_length <= wo_replacement_len && n_masks == 2^vec_length
            # Exact enumeration for small vectors
            masks[range] .= 0:(2^vec_length - 1)
        else
            # Mark for GPU generation
            masks[range] .= -1
        end
        
        vec_ids[range] .= vec_id
        current_pos += n_masks
    end
    
    # GPU generation for marked masks
    if any(==(−1), masks)
        d_masks, d_vec_ids = gpu_generate_remaining_masks(masks, vec_ids, lengths, seed)
        return d_masks, d_vec_ids, masks_needed
    else
        return CuArray(masks), CuArray(vec_ids), masks_needed
    end
end


"""
GPU generation for remaining masks (optimized)
"""
function gpu_generate_remaining_masks(masks, vec_ids, lengths, seed)
    d_lengths = CuArray(lengths)
    d_vec_ids = CuArray(vec_ids)
    d_masks = CuArray(masks)
    
    # Deterministic RNG initialization
    if seed !== nothing
        Random.seed!(seed)
    end
    rng_seeds = rand(UInt32, length(masks))
    d_rng_states = CuArray(rng_seeds)
    
    # Launch with optimal configuration
    launch_kernel_safe(generate_masks_kernel!, d_masks, d_vec_ids, d_lengths, d_rng_states,
                      name="mask_generation")
    
    return d_masks, d_vec_ids
end


# ============================================================================
# Optimized Subset Sum Computation
# ============================================================================

"""
Compute subset sums with memory-efficient GPU implementation
"""
function compute_subset_sums_optimized(vectors, num_samples_per_vec::Int; seed=nothing)
    # Input validation
    validate_inputs(vectors, fill(0.0f0, length(vectors)))  # Dummy target vals for validation
    
    # Prepare data with optimal memory layout
    padded_vecs, lengths = prepare_padded_vectors_optimized(vectors)
    
    # Generate masks
    # @vinfo "Generating masks..."
    d_masks, d_vec_ids, masks_needed = generate_masks_optimized(lengths, num_samples_per_vec; seed=seed)
    
    return safe_gpu_alloc(d_masks, d_vec_ids) do masks, vec_ids
        # Move data to GPU
        d_vecs = CuArray(padded_vecs)
        d_lengths = CuArray(lengths)
        d_sums = CUDA.zeros(Float32, length(masks))
        
        # Launch subset sum kernel
        # @vinfo "Computing subset sums..."
        launch_kernel_safe(subset_sum_kernel!, d_sums, d_vecs, masks, vec_ids, d_lengths,
                          name="subset_sum")
        
        return d_sums, vec_ids, masks_needed
    end
end


"""
Optimized vector padding with better memory efficiency
"""
function prepare_padded_vectors_optimized(vectors)
    if isempty(vectors)
        error("Cannot prepare padded vectors from empty collection")
    end
    
    # Use StaticArrays for small vectors if beneficial
    vec_list = collect(vectors)
    T = eltype(first(vec_list))
    
    lengths = Int32.(length.(vec_list))
    max_len = maximum(lengths)
    M = length(vec_list)
    
    # Pre-allocate with zeros (more cache-friendly)
    padded_matrix = zeros(T, max_len, M)
    
    # Vectorized copying when possible
    for (i, vec) in enumerate(vec_list)
        padded_matrix[1:length(vec), i] = vec
    end
    
    return padded_matrix, lengths
end

# ============================================================================
# Optimized Banzhaf Computation
# ============================================================================

"""
High-performance Banzhaf computation with automatic batching
"""
function compute_banzhaf_optimized(vectors, target_vals; 
                                 num_samples_per_vec::Int=DEFAULT_SAMPLES_PER_VECTOR,
                                 seed=nothing, 
                                 final_nonlinearity=BanzhafInference.FunctorWrapper(identity),
                                 scale_back_function=BanzhafInference.FunctorWrapper(identity),
                                 use_batching::Bool=true)
    
    # For generators, we can't easily validate without materializing
    # So we'll validate during processing or with sampling
    if isa(vectors, AbstractVector)
        validate_inputs(vectors, target_vals)
        num_vectors = length(vectors)
    else
        # For generators, estimate size from target_vals
        num_vectors = length(target_vals)
        _should_log(:verbose) && println("🔄 Processing generator with $num_vectors expected vectors")
    end
    
    # Check if batching is needed
    if use_batching && num_vectors > BATCH_PROCESSING_THRESHOLD
        # @vinfo "Large dataset detected ($num_vectors vectors). Using batching..."
        return process_in_batches(compute_banzhaf_single_batch, vectors, target_vals;
                                num_samples_per_vec=num_samples_per_vec, 
                                seed=seed, 
                                final_nonlinearity=final_nonlinearity,
                                scale_back_function=scale_back_function)
    else
        # @vinfo "Processing all vectors in a single batch ($num_vectors vectors)"
        # For single batch, we need to materialize generators
        if !isa(vectors, AbstractVector)
            # println("📦 Materializing generator for single batch processing")
            vectors = collect(vectors)
        end
        
        return compute_banzhaf_single_batch(vectors, target_vals;
                                          num_samples_per_vec=num_samples_per_vec,
                                          seed=seed, 
                                          final_nonlinearity=final_nonlinearity,
                                          scale_back_function=scale_back_function)
    end
end


"""
Single batch Banzhaf computation (optimized)
"""
function compute_banzhaf_single_batch(vectors, target_vals;
                                    num_samples_per_vec::Int=DEFAULT_SAMPLES_PER_VECTOR,
                                    seed=nothing, 
                                    final_nonlinearity=BanzhafInference.FunctorWrapper(identity),
                                    scale_back_function=BanzhafInference.FunctorWrapper(identity))
    
    # Compute subset sums
    d_sums, d_vec_ids, _ = compute_subset_sums_optimized(vectors, num_samples_per_vec; seed=seed)
    
    # Convert target values to GPU
    d_target_vals = target_vals isa CuArray ? target_vals : CuArray(Float32.(target_vals))
    
    return safe_gpu_alloc(d_sums, d_vec_ids, d_target_vals) do sums, vec_ids, targets
        # Validate GPU arrays
        validate_gpu_arrays(sums, vec_ids, targets)
        # Compute union terms (one thread per sample)
        union_terms = CUDA.zeros(Float32, length(sums))
        threads, blocks = get_optimal_launch_config(length(sums))
        launch_kernel_safe(banzhaf_union_term_kernel!, sums, vec_ids, targets, union_terms;
                          threads=threads, blocks=blocks, name="union_terms")
        
        # Apply nonlinear function then scale back (element-wise on GPU)
        # Composition: scale_back_function(final_nonlinearity(x))
        union_transformed = scale_back_function.(final_nonlinearity.(union_terms))
        sums_transformed = scale_back_function.(final_nonlinearity.(sums))
        difference_terms = union_transformed .- sums_transformed
        
        # Accumulate results (one thread per sample, NOT per vector!)
        num_vectors = length(targets)
        banzhafs = CUDA.zeros(Float32, num_vectors)
        counts = CUDA.zeros(Int32, num_vectors)
        
        threads, blocks = get_optimal_launch_config(length(difference_terms))
        launch_kernel_safe(banzhaf_accumulate_kernel!, banzhafs, counts, difference_terms, vec_ids;
                          threads=threads, blocks=blocks, name="accumulate")
        
        # Compute averages (one thread per vector)
        threads, blocks = get_optimal_launch_config(num_vectors)
        launch_kernel_safe(banzhaf_average_kernel!, banzhafs, counts;
                          threads=threads, blocks=blocks, name="average")
        
        return Array(banzhafs)
    end
end
