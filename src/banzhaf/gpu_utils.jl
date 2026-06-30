
# ============================================================================
# Memory Management
# ============================================================================

"""
Safe GPU computation with memory management
"""
function safe_gpu_alloc(f::Function, arrays...)
    # Just call the function and let Julia's GC handle cleanup
    # Manual unsafe_free! can cause race conditions
    result = f(arrays...)
    
    # Optionally trigger GC to free up memory sooner
    # But don't force unsafe_free! which can cause freed reference errors
    CUDA.reclaim()
    
    return result
end

"""
Enhanced batch processing with memory-aware sizing
"""
function process_in_batches(f::Function, vectors, target_vals; batch_size=nothing, kwargs...)
    
    # Handle regular vectors (collections)
    num_vectors = length(target_vals)
    
    # For generators, process in streaming fashion
    if !isa(vectors, AbstractVector)
        @vinfo "Streaming generator through batches ($num_vectors vectors)"
        return process_generator_in_batches(f, vectors, target_vals, batch_size; kwargs...)
    end

    if batch_size === nothing
        # Auto-calculate batch size based on memory analysis
        batch_size = estimate_optimal_batch_size(vectors, kwargs)
        @vinfo "Auto-calculated batch size: $batch_size"
    end
    
    if batch_size >= num_vectors
        # No batching needed - try single batch first
        try
            return f(vectors, target_vals; kwargs...)
        catch e
            if isa(e, CUDA.OutOfGPUMemoryError)
                @warn "OOM in single batch, forcing batching with size $(num_vectors÷2)"
                batch_size = max(1, num_vectors ÷ 2)
            else
                rethrow(e)
            end
        end
    end
    
    # Process in batches with adaptive sizing
    results = Vector{Float32}(undef, num_vectors)
    
    return process_batches_with_recovery(f, vectors, target_vals, results, batch_size; kwargs...)
end

"""
Process generator in batches by collecting chunks at a time.
"""
function process_generator_in_batches(f::Function, gen, target_vals, batch_size; kwargs...)
    num_vectors = length(target_vals)
    
    # Start with conservative batch size for generators
    if batch_size === nothing
        batch_size = min(2000, num_vectors)
        @vinfo "Using batch size $batch_size for generator processing"
    end
    
    results = Vector{Float32}(undef, num_vectors)
    progress = Progress(num_vectors, desc="Processing generator batches: ", barlen=50)
    
    processed = 0
    batch_vectors = Vector{eltype(gen)}()
    sizehint!(batch_vectors, batch_size)
    
    for (i, vec) in enumerate(gen)
        push!(batch_vectors, vec)
        
        # Process when batch is full or at end
        if length(batch_vectors) >= batch_size || i == num_vectors
            start_idx = processed + 1
            end_idx = processed + length(batch_vectors)
            
            batch_targets = target_vals[start_idx:end_idx]
            
            try
                batch_results = f(batch_vectors, batch_targets; kwargs...)
                results[start_idx:end_idx] = batch_results
                processed += length(batch_vectors)
                update!(progress, processed)
                
                # Clear batch for next iteration
                empty!(batch_vectors)
                sizehint!(batch_vectors, batch_size)
                
                # Try to increase batch size if successful
                if length(batch_vectors) == batch_size && batch_size < min(5000, num_vectors)
                    batch_size = min(Int(ceil(batch_size * 1.2)), num_vectors - processed)
                end
                
            catch e
                if isa(e, CUDA.OutOfGPUMemoryError) && batch_size > 1
                    # OOM - reduce batch size and retry
                    batch_size = max(100, batch_size ÷ 2)
                    @warn "GPU OOM, reducing batch size to $batch_size"
                    CUDA.reclaim()
                    # Don't empty batch_vectors - we'll retry with same data
                    continue
                else
                    finish!(progress)
                    rethrow(e)
                end
            end
        end
    end
    
    finish!(progress)
    return results
end

"""
Process batches with automatic recovery from OOM errors
"""
function process_batches_with_recovery(f::Function, vectors, target_vals, results, 
                                     initial_batch_size; kwargs...)
    num_vectors = length(vectors)
    batch_size = initial_batch_size
    processed = 0
    
    # Create progress bar
    progress = Progress(num_vectors, desc="Processing batches: ", barlen=50)
    
    while processed < num_vectors
        remaining = num_vectors - processed
        current_batch_size = min(batch_size, remaining)
        
        start_idx = processed + 1
        end_idx = processed + current_batch_size
        
        batch_vectors = vectors[start_idx:end_idx]
        batch_targets = target_vals[start_idx:end_idx]
        
        try
            # Attempt batch processing
            batch_results = f(batch_vectors, batch_targets; kwargs...)
            results[start_idx:end_idx] = batch_results
            processed += current_batch_size
            
            # Update progress bar
            update!(progress, processed)
            
            # Success - try to increase batch size for efficiency
            if current_batch_size < batch_size && remaining > current_batch_size
                batch_size = min(batch_size, Int(ceil(batch_size * 1.2)))
            end
            
        catch e
            if isa(e, CUDA.OutOfGPUMemoryError) && current_batch_size > 1
                # OOM - reduce batch size and retry
                new_batch_size = max(1, current_batch_size ÷ 2)
                @warn "GPU OOM at batch size $current_batch_size, reducing to $new_batch_size"
                batch_size = new_batch_size
                CUDA.reclaim()  # Force garbage collection
                continue
            elseif isa(e, ArgumentError) && occursin("invalid GenericMemory size", e.msg) && current_batch_size > 1
                # Memory allocation error - batch is too large
                new_batch_size = max(1, current_batch_size ÷ 2)
                @warn "Memory allocation error at batch size $current_batch_size, reducing to $new_batch_size"
                batch_size = new_batch_size
                GC.gc()  # Force garbage collection
                continue
            else
                finish!(progress)  # Clean up progress bar before error
                rethrow(e)
            end
        end
    end
    
    finish!(progress)  # Complete progress bar
    return results
end


"""
Estimate optimal batch size based on vector characteristics and available memory
"""
function estimate_optimal_batch_size(vectors, kwargs)
    num_samples = get(kwargs, :num_samples_per_vec, DEFAULT_SAMPLES_PER_VECTOR)
    
    # Analyze vector length distribution
    lengths = length.(vectors)
    min_len, max_len = extrema(lengths)
    avg_len = mean(lengths)
    std_len = std(lengths)
    
    # Memory estimation for different scenarios
    small_vec_memory = estimate_memory_usage(1, num_samples, min_len)
    avg_vec_memory = estimate_memory_usage(1, num_samples, Int(ceil(avg_len)))
    large_vec_memory = estimate_memory_usage(1, num_samples, max_len)
    
    free_mem, total_mem = CUDA.memory_info()
    usable_memory = Int(floor(free_mem * MAX_GPU_MEMORY_FRACTION))
    
    # Conservative estimate using worst-case scenario (largest vectors)
    # but with some optimization for typical case
    if std_len / avg_len < 0.1  # Low variance in vector lengths
        target_memory = avg_vec_memory
    else
        # High variance - use weighted average (less conservative than before)
        # Changed from 0.7/0.3 to 0.5/0.5 for less conservative batching
        target_memory = Int(ceil(0.5 * avg_vec_memory + 0.5 * large_vec_memory))
    end
    
    batch_size = max(1, min(
        length(vectors),
        Int(floor(usable_memory / target_memory)),
        BATCH_PROCESSING_THRESHOLD
        # 1000
    ))
    
    @vinfo "Memory analysis" free_mb=round(free_mem/1e6, digits=2) usable_mb=round(usable_memory/1e6, digits=2) target_mb_per_vec=round(target_memory/1e6, digits=4) estimated_batch_size=batch_size
    
    return batch_size
end


# ============================================================================
# GPU Utilities
# ============================================================================

"""
Optimal thread/block configuration for given problem size
"""
function get_optimal_launch_config(problem_size::Int, max_threads::Int=DEFAULT_THREADS_PER_BLOCK)
    threads = min(max_threads, problem_size)
    blocks = cld(problem_size, threads)
    return threads, blocks
end

"""
Launch kernel with error handling and timing
"""
function launch_kernel_safe(kernel_func, args...; threads=DEFAULT_THREADS_PER_BLOCK, 
                           blocks=nothing, name="kernel")
    if blocks === nothing
        problem_size = length(args[1])  # Assume first arg determines size
        threads, blocks = get_optimal_launch_config(problem_size, threads)
    end
    
    try
        @cuda threads=threads blocks=blocks kernel_func(args...)
        CUDA.synchronize()
    catch e
        @error "Kernel $name failed" exception=e
        rethrow(e)
    end
end

# ============================================================================
# Data Validation
# ============================================================================

"""
Validate input data for GPU computation
"""
function validate_inputs(vectors, target_vals)
    @assert !isempty(vectors) "Input vectors cannot be empty"
    @assert length(vectors) == length(target_vals) "Vectors and target values must have same length"
    
    # Check for invalid vector lengths
    for (i, vec) in enumerate(vectors)
        @assert !isempty(vec) "Vector $i cannot be empty"
        # @assert length(vec) <= MAX_SAFE_VECTOR_LENGTH "Vector $i too long ($(length(vec)) > $MAX_SAFE_VECTOR_LENGTH)"
    end
    
    # Check target values
    @assert all(isfinite, target_vals) "All target values must be finite"
end


"""
Validate GPU arrays for bounds checking
"""
function validate_gpu_arrays(d_sums, d_vec_ids, target_vals)
    vec_id_range = extrema(Array(d_vec_ids))
    @assert vec_id_range[1] >= 1 "Vector IDs must be >= 1, got $(vec_id_range[1])"
    @assert vec_id_range[2] <= length(target_vals) "Vector IDs must be <= $(length(target_vals)), got $(vec_id_range[2])"
end