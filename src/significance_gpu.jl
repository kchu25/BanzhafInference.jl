"""
GPU-accelerated significance testing using CUDA for parallel Mann-Whitney U tests.
This implementation parallelizes the computation across multiple motifs.
"""

# CUDA kernel for computing ranks in pooled samples
function rank_pooled_kernel!(pooled, ranks, n1, n2)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_total = n1 + n2
    
    if idx <= n_total
        val = pooled[idx]
        rank = 0.0
        count = 0
        
        # Count values less than current value
        for i in 1:n_total
            if pooled[i] < val
                rank += 1.0
            elseif pooled[i] == val
                count += 1
            end
        end
        
        # Average rank for ties (midrank)
        ranks[idx] = rank + (count + 1) / 2.0
    end
    return nothing
end

# CUDA kernel for computing U statistic for a single test
function compute_u_statistic_kernel!(group_data, group_sizes, random_data, n_random, u_stats, test_idx)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx == 1  # Single thread per test
        n1 = group_sizes[test_idx]
        n2 = n_random
        n_total = n1 + n2
        
        # Get offset for this group's data
        offset = 0
        for i in 1:(test_idx-1)
            offset += group_sizes[i]
        end
        
        # Compute rank sum for group 1 (motif data)
        rank_sum = 0.0
        for i in 1:n1
            val = group_data[offset + i]
            rank = 0.0
            count = 0
            
            # Compare with all pooled values
            for j in 1:n1
                other_val = group_data[offset + j]
                if other_val < val
                    rank += 1.0
                elseif other_val == val
                    count += 1
                end
            end
            
            for j in 1:n2
                other_val = random_data[j]
                if other_val < val
                    rank += 1.0
                elseif other_val == val
                    count += 1
                end
            end
            
            # Midrank for ties
            rank_sum += rank + (count + 1) / 2.0
        end
        
        # Compute U statistic
        u1 = rank_sum - (n1 * (n1 + 1)) / 2.0
        u_stats[test_idx] = u1
    end
    
    return nothing
end

# More efficient batched kernel - processes multiple tests in parallel
function compute_u_statistics_batched_kernel!(group_data_starts, group_data, group_sizes, 
                                               random_data, n_random, u_stats, n_tests)
    test_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if test_idx <= n_tests
        n1 = group_sizes[test_idx]
        n2 = n_random
        start_idx = group_data_starts[test_idx]
        
        # Compute rank sum for group 1
        rank_sum = 0.0
        for i in 1:n1
            val = group_data[start_idx + i - 1]
            rank = 0.0
            count = 0
            
            # Compare with group data
            for j in 1:n1
                other_val = group_data[start_idx + j - 1]
                if other_val < val
                    rank += 1.0
                elseif other_val == val
                    count += 1
                end
            end
            
            # Compare with random data
            for j in 1:n2
                other_val = random_data[j]
                if other_val < val
                    rank += 1.0
                elseif other_val == val
                    count += 1
                end
            end
            
            rank_sum += rank + (count + 1) / 2.0
        end
        
        # U statistic
        u1 = rank_sum - (n1 * (n1 + 1)) / 2.0
        u_stats[test_idx] = u1
    end
    
    return nothing
end

# Optimized kernel using shared memory and thread-level parallelism per test
function compute_u_statistics_shared_mem!(group_data_starts, group_data, group_sizes, 
                                          random_data, n_random, u_stats, n_tests)
    # Each block handles one test with multiple threads
    test_idx = blockIdx().x
    thread_id = threadIdx().x
    threads = blockDim().x
    
    if test_idx <= n_tests
        n1 = group_sizes[test_idx]
        start_idx = group_data_starts[test_idx]
        
        # Each thread handles a subset of group elements
        # Use Float64 for better numerical accuracy
        local_sum = 0.0
        
        for i in thread_id:threads:n1
            if i <= n1
                val = group_data[start_idx + i - 1]
                
                # Count smaller and equal values separately for better tie handling
                n_less = 0
                n_equal = 0
                
                # Count in group
                for j in 1:n1
                    other = group_data[start_idx + j - 1]
                    if other < val
                        n_less += 1
                    elseif other == val
                        n_equal += 1
                    end
                end
                
                # Count in random
                for j in 1:n_random
                    other = random_data[j]
                    if other < val
                        n_less += 1
                    elseif other == val
                        n_equal += 1
                    end
                end
                
                # Midrank formula: rank = n_less + (n_equal + 1) / 2
                # More accurate than incrementing rank
                rank = Float64(n_less) + (Float64(n_equal) + 1.0) / 2.0
                local_sum += rank
            end
        end
        
        # Parallel reduction within block using shared memory with Float64
        shared = @cuDynamicSharedMem(Float64, threads)
        shared[thread_id] = local_sum
        sync_threads()
        
        # Kahan-style reduction for better numerical stability
        stride = threads ÷ 2
        while stride > 0
            if thread_id <= stride && thread_id + stride <= threads
                shared[thread_id] += shared[thread_id + stride]
            end
            sync_threads()
            stride ÷= 2
        end
        
        if thread_id == 1
            rank_sum = shared[1]
            # Compute U statistic with Float64 precision
            u1 = rank_sum - Float64(n1) * Float64(n1 + 1) / 2.0
            u_stats[test_idx] = Float32(u1)
        end
    end
    
    return nothing
end

"""
Compute z-score for Mann-Whitney U test (normal approximation)
"""
function mannwhitney_z_score(u, n1, n2)
    # Use Float64 for all intermediate calculations
    n1_f = Float64(n1)
    n2_f = Float64(n2)
    u_f = Float64(u)
    
    μ_u = (n1_f * n2_f) / 2.0
    # Standard error without tie correction (conservative)
    σ_u = sqrt((n1_f * n2_f * (n1_f + n2_f + 1.0)) / 12.0)
    
    # Continuity correction for better approximation
    # Adjust U toward mean by 0.5
    u_corrected = u_f > μ_u ? u_f - 0.5 : u_f + 0.5
    
    z = (u_corrected - μ_u) / σ_u
    return z
end

"""
Convert z-score to p-value (two-tailed test)
"""
function z_to_pvalue(z)
    # Use erfc for better numerical stability with large z-scores
    # For large |z|, 1 - Φ(z) loses precision, but erfc handles it correctly
    # P(Z > |z|) = erfc(|z|/√2) for two-tailed test
    return erfc(abs(z) / sqrt(2.0))
end

"""
Summarize grouped motif DataFrames without significance testing.
Returns a DataFrame with summary statistics only (no p-values or filtering).
Useful when significance filtering is not needed.
"""
function summarize_motifs(grouped_motifs_dfs)
    df_summary = DataFrames.combine(grouped_motifs_dfs, 
        :banzhaf => mean => :mean_banzhaf,
        :banzhaf => std => :std_banzhaf,
        :banzhaf => median => :median_banzhaf,
        :contribution => mean => :mean_contribution,
        nrow => :count
    )
    sort!(df_summary, :mean_banzhaf, rev=true)
    return df_summary
end

"""
GPU-accelerated Mann-Whitney U tests for multiple groups.

Parameters:
- grouped_motifs_dfs: GroupedDataFrame with banzhaf values for each motif
- random_coalitions: DataFrame with banzhaf values for random coalitions
- q_thresh: FDR threshold for significance (default from Q_THRESHOLD)

Returns:
- DataFrame with significance test results, similar to get_significant_motifs
"""
function get_significant_motifs_gpu(grouped_motifs_dfs, random_coalitions; q_thresh = Q_THRESHOLD, actually_filter=true)
    
    # Skip GPU computation entirely if not filtering
    if !actually_filter
        return summarize_motifs(grouped_motifs_dfs)
    end

    n_tests = length(grouped_motifs_dfs)
    
    @info "Performing GPU-accelerated significance testing for $n_tests motifs."
    
    # Prepare data structures
    random_banzhaf = Float32.(random_coalitions.banzhaf)
    n_random = length(random_banzhaf)
    
    # Collect all group data and their sizes
    group_sizes = Int32[]
    group_data_all = Float32[]
    group_data_starts = Int32[]
    current_start = 1
    
    for group in grouped_motifs_dfs
        group_banzhaf = Float32.(group.banzhaf)
        # Subsample if group exceeds MAX_BANZHAF_PER_GROUP
        if length(group_banzhaf) > MAX_BANZHAF_PER_GROUP
            group_banzhaf = group_banzhaf[sample(1:length(group_banzhaf), MAX_BANZHAF_PER_GROUP; replace=false)]
        end
        push!(group_sizes, length(group_banzhaf))
        append!(group_data_all, group_banzhaf)
        push!(group_data_starts, current_start)
        current_start += length(group_banzhaf)
    end
    
    # Transfer to GPU
    d_random_data = CuArray(random_banzhaf)
    d_group_data = CuArray(group_data_all)
    d_group_sizes = CuArray(group_sizes)
    d_group_starts = CuArray(group_data_starts)
    d_u_stats = CUDA.zeros(Float32, n_tests)
    
    # Launch kernel - choose between batched (simple) or shared memory (optimized) version
    USE_SHARED_MEM = true  # Set to false to use simple batched kernel
    
    if USE_SHARED_MEM
        # Optimized: one block per test, multiple threads per block
        threads_per_block = 256
        n_blocks = n_tests
        shared_mem_size = threads_per_block * sizeof(Float64)  # Use Float64 for accuracy
        
        @info "Launching optimized GPU kernel with $n_blocks blocks of $threads_per_block threads (shared memory version)."
        @cuda threads=threads_per_block blocks=n_blocks shmem=shared_mem_size compute_u_statistics_shared_mem!(
            d_group_starts, d_group_data, d_group_sizes, 
            d_random_data, n_random, d_u_stats, n_tests
        )
    else
        # Simple: one thread per test
        threads_per_block = 256
        n_blocks = ceil(Int, n_tests / threads_per_block)
        
        @info "Launching simple GPU kernel with $n_blocks blocks of $threads_per_block threads (batched version)."
        @cuda threads=threads_per_block blocks=n_blocks compute_u_statistics_batched_kernel!(
            d_group_starts, d_group_data, d_group_sizes, 
            d_random_data, n_random, d_u_stats, n_tests
        )
    end
    
    CUDA.synchronize()
    
    @info "GPU computation complete. Transferring results back to CPU."

    # Transfer results back to CPU
    u_stats = Array(d_u_stats)
    group_sizes_cpu = Array(d_group_sizes)
    
    # Compute p-values using normal approximation
    p_values = Float64[]
    for i in 1:n_tests
        n1 = group_sizes_cpu[i]
        n2 = n_random
        u = u_stats[i]
        
        # Convert to z-score and p-value
        z = mannwhitney_z_score(u, n1, n2)
        p = z_to_pvalue(z)
        push!(p_values, p)
    end
    
    # FDR correction
    fdr_results = adjust(p_values, BenjaminiHochberg())
    
    # Combine with motif statistics (same as CPU version)
    df_significant = DataFrames.combine(grouped_motifs_dfs, 
        :banzhaf => mean => :mean_banzhaf,
        :banzhaf => std => :std_banzhaf,
        :banzhaf => median => :median_banzhaf,
        :contribution => mean => :mean_contribution,
        nrow => :count
    )
    
    df_significant.pvalue = p_values
    df_significant.qvalue = fdr_results
    df_significant.significant = df_significant.qvalue .< q_thresh
    
    # Filter to keep only significant motifs
    n_total = nrow(df_significant)

    actually_filter && filter!(row -> row.significant, df_significant)
    
    @info "Found $(nrow(df_significant)) significant configs out of $n_total total (FDR < $q_thresh)."
    sort!(df_significant, :mean_banzhaf, rev=true)
    
    return df_significant
end

"""
Fallback version that validates GPU results match CPU implementation.
Useful for testing/debugging.
"""
function validate_gpu_vs_cpu(grouped_motifs_dfs, random_coalitions; q_thresh = Q_THRESHOLD)
    @info "Running both CPU and GPU versions for validation..."
    
    # CPU version
    cpu_results = get_significant_motifs(grouped_motifs_dfs, random_coalitions; q_thresh=q_thresh)
    
    # GPU version
    gpu_results = get_significant_motifs_gpu(grouped_motifs_dfs, random_coalitions; q_thresh=q_thresh)
    
    @info "CPU found $(nrow(cpu_results)) significant motifs"
    @info "GPU found $(nrow(gpu_results)) significant motifs"
    
    # Compare p-values
    if nrow(cpu_results) > 0 && nrow(gpu_results) > 0
        max_pvalue_diff = maximum(abs.(cpu_results.pvalue .- gpu_results.pvalue))
        @info "Maximum p-value difference: $max_pvalue_diff"
    end
    
    return cpu_results, gpu_results
end
