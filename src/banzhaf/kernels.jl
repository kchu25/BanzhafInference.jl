"""
GPU Kernels for Banzhaf computation
Separated from host code for better organization and reusability
"""


# ============================================================================
# Mask Generation Kernels
# ============================================================================

"""
GPU kernel to generate random masks for multiple vectors in parallel.
Each thread generates one mask by flipping coins for each bit position.
"""
function generate_masks_kernel!(masks, vec_ids, lengths, rng_states)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(masks)
        # Only generate masks that have sentinel value (-1)
        if masks[idx] == Int32(-1)
            # Find which vector this mask belongs to
            vec_id = vec_ids[idx]
            vec_length = lengths[vec_id]
            
            # Generate mask bit by bit using thread-local RNG
            mask = Int32(0)
            rng_state = rng_states[idx]
            
            # Limit to 31 bits to avoid overflow
            max_bits = min(Int32(vec_length), Int32(31))
            
            for bit_pos in Int32(1):max_bits
                # Simple LCG for random number generation
                rng_state = rng_state * UInt32(1664525) + UInt32(1013904223)
                # Check if bit should be set (use a bit from rng_state)
                if (rng_state & UInt32(1)) != UInt32(0)
                    mask = mask | (Int32(1) << (bit_pos - Int32(1)))
                end
                rng_state = rng_state >> 1  # Use next bit for next iteration
            end
            
            masks[idx] = mask
        end
    end
    return nothing
end

# ============================================================================
# Subset Sum Kernels
# ============================================================================

"""
GPU kernel to compute subset sums for multiple vectors in parallel.
"""
function subset_sum_kernel!(sums, vecs, masks, vec_ids, lengths)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(masks)
        vec_id = vec_ids[idx]
        mask = masks[idx]
        n = lengths[vec_id]
        
        # Compute subset sum using bit mask
        s = zero(eltype(vecs))
        for i in 1:n
            if (mask >> (i-1)) & 1 == 1
                s += vecs[i, vec_id]
            end
        end
        sums[idx] = s
    end
    return nothing
end

# ============================================================================
# Banzhaf Computation Kernels
# ============================================================================

"""
Kernel to compute union terms: subset_sum + target_value
"""
function banzhaf_union_term_kernel!(d_sums, d_vec_ids, target_vals, union_terms)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(d_sums)
        vec_id = d_vec_ids[idx]
        # Add bounds check for vec_id
        if 1 <= vec_id <= length(target_vals)
            union_terms[idx] = d_sums[idx] + target_vals[vec_id]
        end
    end
    return nothing
end

"""
Kernel to accumulate Banzhaf values by vector ID
"""
function banzhaf_accumulate_kernel!(banzhafs, counts, difference_terms, vec_ids)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(difference_terms)
        vec_id = vec_ids[idx]
        # Add bounds check for vec_id
        if 1 <= vec_id <= length(banzhafs)
            CUDA.@atomic banzhafs[vec_id] += difference_terms[idx]
            CUDA.@atomic counts[vec_id] += Int32(1)
        end
    end
    return nothing
end

"""
Kernel to compute final averages
"""
function banzhaf_average_kernel!(banzhafs, counts)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= length(banzhafs)
        if counts[idx] > 0
            banzhafs[idx] = banzhafs[idx] / Float32(counts[idx])
        end
    end
    return nothing
end
