function single_motifs_banzhaf!(ac, ec, contribs_filtered, contributions_df_filtered)

    target_vals = contributions_df_filtered.contribution
    if is_identity(ac.final_nonlinearity) && ac.normalization_method == :identity
        banzhafs = target_vals
    elseif is_identity(ac.final_nonlinearity) && ac.normalization_method == :zscore
        banzhafs = target_vals .* ac.scale_back_function.functor.std
    else
        banzhafs = BanzhafInference.obtain_banzhafs_enhanced(
            contribs_filtered, target_vals;
            num_samples_per_vec=ac.num_samples_per_vec,
            seed=ec.seed,
            final_nonlinearity=ac.final_nonlinearity,
            scale_back_function=ac.scale_back_function,
        )
    end
    contributions_df_filtered.banzhaf = banzhafs
    
    # report stats
    try
        println("Banzhaf stats:")
        println("Max: ", maximum(banzhafs))
        println("Min: ", minimum(banzhafs))
        println("Mean: ", mean(banzhafs))
    catch e
        @warn "Failed to print banzhaf stats: $(e)"
    end
end

function single_motifs_and_significance_filtering!(
    ac, ec, contribs_filtered, contributions_df_filtered, random_coalitions;
    mutegenesis=false, top_and_bot_counts=8)
    single_motifs_banzhaf!(ac, ec, contribs_filtered, contributions_df_filtered)

    if mutegenesis
        columns_of_interest = [:filter_index, :position]
    else
        columns_of_interest = [:filter_index]
    end
    @info "columns_of_interest: $(columns_of_interest)"

    df_significant = filter_and_test_significance(
            contributions_df_filtered, columns_of_interest, random_coalitions
            );
    # now keep only significant motifs in contributions_df_filtered (singletons)
    contributions_df_filtered_singletons = 
        apply_final_filters!(contributions_df_filtered, df_significant, columns_of_interest;
            mutegenesis=mutegenesis, top_and_bot_counts=top_and_bot_counts); 
    return contributions_df_filtered_singletons
end


function extract_motifs_from_sample(activation_dict, ec, motif_size, m_syms)
    df_motifs = EpicHyperSketch.obtain_enriched_configurations_partitioned(
        activation_dict; motif_size, ec.filter_len) # TODO need a seed as well
    BanzhafInference.missing_columns_check(df_motifs, m_syms) 
    BanzhafInference.convert_all_except!(
        df_motifs, BanzhafInference.IntType, :contribution)
    return df_motifs
end

# Read Arrow file → dedup → atomically replace original via a temp file
function flush_dedup!(path, dedup_cols)
    df = DataFrame(Arrow.Table(path); copycols=true)
    n_pre = nrow(df)
    unique!(df, dedup_cols)
    tmp = path * ".tmp"
    Arrow.write(tmp, df)
    mv(tmp, path; force=true)   # atomic rename replaces old file
    @info "  → On-disk dedup: $n_pre rows → $(nrow(df)) rows"
end

function obtain_multi_motifs(ec, seed, m_syms, d_syms, contributions_df_filtered; motif_size=2, dedup_every=3)

    dedup_cols = Not(:contribution)
    n_before = 0

    !isdir(ec.cache_folder_path) && mkpath(ec.cache_folder_path)    
    cached_output_file = joinpath(ec.cache_folder_path, "motifs_size_$(motif_size).arrow")
    chunk_id = 0

    # Collect motifs with periodic deduplication to bound memory
    writer = open(Arrow.Writer, cached_output_file)

    for offset = 0:(ec.num_contrib_samples-1)
        @info "Obtaining motifs from contribution sample $(offset + 1) / $(ec.num_contrib_samples)..."
        contributions_df_sampled = BanzhafInference.subsample_contributions(
            contributions_df_filtered; max_rows_per_group=ec.subsample_rows, 
            verbose=false, seed=seed+offset)
        ad = BanzhafInference.load_activation_dict(contributions_df_sampled)
        df = extract_motifs_from_sample(ad, ec, motif_size, m_syms)
        n_before += nrow(df)
        Arrow.write(writer, df)

        is_dedup_step = (offset + 1) % dedup_every == 0
        if is_dedup_step
            close(writer)
            flush_dedup!(cached_output_file, dedup_cols)
            # rename deduped file as a chunk
            chunk_file = joinpath(ec.cache_folder_path, "motifs_size_$(motif_size)_chunk_$(chunk_id).arrow")
            mv(cached_output_file, chunk_file; force=true)
            chunk_id += 1
            # start a fresh writer for the next batch of new rows
            writer = open(Arrow.Writer, cached_output_file)
        end
    end
    close(writer)

    # Final chunk (remaining rows after last dedup step)
    if isfile(cached_output_file) && filesize(cached_output_file) > 0
        flush_dedup!(cached_output_file, dedup_cols)
        chunk_file = joinpath(ec.cache_folder_path, "motifs_size_$(motif_size)_chunk_$(chunk_id).arrow")
        mv(cached_output_file, chunk_file; force=true)
        chunk_id += 1
    end

    # Pairwise merge chunks with cross-chunk dedup (max 2 chunks in memory at a time)
    chunk_files = [joinpath(ec.cache_folder_path, "motifs_size_$(motif_size)_chunk_$(i).arrow") 
                   for i in 0:(chunk_id-1) if isfile(joinpath(ec.cache_folder_path, "motifs_size_$(motif_size)_chunk_$(i).arrow"))]
    
    total_chunks = length(chunk_files)
    merge_step = 0
    while length(chunk_files) > 1
        a, b = popfirst!(chunk_files), popfirst!(chunk_files)
        merge_step += 1
        @info "Merging chunks ($merge_step / $(total_chunks - 1)): $(basename(a)) + $(basename(b)) → $(length(chunk_files) + 1) chunks remaining"
        merged = vcat(DataFrame(Arrow.Table(a); copycols=true), DataFrame(Arrow.Table(b); copycols=true))
        unique!(merged, dedup_cols)
        @info "  → Merged: $(nrow(merged)) unique rows"
        merged_file = joinpath(ec.cache_folder_path, "motifs_merged_$(length(chunk_files)).arrow")
        Arrow.write(merged_file, merged)
        rm(a; force=true); rm(b; force=true)
        push!(chunk_files, merged_file)
        GC.gc()
    end

    df_motifs = DataFrame(Arrow.Table(chunk_files[1]); copycols=true)
    rm(chunk_files[1]; force=true)

    # Restore column types after Arrow round-trip (Arrow may widen Int32 → Int64)
    BanzhafInference.convert_all_except!(df_motifs, BanzhafInference.IntType, :contribution)

    n_after = nrow(df_motifs)
    @info "Extracted $n_before motifs before deduplication."
    @info "Retained $n_after unique motifs after deduplication."
    @info "The unique motifs is $(round(n_after / n_before * 100, digits=2))% of the original."

    BanzhafInference.add_motif_positions_columns!(df_motifs, m_syms, d_syms, ec.filter_len)
    return df_motifs
end
