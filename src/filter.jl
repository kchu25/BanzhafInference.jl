function filter_via_magnitude(contributions_df, contribs; mag_percentile=MAG_PERCENTILE)
    mag_thresh = StatsBase.quantile(contributions_df.mag, mag_percentile);
    mag_mask = contributions_df.mag .> mag_thresh;
    contributions_df_filtered = contributions_df[mag_mask, :]; # copy for now to prevent parent call error later
    
    # Create filtering generator for both vectors and generators
    if isa(contribs, AbstractVector)
        # For vectors, use view
        contribs_filtered = @view contribs[mag_mask]
    else
        # For generators, create a filtering generator
        contribs_filtered = (vec for (vec, keep) in zip(contribs, mag_mask) if keep)
    end
    
    return contribs_filtered, contributions_df_filtered
end
