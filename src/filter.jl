function filter_via_magnitude(contributions_df, contribs; mag_percentile=0.95)
    mag_thresh = StatsBase.quantile(contributions_df.mag, mag_percentile);
    mag_mask = contributions_df.mag .> mag_thresh;
    contributions_df_filtered = contributions_df[mag_mask, :]; # copy for now to prevent parent call error later
    contribs_filtered = @view contribs[mag_mask];
    return contribs_filtered, contributions_df_filtered
end
