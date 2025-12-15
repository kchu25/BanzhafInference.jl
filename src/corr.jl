function r2_score(y_true, y_pred)
    valid = .!isnan.(y_true) .& .!isnan.(y_pred)
    if sum(valid) < 2
        return NaN
    end
    yt = y_true[valid]
    yp = y_pred[valid]
    ss_res = sum((yt .- yp).^2)
    ss_tot = sum((yt .- mean(yt)).^2)
    return ss_tot ≈ 0 ? (ss_res ≈ 0 ? 1.0 : -Inf) : 1 - ss_res / ss_tot
end

function pearson_r(y_true, y_pred)
    valid = .!isnan.(y_true) .& .!isnan.(y_pred)
    if sum(valid) < 2
        return NaN
    end
    yt = y_true[valid]
    yp = y_pred[valid]
    
    # Center the data
    yt_centered = yt .- mean(yt)
    yp_centered = yp .- mean(yp)
    
    # Compute correlation
    numerator = sum(yt_centered .* yp_centered)
    denominator = sqrt(sum(yt_centered.^2) * sum(yp_centered.^2))
    
    return denominator ≈ 0 ? 0.0 : numerator / denominator
end