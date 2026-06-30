# Verbosity control for BanzhafInference logging.
#
# Levels, in increasing order of detail:
#   :silent   -- suppress all package logging, including timing
#   :minimal  -- only essential output: timing and key result summaries (DEFAULT)
#   :verbose  -- full diagnostic output (the previous default behavior)
#
# Use `set_verbosity!(:verbose)` to restore the old, chatty behavior.

const _VERBOSITY_RANK = Dict(:silent => 0, :minimal => 1, :verbose => 2)

const _VERBOSITY = Ref(:minimal)

"""
    set_verbosity!(level::Symbol) -> Symbol

Set the global logging verbosity for BanzhafInference. Valid levels are
`:silent`, `:minimal` (default), and `:verbose`. Pass `:verbose` to restore the
previous (chatty) behavior. Returns the level that was set.
"""
function set_verbosity!(level::Symbol)
    haskey(_VERBOSITY_RANK, level) ||
        throw(ArgumentError("Unknown verbosity level :$level. Use :silent, :minimal, or :verbose."))
    _VERBOSITY[] = level
    return level
end

"""
    get_verbosity() -> Symbol

Return the current global logging verbosity for BanzhafInference.
"""
get_verbosity() = _VERBOSITY[]

# True if messages tagged at `level` should be emitted under the current setting.
_should_log(level::Symbol) = _VERBOSITY_RANK[_VERBOSITY[]] >= _VERBOSITY_RANK[level]

# Log diagnostic chatter -- shown only at :verbose.
macro vinfo(args...)
    return quote
        if _should_log(:verbose)
            @info($(map(esc, args)...))
        end
    end
end

# Log essential summaries -- shown at :minimal and above.
macro minfo(args...)
    return quote
        if _should_log(:minimal)
            @info($(map(esc, args)...))
        end
    end
end

# Time an expression, printing the timing at :minimal and above; runs silently
# (no timing print) at :silent. Returns the value of the expression in all cases.
macro vtime(expr)
    return quote
        if _should_log(:minimal)
            @time $(esc(expr))
        else
            $(esc(expr))
        end
    end
end
