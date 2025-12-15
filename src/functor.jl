# Functor types for wrapping functions with optional arguments
# These allow functions to be stored in structs and passed around
# while maintaining type stability

"""
    Functor1Arg{F}

Wraps a single-argument function for use in structs.

# Example
```julia
f = Functor1Arg(x -> x^2)
f(3)  # Returns 9
```
"""
struct Functor1Arg{F}
    functor::F
end

"""
    Functor2Arg{F}

Wraps a two-argument function where the second argument is fixed.

# Example
```julia
f = Functor2Arg((x, n) -> x^n, 2)
f(3)  # Returns 9 (3^2)
```
"""
struct Functor2Arg{F}
    functor::F
    second_arg::Int
end

# Make functors callable
@inline (f::Functor1Arg)(x) = f.functor(x)
@inline (f::Functor2Arg)(x) = f.functor(x, f.second_arg)

"""
    FunctorWrapper(f, arg=nothing)

Helper constructor to create the appropriate Functor type.

# Arguments
- `f`: The function to wrap
- `arg`: Optional second argument (Int). If provided, creates Functor2Arg, otherwise Functor1Arg

# Examples
```julia
# Single-argument function
f1 = FunctorWrapper(x -> x^2)  # Creates Functor1Arg

# Two-argument function with fixed second arg
f2 = FunctorWrapper((x, n) -> x^n, 2)  # Creates Functor2Arg
```
"""
FunctorWrapper(f::F, arg::Int) where {F} = Functor2Arg(f, arg)
FunctorWrapper(f::F, ::Nothing=nothing) where {F} = Functor1Arg(f)
