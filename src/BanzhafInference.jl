module BanzhafInference

using RealLabelNormalization
using Flux, DataFrames, CUDA
using StatsBase, Random
using EpicHyperSketch
using GLM, HypothesisTests, MultipleTesting, Printf
using SpecialFunctions

const FloatType = Float32
const IntType = Int32

using ProgressMeter
using Flux: gpu, cpu, gradient
using StatsBase: mean

include("helpers.jl")
include("influence/influence.jl")
# include("thresholding/thresholding.jl")
include("corr.jl")

include("const.jl")
include("functor.jl")
include("prep_banzhaf_compute.jl")
include("banzhaf/banzhaf.jl")
include("random_coalition.jl")
include("subsample.jl")
include("setup.jl")
include("filter.jl")
include("motifs_helpers.jl")
include("motifs.jl")
include("interactions.jl")
include("significance.jl")
include("significance_gpu.jl")

export compute_and_filter_contributions
export compute_random_coalition_banzhafs_per_datapoint
export compute_random_coalition_banzhafs_all_datapoints



end
