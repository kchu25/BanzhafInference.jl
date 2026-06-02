using BanzhafInference
using Test
using DataFrames
using Random
using Statistics

const BI = BanzhafInference

# These tests cover the CPU-only functionality (no CUDA / GPU required):
# small helpers, correlation metrics, functor wrappers, random-coalition
# generation, and the DataFrame preparation / subsampling utilities.

@testset "BanzhafInference.jl" begin

    @testset "helpers" begin
        # tail_reshape drops the leading dimension
        x = reshape(collect(1:6), (1, 2, 3))
        y = BI.tail_reshape(x)
        @test size(y) == (2, 3)
        @test vec(y) == collect(1:6)

        # a leading singleton on a 2D array
        @test size(BI.tail_reshape(reshape(collect(1:4), (1, 4)))) == (4,)

        # round2 rounds to two decimal places
        @test BI.round2(3.14159) == 3.14
        @test BI.round2(2.0) == 2.0
        @test BI.round2(-1.236) == -1.24
    end

    @testset "corr: r2_score" begin
        # perfect prediction -> 1.0
        @test BI.r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) ≈ 1.0

        # predicting the mean -> 0.0
        @test BI.r2_score([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) ≈ 0.0

        # NaNs are dropped; fewer than 2 valid points -> NaN
        @test isnan(BI.r2_score([NaN, 1.0], [1.0, 2.0]))
        @test isnan(BI.r2_score([NaN, NaN, NaN], [1.0, 2.0, 3.0]))

        # NaNs dropped but enough valid points remain
        @test BI.r2_score([1.0, 2.0, 3.0, NaN], [1.0, 2.0, 3.0, 5.0]) ≈ 1.0

        # constant truth: zero residual -> 1.0, nonzero residual -> -Inf
        @test BI.r2_score([2.0, 2.0], [2.0, 2.0]) ≈ 1.0
        @test BI.r2_score([2.0, 2.0], [3.0, 3.0]) == -Inf
    end

    @testset "corr: pearson_r" begin
        # perfect positive / negative correlation
        @test BI.pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) ≈ 1.0
        @test BI.pearson_r([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) ≈ -1.0

        # symmetric in its arguments
        a = [1.0, 5.0, 2.0, 8.0]
        b = [2.0, 4.0, 1.0, 7.0]
        @test BI.pearson_r(a, b) ≈ BI.pearson_r(b, a)

        # in range
        @test -1.0 <= BI.pearson_r(a, b) <= 1.0

        # constant input -> zero denominator -> 0.0
        @test BI.pearson_r([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

        # fewer than 2 valid points -> NaN
        @test isnan(BI.pearson_r([NaN, 1.0], [1.0, 2.0]))
    end

    @testset "functor wrappers" begin
        f1 = BI.Functor1Arg(x -> x^2)
        @test f1(3) == 9

        f2 = BI.Functor2Arg((x, n) -> x^n, 2)
        @test f2(3) == 9

        # FunctorWrapper picks the right type from the arguments
        w1 = BI.FunctorWrapper(x -> x + 1)
        @test w1 isa BI.Functor1Arg
        @test w1(4) == 5

        w2 = BI.FunctorWrapper((x, n) -> x^n, 3)
        @test w2 isa BI.Functor2Arg
        @test w2(2) == 8

        # default (no second arg) is the 1-arg variant
        @test BI.FunctorWrapper(identity) isa BI.Functor1Arg
    end

    @testset "generate_random_coalitions_cpu" begin
        n_items, n_coalitions, min_size, max_size = 10, 50, 2, 5
        coalitions = BI.generate_random_coalitions_cpu(
            n_items, n_coalitions, min_size, max_size; seed=123)

        @test length(coalitions) == n_coalitions
        @test eltype(coalitions) == Vector{Int}

        for c in coalitions
            @test min_size <= length(c) <= max_size   # size within bounds
            @test all(1 .<= c .<= n_items)            # valid indices
            @test issorted(c)                          # sorted output
            @test length(unique(c)) == length(c)       # no repeats
        end

        # reproducible with the same seed, different without
        c1 = BI.generate_random_coalitions_cpu(20, 30, 2, 8; seed=42)
        c2 = BI.generate_random_coalitions_cpu(20, 30, 2, 8; seed=42)
        @test c1 == c2
    end

    @testset "extract_coalition_and_remainder" begin
        contribs = Float32[10, 20, 30, 40]
        coalition = [1, 3]
        coalition_sum, remainder, selected, unselected =
            BI.extract_coalition_and_remainder(contribs, coalition)

        @test coalition_sum == 40.0f0          # 10 + 30
        @test collect(remainder) == Float32[20, 40]
        @test selected == coalition
        @test unselected == [2, 4]

        # partition is complete: coalition_sum + remainder_sum == total
        @test coalition_sum + sum(remainder) ≈ sum(contribs)
        @test sort(vcat(selected, unselected)) == collect(1:length(contribs))
    end

    @testset "filtering_data_pts" begin
        df = DataFrame(
            data_pt_index = [1, 1, 1, 2, 2, 3],
            contribution  = Float32[1, 2, 3, 4, 5, 6],
        )

        # keep groups with more than 2 rows -> only data point 1 survives
        out2 = BI.filtering_data_pts(df; n_rows_threshold=2)
        @test sort(unique(out2.data_pt_index)) == [1]
        @test nrow(out2) == 3

        # keep groups with more than 1 row -> data points 1 and 2 survive
        out1 = BI.filtering_data_pts(df; n_rows_threshold=1)
        @test sort(unique(out1.data_pt_index)) == [1, 2]
        @test nrow(out1) == 5
    end

    @testset "extract_leave_one_out_vectors" begin
        df = DataFrame(
            data_pt_index = [1, 1, 1, 2, 2],
            contribution  = Float32[1, 2, 3, 10, 20],
        )
        loo = collect(BI.extract_leave_one_out_vectors(df))

        # one leave-one-out vector per row
        @test length(loo) == nrow(df)

        # group 1 (size 3): each vector omits one element in turn
        @test collect(loo[1]) == Float32[2, 3]
        @test collect(loo[2]) == Float32[1, 3]
        @test collect(loo[3]) == Float32[1, 2]

        # group 2 (size 2)
        @test collect(loo[4]) == Float32[20]
        @test collect(loo[5]) == Float32[10]
    end

    @testset "sort_by_vector_length" begin
        vectors = [Float32[1, 2, 3], Float32[1], Float32[1, 2]]
        df = DataFrame(id = [1, 2, 3])

        sorted_vectors, sorted_df = BI.sort_by_vector_length(vectors, df)

        @test length.(sorted_vectors) == [1, 2, 3]   # ascending by length
        @test sorted_df.id == [2, 3, 1]              # df rows follow the sort
    end

    @testset "prepare_banzhaf_data" begin
        df = DataFrame(
            data_pt_index = [2, 1, 1, 1, 3, 2],   # unsorted, groups of sizes 2/3/1
            contribution  = Float32[1, 2, 3, 4, 5, 6],
        )

        vectors, df_filtered = BI.prepare_banzhaf_data(df; n_rows_threshold=2)

        # only group 1 (3 rows) survives the >2 threshold, and is sorted
        @test issorted(df_filtered.data_pt_index)
        @test sort(unique(df_filtered.data_pt_index)) == [1]
        @test nrow(df_filtered) == 3

        # one leave-one-out vector per surviving row
        @test length(collect(vectors)) == nrow(df_filtered)
    end

    @testset "subsample_contributions" begin
        Random.seed!(0)
        df = DataFrame(
            data_pt_index = vcat(fill(1, 20), fill(2, 3), fill(3, 7)),
            contribution  = Float32.(1:30),
        )

        sampled = BI.subsample_contributions(df; max_rows_per_group=5,
                                             seed=1, verbose=false)

        # each group is capped at max_rows_per_group; smaller groups untouched
        gsizes = [nrow(g) for g in groupby(sampled, :data_pt_index)]
        @test all(gsizes .<= 5)
        @test sort(unique(sampled.data_pt_index)) == [1, 2, 3]

        # group 2 (only 3 rows) is kept in full
        g2 = sampled[sampled.data_pt_index .== 2, :]
        @test nrow(g2) == 3

        # reproducible with a fixed seed
        s1 = BI.subsample_contributions(df; max_rows_per_group=5, seed=7, verbose=false)
        s2 = BI.subsample_contributions(df; max_rows_per_group=5, seed=7, verbose=false)
        @test s1.contribution == s2.contribution
    end

end
