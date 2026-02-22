using DifferentialEvolution
using JSON3
using Random
using Test

@testset "DifferentialEvolution.optimize" begin
    f(x) = sum(abs2, x)
    lower = fill(-5.0, 3)
    upper = fill(5.0, 3)

    rng1 = MersenneTwister(1234)
    rng2 = MersenneTwister(1234)

    res1 = optimize(f, lower, upper; rng=rng1, maxiters=50)
    res2 = optimize(f, lower, upper; rng=rng2, maxiters=50)

    @test res1.best_x == res2.best_x
    @test res1.best_f == res2.best_f
    @test res1.history == res2.history
    @test length(res1.history) == res1.iterations
end

@testset "bounds respected" begin
    lower = [-1.0, -2.0]
    upper = [1.0, 2.0]

    function bounded_f(x)
        @test all(x .>= lower) && all(x .<= upper)
        return sum(abs2, x)
    end

    rng = MersenneTwister(2024)
    res = optimize(bounded_f, lower, upper; rng=rng, maxiters=30)

    @test all(res.best_x .>= lower)
    @test all(res.best_x .<= upper)
end

@testset "input validation" begin
    f(x) = sum(abs2, x)
    rng = MersenneTwister(1)

    @test_throws ArgumentError optimize(f, [0.0], [0.0]; rng=rng)
    @test_throws ArgumentError optimize(f, [0.0, 1.0], [1.0]; rng=rng)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, popsize=3)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, maxiters=0)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, maxevals=3, popsize=4)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, F=0.0)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, CR=1.5)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, algorithm=:unknown)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, algorithm=:shade, memory_size=0)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, algorithm=:shade, pmax=0.0)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, algorithm=:lshade, pmax=1.5)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, algorithm=:jso, pmax=1.5)
    @test_throws ArgumentError optimize(f, Float64[], Float64[]; rng=rng, algorithm=:jso)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, target=NaN)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, local_method=:unknown)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, local_maxiters=0)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, local_tol=0.0)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, message_every=0)
    @test_throws ArgumentError optimize(f, [0.0], [1.0]; rng=rng, message_mode=:verbose)
    @test_throws ArgumentError find_sublevel_point(f, [0.0], [1.0]; c=Inf, rng=rng)
    @test_throws ArgumentError find_sublevel_point(f, [0.0], [1.0]; c=0.0, c_tol=Inf, rng=rng)
    @test_throws ArgumentError find_sublevel_point(f, [0.0], [1.0]; c=0.0, c_tol=-1e-3, rng=rng)
end

@testset "sanity" begin
    f(x) = sum(abs2, x)
    lower = fill(-5.0, 5)
    upper = fill(5.0, 5)

    rng = MersenneTwister(42)
    res = optimize(f, lower, upper; rng=rng, maxiters=120, popsize=50)

    @test res.best_f < 1e-2
end

@testset "shade, lshade, and jso reproducibility" begin
    f(x) = sum(abs2, x)
    lower = fill(-3.0, 4)
    upper = fill(3.0, 4)

    for alg in (:shade, :lshade, :jso)
        rng1 = MersenneTwister(42)
        rng2 = MersenneTwister(42)

        res1 = optimize(f, lower, upper; rng=rng1, algorithm=alg, maxiters=60, popsize=20)
        res2 = optimize(f, lower, upper; rng=rng2, algorithm=alg, maxiters=60, popsize=20)

        @test res1.best_x == res2.best_x
        @test res1.best_f == res2.best_f
        @test res1.history == res2.history
    end
end

@testset "parallel reproducibility" begin
    f(x) = sum(abs2, x)
    lower = fill(-2.0, 3)
    upper = fill(2.0, 3)

    for alg in (:de, :shade, :lshade, :jso)
        rng1 = MersenneTwister(123)
        rng2 = MersenneTwister(123)

        res1 = optimize(f, lower, upper; rng=rng1, algorithm=alg, maxiters=40, popsize=12, parallel=true)
        res2 = optimize(f, lower, upper; rng=rng2, algorithm=alg, maxiters=40, popsize=12, parallel=true)

        @test res1.best_x == res2.best_x
        @test res1.best_f == res2.best_f
    end
end

@testset "shade, lshade, and jso bounds" begin
    lower = [-1.0, -2.0]
    upper = [1.0, 2.0]

    function bounded_f(x)
        @test all(x .>= lower) && all(x .<= upper)
        return sum(abs2, x)
    end

    for (i, alg) in enumerate((:shade, :lshade, :jso))
        rng = MersenneTwister(2024 + i)
        res = optimize(bounded_f, lower, upper; rng=rng, algorithm=alg, maxiters=30, popsize=12)

        @test all(res.best_x .>= lower)
        @test all(res.best_x .<= upper)
    end
end

@testset "shade, lshade, and jso sanity" begin
    f(x) = sum(abs2, x)
    lower = fill(-5.0, 3)
    upper = fill(5.0, 3)

    for alg in (:shade, :lshade, :jso)
        rng = MersenneTwister(77)
        res = optimize(f, lower, upper; rng=rng, algorithm=alg, maxiters=80, popsize=30)
        @test res.best_f < 1e-1
    end
end

@testset "shade parameter ranges" begin
    rng = MersenneTwister(11)
    dim = 3
    popsize = 8
    lower = fill(-1.0, dim)
    upper = fill(1.0, dim)
    pop = rand(rng, dim, popsize) .* 2 .- 1
    fitness = [sum(abs2, view(pop, :, i)) for i in 1:popsize]
    trial = Vector{Float64}(undef, dim)

    for alg in (:shade, :lshade, :jso)
        state = alg == :shade ?
            DifferentialEvolution._make_shade_state(Float64, popsize, popsize, 0.2) :
            alg == :lshade ?
            DifferentialEvolution._make_lshade_state(Float64, popsize, popsize, 0.2, 100) :
            DifferentialEvolution._make_jso_state(Float64, popsize, 5, 0.25, 100, 10)

        DifferentialEvolution._start_generation!(state, pop, fitness, rng, 1, 10, popsize, 100)
        for i in 1:popsize
            DifferentialEvolution._generate_trial!(state, trial, pop, fitness, i, rng, lower, upper, popsize)
            @test 0.0 <= state.cr_vals[i] <= 1.0
            @test 0.0 < state.f_vals[i] <= 1.0
            @test all(trial .>= lower) && all(trial .<= upper)
        end
    end
end

@testset "lshade population reduction" begin
    rng = MersenneTwister(123)
    dim = 2
    popsize = 10
    pop = reshape(collect(1.0:(dim * popsize)), dim, popsize)
    fitness = collect(1.0:popsize)

    state = DifferentialEvolution._make_lshade_state(Float64, popsize, popsize, 0.2, 100)
    new_pop, new_fit = DifferentialEvolution._end_generation!(state, pop, fitness, rng, 100)

    @test size(new_pop, 2) == 4
    @test length(new_fit) == 4
    @test state.archive_limit == 4
    @test new_fit == fitness[1:4]
end

@testset "shade memory update" begin
    rng = MersenneTwister(2)
    pop = reshape(collect(1.0:12.0), 2, 6)
    fitness = collect(1.0:6.0)

    state = DifferentialEvolution._make_shade_state(Float64, 6, 3, 0.2)
    state.success_cr = [0.2, 0.6]
    state.success_f = [0.4, 0.9]
    state.success_df = [1.0, 3.0]

    DifferentialEvolution._end_generation!(state, pop, fitness, rng, 10)

    expected_mcr = (0.2 * 1.0 + 0.6 * 3.0) / 4.0
    expected_mf = (1.0 * 0.4^2 + 3.0 * 0.9^2) / (1.0 * 0.4 + 3.0 * 0.9)

    @test isapprox(state.MCR[1], expected_mcr; atol=1e-12, rtol=0)
    @test isapprox(state.MF[1], expected_mf; atol=1e-12, rtol=0)
    @test state.k == 2
end

@testset "lshade memory update" begin
    rng = MersenneTwister(3)
    pop = reshape(collect(1.0:12.0), 2, 6)
    fitness = collect(1.0:6.0)

    state = DifferentialEvolution._make_lshade_state(Float64, 6, 2, 0.2, 100)
    state.success_cr = [0.0, 0.0]
    state.success_f = [0.5, 0.8]
    state.success_df = [1.0, 2.0]

    DifferentialEvolution._end_generation!(state, pop, fitness, rng, 10)

    expected_mf = (1.0 * 0.5^2 + 2.0 * 0.8^2) / (1.0 * 0.5 + 2.0 * 0.8)

    @test isnan(state.MCR[1])
    @test isapprox(state.MF[1], expected_mf; atol=1e-12, rtol=0)
    @test state.k == 2
end

@testset "lshade population schedule" begin
    state = DifferentialEvolution._make_lshade_state(Float64, 10, 10, 0.2, 100)
    @test DifferentialEvolution._lshade_population_size(state, 0) == 10
    @test DifferentialEvolution._lshade_population_size(state, 50) == 7
    @test DifferentialEvolution._lshade_population_size(state, 100) == 4
end

@testset "jso success-history and archive update" begin
    rng = MersenneTwister(9)
    parent_pop = [
        1.0 2.0 3.0 4.0;
        1.0 2.0 3.0 4.0;
    ]
    parent_fitness = [sum(abs2, view(parent_pop, :, i)) for i in 1:4]
    trial_pop = [
        0.0 2.0 10.0 0.5;
        0.0 2.0 10.0 0.5;
    ]
    trial_fitness = [sum(abs2, view(trial_pop, :, i)) for i in 1:4]
    jso_state = DifferentialEvolution._make_jso_state(Float64, 4, 5, 0.25, 100, 20)
    jso_state.cr_vals = [0.1, 0.2, 0.3, 0.4]
    jso_state.f_vals = [0.5, 0.6, 0.7, 0.8]

    for index in 1:4
        if trial_fitness[index] <= parent_fitness[index]
            DifferentialEvolution._on_success!(
                jso_state,
                index,
                view(trial_pop, :, index),
                trial_fitness[index],
                parent_fitness[index],
                parent_pop,
                rng,
            )
        end
    end

    @test jso_state.success_cr == [0.1, 0.4]
    @test jso_state.success_f == [0.5, 0.8]
    @test jso_state.success_df == [
        abs(trial_fitness[1] - parent_fitness[1]),
        abs(trial_fitness[4] - parent_fitness[4]),
    ]
    @test length(jso_state.archive) == 2
end

@testset "jso p schedule increases from pmin to pmax" begin
    rng = MersenneTwister(10)
    initial_popsize = 10
    jso_state = DifferentialEvolution._make_jso_state(Float64, initial_popsize, 5, 0.25, 100, 20)
    population = reshape(collect(1.0:(2 * initial_popsize)), 2, initial_popsize)
    fitness = collect(1.0:initial_popsize)

    @test isapprox(jso_state.p, jso_state.pmin; atol=1e-12, rtol=0)

    DifferentialEvolution._end_generation!(jso_state, population, fitness, rng, 20)
    @test isapprox(jso_state.p, jso_state.pmin + (jso_state.pmax - jso_state.pmin) * 0.2; atol=1e-12, rtol=0)

    DifferentialEvolution._end_generation!(jso_state, population, fitness, rng, 100)
    @test isapprox(jso_state.p, jso_state.pmax; atol=1e-12, rtol=0)
end

@testset "jso memory update and fixed memory entry" begin
    rng = MersenneTwister(11)
    initial_popsize = 10
    jso_state = DifferentialEvolution._make_jso_state(Float64, initial_popsize, 5, 0.25, 100, 20)
    population = reshape(collect(1.0:(2 * initial_popsize)), 2, initial_popsize)
    fitness = collect(1.0:initial_popsize)

    jso_state.success_cr = [0.2, 0.8]
    jso_state.success_f = [0.4, 0.9]
    jso_state.success_df = [1.0, 3.0]
    initial_mcr = jso_state.MCR[1]
    initial_mf = jso_state.MF[1]
    expected_mean_cr = (1.0 * 0.2^2 + 3.0 * 0.8^2) / (1.0 * 0.2 + 3.0 * 0.8)
    expected_mean_f = (1.0 * 0.4^2 + 3.0 * 0.9^2) / (1.0 * 0.4 + 3.0 * 0.9)

    DifferentialEvolution._end_generation!(jso_state, population, fitness, rng, 20)

    @test isapprox(jso_state.MCR[1], (initial_mcr + expected_mean_cr) / 2; atol=1e-12, rtol=0)
    @test isapprox(jso_state.MF[1], (initial_mf + expected_mean_f) / 2; atol=1e-12, rtol=0)
    @test jso_state.k == 2

    fixed_index = jso_state.fixed_idx
    @test fixed_index == 5
    jso_state.k = fixed_index
    jso_state.success_cr = [0.5]
    jso_state.success_f = [0.6]
    jso_state.success_df = [1.0]
    fixed_mcr_before = jso_state.MCR[fixed_index]
    fixed_mf_before = jso_state.MF[fixed_index]

    DifferentialEvolution._end_generation!(jso_state, population, fitness, rng, 40)

    @test jso_state.k == 1
    @test jso_state.MCR[fixed_index] == fixed_mcr_before
    @test jso_state.MF[fixed_index] == fixed_mf_before
end

@testset "history off" begin
    f(x) = sum(abs2, x)
    lower = fill(-2.0, 2)
    upper = fill(2.0, 2)

    rng = MersenneTwister(7)
    res = optimize(f, lower, upper; rng=rng, maxiters=10, history=false)

    @test isempty(res.history)
    @test res.iterations == 10
end

@testset "non-finite objective values are treated as Inf" begin
    function always_nan(_)
        return NaN
    end
    function always_string(_)
        return "not-a-number"
    end

    lower = fill(0.0, 2)
    upper = fill(1.0, 2)
    rng = MersenneTwister(808)
    res = optimize(always_nan, lower, upper; rng=rng, maxiters=8, popsize=8, history=false, target=-Inf)
    rng_string = MersenneTwister(809)
    res_string = optimize(always_string, lower, upper; rng=rng_string, maxiters=8, popsize=8, history=false, target=-Inf)

    @test isinf(res.best_f)
    @test res.status == :not_reached
    @test res.de_status in (:maxiters, :maxevals)
    @test isinf(res_string.best_f)
    @test res_string.status == :not_reached
    @test res_string.de_status in (:maxiters, :maxevals)
end

@testset "higher dimension function" begin
    function rosenbrock(x)
        s = 0.0
        for i in 1:length(x)-1
            s += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
        end
        return s
    end

    lower = fill(-2.0, 10)
    upper = fill(2.0, 10)

    rng = MersenneTwister(2025)
    res = optimize(rosenbrock, lower, upper; rng=rng, maxiters=200, popsize=60)

    @test res.best_f < 20.0
end

@testset "parallel trial generation order independence" begin
    f(x) = sum(abs2, x)
    dim = 4
    popsize = 6
    lower = fill(-5.0, dim)
    upper = fill(5.0, dim)

    base_rng = MersenneTwister(1234)
    pop = rand(base_rng, dim, popsize) .* (upper .- lower) .+ lower
    fitness = [f(view(pop, :, i)) for i in 1:popsize]

    for alg in (:de, :shade, :lshade, :jso)
        rng = MersenneTwister(2024)
        state = alg == :de ?
            DifferentialEvolution.DEStrategy(0.5, 0.9) :
            alg == :shade ?
            DifferentialEvolution._make_shade_state(Float64, popsize, popsize, 0.25) :
            alg == :lshade ?
            DifferentialEvolution._make_lshade_state(Float64, popsize, popsize, 0.25, 100) :
            DifferentialEvolution._make_jso_state(Float64, popsize, 5, 0.25, 100, 10)

        DifferentialEvolution._start_generation!(state, pop, fitness, rng, 1, 10, popsize, 100)
        seeds = [rand(rng, UInt64) for _ in 1:popsize]

        state_ref = deepcopy(state)
        state_shuf = deepcopy(state)

        trials_ref = Matrix{Float64}(undef, dim, popsize)
        trials_shuf = Matrix{Float64}(undef, dim, popsize)

        for i in 1:popsize
            local_rng = Random.Xoshiro(seeds[i])
            eval_count = popsize + (i - 1)
            DifferentialEvolution._generate_trial!(
                state_ref,
                view(trials_ref, :, i),
                pop,
                fitness,
                i,
                local_rng,
                lower,
                upper,
                eval_count,
            )
        end

        order = randperm(MersenneTwister(99), popsize)
        for i in order
            local_rng = Random.Xoshiro(seeds[i])
            eval_count = popsize + (i - 1)
            DifferentialEvolution._generate_trial!(
                state_shuf,
                view(trials_shuf, :, i),
                pop,
                fitness,
                i,
                local_rng,
                lower,
                upper,
                eval_count,
            )
        end

        @test trials_ref == trials_shuf
    end
end

@testset "parallel selection update integrity" begin
    f(x) = sum(abs2, x)
    dim = 2
    popsize = 4
    lower = fill(-5.0, dim)
    upper = fill(5.0, dim)

    parent_pop = [
        1.0 2.0 3.0 4.0;
        1.0 2.0 3.0 4.0;
    ]
    parent_fitness = [f(view(parent_pop, :, i)) for i in 1:popsize]

    trials = [
        0.0 2.0 10.0 0.5;
        0.0 2.0 10.0 0.5;
    ]
    f_trials = [f(view(trials, :, i)) for i in 1:popsize]

    pop = copy(parent_pop)
    fitness = copy(parent_fitness)

    state = DifferentialEvolution._make_shade_state(Float64, popsize, popsize, 0.25)
    state.cr_vals = [0.1, 0.2, 0.3, 0.4]
    state.f_vals = [0.5, 0.6, 0.7, 0.8]

    rng = MersenneTwister(7)
    accept = falses(popsize)
    for i in 1:popsize
        if f_trials[i] <= parent_fitness[i]
            accept[i] = true
            fitness[i] = f_trials[i]
            copyto!(view(pop, :, i), view(trials, :, i))
        end
    end

    for i in 1:popsize
        if accept[i]
            DifferentialEvolution._on_success!(
                state,
                i,
                view(trials, :, i),
                f_trials[i],
                parent_fitness[i],
                parent_pop,
                rng,
            )
        end
    end

    @test accept == Bool[true, true, false, true]
    @test fitness[1] == f_trials[1]
    @test fitness[2] == f_trials[2]
    @test fitness[3] == parent_fitness[3]
    @test fitness[4] == f_trials[4]
    @test pop[:, 1] == trials[:, 1]
    @test pop[:, 2] == trials[:, 2]
    @test pop[:, 3] == parent_pop[:, 3]
    @test pop[:, 4] == trials[:, 4]

    expected_cr = [0.1, 0.4]
    expected_f = [0.5, 0.8]
    expected_df = [abs(f_trials[1] - parent_fitness[1]), abs(f_trials[4] - parent_fitness[4])]
    @test state.success_cr == expected_cr
    @test state.success_f == expected_f
    @test state.success_df == expected_df
end

@testset "maxevals improves or maintains best_f" begin
    function sphere(x)
        return sum(abs2, x)
    end

    function ackley(x)
        n = length(x)
        sumsq = 0.0
        sumcos = 0.0
        for i in 1:n
            sumsq += x[i]^2
            sumcos += cos(2.0 * pi * x[i])
        end
        term1 = -20.0 * exp(-0.2 * sqrt(sumsq / n))
        term2 = -exp(sumcos / n)
        return term1 + term2 + 20.0 + exp(1.0)
    end

    function rosenbrock(x)
        s = 0.0
        for i in 1:length(x)-1
            s += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
        end
        return s
    end

    tasks = [
        (sphere, -5.0, 5.0),
        (ackley, -5.0, 5.0),
        (rosenbrock, -2.0, 2.0),
    ]

    dim = 5
    popsize = 12
    maxevals_small = 500
    maxevals_large = 1500
    maxiters_small = max(1, Int(floor(maxevals_small / popsize)) - 1)
    maxiters_large = max(1, Int(floor(maxevals_large / popsize)) - 1)

    # L-SHADE/JSO adapt their behavior based on maxevals/maxiters,
    # so changing maxevals is not an extension of the same run.
    for alg in (:de, :shade)
        for (func, lower_v, upper_v) in tasks
            lower = fill(lower_v, dim)
            upper = fill(upper_v, dim)

            rng1 = MersenneTwister(2026)
            res_small = optimize(
                func,
                lower,
                upper;
                rng=rng1,
                algorithm=alg,
                popsize=popsize,
                maxiters=maxiters_small,
                maxevals=maxevals_small,
                target=-Inf,
                history=false,
            )

            rng2 = MersenneTwister(2026)
            res_large = optimize(
                func,
                lower,
                upper;
                rng=rng2,
                algorithm=alg,
                popsize=popsize,
                maxiters=maxiters_large,
                maxevals=maxevals_large,
                target=-Inf,
                history=false,
            )

            @test res_large.best_f <= res_small.best_f + 1e-12
        end
    end
end

@testset "maxiters early stop vs full (lshade/jso)" begin
    function sphere(x)
        return sum(abs2, x)
    end

    function ackley(x)
        n = length(x)
        sumsq = 0.0
        sumcos = 0.0
        for i in 1:n
            sumsq += x[i]^2
            sumcos += cos(2.0 * pi * x[i])
        end
        term1 = -20.0 * exp(-0.2 * sqrt(sumsq / n))
        term2 = -exp(sumcos / n)
        return term1 + term2 + 20.0 + exp(1.0)
    end

    function rosenbrock(x)
        s = 0.0
        for i in 1:length(x)-1
            s += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
        end
        return s
    end

    tasks = [
        (sphere, -5.0, 5.0),
        (ackley, -5.0, 5.0),
        (rosenbrock, -2.0, 2.0),
    ]

    dim = 5
    popsize = 12
    maxevals = 1500
    maxiters_small = 10
    maxiters_large = 40

    # L-SHADE does not use maxiters internally, so maxiters acts as a pure stop condition.
    for (func, lower_v, upper_v) in tasks
        lower = fill(lower_v, dim)
        upper = fill(upper_v, dim)

        rng1 = MersenneTwister(2027)
        res_small = optimize(
            func,
            lower,
            upper;
            rng=rng1,
            algorithm=:lshade,
            popsize=popsize,
            maxiters=maxiters_small,
            maxevals=maxevals,
            target=-Inf,
            history=false,
        )

        rng2 = MersenneTwister(2027)
        res_large = optimize(
            func,
            lower,
            upper;
            rng=rng2,
            algorithm=:lshade,
            popsize=popsize,
            maxiters=maxiters_large,
            maxevals=maxevals,
            target=-Inf,
            history=false,
        )

        @test res_large.best_f <= res_small.best_f + 1e-12
    end

    # JSO uses maxiters internally; instead verify monotone best_f over iterations.
    for (func, lower_v, upper_v) in tasks
        lower = fill(lower_v, dim)
        upper = fill(upper_v, dim)
        rng = MersenneTwister(2027)
        res = optimize(
            func,
            lower,
            upper;
            rng=rng,
            algorithm=:jso,
            popsize=popsize,
            maxiters=maxiters_large,
            maxevals=maxevals,
            target=-Inf,
            history=true,
        )
        for i in 2:length(res.history)
            @test res.history[i] <= res.history[i - 1] + 1e-12
        end
        @test res.best_f == res.history[end]
    end
end

@testset "local refinement improves or keeps DE best" begin
    f(x) = sum(abs2, x)
    lower = fill(-5.0, 4)
    upper = fill(5.0, 4)

    for method in (:nelder_mead, :lbfgs)
        rng_de = MersenneTwister(314)
        rng_hybrid = MersenneTwister(314)

        res_de = optimize(f, lower, upper; rng=rng_de, maxiters=60, popsize=24, history=false)
        res_hybrid = optimize(
            f,
            lower,
            upper;
            rng=rng_hybrid,
            maxiters=60,
            popsize=24,
            history=false,
            local_refine=true,
            local_method=method,
            local_maxiters=120,
            local_tol=1e-10,
        )

        @test res_hybrid.de_best_x == res_de.best_x
        @test res_hybrid.de_best_f == res_de.best_f
        @test res_hybrid.best_f <= res_hybrid.de_best_f + 1e-12
        @test res_hybrid.local_status in (:success, :stopped)
        @test res_hybrid.total_evaluations == res_hybrid.de_evaluations + res_hybrid.local_evaluations
        @test res_hybrid.evaluations == res_hybrid.total_evaluations
        @test res_hybrid.local_evaluations > 0
        @test isfinite(res_hybrid.elapsed_de_sec)
        @test isfinite(res_hybrid.elapsed_local_sec)
        @test isfinite(res_hybrid.elapsed_total_sec)
    end
end

@testset "status is final outcome and de_status is DE stop reason" begin
    f(x) = sum(abs2, x)
    lower = fill(-5.0, 4)
    upper = fill(5.0, 4)

    rng_target = MersenneTwister(515)
    res_target = optimize(
        f,
        lower,
        upper;
        rng=rng_target,
        maxiters=1,
        popsize=20,
        target=1e-12,
        history=false,
        local_refine=true,
        local_method=:lbfgs,
        local_maxiters=300,
        local_tol=1e-12,
    )
    @test res_target.de_status == :maxiters
    @test res_target.status == (res_target.best_f <= 1e-12 ? :target_reached : :not_reached)

    rng_no_target = MersenneTwister(516)
    res_no_target = optimize(
        f,
        lower,
        upper;
        rng=rng_no_target,
        maxiters=5,
        popsize=20,
        target=-Inf,
        history=false,
    )
    @test res_no_target.status == :not_reached
    @test res_no_target.de_status == :maxiters
end

@testset "find_sublevel_point returns overlap=true and stops immediately when threshold is hit" begin
    lower = fill(-1.0, 2)
    upper = fill(1.0, 2)

    for algorithm in (:de, :shade, :lshade)
        evaluation_counter = Ref(0)
        function threshold_objective(_)
            evaluation_counter[] += 1
            return evaluation_counter[] >= 15 ? -1.0 : 1.0
        end

        rng = MersenneTwister(1201)
        result = find_sublevel_point(
            threshold_objective,
            lower,
            upper;
            c=0.0,
            rng=rng,
            algorithm=algorithm,
            popsize=8,
            maxiters=50,
            maxevals=200,
        )

        @test result.overlap == true
        @test result.best_f <= 0.0
        @test result.stop_reason == :overlap_found
        @test result.evaluations == 15
        @test evaluation_counter[] == result.evaluations
    end
end

@testset "find_sublevel_point returns overlap=false when threshold is not found" begin
    lower = fill(-1.0, 3)
    upper = fill(1.0, 3)

    for algorithm in (:de, :shade, :lshade)
        constant_objective(_) = 10.0
        rng = MersenneTwister(1202)
        result = find_sublevel_point(
            constant_objective,
            lower,
            upper;
            c=0.0,
            rng=rng,
            algorithm=algorithm,
            popsize=8,
            maxiters=100,
            maxevals=30,
        )

        @test result.overlap == false
        @test result.best_f == 10.0
        @test result.stop_reason == :maxevals
        @test result.evaluations == 30
    end
end

@testset "local refinement lbfgs falls back to finite diff" begin
    function no_dual_numbers(x)
        if !(eltype(x) <: AbstractFloat)
            error("unsupported element type")
        end
        return sum(abs2, x)
    end

    lower = fill(-2.0, 3)
    upper = fill(2.0, 3)
    warning_text = mktemp() do _, temp_io
        res_local = nothing
        redirect_stderr(temp_io) do
            rng = MersenneTwister(91)
            res_local = optimize(
                no_dual_numbers,
                lower,
                upper;
                rng=rng,
                maxiters=40,
                popsize=20,
                local_refine=true,
                local_method=:lbfgs,
                history=false,
                message=false,
            )
        end
        flush(temp_io)
        seekstart(temp_io)
        warning_log = read(temp_io, String)
        @test occursin("lbfgs autodiff=:forward failed; retrying autodiff=:finite", warning_log)
        res = res_local::Result
        @test res.local_status in (:success, :stopped)
        @test res.best_f <= res.de_best_f + 1e-12
        @test res.local_evaluations > 0
        return warning_log
    end

    @test !isempty(warning_text)
end

@testset "trace history and settings are recorded" begin
    f(x) = sum(abs2, x)
    lower = fill(-3.0, 3)
    upper = fill(3.0, 3)
    rng = MersenneTwister(202)
    res = optimize(
        f,
        lower,
        upper;
        rng=rng,
        algorithm=:shade,
        popsize=12,
        maxiters=20,
        history=false,
        local_refine=true,
        local_method=:nelder_mead,
        local_maxiters=50,
        local_tol=1e-9,
        trace_history=true,
        job_id=77,
    )

    de_rows = filter(trace_row -> trace_row.phase == :de_generation, res.trace)
    @test length(de_rows) == res.iterations
    @test all(trace_row.job_id == 77 for trace_row in res.trace)
    @test all(de_rows[index].generation <= de_rows[index + 1].generation for index in 1:length(de_rows)-1)
    @test de_rows[end].best_f == res.de_best_f
    @test de_rows[end].best_x == res.de_best_x

    local_start_index = findfirst(trace_row -> trace_row.phase == :local_start, res.trace)
    local_end_index = findfirst(trace_row -> trace_row.phase == :local_end, res.trace)
    @test !isnothing(local_start_index)
    @test !isnothing(local_end_index)
    @test res.trace[local_start_index].best_f == res.de_best_f
    @test res.trace[local_start_index].best_x == res.de_best_x
    @test res.trace[local_end_index].best_f == res.local_best_f
    @test res.trace[local_end_index].best_x == res.local_best_x
    @test res.settings.algorithm == :shade
    @test res.settings.popsize == 12
    @test res.settings.local_refine == true
    @test res.settings.local_method == :nelder_mead
    @test res.settings.trace_history == true
    @test res.settings.job_id == 77
    @test res.settings.message == false
    @test res.settings.message_every == 1
    @test res.settings.message_mode == :compact
end

@testset "trace csv export" begin
    f(x) = sum(abs2, x)
    lower = fill(-2.0, 2)
    upper = fill(2.0, 2)
    rng = MersenneTwister(303)
    res = optimize(
        f,
        lower,
        upper;
        rng=rng,
        popsize=10,
        maxiters=12,
        history=false,
        trace_history=true,
        job_id=5,
    )

    mktempdir() do temp_dir
        output_path = joinpath(temp_dir, "trace.csv")
        write_trace_csv(res, output_path)
        @test isfile(output_path)
        csv_lines = readlines(output_path)
        @test length(csv_lines) == length(res.trace) + 1
        @test startswith(csv_lines[1], "job_id,generation,phase,evaluations,best_f,best_x_1,best_x_2")
        first_data_cells = split(csv_lines[2], ",")
        @test parse(Int, first_data_cells[1]) == 5
        @test first_data_cells[3] == "de_generation"
    end
end

@testset "message progress output" begin
    f(x) = sum(abs2, x)
    lower = fill(-2.0, 2)
    upper = fill(2.0, 2)

    message_text = mktemp() do _, temp_io
        redirect_stdout(temp_io) do
            rng = MersenneTwister(404)
            optimize(
                f,
                lower,
                upper;
                rng=rng,
                popsize=10,
                maxiters=6,
                history=false,
                local_refine=true,
                local_method=:nelder_mead,
                local_maxiters=20,
                message=true,
                message_every=2,
                job_id=11,
            )
        end
        flush(temp_io)
        seekstart(temp_io)
        read(temp_io, String)
    end
    @test occursin("[DE] job=11 generation=2/6", message_text)
    @test occursin("[DE] job=11 generation=4/6", message_text)
    @test occursin("[DE] job=11 generation=6/6", message_text)
    @test !occursin("[DE] job=11 generation=1/6", message_text)
    @test occursin("delta_best=", message_text)
    @test occursin("stall_generations=", message_text)
    @test !occursin("best_x=", message_text)
    @test occursin("[LOCAL-START] job=11", message_text)
    @test occursin("[LOCAL-END] job=11", message_text)
end

@testset "message detailed output includes best_x" begin
    f(x) = sum(abs2, x)
    lower = fill(-2.0, 2)
    upper = fill(2.0, 2)

    message_text = mktemp() do _, temp_io
        redirect_stdout(temp_io) do
            rng = MersenneTwister(405)
            optimize(
                f,
                lower,
                upper;
                rng=rng,
                popsize=10,
                maxiters=4,
                history=false,
                local_refine=true,
                local_method=:nelder_mead,
                local_maxiters=20,
                message=true,
                message_every=2,
                message_mode=:detailed,
                job_id=12,
            )
        end
        flush(temp_io)
        seekstart(temp_io)
        read(temp_io, String)
    end

    @test occursin("[DE] job=12 generation=2/4", message_text)
    @test occursin("best_x=", message_text)
    @test occursin("[LOCAL-START] job=12", message_text)
    @test occursin("[LOCAL-END] job=12", message_text)
end

@testset "summarize script skips malformed json files" begin
    mktempdir() do temp_dir
        valid_payload = Dict(
            "seed" => 1,
            "best_f" => 0.1,
            "status" => "not_reached",
            "de_status" => "maxiters",
            "local_status" => "disabled",
        )
        open(joinpath(temp_dir, "seed_1.json"), "w") do io
            JSON3.pretty(io, valid_payload)
        end
        write(joinpath(temp_dir, "seed_2.json"), "{not-json")

        project_dir = dirname(dirname(pathof(DifferentialEvolution)))
        run(
            `$(Base.julia_cmd()) --project=$project_dir $(joinpath(project_dir, "scripts", "summarize_runs.jl")) --results_dir $temp_dir --top_k 5`,
        )

        summary_data = JSON3.read(read(joinpath(temp_dir, "summary.json"), String), Dict{String, Any})
        @test summary_data["num_runs"] == 1
        @test summary_data["num_skipped_runs"] == 1
        @test occursin("invalid_json", String(summary_data["skipped_runs"][1]["reason"]))
    end
end

@testset "archive push matches reference semantics" begin
    function archive_push_reference!(archive, limit, vec, rng)
        if limit <= 0
            return nothing
        end
        push!(archive, copy(vec))
        if length(archive) > limit
            idx = rand(rng, 1:length(archive))
            archive[idx] = archive[end]
            pop!(archive)
        end
        return nothing
    end

    rng_reference = MersenneTwister(12345)
    rng_current = MersenneTwister(12345)
    archive_reference = Vector{Vector{Float64}}()
    archive_current = Vector{Vector{Float64}}()
    vector_rng = MersenneTwister(67890)

    for _ in 1:200
        vec = rand(vector_rng, 4)
        archive_push_reference!(archive_reference, 7, vec, rng_reference)
        DifferentialEvolution._archive_push!(archive_current, 7, vec, rng_current)
        @test archive_current == archive_reference
    end

    @test rand(rng_current, UInt64) == rand(rng_reference, UInt64)

    rng_reference_zero = MersenneTwister(2027)
    rng_current_zero = MersenneTwister(2027)
    archive_reference_zero = [[1.0, 2.0]]
    archive_current_zero = [[1.0, 2.0]]
    vec = [3.0, 4.0]
    archive_push_reference!(archive_reference_zero, 0, vec, rng_reference_zero)
    DifferentialEvolution._archive_push!(archive_current_zero, 0, vec, rng_current_zero)
    @test archive_current_zero == archive_reference_zero
    @test rand(rng_current_zero, UInt64) == rand(rng_reference_zero, UInt64)
end

@testset "message and trace flags do not change optimization results" begin
    f(x) = sum(abs2, x)
    lower = fill(-3.0, 4)
    upper = fill(3.0, 4)

    for (offset, alg) in enumerate((:de, :shade, :lshade, :jso))
        seed = 3000 + offset
        rng_base = MersenneTwister(seed)
        res_base = optimize(
            f,
            lower,
            upper;
            rng=rng_base,
            algorithm=alg,
            maxiters=40,
            popsize=20,
            history=true,
            message=false,
            trace_history=false,
        )

        res_compact = nothing
        redirect_stdout(devnull) do
            rng_compact = MersenneTwister(seed)
            res_compact = optimize(
                f,
                lower,
                upper;
                rng=rng_compact,
                algorithm=alg,
                maxiters=40,
                popsize=20,
                history=true,
                message=true,
                message_every=2,
                message_mode=:compact,
                trace_history=false,
            )
        end
        compact_result = res_compact::Result

        res_detailed = nothing
        redirect_stdout(devnull) do
            rng_detailed = MersenneTwister(seed)
            res_detailed = optimize(
                f,
                lower,
                upper;
                rng=rng_detailed,
                algorithm=alg,
                maxiters=40,
                popsize=20,
                history=true,
                message=true,
                message_every=2,
                message_mode=:detailed,
                trace_history=false,
            )
        end
        detailed_result = res_detailed::Result

        rng_trace = MersenneTwister(seed)
        trace_result = optimize(
            f,
            lower,
            upper;
            rng=rng_trace,
            algorithm=alg,
            maxiters=40,
            popsize=20,
            history=true,
            message=false,
            trace_history=true,
        )

        for compared_result in (compact_result, detailed_result, trace_result)
            @test compared_result.best_x == res_base.best_x
            @test compared_result.best_f == res_base.best_f
            @test compared_result.history == res_base.history
            @test compared_result.de_best_x == res_base.de_best_x
            @test compared_result.de_best_f == res_base.de_best_f
            @test compared_result.evaluations == res_base.evaluations
            @test compared_result.iterations == res_base.iterations
            @test compared_result.de_status == res_base.de_status
            @test compared_result.status == res_base.status
        end
    end
end
