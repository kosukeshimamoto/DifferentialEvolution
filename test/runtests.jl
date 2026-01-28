using DifferentialEvolution
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

@testset "history off" begin
    f(x) = sum(abs2, x)
    lower = fill(-2.0, 2)
    upper = fill(2.0, 2)

    rng = MersenneTwister(7)
    res = optimize(f, lower, upper; rng=rng, maxiters=10, history=false)

    @test isempty(res.history)
    @test res.iterations == 10
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
