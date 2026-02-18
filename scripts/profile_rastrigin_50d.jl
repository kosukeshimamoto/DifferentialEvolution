using DifferentialEvolution
using Random
using Statistics
using Profile
using Printf
using Dates

function rastrigin(x)
    n = length(x)
    s = 10.0 * n
    @inbounds for i in 1:n
        s += x[i]^2 - 10.0 * cos(2.0 * pi * x[i])
    end
    return s
end

function run_profile(alg; dim=50, maxevals=200000, popsize=250, F=0.5, CR=0.9, pmax=0.25, profile_runs=3)
    lower = fill(-5.12, dim)
    upper = fill(5.12, dim)
    maxiters = max(1, Int(floor(maxevals / popsize)) - 1)

    # warmup
    rng = MersenneTwister(2024)
    DifferentialEvolution.optimize(
        rastrigin,
        lower,
        upper;
        rng=rng,
        algorithm=alg,
        popsize=popsize,
        maxiters=maxiters,
        maxevals=maxevals,
        F=F,
        CR=CR,
        pmax=pmax,
        history=false,
    )

    rng = MersenneTwister(2024)
    elapsed = @elapsed DifferentialEvolution.optimize(
        rastrigin,
        lower,
        upper;
        rng=rng,
        algorithm=alg,
        popsize=popsize,
        maxiters=maxiters,
        maxevals=maxevals,
        F=F,
        CR=CR,
        pmax=pmax,
        history=false,
    )

    rng = MersenneTwister(2024)
    alloc = @allocated DifferentialEvolution.optimize(
        rastrigin,
        lower,
        upper;
        rng=rng,
        algorithm=alg,
        popsize=popsize,
        maxiters=maxiters,
        maxevals=maxevals,
        F=F,
        CR=CR,
        pmax=pmax,
        history=false,
    )

    Profile.clear()
    @profile begin
        for r in 1:profile_runs
            rng = MersenneTwister(2024 + r)
            DifferentialEvolution.optimize(
                rastrigin,
                lower,
                upper;
                rng=rng,
                algorithm=alg,
                popsize=popsize,
                maxiters=maxiters,
                maxevals=maxevals,
                F=F,
                CR=CR,
                pmax=pmax,
                history=false,
            )
        end
    end

    return elapsed, alloc
end

function main()
    outdir = joinpath(pwd(), "reports", "benchmarks", "profiles")
    mkpath(outdir)
    timestamp = Dates.format(Dates.now(), dateformat"yyyy-mm-dd HH:MM:SS")

    alg_arg = get(ENV, "PROFILE_ALGS", "de,shade,lshade,jso")
    algs = Symbol.(filter(!isempty, strip.(split(alg_arg, ","))))
    dim = parse(Int, get(ENV, "PROFILE_DIM", "50"))
    maxevals = parse(Int, get(ENV, "PROFILE_MAXEVALS", "200000"))
    popsize = parse(Int, get(ENV, "PROFILE_POPSZ", "250"))
    profile_runs = parse(Int, get(ENV, "PROFILE_RUNS", "3"))
    profile_prefix = "rastrigin_" * string(dim) * "d"

    summary_path = joinpath(outdir, profile_prefix * "_summary.txt")
    open(summary_path, "w") do io
        println(io, "Rastrigin ", dim, "D profile summary")
        println(io, "generated: ", timestamp)
        println(io, "maxevals: ", maxevals)
        println(io, "popsize: ", popsize)
        println(io, "profile runs per algorithm: ", profile_runs)
        println(io, "F: 0.5, CR: 0.9, pmax: 0.25")
        println(io)
        for alg in algs
            elapsed, alloc = run_profile(alg; dim=dim, maxevals=maxevals, popsize=popsize, profile_runs=profile_runs)
            per_eval = elapsed / maxevals
            println(io, @sprintf("%s: elapsed=%.3f s, per_eval=%.3e s, allocated=%.1f MB", uppercase(String(alg)), elapsed, per_eval, alloc / 1024^2))

            profile_path = joinpath(outdir, profile_prefix * "_profile_" * String(alg) * ".txt")
            open(profile_path, "w") do pio
                Profile.print(pio; format=:flat, sortedby=:count, maxdepth=20)
            end
        end
    end

    println("saved: ", summary_path)
end

main()
