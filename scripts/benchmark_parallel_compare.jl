using DifferentialEvolution
using Random
using Statistics
using Printf
using Dates
using DelimitedFiles
using Base.Threads

function sphere(x)
    s = 0.0
    for i in 1:length(x)
        s += x[i]^2
    end
    return s
end

function rosenbrock(x)
    s = 0.0
    for i in 1:length(x)-1
        s += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
    end
    return s
end

function rastrigin(x)
    n = length(x)
    s = 10.0 * n
    for i in 1:n
        s += x[i]^2 - 10.0 * cos(2.0 * pi * x[i])
    end
    return s
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

function griewank(x)
    sumsq = 0.0
    prodcos = 1.0
    for i in 1:length(x)
        sumsq += x[i]^2
        prodcos *= cos(x[i] / sqrt(i))
    end
    return sumsq / 4000.0 - prodcos + 1.0
end

function schwefel(x)
    n = length(x)
    s = 0.0
    for i in 1:n
        s += x[i] * sin(sqrt(abs(x[i])))
    end
    return 418.9829 * n - s
end

const _HEAVY_SINK = Threads.Atomic{Float64}(0.0)

@noinline function heavy_work(x, iters)
    s = 0.0
    n = length(x)
    @inbounds for k in 1:iters
        s += sin(x[1] + k * 0.001) * cos(x[n] - k * 0.002)
    end
    Threads.atomic_xchg!(_HEAVY_SINK, s)
    return nothing
end

function maybe_heavy(f, iters)
    if iters <= 0
        return f
    end
    return x -> (heavy_work(x, iters); f(x))
end

function parse_int_list(env_name, default_values)
    raw = get(ENV, env_name, "")
    if isempty(raw)
        return default_values
    end
    values = Int[]
    for part in split(raw, ",")
        part = strip(part)
        if !isempty(part)
            push!(values, parse(Int, part))
        end
    end
    return isempty(values) ? default_values : values
end

function mean_std(values::Vector{Float64})
    finite = filter(isfinite, values)
    if isempty(finite)
        return (mean=NaN, std=NaN)
    end
    m = mean(finite)
    s = length(finite) > 1 ? std(finite) : 0.0
    return (mean=m, std=s)
end

function format_float(x)
    if !isfinite(x)
        return "NA"
    end
    return @sprintf("%.6e", x)
end

function format_time(x)
    if !isfinite(x)
        return "NA"
    end
    return @sprintf("%.3f", x)
end

function run_de_trial(f, lower, upper, rng; algorithm, popsize, maxevals, F, CR, pmax, target, parallel)
    maxiters = max(1, Int(floor(maxevals / popsize)) - 1)
    t0 = time_ns()
    res = DifferentialEvolution.optimize(
        f,
        lower,
        upper;
        rng=rng,
        algorithm=algorithm,
        popsize=popsize,
        maxiters=maxiters,
        maxevals=maxevals,
        F=F,
        CR=CR,
        pmax=pmax,
        target=target,
        history=false,
        parallel=parallel,
    )
    elapsed = (time_ns() - t0) / 1e9
    return res.best_f, elapsed
end

function run_benchmark()
    maxevals = parse(Int, get(ENV, "DE_PAR_BENCH_MAXEVALS", "200000"))
    dims = parse_int_list("DE_PAR_BENCH_DIMS", [1, 5, 10, 50, 100])
    runs = parse(Int, get(ENV, "DE_PAR_BENCH_RUNS", "5"))
    base_seed = parse(Int, get(ENV, "DE_PAR_BENCH_SEED", "2025"))
    target_tol = parse(Float64, get(ENV, "DE_PAR_BENCH_TARGET_TOL", "1e-2"))
    de_F = parse(Float64, get(ENV, "DE_PAR_BENCH_F", "0.5"))
    de_CR = parse(Float64, get(ENV, "DE_PAR_BENCH_CR", "0.9"))
    pmax = parse(Float64, get(ENV, "DE_PAR_BENCH_PMAX", "0.25"))
    heavy_iters = parse(Int, get(ENV, "DE_PAR_BENCH_HEAVY_ITERS", "0"))
    tag = strip(get(ENV, "DE_PAR_BENCH_TAG", ""))

    outdir = joinpath(pwd(), "reports", "benchmarks", isempty(tag) ? "parallel_compare" : "parallel_compare_" * tag)
    mkpath(outdir)
    md_path = joinpath(outdir, "benchmark_summary.md")
    csv_path = joinpath(outdir, "benchmark_summary.csv")

    tasks = [
        (name="Sphere", f=maybe_heavy(sphere, heavy_iters), lower=-5.0, upper=5.0, optimum=0.0),
        (name="Rosenbrock", f=maybe_heavy(rosenbrock, heavy_iters), lower=-2.0, upper=2.0, optimum=0.0),
        (name="Rastrigin", f=maybe_heavy(rastrigin, heavy_iters), lower=-5.12, upper=5.12, optimum=0.0),
        (name="Ackley", f=maybe_heavy(ackley, heavy_iters), lower=-32.768, upper=32.768, optimum=0.0),
        (name="Griewank", f=maybe_heavy(griewank, heavy_iters), lower=-600.0, upper=600.0, optimum=0.0),
        (name="Schwefel", f=maybe_heavy(schwefel, heavy_iters), lower=-500.0, upper=500.0, optimum=0.0),
    ]

    algorithms = [
        ("DE", :de),
        ("SHADE", :shade),
        ("L-SHADE", :lshade),
        ("JSO", :jso),
    ]

    modes = [
        ("serial", false),
        ("parallel", true),
    ]

    results = NamedTuple[]
    config_id = 0
    for dim in dims
        for (mode_label, mode_flag) in modes
            for (alg_index, (alg_label, alg_id)) in enumerate(algorithms)
                config_id += 1
                popsize = max(5 * dim, 4)
                for task in tasks
                    lower = fill(task.lower, dim)
                    upper = fill(task.upper, dim)
                    target = task.optimum + target_tol

                    final_errors = Float64[]
                    times_list = Float64[]

                    for run in 1:runs
                        seed_alg = base_seed + run + 1000 * alg_index + 100000 * config_id + 10 * dim
                        rng = MersenneTwister(seed_alg)
                        best_f, elapsed = run_de_trial(
                            task.f,
                            lower,
                            upper,
                            rng;
                            algorithm=alg_id,
                            popsize=popsize,
                            maxevals=maxevals,
                            F=de_F,
                            CR=de_CR,
                            pmax=pmax,
                            target=target,
                            parallel=mode_flag,
                        )

                        push!(final_errors, abs(best_f - task.optimum) / dim)
                        push!(times_list, elapsed)
                    end

                    err_stats = mean_std(final_errors)
                    time_stats = mean_std(times_list)

                    push!(results, (
                        task=task.name,
                        dim=dim,
                        algorithm=alg_label,
                        mode=mode_label,
                        runs=runs,
                        maxevals=maxevals,
                        popsize=popsize,
                        threads=nthreads(),
                        error_mean=err_stats.mean,
                        error_std=err_stats.std,
                        time_mean=time_stats.mean,
                        time_std=time_stats.std,
                    ))
                end
            end
        end
    end

    timestamp = Dates.format(Dates.now(), dateformat"yyyy-mm-dd HH:MM:SS")

    open(md_path, "w") do io
        println(io, "# Parallel vs serial benchmark summary")
        println(io)
        println(io, "- generated: ", timestamp)
        println(io, "- runs: ", runs)
        println(io, "- dims: ", join(dims, ", "))
        println(io, "- maxevals: ", maxevals)
        println(io, "- popsize: 5 * D for all algorithms")
        println(io, "- algorithms: ", join(first.(algorithms), ", "))
        println(io, "- modes: serial, parallel")
        println(io, "- threads: ", nthreads())
        println(io, "- heavy_iters: ", heavy_iters)
        println(io, "- target_tol: ", target_tol)
        println(io, "- DE params: F=", de_F, ", CR=", de_CR, ", pmax=", pmax)
        println(io, "- error metric: mean(|f - f*|) / dim")
        println(io)

        for task in tasks
            println(io, "## ", task.name)
            println(io)
            println(io, "| Dim | Algorithm | Mode | Runs | Maxevals | Popsize | Threads | Error mean / dim | Error std / dim | Time mean (s) | Time std (s) |")
            println(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for dim in dims
                for (alg_label, _) in algorithms
                    for (mode_label, _) in modes
                        row = only(filter(r ->
                            r.task == task.name &&
                            r.dim == dim &&
                            r.algorithm == alg_label &&
                            r.mode == mode_label,
                            results
                        ))
                        println(
                            io,
                            dim, " | ",
                            alg_label, " | ",
                            mode_label, " | ",
                            row.runs, " | ",
                            row.maxevals, " | ",
                            row.popsize, " | ",
                            row.threads, " | ",
                            format_float(row.error_mean), " | ",
                            format_float(row.error_std), " | ",
                            format_time(row.time_mean), " | ",
                            format_time(row.time_std), " |"
                        )
                    end
                end
            end
            println(io)
        end
    end

    open(csv_path, "w") do io
        println(io, "task,dim,algorithm,mode,runs,maxevals,popsize,threads,error_mean,error_std,time_mean_s,time_std_s")
        for row in results
            println(
                io,
                row.task, ",",
                row.dim, ",",
                row.algorithm, ",",
                row.mode, ",",
                row.runs, ",",
                row.maxevals, ",",
                row.popsize, ",",
                row.threads, ",",
                format_float(row.error_mean), ",",
                format_float(row.error_std), ",",
                format_time(row.time_mean), ",",
                format_time(row.time_std)
            )
        end
    end

    println("saved to: ", md_path)
    println("saved to: ", csv_path)
end

run_benchmark()
