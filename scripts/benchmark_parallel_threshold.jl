using DifferentialEvolution
using Random
using Statistics
using Printf
using Dates
using DelimitedFiles
using Base.Threads

function rastrigin(x)
    n = length(x)
    s = 10.0 * n
    @inbounds for i in 1:n
        s += x[i]^2 - 10.0 * cos(2.0 * pi * x[i])
    end
    return s
end

const _HEAVY_SINK = Threads.Atomic{Float64}(0.0)

@noinline function burn_for(x, delay_s)
    if delay_s <= 0
        return nothing
    end
    t0 = time_ns()
    limit = Int64(round(delay_s * 1e9))
    s = 0.0
    n = length(x)
    @inbounds while (time_ns() - t0) < limit
        s += sin(x[1]) * cos(x[n])
        if s > 1e6
            s *= 0.5
        end
    end
    Threads.atomic_xchg!(_HEAVY_SINK, s)
    return nothing
end

function make_delayed(f, delay_s)
    if delay_s <= 0
        return f
    end
    return x -> (burn_for(x, delay_s); f(x))
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
    return @sprintf("%.6f", x)
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

function parse_float_list(env_name, default_values)
    raw = get(ENV, env_name, "")
    if isempty(raw)
        return default_values
    end
    values = Float64[]
    for part in split(raw, ",")
        part = strip(part)
        if !isempty(part)
            push!(values, parse(Float64, part))
        end
    end
    return isempty(values) ? default_values : values
end

function run_benchmark()
    dim = parse(Int, get(ENV, "DE_PAR_THRESH_DIM", "50"))
    runs = parse(Int, get(ENV, "DE_PAR_THRESH_RUNS", "1"))
    base_seed = parse(Int, get(ENV, "DE_PAR_THRESH_SEED", "2025"))
    target_tol = parse(Float64, get(ENV, "DE_PAR_THRESH_TARGET_TOL", "1e-2"))
    de_F = parse(Float64, get(ENV, "DE_PAR_THRESH_F", "0.5"))
    de_CR = parse(Float64, get(ENV, "DE_PAR_THRESH_CR", "0.9"))
    pmax = parse(Float64, get(ENV, "DE_PAR_THRESH_PMAX", "0.25"))
    popsize = parse(Int, get(ENV, "DE_PAR_THRESH_POPSZ", "10"))
    maxevals = parse(Int, get(ENV, "DE_PAR_THRESH_MAXEVALS", "20"))

    delays = parse_float_list("DE_PAR_THRESH_DELAYS", [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    outdir = joinpath(pwd(), "reports", "benchmarks", "parallel_threshold")
    mkpath(outdir)
    md_path = joinpath(outdir, "benchmark_summary.md")
    csv_path = joinpath(outdir, "benchmark_summary.csv")

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
    for delay in delays
        f = make_delayed(rastrigin, delay)
        for (mode_label, mode_flag) in modes
            for (alg_index, (alg_label, alg_id)) in enumerate(algorithms)
                config_id += 1
                lower = fill(-5.12, dim)
                upper = fill(5.12, dim)
                target = target_tol

                final_errors = Float64[]
                times_list = Float64[]

                for run in 1:runs
                    seed_alg = base_seed + run + 1000 * alg_index + 100000 * config_id
                    rng = MersenneTwister(seed_alg)
                    best_f, elapsed = run_de_trial(
                        f,
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

                    push!(final_errors, abs(best_f) / dim)
                    push!(times_list, elapsed)
                end

                err_stats = mean_std(final_errors)
                time_stats = mean_std(times_list)

                push!(results, (
                    delay_s=delay,
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

    timestamp = Dates.format(Dates.now(), dateformat"yyyy-mm-dd HH:MM:SS")

    open(md_path, "w") do io
        println(io, "# Parallel threshold benchmark summary")
        println(io)
        println(io, "- generated: ", timestamp)
        println(io, "- dim: ", dim)
        println(io, "- runs: ", runs)
        println(io, "- maxevals: ", maxevals)
        println(io, "- popsize: ", popsize)
        println(io, "- algorithms: ", join(first.(algorithms), ", "))
        println(io, "- modes: serial, parallel")
        println(io, "- threads: ", nthreads())
        println(io, "- delay targets (s): ", join(delays, ", "))
        println(io, "- base function: Rastrigin")
        println(io, "- error metric: mean(|f - f*|) / dim (f* = 0)")
        println(io)

        println(io, "| Delay (s) | Algorithm | Mode | Runs | Maxevals | Popsize | Threads | Error mean / dim | Error std / dim | Time mean (s) | Time std (s) |")
        println(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for delay in delays
            for (alg_label, _) in algorithms
                for (mode_label, _) in modes
                    row = only(filter(r ->
                        r.delay_s == delay &&
                        r.algorithm == alg_label &&
                        r.mode == mode_label,
                        results
                    ))
                    println(
                        io,
                        format_float(row.delay_s), " | ",
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
    end

    open(csv_path, "w") do io
        println(io, "delay_s,dim,algorithm,mode,runs,maxevals,popsize,threads,error_mean,error_std,time_mean_s,time_std_s")
        for row in results
            println(
                io,
                format_float(row.delay_s), ",",
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
