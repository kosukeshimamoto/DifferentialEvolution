using DifferentialEvolution
using Optim
using Random
using Statistics
using Printf
using Dates
using DelimitedFiles

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

function make_x0(lower, upper, rng)
    return [lower[i] + rand(rng) * (upper[i] - lower[i]) for i in eachindex(lower)]
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

function run_de_trial(f, lower, upper, rng; algorithm, popsize, maxevals, F, CR, target, pmax)
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
    )
    elapsed = (time_ns() - t0) / 1e9
    return res.best_f, elapsed
end

function run_optim_trial(f, lower, upper, x0, method; maxevals)
    best = Inf

    function f_box(x)
        return f(clamp.(x, lower, upper))
    end

    function fcount(x)
        v = f_box(x)
        if v < best
            best = v
        end
        return v
    end

    opts = Optim.Options(
        f_calls_limit=maxevals,
        iterations=maxevals,
        x_abstol=0.0,
        x_reltol=0.0,
        f_abstol=0.0,
        f_reltol=0.0,
        g_abstol=0.0,
        successive_f_tol=0,
        store_trace=false,
        show_trace=false,
        show_warnings=false,
    )

    t0 = time_ns()
    Optim.optimize(fcount, x0, method, opts)
    elapsed = (time_ns() - t0) / 1e9

    return best, elapsed
end

function slugify(name)
    return replace(lowercase(name), r"[^a-z0-9]+" => "_")
end

function plot_benchmark_bars(csv_path, plotdir)
    rscript = Sys.which("Rscript")
    if rscript === nothing
        error("Rscript not found; install R to render benchmark plots")
    end
    script_path = joinpath(pwd(), "scripts", "plot_benchmark_bars.R")
    cmd = `$rscript $script_path --csv $csv_path --outdir $plotdir`
    return run(cmd)
end

maxevals = parse(Int, get(ENV, "DE_BENCH_MAXEVALS", "200000"))
runs = parse(Int, get(ENV, "DE_BENCH_RUNS", "5"))
base_seed = parse(Int, get(ENV, "DE_BENCH_SEED", "2025"))
target_tol = parse(Float64, get(ENV, "DE_BENCH_TARGET_TOL", "1e-2"))
de_F = parse(Float64, get(ENV, "DE_BENCH_F", "0.5"))
de_CR = parse(Float64, get(ENV, "DE_BENCH_CR", "0.9"))
pmax = parse(Float64, get(ENV, "DE_BENCH_PMAX", "0.25"))

outdir = joinpath(pwd(), "reports", "benchmarks", "algorithm_compare")
plotdir = joinpath(outdir, "plots")
mkpath(outdir)
mkpath(plotdir)
md_path = joinpath(outdir, "benchmark_summary.md")
csv_path = joinpath(outdir, "benchmark_summary.csv")

plot_only = get(ENV, "DE_BENCH_PLOT_ONLY", "") != "" && isfile(csv_path)

dims = [1, 5, 10, 50, 100]

tasks = [
    (name="Sphere", f=sphere, lower=-5.0, upper=5.0, optimum=0.0),
    (name="Rosenbrock", f=rosenbrock, lower=-2.0, upper=2.0, optimum=0.0),
    (name="Rastrigin", f=rastrigin, lower=-5.12, upper=5.12, optimum=0.0),
    (name="Ackley", f=ackley, lower=-32.768, upper=32.768, optimum=0.0),
    (name="Griewank", f=griewank, lower=-600.0, upper=600.0, optimum=0.0),
    (name="Schwefel", f=schwefel, lower=-500.0, upper=500.0, optimum=0.0),
]

algorithms = [
    ("BFGS", :bfgs),
    ("Nelder-Mead", :nelder_mead),
    ("DE", :de),
    ("SHADE", :shade),
    ("L-SHADE", :lshade),
    ("JSO", :jso),
]

results = NamedTuple[]

if plot_only
    data = readdlm(csv_path, ',', String)
    for row in eachrow(data[2:end, :])
        push!(results, (
            task=row[1],
            dim=parse(Int, row[2]),
            algorithm=row[3],
            runs=parse(Int, row[4]),
            maxevals=parse(Int, row[5]),
            popsize=parse(Int, row[6]),
            error_mean=row[7] == "NA" ? NaN : parse(Float64, row[7]),
            error_std=row[8] == "NA" ? NaN : parse(Float64, row[8]),
            time_mean=row[9] == "NA" ? NaN : parse(Float64, row[9]),
            time_std=row[10] == "NA" ? NaN : parse(Float64, row[10]),
        ))
    end
    if !isempty(results)
        runs = results[1].runs
        maxevals = results[1].maxevals
    end
else
    for task in tasks
        for dim in dims
            lower = fill(task.lower, dim)
            upper = fill(task.upper, dim)
            popsize = max(5 * dim, 4)
            target = task.optimum + target_tol

            for (alg_label, alg_id) in algorithms
                final_errors = Float64[]
                times_list = Float64[]

                for run in 1:runs
                    seed_x0 = base_seed + run + 10 * dim
                    rng_x0 = MersenneTwister(seed_x0)
                    x0 = make_x0(lower, upper, rng_x0)

                    if alg_id in (:de, :shade, :lshade, :jso)
                        seed_alg = base_seed + run + 1000 * findfirst(x -> x[2] == alg_id, algorithms) + 10 * dim
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
                        )
                    elseif alg_id == :nelder_mead
                        best_f, elapsed = run_optim_trial(
                            task.f,
                            lower,
                            upper,
                            x0,
                            Optim.NelderMead();
                            maxevals=maxevals,
                        )
                    else
                        best_f, elapsed = run_optim_trial(
                            task.f,
                            lower,
                            upper,
                            x0,
                            Optim.BFGS();
                            maxevals=maxevals,
                        )
                    end

                push!(final_errors, abs(best_f - task.optimum) / dim)
                    push!(times_list, elapsed)
                end

                err_stats = mean_std(final_errors)
                time_stats = mean_std(times_list)

                push!(results, (
                    task=task.name,
                    dim=dim,
                    algorithm=alg_label,
                    runs=runs,
                    maxevals=maxevals,
                    popsize=popsize,
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
    println(io, "# Benchmark summary")
    println(io)
    println(io, "- generated: ", timestamp)
    println(io, "- runs: ", runs)
    println(io, "- maxevals: ", maxevals)
    println(io, "- dims: ", join(dims, ", "))
    println(io, "- target_tol: ", target_tol)
    println(io, "- popsize: 5 * D for DE variants (DE/SHADE/L-SHADE/JSO)")
    println(io, "- DE params: F=", de_F, ", CR=", de_CR, ", pmax=", pmax)
    println(io, "- Optim methods: Nelder-Mead, BFGS (objective clamped to bounds)")
    println(io, "- error metric: mean(|f - f*|) / dim")
    println(io, "- plots: grouped bars by dimension (color=algorithm), linear scale, no error bars (PNG)")
    println(io)

    # Plots are saved per function; a combined HTML overview is emitted below.

    for task in tasks
        println(io, "## ", task.name)
        println(io)
        println(io, "| Dim | Algorithm | Runs | Maxevals | Popsize | Error mean / dim | Error std / dim | Time mean (s) | Time std (s) |")
        println(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for dim in dims
            for (alg_label, _) in algorithms
                row = only(filter(r -> r.task == task.name && r.dim == dim && r.algorithm == alg_label, results))
                println(io,
                    "| ", row.dim,
                    " | ", row.algorithm,
                    " | ", row.runs,
                    " | ", row.maxevals,
                    " | ", row.popsize,
                    " | ", format_float(row.error_mean),
                    " | ", format_float(row.error_std),
                    " | ", format_time(row.time_mean),
                    " | ", format_time(row.time_std),
                    " |"
                )
            end
        end
        println(io)

        error_plot = joinpath(plotdir, slugify(task.name) * "_error.png")
        time_plot = joinpath(plotdir, slugify(task.name) * "_time.png")

        println(io, "![", task.name, " error](plots/", basename(error_plot), ")")
        println(io)
        println(io, "![", task.name, " time](plots/", basename(time_plot), ")")
        println(io)
    end
    println(io, "- overview_html: benchmark_overview.html")
    println(io)
end

open(csv_path, "w") do io
    println(io, "task,dim,algorithm,runs,maxevals,popsize,error_mean,error_std,time_mean_s,time_std_s")
    for row in results
        println(io,
            row.task, ",",
            row.dim, ",",
            row.algorithm, ",",
            row.runs, ",",
            row.maxevals, ",",
            row.popsize, ",",
            format_float(row.error_mean), ",",
            format_float(row.error_std), ",",
            format_time(row.time_mean), ",",
            format_time(row.time_std)
        )
    end
end

plot_benchmark_bars(csv_path, plotdir)

html_path = joinpath(outdir, "benchmark_overview.html")
open(html_path, "w") do io
    println(io, "<!doctype html>")
    println(io, "<html lang=\"en\">")
    println(io, "<head>")
    println(io, "  <meta charset=\"utf-8\">")
    println(io, "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
    println(io, "  <title>Benchmark Overview</title>")
    println(io, "  <style>")
    println(io, "    body { font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; margin: 20px; color: #222; }")
    println(io, "    h1 { margin: 0 0 12px; }")
    println(io, "    h2 { margin: 18px 0 8px; }")
    println(io, "    .task { display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 16px; align-items: start; }")
    println(io, "    .panel { margin: 0; }")
    println(io, "    .panel h3 { margin: 0 0 6px; font-size: 13px; color: #444; }")
    println(io, "    img { width: 100%; max-width: 560px; height: auto; border: 1px solid #ddd; }")
    println(io, "    @media (max-width: 1100px) { .task { grid-template-columns: 1fr; } }")
    println(io, "  </style>")
    println(io, "</head>")
    println(io, "<body>")
    println(io, "  <h1>Benchmark Overview</h1>")
    for task in tasks
        err_img = "plots/" * slugify(task.name) * "_error.png"
        time_img = "plots/" * slugify(task.name) * "_time.png"
        println(io, "  <h2>", task.name, "</h2>")
        println(io, "  <div class=\"task\">")
        println(io, "    <div class=\"panel\">")
        println(io, "      <h3>Mean error / dim</h3>")
        println(io, "      <img src=\"", err_img, "\" alt=\"", task.name, " mean error per dim\">")
        println(io, "    </div>")
        println(io, "    <div class=\"panel\">")
        println(io, "      <h3>Mean time</h3>")
        println(io, "      <img src=\"", time_img, "\" alt=\"", task.name, " mean time\">")
        println(io, "    </div>")
        println(io, "  </div>")
    end
    println(io, "</body>")
    println(io, "</html>")
end

println("saved to: ", md_path)
println("saved to: ", csv_path)
println("saved to: ", html_path)
println("plots saved to: ", plotdir)
