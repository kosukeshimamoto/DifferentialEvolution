using DifferentialEvolution
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

function run_de_trial(f, lower, upper, rng; algorithm, popsize, maxevals, F, CR, pmax, target)
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

function slugify(name)
    return replace(lowercase(name), r"[^a-z0-9]+" => "_")
end

function plot_benchmark_cells(csv_path, plotdir; algorithms)
    rscript = Sys.which("Rscript")
    if rscript === nothing
        error("Rscript not found; install R to render benchmark plots")
    end
    script_path = joinpath(pwd(), "scripts", "plot_benchmark_sweep_cells.R")
    alg_arg = join(algorithms, ",")
    cmd = `$rscript $script_path --csv $csv_path --outdir $plotdir --algorithms $alg_arg`
    return run(cmd)
end

function run_sweep()
    maxevals_list = parse_int_list("DE_SWEEP_MAXEVALS", [200000, 500000])
    popsizes = parse_int_list("DE_SWEEP_POPSIZES", [20, 50])
    pmax_list = parse_float_list("DE_SWEEP_PMAX", [0.25])
    dims = parse_int_list("DE_SWEEP_DIMS", [10, 50, 100])
    runs = parse(Int, get(ENV, "DE_SWEEP_RUNS", "5"))
    base_seed = parse(Int, get(ENV, "DE_SWEEP_SEED", "2025"))
    target_tol = parse(Float64, get(ENV, "DE_SWEEP_TARGET_TOL", "1e-2"))
    de_F = parse(Float64, get(ENV, "DE_SWEEP_F", "0.5"))
    de_CR = parse(Float64, get(ENV, "DE_SWEEP_CR", "0.9"))

    outdir = joinpath(pwd(), "reports", "benchmarks", "de_param_sweep")
    plotdir = joinpath(outdir, "plots")
    mkpath(outdir)
    mkpath(plotdir)
    md_path = joinpath(outdir, "benchmark_summary.md")
    csv_path = joinpath(outdir, "benchmark_summary.csv")
    html_path = joinpath(outdir, "benchmark_overview.html")

    plot_only = get(ENV, "DE_SWEEP_PLOT_ONLY", "") != "" && isfile(csv_path)

    tasks = [
        (name="Sphere", f=sphere, lower=-5.0, upper=5.0, optimum=0.0),
        (name="Rosenbrock", f=rosenbrock, lower=-2.0, upper=2.0, optimum=0.0),
        (name="Rastrigin", f=rastrigin, lower=-5.12, upper=5.12, optimum=0.0),
        (name="Ackley", f=ackley, lower=-32.768, upper=32.768, optimum=0.0),
        (name="Griewank", f=griewank, lower=-600.0, upper=600.0, optimum=0.0),
        (name="Schwefel", f=schwefel, lower=-500.0, upper=500.0, optimum=0.0),
    ]

    algorithms = [
        ("DE", :de),
        ("SHADE", :shade),
        ("L-SHADE", :lshade),
        ("JSO", :jso),
    ]

    results = NamedTuple[]

    if plot_only
        rows = DelimitedFiles.readdlm(csv_path, ',', String; skipstart=1)
        for row in eachrow(rows)
            push!(results, (
                task=row[1],
                dim=parse(Int, row[2]),
                algorithm=row[3],
                runs=parse(Int, row[4]),
                maxevals=parse(Int, row[5]),
                popsize=parse(Int, row[6]),
                pmax=parse(Float64, row[7]),
                error_mean=row[8] == "NA" ? NaN : parse(Float64, row[8]),
                error_std=row[9] == "NA" ? NaN : parse(Float64, row[9]),
                time_mean=row[10] == "NA" ? NaN : parse(Float64, row[10]),
                time_std=row[11] == "NA" ? NaN : parse(Float64, row[11]),
            ))
        end

        if !isempty(results)
            dims = sort(unique([r.dim for r in results]))
            maxevals_list = sort(unique([r.maxevals for r in results]))
            popsizes = sort(unique([r.popsize for r in results]))
            pmax_list = sort(unique([r.pmax for r in results]))
            runs = results[1].runs
        end
    else
        config_id = 0
        for maxevals in maxevals_list
            for popsize in popsizes
                for pmax in pmax_list
                    config_id += 1
                    popsize = max(popsize, 4)
                    for task in tasks
                        for dim in dims
                            lower = fill(task.lower, dim)
                            upper = fill(task.upper, dim)
                            target = task.optimum + target_tol

                            for (alg_index, (alg_label, alg_id)) in enumerate(algorithms)
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
                                    runs=runs,
                                    maxevals=maxevals,
                                    popsize=popsize,
                                    pmax=pmax,
                                    error_mean=err_stats.mean,
                                    error_std=err_stats.std,
                                    time_mean=time_stats.mean,
                                    time_std=time_stats.std,
                                ))
                            end
                        end
                    end
                end
            end
        end
    end

    timestamp = Dates.format(Dates.now(), dateformat"yyyy-mm-dd HH:MM:SS")

    open(md_path, "w") do io
        println(io, "# Benchmark sweep summary")
        println(io)
        println(io, "- generated: ", timestamp)
        println(io, "- runs: ", runs)
        println(io, "- dims: ", join(dims, ", "))
        println(io, "- maxevals list: ", join(maxevals_list, ", "))
        println(io, "- popsize list: ", join(popsizes, ", "))
        println(io, "- pmax list: ", join(pmax_list, ", "))
        println(io, "- algorithms: ", join(first.(algorithms), ", "))
        println(io, "- target_tol: ", target_tol)
        println(io, "- DE params: F=", de_F, ", CR=", de_CR, ", pmax variable")
        println(io, "- error metric: mean(|f - f*|) / dim")
        println(io, "- plots: per-dimension x algorithm grid; each cell compares maxevals vs popsize (PNG), facets by pmax")
        println(io)

        for pmax in pmax_list
            for maxevals in maxevals_list
                for popsize in popsizes
                    println(io, "## pmax=", format_float(pmax), ", maxevals=", maxevals, ", popsize=", popsize)
                    println(io)
                    for task in tasks
                        println(io, "### ", task.name)
                        println(io)
                        println(io, "| Dim | Algorithm | Runs | Maxevals | Popsize | Pmax | Error mean / dim | Error std / dim | Time mean (s) | Time std (s) |")
                        println(io, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
                        for dim in dims
                            for (alg_label, _) in algorithms
                                row = only(filter(r ->
                                    r.task == task.name &&
                                    r.dim == dim &&
                                    r.algorithm == alg_label &&
                                    r.maxevals == maxevals &&
                                    r.popsize == popsize &&
                                    r.pmax == pmax,
                                    results
                                ))
                                println(
                                    io,
                                    dim, " | ",
                                    alg_label, " | ",
                                    row.runs, " | ",
                                    row.maxevals, " | ",
                                    row.popsize, " | ",
                                    format_float(row.pmax), " | ",
                                    format_float(row.error_mean), " | ",
                                    format_float(row.error_std), " | ",
                                    format_time(row.time_mean), " | ",
                                    format_time(row.time_std), " |"
                                )
                            end
                        end
                        println(io)
                    end
                end
            end
        end
    end

    open(csv_path, "w") do io
        println(io, "task,dim,algorithm,runs,maxevals,popsize,pmax,error_mean,error_std,time_mean_s,time_std_s")
        for row in results
            println(
                io,
                row.task, ",",
                row.dim, ",",
                row.algorithm, ",",
                row.runs, ",",
                row.maxevals, ",",
                row.popsize, ",",
                format_float(row.pmax), ",",
                format_float(row.error_mean), ",",
                format_float(row.error_std), ",",
                format_time(row.time_mean), ",",
                format_time(row.time_std)
            )
        end
    end

    plot_benchmark_cells(csv_path, plotdir; algorithms=first.(algorithms))

    open(html_path, "w") do io
        println(io, "<!doctype html>")
        println(io, "<html lang=\"en\">")
        println(io, "<head>")
        println(io, "  <meta charset=\"utf-8\">")
        println(io, "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
        println(io, "  <title>DE Sweep Benchmark Overview</title>")
        println(io, "  <style>")
        println(io, "    body { font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; margin: 24px; color: #222; }")
        println(io, "    h1 { margin: 0 0 16px; }")
        println(io, "    h2 { margin: 28px 0 8px; }")
        println(io, "    h3 { margin: 18px 0 6px; }")
        println(io, "    .panel { margin: 8px 0 18px; }")
        println(io, "    .panel h4 { margin: 0 0 6px; font-size: 14px; color: #444; }")
        println(io, "    .note { font-size: 12px; color: #555; margin: 4px 0 12px; }")
        println(io, "    table.grid { border-collapse: collapse; margin: 8px 0 20px; }")
        println(io, "    table.grid th, table.grid td { border: 1px solid #ddd; padding: 6px; text-align: center; vertical-align: top; }")
        println(io, "    table.grid th.row { text-align: left; background: #f7f7f7; }")
        println(io, "    img.cell { width: 230px; height: auto; border: 1px solid #ddd; }")
        println(io, "  </style>")
        println(io, "</head>")
        println(io, "<body>")
        println(io, "  <h1>DE Sweep Benchmark Overview</h1>")
        for task in tasks
            task_slug = slugify(task.name)
            println(io, "  <h2>", task.name, "</h2>")
            println(io, "  <div class=\"note\">Bars compare maxevals (x) and popsize (color) for each dimension/algorithm. Rows inside each plot correspond to pmax values.</div>")

            for metric in (("Mean error / dim", "error"), ("Mean time", "time"))
                println(io, "  <div class=\"panel\">")
                println(io, "    <h3>", metric[1], "</h3>")
                println(io, "    <table class=\"grid\">")
                println(io, "      <thead>")
                println(io, "        <tr>")
                println(io, "          <th></th>")
                for (alg_label, _) in algorithms
                    println(io, "          <th>", alg_label, "</th>")
                end
                println(io, "        </tr>")
                println(io, "      </thead>")
                println(io, "      <tbody>")
                for dim in dims
                    println(io, "        <tr>")
                    println(io, "          <th class=\"row\">", dim, "D</th>")
                    for (alg_label, _) in algorithms
                        alg_slug = slugify(alg_label)
                        img_path = "plots/" * task_slug * "/" * metric[2] * "/dim_" * string(dim) * "_" * alg_slug * ".png"
                        println(io, "          <td><img class=\"cell\" src=\"", img_path, "\" alt=\"", task.name, " ", dim, "D ", alg_label, " ", metric[1], "\"></td>")
                    end
                    println(io, "        </tr>")
                end
                println(io, "      </tbody>")
                println(io, "    </table>")
                println(io, "  </div>")
            end
        end
        println(io, "</body>")
        println(io, "</html>")
    end

    println("saved to: ", md_path)
    println("saved to: ", csv_path)
    println("saved to: ", html_path)
    println("plots saved to: ", plotdir)
end

run_sweep()
