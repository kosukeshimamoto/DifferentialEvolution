ENV["GKSwstype"] = "100"

using DifferentialEvolution
using Optim
using Random
using Plots

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

function cummin(values)
    out = similar(values)
    best = values[1]
    out[1] = best
    for i in 2:length(values)
        v = values[i]
        if v < best
            best = v
        end
        out[i] = best
    end
    return out
end

function state_value(state)
    if hasproperty(state, :value)
        return getproperty(state, :value)
    elseif hasproperty(state, :f_lowest)
        return getproperty(state, :f_lowest)
    elseif hasproperty(state, :f_x)
        return getproperty(state, :f_x)
    else
        return NaN
    end
end

function run_de_curve(f, lower, upper; seed, maxevals, popsize, F, CR, algorithm=:de)
    rng = MersenneTwister(seed)
    maxiters = max(1, Int(floor(maxevals / popsize)) - 1)
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
        history=true,
    )
    if isempty(res.history)
        return Int[], Float64[]
    end
    evals = popsize .* (2:(res.iterations + 1))
    best_vals = cummin(res.history)
    best_vals = max.(best_vals, eps(Float64))
    return evals, best_vals
end

function run_optim_curve(f, x0, method; maxevals)
    count = Ref(0)
    function fcount(x)
        count[] += 1
        return f(x)
    end
    evals = Int[]
    values = Float64[]
    callback = function (state)
        v = state_value(state)
        if isfinite(v)
            push!(evals, count[])
            push!(values, v)
        end
        return false
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
        callback=callback,
    )

    Optim.optimize(fcount, x0, method, opts)

    if isempty(values)
        return Int[], Float64[]
    end
    best_vals = cummin(values)
    best_vals = max.(best_vals, eps(Float64))
    return evals, best_vals
end

function plot_convergence(title_str, series, outpath)
    plt = plot(
        xlabel="function evaluations",
        ylabel="best f(x)",
        yscale=:log10,
        title=title_str,
        legend=:topright,
    )
    for (label, x, y) in series
        if !isempty(x)
            plot!(plt, x, y, label=label, linewidth=2)
        end
    end
    savefig(plt, outpath)
end

function make_x0(lower, upper, seed)
    rng = MersenneTwister(seed)
    return [lower[i] + rand(rng) * (upper[i] - lower[i]) for i in eachindex(lower)]
end

function slugify(name)
    return replace(lowercase(name), r"[^a-z0-9]+" => "_")
end

outdir = joinpath(pwd(), "reports", "benchmarks", "convergence")
mkpath(outdir)

seed = 1
maxevals = 10_000_000
n = 30

functions = [
    ("Sphere 30D", sphere, fill(-5.0, n), fill(5.0, n)),
    ("Rosenbrock 30D", rosenbrock, fill(-2.0, n), fill(2.0, n)),
    ("Rastrigin 30D", rastrigin, fill(-5.12, n), fill(5.12, n)),
]

algorithms = [
    ("DE", :de),
    ("SHADE", :shade),
    ("L-SHADE", :lshade),
]

for (name, f, lower, upper) in functions
    series = Tuple{String, Vector{Int}, Vector{Float64}}[]

    for (label, alg) in algorithms
        x, y = run_de_curve(
            f,
            lower,
            upper;
            seed=seed,
            maxevals=maxevals,
            popsize=5 * n,
            F=0.5,
            CR=0.9,
            algorithm=alg,
        )
        push!(series, (label, x, y))
    end

    x0 = make_x0(lower, upper, seed)
    x, y = run_optim_curve(f, x0, Optim.NelderMead(); maxevals=maxevals)
    push!(series, ("Nelder-Mead", x, y))

    x, y = run_optim_curve(f, x0, Optim.BFGS(); maxevals=maxevals)
    push!(series, ("BFGS", x, y))

    outpath = joinpath(outdir, "convergence_compare_" * slugify(name) * ".png")
    plot_convergence("Comparison: " * name, series, outpath)
end

println("saved to: ", outdir)
