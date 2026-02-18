ENV["GKSwstype"] = "100"

using DifferentialEvolution
using Optim
using Random
using Plots

function sphere_vec(x)
    return x[1]^2 + x[2]^2
end

function rosenbrock_vec(x)
    return 500.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2
end

function rastrigin_vec(x)
    return 20.0 + (x[1]^2 - 10.0 * cos(2.0 * pi * x[1])) +
        (x[2]^2 - 10.0 * cos(2.0 * pi * x[2]))
end

function ackley_vec(x)
    term1 = -20.0 * exp(-0.2 * sqrt(0.5 * (x[1]^2 + x[2]^2)))
    term2 = -exp(0.5 * (cos(2.0 * pi * x[1]) + cos(2.0 * pi * x[2])))
    return term1 + term2 + 20.0 + exp(1.0)
end

function griewank_vec(x)
    sum_term = (x[1]^2 + x[2]^2) / 4000.0
    prod_term = cos(x[1]) * cos(x[2] / sqrt(2.0))
    return sum_term - prod_term + 1.0
end

function schwefel_vec(x)
    return 837.9658 - (x[1] * sin(sqrt(abs(x[1]))) + x[2] * sin(sqrt(abs(x[2]))))
end

function sphere_xy(x, y)
    return x^2 + y^2
end

function rosenbrock_xy(x, y)
    return 500.0 * (y - x^2)^2 + (1.0 - x)^2
end

function rastrigin_xy(x, y)
    return 20.0 + (x^2 - 10.0 * cos(2.0 * pi * x)) +
        (y^2 - 10.0 * cos(2.0 * pi * y))
end

function ackley_xy(x, y)
    term1 = -20.0 * exp(-0.2 * sqrt(0.5 * (x^2 + y^2)))
    term2 = -exp(0.5 * (cos(2.0 * pi * x) + cos(2.0 * pi * y)))
    return term1 + term2 + 20.0 + exp(1.0)
end

function griewank_xy(x, y)
    sum_term = (x^2 + y^2) / 4000.0
    prod_term = cos(x) * cos(y / sqrt(2.0))
    return sum_term - prod_term + 1.0
end

function schwefel_xy(x, y)
    return 837.9658 - (x * sin(sqrt(abs(x))) + y * sin(sqrt(abs(y))))
end

function trace_evolution(f, lower, upper; seed, popsize, maxiters, maxevals, algorithm, F, CR)
    rng = MersenneTwister(seed)
    dim = 2
    T = Float64
    lower_t = Vector{T}(lower)
    upper_t = Vector{T}(upper)
    pop = Matrix{T}(undef, dim, popsize)
    fitness = Vector{T}(undef, popsize)
    trial = Vector{T}(undef, dim)

    for i in 1:popsize
        for j in 1:dim
            pop[j, i] = lower_t[j] + rand(rng, T) * (upper_t[j] - lower_t[j])
        end
        fitness[i] = T(f(view(pop, :, i)))
    end

    strategy = if algorithm == :de
        DifferentialEvolution.DEStrategy(T(F), T(CR))
    elseif algorithm == :shade
        DifferentialEvolution._make_shade_state(T, popsize, popsize, T(0.2))
    elseif algorithm == :lshade
        DifferentialEvolution._make_lshade_state(T, popsize, popsize, T(0.2), maxevals)
    elseif algorithm == :jso
        DifferentialEvolution._make_jso_state(T, popsize, 5, T(0.25), maxevals, maxiters)
    else
        error("unsupported algorithm: $(algorithm)")
    end

    pop_history = Matrix{T}[]
    best_history = NTuple{2, T}[]

    best_idx = argmin(fitness)
    push!(pop_history, copy(pop))
    push!(best_history, (pop[1, best_idx], pop[2, best_idx]))

    iterations = 0
    evaluations = popsize
    stop_due_to_evals = false

    while iterations < maxiters && evaluations < maxevals
        iterations += 1
        DifferentialEvolution._start_generation!(strategy, pop, fitness, rng, iterations, maxiters, evaluations, maxevals)
        popsize = size(pop, 2)
        for i in 1:popsize
            if evaluations >= maxevals
                stop_due_to_evals = true
                break
            end
            DifferentialEvolution._generate_trial!(strategy, trial, pop, fitness, i, rng, lower_t, upper_t, evaluations)
            f_trial = T(f(trial))
            evaluations += 1
            if f_trial <= fitness[i]
                DifferentialEvolution._on_success!(strategy, i, trial, f_trial, fitness[i], pop, rng)
                fitness[i] = f_trial
                copyto!(view(pop, :, i), trial)
            end
        end

        pop, fitness = DifferentialEvolution._end_generation!(strategy, pop, fitness, rng, evaluations)
        best_idx = argmin(fitness)
        push!(pop_history, copy(pop))
        push!(best_history, (pop[1, best_idx], pop[2, best_idx]))

        if stop_due_to_evals
            break
        end
    end

    return pop_history, best_history
end

function current_point(state)
    if hasproperty(state, :x_lowest)
        return getproperty(state, :x_lowest)
    elseif hasproperty(state, :x)
        return getproperty(state, :x)
    elseif hasproperty(state, :metadata) && haskey(state.metadata, "x")
        return state.metadata["x"]
    else
        return nothing
    end
end

function optim_trace(f, x0, method, lower, upper; maxiters, maxevals)
    points = NTuple{2, Float64}[]
    simplex_history = Vector{NTuple{2, Float64}}[]
    function fcount(x)
        return f(x)
    end
    callback = function (state)
        x = current_point(state)
        if x !== nothing
            xc = clamp.(x, lower, upper)
            push!(points, (xc[1], xc[2]))
        end
        if hasproperty(state, :simplex)
            verts = NTuple{2, Float64}[]
            for v in getproperty(state, :simplex)
                vc = clamp.(v, lower, upper)
                push!(verts, (vc[1], vc[2]))
            end
            push!(simplex_history, verts)
        end
        return false
    end

    opts = Optim.Options(
        f_calls_limit=maxevals,
        iterations=maxiters,
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
    return points, simplex_history
end

function grid_values(fxy, xs, ys; zmax)
    Z = Matrix{Float64}(undef, length(ys), length(xs))
    for (j, y) in enumerate(ys)
        for (i, x) in enumerate(xs)
            Z[j, i] = fxy(x, y)
        end
    end
    Z = min.(Z, zmax)
    return log10.(Z .+ 1.0)
end

function plot_background(xs, ys, Zlog, title_str, lower, upper, legend_pos)
    plt = contourf(
        xs,
        ys,
        Zlog;
        levels=30,
        c=cgrad(:blues, rev=true),
        colorbar=false,
        xlabel="",
        ylabel="",
        title=title_str,
        aspect_ratio=:equal,
        legend=legend_pos,
        framestyle=:box,
        xaxis=false,
        yaxis=false,
        xticks=false,
        yticks=false,
        grid=false,
        xlims=(lower[1], upper[1]),
        ylims=(lower[2], upper[2]),
    )
    bx = [lower[1], upper[1], upper[1], lower[1], lower[1]]
    by = [lower[2], lower[2], upper[2], upper[2], lower[2]]
    plot!(plt, bx, by, color=:black, linewidth=1.2, label="")
    return plt
end

function animate_population(name, label, xs, ys, Zlog, pop_history, best_history, optimum, outpath, lower, upper, legend_pos)
    anim = @animate for t in 1:length(pop_history)
        plt = plot_background(xs, ys, Zlog, name * " (" * label * ")", lower, upper, legend_pos)
        pop = pop_history[t]
        scatter!(plt, pop[1, :], pop[2, :], markersize=3, color=:black, alpha=0.5, label="population")
        bx = [p[1] for p in best_history[1:t]]
        by = [p[2] for p in best_history[1:t]]
        plot!(plt, bx, by, color=:cyan, linewidth=2, label="path")
        scatter!(
            plt,
            [bx[end]],
            [by[end]],
            color=:white,
            markerstrokecolor=:cyan,
            markersize=6,
            label="",
        )
        scatter!(
            plt,
            [optimum[1]],
            [optimum[2]],
            marker=:diamond,
            color=:white,
            markerstrokecolor=:black,
            markersize=7,
            label="true",
        )
    end
    gif(anim, outpath; fps=10, loop=0)
end

function animate_path(name, xs, ys, Zlog, path, optimum, outpath, lower, upper, legend_pos)
    anim = @animate for t in 1:length(path)
        plt = plot_background(xs, ys, Zlog, name, lower, upper, legend_pos)
        px = [p[1] for p in path[1:t]]
        py = [p[2] for p in path[1:t]]
        plot!(plt, px, py, color=:cyan, linewidth=2, label="path")
        scatter!(
            plt,
            [px[end]],
            [py[end]],
            color=:white,
            markerstrokecolor=:cyan,
            markersize=6,
            label="",
        )
        scatter!(
            plt,
            [optimum[1]],
            [optimum[2]],
            marker=:diamond,
            color=:white,
            markerstrokecolor=:black,
            markersize=7,
            label="true",
        )
    end
    gif(anim, outpath; fps=10, loop=0)
end

function animate_simplex(name, xs, ys, Zlog, simplex_history, path, optimum, outpath, lower, upper, legend_pos)
    anim = @animate for t in 1:length(simplex_history)
        plt = plot_background(xs, ys, Zlog, name, lower, upper, legend_pos)
        verts = simplex_history[t]
        sx = [v[1] for v in verts]
        sy = [v[2] for v in verts]
        scatter!(plt, sx, sy, markersize=5, color=:orange, markerstrokecolor=:black, label="simplex")
        if !isempty(path)
            tpath = min(t, length(path))
            px = [p[1] for p in path[1:tpath]]
            py = [p[2] for p in path[1:tpath]]
            plot!(plt, px, py, color=:cyan, linewidth=2, label="path")
            scatter!(
                plt,
                [px[end]],
                [py[end]],
                color=:white,
                markerstrokecolor=:cyan,
                markersize=6,
                label="",
            )
        end
        scatter!(
            plt,
            [optimum[1]],
            [optimum[2]],
            marker=:diamond,
            color=:white,
            markerstrokecolor=:black,
            markersize=7,
            label="true",
        )
    end
    gif(anim, outpath; fps=10, loop=0)
end

function make_x0(lower, upper, seed)
    rng = MersenneTwister(seed)
    return [lower[1] + rand(rng) * (upper[1] - lower[1]),
            lower[2] + rand(rng) * (upper[2] - lower[2])]
end

outdir = joinpath(pwd(), "reports", "2d")
mkpath(outdir)

seed = 1
maxiters = 120
maxevals = 4000
popsize = 30

configs = [
    (
        "Sphere",
        sphere_vec,
        sphere_xy,
        [-10.0, -10.0],
        [10.0, 10.0],
        (0.0, 0.0),
        200.0,
        (F=0.5, CR=0.9),
    ),
    (
        "Rosenbrock",
        rosenbrock_vec,
        rosenbrock_xy,
        [-3.0, -3.0],
        [3.0, 3.0],
        (1.0, 1.0),
        4000.0,
        (F=0.5, CR=0.9),
    ),
    (
        "Rastrigin",
        rastrigin_vec,
        rastrigin_xy,
        [-8.0, -8.0],
        [8.0, 8.0],
        (0.0, 0.0),
        200.0,
        (F=0.5, CR=0.3),
    ),
    (
        "Ackley",
        ackley_vec,
        ackley_xy,
        [-32.768, -32.768],
        [32.768, 32.768],
        (0.0, 0.0),
        30.0,
        (F=0.5, CR=0.3),
    ),
    (
        "Griewank",
        griewank_vec,
        griewank_xy,
        [-600.0, -600.0],
        [600.0, 600.0],
        (0.0, 0.0),
        200.0,
        (F=0.5, CR=0.3),
    ),
    (
        "Schwefel",
        schwefel_vec,
        schwefel_xy,
        [-500.0, -500.0],
        [500.0, 500.0],
        (420.9687, 420.9687),
        1200.0,
        (F=0.5, CR=0.3),
    ),
]

for (name, fvec, fxy, lower, upper, optimum, zmax, params) in configs
    xs = range(lower[1], upper[1], length=200)
    ys = range(lower[2], upper[2], length=200)
    Zlog = grid_values(fxy, xs, ys; zmax=zmax)
    legend_pos = name == "Schwefel" ? :bottomleft : :topright

    for (label, alg, slug) in (("DE", :de, "de"), ("SHADE", :shade, "shade"), ("L-SHADE", :lshade, "lshade"), ("JSO", :jso, "jso"))
        pop_hist, best_hist = trace_evolution(
            fvec,
            lower,
            upper;
            seed=seed,
            popsize=popsize,
            maxiters=maxiters,
            maxevals=maxevals,
            algorithm=alg,
            F=params.F,
            CR=params.CR,
        )
        outpath = joinpath(outdir, "2d_" * lowercase(name) * "_" * slug * ".gif")
        animate_population(name, label, xs, ys, Zlog, pop_hist, best_hist, optimum, outpath, lower, upper, legend_pos)
    end

    # Nelder-Mead animation
    x0 = make_x0(lower, upper, seed)
    nm_path, nm_simplex = optim_trace(fvec, x0, Optim.NelderMead(), lower, upper; maxiters=maxiters, maxevals=maxevals)
    nm_out = joinpath(outdir, "2d_" * lowercase(name) * "_nelder_mead.gif")
    animate_simplex(name * " (Nelder-Mead)", xs, ys, Zlog, nm_simplex, nm_path, optimum, nm_out, lower, upper, legend_pos)

    # BFGS animation
    x0 = make_x0(lower, upper, seed)
    bfgs_path, _ = optim_trace(fvec, x0, Optim.BFGS(), lower, upper; maxiters=maxiters, maxevals=maxevals)
    bfgs_out = joinpath(outdir, "2d_" * lowercase(name) * "_bfgs.gif")
    animate_path(name * " (BFGS)", xs, ys, Zlog, bfgs_path, optimum, bfgs_out, lower, upper, legend_pos)
end

println("saved to: ", outdir)
