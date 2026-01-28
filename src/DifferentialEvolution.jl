module DifferentialEvolution

export Result, optimize

using Random

include("internal.jl")

"""
    Result{T}

Optimization result returned by `optimize`.

Fields:
- `best_x::Vector{T}`: best solution vector
- `best_f::T`: objective value at `best_x`
- `status::Symbol`: `:target_reached`, `:maxiters`, `:maxevals`, or `:stopped`
- `evaluations::Int`: number of objective evaluations
- `iterations::Int`: number of iterations executed
- `history::Vector{T}`: best objective value after each iteration (empty if disabled)
"""
struct Result{T}
    best_x::Vector{T}
    best_f::T
    status::Symbol
    evaluations::Int
    iterations::Int
    history::Vector{T}
end

"""
    optimize(f, lower, upper; rng, algorithm, popsize, maxiters, maxevals, F, CR, memory_size, pmax, target, history, parallel)

Minimize `f` over a box defined by `lower` and `upper`.
`rng` is required and controls all randomness.

If `parallel=true`, trials are generated and evaluated in parallel across threads
using a generation-synchronous update. Results can differ from the default
asynchronous update, but remain reproducible for deterministic `f`.
"""
function optimize(
    f,
    lower::AbstractVector,
    upper::AbstractVector;
    rng::AbstractRNG,
    algorithm::Symbol = :de,
    popsize::Union{Int, Nothing} = nothing,
    maxiters::Int = 1000,
    maxevals::Union{Int, Nothing} = nothing,
    F::Real = 0.8,
    CR::Real = 0.9,
    memory_size::Union{Int, Nothing} = nothing,
    pmax::Real = 0.2,
    target::Real = -Inf,
    history::Bool = true,
    parallel::Bool = false,
)
    dim = length(lower)
    if isnothing(popsize)
        if algorithm == :jso
            popsize = max(4, round(Int, 25 * log(dim) * sqrt(dim)))
        else
            popsize = max(10 * dim, 4)
        end
    end
    maxevals = isnothing(maxevals) ? popsize * (maxiters + 1) : maxevals

    _validate_inputs(lower, upper, popsize, maxiters, maxevals, F, CR, algorithm, memory_size, pmax)

    T = promote_type(Float64, eltype(lower), eltype(upper), typeof(F), typeof(CR), typeof(target))
    lower_t = Vector{T}(lower)
    upper_t = Vector{T}(upper)
    F_t = T(F)
    CR_t = T(CR)
    pmax_t = T(pmax)
    target_t = T(target)
    if isnothing(memory_size)
        memory_size = algorithm == :jso ? 5 : popsize
    end

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
        DEStrategy(F_t, CR_t)
    elseif algorithm == :shade
        _make_shade_state(T, popsize, memory_size, pmax_t)
    elseif algorithm == :lshade
        _make_lshade_state(T, popsize, memory_size, pmax_t, maxevals)
    else
        _make_jso_state(T, popsize, memory_size, pmax_t, maxevals, maxiters)
    end

    best_x, best_f, evaluations, iterations, history_best = _run_evolution!(
        f,
        pop,
        fitness,
        trial,
        lower_t,
        upper_t,
        rng,
        strategy,
        maxiters,
        maxevals,
        target_t,
        history,
        parallel,
    )

    status = if best_f <= target_t
        :target_reached
    elseif iterations >= maxiters
        :maxiters
    elseif evaluations >= maxevals
        :maxevals
    else
        :stopped
    end

    return Result(best_x, best_f, status, evaluations, iterations, history_best)
end

end # module
