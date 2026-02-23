module DifferentialEvolution

export Result, RunSettings, TraceRecord, SublevelSearchResult, optimize, find_sublevel_point, write_trace_csv

using Optim
using Random

include("internal.jl")

"""
    TraceRecord{T}

Single trace row for one optimization run.

Fields:
- `generation::Int`: DE generation index (1-based). Local-phase rows use the next index.
- `job_id::Int`: user-provided job identifier
- `phase::Symbol`: `:de_generation`, `:local_start`, or `:local_end`
- `best_x::Vector{T}`: best parameter vector at this row
- `best_f::T`: objective value at `best_x`
- `evaluations::Int`: cumulative objective evaluations at this row
"""
struct TraceRecord{T}
    generation::Int
    job_id::Int
    phase::Symbol
    best_x::Vector{T}
    best_f::T
    evaluations::Int
end

"""
    RunSettings{T}

Resolved settings used in one optimization run.
"""
struct RunSettings{T}
    algorithm::Symbol
    popsize::Int
    maxiters::Int
    maxevals::Int
    F::T
    CR::T
    memory_size::Int
    pmax::T
    target::T
    history::Bool
    parallel::Bool
    local_refine::Bool
    local_method::Symbol
    local_maxiters::Int
    local_tol::T
    trace_history::Bool
    job_id::Int
    message::Bool
    message_every::Int
    message_mode::Symbol
end

"""
    Result{T}

Optimization result returned by `optimize`.

Fields:
- `best_x::Vector{T}`: best solution vector
- `best_f::T`: objective value at `best_x`
- `status::Symbol`: final outcome, `:target_reached` or `:not_reached`
- `de_status::Symbol`: DE-phase stop reason, `:target_reached`, `:maxiters`, `:maxevals`, or `:stopped`
- `evaluations::Int`: total number of objective evaluations (DE + local if enabled)
- `iterations::Int`: number of iterations executed
- `history::Vector{T}`: best objective value after each iteration (empty if disabled)
- `de_best_x::Vector{T}` / `de_best_f::T`: best solution and objective from DE phase
- `local_best_x::Vector{T}` / `local_best_f::T`: best solution and objective from local phase
- `local_status::Symbol`: `:disabled`, `:success`, `:failed`, or `:stopped`
- `de_evaluations::Int`: number of DE objective evaluations
- `local_evaluations::Int`: number of local-phase objective evaluations
- `total_evaluations::Int`: `de_evaluations + local_evaluations`
- `elapsed_de_sec::Float64`: elapsed DE time in seconds
- `elapsed_local_sec::Float64`: elapsed local-phase time in seconds
- `elapsed_total_sec::Float64`: total elapsed time in seconds
- `settings::RunSettings{T}`: resolved run settings
- `trace::Vector{TraceRecord{T}}`: per-generation and local-phase trace rows
"""
struct Result{T}
    best_x::Vector{T}
    best_f::T
    status::Symbol
    de_status::Symbol
    evaluations::Int
    iterations::Int
    history::Vector{T}
    de_best_x::Vector{T}
    de_best_f::T
    local_best_x::Vector{T}
    local_best_f::T
    local_status::Symbol
    de_evaluations::Int
    local_evaluations::Int
    total_evaluations::Int
    elapsed_de_sec::Float64
    elapsed_local_sec::Float64
    elapsed_total_sec::Float64
    settings::RunSettings{T}
    trace::Vector{TraceRecord{T}}
end

"""
    SublevelSearchResult{T}

Result returned by `find_sublevel_point`.

Fields:
- `overlap::Bool`: `true` if a point with `f(x) <= c + c_tol` was found
- `best_x::Vector{T}`: best solution found during the run
- `best_f::T`: objective value at `best_x`
- `evaluations::Int`: number of objective evaluations used by DE phase
- `stop_reason::Symbol`: `:overlap_found`, `:maxevals`, `:maxiters`, or `:stopped`
"""
struct SublevelSearchResult{T}
    overlap::Bool
    best_x::Vector{T}
    best_f::T
    evaluations::Int
    stop_reason::Symbol
end

function Result(best_x::Vector{T}, best_f::T, de_status::Symbol, evaluations::Int, iterations::Int, history::Vector{T}) where {T}
    status = de_status == :target_reached ? :target_reached : :not_reached
    default_settings = RunSettings(
        :de,
        0,
        iterations,
        evaluations,
        zero(T),
        zero(T),
        0,
        zero(T),
        zero(T),
        !isempty(history),
        false,
        false,
        :nelder_mead,
        0,
        zero(T),
        false,
        0,
        false,
        1,
        :compact,
    )
    return Result(
        best_x,
        best_f,
        status,
        de_status,
        evaluations,
        iterations,
        history,
        copy(best_x),
        best_f,
        copy(best_x),
        best_f,
        :disabled,
        evaluations,
        0,
        evaluations,
        NaN,
        0.0,
        NaN,
        default_settings,
        TraceRecord{T}[],
    )
end

"""
    optimize(f, lower, upper; rng, algorithm, popsize, maxiters, maxevals, F, CR, memory_size, pmax, target, history, parallel, local_refine, local_method, local_maxiters, local_tol, trace_history, job_id, message, message_every, message_mode)

Minimize `f` over a box defined by `lower` and `upper`.
`rng` is required and controls all randomness.

If `parallel=true`, trials are generated and evaluated in parallel across threads
using a generation-synchronous update. Results can differ from the default
asynchronous update, but remain reproducible for deterministic `f`.
If `parallel=:auto`, the mode is chosen automatically (`true` when
`Threads.nthreads() > 1`, otherwise `false`).

If `local_refine=true`, a local optimization phase is started from the DE best
solution. `local_method` selects `:nelder_mead` or `:lbfgs`. If local
optimization fails, the DE best solution is returned safely.

If `trace_history=true`, per-generation best records are stored in `result.trace`
and can be exported with `write_trace_csv`.

If `message=true`, progress is printed during optimization. `message_mode=:compact`
prints a lightweight summary; `message_mode=:detailed` also prints full `best_x`.
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
    parallel::Union{Bool, Symbol} = false,
    local_refine::Bool = false,
    local_method::Symbol = :nelder_mead,
    local_maxiters::Int = 200,
    local_tol::Real = 1e-8,
    trace_history::Bool = false,
    job_id::Int = 0,
    message::Bool = false,
    message_every::Int = 1,
    message_mode::Symbol = :compact,
)
    total_start_ns = time_ns()
    if isempty(lower) || isempty(upper)
        throw(ArgumentError("lower and upper must be non-empty"))
    end
    if length(lower) != length(upper)
        throw(ArgumentError("lower and upper must have the same length"))
    end
    dim = length(lower)
    if isnothing(popsize)
        if algorithm == :jso
            popsize = max(4, round(Int, 25 * log(dim) * sqrt(dim)))
        else
            popsize = max(10 * dim, 4)
        end
    end
    maxevals = isnothing(maxevals) ? popsize * (maxiters + 1) : maxevals

    _validate_inputs(
        lower,
        upper,
        popsize,
        maxiters,
        maxevals,
        F,
        CR,
        algorithm,
        memory_size,
        pmax,
        target,
        parallel,
        local_method,
        local_maxiters,
        local_tol,
        message_every,
        message_mode,
    )

    T = promote_type(Float64, eltype(lower), eltype(upper), typeof(F), typeof(CR), typeof(target))
    lower_t = Vector{T}(lower)
    upper_t = Vector{T}(upper)
    F_t = T(F)
    CR_t = T(CR)
    pmax_t = T(pmax)
    target_t = T(target)
    local_tol_t = T(local_tol)
    if isnothing(memory_size)
        memory_size = algorithm == :jso ? 5 : popsize
    end
    resolved_parallel = _resolve_parallel(parallel)

    pop = Matrix{T}(undef, dim, popsize)
    fitness = Vector{T}(undef, popsize)
    trial = Vector{T}(undef, dim)
    objective_value_or_inf = message ? _objective_value_or_inf_verbose : _objective_value_or_inf_quiet

    for i in 1:popsize
        for j in 1:dim
            pop[j, i] = lower_t[j] + rand(rng, T) * (upper_t[j] - lower_t[j])
        end
        fitness[i] = objective_value_or_inf(
            f,
            view(pop, :, i),
            T,
            job_id,
            :initialization,
            i,
            i,
        )
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

    de_start_ns = time_ns()
    trace_rows = trace_history ? NamedTuple[] : nothing
    if !isnothing(trace_rows)
        sizehint!(trace_rows, maxiters + (local_refine ? 2 : 0))
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
        resolved_parallel,
        objective_value_or_inf,
        trace_rows,
        job_id,
        message,
        message_every,
        message_mode,
    )
    elapsed_de_sec = (time_ns() - de_start_ns) / 1e9

    de_status = if best_f <= target_t
        :target_reached
    elseif iterations >= maxiters
        :maxiters
    elseif evaluations >= maxevals
        :maxevals
    else
        :stopped
    end

    de_best_x = copy(best_x)
    de_best_f = best_f
    de_evaluations = evaluations
    trace = TraceRecord{T}[]
    if trace_history
        for trace_row in trace_rows
            push!(
                trace,
                TraceRecord{T}(
                    trace_row.generation,
                    trace_row.job_id,
                    trace_row.phase,
                    trace_row.best_x,
                    trace_row.best_f,
                    trace_row.evaluations,
                ),
            )
        end
    end
    local_best_x = copy(de_best_x)
    local_best_f = de_best_f
    local_status = :disabled
    local_evaluations = 0
    elapsed_local_sec = 0.0

    if local_refine
        if message
            if message_mode == :detailed
                println(
                    "[LOCAL-START] job=",
                    job_id,
                    " generation=",
                    iterations + 1,
                    " evaluations=",
                    de_evaluations,
                    " method=",
                    local_method,
                    " de_best_f=",
                    de_best_f,
                    " de_best_x=",
                    de_best_x,
                )
            else
                println(
                    "[LOCAL-START] job=",
                    job_id,
                    " generation=",
                    iterations + 1,
                    " evaluations=",
                    de_evaluations,
                    " method=",
                    local_method,
                    " de_best_f=",
                    de_best_f,
                )
            end
        end
        if trace_history
            push!(
                trace,
                TraceRecord{T}(
                    iterations + 1,
                    job_id,
                    :local_start,
                    copy(de_best_x),
                    de_best_f,
                    de_evaluations,
                ),
            )
        end
        local_best_x, local_best_f, local_status, local_evaluations, elapsed_local_sec = _run_local_refinement(
            f,
            de_best_x,
            de_best_f,
            lower_t,
            upper_t,
            local_method,
            local_maxiters,
            local_tol,
            job_id,
            de_evaluations,
        )
        if trace_history
            push!(
                trace,
                TraceRecord{T}(
                    iterations + 1,
                    job_id,
                    :local_end,
                    copy(local_best_x),
                    local_best_f,
                    de_evaluations + local_evaluations,
                ),
            )
        end
        if message
            if message_mode == :detailed
                println(
                    "[LOCAL-END] job=",
                    job_id,
                    " generation=",
                    iterations + 1,
                    " evaluations=",
                    de_evaluations + local_evaluations,
                    " status=",
                    local_status,
                    " local_best_f=",
                    local_best_f,
                    " local_best_x=",
                    local_best_x,
                )
            else
                println(
                    "[LOCAL-END] job=",
                    job_id,
                    " generation=",
                    iterations + 1,
                    " evaluations=",
                    de_evaluations + local_evaluations,
                    " status=",
                    local_status,
                    " local_best_f=",
                    local_best_f,
                )
            end
        end
        if local_best_f < de_best_f
            best_x = copy(local_best_x)
            best_f = local_best_f
        else
            best_x = copy(de_best_x)
            best_f = de_best_f
        end
    end

    total_evaluations = de_evaluations + local_evaluations
    elapsed_total_sec = (time_ns() - total_start_ns) / 1e9
    status = best_f <= target_t ? :target_reached : :not_reached
    run_settings = RunSettings(
        algorithm,
        popsize,
        maxiters,
        maxevals,
        F_t,
        CR_t,
        memory_size,
        pmax_t,
        target_t,
        history,
        resolved_parallel,
        local_refine,
        local_method,
        local_maxiters,
        local_tol_t,
        trace_history,
        job_id,
        message,
        message_every,
        message_mode,
    )
    return Result(
        best_x,
        best_f,
        status,
        de_status,
        total_evaluations,
        iterations,
        history_best,
        de_best_x,
        de_best_f,
        local_best_x,
        local_best_f,
        local_status,
        de_evaluations,
        local_evaluations,
        total_evaluations,
        elapsed_de_sec,
        elapsed_local_sec,
        elapsed_total_sec,
        run_settings,
        trace,
    )
end

"""
    find_sublevel_point(f, lower, upper; c, c_tol=0.0, rng, algorithm=:de, popsize=nothing, maxiters=1000, maxevals=nothing, F=0.8, CR=0.9, memory_size=nothing, pmax=0.2, history=false, parallel=false, trace_history=false, job_id=0, message=false, message_every=1, message_mode=:compact)

Search for a point satisfying `f(x) <= c + c_tol` inside the box `[lower, upper]`.

Returns `SublevelSearchResult` and stops early when the threshold is found.
"""
function find_sublevel_point(
    f,
    lower::AbstractVector,
    upper::AbstractVector;
    c::Real,
    c_tol::Real = 0.0,
    rng::AbstractRNG,
    algorithm::Symbol = :de,
    popsize::Union{Int, Nothing} = nothing,
    maxiters::Int = 1000,
    maxevals::Union{Int, Nothing} = nothing,
    F::Real = 0.8,
    CR::Real = 0.9,
    memory_size::Union{Int, Nothing} = nothing,
    pmax::Real = 0.2,
    history::Bool = false,
    parallel::Union{Bool, Symbol} = false,
    trace_history::Bool = false,
    job_id::Int = 0,
    message::Bool = false,
    message_every::Int = 1,
    message_mode::Symbol = :compact,
)
    if !isfinite(c)
        throw(ArgumentError("c must be finite"))
    end
    if !isfinite(c_tol)
        throw(ArgumentError("c_tol must be finite"))
    end
    if c_tol < 0
        throw(ArgumentError("c_tol must be >= 0"))
    end
    threshold = c + c_tol
    if !isfinite(threshold)
        throw(ArgumentError("c + c_tol must be finite"))
    end

    optimize_result = optimize(
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
        memory_size=memory_size,
        pmax=pmax,
        target=threshold,
        history=history,
        parallel=parallel,
        local_refine=false,
        trace_history=trace_history,
        job_id=job_id,
        message=message,
        message_every=message_every,
        message_mode=message_mode,
    )

    overlap = optimize_result.de_best_f <= threshold
    stop_reason = if overlap
        :overlap_found
    elseif optimize_result.de_status == :maxevals
        :maxevals
    elseif optimize_result.de_status == :maxiters
        :maxiters
    else
        :stopped
    end

    return SublevelSearchResult(
        overlap,
        copy(optimize_result.de_best_x),
        optimize_result.de_best_f,
        optimize_result.de_evaluations,
        stop_reason,
    )
end

"""
    write_trace_csv(result, output_path)

Write `result.trace` to a CSV file.
Columns:
- `job_id,generation,phase,evaluations,best_f`
- `best_x_1 ... best_x_D`
"""
function write_trace_csv(result::Result{T}, output_path::AbstractString) where {T}
    dimension = isempty(result.trace) ? length(result.best_x) : length(result.trace[1].best_x)
    open(output_path, "w") do io
        print(io, "job_id,generation,phase,evaluations,best_f")
        for dimension_index in 1:dimension
            print(io, ",best_x_", dimension_index)
        end
        print(io, "\n")
        for trace_row in result.trace
            print(
                io,
                trace_row.job_id,
                ",",
                trace_row.generation,
                ",",
                trace_row.phase,
                ",",
                trace_row.evaluations,
                ",",
                trace_row.best_f,
            )
            for parameter_value in trace_row.best_x
                print(io, ",", parameter_value)
            end
            print(io, "\n")
        end
    end
    return output_path
end

end # module
