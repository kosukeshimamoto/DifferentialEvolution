const _ALGORITHMS = (:de, :shade, :lshade, :jso)
const _LOCAL_METHODS = (:nelder_mead, :lbfgs)
const _MESSAGE_MODES = (:compact, :detailed)

function _validate_inputs(
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
    local_method,
    local_maxiters,
    local_tol,
    message_every,
    message_mode,
)
    if isempty(lower) || isempty(upper)
        throw(ArgumentError("lower and upper must be non-empty"))
    end
    if length(lower) != length(upper)
        throw(ArgumentError("lower and upper must have the same length"))
    end
    for i in eachindex(lower, upper)
        if !isfinite(lower[i]) || !isfinite(upper[i])
            throw(ArgumentError("bounds must be finite"))
        end
        if !(lower[i] < upper[i])
            throw(ArgumentError("lower must be strictly less than upper for each dimension"))
        end
    end
    if popsize < 4
        throw(ArgumentError("popsize must be >= 4"))
    end
    if maxiters <= 0
        throw(ArgumentError("maxiters must be positive"))
    end
    if maxevals < popsize
        throw(ArgumentError("maxevals must be >= popsize"))
    end
    if algorithm ∉ _ALGORITHMS
        throw(ArgumentError("algorithm must be one of :de, :shade, :lshade, or :jso"))
    end
    if !(F > 0 && F <= 2)
        throw(ArgumentError("F must be in (0, 2]"))
    end
    if !(CR >= 0 && CR <= 1)
        throw(ArgumentError("CR must be in [0, 1]"))
    end
    if isnan(target)
        throw(ArgumentError("target must not be NaN"))
    end
    if local_method ∉ _LOCAL_METHODS
        throw(ArgumentError("local_method must be one of :nelder_mead or :lbfgs"))
    end
    if local_maxiters <= 0
        throw(ArgumentError("local_maxiters must be positive"))
    end
    if !(isfinite(local_tol) && local_tol > 0)
        throw(ArgumentError("local_tol must be finite and positive"))
    end
    if message_every <= 0
        throw(ArgumentError("message_every must be positive"))
    end
    if message_mode ∉ _MESSAGE_MODES
        throw(ArgumentError("message_mode must be one of :compact or :detailed"))
    end
    if algorithm != :de
        if !isnothing(memory_size) && memory_size <= 0
            throw(ArgumentError("memory_size must be positive"))
        end
        if !(pmax > 0 && pmax <= 1)
            throw(ArgumentError("pmax must be in (0, 1]"))
        end
    end
    return nothing
end

function _print_generation_message(
    message::Bool,
    message_every::Int,
    message_mode::Symbol,
    job_id::Int,
    iterations::Int,
    maxiters::Int,
    evaluations::Int,
    maxevals::Int,
    best_f,
    best_x,
    delta_best,
    stall_generations::Int,
)
    if !message
        return nothing
    end
    if iterations % message_every != 0
        return nothing
    end
    if message_mode == :detailed
        println(
            "[DE] job=",
            job_id,
            " generation=",
            iterations,
            "/",
            maxiters,
            " evaluations=",
            evaluations,
            "/",
            maxevals,
            " best_f=",
            best_f,
            " delta_best=",
            delta_best,
            " stall_generations=",
            stall_generations,
            " best_x=",
            best_x,
        )
    else
        println(
            "[DE] job=",
            job_id,
            " generation=",
            iterations,
            "/",
            maxiters,
            " evaluations=",
            evaluations,
            "/",
            maxevals,
            " best_f=",
            best_f,
            " delta_best=",
            delta_best,
            " stall_generations=",
            stall_generations,
        )
    end
    return nothing
end

function _objective_value_or_inf_quiet(f, x, T, job_id, phase, evaluations, candidate_index)
    objective_value = f(x)
    converted_value = try
        T(objective_value)
    catch
        return T(Inf)
    end
    if isfinite(converted_value)
        return converted_value
    end
    return T(Inf)
end

function _objective_value_or_inf_verbose(f, x, T, job_id, phase, evaluations, candidate_index)
    objective_value = f(x)
    converted_value = try
        T(objective_value)
    catch conversion_error
        println(
            stderr,
            "[WARNING] job=",
            job_id,
            " phase=",
            phase,
            " evaluations=",
            evaluations,
            " candidate=",
            candidate_index,
            " objective conversion failed with error=",
            typeof(conversion_error),
            " value_type=",
            typeof(objective_value),
            " value=",
            objective_value,
            " treated_as=Inf",
        )
        return T(Inf)
    end
    if isfinite(converted_value)
        return converted_value
    end
    println(
        stderr,
        "[WARNING] job=",
        job_id,
        " phase=",
        phase,
        " evaluations=",
        evaluations,
        " candidate=",
        candidate_index,
        " objective returned non-finite value=",
        converted_value,
        " treated_as=Inf",
    )
    return T(Inf)
end

function _objective_value_or_inf(f, x, T, message, job_id, phase, evaluations, candidate_index)
    if message
        return _objective_value_or_inf_verbose(f, x, T, job_id, phase, evaluations, candidate_index)
    end
    return _objective_value_or_inf_quiet(f, x, T, job_id, phase, evaluations, candidate_index)
end

function _rand1distinct(rng, n, exclude)
    r = rand(rng, 1:n)
    while r == exclude
        r = rand(rng, 1:n)
    end
    return r
end

function _rand3distinct(rng, n, exclude)
    r1 = rand(rng, 1:n)
    while r1 == exclude
        r1 = rand(rng, 1:n)
    end
    r2 = rand(rng, 1:n)
    while r2 == exclude || r2 == r1
        r2 = rand(rng, 1:n)
    end
    r3 = rand(rng, 1:n)
    while r3 == exclude || r3 == r1 || r3 == r2
        r3 = rand(rng, 1:n)
    end
    return r1, r2, r3
end

function _rand_cauchy(rng, mean, scale)
    u = rand(rng)
    return mean + scale * tan(pi * (u - 0.5))
end

function _weighted_arith_mean(values, weights)
    total = sum(weights)
    return sum(values .* weights) / total
end

function _weighted_lehmer_mean(values, weights)
    num = sum(weights .* (values .^ 2))
    den = sum(weights .* values)
    return num / den
end

function _archive_push!(archive, limit, vec, rng)
    if limit <= 0
        return nothing
    end
    archive_len = length(archive)
    if archive_len < limit
        push!(archive, copy(vec))
        return nothing
    end
    idx = rand(rng, 1:(archive_len + 1))
    if idx <= archive_len
        copyto!(archive[idx], vec)
    end
    return nothing
end

function _archive_trim!(archive, limit, rng)
    while length(archive) > limit
        idx = rand(rng, 1:length(archive))
        archive[idx] = archive[end]
        pop!(archive)
    end
    return nothing
end

function _select_union_vector(rng, pop, archive, popsize, i, r1)
    archive_len = length(archive)
    total = popsize + archive_len
    while true
        idx = rand(rng, 1:total)
        if idx <= popsize
            if idx != i && idx != r1
                return view(pop, :, idx)
            end
        else
            return archive[idx - popsize]
        end
    end
end

function _select_pbest_index(rng, sorted_idx, pcount, i)
    if pcount <= 1
        return sorted_idx[1]
    end
    while true
        idx = sorted_idx[rand(rng, 1:pcount)]
        if idx != i || pcount == 1
            return idx
        end
    end
end

abstract type AbstractStrategy end

struct DEStrategy{T} <: AbstractStrategy
    F::T
    CR::T
end

mutable struct SHADEState{T} <: AbstractStrategy
    H::Int
    MCR::Vector{T}
    MF::Vector{T}
    k::Int
    pmax::T
    cr_vals::Vector{T}
    f_vals::Vector{T}
    success_cr::Vector{T}
    success_f::Vector{T}
    success_df::Vector{T}
    sorted_idx::Vector{Int}
    archive::Vector{Vector{T}}
    archive_limit::Int
end

mutable struct LSHADEState{T} <: AbstractStrategy
    H::Int
    MCR::Vector{T}
    MF::Vector{T}
    k::Int
    pmax::T
    cr_vals::Vector{T}
    f_vals::Vector{T}
    success_cr::Vector{T}
    success_f::Vector{T}
    success_df::Vector{T}
    sorted_idx::Vector{Int}
    archive::Vector{Vector{T}}
    archive_limit::Int
    init_popsize::Int
    min_popsize::Int
    maxevals::Int
end

mutable struct JSOState{T} <: AbstractStrategy
    H::Int
    MCR::Vector{T}
    MF::Vector{T}
    k::Int
    pmax::T
    pmin::T
    p::T
    cr_vals::Vector{T}
    f_vals::Vector{T}
    success_cr::Vector{T}
    success_f::Vector{T}
    success_df::Vector{T}
    sorted_idx::Vector{Int}
    archive::Vector{Vector{T}}
    archive_limit::Int
    init_popsize::Int
    min_popsize::Int
    maxevals::Int
    maxiters::Int
    iteration::Int
    fixed_idx::Int
end

function _make_shade_state(T, popsize, memory_size, pmax)
    H = memory_size
    success_cr = T[]
    sizehint!(success_cr, popsize)
    success_f = T[]
    sizehint!(success_f, popsize)
    success_df = T[]
    sizehint!(success_df, popsize)
    archive = Vector{Vector{T}}()
    sizehint!(archive, popsize)
    return SHADEState(
        H,
        fill(T(0.5), H),
        fill(T(0.5), H),
        1,
        T(pmax),
        Vector{T}(undef, popsize),
        Vector{T}(undef, popsize),
        success_cr,
        success_f,
        success_df,
        Vector{Int}(undef, popsize),
        archive,
        popsize,
    )
end

function _make_lshade_state(T, popsize, memory_size, pmax, maxevals)
    H = memory_size
    success_cr = T[]
    sizehint!(success_cr, popsize)
    success_f = T[]
    sizehint!(success_f, popsize)
    success_df = T[]
    sizehint!(success_df, popsize)
    archive = Vector{Vector{T}}()
    sizehint!(archive, popsize)
    return LSHADEState(
        H,
        fill(T(0.5), H),
        fill(T(0.5), H),
        1,
        T(pmax),
        Vector{T}(undef, popsize),
        Vector{T}(undef, popsize),
        success_cr,
        success_f,
        success_df,
        Vector{Int}(undef, popsize),
        archive,
        popsize,
        popsize,
        4,
        maxevals,
    )
end

function _make_jso_state(T, popsize, memory_size, pmax, maxevals, maxiters)
    H = memory_size
    MCR = fill(T(0.8), H)
    MF = fill(T(0.3), H)
    fixed_idx = H > 1 ? H : 0
    if fixed_idx > 0
        MCR[fixed_idx] = T(0.9)
        MF[fixed_idx] = T(0.9)
    end
    pmax_t = T(pmax)
    pmin_t = pmax_t / 2
    success_cr = T[]
    sizehint!(success_cr, popsize)
    success_f = T[]
    sizehint!(success_f, popsize)
    success_df = T[]
    sizehint!(success_df, popsize)
    archive = Vector{Vector{T}}()
    sizehint!(archive, popsize)
    return JSOState(
        H,
        MCR,
        MF,
        1,
        pmax_t,
        pmin_t,
        pmin_t,
        Vector{T}(undef, popsize),
        Vector{T}(undef, popsize),
        success_cr,
        success_f,
        success_df,
        Vector{Int}(undef, popsize),
        archive,
        popsize,
        popsize,
        4,
        maxevals,
        maxiters,
        0,
        fixed_idx,
    )
end

_start_generation!(::AbstractStrategy, pop, fitness, rng, iterations, maxiters, evaluations, maxevals) = nothing

function _prepare_generation!(state, pop, fitness)
    popsize = size(pop, 2)
    if length(state.sorted_idx) != popsize
        resize!(state.sorted_idx, popsize)
    end
    sortperm!(state.sorted_idx, fitness)
    empty!(state.success_cr)
    empty!(state.success_f)
    empty!(state.success_df)
    if length(state.cr_vals) != popsize
        resize!(state.cr_vals, popsize)
    end
    if length(state.f_vals) != popsize
        resize!(state.f_vals, popsize)
    end
    return nothing
end

function _start_generation!(state::SHADEState, pop, fitness, rng, iterations, maxiters, evaluations, maxevals)
    return _prepare_generation!(state, pop, fitness)
end

function _start_generation!(state::LSHADEState, pop, fitness, rng, iterations, maxiters, evaluations, maxevals)
    return _prepare_generation!(state, pop, fitness)
end

function _start_generation!(state::JSOState, pop, fitness, rng, iterations, maxiters, evaluations, maxevals)
    _prepare_generation!(state, pop, fitness)
    state.iteration = iterations
    return nothing
end

function _generate_trial!(state::DEStrategy, trial, pop, fitness, i, rng, lower_t, upper_t, evaluations)
    dim = size(pop, 1)
    popsize = size(pop, 2)
    r1, r2, r3 = _rand3distinct(rng, popsize, i)
    jrand = rand(rng, 1:dim)
    @inbounds for j in 1:dim
        mutant = pop[j, r1] + state.F * (pop[j, r2] - pop[j, r3])
        if rand(rng) < state.CR || j == jrand
            trial[j] = mutant
        else
            trial[j] = pop[j, i]
        end
        if trial[j] < lower_t[j]
            trial[j] = lower_t[j]
        elseif trial[j] > upper_t[j]
            trial[j] = upper_t[j]
        end
    end
    return nothing
end

function _shade_trial!(state, trial, pop, fitness, i, rng, lower_t, upper_t, evaluations)
    dim = size(pop, 1)
    popsize = size(pop, 2)
    ri = rand(rng, 1:state.H)
    mcr = state.MCR[ri]
    if isnan(mcr)
        CR = zero(mcr)
    else
        CR = mcr + 0.1 * randn(rng)
        if CR < 0
            CR = zero(mcr)
        elseif CR > 1
            CR = one(mcr)
        end
    end

    F = _rand_cauchy(rng, state.MF[ri], 0.1)
    while F <= 0
        F = _rand_cauchy(rng, state.MF[ri], 0.1)
    end
    if F > 1
        F = one(F)
    end

    state.cr_vals[i] = CR
    state.f_vals[i] = F

    pmin = 2 / popsize
    pmax = state.pmax
    p = pmin
    if pmax > pmin
        p = pmin + rand(rng) * (pmax - pmin)
    end
    pcount = max(2, floor(Int, p * popsize))
    pcount = min(pcount, popsize)
    pbest_idx = _select_pbest_index(rng, state.sorted_idx, pcount, i)

    r1 = _rand1distinct(rng, popsize, i)
    r2 = _select_union_vector(rng, pop, state.archive, popsize, i, r1)

    jrand = rand(rng, 1:dim)
    @inbounds for j in 1:dim
        mutant = pop[j, i] + F * (pop[j, pbest_idx] - pop[j, i]) + F * (pop[j, r1] - r2[j])
        if mutant < lower_t[j]
            mutant = (lower_t[j] + pop[j, i]) / 2
        elseif mutant > upper_t[j]
            mutant = (upper_t[j] + pop[j, i]) / 2
        end
        if rand(rng) < CR || j == jrand
            trial[j] = mutant
        else
            trial[j] = pop[j, i]
        end
    end
    return nothing
end

function _generate_trial!(state::SHADEState, trial, pop, fitness, i, rng, lower_t, upper_t, evaluations)
    return _shade_trial!(state, trial, pop, fitness, i, rng, lower_t, upper_t, evaluations)
end

function _generate_trial!(state::LSHADEState, trial, pop, fitness, i, rng, lower_t, upper_t, evaluations)
    return _shade_trial!(state, trial, pop, fitness, i, rng, lower_t, upper_t, evaluations)
end

function _generate_trial!(state::JSOState, trial, pop, fitness, i, rng, lower_t, upper_t, evaluations)
    dim = size(pop, 1)
    popsize = size(pop, 2)
    ri = rand(rng, 1:state.H)
    mcr = state.MCR[ri]
    if isnan(mcr)
        CR = zero(mcr)
    else
        CR = mcr + 0.1 * randn(rng)
        if CR < 0
            CR = zero(mcr)
        elseif CR > 1
            CR = one(mcr)
        end
    end

    F = _rand_cauchy(rng, state.MF[ri], 0.1)
    while F <= 0
        F = _rand_cauchy(rng, state.MF[ri], 0.1)
    end
    if F > 1
        F = one(F)
    end

    if state.maxiters > 0
        ratio = state.iteration / state.maxiters
        if ratio < 0.25
            CR = max(CR, oftype(CR, 0.7))
        elseif ratio < 0.5
            CR = max(CR, oftype(CR, 0.6))
        end
        if ratio < 0.6 && F > 0.7
            F = oftype(F, 0.7)
        end
    end

    state.cr_vals[i] = CR
    state.f_vals[i] = F

    p = state.p
    pcount = max(2, floor(Int, p * popsize))
    pcount = min(pcount, popsize)
    pbest_idx = _select_pbest_index(rng, state.sorted_idx, pcount, i)

    r1 = _rand1distinct(rng, popsize, i)
    r2 = _select_union_vector(rng, pop, state.archive, popsize, i, r1)

    Fw = F
    if state.maxevals > 0
        eval_ratio = evaluations / state.maxevals
        if eval_ratio < 0.2
            Fw = oftype(F, 0.7) * F
        elseif eval_ratio < 0.4
            Fw = oftype(F, 0.8) * F
        else
            Fw = oftype(F, 1.2) * F
        end
    end

    jrand = rand(rng, 1:dim)
    @inbounds for j in 1:dim
        mutant = pop[j, i] + Fw * (pop[j, pbest_idx] - pop[j, i]) + F * (pop[j, r1] - r2[j])
        if mutant < lower_t[j] || mutant > upper_t[j]
            mutant = lower_t[j] + rand(rng, eltype(lower_t)) * (upper_t[j] - lower_t[j])
        end
        if rand(rng) < CR || j == jrand
            trial[j] = mutant
        else
            trial[j] = pop[j, i]
        end
    end
    return nothing
end

_on_success!(::AbstractStrategy, i, trial, f_trial, f_parent, pop, rng) = nothing

function _on_success!(state::Union{SHADEState, LSHADEState}, i, trial, f_trial, f_parent, pop, rng)
    if f_trial < f_parent
        push!(state.success_cr, state.cr_vals[i])
        push!(state.success_f, state.f_vals[i])
        push!(state.success_df, abs(f_trial - f_parent))
        _archive_push!(state.archive, state.archive_limit, view(pop, :, i), rng)
    end
    return nothing
end

function _on_success!(state::JSOState, i, trial, f_trial, f_parent, pop, rng)
    if f_trial < f_parent
        push!(state.success_cr, state.cr_vals[i])
        push!(state.success_f, state.f_vals[i])
        push!(state.success_df, abs(f_trial - f_parent))
        _archive_push!(state.archive, state.archive_limit, view(pop, :, i), rng)
    end
    return nothing
end

_end_generation!(state::AbstractStrategy, pop, fitness, rng, evaluations) = (pop, fitness)

function _end_generation!(state::SHADEState, pop, fitness, rng, evaluations)
    if !isempty(state.success_cr)
        state.MCR[state.k] = _weighted_arith_mean(state.success_cr, state.success_df)
        state.MF[state.k] = _weighted_lehmer_mean(state.success_f, state.success_df)
        state.k = state.k == state.H ? 1 : state.k + 1
    end
    return pop, fitness
end

function _lshade_population_size(state::Union{LSHADEState, JSOState}, evaluations)
    nfe = min(evaluations, state.maxevals)
    slope = (state.min_popsize - state.init_popsize) / state.maxevals
    new_size = round(Int, slope * nfe + state.init_popsize)
    return clamp(new_size, state.min_popsize, state.init_popsize)
end

function _reduce_population(pop, fitness, new_size)
    idx = sortperm(fitness)
    keep = idx[1:new_size]
    return pop[:, keep], fitness[keep]
end

function _end_generation!(state::LSHADEState, pop, fitness, rng, evaluations)
    if !isempty(state.success_cr)
        if maximum(state.success_cr) == 0
            state.MCR[state.k] = oftype(state.MCR[state.k], NaN)
        else
            state.MCR[state.k] = _weighted_lehmer_mean(state.success_cr, state.success_df)
        end
        state.MF[state.k] = _weighted_lehmer_mean(state.success_f, state.success_df)
        state.k = state.k == state.H ? 1 : state.k + 1
    end

    new_size = _lshade_population_size(state, evaluations)
    if new_size < size(pop, 2)
        pop, fitness = _reduce_population(pop, fitness, new_size)
        state.archive_limit = new_size
        _archive_trim!(state.archive, state.archive_limit, rng)
    end

    return pop, fitness
end

function _end_generation!(state::JSOState, pop, fitness, rng, evaluations)
    if !isempty(state.success_cr)
        if state.fixed_idx > 0 && state.k == state.fixed_idx
            state.k = 1
        else
            if maximum(state.success_cr) == 0
                mean_cr = oftype(state.MCR[state.k], NaN)
            else
                mean_cr = _weighted_lehmer_mean(state.success_cr, state.success_df)
            end
            mean_f = _weighted_lehmer_mean(state.success_f, state.success_df)
            state.MCR[state.k] = (state.MCR[state.k] + mean_cr) / 2
            state.MF[state.k] = (state.MF[state.k] + mean_f) / 2
            state.k = state.k == state.H ? 1 : state.k + 1
        end
    end

    if state.maxevals > 0
        nfe = min(evaluations, state.maxevals)
        progress = oftype(state.p, nfe) / oftype(state.p, state.maxevals)
        state.p = state.pmin + (state.pmax - state.pmin) * progress
        state.p = clamp(state.p, state.pmin, state.pmax)
    end

    new_size = _lshade_population_size(state, evaluations)
    if new_size < size(pop, 2)
        pop, fitness = _reduce_population(pop, fitness, new_size)
        state.archive_limit = new_size
        _archive_trim!(state.archive, state.archive_limit, rng)
    end

    return pop, fitness
end

function _run_evolution_serial!(
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
    objective_value_or_inf,
    trace_rows,
    job_id,
    message,
    message_every,
    message_mode,
)
    T = eltype(pop)
    popsize = size(pop, 2)
    track_best_vector = !isnothing(trace_rows) || (message && message_mode == :detailed)
    best_idx = argmin(fitness)
    best_f = fitness[best_idx]
    best_x = copy(view(pop, :, best_idx))

    history_best = history ? Vector{T}(undef, maxiters) : T[]
    iterations = 0
    evaluations = popsize
    stop_due_to_evals = false
    stop_due_to_target = false
    previous_best_f = best_f
    stall_generations = 0

    while iterations < maxiters && evaluations < maxevals && best_f > target_t
        iterations += 1
        _start_generation!(strategy, pop, fitness, rng, iterations, maxiters, evaluations, maxevals)
        popsize = size(pop, 2)
        for i in 1:popsize
            if evaluations >= maxevals
                stop_due_to_evals = true
                break
            end
            _generate_trial!(strategy, trial, pop, fitness, i, rng, lower_t, upper_t, evaluations)
            f_trial = objective_value_or_inf(
                f,
                trial,
                T,
                job_id,
                :de_generation,
                evaluations + 1,
                i,
            )
            evaluations += 1
            if f_trial <= fitness[i]
                _on_success!(strategy, i, trial, f_trial, fitness[i], pop, rng)
                fitness[i] = f_trial
                copyto!(view(pop, :, i), trial)
                if f_trial < best_f
                    best_f = f_trial
                    if track_best_vector
                        copyto!(best_x, trial)
                    end
                end
            end
            if best_f <= target_t
                stop_due_to_target = true
                break
            end
        end

        pop, fitness = _end_generation!(strategy, pop, fitness, rng, evaluations)
        best_idx = argmin(fitness)
        best_f = fitness[best_idx]
        if track_best_vector
            copyto!(best_x, view(pop, :, best_idx))
        end
        if history
            history_best[iterations] = best_f
        end
        if !isnothing(trace_rows)
            push!(
                trace_rows,
                (
                    generation=iterations,
                    job_id=job_id,
                    phase=:de_generation,
                    best_x=copy(best_x),
                    best_f=best_f,
                    evaluations=evaluations,
                ),
            )
        end
        if message
            delta_best = if isfinite(previous_best_f) && isfinite(best_f)
                max(previous_best_f - best_f, zero(T))
            elseif isfinite(best_f) && !isfinite(previous_best_f)
                T(Inf)
            else
                zero(T)
            end
            if best_f < previous_best_f
                stall_generations = 0
            else
                stall_generations += 1
            end
            _print_generation_message(
                message,
                message_every,
                message_mode,
                job_id,
                iterations,
                maxiters,
                evaluations,
                maxevals,
                best_f,
                best_x,
                delta_best,
                stall_generations,
            )
            previous_best_f = best_f
        end
        if stop_due_to_evals || stop_due_to_target
            break
        end
    end

    if history
        resize!(history_best, iterations)
    end
    copyto!(best_x, view(pop, :, best_idx))

    return best_x, best_f, evaluations, iterations, history_best
end

function _run_evolution_parallel!(
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
    objective_value_or_inf,
    trace_rows,
    job_id,
    message,
    message_every,
    message_mode,
)
    T = eltype(pop)
    dim = size(pop, 1)
    popsize = size(pop, 2)
    track_best_vector = !isnothing(trace_rows) || (message && message_mode == :detailed)
    best_idx = argmin(fitness)
    best_f = fitness[best_idx]
    best_x = copy(view(pop, :, best_idx))

    history_best = history ? Vector{T}(undef, maxiters) : T[]
    iterations = 0
    evaluations = popsize
    stop_due_to_evals = false
    previous_best_f = best_f
    stall_generations = 0
    parent_pop_buffer = Matrix{T}(undef, dim, popsize)
    parent_fitness_buffer = Vector{T}(undef, popsize)
    seeds = Vector{UInt64}(undef, popsize)
    trials = Matrix{T}(undef, dim, popsize)
    f_trials = Vector{T}(undef, popsize)
    accept = Vector{Bool}(undef, popsize)

    while iterations < maxiters && evaluations < maxevals && best_f > target_t
        iterations += 1
        _start_generation!(strategy, pop, fitness, rng, iterations, maxiters, evaluations, maxevals)
        popsize = size(pop, 2)

        remaining = maxevals - evaluations
        n_eval = min(popsize, remaining)
        if n_eval <= 0
            stop_due_to_evals = true
            break
        end

        parent_pop = view(parent_pop_buffer, :, 1:popsize)
        parent_fitness = view(parent_fitness_buffer, 1:popsize)
        copyto!(parent_pop, pop)
        copyto!(parent_fitness, fitness)

        @inbounds for i in 1:n_eval
            seeds[i] = rand(rng, UInt64)
        end

        Threads.@threads for i in 1:n_eval
            local_rng = Random.Xoshiro(seeds[i])
            trial_i = view(trials, :, i)
            eval_count = evaluations + (i - 1)
            _generate_trial!(strategy, trial_i, parent_pop, parent_fitness, i, local_rng, lower_t, upper_t, eval_count)
            f_trials[i] = objective_value_or_inf(
                f,
                trial_i,
                T,
                job_id,
                :de_generation,
                eval_count + 1,
                i,
            )
        end
        evaluations += n_eval

        Threads.@threads for i in 1:n_eval
            if f_trials[i] <= parent_fitness[i]
                accept[i] = true
                fitness[i] = f_trials[i]
                copyto!(view(pop, :, i), view(trials, :, i))
            else
                accept[i] = false
            end
        end

        for i in 1:n_eval
            if accept[i]
                _on_success!(strategy, i, view(trials, :, i), f_trials[i], parent_fitness[i], parent_pop, rng)
            end
        end

        pop, fitness = _end_generation!(strategy, pop, fitness, rng, evaluations)
        best_idx = argmin(fitness)
        best_f = fitness[best_idx]
        if track_best_vector
            copyto!(best_x, view(pop, :, best_idx))
        end
        if history
            history_best[iterations] = best_f
        end
        if !isnothing(trace_rows)
            push!(
                trace_rows,
                (
                    generation=iterations,
                    job_id=job_id,
                    phase=:de_generation,
                    best_x=copy(best_x),
                    best_f=best_f,
                    evaluations=evaluations,
                ),
            )
        end
        if message
            delta_best = if isfinite(previous_best_f) && isfinite(best_f)
                max(previous_best_f - best_f, zero(T))
            elseif isfinite(best_f) && !isfinite(previous_best_f)
                T(Inf)
            else
                zero(T)
            end
            if best_f < previous_best_f
                stall_generations = 0
            else
                stall_generations += 1
            end
            _print_generation_message(
                message,
                message_every,
                message_mode,
                job_id,
                iterations,
                maxiters,
                evaluations,
                maxevals,
                best_f,
                best_x,
                delta_best,
                stall_generations,
            )
            previous_best_f = best_f
        end
        if evaluations >= maxevals
            stop_due_to_evals = true
        end
        if stop_due_to_evals
            break
        end
    end

    if history
        resize!(history_best, iterations)
    end
    copyto!(best_x, view(pop, :, best_idx))

    return best_x, best_f, evaluations, iterations, history_best
end

function _run_evolution!(
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
    parallel::Bool,
    objective_value_or_inf,
    trace_rows,
    job_id,
    message,
    message_every,
    message_mode,
)
    if parallel
        return _run_evolution_parallel!(
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
            objective_value_or_inf,
            trace_rows,
            job_id,
            message,
            message_every,
            message_mode,
        )
    end
    return _run_evolution_serial!(
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
        objective_value_or_inf,
        trace_rows,
        job_id,
        message,
        message_every,
        message_mode,
    )
end

function _local_inner_method(local_method::Symbol)
    if local_method == :nelder_mead
        return Optim.NelderMead()
    end
    return Optim.LBFGS()
end

function _extract_real_value(value)
    if value isa Real
        return value
    end
    if hasproperty(value, :value)
        return _extract_real_value(getproperty(value, :value))
    end
    throw(ArgumentError("Objective value type cannot be converted to Real"))
end

function _convert_objective_value(value, ::Type{T}) where {T}
    try
        return T(value)
    catch
        return T(_extract_real_value(value))
    end
end

function _run_lbfgs_with_fallback(
    bounded_objective,
    lower_t,
    upper_t,
    de_best_x,
    local_options,
    job_id,
    de_evaluations,
    objective_calls,
)
    try
        return Optim.optimize(
            bounded_objective,
            lower_t,
            upper_t,
            de_best_x,
            Optim.Fminbox(Optim.LBFGS()),
            local_options;
            autodiff=:forward,
        )
    catch local_error
        println(
            stderr,
            "[WARNING] job=",
            job_id,
            " phase=local_refine",
            " evaluations=",
            de_evaluations + objective_calls[],
            " lbfgs autodiff=:forward failed; retrying autodiff=:finite.",
            " error=",
            typeof(local_error),
            " message=",
            local_error,
        )
        return Optim.optimize(
            bounded_objective,
            lower_t,
            upper_t,
            de_best_x,
            Optim.Fminbox(Optim.LBFGS()),
            local_options;
            autodiff=:finite,
        )
    end
end

function _run_local_refinement(
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
    T = eltype(de_best_x)
    local_best_x = copy(de_best_x)
    local_best_f = de_best_f
    local_status = :failed
    local_evaluations = 0
    local_start_ns = time_ns()

    objective_calls = Ref(0)
    bounded_objective = function (x)
        objective_calls[] += 1
        return f(x)
    end

    local_options = Optim.Options(
        iterations=local_maxiters,
        x_abstol=local_tol,
        x_reltol=local_tol,
        f_abstol=local_tol,
        f_reltol=local_tol,
        g_abstol=local_tol,
        store_trace=false,
        show_trace=false,
        show_warnings=false,
    )

    local_result = nothing
    try
        if local_method == :lbfgs
            local_result = _run_lbfgs_with_fallback(
                bounded_objective,
                lower_t,
                upper_t,
                de_best_x,
                local_options,
                job_id,
                de_evaluations,
                objective_calls,
            )
        else
            local_result = Optim.optimize(
                bounded_objective,
                lower_t,
                upper_t,
                de_best_x,
                Optim.Fminbox(_local_inner_method(local_method)),
                local_options,
            )
        end
        local_best_x = Vector{T}(Optim.minimizer(local_result))
        local_best_f = _convert_objective_value(Optim.minimum(local_result), T)
        if isfinite(local_best_f)
            local_status = Optim.converged(local_result) ? :success : :stopped
        else
            local_best_x = copy(de_best_x)
            local_best_f = de_best_f
            local_status = :failed
        end
    catch local_error
        println(
            stderr,
            "[WARNING] job=",
            job_id,
            " phase=local_refine",
            " evaluations=",
            de_evaluations + objective_calls[],
            " local optimization failed with error=",
            typeof(local_error),
            " message=",
            local_error,
        )
        local_best_x = copy(de_best_x)
        local_best_f = de_best_f
        local_status = :failed
    end

    local_evaluations = objective_calls[]
    elapsed_local_sec = (time_ns() - local_start_ns) / 1e9
    return local_best_x, local_best_f, local_status, local_evaluations, elapsed_local_sec
end
