using DifferentialEvolution
using JSON3
using LinearAlgebra
using Random

function get_arg_value(args, name)
    long_flag = "--" * name
    prefix = long_flag * "="
    for i in eachindex(args)
        arg = args[i]
        if arg == long_flag
            if i == length(args)
                error("missing value for " * long_flag)
            end
            return args[i + 1]
        end
        if startswith(arg, prefix)
            return arg[length(prefix)+1:end]
        end
    end
    return nothing
end

function get_setting(name; default=nothing)
    from_env = get(ENV, uppercase(name), "")
    if !isempty(from_env)
        return from_env
    end
    from_args = get_arg_value(ARGS, lowercase(name))
    if !isnothing(from_args)
        return from_args
    end
    return default
end

function parse_bool(raw, default_value)
    if raw === nothing
        return default_value
    end
    lowered = lowercase(strip(String(raw)))
    if lowered in ("1", "true", "yes", "on")
        return true
    end
    if lowered in ("0", "false", "no", "off")
        return false
    end
    error("invalid bool value: " * String(raw))
end

function parse_optional_int(raw)
    if raw === nothing || isempty(strip(String(raw)))
        return nothing
    end
    return parse(Int, String(raw))
end

function require_positive_int(name, value)
    if value <= 0
        error(name * " must be a positive integer")
    end
    return value
end

function require_positive_optional_int(name, value)
    if !isnothing(value) && value <= 0
        error(name * " must be a positive integer when provided")
    end
    return value
end

function json_safe_number(x)
    if x isa Real && !isfinite(x)
        return string(x)
    end
    return x
end

function sphere(x)
    return sum(abs2, x)
end

function rosenbrock(x)
    total = 0.0
    for i in 1:length(x)-1
        total += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
    end
    return total
end

function rastrigin(x)
    n = length(x)
    total = 10.0 * n
    for i in 1:n
        total += x[i]^2 - 10.0 * cos(2.0 * pi * x[i])
    end
    return total
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
    total = 0.0
    for i in 1:n
        total += x[i] * sin(sqrt(abs(x[i])))
    end
    return 418.9829 * n - total
end

function objective_and_bounds(name::String)
    if name == "sphere"
        return sphere, -5.0, 5.0
    end
    if name == "rosenbrock"
        return rosenbrock, -2.0, 2.0
    end
    if name == "rastrigin"
        return rastrigin, -5.12, 5.12
    end
    if name == "ackley"
        return ackley, -32.768, 32.768
    end
    if name == "griewank"
        return griewank, -600.0, 600.0
    end
    if name == "schwefel"
        return schwefel, -500.0, 500.0
    end
    error("unsupported objective: " * name)
end

seed_raw = get_setting("seed"; default=get(ENV, "SLURM_ARRAY_TASK_ID", "1"))
seed = parse(Int, String(seed_raw))
objective_name = lowercase(String(get_setting("objective"; default="rastrigin")))
dim = require_positive_int("dim", parse(Int, String(get_setting("dim"; default="10"))))
algorithm = Symbol(String(get_setting("algorithm"; default="de")))
parallel = parse_bool(get_setting("parallel"; default="false"), false)
local_refine = parse_bool(get_setting("local_refine"; default="false"), false)
local_method = Symbol(String(get_setting("local_method"; default="nelder_mead")))
local_maxiters = require_positive_int("local_maxiters", parse(Int, String(get_setting("local_maxiters"; default="300"))))
local_tol = parse(Float64, String(get_setting("local_tol"; default="1e-8")))

popsize = require_positive_optional_int("popsize", parse_optional_int(get_setting("popsize"; default=nothing)))
maxiters = require_positive_int("maxiters", parse(Int, String(get_setting("maxiters"; default="1000"))))
maxevals = require_positive_optional_int("maxevals", parse_optional_int(get_setting("maxevals"; default=nothing)))
mutation_factor = parse(Float64, String(get_setting("f"; default="0.8")))
crossover_rate = parse(Float64, String(get_setting("cr"; default="0.9")))
memory_size = require_positive_optional_int("memory_size", parse_optional_int(get_setting("memory_size"; default=nothing)))
pmax = parse(Float64, String(get_setting("pmax"; default="0.2")))
target = parse(Float64, String(get_setting("target"; default="-Inf")))
history = parse_bool(get_setting("history"; default="false"), false)
trace_csv = parse_bool(get_setting("trace_csv"; default="false"), false)
message = parse_bool(get_setting("message"; default="false"), false)
message_every = require_positive_int("message_every", parse(Int, String(get_setting("message_every"; default="1"))))

results_dir = String(get_setting("results_dir"; default="results"))
if isempty(strip(results_dir))
    error("results_dir must be non-empty")
end
mkpath(results_dir)
out_path = joinpath(results_dir, "seed_" * string(seed) * ".json")

f, default_lower, default_upper = objective_and_bounds(objective_name)
lower_bound = parse(Float64, String(get_setting("lower"; default=string(default_lower))))
upper_bound = parse(Float64, String(get_setting("upper"; default=string(default_upper))))
lower = fill(lower_bound, dim)
upper = fill(upper_bound, dim)

# Avoid nested parallelism when the objective (or downstream libs) uses BLAS.
LinearAlgebra.BLAS.set_num_threads(1)

rng = MersenneTwister(seed)
started_ns = time_ns()
result = DifferentialEvolution.optimize(
    f,
    lower,
    upper;
    rng=rng,
    algorithm=algorithm,
    popsize=popsize,
    maxiters=maxiters,
    maxevals=maxevals,
    F=mutation_factor,
    CR=crossover_rate,
    memory_size=memory_size,
    pmax=pmax,
    target=target,
    history=history,
    parallel=parallel,
    local_refine=local_refine,
    local_method=local_method,
    local_maxiters=local_maxiters,
    local_tol=local_tol,
    trace_history=trace_csv,
    job_id=seed,
    message=message,
    message_every=message_every,
)
wall_clock_sec = (time_ns() - started_ns) / 1e9

payload = Dict(
    "seed" => seed,
    "objective" => objective_name,
    "dim" => dim,
    "algorithm" => String(algorithm),
    "parallel" => parallel,
    "local_refine" => local_refine,
    "local_method" => String(local_method),
    "best_f" => json_safe_number(result.best_f),
    "best_x" => result.best_x,
    "status" => String(result.status),
    "de_status" => String(result.de_status),
    "de_best_f" => json_safe_number(result.de_best_f),
    "de_best_x" => result.de_best_x,
    "local_best_f" => json_safe_number(result.local_best_f),
    "local_best_x" => result.local_best_x,
    "local_status" => String(result.local_status),
    "de_evaluations" => result.de_evaluations,
    "local_evaluations" => result.local_evaluations,
    "total_evaluations" => result.total_evaluations,
    "iterations" => result.iterations,
    "elapsed_de_sec" => json_safe_number(result.elapsed_de_sec),
    "elapsed_local_sec" => json_safe_number(result.elapsed_local_sec),
    "elapsed_total_sec" => json_safe_number(result.elapsed_total_sec),
    "wall_clock_sec" => json_safe_number(wall_clock_sec),
    "trace_rows" => length(result.trace),
    "settings" => Dict(
        "algorithm" => String(result.settings.algorithm),
        "popsize" => result.settings.popsize,
        "maxiters" => result.settings.maxiters,
        "maxevals" => result.settings.maxevals,
        "F" => result.settings.F,
        "CR" => result.settings.CR,
        "memory_size" => result.settings.memory_size,
        "pmax" => result.settings.pmax,
        "target" => json_safe_number(result.settings.target),
        "history" => result.settings.history,
        "parallel" => result.settings.parallel,
        "local_refine" => result.settings.local_refine,
        "local_method" => String(result.settings.local_method),
        "local_maxiters" => result.settings.local_maxiters,
        "local_tol" => result.settings.local_tol,
        "trace_history" => result.settings.trace_history,
        "job_id" => result.settings.job_id,
        "message" => result.settings.message,
        "message_every" => result.settings.message_every,
    ),
    "julia_num_threads" => Threads.nthreads(),
    "slurm_cpus_per_task" => get(ENV, "SLURM_CPUS_PER_TASK", ""),
    "config" => Dict(
        "lower" => lower_bound,
        "upper" => upper_bound,
        "maxiters" => maxiters,
        "maxevals" => isnothing(maxevals) ? "" : maxevals,
        "popsize" => isnothing(popsize) ? "" : popsize,
        "F" => mutation_factor,
        "CR" => crossover_rate,
        "memory_size" => isnothing(memory_size) ? "" : memory_size,
        "pmax" => pmax,
        "target" => json_safe_number(target),
        "history" => history,
    ),
)

open(out_path, "w") do io
    JSON3.pretty(io, payload)
end

println("saved: ", out_path)

if trace_csv
    DifferentialEvolution.write_trace_csv(
        result,
        joinpath(results_dir, "seed_" * string(seed) * "_trace.csv"),
    )
    println("trace saved: ", joinpath(results_dir, "seed_" * string(seed) * "_trace.csv"))
end
