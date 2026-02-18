using JSON3
using Statistics

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

function finite_summary(values)
    finite_values = filter(isfinite, values)
    if isempty(finite_values)
        return Dict(
            "count" => 0,
            "min" => NaN,
            "max" => NaN,
            "mean" => NaN,
            "median" => NaN,
            "std" => NaN,
        )
    end
    std_value = length(finite_values) > 1 ? std(finite_values) : 0.0
    return Dict(
        "count" => length(finite_values),
        "min" => minimum(finite_values),
        "max" => maximum(finite_values),
        "mean" => mean(finite_values),
        "median" => median(finite_values),
        "std" => std_value,
    )
end

function parse_best_f(raw_best_f)
    if raw_best_f isa Real
        parsed_value = Float64(raw_best_f)
        return isfinite(parsed_value) ? parsed_value : nothing
    end
    if raw_best_f isa AbstractString
        try
            parsed_value = parse(Float64, raw_best_f)
            return isfinite(parsed_value) ? parsed_value : nothing
        catch
            return nothing
        end
    end
    return nothing
end

function warn_skipped_run(path, seed, reason)
    println(
        stderr,
        "WARNING: skipped invalid run file path=",
        path,
        " seed=",
        seed,
        " reason=",
        reason,
    )
end

function require_positive_int(name, value)
    if value <= 0
        error(name * " must be a positive integer")
    end
    return value
end

results_dir = String(get_setting("results_dir"; default="results"))
top_k = require_positive_int("top_k", parse(Int, String(get_setting("top_k"; default="5"))))
summary_path = joinpath(results_dir, "summary.json")
if isempty(strip(results_dir))
    error("results_dir must be non-empty")
end

if !isdir(results_dir)
    error("results directory not found: " * results_dir)
end

result_files = sort(filter(
    path -> occursin(r"seed_\d+\.json$", basename(path)),
    readdir(results_dir; join=true),
))

if isempty(result_files)
    error("no seed result files found under: " * results_dir)
end

runs = NamedTuple[]
skipped_runs = Vector{Dict{String, Any}}()
for path in result_files
    parsed = try
        JSON3.read(read(path, String), Dict{String, Any})
    catch json_error
        warn_skipped_run(path, "", "invalid_json: " * string(typeof(json_error)))
        push!(
            skipped_runs,
            Dict(
                "path" => path,
                "seed" => "",
                "reason" => "invalid_json: " * string(typeof(json_error)),
            ),
        )
        continue
    end
    best_f = parse_best_f(get(parsed, "best_f", nothing))
    if isnothing(best_f)
        warn_skipped_run(path, get(parsed, "seed", ""), "best_f is missing, non-numeric, or non-finite")
        push!(
            skipped_runs,
            Dict(
                "path" => path,
                "seed" => get(parsed, "seed", ""),
                "reason" => "best_f is missing, non-numeric, or non-finite",
            ),
        )
        continue
    end
    if !haskey(parsed, "seed") || !haskey(parsed, "status") || !haskey(parsed, "local_status")
        warn_skipped_run(path, get(parsed, "seed", ""), "required fields are missing")
        push!(
            skipped_runs,
            Dict(
                "path" => path,
                "seed" => get(parsed, "seed", ""),
                "reason" => "required fields are missing",
            ),
        )
        continue
    end
    try
        push!(runs, (
            seed=Int(parsed["seed"]),
            best_f=best_f,
            status=String(parsed["status"]),
            de_status=String(get(parsed, "de_status", "unknown")),
            local_status=String(parsed["local_status"]),
            path=path,
        ))
    catch
        warn_skipped_run(path, get(parsed, "seed", ""), "failed to parse required fields")
        push!(
            skipped_runs,
            Dict(
                "path" => path,
                "seed" => get(parsed, "seed", ""),
                "reason" => "failed to parse required fields",
            ),
        )
    end
end

if isempty(runs)
    error("no valid seed result files after skipping invalid runs under: " * results_dir)
end

sorted_runs = sort(runs; by=r -> r.best_f)
k = min(top_k, length(sorted_runs))
top_runs = sorted_runs[1:k]
best_run = first(sorted_runs)
best_f_values = [r.best_f for r in sorted_runs]

status_counts = Dict{String, Int}()
de_status_counts = Dict{String, Int}()
local_status_counts = Dict{String, Int}()
for run in sorted_runs
    status_counts[run.status] = get(status_counts, run.status, 0) + 1
    de_status_counts[run.de_status] = get(de_status_counts, run.de_status, 0) + 1
    local_status_counts[run.local_status] = get(local_status_counts, run.local_status, 0) + 1
end

summary = Dict(
    "results_dir" => results_dir,
    "num_runs" => length(sorted_runs),
    "num_skipped_runs" => length(skipped_runs),
    "skipped_runs" => skipped_runs,
    "best_seed" => best_run.seed,
    "best_f" => best_run.best_f,
    "best_result_path" => best_run.path,
    "best_f_summary" => finite_summary(best_f_values),
    "status_counts" => status_counts,
    "de_status_counts" => de_status_counts,
    "local_status_counts" => local_status_counts,
    "top_k" => [Dict(
        "rank" => i,
        "seed" => top_runs[i].seed,
        "best_f" => top_runs[i].best_f,
        "status" => top_runs[i].status,
        "de_status" => top_runs[i].de_status,
        "local_status" => top_runs[i].local_status,
        "path" => top_runs[i].path,
    ) for i in 1:length(top_runs)],
)

open(summary_path, "w") do io
    JSON3.pretty(io, summary)
end

println("best seed: ", best_run.seed)
println("best f: ", best_run.best_f)
println("skipped runs: ", length(skipped_runs))
if !isempty(skipped_runs)
    println(stderr, "WARNING: ", length(skipped_runs), " invalid run file(s) were skipped")
end
println("summary saved: ", summary_path)
