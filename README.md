# DifferentialEvolution.jl

Small, reproducible Differential Evolution (DE) optimizer for continuous, bound-
constrained minimization. The implementation follows the classic DE/rand/1/bin
scheme with binomial crossover and bound clipping.

## Minimal example

```julia
using DifferentialEvolution
using Random

f(x) = sum(abs2, x)

lower = fill(-5.0, 5)
upper = fill(5.0, 5)

rng = MersenneTwister(42)
result = optimize(f, lower, upper; rng=rng, maxiters=200)

@show result.best_x
@show result.best_f
@show result.status
```

Parallel evaluation (generation-synchronous):

```julia
rng = MersenneTwister(42)
res_parallel = optimize(f, lower, upper; rng=rng, maxiters=200, parallel=true)
```

Algorithm selection (using the same `f`, `lower`, `upper`):

```julia
rng = MersenneTwister(42)
res_shade = optimize(f, lower, upper; rng=rng, algorithm=:shade, maxiters=200)

rng = MersenneTwister(42)
res_lshade = optimize(f, lower, upper; rng=rng, algorithm=:lshade, maxiters=200)

rng = MersenneTwister(42)
res_jso = optimize(f, lower, upper; rng=rng, algorithm=:jso, maxiters=200, pmax=0.25)
```

## API

```
optimize(f, lower, upper; rng, algorithm, popsize, maxiters, maxevals, F, CR, memory_size, pmax, target, history, parallel)
```

- `f`: objective function taking an `AbstractVector`
- `lower`, `upper`: vectors of bounds (same length, `lower[i] < upper[i]`)

Keywords:

- `rng` (required): random number generator used internally
- `algorithm`: `:de` (default), `:shade`, `:lshade`, or `:jso`
- `popsize`: population size (default `max(10*D, 4)`)
- `maxiters`: maximum number of iterations (default `1000`)
- `maxevals`: maximum number of objective evaluations (default `popsize * (maxiters + 1)`)
- `F`: mutation factor in `(0, 2]` (default `0.8`)
- `CR`: crossover probability in `[0, 1]` (default `0.9`)
- `memory_size`: history size for SHADE/L-SHADE (default `popsize`; for jSO default `5`)
- `pmax`: upper bound for p-best fraction in SHADE/L-SHADE/jSO (default `0.2`; jSO paper uses `0.25`)
- `target`: stop when `best_f <= target` (default `-Inf`)
- `history`: store best objective after each iteration (default `true`)
- `parallel`: evaluate a generation in parallel using threads (default `false`)

Notes:

- `F` and `CR` are used only when `algorithm=:de`.
- `algorithm=:shade` uses success-history based parameter adaptation.
- `algorithm=:lshade` extends SHADE with linear population size reduction
  (population shrinks linearly to 4 by the evaluation budget).
- `algorithm=:jso` is a jSO-style variant of iL-SHADE with weighted
  current-to-pbest mutation, linear population size reduction, and a
  linearly decreasing p schedule.
- `parallel=true` switches to a generation-synchronous update. Trials are
  generated from the same parent population and evaluated in parallel. Results
  can differ from the default asynchronous update but remain reproducible when
  `f` is deterministic.

Return value `Result` fields:

- `best_x`: best solution vector
- `best_f`: objective value at `best_x`
- `status`: `:target_reached`, `:maxiters`, `:maxevals`, or `:stopped`
- `evaluations`: number of objective evaluations
- `iterations`: number of iterations executed
- `history`: best objective value after each iteration (empty when `history=false`)

## Reproducibility

All randomness is controlled by the `rng` you pass in. Using the same seed yields
the same result:

```julia
rng1 = MersenneTwister(1234)
rng2 = MersenneTwister(1234)

res1 = optimize(f, lower, upper; rng=rng1, maxiters=100)
res2 = optimize(f, lower, upper; rng=rng2, maxiters=100)

@assert res1.best_x == res2.best_x
@assert res1.best_f == res2.best_f
```
