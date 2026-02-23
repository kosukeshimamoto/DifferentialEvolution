# Reports

This repository keeps the latest benchmark and visualization outputs under
`reports/` so the results can be reviewed without rerunning scripts.

- `reports/README.md` describes each report folder and how to reproduce it.
- `reports/2d/` contains 2D convergence GIFs and a gallery HTML file.
- `reports/benchmarks/` contains benchmark summaries (CSV/MD/HTML) and plots.

If you want to regenerate any output, see the commands listed in
`reports/README.md`.

## Performance Log (Core Optimization Loop)

This section keeps a human-readable memo of major speed improvements so they can
be tracked across commits.

### Measurement protocol (used for the summary below)

- Objective: 50D Rastrigin
- Algorithms: `:de`, `:shade`, `:lshade`, `:jso`
- Budget: `popsize=200`, `maxevals=100000`
- Repetitions: `reps=20`, `warmups=3`
- Environment: `Threads.nthreads() = 1`
- Baseline commit: `0aa6657`
- Current commit: `d25f102`

### Total speedup summary

- `parallel=true` weighted speedup: `+12.35%`
- `parallel=false` (default-only) weighted speedup: `-0.85%`
- Combined reference (serial + parallel equal weight): `+7.28%`

### `parallel=true` per-algorithm speedup

| Algorithm | Baseline mean (s) | Current mean (s) | Speedup |
| --- | ---: | ---: | ---: |
| `:de` | 0.162262236 | 0.142438127 | +12.22% |
| `:shade` | 0.170986098 | 0.152250333 | +10.96% |
| `:lshade` | 0.123137429 | 0.104970802 | +14.75% |
| `:jso` | 0.113566790 | 0.099910960 | +12.02% |

### Update rule

When a new optimization batch is added:

1. Re-run the same protocol against a fixed baseline commit.
2. Append a new memo block with commit hashes and weighted speedups.
3. Keep this section as the single source of truth for top-line performance deltas.
