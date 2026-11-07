# Minimizers

Global optimisers used in calibrations.

## Differential evolution (`de.jl`)
- `makeDifferentialEvolutionOptimizer(params, problem, rng, strategy)` builds an optimiser over a bounded `Problem`.
- `makeDifferentialEvolutionOptimizerWithElite` keeps elite members across generations.
- Mutation strategies: `Best1Bin`, `Best2Bin`, `Rand1Bin`, `Rand1EitherOr`, `Rand1Exp`.
- `optimize(o, TerminationCriteria(...))` runs it; `minimum`, `minimizer` and `numberOfEvaluations` report the result. Optional quasi-random initialisation (`useQuasiInit`) and a starting point (`x0`) are supported.

## Simulated annealing (`sa.jl`)
- `SimulatedAnnealing(problem, rng)` with `optimize`, `minimum`, `minimizer` and `numberOfEvaluations`. Status is reported with `SAStatusCode`.

Both work on a `Problem` holding the objective and the box bounds.
