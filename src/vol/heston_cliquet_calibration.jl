using Optim
import Random: seed!
import AQFED.Random: Chacha8SIMD, ScrambledSobolSeq, Owen, ScramblingRngAdapter, Well1024a
import AQFED.MonteCarlo: simulateIter, GridConfig, DVSS2X, ListPayoff, CliquetOption, PeriodicCashFlow, ndims
import AQFED.Math: TanhTransformation, inv
import AQFED.GlobalOptimization: Best1Bin, Best2Bin, Rand1Bin, Problem, OptimizerParams, makeDifferentialEvolutionOptimizer, optimize, TerminationCriteria, makeDifferentialEvolutionOptimizerWithElite, extract_population


# ── Build HestonParams from base + overrides ──
function build_params(base::HestonModel, overrides::NamedTuple{names, TupTypes}) where {names, TupTypes}
    params = merge(
        (v0=base.v0, κ=base.κ, θ=base.θ, ρ=base.ρ, σ=base.σ),
        overrides
    )
    # TupTypes is e.g. Tuple{Dual{...}, Dual{...}, Dual{...}}
    # eltype(TupTypes) is the numeric type we need
    T = eltype(TupTypes)  # or promote_type(eltype(TupTypes), Float64)
    HestonModel(
        HestonParams{T}(
            T(params.v0), T(params.κ), T(params.θ), T(params.ρ), T(params.σ)
        ),
        base.driftCurve,
        base.discountCurve
    )
end

# ── MC simulation: takes a full HestonParams ──
function simulate_outputs(model, spot, payoff, n, config, ndim; seed=42)
	rng = Chacha8SIMD(UInt32)
	seed!(rng, seed)
	seq = ScrambledSobolSeq(ndim, n, Owen(31, ScramblingRngAdapter(rng)))
	return simulateIter(seq, model, spot, payoff, 0, n, config)
end
function interiorize(x0, lower, upper; margin_frac = 0.02)
	x = copy(x0)

	for i in eachindex(x)
		margin = margin_frac * (upper[i] - lower[i])
		x[i] = clamp(x[i], lower[i] + margin, upper[i] - margin)
	end

	return x
end
"""
	de_bounds(x0, paramNames, lower_abs, upper_abs; factor=0.3)

Scale x0 toward absolute lower and upper bounds by a single factor.
For each parameter i:
	lower_i = x0[i] - factor * (x0[i] - lower_abs[i])
	upper_i = x0[i] + factor * (upper_abs[i] - x0[i])

Clamps to absolute bounds to avoid touching the boundary.
"""
function de_bounds(
	x0::AbstractVector{T},
	paramNames::NTuple{N, Symbol},
	lower_abs::AbstractVector,
	upper_abs::AbstractVector;
	factor::Real = 0.5,
) where {T <: Real, N}

	length(x0) == N || throw(ArgumentError("x0 length != paramNames length"))
	length(lower_abs) == N || throw(ArgumentError("lower_abs length != N"))
	length(upper_abs) == N || throw(ArgumentError("upper_abs length != N"))

	lower = similar(x0, T)
	upper = similar(x0, T)

	for i in 1:N
		xi = x0[i]
		lo = T(lower_abs[i])
		hi = T(upper_abs[i])

		lower[i] = max(lo, xi - T(factor) * (xi - lo))
		upper[i] = min(hi, xi + T(factor) * (hi - xi))
	end

	return lower, upper
end

# Convenience: extract x0 from model
function de_bounds(
	model::HestonModel,
	paramNames::NTuple{N, Symbol},
	lower_abs::AbstractVector,
	upper_abs::AbstractVector;
	kwargs...,
) where N

	x0 = [getproperty(model, name) for name in paramNames]
	de_bounds(x0, paramNames, lower_abs, upper_abs; kwargs...)
end
# ── Generic calibrate function ──
function calibrateCliquets(base::HestonModel, paramNames::Tuple, lower, upper, refValues; abstol = 1e-8, config::GridConfig{T} = GridConfig(base, specTimes, 1.0 / 52, DVSS2X(withBB = false), mindt = 0.5/365, tstop = 1.0), n_schedule = [4*1024],
	spot = 1.0, payoff = ListPayoff([CliquetOption(
		cap, -1.0, 1e100, 0.0, 0.0,
		PeriodicCashFlow(
			collect(range(1.0/12, 1.0, length = 12)),
			1.0,
			1, 0.0),
		0.0) for cap in [0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.03, 0.035, 0.04]]), polish = false,
	useSA = false, isProgressive = false, useNM = true, useDE = false, useLBFGS=false,
	lowerGuess = interiorize(lower, lower, upper), upperGuess = interiorize(upper, lower, upper),
    seed = 42, isRelative=false, n_polish = n_schedule[end]
    ) where {T}
	transforms = map((lo, hi) -> TanhTransformation(lo, hi), lower, upper)
	x0 = [getproperty(base, name) for name in paramNames]
	rx0 = map((t, xi) -> inv(t, xi), transforms, x0)
	ndim = ndims(base, config)


	function make_objective(n)
		function objective(xr)
			x = map((t, xri) -> t(xri), transforms, xr)
			overrides = NamedTuple{paramNames}(x)
			model = build_params(base, overrides)
			vals, _ = simulate_outputs(model, spot, payoff, n, config, ndim, seed=seed)
			return isRelative ? sum(abs2, vals ./ refValues .- 1) : sum(abs2, vals .- refValues)
		end
		return objective
	end

	function make_objective_de(n)
		function objective(x)
			if any(x .< lower) || any(x .> upper)
				return Inf
			end
			overrides = NamedTuple{paramNames}(x)
			model = build_params(base, overrides)
			vals, _ = simulate_outputs(model, spot, payoff, n, config, ndim, seed=seed)
			return isRelative ? sum(abs2, vals ./ refValues .- 1) : sum(abs2, vals .- refValues)
		end
		return objective
	end

	rx_best = rx0
	f_best = Inf

	if useDE
		#lowerGuess, upperGuess = de_bounds(base, paramNames, lower, upper)
		objective_de = make_objective_de(n_schedule[1])
		problem = GlobalOptimization.Problem(
			length(x0),
			objective_de,
			lowerGuess,
			upperGuess,
		)
		de = GlobalOptimization.makeDifferentialEvolutionOptimizer(
			GlobalOptimization.OptimizerParams(15),
			problem,
			Chacha8SIMD(UInt32),
			GlobalOptimization.Best1Bin(0.9, 0.5),
			#x0 = x0,
		)
		maxGen = 100
		stagnationMax = 30
		de_result = GlobalOptimization.optimize(
			de,
			GlobalOptimization.TerminationCriteria(
				maxGen,
				stagnationMax,
				1e-8,
				1e-10,
			),
		)

		# Adjust this field name if your GlobalOptimization result differs.
		x_de = GlobalOptimization.minimizer(de)
		println("DE candidate: ", NamedTuple{paramNames}(x_de))
		rx_best = map((t, xi) -> inv(t, xi), transforms, x_de)
	end
	if useSA
		objective_de = make_objective_de(n_schedule[1])
		problem = GlobalOptimization.Problem(
			length(x0),
			objective_de,
			lowerGuess,
			upperGuess,
		)
		optim = GlobalOptimization.SimulatedAnnealing(problem, Well1024a())
		GlobalOptimization.optimize(optim, x0)

		# Adjust this field name if your GlobalOptimization result differs.
		x_de = GlobalOptimization.minimizer(optim)
		println("SA candidate: ", NamedTuple{paramNames}(x_de))
		rx_best = map((t, xi) -> inv(t, xi), transforms, x_de)

	end

	if useNM
		for (stage, n) in enumerate(n_schedule)
			if stage == 1
				# First stage: large simplex for global exploration
				simplexer = Optim.AffineSimplexer(0.15, 0.5)
			else
				# Successive stages: shrink proportional to noise reduction
				n_prev = n_schedule[stage-1]
				scale = 0.15 * sqrt(n_prev / n)
				simplexer = Optim.AffineSimplexer(scale, 0.5)
			end
			xbest = map((t, xi) -> t(xi), transforms, rx_best)
			xbest = interiorize(xbest, lower, upper)
			rx_best = map((t, xi) -> inv(t, xi), transforms, xbest)

			obj = make_objective(n)
			res = Optim.optimize(obj, rx_best,
				NelderMead(),
				#   NelderMead(initial_simplex = simplexer), 
				Optim.Options(f_abstol = abstol, iterations = 300))
			rx_best = Optim.minimizer(res)
			f_best = Optim.minimum(res)
			println("n=$(n): obj=$(f_best), params=$(map((t, xi) -> t(xi), transforms, rx_best))")
		end
	end
	xCalib = map((t, xi) -> t(xi), transforms, rx_best)
	println("Calibrated: ", NamedTuple{paramNames}(xCalib))
	println("Objective  : ", f_best)

	if useLBFGS
		seed = seed*13
		res_lbfgs = Optim.optimize(
		     make_objective(n_polish),
		     rx_best,
		     LBFGS(),
		     Optim.Options(f_abstol=abstol/100, allow_f_increases=true, iterations=50),
		     autodiff=:forward
		 )
		 if Optim.minimum(res_lbfgs) < f_best
		     rx_best = Optim.minimizer(res_lbfgs)
		     f_best = Optim.minimum(res_lbfgs)
		     xCalib = map((t, xi) -> t(xi), transforms, rx_best)
		     println("Calibrated: ", NamedTuple{paramNames}(xCalib))
		     println("Objective  : ", f_best)
		end
	end

	if polish
		seed = seed*13
		#rx_best = map((transformi, xri) -> inv(transformi, xri), transforms, x)
		out = zeros(length(refValues))
		function objectiveND(F, xr)
			x = map((t, xri) -> t(xri), transforms, xr)
			overrides = NamedTuple{paramNames}(x)
			model = build_params(base, overrides)
			vals, _ = simulate_outputs(model, spot, payoff, n_polish, config, ndim, seed=seed)
			F .= (vals - refValues) .* 100
			return F
		end
		gnRmse = GaussNewton.optimize!(objectiveND, rx_best, out, autodiff = :forward)
		#gnRmse = GaussNewton.optimize!(objectiveND, rx_best, out, autodiff = :single, derivstp = 0.001)
		xCalib = map((t, xi) -> t(xi), transforms, rx_best)
		println("Calibrated: ", NamedTuple{paramNames}(xCalib))
		println("Objective  : ", gnRmse)	
	end

	overrides = NamedTuple{paramNames}(xCalib)
	model_calibrated = build_params(base, overrides)
	return model_calibrated, f_best
end



function calibrateCliquetsMultiseed(base::HestonModel, paramNames::Tuple, lower, upper, refValues; abstol = 1e-8, config::GridConfig{T} = GridConfig(base, specTimes, 1.0 / 52, DVSS2X(withBB = false), mindt = 0.5/365, tstop = 1.0), n_schedule=[4*1024], n_polish=n_schedule[end],
	spot = 1.0, payoff = ListPayoff([CliquetOption(
		cap, -1.0, 1e100, 0.0, 0.0,
		PeriodicCashFlow(
			collect(range(1.0/12, 1.0, length = 12)),
			1.0,
			1, 0.0),
		0.0) for cap in [0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.03, 0.035, 0.04]]), polish = false,
    seeds = [42,43,44,45]) where {T}
	transforms = map((lo, hi) -> TanhTransformation(lo, hi), lower, upper)
	x0 = [getproperty(base, name) for name in paramNames]
	rx0 = map((t, xi) -> inv(t, xi), transforms, x0)
	ndim = ndims(base, config)


	function make_objective(n)
		function objective(xr)
			x = map((t, xri) -> t(xri), transforms, xr)
			overrides = NamedTuple{paramNames}(x)
			model = build_params(base, overrides)
			av = 0.0
			for seed in seeds
				vals, _ = simulate_outputs(model, spot, payoff, n, config, ndim, seed=seed)
				err = sum(abs2, vals .- refValues)
				av+=err
			end
			err /= length(seeds)
			err
		end
		return objective
	end


	rx_best = rx0
	f_best = Inf
		for (stage, n) in enumerate(n_schedule)
			xbest = map((t, xi) -> t(xi), transforms, rx_best)
			xbest = interiorize(xbest, lower, upper)
			rx_best = map((t, xi) -> inv(t, xi), transforms, xbest)

			obj = make_objective(n)
			res = Optim.optimize(obj, rx_best,
				NelderMead(),
				#   NelderMead(initial_simplex = simplexer), 
				Optim.Options(f_abstol = abstol, iterations = 400))
			rx_best = Optim.minimizer(res)
			f_best = Optim.minimum(res)
			println("n=$(n): obj=$(f_best), params=$(map((t, xi) -> t(xi), transforms, rx_best))")
		end
	xCalib = map((t, xi) -> t(xi), transforms, rx_best)
	println("Calibrated: ", NamedTuple{paramNames}(xCalib))
	println("Objective  : ", f_best)


	if polish
		#rx_best = map((transformi, xri) -> inv(transformi, xri), transforms, x)
		out = zeros(length(refValues))
		function objectiveND(F, xr)
			x = map((t, xri) -> t(xri), transforms, xr)
			overrides = NamedTuple{paramNames}(x)
			model = build_params(base, overrides)
			vals, _ = simulate_outputs(model, spot, payoff, n_polish, config, ndim, seed=2026)
			F .=  (vals - refValues) .* 100
			return F
		end
		gnRmse = GaussNewton.optimize!(objectiveND, rx_best, out, autodiff = :forward)
		xCalib = map((t, xi) -> t(xi), transforms, rx_best)
		println("Calibrated: ", NamedTuple{paramNames}(xCalib))
		println("Objective  : ", gnRmse)
		# Stage 2: LBFGS from NelderMead solution
		# res_lbfgs = Optim.optimize(
		#     objective,
		#     rx_best,
		#     LBFGS(),
		#     Optim.Options(f_abstol=abstol/100, allow_f_increases=true),
		#     #autodiff=:forward
		# )
		# if Optim.minimum(res_lbfgs) < f_best
		#     rx_best = Optim.minimizer(res_lbfgs)
		#     f_best = Optim.minimum(res_lbfgs)
		#     xCalib = map((t, xi) -> t(xi), transforms, rx_best)
		#     println("Calibrated: ", NamedTuple{paramNames}(xCalib))
		#     println("Objective  : ", f_best)
		# end
	end

	overrides = NamedTuple{paramNames}(xCalib)
	model_calibrated = build_params(base, overrides)
	return model_calibrated, f_best
end


"""
	quadratic_minimum_3best(κs, fs)

Fit a parabola through the 3 grid points with lowest loss
and return the analytic minimum, clamped to their convex hull.
"""
function quadratic_minimum_3best(κs, fs)
	p = sortperm(fs)
	κ_local = κs[p[1:3]]
	f_local = log.(fs[p[1:3]])

	# Fit a*κ^2 + b*κ + c
	A = hcat(κ_local .^ 2, κ_local, ones(3))
	abc = A \ f_local
	a, b, c = abc

	if a > 0
		κ_min = -b / (2a)
		return clamp(κ_min, minimum(κ_local), maximum(κ_local))
	else
		# Not convex: fall back to best grid point
		return κ_local[argmin(f_local)]
	end
end

"""
	calibrateCliquets_kappa_grid(base, paramNames2d, lower2d, upper2d, refValues;
		kappa_grid, kappa_limits, config, n, spot, payoff, ...)

Grid search on κ (1D), 2D Nelder-Mead on (σ,ρ) at each grid point,
quadratic interpolation to best κ, then one 3D Nelder-Mead polish.
"""
function calibrateCliquets_kappa_grid(
	base::HestonModel,
	paramNames2d::NTuple{2, Symbol},
	lower2d,
	upper2d,
	refValues;
	kappa_grid = 1.0:1.0:10.0,
	kappa_limits = (1.0, 10.0),
	abstol = 1e-8,
	config = nothing,
	n = 16*1024,
	spot = 1.0,
	payoff = ListPayoff([CliquetOption(
		cap, -1.0, 1e100, 0.0, 0.0,
		PeriodicCashFlow(
			collect(range(1.0/12, 1.0, length = 12)),
			1.0, 1, 0.0),
		0.0) for cap in [0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.03, 0.035, 0.04]]),
	lowerGuess2d = interiorize(lower2d, lower2d, upper2d),
	upperGuess2d = interiorize(upper2d, lower2d, upper2d),
)

	# ── Stage 1: 2D calibration for each κ on the grid ──
	n_grid = length(kappa_grid)
	κs = collect(kappa_grid)
	fs = Vector{Float64}(undef, n_grid)
	σs = Vector{Float64}(undef, n_grid)
	ρs = Vector{Float64}(undef, n_grid)

	println("=== Kappa grid search ===")

	for (i, κ) in enumerate(kappa_grid)
		# Fix κ in the base model
		model_κ = build_params(base, (κ = κ,))

		cal_model, f_val = calibrateCliquets(
			model_κ,
			paramNames2d,
			lower2d,
			upper2d,
			refValues;
			abstol = abstol,
			config = config,
			n_schedule = [n],
			spot = spot,
			payoff = payoff,
			polish = false,
			lowerGuess = lowerGuess2d,
			upperGuess = upperGuess2d,
		)

		σs[i] = cal_model.σ
		ρs[i] = cal_model.ρ
		fs[i] = f_val

		println("κ=$(κ): f=$(f_val), σ=$(σs[i]), ρ=$(ρs[i])")
	end

	# ── Stage 2: quadratic interpolation to best κ ──
	κ_interp = quadratic_minimum_3best(κs, fs)

	# Pick the best grid point as the source for σ, ρ
	ibest = argmin(fs)
	σ_start = σs[ibest]
	ρ_start = ρs[ibest]

	println("\n=== Interpolated κ = $(κ_interp) ===")
	println("Best grid point: κ=$(κs[ibest]), σ=$(σ_start), ρ=$(ρ_start), f=$(fs[ibest])")

	# ── Stage 3: single 3D Nelder-Mead from interpolated start ──
	model_3d_start = build_params(base, (κ = κ_interp, σ = σ_start, ρ = ρ_start))

	lower3d = [lower2d[1], lower2d[2], kappa_limits[1]]
	upper3d = [upper2d[1], upper2d[2], kappa_limits[2]]

	println("\n=== Final 3D calibration ===")

	cal_final, f_final = calibrateCliquets(
		model_3d_start,
		(:σ, :ρ, :κ),
		lower3d,
		upper3d,
		refValues;
		abstol = abstol / 100,      # tighter tolerance for final polish
		config = config,
		n_schedule = [n, n*4],
		spot = spot,
		payoff = payoff,
		polish = false,
		lowerGuess = interiorize(lower3d, lower3d, upper3d),
		upperGuess = interiorize(upper3d, lower3d, upper3d),
	)

	return cal_final, f_final, κs, fs, σs, ρs, κ_interp
end
function quadratic_minimum_log(κs, fs)
	log_fs = log.(fs)
	A = hcat(κs .^ 2, κs, ones(length(κs)))
	abc = A \ log_fs
	a, b, c = abc

	if a > 0
		κ_min = -b / (2a)
		# Clamp to the convex hull of the data
		return clamp(κ_min, minimum(κs), maximum(κs))
	else
		return κs[argmin(fs)]
	end
end

"""
	log_quadratic_minimum_3best(κs, fs; clamp_to_hull=true)

Select the 3 points with lowest loss, fit a quadratic in log-loss space,
and return the analytic minimum.
"""
function log_quadratic_minimum_3best(κs, fs; clamp_to_hull = true, size = min(3, length(fs)))
	# Select 3 best points by loss
	p = sortperm(fs)
	κ3 = κs[p[1:size]]
	f3 = fs[p[1:size]]

	# Fit log(loss) = a*κ^2 + b*κ + c
	A = hcat(κ3 .^ 2, κ3, ones(size))
	log_f3 = log.(f3)
	abc = A \ log_f3
	a, b, c = abc

	# Analytic minimum of the parabola
	if a > 0
		κ_min = -b / (2a)
	else
		# Not convex: fall back to best grid point
		return κ3[argmin(f3)]
	end

	if clamp_to_hull
		# Only trust interpolation within the convex hull of the 3 points
		κ_min = clamp(κ_min, minimum(κ3), maximum(κ3))
	end

	return κ_min
end




# ── Main progressive DE calibration ──
function calibrateCliquetsProgressiveDE(
	base::HestonModel,
	paramNames::NTuple{N, Symbol},
	lower,
	upper,
	refValues;
	n_schedule = [256, 512, 1024, 4096],
	pop_schedule = [15, 15, 15, 10],
	elite_fraction = 0.75,
	n_generations = [300, 200, 100, 50],
	stagnation = [30, 20, 20, 20],
	strategy = Best1Bin(0.9, 0.5),
	rng = Chacha8SIMD(UInt32),
	abstol = 1e-8,
	config::GridConfig = GridConfig(base, specTimes, 1.0/52, DVSS2X(withBB = false), mindt = 0.5/365, tstop = 1.0),
	spot = 1.0,
	payoff = ListPayoff([CliquetOption(
		cap, -1.0, 1e100, 0.0, 0.0,
		PeriodicCashFlow(
			collect(range(1.0/12, 1.0, length = 12)),
			1.0, 1, 0.0),
		0.0) for cap in [0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.03, 0.035, 0.04]]),
        seed = 42
) where N

	ndim = length(paramNames)
	ndim_mc = ndims(base, config)

	function make_objective(n)
		function objective(x)
			if any(x .< lower) || any(x .> upper)
				return floatmax(typeof(x[1]))
			end
			overrides = NamedTuple{paramNames}(x)
			model = build_params(base, overrides)
			vals, _ = simulate_outputs(model, spot, payoff, n, config, ndim_mc, seed=seed)
			return sum(abs2, vals .- refValues)
		end
		return objective
	end

	x0 = [getproperty(base, name) for name in paramNames]

	# ── Stage 1: cheap exploration ──
	println("=== Stage 1: n=$(n_schedule[1]) ===")
	obj1 = make_objective(n_schedule[1])
	prob1 = Problem(ndim, obj1, collect(lower), collect(upper))

	de1 = makeDifferentialEvolutionOptimizer(
		OptimizerParams(pop_schedule[1]),
		prob1,
		rng,
		strategy,
		x0 = x0,
	)

	_, stat1 = optimize(de1, TerminationCriteria(
		n_generations[1],
		stagnation[1],
		abstol,
		abstol / 100,
	))

	println("  best = $(GlobalOptimization.minimum(de1)), evals = $(GlobalOptimization.numberOfEvaluations(de1))")

	pop, costs = extract_population(de1)
	elite_count = max(1, round(Int, elite_fraction * pop_schedule[1]))
	elite_idx = sortperm(costs)[1:elite_count]
	elite_pop = pop[elite_idx, :]
	elite_costs = costs[elite_idx]

	best = GlobalOptimization.minimizer(de1)

	# ── Stages 2+: warm-start with elite population ──
	for stage in 2:length(n_schedule)
		n = n_schedule[stage]
		np = pop_schedule[stage]
		n_elite = min(elite_count, np)

		println("=== Stage $stage: n=$n ===")

		obj = make_objective(n)
		prob = Problem(ndim, obj, collect(lower), collect(upper))

		de = makeDifferentialEvolutionOptimizerWithElite(
			OptimizerParams(np),
			prob,
			rng,
			strategy,
			elite_pop,
			elite_costs;
			reevaluate = true,  # objective changed due to different n
		)

		_, stat = optimize(de, TerminationCriteria(
			n_generations[stage],
			stagnation[stage],
			abstol,
			abstol / 100,
		))


		# Prepare elite for next stage
		pop, costs = extract_population(de)
		elite_count = max(1, round(Int, elite_fraction * np))
		elite_idx = sortperm(costs)[1:elite_count]
		elite_pop = pop[elite_idx, :]
		elite_costs = costs[elite_idx]

		best = GlobalOptimization.minimizer(de)
		println("  best = $(GlobalOptimization.minimum(de)), evals = $(GlobalOptimization.numberOfEvaluations(de)), minimizer = $(best)")
	end
	# ── Final polish: Nelder-Mead at highest n ──
	println("=== Final polish: Nelder-Mead (n=$(n_schedule[end])) ===")
	transforms = map((lo, hi) -> TanhTransformation(lo, hi), lower, upper)
	rx0 = map((t, xi) -> inv(t, xi), transforms, interiorize(best, lower, upper; margin_frac=0.0001))
	ndim = ndims(base, config)


	function make_objective_constrained(n)
		function objective(xr)
			x = map((t, xri) -> t(xri), transforms, xr)
			overrides = NamedTuple{paramNames}(x)
			model = build_params(base, overrides)
			vals, _ = simulate_outputs(model, spot, payoff, n, config, ndim, seed=seed)
			return sum(abs2, vals .- refValues)
		end
		return objective
	end
	obj_final = make_objective_constrained(n_schedule[end])

	nm_res = Optim.optimize(
		obj_final,
		rx0,
		NelderMead(),
		Optim.Options(f_abstol = abstol/100, iterations = 100),
	)

	rx_final = Optim.minimizer(nm_res)
	x_final = map((t, xi) -> t(xi), transforms, rx_final)
	println("  NM best = $(Optim.minimum(nm_res))")
	println("  Final params: ", NamedTuple{paramNames}(x_final))

	model_final = build_params(base, NamedTuple{paramNames}(x_final))
	return model_final, Optim.minimum(nm_res)
end
