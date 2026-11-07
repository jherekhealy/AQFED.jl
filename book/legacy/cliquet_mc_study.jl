using AQFED: AQFED
using Test
using PPInterpolation
using CharFuncPricing
using Statistics
using Printf
using LaTeXStrings
using DataFrames
using Optim
using Statistics
using LinearAlgebra
import AQFED.Random
import AQFED.Random:
	AbstractRNGSeq,
	ZRNGSeq,
	OriginalScramblingRng,
	ScramblingRngAdapter,
	ScrambledSobolSeq,
	Owen, FaureTezuka, NoScrambling, OwenFaureTezuka

using Random123: Random123, Philox4x
import RandomNumbers: AbstractRNG
import Random: MersenneTwister, rand!
using Statistics
import AQFED.TermStructure:
	SVISection,
	VarianceSurfaceBySection,
	varianceByLogmoneyness,
	HestonModel,
	DoubleHestonModel,
	LocalVolatilityModel,
	ConstantBlackModel,
	TSBlackModel,
	ConstantRateCurve,
	InterpolatedLogDiscountFactorCurve

include("market_data.jl")
	
function estimateError(values::Vector{Float64}, k::Int, rng)
	n = length(values)
	mv = Vector{Float64}(undef, k)
	#stderr =  stdm(values, meanv) / sqrt(length(values))
	for i ∈ 1:k
		indices = (rand(rng, UInt32, 100) .% n) .+ 1
		v = values[indices]
		mv[i] = mean(v)
	end
	stderr = stdm(mv, mean(values)) / sqrt(length(values) / 100)
	return stderr
end

#the simulate*Iter functions return a vector of MCConvergenceResult (one per payoff): (mean, stderr) of the last sample size
mcStat(res::Tuple) = res
mcStat(res::AQFED.MonteCarlo.MCConvergenceResult) = AQFED.MonteCarlo.last_stat(res)
mcStat(res::AbstractVector) = length(res) == 1 ? mcStat(only(res)) : AQFED.MonteCarlo.last_stat(res)




struct RQMCRow
	method::String
	timestep::Int
	price::Float64
	error::Float64
	stderr::Float64
	pathstderr::Float64
	mean_time::Float64
	total_time::Float64
end

function dvss_runner(seq, model, spot, payoff, start, n, config)
	return mcStat(AQFED.MonteCarlo.simulateDVSS2XIter(
		seq, model, spot, payoff, start, n, config;
	))
end

function qe_runner(seq, model, spot, payoff, start, n, config)
	return mcStat(AQFED.MonteCarlo.simulateQEIter(
		seq, model, spot, payoff, start, n, config;
	))
end

function one_rqmc_run(simfun, ndim, n, model, spot, payoff, start, config, seed::UInt64)
	rng = Philox4x((UInt32(seed % UInt32), UInt32(seed >> 32)))
	seq = ScrambledSobolSeq(ndim, n, OwenFaureTezuka(31, ScramblingRngAdapter(rng)))
	t = @elapsed value, pathstderr = simfun(seq, model, spot, payoff, start, n, config)
	return value, pathstderr, t
end

function run_rqmc_method(
	method_name,
	simfun,
	timesteps;
	model,
	specTimes,
	spot,
	payoff,
	start,
	n,
	configFactory,
	mrep = 128,
	ref_price = nothing,
	master_seed = 0x12345678abcdef01,
)
	rows = RQMCRow[]
	master_rng = Philox4x((UInt32(master_seed % UInt32), UInt32(master_seed >> 32)))

	for timestep in timesteps
		dt = 1.0 / timestep
		config = configFactory(dt)
		ndim = AQFED.MonteCarlo.ndims(model, config)

		prices = Vector{Float64}(undef, mrep)
		pathstderrs = Vector{Float64}(undef, mrep)
		times = Vector{Float64}(undef, mrep)

		for j in 1:mrep
			seed = rand(master_rng, UInt64)
			price, pathstderr, elapsed = one_rqmc_run(
				simfun, ndim, n, model, spot, payoff, start, config, seed,
			)
			prices[j] = price
			pathstderrs[j] = pathstderr
			times[j] = elapsed
		end

		mean_price = mean(prices)
		rqmc_stderr = std(prices; corrected = true) / sqrt(mrep)
		mean_pathstderr = mean(pathstderrs)
		err = isnothing(ref_price) ? NaN : abs(mean_price - ref_price)
		mean_time = mean(times)
		total_time = sum(times)

		push!(rows, RQMCRow(
			method_name,
			timestep,
			mean_price,
			err,
			rqmc_stderr,
			mean_pathstderr,
			mean_time,
			total_time,
		))
	end

	return rows
end
using Base.Threads

function one_rqmc_run_threaded(simfun, ndim, n, model, spot, payoff, start, config, seed::UInt64)
	rng = Philox4x((UInt32(seed % UInt32), UInt32(seed >> 32)))
	seq = ScrambledSobolSeq(ndim, n, Owen(31, ScramblingRngAdapter(rng)))
	local_payoff = deepcopy(payoff)   # safest choice
	t = @elapsed value, pathstderr = simfun(seq, model, spot, local_payoff, start, n, config)
	return value, pathstderr, t
end

function run_rqmc_method_threaded(
	method_name,
	simfun,
	timesteps;
	model,
	specTimes,
	spot,
	payoff,
	start,
	n,
	configFactory,
	mrep = 128,
	ref_price = nothing,
	master_seed = 0x12345678abcdef01,
)
	rows = RQMCRow[]
	master_rng = Philox4x((UInt32(master_seed % UInt32), UInt32(master_seed >> 32)))

	for timestep in timesteps
		dt = 1.0 / timestep
		config = configFactory(dt)
		ndim = AQFED.MonteCarlo.ndims(model, config)

		prices = Vector{Float64}(undef, mrep)
		pathstderrs = Vector{Float64}(undef, mrep)
		times = Vector{Float64}(undef, mrep)
		seeds = rand(master_rng, UInt64, mrep)

		@threads for j in 1:mrep
			price, pathstderr, elapsed = one_rqmc_run_threaded(
				simfun, ndim, n, model, spot, payoff, start, config, seeds[j],
			)
			prices[j] = price
			pathstderrs[j] = pathstderr
			times[j] = elapsed
		end

		mean_price = mean(prices)
		rqmc_stderr = std(prices; corrected = true) / sqrt(mrep)
		mean_pathstderr = mean(pathstderrs)
		err = isnothing(ref_price) ? NaN : abs(mean_price - ref_price)
		mean_time = mean(times)
		total_time = sum(times)

		push!(rows, RQMCRow(
			method_name,
			timestep,
			mean_price,
			err,
			rqmc_stderr,
			mean_pathstderr,
			mean_time,
			total_time,
		))
	end

	return rows
end
function latex_table(rows; caption = "RQMC results", label = "tab:rqmc_results")
	io = IOBuffer()

	println(io, raw"\begin{table}[htbp]")
	println(io, raw"\centering")
	println(io, raw"\small")
	println(io, raw"\setlength{\tabcolsep}{4pt}")
	println(io, raw"\begin{tabular}{lrrrrrr}")
	println(io, raw"\toprule")
	println(io, raw"Method & Steps/year & Price & Error & RQMC s.e. & Path s.e. & Time (s) \\")
	println(io, raw"\midrule")

	for r in rows
		err_str = isnan(r.error) ? "--" : @sprintf("%.6e", r.error)
		println(
			io,
			@sprintf(
				"%s & %d & %.10f & %s & %.6e & %.6e & %.4f \\\\",
				r.method,
				r.timestep,
				r.price,
				err_str,
				r.stderr,
				r.pathstderr,
				r.mean_time,
			)
		)
	end

	println(io, raw"\bottomrule")
	println(io, raw"\end{tabular}")
	println(io, "\\caption{$caption}")
	println(io, "\\label{$label}")
	println(io, raw"\end{table}")

	return String(take!(io))
end
function latex_table_siunitx(rows; caption = "RQMC results", label = "tab:rqmc_results", ref_price = nothing)
	io = IOBuffer()

	println(io, raw"\begin{table}[htbp]")
	println(io, raw"\centering")
	println(io, raw"\small")
	println(io, "\\caption{$caption}")
	println(io, "\\label{$label}")

	println(io, raw"\begin{tabular}{l S[table-format=3.0] S[table-format=1.4] S[table-format=3.2] S[table-format=1.2e2] S[table-format=1.2e2] S[table-format=1.2e2] S[table-format=2.2]}")
	println(io, raw"\toprule")

	# FIX: Every header in an S column must be wrapped in an extra set of {} 
	# e.g., {{\textbf{Price}}} instead of {\textbf{Price}}
	println(io, raw"\textbf{Method} & {{\textbf{Steps/yr}}} & {{\textbf{Price}}} & {{\textbf{Rel. Err (\%)}}}  & {{\textbf{Abs. Error}}}  & {{\textbf{RQMC s.e.}}} & {{\textbf{Path s.e.}}} & {{\textbf{Time (s)}}} \\\\")
	println(io, raw"\midrule")

	methods = unique([r.method for r in rows])

	for (i, m) in enumerate(methods)
		m_rows = filter(r -> r.method == m, rows)
		n = length(m_rows)

		for (j, r) in enumerate(m_rows)
			# FIX: \multirow in an S column must also be protected with {}
			method_str = (j == 1) ? "{$m}" : "{}"

			if isnan(r.error)
				abs_err_str = "{--}"
				rel_err_str = "{--}"
			elseif isnothing(ref_price)
				abs_err_str = @sprintf("%.2e", r.error)
				rel_err_str = "{--}"
			else
				abs_err_str = @sprintf("%.2e", r.error)
				pct_err = (r.error / ref_price) * 100
				rel_err_str = @sprintf("%.2f", pct_err)
			end

			println(
				io,
				@sprintf(
					"%-22s & %4d & %.4f & %s & %s & %.2e & %.2e & %8.2f \\\\",
					method_str,
					r.timestep,
					r.price*100,
					rel_err_str,
					abs_err_str,
					r.stderr,
					r.pathstderr,
					r.mean_time
				)
			)
		end

		if i < length(methods)
			println(io, raw"\midrule")
		end
	end

	println(io, raw"\bottomrule")
	println(io, raw"\end{tabular}")
	println(io, raw"\end{table}")

	return String(take!(io))
end

#=
using CSV
using DataFrames
using Dates

function makeMarketDataFromFile(path::AbstractString)
    df = DataFrame(CSV.File(path))

    rename!(df, Dict(
        "ValuationDate" => :valuation_date,
        "MarketType" => :market_type,
        "Underlying" => :underlying,
        "Stock Name" => :stock_name,
        "Volatility Configuration" => :vol_config,
        "Spot Price" => :spot,
        "Expiration Date" => :expiry_date,
        "Relative Strike" => :relative_strike,
        "Strike Price" => :strike_price,
        "Volatility" => :volatility,
        "Discount Factor" => :discount_factor,
        "Depth" => :depth,
        "Forward Price" => :forward_price,
        "Dividend Yield" => :dividend_yield
    ))

    df.valuation_date = Date.(df.valuation_date, dateformat"yyyy/mm/dd")
    df.expiry_date    = Date.(df.expiry_date,    dateformat"yyyy/mm/dd")

    valdate = only(unique(df.valuation_date))
    spot    = only(unique(df.spot))

    expiry_order = sort(unique(df.expiry_date))
    strike_order = sort(unique(df.relative_strike))

    nT = length(expiry_order)
    nK = length(strike_order)

    ts = zeros(Float64, nT)
    dfs = zeros(Float64, nT)
    dividendYields = zeros(Float64, nT)
    uForwards = zeros(Float64, nT)
    vols = Matrix{Float64}(undef, nT, nK)

    expiry_index = Dict(d => i for (i, d) in enumerate(expiry_order))
    strike_index = Dict(k => j for (j, k) in enumerate(strike_order))

    for subdf in groupby(df, :expiry_date)
        i = expiry_index[subdf.expiry_date[1]]

        ts[i] = Dates.value(subdf.expiry_date[1] - valdate) / 365.0
        dfs[i] = subdf.discount_factor[1]
        dividendYields[i] = subdf.dividend_yield[1]
        uForwards[i] = subdf.forward_price[1] / spot

        for row in eachrow(subdf)
            j = strike_index[row.relative_strike]
            vols[i, j] = row.volatility
        end
    end

    strikesV = collect(strike_order) ./ 100.0
    strikes = ones(nT, 1) * permutedims(strikesV)

    return spot, ts, dfs, dividendYields, uForwards, strikes, vols
end
=#

# Helper to add nsteps column for a given scheme + ramp settings
function add_nsteps!(
    df::DataFrame,
    model,
    specTimes::Vector{Float64},
    method;        # e.g. AQFED.MonteCarlo.DVSS2X(withBB=false)
    mindt::Union{Nothing,Float64}=nothing,
    tstop::Union{Nothing,Float64}=nothing,
)
    nsteps = similar(df.timestep, Int)
    for (i, n) in enumerate(df.timestep)
		dt = 1.0 / n
        cfg = isnothing(mindt) && isnothing(tstop) ?
            AQFED.MonteCarlo.GridConfig(model, specTimes, dt, method) :
            AQFED.MonteCarlo.GridConfig(
                model, specTimes, dt, method;
                mindt = mindt,
                tstop = tstop,
            )
        nsteps[i] = length(cfg.genTimes) - 1   # one step per interval
    end
    df.nsteps = nsteps
    return df
end

function makeDataFrameFromRows(rows_qe)
	rows_qe_df = DataFrame(
    method      = getfield.(rows_qe, :method),
    timestep    = getfield.(rows_qe, :timestep),
    price       = getfield.(rows_qe, :price),
    error       = getfield.(rows_qe, :error),
    pathstderr  = getfield.(rows_qe, :pathstderr),
    stderr      = getfield.(rows_qe, :stderr),
    mean_time   = getfield.(rows_qe, :mean_time),
    total_time  = getfield.(rows_qe, :total_time),
)	
return rows_qe_df
end

@testset "HestonCliquetSim" begin
	spot, ts, dfs, uForwards, strikes, vols = makeMarketData20260512()
	uPrices, isCall, uWeights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ts, uForwards, strikes, vols, vegaFloor = 1e-2)

	params, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, ones(size(vols)))

	logdfInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, log.(dfs)))
	logForwardInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, -log.(uForwards)))
	discountCurve = InterpolatedLogDiscountFactorCurve(logdfInterp)
	driftCurve = InterpolatedLogDiscountFactorCurve(logForwardInterp)
	hParams = params

	hParams = HestonParams(0.0399, 4.435, 0.0691, -0.7114, 1.947) #val=2021/03/15, sp500 Feller strongly violated
	hParams = HestonParams(0.0550, 5.0, 0.0527, -0.688, 3.237)
	lag = 3.0/12
	payoff = AQFED.MonteCarlo.CliquetOption(
		0.098, 0.0, 1e8, 0.0, 0.0547*5,
		AQFED.MonteCarlo.PeriodicCashFlow([lag, 1.0+lag, 2.0+lag, 3.0+lag, 4.0+lag, 5.0+lag], 5.0+lag, 1, 0.0),
		0.0)

	specTimes = AQFED.MonteCarlo.specificTimes(payoff)
	model = HestonModel(hParams, driftCurve, discountCurve)
	start = 0
	dvss_method = AQFED.MonteCarlo.DVSS2X(withBB = false)
	qe_method   = AQFED.MonteCarlo.QE(withBB = false, withMartingale = true)

	config = AQFED.MonteCarlo.GridConfig(model, specTimes, 1.0 / 365, dvss_method, mindt=0.5/365, tstop=0.0)
	ndim = AQFED.MonteCarlo.ndims(model, config)
	rng = AQFED.Random.Well1024a()
	#seq = AbstractRNGSeq(rng, ndim)
	seq = ScrambledSobolSeq(ndim, 1024*1024, OwenFaureTezuka(31, ScramblingRngAdapter(rng)))
	ref_price, stderror =
		mcStat(AQFED.MonteCarlo.simulateDVSS2XIter(seq, model, spot, payoff, start, 1024*1024, config))
	println("ref = ", ref_price)


	n = 16*1024

	timesteps = [4, 8, 12, 16, 24, 32, 52, 64, 96, 128]
	mrep = 128
	rows_dvss_uni = run_rqmc_method_threaded(
		"DVSS2X",
		dvss_runner,
		timesteps;
		model = model,
		specTimes = specTimes,
		spot = spot,
		payoff = payoff,
		start = start,
		n = n,
		configFactory = dt -> AQFED.MonteCarlo.GridConfig(model, specTimes, dt, dvss_method, mindt=0.5/365, tstop=0.0),
		mrep = mrep,
		ref_price = ref_price,
		master_seed = 0x1111111111111111,
	)

	rows_qe_uni = run_rqmc_method_threaded(
		"QE-M",
		qe_runner,
		timesteps;
		model = model,
		specTimes = specTimes,
		spot = spot,
		payoff = payoff,
		start = start,
		n = n,
		configFactory = dt -> AQFED.MonteCarlo.GridConfig(model, specTimes, dt, qe_method, mindt=0.5/365, tstop=0.0),
		mrep = mrep,
		ref_price = ref_price,
		master_seed = 0x2222222222222222,
	)

	rows_dvss_ramp = run_rqmc_method_threaded(
		"DVSS2X",
		dvss_runner,
		timesteps;
		model = model,
		specTimes = specTimes,
		spot = spot,
		payoff = payoff,
		start = start,
		n = n,
		configFactory = dt -> AQFED.MonteCarlo.GridConfig(model, specTimes, dt, dvss_method, mindt=0.5/365, tstop=1.0),
		mrep = mrep,
		ref_price = ref_price,
		master_seed = 0x1111111111111111,
	)

	rows_qe_ramp = run_rqmc_method_threaded(
		"QE-M",
		qe_runner,
		timesteps;
		model = model,
		specTimes = specTimes,
		spot = spot,
		payoff = payoff,
		start = start,
		n = n,
		configFactory = dt -> AQFED.MonteCarlo.GridConfig(model, specTimes, dt, qe_method, mindt=0.5/365, tstop=1.0),
		mrep = mrep,
		ref_price = ref_price,
		master_seed = 0x2222222222222222,
	)



	rows = vcat(rows_dvss_uni, rows_qe_uni)

	for r in rows
		@printf(
			"%-6s %4d  price=% .10f  error=% .6e  rqmc_se=% .6e  path_se=% .6e  time=% .4f\n",
			r.method, r.timestep, r.price, r.error, r.stderr, r.pathstderr, r.mean_time
		)
	end

	tex = latex_table_siunitx(
		rows;
		caption = "Randomized QMC prices for the cliquet under Heston.",
		label = "tbl:cliquet_rqmc",
		ref_price = ref_price,
	)

	println(tex)

	rows_qe_uni_df = makeDataFrameFromRows(rows_qe_uni)	
	add_nsteps!(rows_qe_uni_df, model, specTimes, qe_method;
               mindt = 0.5/365, tstop = 0.0)
	rows_qe_ramp_df = makeDataFrameFromRows(rows_qe_ramp)	
	add_nsteps!(rows_qe_ramp_df, model, specTimes, qe_method;
               mindt = 0.5/365, tstop = 1.0)
rows_dvss2_uni_df = makeDataFrameFromRows(rows_dvss_uni)	
	add_nsteps!(rows_dvss2_uni_df, model, specTimes, qe_method;
               mindt = 0.5/365, tstop = 0.0)
	rows_dvss2_ramp_df = makeDataFrameFromRows(rows_dvss_ramp)	
	add_nsteps!(rows_dvss2_ramp_df, model, specTimes, qe_method;
               mindt = 0.5/365, tstop = 1.0)

end
#=
using AlgebraOfGraphics, CairoMakie

ows_qe_uni_df.scheme .= "QE"
rows_qe_uni_df.grid   .= "almost uniform"

rows_qe_ramp_df.scheme    .= "QE"
rows_qe_ramp_df.grid      .= "ramp to 1Y"

rows_dvss2_uni_df.scheme .= "DVSS2X"
rows_dvss2_uni_df.grid   .= "almost uniform"

rows_dvss2_ramp_df.scheme    .= "DVSS2X"
rows_dvss2_ramp_df.grid      .= "ramp to 1Y"
df_all = vcat(rows_dvss2_uni_df, rows_dvss2_ramp_df,
              rows_qe_uni_df, rows_qe_ramp_df)

set_aog_theme!()


lines_layer =
    data(df_all) *
    mapping(
        :nsteps => "number of steps",
        :error  => "absolute error";
        color   = :scheme,
        group   = (:scheme, :grid),
    ) *
    visual(Lines)

scatter_layer =
    data(df_all) *
    mapping(
        :nsteps => "number of steps",
        :error  => "absolute error";
        color   = :scheme,
        marker  = :grid,                 # only here; Lines doesn't see it
        group   = (:scheme, :grid),
    ) *
    visual(Scatter)

xs = sort(unique(df_all.nsteps))[1:2:end]

fig = draw(
    lines_layer + scatter_layer;
    axis = (
        xscale = log10,
        yscale = log10,
        xticks = (xs, string.(xs)),  # positions + integer labels
        xlabel = "number of steps",
        ylabel = "absolute error",
    ),
    figure = (size = (800, 300),),
)
 save("cliquet_5y_error_vs_nsteps_dvss_qe.pdf", fig)


 cfg_uniform = AQFED.MonteCarlo.GridConfig(model, specTimes, 1.0/6, dvss_method, mindt=0.5/365, tstop=0.0)
cfg_ramp = AQFED.MonteCarlo.GridConfig(model, specTimes, 1.0 / 6, dvss_method, mindt=0.5/365, tstop=1.0)
gen_uniform = cfg_uniform.genTimes
gen_ramp    = cfg_ramp.genTimes

t_uniform = gen_uniform[1:end-1]
dt_uniform = gen_uniform[2:end] .- gen_uniform[1:end-1]

t_ramp = gen_ramp[1:end-1]
dt_ramp = gen_ramp[2:end] .- gen_ramp[1:end-1]

fig = Figure(resolution = (800, 300))
ax  = Axis(fig[1, 1],
           xlabel = "time",
           ylabel = "local step size",
           title  = "Uniform grid vs linear ramp grid")

scatter!(ax, t_uniform, dt_uniform,  label = "uniform")

scatter!(ax, t_ramp, dt_ramp, color = :red, label = "linear ramp")

axislegend(ax, position = :rb)
fig
fig = Figure(resolution = (800, 200))
ax  = Axis(fig[1, 1],
           xlabel = "time",
           ylabel = "",
           title  = "Sampling points: uniform vs linear ramp",
           yticks = ([1.0, 0.0], ["uniform", "linear ramp"]),
           yminorticksvisible = false,
           ygridvisible = false)

# Plot grid points as horizontal dot-rows
scatter!(ax, gen_uniform, fill(1.0, length(gen_uniform)),
         markersize = 6)

scatter!(ax, gen_ramp, fill(0.0, length(gen_ramp)),
         markersize = 6)

fig
save("gridpoints_uniform_vs_ramp.pdf", fig)
=#


@testset "LVMonthlyCliquetSim" begin
	spot, ts, dfs, uForwards, strikes, vols = makeMarketData20260512()
	uPrices, isCall, uWeights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ts, uForwards, strikes, vols, vegaFloor = 1e-1)

	params, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, ones(size(vols)))

	logdfInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, log.(dfs)))
	logForwardInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, -log.(uForwards)))
	discountCurve = InterpolatedLogDiscountFactorCurve(logdfInterp)
	driftCurve = InterpolatedLogDiscountFactorCurve(logForwardInterp)

	surface = AQFED.VolatilityModels.SplineSurface(ts, uForwards, log.(strikes ./ uForwards), vols)
	lvModel = AQFED.TermStructure.LocalVolatilityModel(surface, driftCurve, discountCurve)


	volsH = zeros(size(vols))
	for i in axes(strikes, 1), j in axes(strikes, 2)

		pricer = CharFuncPricing.ALCharFuncPricer(CharFuncPricing.DefaultCharFunc(params))
		isCall = strikes[i, j] > uForwards[i]
		price = CharFuncPricing.priceEuropean(pricer, isCall, strikes[i, j], uForwards[i], ts[i], 1.0)
		volsH[i, j] = AQFED.Black.impliedVolatility.(isCall, price, uForwards[i], strikes[i, j], ts[i], 1.0)
	end
	surfaceH = AQFED.VolatilityModels.SplineSurface(ts, uForwards, log.(strikes ./ uForwards), volsH)
	lvModelH = AQFED.TermStructure.LocalVolatilityModel(surfaceH, driftCurve, discountCurve)
	#two lv models: 1) market 2) heston. Prices of cliquets are too low. no flexibility to adjust them, and it is inherent to hte model. This is true Even if we calibrate the localvolatility to the heston model implied volatilities instead of the market volatilities. This is particularly visible for the low cap values on this example. More importantly, the LV model is inadequate to match cliquet prices from counterparties, as it is already fully defined by the vanillas.

	caps = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 3.50, 4.0] ./ 100
	obsTimes = collect(range(0.0/12, 1.0, length = 13))
	#obsTimes = vcat(0.0001, obsTimes)

	#=
	hParams = HestonParams(0.18^2, 7.6276,0.25^2,  -0.887, 1.347) #1M cliquet candidate
	hParams = HestonParams(0.021904, 7.6276, 0.060661, -0.9298, 1.3368) #v0 and theta should not be so low! var swaps are higher. Correct fit may be t[1] exact, t[end]exact and then calibrate kappa only. interestingly while using t[end] is okish for theta, using t[1] directly for v0 without adj leads to a higher varswap heston price compared to vsTheo[1]. (but this is what seems to have been done by the csystem)
    hParams = HestonParams(0.021165, 6.4657, 0.060607, -0.9730, 1.1596)
	=#
	model = HestonModel(hParams, driftCurve, discountCurve)
	timestep = 52
    n = 64*1024
    start=0
	for cap in caps
		payoff = AQFED.MonteCarlo.CliquetOption(
			cap, -1.0, 1e100, 0.0, 0.0,
			AQFED.MonteCarlo.PeriodicCashFlow(obsTimes, obsTimes[end], 1, 0.0),
			0.0)
		specTimes = AQFED.MonteCarlo.specificTimes(payoff)

		config = AQFED.MonteCarlo.GridConfig(model, specTimes, 1.0 / timestep, AQFED.MonteCarlo.DVSS2X(withBB = false), mindt=0.5/365, tstop=0.75)
		ndim = AQFED.MonteCarlo.ndims(model, config)
		rng = AQFED.Random.Well1024a()
		#seq = AbstractRNGSeq(rng, ndim)
		seq = ScrambledSobolSeq(ndim, n, Owen(31, ScramblingRngAdapter(rng)))
		time = @elapsed value, stderror =
			mcStat(AQFED.MonteCarlo.simulateDVSS2XIter(seq, model, spot, payoff, start, n, config))
		println("Heston ",
			timestep,
			" ",
			value,
			" ",
			stderror,
			" ",
			time,
		)
		rng = AQFED.Random.Well1024a()
		nd = AQFED.MonteCarlo.ndims(lvModel, specTimes, 1.0 / timestep)

		#seq = AbstractRNGSeq(rng, ndim)
		seq = ScrambledSobolSeq(nd, n, Owen(31, ScramblingRngAdapter(rng)))
		time = @elapsed value, stderror =
			mcStat(AQFED.MonteCarlo.simulate(seq, lvModel, spot, payoff, start, n, 1.0 / timestep))
		println("LV ",
			timestep,
			" ",
			value,
			" ",
			stderror,
			" ",
			time,
		)
		seq = ScrambledSobolSeq(nd, n, Owen(31, ScramblingRngAdapter(rng)))
		time = @elapsed value, stderror =
			mcStat(AQFED.MonteCarlo.simulate(seq, lvModelH, spot, payoff, start, n, 1.0 / timestep))
		println("LV-H ",
			timestep,
			" ",
			value,
			" ",
			stderror,
			" ",
			time,
		)
	end
end


@testset "MonthlyCliquetCalibration" begin
	spot, ts, dfs, uForwards, strikes, vols = makeMarketData20260512()
	refValues = [0.86,1.27 ,1.73 ,2.22 ,2.73 ,3.24 ,3.74 ,4.67 ,5.49 ,6.18 ] ./ 100

	spot, ts, dfs, yields, uForwards, strikes, vols = makeMarketData20260505()
	refValues = [0.009399999999999999, 0.0137, 0.018500000000000003, 0.023700000000000002, 0.028900000000000002 ,0.0341, 0.039, 0.0481, 0.0559, 0.062400000000000004]

	spot, ts, dfs, yields, uForwards, strikes, vols = makeMarketData20260331()
	refValues = [0.80,1.10,1.49,1.93,2.40,2.89,3.39,4.38,5.31,6.06] ./100


	spot, ts, dfs, yields, uForwards, strikes, vols = makeMarketData20260602()
	refValues = [0.89,1.30,1.77,2.28,2.80,3.32,3.83,4.77,5.58,6.26] ./ 100
	
	uPrices, isCall, uWeights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ts, uForwards, strikes, vols, vegaFloor=1e-2, truncationDev=2.0)
params2, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, uWeights)
#params05, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, ones(size(uPrices)))
 volsF, strikesF = AQFED.VolatilityModels.filterVolsBySimpleDelta(ts, uForwards, strikes, vols, deltas=[0.1, 0.25, 0.5, 0.75,0.9])
 params05, rmse05 = AQFED.VolatilityModels.calibrateHestonFromVols(ts, uForwards, strikesF, volsF,vegaFloor=1e-2)

	vsTheo = [AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(false), t, log.(strikes[1,:] / uForwards[i]), vols[i, :] .^ 2, 1.0) for (i, t) ∈ enumerate(ts)]

	vsTheoFlat = [AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(false), t, log.(strikes[1,:] / uForwards[i]), vols[i, :] .^ 2, 1.0) for (i, t) ∈ enumerate(ts)]
	vsTheoLinear = [AQFED.VolatilityModels.priceVarianceSwap(AQFED.VolatilityModels.FukasawaVarianceSwapReplication(true), t, log.(strikes[1,:] / uForwards[i]), vols[i, :] .^ 2, 1.0) for (i, t) ∈ enumerate(ts)]

	reduction = AQFED.VolatilityModels.makeVarianceSwapReduction(ts, uForwards, strikes, vols, isLinearExtrapolation=true)
	reduction1 = AQFED.VolatilityModels.makeVarianceSwapReduction1D(ts, uForwards, strikes, vols, isLinearExtrapolation=true)
	reductionE =  AQFED.VolatilityModels.makeVarianceSwapReductionExplicit(ts, uForwards, strikes, vols, isLinearExtrapolation=true,tte=ts[end-2])
	reductionE =  AQFED.VolatilityModels.makeVarianceSwapReductionExplicit(ts, uForwards, strikes, vols, isLinearExtrapolation=true,tte=1.0)
	reductionE =  AQFED.VolatilityModels.makeVarianceSwapReductionExplicit(ts, uForwards, strikes, vols, isLinearExtrapolation=false,tte=1.0)

	logdfInterp = PPInterpolation.makeLinearPP(vcat(0.0, ts), vcat(0.0, log.(dfs)))
	logForwardInterp = PPInterpolation.makeLinearPP(vcat(0.0, ts), vcat(0.0, -log.(uForwards)))
	logdfInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, log.(dfs)))
	logForwardInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, -log.(uForwards)))
	discountCurve = InterpolatedLogDiscountFactorCurve(logdfInterp)
	driftCurve = InterpolatedLogDiscountFactorCurve(logForwardInterp)


	#=
tdense=range(0.0,ts[end],length=100)
using CairoMakie

fig = Figure(size = (860, 420))
ax = Axis(
    fig[1, 1],
    xlabel = "Expiry",
    ylabel = "Price in volatility %",
    title = "Variance swap prices: replication vs Heston fits",
)

scatter!(ax, ts, sqrt.(vsTheoFlat);
    color = :black,
    marker = :circle,
    markersize = 11,
    strokecolor = :white,
    strokewidth = 0.8,
    label = "Flat extrapolation",
)

scatter!(ax, ts, sqrt.(vsTheoLinear);
    color = :black,
    marker = :utriangle,
    markersize = 12,
    strokecolor = :white,
    strokewidth = 0.8,
    label = "Linear extrapolation",
)

for (label, red, style) in [
    ("3D LS",    reduction,  nothing),
    ("1D LS",    reduction1, :dash),
    ("Explicit", reductionE, :dot),
    ("Vanillas 0.5 dev",  params05,     :dashdot),
    ("Vanillas 2.0 dev",  params2,     :dashdot),
]
    y = sqrt.(10000 .* map(t ->
        AQFED.VolatilityModels.integratedVarianceExpectation(
            red.v0, red.κ, red.θ; dt = t
        ),
        tdense
    ))

    if isnothing(style)
        lines!(ax, tdense, y; linewidth = 2.5, label = label)
    else
        lines!(ax, tdense, y; linewidth = 2.5, linestyle = style, label = label)
    end
end

axislegend(ax; position = :rb, framevisible = false)
fig
save("heston_varswap_oct2024.pdf", fig)
	=#

### 
caps = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 3.50, 4.0] ./ 100
	obsTimes = collect(range(0.0/12, 1.0, length = 13))
	payoff = AQFED.MonteCarlo.ListPayoff([AQFED.MonteCarlo.CliquetOption(
			cap, -1.0, 1e100, 0.0, 0.0,
			AQFED.MonteCarlo.PeriodicCashFlow(obsTimes, obsTimes[end], 1, 0.0),
			0.0) for cap in caps])
	logdfInterp = PPInterpolation.makeLinearPP(vcat(0.0, ts), vcat(0.0, log.(dfs)))
	logForwardInterp = PPInterpolation.makeLinearPP(vcat(0.0, ts), vcat(0.0, -log.(uForwards)))
	logdfInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, log.(dfs)))
	logForwardInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, -log.(uForwards)))
	discountCurve = InterpolatedLogDiscountFactorCurve(logdfInterp)
	driftCurve = InterpolatedLogDiscountFactorCurve(logForwardInterp)
model = HestonModel(hParams, driftCurve, discountCurve)
	timestep = 64
    n = 16*1024
    start=0
	specTimes = AQFED.MonteCarlo.specificTimes(payoff)
	config = AQFED.MonteCarlo.GridConfig(model, specTimes, 1.0 / timestep, AQFED.MonteCarlo.QE(withBB = false), mindt=0.5/365, tstop=0.0)
		ndim = AQFED.MonteCarlo.ndims(model, config)
		rng = AQFED.Random.Well1024a()
		seq = ScrambledSobolSeq(ndim, n, Owen(31, ScramblingRngAdapter(rng)))
		values, stddevs = mcStat(AQFED.MonteCarlo.simulateIter(seq, model, spot, payoff, start, n, config))

		#=
using StatsBase
using LinearAlgebra
using CairoMakie

rhos   = range(-0.99, -0.5, length=101)
sigmas = range(0.5, 2.5, length=101)

#zoom shows waves in error due to MC.
rhos   = range(-1.0, -0.8, length=101)
sigmas = range(1.0, 1.5, length=101)

n = 256
config = AQFED.MonteCarlo.GridConfig(model, specTimes, 1.0 / 64, AQFED.MonteCarlo.QE(true, true), mindt=0.5/365, tstop=0.0)

Z = Matrix{Float64}(undef, length(sigmas), length(rhos))
    for (j, ρ) in enumerate(rhos)
for (i, σ) in enumerate(sigmas)
        params0 = HestonParams(reductionE.v0, 7.0, reductionE.θ, ρ, σ)
        model = HestonModel(params0, driftCurve, discountCurve)
		ndim = AQFED.MonteCarlo.ndims(model, config)
		rng = AQFED.Random.Chacha8SIMD(UInt32)
	Random.seed!(rng, 42)
		seq = ScrambledSobolSeq(ndim, n, Owen(31, ScramblingRngAdapter(rng)))	
        values, stddevs = mcStat(AQFED.MonteCarlo.simulateIterParallel(seq, model, spot, payoff, start, n, config))

        Z[j, i] = rmsd(values, refValues)
    end
end

fig = Figure(size = (1200, 500))
logrmse = log10.(Z)
ax1 = Axis3(fig[1, 1], xlabel = "rho", ylabel = "sigma", zlabel = "log10(RMSE)")
surface!(ax1, rhos, sigmas, logrmse; colormap = :viridis)

ax2 = Axis(fig[1, 2], xlabel = "rho", ylabel = "sigma")
hm = heatmap!(ax2, rhos, sigmas, logrmse; colormap = :viridis)
Colorbar(fig[1, 3], hm)

fig
		=#

		#= plot all schemes accuracy
using CairoMakie
using LaTeXStrings
using AQFED.MonteCarlo
using AQFED.Random
using AQFED.TermStructure

CairoMakie.activate!(type = "pdf")

RHO_RANGE   = range(-1.0, -0.8, length=101)
SIGMA_RANGE = range(1.0, 1.5, length=101)

function make_seq(ndim::Int, n::Int)
    rng = AQFED.Random.Chacha8SIMD(UInt32)
    return ScrambledSobolSeq(ndim, n, Owen(31, ScramblingRngAdapter(rng)))
end

function build_model(v0, κ, θ, ρ, σ, driftCurve, discountCurve)
    params = HestonParams(v0, κ, θ, ρ, σ)
    return HestonModel(params, driftCurve, discountCurve)
end

function eval_surface!(
    Z::AbstractMatrix,
    method_builder,
    n::Int,
    rhos,
    sigmas,
    base_v0,
    base_theta,
    driftCurve,
    discountCurve,
    spot,
    payoff,
    specTimes,
    start,
    refValues;
    κ = 7.0,
)
    for (j, ρ) in enumerate(rhos)
        for (i, σ) in enumerate(sigmas)
            model = build_model(base_v0, κ, base_theta, ρ, σ, driftCurve, discountCurve)
            config = method_builder(model, specTimes)
            ndim = AQFED.MonteCarlo.ndims(model, config)
            seq = make_seq(ndim, n)
            values, stddevs = mcStat(AQFED.MonteCarlo.simulateIterParallel(
                seq, model, spot, payoff, start, n, config
            ))
            Z[j, i] = rmsd(values, refValues)
        end
    end
    return Z
end

function panel_figure(
    rhos,
    sigmas,
    Z;
    title3d = "",
    size = (1200, 500),
)
    logZ = log10.(Z)

    fig = Figure(size = size)

    ax1 = Axis3(
        fig[1, 1],
        xlabel = "rho",
        ylabel = "sigma",
        zlabel = "log10(RMSE)",
        title = title3d,
    )
    s = surface!(ax1, rhos, sigmas, logZ; colormap = :viridis)
    s.rasterize = 4   # try 2, 3, or 4

    ax2 = Axis(fig[1, 2], xlabel = "rho", ylabel = "sigma")
    hm = heatmap!(ax2, rhos, sigmas, logZ; colormap = :viridis)
    hm.rasterize = 3  # optional, but usually helps a lot too

    Colorbar(fig[1, 3], hm)

    return fig, logZ
end

function save_surface_heatmap_pdf(
    filename,
    rhos,
    sigmas,
    Z;
    kwargs...
)
    fig, _ = panel_figure(rhos, sigmas, Z; kwargs...)
    save(filename, fig, pdf_version="1.4")
    return fig
end

function common_colorrange(mats...)
    vals = vcat([vec(log10.(permutedims(M))) for M in mats]...)
    return (minimum(vals), maximum(vals))
end

# --- scheme-specific config builders ---

qe_config_builder(; dt = 1/64, mindt = 0.5/365, tstop = 0.0, maxdt = dt, withBB = true, withMartingale = true) =
    (model, specTimes) -> AQFED.MonteCarlo.GridConfig(
        model, specTimes, dt, AQFED.MonteCarlo.QE(withBB=withBB, withMartingale=withMartingale),
        mindt=mindt, tstop=tstop, maxdt=maxdt
    )

dvss2x_config_builder(; dt = 1/64, mindt = 0.5/365, tstop = 0.0, maxdt = dt, withBB = true) =
    (model, specTimes) -> AQFED.MonteCarlo.GridConfig(
        model, specTimes, dt, AQFED.MonteCarlo.DVSS2X(withBB=withBB),
        mindt=mindt, tstop=tstop, maxdt=maxdt
    )

fulltrunc_config_builder(; dt = 1/256, mindt = 0.5/365, tstop = 0.0, maxdt = dt, withBB = true) =
    (model, specTimes) -> AQFED.MonteCarlo.GridConfig(
        model, specTimes, dt, AQFED.MonteCarlo.FullTruncation(withBB),
        mindt=mindt, tstop=tstop, maxdt=maxdt
    )

alfonsi2_config_builder(; dt = 1/64, mindt = 0.5/365, tstop = 0.0, maxdt = dt, withBB = true) =
    (model, specTimes) -> AQFED.MonteCarlo.GridConfig(
        model, specTimes, dt, AQFED.MonteCarlo.Alfonsi2(withBB=withBB),
        mindt=mindt, tstop=tstop, maxdt=maxdt
    )

function generate_all_scheme_figures(;
    rhos = RHO_RANGE,
    sigmas = SIGMA_RANGE,
    n_qe_small = 256,
    n_qe_large = 1024,
    n_dvss = 256,
    n_dvss_large = 1024,
    n_alf = 256,
    n_ft = 256,
    base_v0,
    base_theta,
    driftCurve,
    discountCurve,
    spot,
    payoff,
    specTimes,
    start,
    refValues,
    κ = 7.0,
)
    Z_qe_256 = Matrix{Float64}(undef, length(rhos), length(sigmas))
    Z_qe_1024 = Matrix{Float64}(undef, length(rhos), length(sigmas))
    Z_dvss_256 = Matrix{Float64}(undef, length(rhos), length(sigmas))
    Z_dvss_1024 = Matrix{Float64}(undef, length(rhos), length(sigmas))
    Z_alf_256 = Matrix{Float64}(undef, length(rhos), length(sigmas))
    Z_ft_256 = Matrix{Float64}(undef, length(rhos), length(sigmas))

    eval_surface!(Z_qe_256,  qe_config_builder(),        n_qe_small, rhos, sigmas, base_v0, base_theta, driftCurve, discountCurve, spot, payoff, specTimes, start, refValues; κ=κ)
    eval_surface!(Z_qe_1024, qe_config_builder(),        n_qe_large, rhos, sigmas, base_v0, base_theta, driftCurve, discountCurve, spot, payoff, specTimes, start, refValues; κ=κ)
    eval_surface!(Z_dvss_256, dvss2x_config_builder(),   n_dvss,     rhos, sigmas, base_v0, base_theta, driftCurve, discountCurve, spot, payoff, specTimes, start, refValues; κ=κ)
    eval_surface!(Z_dvss_1024,dvss2x_config_builder(),   n_dvss_large,rhos, sigmas, base_v0, base_theta, driftCurve, discountCurve, spot, payoff, specTimes, start, refValues; κ=κ)
    eval_surface!(Z_alf_256,  alfonsi2_config_builder(), n_alf,      rhos, sigmas, base_v0, base_theta, driftCurve, discountCurve, spot, payoff, specTimes, start, refValues; κ=κ)
    eval_surface!(Z_ft_256,   fulltrunc_config_builder(),n_ft,       rhos, sigmas, base_v0, base_theta, driftCurve, discountCurve, spot, payoff, specTimes, start, refValues; κ=κ)

    cr_all = common_colorrange(Z_qe_256, Z_qe_1024, Z_dvss_256, Z_dvss_1024, Z_alf_256, Z_ft_256)

    save_surface_heatmap_pdf(
        "qe_256_surface_heatmap.pdf", rhos, sigmas, Z_qe_256;
        title3d = "QE, n = 256",
              
    )

    save_surface_heatmap_pdf(
        "qe_1024_surface_heatmap.pdf", rhos, sigmas, Z_qe_1024;
        title3d = "QE, n = 1024",
         )

    save_surface_heatmap_pdf(
        "dvss2x_256_surface_heatmap.pdf", rhos, sigmas, Z_dvss_256;
        title3d = "DVSS2X, n = 256",
         
    )

    save_surface_heatmap_pdf(
        "dvss2x_1024_surface_heatmap.pdf", rhos, sigmas, Z_dvss_1024;
        title3d = "DVSS2X, n = 1024",
         
    )

    save_surface_heatmap_pdf(
        "alfonsi2_256_surface_heatmap.pdf", rhos, sigmas, Z_alf_256;
        title3d = "Alfonsi2, n = 256",
        
    )

    save_surface_heatmap_pdf(
        "fulltrunc_256_surface_heatmap.pdf", rhos, sigmas, Z_ft_256;
        title3d = "Full Truncation, n = 256",
        
    )

    return (
        Z_qe_256 = Z_qe_256,
        Z_qe_1024 = Z_qe_1024,
        Z_dvss_256 = Z_dvss_256,
        Z_dvss_1024 = Z_dvss_1024,
        Z_alf_256 = Z_alf_256,
        Z_ft_256 = Z_ft_256,
    )
end

generate_all_scheme_figures(base_v0=reductionE.v0, base_theta=reductionE.θ, driftCurve=driftCurve, discountCurve=discountCurve, spot=spot, payoff=payoff, specTimes=specTimes, start=0, refValues=refValues)

		=#

		#= kappa grid RMSE
	using Printf
	"""
		Compute RMSE profile versus kappa, with (σ, ρ) recalibrated by calibrateCliquets
		for each fixed kappa.

		Returns a named tuple with vectors:
			kappa, sigma, rho, rmse, model
		"""
		function qe_kappa_profile_calibrateCliquets(
			kappas;
			v0,
			θ,
			driftCurve,
			discountCurve,
			spot,
			payoff,
			refValues,
			config,
			n_schedule = [1024],
			sigma_bounds = [0.01, 5.0],
			rho_bounds   = [-0.99, 0.5],
			seed=42
		)
			K = Float64[]
			S = Float64[]
			R = Float64[]
			E = Float64[]
			M = Any[]

			for κ in kappas
				params0 = HestonParams(v0, κ, θ, rho_bounds[1], sigma_bounds[1])
				model0 = HestonModel(params0, driftCurve, discountCurve)

				modelc, rmse = AQFED.VolatilityModels.calibrateCliquets(
					model0,
					(:σ, :ρ),
					[sigma_bounds[1], rho_bounds[1]],
					[sigma_bounds[2], rho_bounds[2]],
					refValues;
					config = config,
					n_schedule = n_schedule,
					spot = spot,
					payoff = payoff,
					polish = false,
					useDE = false,
					useNM = true,
					seed=seed
				)

				push!(K, κ)
				push!(S, modelc.σ)
				push!(R, modelc.ρ)
				push!(E, rmse)
				push!(M, modelc)

				@printf("kappa=%8.4f  sigma*=%9.6f  rho*=%9.6f  rmse=%12.8e  n_schedule=%s\n",
						κ, modelc.σ, modelc.ρ, rmse, string(n_schedule))
			end

			return (
				kappa = K,
				sigma = S,
				rho = R,
				rmse = E,
				model = M,
			)
		end


		function qe_kappa_profile_calibrateCliquetsDE(
			kappas;
			v0,
			θ,
			driftCurve,
			discountCurve,
			spot,
			payoff,
			refValues,
			config,
			n_schedule = [1024],
			pop_schedule = [15],
			n_generations = [100],
			sigma_bounds = [0.01, 5.0],
			rho_bounds   = [-0.99, 0.5],
			seed=42
		)
			K = Float64[]
			S = Float64[]
			R = Float64[]
			E = Float64[]
			M = Any[]

			for κ in kappas
				params0 = HestonParams(v0, κ, θ, rho_bounds[1], sigma_bounds[1])
				model0 = HestonModel(params0, driftCurve, discountCurve)

				modelc, rmse = AQFED.VolatilityModels.calibrateCliquetsProgressiveDE(
					model0,
					(:σ, :ρ),
					[sigma_bounds[1], rho_bounds[1]],
					[sigma_bounds[2], rho_bounds[2]],
					refValues;
					config = config,
					n_schedule = n_schedule,
					pop_schedule = pop_schedule,
					n_generations = n_generations,
					spot = spot,
					payoff = payoff,
					seed=seed
				)

				push!(K, κ)
				push!(S, modelc.σ)
				push!(R, modelc.ρ)
				push!(E, rmse)
				push!(M, modelc)

				@printf("kappa=%8.4f  sigma*=%9.6f  rho*=%9.6f  rmse=%12.8e  n_schedule=%s\n",
						κ, modelc.σ, modelc.ρ, rmse, string(n_schedule))
			end

			return (
				kappa = K,
				sigma = S,
				rho = R,
				rmse = E,
				model = M,
			)
		end
	

	kappas = collect(range(1.0, 12.0, length=31))
		seed = 42
	profile_1024 = qe_kappa_profile_calibrateCliquets(
		kappas;
		v0 = reductionE.v0,
		θ = reductionE.θ,
		driftCurve = driftCurve,
		discountCurve = discountCurve,
		spot = spot,
		payoff = payoff,
		refValues = refValues,
		config = config,
		n_schedule = [1024],
		seed = seed
	)

	profile_4096 = qe_kappa_profile_calibrateCliquets(
		kappas;
		v0 = reductionE.v0,
		θ = reductionE.θ,
		driftCurve = driftCurve,
		discountCurve = discountCurve,
		spot = spot,
		payoff = payoff,
		refValues = refValues,
		config = config,
		n_schedule = [4096],
		seed = seed
	)

	profile_sched = qe_kappa_profile_calibrateCliquets(
		kappas;
		v0 = reductionE.v0,
		θ = reductionE.θ,
		driftCurve = driftCurve,
		discountCurve = discountCurve,
		spot = spot,
		payoff = payoff,
		refValues = refValues,
		config = config,
		n_schedule = [512, 4096],
		seed = seed
	)

profile_1024_de = qe_kappa_profile_calibrateCliquetsDE(
    kappas;
    v0 = reductionE.v0,
    θ = reductionE.θ,
    driftCurve = driftCurve,
    discountCurve = discountCurve,
    spot = spot,
    payoff = payoff,
    refValues = refValues,
    config = config,
    n_schedule = [1024],
	seed=  seed
)

profile_4096_de = qe_kappa_profile_calibrateCliquetsDE(
    kappas;
    v0 = reductionE.v0,
    θ = reductionE.θ,
    driftCurve = driftCurve,
    discountCurve = discountCurve,
    spot = spot,
    payoff = payoff,
    refValues = refValues,
    config = config,
    n_schedule = [4096],
	   pop_schedule = [10],
    	n_generations = [50],
		seed=seed
)
profile_sched_de = qe_kappa_profile_calibrateCliquetsDE(
           kappas;
           v0 = reductionE.v0,
           θ = reductionE.θ,
           driftCurve = driftCurve,
           discountCurve = discountCurve,
           spot = spot,
           payoff = payoff,
           refValues = refValues,
           config = config,
           n_schedule = [256,512, 4096],
                  pop_schedule = [25, 20, 5],
                   n_generations = [250, 100, 10],
                   
       )

	   function save_kappa_profile_csv(filename, profile)
    open(filename, "w") do io
        println(io, "kappa,sigma_opt,rho_opt,rmse")
        for i in eachindex(profile.kappa)
            @printf(io, "%.8f,%.8f,%.8f,%.12e\n",
                    profile.kappa[i],
                    profile.sigma[i],
                    profile.rho[i],
                    profile.rmse[i])
        end
    end
end

save_kappa_profile_csv("qe_profile_1024.csv", profile_1024)
save_kappa_profile_csv("qe_profile_4096.csv", profile_4096)
save_kappa_profile_csv("qe_profile_512_4096.csv", profile_sched)
save_kappa_profile_csv("qe_profile_1024_de.csv", profile_1024_de)
save_kappa_profile_csv("qe_profile_4096_de.csv", profile_4096_de)
save_kappa_profile_csv("qe_profile_256_4096_de.csv", profile_sched_de)

using CairoMakie
rmse_1024 = sqrt.(profile_1024.rmse ./ length(refValues))
rmse_4096 = sqrt.(profile_4096.rmse ./ length(refValues))
rmse_sched = sqrt.(profile_sched.rmse ./ length(refValues))
rmse_1024_de = sqrt.(profile_1024_de.rmse ./ length(refValues))
rmse_4096_de = sqrt.(profile_4096_de.rmse ./ length(refValues))
rmse_sched_de = sqrt.(profile_sched_de.rmse ./ length(refValues))

using CairoMakie

# Thin markers for print (full line, ~20 markers)
thin(κ, y; k=22) = (idx = 1:max(1, length(κ) ÷ k):length(κ); (κ[idx], y[idx]))

function series!(ax, κ, rmse; color, linestyle, marker, label)
    κm, ym = thin(κ, rmse)
    lines!(ax, κ, rmse; color=color, linewidth=2.0, linestyle=linestyle)
    # filled markers read better than hollow when every series has its own color
    sc = scatter!(ax, κm, ym;
        color=color, marker=marker, markersize=11,
        strokecolor=:white, strokewidth=0.8,  # thin halo → readable on overlaps
        label=label,
    )
    return sc
end

fig = with_theme(theme_minimal()) do
    fig = Figure(
        size = (850, 360),
        fontsize = 12,
        figure_padding = (2, 8, 2, 8),
    )

    ax = Axis(
        fig[1, 1];
        xlabel = L"\kappa",
        ylabel = "RMSE",
        yscale = log10,
        topspinevisible = false,
        rightspinevisible = false,
        spinewidth = 1.0,
        xtickwidth = 1.0,
        ytickwidth = 1.0,
        xminorticksvisible = true,
        yminorticksvisible = true,
        xminorticks = IntervalsBetween(5),
        yminorticks = IntervalsBetween(5),
        xgridvisible = false,          # less ink (classic journal look)
        ygridvisible = true,
        ygridstyle = :dot,
        ygridcolor = (:gray, 0.4),
        yminorgridvisible = false,
    )

    # Cycled(i) → Makie default Wong palette (same as bare lines!)
    # Order: group by optimizer, then by n — easy legend scanning
    series!(ax, profile_1024.kappa, rmse_1024;
        color=Cycled(1), linestyle=:solid, marker=:circle,
        label="NM, n = 1024")

    series!(ax, profile_4096.kappa, rmse_4096;
        color=Cycled(2), linestyle=:solid, marker=:rect,
        label="NM, n = 4096")

    series!(ax, profile_sched.kappa, rmse_sched;        color=Cycled(3), linestyle=:solid, marker=:utriangle,        label="NM warm-start (512 → 4096)")

    series!(ax, profile_1024_de.kappa, rmse_1024_de;
        color=Cycled(4), linestyle=:dash, marker=:diamond,
        label="DE, n = 1024")

    series!(ax, profile_4096_de.kappa, rmse_4096_de;
        color=Cycled(5), linestyle=:dash, marker=:rtriangle,
        label="DE, n = 4096")

	#series!(ax, profile_sched.kappa, rmse_sched_de;        color=Cycled(6), linestyle=:solid, marker=:ucircle,        label="DE warm-start (256 → 4096)") # actually worse than 4096, and not that suitable. May need more steps.


    # Side legend: does not cover curves; standard in papers
    Legend(
        fig[1, 2], ax;
        framevisible = false,
        labelsize = 11,
        rowgap = 4,
        patchsize = (28, 14),
        padding = (4, 4, 4, 4),
    )

    colgap!(fig.layout, 8)
    fig
end
save("qe_kappa_profile_compare.pdf", fig)


		=#

		#=
#Error per cap

modelc, rmse = AQFED.VolatilityModels.calibrateCliquets(
           HestonModel(HestonParams(reductionE.v0, params05.κ, reductionE.θ, params05.ρ, params05.σ),driftCurve,discountCurve),
           (:κ, :ρ, :σ),
           [0.1,-0.99,0.1],
           [15.0, 0.5, 5.0],
           refValues;
           config, n_schedule=[4096], spot, payoff,
           polish = false, n_polish=4096,
           useSA = false,
           useDE=false, useNM=false, useLBFGS=true,
           seed = 2,
       )
values_lbfgs_4, stddevs = AQFED.MonteCarlo.simulateIter(seq, modelc,spot, payoff, start, 64*4*1024, config)

modelc, rmse = AQFED.VolatilityModels.calibrateCliquets(
           HestonModel(HestonParams(reductionE.v0, params05.κ, reductionE.θ, params05.ρ, params05.σ),driftCurve,discountCurve),
           (:κ, :ρ, :σ),
           [0.1,-0.99,0.1],
           [15.0, 0.5, 5.0],
           refValues;
           config, n_schedule=[4096], spot, payoff,
           polish = false, n_polish=4096*4,
           useSA = false,
           useDE=false, useNM=false, useLBFGS=true,
           seed = 2,
       )

modelc, rmse = AQFED.VolatilityModels.calibrateCliquets(
           HestonModel(HestonParams(reductionE.v0, params05.κ, reductionE.θ, params05.ρ, params05.σ),driftCurve,discountCurve),
           (:κ, :ρ, :σ),
           [0.1,-0.99,0.1],
           [15.0, 0.5, 5.0],
           refValues;
           config, n_schedule=[4096], spot, payoff,
           polish = false, n_polish=4096*4,
           useSA = false,
           useDE=false, useNM=false, useLBFGS=true,
           seed = 1,
       )

values_lbfgs_16, stddevs = AQFED.MonteCarlo.simulateIter(seq, modelc,spot, payoff, start, 64*4*1024, config)

 modelc, rmse = AQFED.VolatilityModels.calibrateCliquetsMultiseed(
           HestonModel(HestonParams(reductionE.v0, params05.κ, reductionE.θ, params05.ρ, params05.σ),driftCurve,discountCurve),
           (:κ, :ρ, :σ),
           [0.1,-0.99,0.1],
           [15.0, 0.5, 5.0],
           refValues;
           config, n_schedule=[1024], spot, payoff,
           polish = true,seeds=[1,2,3,4,5,6,7,8], n_polish=64*1024)
values_ms_64, stddevs = AQFED.MonteCarlo.simulateIter(seq, modelc,spot, payoff, start, 64*4*1024, config)
using CairoMakie

# Example: caps = [0.01, 0.0125, ..., 0.04]
cap_pct = 100 .* caps

err_lbfgs_16 = 10_000 .* (values_lbfgs_16_1 .- refValues)
err_lbfgs_4  = 10_000 .* (values_lbfgs_4  .- refValues)
err_ms_64     = 10_000 .* (values_ms_64     .- refValues)

# Labels such as "1.00%", "1.25%", ...
cap_labels = [@sprintf("%.2f%%", c) for c in cap_pct]
cap_pct = 100 .* caps
cap_labels = [@sprintf("%.2f%%", c) for c in cap_pct]


fig = Figure(
    size = (820, 360),
    fontsize = 12,
    figure_padding = (2, 12, 2, 8),
)

ax = Axis(
    fig[1, 1];
    xlabel = "Cap",
    ylabel = "Error (bp)",
    title = "Cliquet calibration error",
    topspinevisible = false,
    rightspinevisible = false,
    xminorticksvisible = true,
    yminorticksvisible = true,
    xminorticks = IntervalsBetween(5),
    yminorticks = IntervalsBetween(5),
    xticks = (cap_pct, cap_labels),
    xgridvisible = false,
    ygridvisible = true,
    ygridstyle = :dot,
    ygridcolor = (:gray, 0.4),
)

scatterlines!(
    ax,
    cap_pct,
    err_lbfgs_16;
    linewidth = 2.0,
    marker = :circle,
    markersize = 8,
    color = :steelblue,
    strokecolor = :white,
    strokewidth = 0.5,
    label = "LBFGS 16K",
)

scatterlines!(
    ax,
    cap_pct,
    err_lbfgs_4;
    linewidth = 2.0,
    marker = :rect,
    markersize = 8,
    color = :darkorange,
    strokecolor = :white,
    strokewidth = 0.5,
    label = "LBFGS 4K",
)

scatterlines!(
    ax,
    cap_pct,
    err_ms_64;
    linewidth = 2.0,
    marker = :diamond,
    markersize = 9,
    color = :seagreen,
    strokecolor = :white,
    strokewidth = 0.5,
    label = "Multiseed 64K",
)

axislegend(
    ax;
    position = :rt,
    framevisible = false,
    labelsize = 10,
)

fig
save("cliquet_pricing_errors_bp.png", fig; px_per_unit = 2)
		=#
using LinearAlgebra, Statistics, Printf

"""
Fit a parabola through the 3 grid points with lowest RMSE
and return its vertex (κ_min). Falls back to NaN on failure.
"""
function quadratic_kappa_min(kappas, rmse; mode=:best3)
    n = length(kappas)
    n < 3 && return (NaN, false)

    if mode === :best3
        idx = sortperm(rmse)[1:3]
        x, y = kappas[idx], rmse[idx]
    else  # :neighbors → best point ± 1
        i0 = argmin(rmse)
        lo, hi = max(i0 - 1, 1), min(i0 + 1, n)
        hi - lo < 2 && return (NaN, false)
        x, y = kappas[lo:hi], rmse[lo:hi]
    end

    A = hcat(x .^ 2, x, ones(length(x)))
    try
        a, b, c = A \ y
        (isfinite(a) && a > 0) || return (NaN, false)
        κ_min = -b / (2a)
        (minimum(x) - 0.5 <= κ_min <= maximum(x) + 0.5) || return (NaN, false)
        return (κ_min, true)
    catch
        return (NaN, false)
    end
end
function kappa_profiles_multiseed_DE(
    kappas;
    seeds,
    v0, θ,
    driftCurve, discountCurve,
    spot, payoff,
    refValues,
    config,
    n_schedule = [1024],
    pop_schedule = [15],
    n_generations = [100],
    sigma_bounds = [0.01, 5.0],
    rho_bounds   = [-0.99, 0.5],
)
    nq = length(refValues)
    ns = length(seeds)
    profiles = Vector{Any}(undef, ns)
    rmse_mat = zeros(length(kappas), ns)

    for (j, seed) in enumerate(seeds)
        prof = qe_kappa_profile_calibrateCliquetsDE(
            kappas;
            v0 = v0, θ = θ,
            driftCurve = driftCurve,
            discountCurve = discountCurve,
            spot = spot, payoff = payoff,
            refValues = refValues, config = config,
            n_schedule = n_schedule,
            pop_schedule = pop_schedule,
            n_generations = n_generations,
            sigma_bounds = sigma_bounds,
            rho_bounds = rho_bounds,
            seed = seed,
        )
        profiles[j] = prof
        rmse_mat[:, j] .= sqrt.(prof.rmse ./ nq)
        @printf("seed=%4d done, min RMSE = %e\n", seed, minimum(rmse_mat[:, j]))
    end

    kappa_min_grid = similar(kappas, ns)
    kappa_min_quad = similar(kappas, ns)
    ok_quad = trues(ns)

    for j in 1:ns
        rmse = rmse_mat[:, j]
        i★ = argmin(rmse)
        kappa_min_grid[j] = kappas[i★]
        κq, ok = quadratic_kappa_min(kappas, rmse)
        kappa_min_quad[j] = ok ? κq : kappas[i★]
        ok_quad[j] = ok
    end

    return (;
        kappas,
        seeds = collect(seeds),
        profiles,
        rmse_mat,
        rmse_med = mapslices(median, rmse_mat; dims=2)[:],
        rmse_lo  = mapslices(x -> quantile(x, 0.1), rmse_mat; dims=2)[:],
        rmse_hi  = mapslices(x -> quantile(x, 0.9), rmse_mat; dims=2)[:],
        kappa_min_grid,
        kappa_min_quad,
        ok_quad,
        kappa_med = median(kappa_min_quad),
    )
end
kappas = collect(range(1.0, 12.0, length=31))
seeds  = 1:20

res_de_1024 = kappa_profiles_multiseed_DE(
    kappas;
    seeds = seeds,
    v0 = reductionE.v0,
    θ = reductionE.θ,
    driftCurve = driftCurve,
    discountCurve = discountCurve,
    spot = spot, payoff = payoff,
    refValues = refValues, config = config,
    n_schedule = [1024],
    pop_schedule = [10],
    n_generations = [50],
)

@info "DE κ minimizers" median = res_de_1024.kappa_med range = extrema(res_de_1024.kappa_min_quad)

using CairoMakie

function plot_kappa_multiseed_DE(res; title = "DE κ-profile across MC seeds, n = 1024")
    fig = with_theme(theme_minimal()) do
        fig = Figure(size = (820, 360), fontsize = 12, figure_padding = (2, 12, 2, 8))
        ax = Axis(fig[1, 1];
            xlabel = L"\kappa", ylabel = "RMSE", yscale = log10,
            title = title,
            topspinevisible = false, rightspinevisible = false,
            xminorticksvisible = true, yminorticksvisible = true,
            xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5),
            xgridvisible = false, ygridvisible = true,
            ygridstyle = :dot, ygridcolor = (:gray, 0.4),
        )

        κ = res.kappas

        # individual seed profiles (light spaghetti)
        for j in 1:size(res.rmse_mat, 2)
            lines!(ax, κ, res.rmse_mat[:, j]; color = (:gray, 0.25), linewidth = 0.9)
        end

        # 10–90% band + median profile
        band!(ax, κ, res.rmse_lo, res.rmse_hi;
            label = "10–90% across seeds")
        scatterlines!(ax, κ, res.rmse_med;
            linewidth = 2.2, linestyle = :dash,
            marker = :diamond, markersize = 10,
            strokecolor = :white, strokewidth = 0.6,
            label = "median profile")

        # per-seed quadratic minima (placed at their seed's min RMSE)
        for j in 1:length(res.seeds)
            scatter!(ax, [res.kappa_min_quad[j]], [minimum(res.rmse_mat[:, j])];
                color = :white, strokewidth = 1.2,
                marker = :diamond, markersize = 9)
        end

        # robust κ recommendation
        vlines!(ax, [res.kappa_med]; linestyle = :dot, linewidth = 1.8,
            label = "median κ* = $(round(res.kappa_med, digits=2))")

        axislegend(ax; position = :rt, framevisible = false, labelsize = 10)
        fig
    end
    return fig
end

fig = plot_kappa_multiseed_DE(res_de_1024)
save("qe_kappa_profile_DE_multiseed_1024.pdf", fig; pt_per_unit = 1)
function plot_kappa_min_hist(res)
    fig = Figure(size = (600, 300), fontsize = 12)
    ax = Axis(fig[1, 1];
        xlabel = L"\kappa^{\star}", ylabel = "count",
        topspinevisible = false, rightspinevisible = false)
    hist!(ax, res.kappa_min_quad; bins = 10,
        strokecolor = :black, strokewidth = 0.5)
    vlines!(ax, [res.kappa_med]; linestyle = :dash, linewidth = 2)
    return fig
end

fig2 = plot_kappa_min_hist(res_de_1024)
save("qe_kappa_min_hist_DE_1024.pdf", fig2; pt_per_unit = 1)






using Random, Statistics, CairoMakie

"""
Repeated 3D NM calibrations of (κ, ρ, σ) with different MC seeds.
v0, θ fixed from term structure / vanillas.
"""
function calibrate_3d_multiseed(
    v0, θ,
    driftCurve, discountCurve,
    kappa_bounds, rho_bounds, sigma_bounds,
    refValues;
    config,
    n_schedule,
    spot,
    payoff,
    seeds = 1:30,
    κ0 = kappa_bounds[1],   # or mid-range start
    ρ0 = rho_bounds[1],
    σ0 = sigma_bounds[1],
)
    ns = length(seeds)
    rmse  = fill(NaN, ns)
    κ_hat = fill(NaN, ns)
    ρ_hat = fill(NaN, ns)
    σ_hat = fill(NaN, ns)

    # 3D free parameters
    names = (:κ, :ρ, :σ)
    x_lb  = [kappa_bounds[1], rho_bounds[1], sigma_bounds[1]]
    x_ub  = [kappa_bounds[2], rho_bounds[2], sigma_bounds[2]]

    for (j, seed) in enumerate(seeds)
        params0 = HestonParams(v0, κ0, θ, ρ0, σ0)
        model0  = HestonModel(params0, driftCurve, discountCurve)

        modelc, rmse_j = AQFED.VolatilityModels.calibrateCliquets(
            model0,
            names,
            x_lb,
            x_ub,
            refValues;
            config = config,
            n_schedule = n_schedule,
            spot = spot,
            payoff = payoff,
            polish = false,
            useDE = false,
            useNM = true,
            seed = seed,   # <-- method parameter
        )

        rmse[j]  = rmse_j
        κ_hat[j] = modelc.params.κ   # adjust field names if needed
        ρ_hat[j] = modelc.params.ρ
        σ_hat[j] = modelc.params.σ
    end

    return (;
        seeds = collect(seeds),
        rmse, κ_hat, ρ_hat, σ_hat,
        rmse_med = median(rmse),
        rmse_lo  = quantile(rmse, 0.1),
        rmse_hi  = quantile(rmse, 0.9),
    )
end



"""
Repeated 3D NM calibrations of (κ, ρ, σ) with different MC seeds.
v0, θ fixed from term structure / vanillas.
"""
function calibrate_3d_multiseed_average(
    v0, θ,
    driftCurve, discountCurve,
    kappa_bounds, rho_bounds, sigma_bounds,
    refValues;
    config,
    n_schedule,
    spot,
    payoff,
    seeds = 1:4,
    κ0 = (kappa_bounds[1]+kappa_bounds[2])/2, 
    ρ0 = (rho_bounds[1]+rho_bounds[2])/2,
    σ0 = (sigma_bounds[1]+sigma_bounds[2])/2,
)
    ns = length(seeds)
    rmse  = fill(NaN, ns)
    κ_hat = fill(NaN, ns)
    ρ_hat = fill(NaN, ns)
    σ_hat = fill(NaN, ns)

    # 3D free parameters
    names = (:κ, :ρ, :σ)
    x_lb  = [kappa_bounds[1], rho_bounds[1], sigma_bounds[1]]
    x_ub  = [kappa_bounds[2], rho_bounds[2], sigma_bounds[2]]

    for (j, seed) in enumerate(seeds)
        params0 = HestonParams(v0, κ0, θ, ρ0, σ0)
        model0  = HestonModel(params0, driftCurve, discountCurve)

        modelc, rmse_j = AQFED.VolatilityModels.calibrateCliquets(
            model0,
            names,
            x_lb,
            x_ub,
            refValues;
            config = config,
            n_schedule = n_schedule,
            spot = spot,
            payoff = payoff,
            polish = false,
            useDE = false,
            useNM = true,
            seed = seed,   # <-- method parameter
        )

        rmse[j]  = rmse_j
        κ_hat[j] = modelc.params.κ   # adjust field names if needed
        ρ_hat[j] = modelc.params.ρ
        σ_hat[j] = modelc.params.σ
    end

    return (;
        seeds = collect(seeds),
        rmse, κ_hat, ρ_hat, σ_hat,
        rmse_med = median(rmse),
        rmse_lo  = quantile(rmse, 0.1),
        rmse_hi  = quantile(rmse, 0.9),
    )
end



using Random, Statistics, CairoMakie

"""
Repeated 3D NM calibrations of (κ, ρ, σ) with different MC seeds.
v0, θ fixed from term structure / vanillas.
"""
function calibrate_3d_multiseed(
    v0, θ,
    driftCurve, discountCurve,
    kappa_bounds, rho_bounds, sigma_bounds,
    refValues;
    config,
    n_schedule,
    spot,
    payoff,
    seeds = 1:30,
	useDE = false,
	isRelative = false,
    κ0 = kappa_bounds[1],   # or mid-range start
    ρ0 = rho_bounds[1],
    σ0 = sigma_bounds[1],
)
    ns = length(seeds)
    rmse  = fill(NaN, ns)
    κ_hat = fill(NaN, ns)
    ρ_hat = fill(NaN, ns)
    σ_hat = fill(NaN, ns)

    # 3D free parameters
    names = (:κ, :ρ, :σ)
    x_lb  = [kappa_bounds[1], rho_bounds[1], sigma_bounds[1]]
    x_ub  = [kappa_bounds[2], rho_bounds[2], sigma_bounds[2]]

    for (j, seed) in enumerate(seeds)
        params0 = HestonParams(v0, κ0, θ, ρ0, σ0)
        model0  = HestonModel(params0, driftCurve, discountCurve)

        modelc, rmse_j = AQFED.VolatilityModels.calibrateCliquets(
            model0,
            names,
            x_lb,
            x_ub,
            refValues;
            config = config,
            n_schedule = n_schedule,
            spot = spot,
            payoff = payoff,
            polish = false,
            useDE = useDE,
            useNM = true,
            seed = seed, 
			isRelative = isRelative  # <-- method parameter
        )

        rmse[j]  = rmse_j
        κ_hat[j] = modelc.κ   # adjust field names if needed
        ρ_hat[j] = modelc.ρ
        σ_hat[j] = modelc.σ
    end

    return (;
        seeds = collect(seeds),
        rmse, κ_hat, ρ_hat, σ_hat,
        rmse_med = median(rmse),
        rmse_lo  = quantile(rmse, 0.1),
        rmse_hi  = quantile(rmse, 0.9),
    )
end
using Statistics

"""
Given rmse[seed] and parameter arrays κ_hat[seed], ρ_hat[seed], σ_hat[seed],
return the parameters and rmse of the seed with median RMSE.
For even length, the two central seeds are returned.
"""
function median_seed_results(rmse, κ_hat, ρ_hat, σ_hat)
    @assert length(rmse) == length(κ_hat) == length(ρ_hat) == length(σ_hat)
    idx = sortperm(rmse)
    n = length(rmse)

    if isodd(n)
        j = idx[(n + 1) ÷ 2]
        return (;
            seed_index = j,
            rmse = rmse[j],
            κ = κ_hat[j],
            ρ = ρ_hat[j],
            σ = σ_hat[j],
        )
    else
        j1 = idx[n ÷ 2]
        j2 = idx[n ÷ 2 + 1]
        # report both; caller can pick (e.g. lower κ, or average if desired)
        return (;
            seed_indices = (j1, j2),
            rmse = (rmse[j1], rmse[j2]),
            κ = (κ_hat[j1], κ_hat[j2]),
            ρ = (ρ_hat[j1], ρ_hat[j2]),
            σ = (σ_hat[j1], σ_hat[j2]),
        )
    end
end
seeds = 1:30   # 20–50 is typical for a stability cloud

res3d_1024_p2 = calibrate_3d_multiseed(
           reductionE.v0, reductionE.θ, driftCurve, discountCurve,
           [0.1,15.0], [-0.99,0.5], [0.1,5.0],
           refValues,
           config=config, n_schedule=[1024], spot=spot, payoff=payoff,
           seeds = seeds,
           # optional: common start interior to the box
           κ0 = params2.κ, ρ0 = params2.ρ, σ0 = params2.σ,
       )

@info "3D NM seed stability" median_rmse=res3d.rmse_med κ_range=extrema(res3d.κ_hat)
function plot_3d_seed_stability(res; title = "3D NM, varying MC seed")
    fig = Figure(size = (900, 320), fontsize = 12)

    # colour = RMSE (log scale via rank or log10)
    c = log10.(res.rmse)
    common = (
        color = c,
        colormap = :viridis,
        markersize = 11,
        strokecolor = :white,
        strokewidth = 0.6,
    )

    ax1 = Axis(fig[1, 1]; xlabel = L"\kappa", ylabel = L"\rho",
        title = title, topspinevisible=false, rightspinevisible=false)
    ax2 = Axis(fig[1, 2]; xlabel = L"\kappa", ylabel = L"\sigma",
        topspinevisible=false, rightspinevisible=false)
    ax3 = Axis(fig[1, 3]; xlabel = L"\rho", ylabel = L"\sigma",
        topspinevisible=false, rightspinevisible=false)

    sc1 = scatter!(ax1, res.κ_hat, res.ρ_hat; common...)
    scatter!(ax2, res.κ_hat, res.σ_hat; common...)
    scatter!(ax3, res.ρ_hat, res.σ_hat; common...)

    Colorbar(fig[1, 4], sc1; label = L"\log_{10}(\mathrm{RMSE})", width = 12)

    # median point
    for (ax, x, y) in (
        (ax1, median(res.κ_hat), median(res.ρ_hat)),
        (ax2, median(res.κ_hat), median(res.σ_hat)),
        (ax3, median(res.ρ_hat), median(res.σ_hat)),
    )
        scatter!(ax, [x], [y]; color=:red, marker=:star5, markersize=16)
    end

    colgap!(fig.layout, 8)
    return fig
end

fig = plot_3d_seed_stability(res3d)
save("heston_3d_seed_stability.pdf", fig; pt_per_unit = 1)
function plot_3d_marginals(res)
    fig = Figure(size = (850, 280), fontsize = 12)
    for (col, (vals, lab)) in enumerate((
        (res.κ_hat, L"\hat{\kappa}"),
        (res.ρ_hat, L"\hat{\rho}"),
        (res.σ_hat, L"\hat{\sigma}"),
        (res.rmse, "RMSE"),
    ))
        ax = Axis(fig[1, col]; xlabel = lab, ylabel = col == 1 ? "count" : "",
            topspinevisible=false, rightspinevisible=false,
            yticklabelsvisible = col == 1)
        hist!(ax, vals; bins = min(12, max(5, length(vals) ÷ 3)),
            color = (:steelblue, 0.75), strokecolor = :black, strokewidth = 0.5)
        vlines!(ax, [median(vals)]; color=:red, linewidth=2, linestyle=:dash)
    end
    return fig
end
function plot_rmse_vs_kappa_seeds(res)
    fig = Figure(size = (700, 320), fontsize = 12)
    ax = Axis(fig[1, 1];
        xlabel = L"\hat{\kappa}", ylabel = "RMSE", yscale = log10,
        title = "3D NM terminals across seeds",
        topspinevisible=false, rightspinevisible=false,
        ygridstyle=:dot, ygridcolor=(:gray, 0.4),
    )
    scatter!(ax, res.κ_hat, res.rmse;
        color = :steelblue, markersize = 12,
        strokecolor=:white, strokewidth=0.8,
        label = "one seed = one calibration")
    # optional: horizontal band for objective noise
    hlines!(ax, [res.rmse_med]; color=:red, linestyle=:dash, label="median RMSE")
    axislegend(ax; position=:rt, framevisible=false)
    return fig
end
starts = [
    (κ=1.0, ρ=-0.3, σ=0.3),
    (κ=2.0, ρ=-0.6, σ=0.5),
    (κ=4.0, ρ=-0.2, σ=0.8),
]
rows = []
#= placeholder call: calibrate_3d_multiseed(...) arguments to fill in
for st in starts, seed in 1:10
    r = calibrate_3d_multiseed(...; seeds=seed:seed, κ0=st.κ, ρ0=st.ρ, σ0=st.σ)
    push!(rows, (; st..., seed, κ=r.κ_hat[1], ρ=r.ρ_hat[1], σ=r.σ_hat[1], rmse=r.rmse[1]))
end
=#
modelc, rmse = AQFED.VolatilityModels.calibrateCliquets(
    model0,
    (:κ, :ρ, :σ),
    [kappa_bounds[1], rho_bounds[1], sigma_bounds[1]],
    [kappa_bounds[2], rho_bounds[2], sigma_bounds[2]],
    refValues;
    config, n_schedule, spot, payoff,
    polish = false,
    useDE = false,
    useNM = true,
    seed = seed,
)






v00 = reduction1.v0
κ0 = reduction1.κ
θ0 = reduction1.θ
params0 = HestonParams(v00, κ0, θ0, params.ρ, params.σ)

AQFED.VolatilityModels.calibrateCliquets(HestonModel(
params0, driftCurve, discountCurve), (:σ, :ρ, :κ), [0.01, -0.99,0.1],[10.0, 0.99,10.0], refValues, config=config, n=16*1024, spot=spot, payoff=payoff)

AQFED.VolatilityModels.calibrateCliquets(HestonModel(
params0, driftCurve, discountCurve), (:σ, :ρ), [0.01, -0.99],[10.0, 0.99], refValues, config=config, n=16*1024, spot=spot, payoff=payoff)


hParams_calibrated = HestonParams{Float64}(v00, κ0, θ0, ρopt, σopt)	


###
	uPrices, isCall, uWeights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ts, uForwards, strikes, vols, vegaFloor = 1e-1)

	params, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, ones(size(vols)))

	surface = AQFED.VolatilityModels.SplineSurface(ts, uForwards, log.(strikes ./ uForwards), vols)
	lvModel = AQFED.TermStructure.LocalVolatilityModel(surface, driftCurve, discountCurve)



	volsH = zeros(size(vols))
	for i in axes(strikes, 1), j in axes(strikes, 2)

		pricer = CharFuncPricing.ALCharFuncPricer(CharFuncPricing.DefaultCharFunc(hParams))
		isCall = strikes[i, j] > uForwards[i]
		price = CharFuncPricing.priceEuropean(pricer, isCall, strikes[i, j], uForwards[i], ts[i], 1.0)
		volsH[i, j] = AQFED.Black.impliedVolatility.(isCall, price, uForwards[i], strikes[i, j], ts[i], 1.0)
	end
	surfaceH = AQFED.VolatilityModels.SplineSurface(ts, uForwards, log.(strikes ./ uForwards), volsH)
	lvModelH = AQFED.TermStructure.LocalVolatilityModel(surfaceH, driftCurve, discountCurve)
	#two lv models: 1) market 2) heston. Prices of cliquets are too low. no flexibility to adjust them, and it is inherent to hte model. This is true Even if we calibrate the localvolatility to the heston model implied volatilities instead of the market volatilities. This is particularly visible for the low cap values on this example. More importantly, the LV model is inadequate to match cliquet prices from counterparties, as it is already fully defined by the vanillas.

	caps = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 3.50, 4.0] ./ 100
	obsTimes = collect(range(0.0/12, 1.0, length = 13))
	#obsTimes = vcat(0.0001, obsTimes)

	#=
	hParams = HestonParams(0.18^2, 7.6276,0.25^2,  -0.887, 1.347) #1M cliquet candidate
	hParams = HestonParams(0.021904, 7.6276, 0.060661, -0.9298, 1.3368) #v0 and theta should not be so low! var swaps are higher. Correct fit may be t[1] exact, t[end]exact and then calibrate kappa only. interestingly while using t[end] is okish for theta, using t[1] directly for v0 without adj leads to a higher varswap heston price compared to vsTheo[1]. (but this is what seems to have been done by the csystem)
    hParams = HestonParams(0.021165, 6.4657, 0.060607, -0.9730, 1.1596)
	=#
	model = HestonModel(hParams, driftCurve, discountCurve)
	timestep = 52
    n = 64*1024
    start=0
	for cap in caps
		payoff = AQFED.MonteCarlo.CliquetOption(
			cap, -1.0, 1e100, 0.0, 0.0,
			AQFED.MonteCarlo.PeriodicCashFlow(obsTimes, obsTimes[end], 1, 0.0),
			0.0)
		specTimes = AQFED.MonteCarlo.specificTimes(payoff)

		config = AQFED.MonteCarlo.GridConfig(model, specTimes, 1.0 / timestep, AQFED.MonteCarlo.DVSS2X(withBB = false), mindt=0.5/365, tstop=0.75)
		ndim = AQFED.MonteCarlo.ndims(model, config)
		rng = AQFED.Random.Well1024a()
		#seq = AbstractRNGSeq(rng, ndim)
		seq = ScrambledSobolSeq(ndim, n, Owen(31, ScramblingRngAdapter(rng)))
		time = @elapsed value, stderror =
			mcStat(AQFED.MonteCarlo.simulateDVSS2XIter(seq, model, spot, payoff, start, n, config))
		println("Heston ",
			timestep,
			" ",
			value,
			" ",
			stderror,
			" ",
			time,
		)
		rng = AQFED.Random.Well1024a()
		nd = AQFED.MonteCarlo.ndims(lvModel, specTimes, 1.0 / timestep)

	
	end


	#TODO multiplicative cliquet. Note: additive cliquet already has issues with QE and 8 time steps (and to some extent 16).
	#TODO arith cliquet monthly. QE vs DVSS2X.
	# => plot profile of error with dt=1/8 vs 1/12 vs 1/16 1/128 for QE-M and DVSS2X. The error is actually up to 8% relative, and generally close to 3% for low caps (< 2%). estimate actual RQMC stderror. 32K < 0.01% meaning < 1/16 step even with QE-M. BB does not help DVSS2X (is there a bug?) 1stddev seems to be real effective error with RQMC (to verify).

	#mult cliquet params:
	# val=2021/03/15, sp500, not= 1M, start = 2020/02/20, end=2026/02/18, obs=2021/02/17, 2022/02/16, 2023/02/15, 2024/02/14, 2025/02/19, localCap=10.5%, start date added as ref obs.localfloor=0,localbuffer=0.10, globalcap none, global floor none.
	#kappa=4.435 theta=0.0691 sigma=1.947, v0=0.0399, rho=-0.7114
	#heston RUSSEL params kappa=0.702, theta=0.0756, sigma=0.6148, v0=0.0928, rho=-0.6997
	#heston filtered fixed kappa. kappa=0.25, theta=0.1297, sigma=0.6438, v0=0.02902, rho=-0.6979 for cliquet start date 2024/06/06, maturity 2031/06/06, yearly obs including start date. val date=2025/10/8.localcap=9.98% gearing=0.97, premium=5.47%. multiplcative with premium paid at each obs. pi_i gearing*(1+R_i).

	#arith cliquet: 1Y, monthly, globalcap=35% globalfloor=5% localcap=7% localfloor=0%

end

function test_slv_vanilla_and_cliquets()
	spot, ts, dfs, uForwards, strikes, vols = makeMarketData20260512()
	refValues = [0.86,1.27 ,1.73 ,2.22 ,2.73 ,3.24 ,3.74 ,4.67 ,5.49 ,6.18 ] ./ 100
	logdfInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, log.(dfs)))
	logForwardInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, -log.(uForwards)))
	discountCurve = InterpolatedLogDiscountFactorCurve(logdfInterp)
	driftCurve = InterpolatedLogDiscountFactorCurve(logForwardInterp)
	surface = AQFED.VolatilityModels.SplineSurface(ts, uForwards, log.(strikes ./ uForwards), vols)
	lvModel = LocalVolatilityModel(surface, driftCurve, discountCurve)
volsF, strikesF = AQFED.VolatilityModels.filterVolsBySimpleDelta(ts, uForwards, strikes, vols, deltas=[0.1, 0.25, 0.5, 0.75,0.9])
 params05, rmse05 = AQFED.VolatilityModels.calibrateHestonFromVols(ts, uForwards, strikesF, volsF,vegaFloor=1e-2)

	modelc, rmse = AQFED.VolatilityModels.calibrateCliquetsMultiseed(
    HestonModel(params05,driftCurve,discountCurve),
    (:κ, :ρ, :σ),
    [0.1,-0.99,0.1],
    [15.0, 0.5, 5.0],
    refValues;
    config, n_schedule=[1024], spot, payoff,
    polish = true,seeds=[1,2,3,4,5,6,7,8], n_polish=64*1024)

	slvModel = HestonLVModel(modelc, lvModel)
	hestonModel = modelc

	
	println("\n=== 1. Testing SLV Vanilla Call Option Pricing vs Ref Model ===")
	ttes = [0.25, 0.5, 1.0, 4.0]
	Krels = [0.8, 0.9, 0.95, 1.0,1.05, 1.1, 1.2,0.0]
	ivMatrix = zeros(length(ttes), length(Krels)-1)
	ivSLVMatrix = zeros(length(ttes), length(Krels)-1)
	ivHestonMatrix = zeros(length(ttes), length(Krels)-1)
	ivLVMatrix = zeros(length(ttes), length(Krels)-1)
	for (i,tte) = enumerate(ttes)
		payoff = AQFED.MonteCarlo.ListPayoff([AQFED.MonteCarlo.VanillaOption(true, spot*K, AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0) for K in Krels])
		specTimes = AQFED.MonteCarlo.specificTimes(payoff)
		config = AQFED.MonteCarlo.GridConfig(hestonModel, specTimes, 1.0 / 64, AQFED.MonteCarlo.QE(true,true), mindt=0.5/365, tstop=1.0)

        nd = AQFED.MonteCarlo.ndims(lvModel, specTimes, 1.0 / 64)

        #seq = AbstractRNGSeq(rng, ndim)
        seq = ScrambledSobolSeq(nd, 16384*4, OwenFaureTezuka(31, ScramblingRngAdapter(Philox4x((UInt32(1234), UInt32(5678))))))
        lv_price, stderrorLV =
                AQFED.MonteCarlo.simulate(seq, lvModel, spot, payoff, start, 16384*4, 1.0 / 64)


        ndim = AQFED.MonteCarlo.ndims(model, config)

        philox = Philox4x((UInt32(1234), UInt32(5678)))
        rngCalib = ScrambledSobolSeq(ndim, 8192*2, OwenFaureTezuka(31, ScramblingRngAdapter(philox)))

        philox2 = Philox4x((UInt32(9999), UInt32(8888)))
        rngSim = ScrambledSobolSeq(ndim, 16384*4, OwenFaureTezuka(31, ScramblingRngAdapter(philox2)))

        slv_price, slv_err = AQFED.MonteCarlo.simulateScheme(rngSim, slvModel, spot, payoff, 0, 16384*4,config; leverageFunctions=nothing,lcCalc=AQFED.MonteCarlo.CompactSupportCalculator(gridSize=64,inverse=false),rngCalib=rngCalib,nSimCalib=2*8192)
        heston_price, heston_err = mcStat(AQFED.MonteCarlo.simulateIter(rngSim, modelc,spot, payoff, 0, 16384*4, config))

		df = AQFED.TermStructure.discountFactor(discountCurve,tte)
		forward = spot*exp(- AQFED.TermStructure.logDiscountFactor(driftCurve, tte))*df
		for (j, K_rel) = enumerate(Krels)
			if K_rel == 0 
			@printf("forward %5.2f%% SLV %6.4f Heston %6.4f LV %6.4f",forward,slv_price[j],heston_price[j],lv_price[j])
			else 
			strike = K_rel*spot

		logmoneyness=log(strike / spot)+AQFED.TermStructure.logDiscountFactor(driftCurve, tte)
		w_iv = AQFED.TermStructure.varianceByLogmoneyness(surface, logmoneyness,tte)
		iv = sqrt(max(w_iv, 1e-4))
		d1 = (-logmoneyness + 0.5 * iv^2 * tte) / (iv * sqrt(tte))
		d2 = d1 - iv * sqrt(tte)
		ref_call = strike * df * (exp(-logmoneyness)*AQFED.Math.normcdf(d1) - AQFED.Math.normcdf(d2))

		rel_diff = abs(slv_price[j] - ref_call) / ref_call
		iv_slv = AQFED.Black.impliedVolatility(true, slv_price[j], strike*exp(-logmoneyness),strike, tte, df)
		iv_heston = AQFED.Black.impliedVolatility(true, heston_price[j], strike*exp(-logmoneyness),strike, tte, df)
		iv_lv = AQFED.Black.impliedVolatility(true, lv_price[j], strike*exp(-logmoneyness),strike, tte, df)
		ivMatrix[i, j] = iv
		ivLVMatrix[i, j] = iv_lv
		ivSLVMatrix[i, j] = iv_slv
		ivHestonMatrix[i, j] = iv_heston
		@printf("Strike %6.1f (rel %.2f): SLV Price = %8.4f ± %6.4f, BS Ref Price = %8.4f, RelDiff = %5.2f%%, IV SLV=%5.2f%% IV Heston=%5.2f%% IV Ref=%5.2f%%\n",
			strike, K_rel, slv_price[j], slv_err[j], ref_call, rel_diff * 100, iv_slv*100, iv_heston*100, iv*100)
			end
	end
end
	println("\n=== 2. Testing SLV Pricing of Cliquet Option List (with t=0 observation) ===")
	caps = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 3.50, 4.0] ./ 100
	obsTimes = collect(range(0.0/12, 1.0, length = 13))
	payoff = AQFED.MonteCarlo.ListPayoff([AQFED.MonteCarlo.CliquetOption(
			cap, -1.0, 1e100, 0.0, 0.0,
			AQFED.MonteCarlo.PeriodicCashFlow(obsTimes, obsTimes[end], 1, 0.0),
			0.0) for cap in caps])
	specTimes = AQFED.MonteCarlo.specificTimes(payoff)
			config = AQFED.MonteCarlo.GridConfig(hestonModel, specTimes, 1.0 / 64, AQFED.MonteCarlo.QE(true,true), mindt=0.5/365, tstop=1.0)
		ndim = AQFED.MonteCarlo.ndims(model, config)

		philoxCalib = Philox4x((UInt32(42), UInt32(43)))
		rngCalib = ScrambledSobolSeq(ndim, 8192*2, Owen(31, ScramblingRngAdapter(philoxCalib)))

		philoxSim = Philox4x((UInt32(100), UInt32(42)))
		rngSim = ScrambledSobolSeq(ndim, 16384*4, Owen(31, ScramblingRngAdapter(philoxSim)))

		slv_price, slv_err = AQFED.MonteCarlo.simulateScheme(rngSim, slvModel, spot, payoff, 0, 16384*4, config; leverageFunctions=nothing,lcCalc=AQFED.MonteCarlo.CompactSupportCalculator(gridSize=64,inverse=false,bandwidthFactor=3.0),rngCalib=rngCalib,nSimCalib=2*8192)
	
end
#=
ivMatrix = [0.30070397543947375 0.22815581869548945 0.19361403582266 0.16212796273302968 0.1401929913337684 0.13201596831826304 0.14570097598528986; 0.2773066597762941 0.22581191295587294 0.20008849993724043 0.1757073090333437 0.15582488801804897 0.14369479190420642 0.13757473238565632; 0.26098730668252706 0.22488250180883884 0.2068411645150171 0.18904548174616115 0.17266376276765102 0.15899976809171715 0.1438478807091887; 0.23949910456204257 0.22341291112268663 0.215660222731908 0.20806148349657286 0.20039872817443957 0.19285491382236858 0.17878438272169817]

ivLVMatrix = [0.29532975995059546 0.22720793263600977 0.19495106099584944 0.1651729735364258 0.14348611041198917 0.13524877182419412 0.15133729224572892; 0.27793809656521357 0.2269419077525798 0.20186371078404464 0.17818360755332605 0.15869198081165192 0.14641126552598957 0.14247421649994543; 0.264632821124973 0.2275052027830284 0.2092643927900695 0.19140428301429654 0.17497922528674784 0.16154442099483066 0.14688731733533722; 0.24189789012171753 0.2251615117831872 0.21716482169701434 0.20931928410634174 0.20155189495433365 0.1939566165052939 0.1798384800019119]

ivSLVMatrix = [0.3099557807520913 0.23334340549187313 0.19795096594617845 0.16538826646452554 0.14137438064385924 0.13217030693599346 0.15221550602171996; 0.27951926637574953 0.22686281500344022 0.20136754465271708 0.17715166863359708 0.1570833416482451 0.14469070937433406 0.1401180087528163; 0.26619834118204966 0.2275673227647401 0.20898954313882406 0.19088852615011145 0.17422702802543374 0.1601378497023744 0.1446213368439936; 0.24114470419434228 0.22476764684735273 0.21695116426896863 0.20929574278310922 0.20172740584302698 0.19427175078224365 0.18030708672262494]
ivHestonMatrix = [0.31940095737635443 0.26048884823662327 0.2287196843167961 0.19478935582533444 0.15957816631073804 0.12975106782920204 0.11914118139590445; 0.29776660380701375 0.25421162829548394 0.2322593897702537 0.21024487429441552 0.18807045544787 0.16635490973258935 0.13297776899641203; 0.2811115247281224 0.25333057921938784 0.23989439621714376 0.22666291454330273 0.21361490926496465 0.20061757487222395 0.17615992495568517; 0.264408424021982 0.2554794844634732 0.2513350904311263 0.2473580444639568 0.24354076782063425 0.23986819626433448 0.23297893198677283]

keep = ttes .!= 2.0

Krels0 = Krels[1:end-1]
plot_ttes = ttes[keep]

ivMarket_pct = 100 .* ivMatrix[keep, :]
ivHeston_pct = 100 .* ivHestonMatrix[keep, :]
ivSLV_pct    = 100 .* ivSLVMatrix[keep, :]
ivLV_pct     = 100 .* ivLVMatrix[keep, :]

k_labels = [@sprintf("%.1f", k) for k in Krels0]

# ------------------------------------------------------------------
# Common y-axis limits
# ------------------------------------------------------------------

all_iv = vcat(
    vec(ivMarket_pct),
    vec(ivHeston_pct),
    vec(ivSLV_pct),
    vec(ivLV_pct),
)

ymin = floor(minimum(all_iv) * 10) / 10 - 0.5
ymax = ceil(maximum(all_iv) * 10) / 10 + 0.5

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

fig = Figure(
    size = (720, 720),
    fontsize = 12,
    figure_padding = (8, 10, 8, 8),
)

legend_plots = nothing

for j in 1:4

    row = j <= 2 ? 1 : 2
    col = isodd(j) ? 1 : 2

    ax = Axis(
        fig[row, col];
        title = @sprintf("T = %.2f", plot_ttes[j]),

        xlabel = row == 2 ? "Relative strike, K/S" : "",
        ylabel = col == 1 ? "Implied volatility (%)" : "",

        xticks = (Krels0, k_labels),
        limits = (0.78, 1.22, ymin, ymax),

        topspinevisible = false,
        rightspinevisible = false,

        xgridvisible = false,
        ygridvisible = true,
        ygridstyle = :dot,
        ygridcolor = (:gray, 0.35),

        yminorticksvisible = true,
        yminorticks = IntervalsBetween(2),

        titlesize = 13,
        xlabelsize = 11,
        ylabelsize = 11,
    )

# --------------------------------------------------------------
# Heston
# --------------------------------------------------------------

p_heston = scatterlines!(
    ax,
    Krels0,
    ivHeston_pct[j, :];
    linewidth = 2.0,
    linestyle = :dash,
    marker = :rect,
    markersize = 8,
    strokecolor = :white,
    strokewidth = 0.7,
    label = "Heston",
)

# --------------------------------------------------------------
# Heston-LV
# --------------------------------------------------------------

p_slv = scatterlines!(
    ax,
    Krels0,
    ivSLV_pct[j, :];
    linewidth = 2.0,
    linestyle = :solid,
    marker = :diamond,
    markersize = 9,
    strokecolor = :white,
    strokewidth = 0.7,
    label = "Heston-LV",
)

# --------------------------------------------------------------
# Simulated LV
# --------------------------------------------------------------

p_lv = scatterlines!(
    ax,
    Krels0,
    ivLV_pct[j, :];
    linewidth = 2.0,
    linestyle = :dot,
    marker = :utriangle,
    markersize = 8,
    strokecolor = :white,
    strokewidth = 0.7,
    label = "LV",
)

# --------------------------------------------------------------
# Market: draw last so that it is in the foreground
# --------------------------------------------------------------

p_market = scatter!(
    ax,
    Krels0,
    ivMarket_pct[j, :];
    marker = :circle,
    markersize = 10,
    strokecolor = :white,
    strokewidth = 1.0,
	color = :black,
    label = "Market",
)

if j == 1
    # Legend order is independent of plotting order.
    legend_plots = [p_market, p_heston, p_slv, p_lv]
end
    if j == 1
        legend_plots = [p_market, p_heston, p_slv, p_lv]
    end
end

# ------------------------------------------------------------------
# Common legend
# ------------------------------------------------------------------

Legend(
    fig[3, 1:2],
    legend_plots,
    ["Market", "Heston", "Heston-LV", "LV"];
    orientation = :horizontal,
    tellwidth = false,
    framevisible = false,
    labelsize = 11,
    patchsize = (20, 12),
)

fig

save(
    "vanilla_heston_slv_lv_cliquet_params20260512_particle.pdf",
    fig,
)=#

