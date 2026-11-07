# Chapter 9: Heston calibration by Levenberg-Marquardt from the smart guess, differential evolution, simulated annealing
# and particle swarm optimization. The timings are only meaningful on a single thread:
#   julia --project=book -t1 book/run.jl --compute 'heston_calibration'

using AQFED, AQFED.Black, CharFuncPricing, Random, LinearAlgebra
import AQFED.GlobalOptimization, AQFED.VolatilityModels
import LeastSquaresOptim
import RandomNumbers.MersenneTwisters: MT19937

const CAL_LOW_GLOBAL = [1e-4, 0.0, 1e-4, 1e-4, -0.9999] #v0, κ, θ, σ, ρ
const CAL_HIGH_GLOBAL = [1.0, 16.0, 1.0, 4.0, 0.9999]
const CAL_LOW_LM = [0.001, 0.0, 0.0, 1e-4, -0.99]
const CAL_HIGH_LM = [1.0, 16.0, 1.0, 4.0, 0.99]
const CAL_SEED = 5489

struct CalOption
    τ::Float64
    strike::Float64
    vol::Float64
    weight::Float64
    price::Float64
    forward::Float64
    df::Float64
end
calIsCall(o::CalOption) = o.forward <= o.strike

function loadCalibrationSurface(name)
    lines = [split(l, ",") for l in readlines(datapath(name)) if !isempty(strip(l))]
    spot = parse(Float64, lines[1][4])
    isRelative = strip(lines[1][6]) == "true"
    strikes = parse.(Float64, lines[2][2:end]) .* (isRelative ? spot : 1.0)
    expiries = [parse(Float64, l[1]) for l in lines[3:end]]
    vols = [parse(Float64, l[j+1]) for l in lines[3:end], j in eachindex(strikes)]
    return (spot=spot, strikes=strikes, expiries=expiries, vols=vols)
end

#the out-of-the-money options, without the ones of price below minRelPrice × spot, weighted equally or by the capped inverse vega
function calibrationOptions(data, r, q; useVol=true, minRelPrice=1e-6)
    list = CalOption[]
    for (i, τ) in enumerate(data.expiries), (j, strike) in enumerate(data.strikes)
        forward, df, vol = data.spot * exp((r - q) * τ), exp(-r * τ), data.vols[i, j]
        price = Black.blackScholesFormula(forward <= strike, strike, forward, vol^2 * τ, 1.0, df)
        price >= minRelPrice * data.spot && push!(list, CalOption(τ, strike, vol, 1.0, price, forward, df))
    end
    if !useVol
        weights = map(list) do o
            min(1000 / o.forward, 1 / Black.blackScholesVega(o.strike, o.forward, o.vol^2 * o.τ, 1.0, o.df, o.τ))
        end
        weights ./= norm(weights)
        list = [CalOption(o.τ, o.strike, o.vol, w, o.price, o.forward, o.df) for (o, w) in zip(list, weights)]
    end
    return list
end

#variance at the given strike, interpolated linearly in total variance per unit of strike
function interpolateVariance(vols, strikes, k)
    i = clamp(searchsortedlast(strikes, k), 1, length(strikes) - 1)
    return (vols[i]^2 * (strikes[i+1] - k) + vols[i+1]^2 * (k - strikes[i])) / (strikes[i+1] - strikes[i])
end

function smartGuess(data; x0=0.05)
    t1, t2 = 2, length(data.expiries) - 1
    variance(i, k) = interpolateVariance(data.vols[i, :], data.strikes, k)
    input = VolatilityModels.FordeGuessInput5D(x0, data.expiries[t1], data.expiries[t2], variance(1, data.spot),
        variance(t1, data.spot * exp(x0)), variance(t1, data.spot * exp(-x0)), variance(t2, data.spot * exp(x0)), variance(t2, data.spot * exp(-x0)))
    return VolatilityModels.computeHestonFordeGuess(input, kappa_min=0.01, theta_min=0.0)
end

calAsVector(p) = [p.v0, p.κ, p.θ, p.σ, p.ρ]

#fills the weighted differences. As an extreme parameter set may not be priced, an error results in a zero price or volatility.
function calibrationResiduals!(out, make, x, list, useVol)
    cf = DefaultCharFunc(HestonParams(x[1], x[2], x[3], x[5], x[4]))
    pricer, τ = nothing, -1.0
    for (i, o) in enumerate(list)
        if o.τ != τ
            τ = o.τ
            pricer = try
                make(cf, τ)
            catch e
                e isa InterruptException && rethrow()
                nothing
            end
        end
        price = 0.0
        if pricer !== nothing
            price = try
                CharFuncPricing.priceEuropean(pricer, calIsCall(o), o.strike, o.forward, τ, o.df)
            catch e
                e isa InterruptException && rethrow()
                0.0
            end
        end
        if useVol
            vol = try
                Black.impliedVolatility(calIsCall(o), price, o.forward, o.strike, τ, o.df)
            catch e
                e isa InterruptException && rethrow()
                0.0
            end
            out[i] = o.weight * ((isnan(vol) ? 0.0 : vol) - o.vol)
        else
            out[i] = o.weight * ((isnan(price) || price < 0 ? 0.0 : price) - o.price)
        end
    end
    return out
end

calibrationRMSE(residuals, list) = sqrt(sum(abs2, residuals) / sum(o -> o.weight^2, list))

function rmseObjective(make, list, useVol)
    residuals = zeros(length(list))
    return x -> calibrationRMSE(calibrationResiduals!(residuals, make, x, list, useVol), list)
end

## Levenberg-Marquardt on the transformed parameters y, x = (l + h)/2 + (l - h)/2 cos(y), with forward differences for the Jacobian
function calibrateLM(make, list, guess, useVol; tol=1e-8, step=sqrt(eps()), iterations=Ref(0))
    low, high = CAL_LOW_LM, CAL_HIGH_LM
    toX(y) = @. (low + high) / 2 + (low - high) / 2 * cos(y)
    y0 = @. acos(clamp((clamp($calAsVector(guess), low, high) - (low + high) / 2) / ((low - high) / 2), -1.0, 1.0))
    #the cosine has a null derivative at the bounds
    y0 = clamp.(y0, 1e-4, π - 1e-4)
    f!(out, y) = calibrationResiduals!(out, make, toX(y), list, useVol)
    function g!(J, y)
        r0, r1 = zeros(length(list)), zeros(length(list))
        f!(r0, y)
        for j in eachindex(y)
            h = step * max(abs(y[j]), 1e-3)
            yh = copy(y)
            yh[j] += h
            J[:, j] .= (f!(r1, yh) .- r0) ./ h
        end
        return J
    end
    problem = LeastSquaresOptim.LeastSquaresProblem(x=y0, f! =f!, g! =g!, J=zeros(length(list), 5), output_length=length(list))
    result = LeastSquaresOptim.optimize!(problem, LeastSquaresOptim.LevenbergMarquardt(), x_tol=tol, f_tol=tol, g_tol=tol, iterations=200)
    iterations[] = result.iterations
    return toX(result.minimizer)
end

function calibrateDE(make, list, useVol; np=10)
    problem = GlobalOptimization.Problem(5, rmseObjective(make, list, useVol), CAL_LOW_GLOBAL, CAL_HIGH_GLOBAL)
    optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(np), problem,
        MT19937(CAL_SEED), GlobalOptimization.Best1Bin(0.9, 0.5))
    GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(500, 100, 1e-3, 1e-3))
    return copy(GlobalOptimization.minimizer(optim))
end

#differential evolution on the second order small-time expansion, on volatilities
function calibrateSmallTime(list; tol=1e-6)
    residuals = zeros(length(list))
    function objective(x)
        p = HestonParams(x[1], x[2], x[3], x[5], x[4])
        τ, expansion = -1.0, nothing
        for (i, o) in enumerate(list)
            if o.τ != τ
                τ = o.τ
                expansion = VolatilityModels.LPPHestonExpansion(p, τ, order=2)
            end
            residuals[i] = o.weight * (VolatilityModels.impliedVolatility(expansion, o.strike, o.forward) - o.vol)
        end
        return calibrationRMSE(residuals, list)
    end
    problem = GlobalOptimization.Problem(5, objective, CAL_LOW_GLOBAL, CAL_HIGH_GLOBAL)
    optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(15), problem,
        MT19937(CAL_SEED), GlobalOptimization.Best1Bin(0.9, 0.5))
    GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(1000, 100, tol, tol * 1e-2))
    return copy(GlobalOptimization.minimizer(optim))
end

function calibrateSA(make, list, useVol)
    rng = MT19937(CAL_SEED)
    problem = GlobalOptimization.Problem(5, rmseObjective(make, list, useVol), CAL_LOW_GLOBAL, CAL_HIGH_GLOBAL)
    x0 = CAL_LOW_GLOBAL .+ (CAL_HIGH_GLOBAL .- CAL_LOW_GLOBAL) .* rand(rng, 5)
    optim = GlobalOptimization.SimulatedAnnealing(problem, rng)
    GlobalOptimization.optimize(optim, x0, rt=0.25, eps=1e-4, ns=5, nt=5, neps=4, maxevl=3000, t=1.0)
    return copy(GlobalOptimization.minimizer(optim))
end

#particle swarm with constriction factor and von Neumann topology: the neighbors of a particle are at ±1 and ±√S.
#A particle out of bounds is put on the boundary with a null speed and does not update the best positions.
function particleSwarm(f, low, high; swarmSize=16, maxIterations=100, c=2.1, tol=1e-4, relTol=1e-6, rng=MT19937(CAL_SEED))
    S, D = swarmSize, length(low)
    sqrtS = round(Int, sqrt(S))
    χ = 2 / abs(2 - 2c - sqrt((2c)^2 - 8c))
    vMax = abs.(high .- low) ./ 2
    x = [low .+ (high .- low) .* rand(rng, D) for _ in 1:S]
    v = [-vMax .+ 2 .* vMax .* rand(rng, D) for _ in 1:S]
    p = deepcopy(x)
    fp = [f(xi) for xi in x]
    iBest = argmin(fp)
    g, fg = copy(p[iBest]), fp[iBest]
    neighbors(i) = (mod1(i - 1, S), mod1(i + 1, S), mod1(i - sqrtS, S), mod1(i + sqrtS, S))
    nSame, maxSame = 0, max(10, maxIterations ÷ 20)
    for _ in 0:maxIterations
        previous = fg
        for i in 1:S
            local gl = p[argmin(n -> fp[n], neighbors(i))]
            inBounds = true
            for j in 1:D
                v[i][j] = χ * (v[i][j] + c * rand(rng) * (p[i][j] - x[i][j]) + c * rand(rng) * (gl[j] - x[i][j]))
                x[i][j] += v[i][j]
                if x[i][j] < low[j] || x[i][j] > high[j]
                    x[i][j] = clamp(x[i][j], low[j], high[j])
                    v[i][j] = 0.0
                    inBounds = false
                end
            end
            if inBounds
                fx = f(x[i])
                if fx < fp[i]
                    p[i] .= x[i]
                    fp[i] = fx
                    if fx < fg
                        g .= x[i]
                        fg = fx
                    end
                end
            end
        end
        fg < tol && break
        nSame = abs(previous - fg) <= relTol * fg ? nSame + 1 : 0
        nSame >= maxSame && break
    end
    return g
end

calibratePSO(make, list, useVol; kwargs...) = particleSwarm(rmseObjective(make, list, useVol), CAL_LOW_GLOBAL, CAL_HIGH_GLOBAL; kwargs...)

function calibrationRow(table, method, calibrate, make, list, useVol; n=1)
    objective = rmseObjective(make, list, useVol)
    objective([0.07, 1.4, 0.08, 0.57, -0.37]) #compilation, not part of the timing
    x, t = [0.0], Inf
    for _ in 1:n
        s = @elapsed x = calibrate()
        t = min(t, s)
    end
    return (Table=table, Method=method, v0=x[1], kappa=x[2], theta=x[3], sigma=x[4], rho=x[5], RMSE=objective(x), TimeSeconds=t)
end

const CAL_COS200 = (c, τ) -> makeCosCharFuncPricer(c, τ, 200, 8)
const CAL_COS128 = (c, τ) -> makeCosCharFuncPricer(c, τ, 128, 8)
const CAL_COS_ADAPTIVE = (c, τ) -> makeCosCharFuncPricer(c, τ, 8)
const CAL_AL = (c, τ) -> ALCharFuncPricer(c, n=200)
const CAL_FLINN_8 = (c, τ) -> FlinnCharFuncPricer(c, τ, tTol=1e-8, qTol=1e-8)

function localCalibrationRows(table, data, r, q, useVol, methods)
    list = calibrationOptions(data, r, q, useVol=useVol)
    tGuess = minimum(@elapsed(smartGuess(data)) for _ in 1:100)
    guess = smartGuess(data)
    x = calAsVector(guess)
    rows = NamedTuple[(Table=table, Method="Smart guess", v0=x[1], kappa=x[2], theta=x[3], sigma=x[4], rho=x[5],
        RMSE=rmseObjective(CAL_AL, list, useVol)(x), TimeSeconds=tGuess)]
    for (method, make) in methods
        push!(rows, calibrationRow(table, method, () -> calibrateLM(make, list, guess, useVol), make, list, useVol, n=5))
    end
    return rows
end

## Tables tbl:spx500_2010_heston_vol_de, tbl:spx500_2010_heston_vol_sa, tbl:spx500_2010_heston_vol_pso, tbl:kahale_heston_vol,
## tbl:kahale_heston_price, tbl:spx500_2010_heston_vol and tbl:spx500_2010_heston_price
@compute "heston_calibration" begin
    data1995, data2010 = loadCalibrationSurface("spx500_1995.csv"), loadCalibrationSurface("spx500_2010.csv")
    r1995, q1995, r2010, q2010 = 0.06, 0.0262, 0.02, 0.01
    rows = NamedTuple[]
    save() = writetable("heston_calibration.csv", DataFrame(rows))
    append!(rows, localCalibrationRows("kahale_vol", data1995, r1995, q1995, true, ["Cos M=200" => CAL_COS200, "Andersen-Lake" => CAL_AL, "Flinn" => CAL_FLINN_8]))
    append!(rows, localCalibrationRows("kahale_price", data1995, r1995, q1995, false, ["Cos M=200" => CAL_COS200, "Andersen-Lake" => CAL_AL, "Flinn" => CAL_FLINN_8]))
    append!(rows, localCalibrationRows("spx2010_vol", data2010, r2010, q2010, true, ["Andersen-Lake" => CAL_AL, "Flinn" => CAL_FLINN_8]))
    append!(rows, localCalibrationRows("spx2010_price", data2010, r2010, q2010, false, ["Andersen-Lake" => CAL_AL, "Flinn" => CAL_FLINN_8]))
    save()
    list = calibrationOptions(data2010, r2010, q2010)
    guess2010 = smartGuess(data2010)
    push!(rows, calibrationRow("de", "LM", () -> calibrateLM(CAL_FLINN_8, list, guess2010, true), CAL_FLINN_8, list, true, n=5))
    for (method, make) in ("Cos M=128" => CAL_COS128, "Cos-Adaptive" => CAL_COS_ADAPTIVE, "Andersen-Lake" => CAL_AL, "Flinn" => CAL_FLINN_8)
        push!(rows, calibrationRow("de", method, () -> calibrateDE(make, list, true), make, list, true))
        save()
    end
    for (method, make) in ("Cos M=200" => CAL_COS200, "Andersen-Lake" => CAL_AL, "Flinn" => CAL_FLINN_8)
        push!(rows, calibrationRow("sa", method, () -> calibrateSA(make, list, true), make, list, true))
        save()
    end
    for (table, settings) in (("pso_50", (swarmSize=50, maxIterations=100, c=2.2)), ("pso_13", (swarmSize=13, maxIterations=200, c=2.1)))
        for (method, make) in ("Andersen-Lake" => CAL_AL, "Flinn" => CAL_FLINN_8)
            push!(rows, calibrationRow(table, method, () -> calibratePSO(make, list, true; settings...), make, list, true))
            save()
        end
    end
end

## Table tbl:heston_parameters: all the methods, on Flinn with a tolerance of 1e-8. The RMSE is measured with Andersen-Lake.
## The time of a method followed by LM includes both.
@compute "heston_calibration_methods" begin
    data = loadCalibrationSurface("spx500_2010.csv")
    list = calibrationOptions(data, 0.02, 0.01)
    rmse = rmseObjective(CAL_AL, list, true)
    rmse([0.07, 1.4, 0.08, 0.57, -0.37]) #compilation, not part of the timing
    rmseObjective(CAL_FLINN_8, list, true)([0.07, 1.4, 0.08, 0.57, -0.37])
    calibrateSmallTime(list)
    asParams(x) = HestonParams(x[1], x[2], x[3], x[5], x[4])
    rows = NamedTuple[]
    add(method, x, t, iterations=0) = push!(rows, (Method=method, RMSE=rmse(x), v0=x[1], kappa=x[2], theta=x[3], sigma=x[4], rho=x[5], TimeMs=1e3t, LMIterations=iterations))
    timed(f) = (s = @elapsed x = f(); (x, s))
    best(f) = first(sort([timed(f) for _ in 1:5], by=last))
    guess = smartGuess(data)
    add("Smart", calAsVector(guess), minimum(@elapsed(smartGuess(data)) for _ in 1:100))
    small, tSmall = best(() -> calibrateSmallTime(list))
    add("Small", small, tSmall)
    n = Ref(0)
    x, t = best(() -> calibrateLM(CAL_FLINN_8, list, guess, true, iterations=n))
    add("Smart-LM", x, t, n[])
    x, t = best(() -> calibrateLM(CAL_FLINN_8, list, asParams(small), true, iterations=n))
    add("Small-LM", x, tSmall + t, n[])
    de, tDE = timed(() -> calibrateDE(CAL_FLINN_8, list, true))
    add("DE", de, tDE)
    x, t = timed(() -> calibrateSA(CAL_FLINN_8, list, true))
    add("SA", x, t)
    x, t = timed(() -> calibratePSO(CAL_FLINN_8, list, true))
    add("PSO", x, t)
    x, t = best(() -> calibrateLM(CAL_FLINN_8, list, asParams(de), true, iterations=n))
    add("DE-LM", x, tDE + t, n[])
    writetable("heston_calibration_methods.csv", DataFrame(rows))
end
