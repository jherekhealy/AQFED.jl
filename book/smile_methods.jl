# Smile interpolation methods of chapter 7, wrapped with a common interface to evaluate the implied volatility and
# the probability density on a fine grid of strikes. Used by the ch07_* figure files.

using AQFED, AQFED.Black, AQFED.Collocation, AQFED.PDDE, AQFED.Math, AQFED.VolatilityModels, ForwardDiff, PPInterpolation
import AQFED.VolatilityModels: priceEuropean
isdefined(@__MODULE__, :SmileData) || include(joinpath(@__DIR__, "..", "test", "smile_data.jl"))

"""
A calibrated smile. `price(isCall, strike)` is the undiscounted option price, `density(strike)` the probability density
of the strike. `otm`: the implied volatility is computed from out-of-the-money options instead of calls.
`vol(strike)`, when given, is used instead of inverting the price.
"""
struct SmileModel
    price::Function
    density::Function
    otm::Bool
    vol::Union{Nothing,Function}
end
SmileModel(price, density; otm=false, vol=nothing) = SmileModel(price, density, otm, vol)

function impliedVol(m::SmileModel, d::SmileData, k)
    m.vol === nothing || return m.vol(k)
    isCall = m.otm ? k >= d.forward : true
    try
        return Black.impliedVolatility(isCall, max(m.price(isCall, k), 1e-32), d.forward, k, d.tte, 1.0)
    catch
        return NaN
    end
end

#density from the second derivative of the call price
d2CallPrice(price, k) = ForwardDiff.derivative(y -> ForwardDiff.derivative(x -> price(true, x), y), k)

## builders: each returns a SmileModel

function expBSplineModel(d::SmileData, prices, w; size=10, rawFit=false, kwargs...)
    b, _ = Collocation.makeExpBSplineCollocation(d.strikes, prices, w, d.tte, d.forward, 1.0; penalty=0.0, size=size, minSlope=1e-8, rawFit=rawFit, kwargs...)
    return SmileModel((c, k) -> Collocation.priceEuropean(b, c, k, d.forward, 1.0), k -> Collocation.density(b, k))
end

function cubicBSplineModel(d::SmileData, prices, w; size=10, rawFit=false, kwargs...)
    b, _ = Collocation.makeBSplineCollocation(d.strikes, prices, w, d.tte, d.forward, 1.0; penalty=0.0, size=size, minSlope=1e-8, rawFit=rawFit, N=3, kwargs...)
    return SmileModel((c, k) -> Collocation.priceEuropean(b, c, k, d.forward, 1.0), k -> Collocation.density(b, k))
end

lvgDensity(l, k) = (PDDE.derivativePrice(l, true, k + 1e-4) - PDDE.derivativePrice(l, true, k)) * 1e4

function quadraticLVGModel(d::SmileData, prices, w; otm=false, kwargs...)
    l = PDDE.calibrateQuadraticLVG(d.tte, d.forward, d.strikes, prices, w; model=PDDE.Quadratic(), kwargs...)
    return SmileModel((c, k) -> PDDE.priceEuropean(l, c, k), k -> lvgDensity(l, k), otm=otm)
end

function linearBlackLVGModel(d::SmileData, prices, w; kwargs...)
    l = PDDE.calibrateLinearBlackLVG(d.tte, d.forward, d.strikes, prices, w; useVol=true, L=d.strikes[1] / 2, U=d.strikes[end] * 2, kwargs...)
    return SmileModel((c, k) -> PDDE.priceEuropean(l, c, k), k -> lvgDensity(l, k))
end

#Andreasen-Huge with a quadratic B-spline representation (discrete log LVG) on `discreteSize` grid points
function andreasenHugeModel(d::SmileData, prices, w; discreteSize)
    ah = PDDE.calibrateDiscreteLogLVG(d.tte, d.forward, d.strikes, prices, w, useVol=true, model=PDDE.Quadratic(), location="Equidistributed",
        optimizer="GN-MQ", L=d.strikes[1] / 2, U=d.strikes[end] * 2, isC3=true, size=10, discreteSize=discreteSize)
    price(c, k) = PDDE.priceEuropean(ah, c, k)
    return SmileModel(price, k -> d2CallPrice(price, k), otm=true)
end

#Andreasen-Huge as above, with quadratic Lagrange interpolation of the discrete prices in between the grid points
function andreasenHugeLagrangeModel(d::SmileData, prices, w; discreteSize)
    ah = PDDE.calibrateDiscreteLogLVG(d.tte, d.forward, d.strikes, prices, w, useVol=true, model=PDDE.Quadratic(), location="Equidistributed",
        optimizer="GN-MQ", L=d.strikes[1] / 2, U=d.strikes[end] * 2, isC3=true, size=10, discreteSize=discreteSize,
        interpolation=PDDE.QuadraticLagrangePriceInterpolation())
    price(c, k) = PDDE.priceEuropean(ah, c, k)
    return SmileModel(price, k -> d2CallPrice(price, k), otm=true)
end

function quadraticBSplineModel(d::SmileData, prices, w; size=10, rawFit=false, kwargs...)
    b, _ = Collocation.makeBSplineCollocation(d.strikes, prices, w, d.tte, d.forward, 1.0; penalty=0.0, size=size, minSlope=1e-8, rawFit=rawFit, N=2, kwargs...)
    return SmileModel((c, k) -> Collocation.priceEuropean(b, c, k, d.forward, 1.0), k -> Collocation.density(b, k))
end

function schabackModel(cs)
    return SmileModel((c, k) -> c ? cs(k) : cs(k) - (cs(0.0) - k), k -> Math.evaluateSecondDerivative(cs, k))
end

#convex rational spline exact on the convex subset of the quotes, with extra points at 0 and far out-of-the-money
function schabackExact(d::SmileData, strikes, prices; upper)
    return Math.makeConvexSchabackRationalSpline(vcat(0.0, strikes, upper), vcat(d.forward, prices, 0.0),
        Math.FirstDerivativeBoundary(-1.0), Math.FirstDerivativeBoundary(0.0), iterations=128)
end

#least-squares convex rational spline with a smoothing penalty
function schabackFit(d::SmileData, strikes, prices, w; upper, penalty)
    cs, _ = Math.fitConvexSchabackRationalSpline(vcat(0.0, strikes, upper), vcat(d.forward, prices, 0.0), vcat(sum(w), w, sum(w)),
        Math.FirstDerivativeBoundary(-1.0), Math.FirstDerivativeBoundary(0.0); penalty=penalty)
    return cs
end

#exact rational spline on 8 knots taken from the convex subset of the quotes
function schaback8Knots(d::SmileData, strikesf, pricesf; upper)
    step = floor(Int, length(strikesf) / 7)
    subset = collect(1:step:length(strikesf))
    subset[end] != length(strikesf) && push!(subset, length(strikesf))
    return schabackExact(d, strikesf[subset], pricesf[subset], upper=upper)
end

function fenglerModel(d::SmileData, prices, w; λ, eps=1e-7, strikes=d.strikes, solver="GI")
    f = VolatilityModels.calibrateFenglerSlice(d.tte, d.forward, strikes, prices, w, λ=λ, eps=eps, solver=solver)
    return SmileModel((c, k) -> c ? max(f(k), 1e-16) : max(f(k), 1e-16) - (d.forward - k), k -> VolatilityModels.evaluateSecondDerivative(f, k))
end

#smile defined by a total variance function of the log-moneyness y
function varianceModel(d::SmileData, variance)
    price(c, k) = Black.blackScholesFormula(c, k, d.forward, variance(log(k / d.forward)) * d.tte, 1.0, 1.0)
    return SmileModel(price, k -> d2CallPrice(price, k), vol=k -> sqrt(variance(log(k / d.forward))))
end

function rbfModel(d::SmileData, weights; knots)
    rbf = VolatilityModels.calibrateMultiquadric(d.tte, d.forward, logmoneyness(d), d.vols, weights, knots=knots)
    return varianceModel(d, y -> rbf(y)), rbf
end

function mixtureModel(kernel)
    return SmileModel((c, k) -> priceEuropean(kernel, c, k), k -> VolatilityModels.density(kernel, k), otm=true)
end

## evaluation

"""
    smileCurves(d, models; n=1001, lower=d.strikes[1], upper=d.strikes[end])

DataFrame with the implied volatility and the density of each model (a vector of name => SmileModel) on n strikes.
"""
function smileCurves(d::SmileData, models; n=1001, lower=d.strikes[1], upper=d.strikes[end])
    k = exp.(range(log(lower), stop=log(upper), length=n))
    df = DataFrame(Model=String[], Logmoneyness=Float64[], Strike=Float64[], Vol=Float64[], Density=Float64[])
    for (name, m) in models
        dens = [try m.density(ki) catch; NaN end for ki in k]
        append!(df, DataFrame(Model=name, Logmoneyness=log.(k ./ d.forward), Strike=k, Vol=impliedVol.(Ref(m), Ref(d), k), Density=dens))
    end
    return df
end

## figures

"""
    smileVolFigure(dataset, models; labels, reference=true, ...)

Market quotes and implied volatilities of the models (names of the curves computed for `dataset`) against the log-moneyness.
"""
function smileVolFigure(dataset, models; labels=Dict(), reference=true, size=(480, 380), xlims=nothing, ylims=nothing,
    legendposition=:rt, offset=0, referenceAlpha=0.5, colors=Dict())
    curves = readcsv("smile_$(dataset).csv")
    fig = bookFigure(size=size)
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Volatility in %")
    if reference
        d = loadSmile(dataset)
        referenceScatter!(ax, logmoneyness(d), d.vols .* 100)
    end
    c = rows(curves, :Model, models...)
    groupLines!(ax, c.Logmoneyness, c.Vol .* 100, c.Model, order=collect(models), labels=labels, offset=offset, colors=colors)
    xlims === nothing || xlims!(ax, xlims...)
    ylims === nothing || ylims!(ax, ylims...)
    bookLegend!(ax, position=legendposition)
    return fig
end

#ticks at integer powers of ten (at most 6), labelled 10ⁿ
function decadeTicks(lo, hi)
    a, b = ceil(Int, log10(lo)), floor(Int, log10(hi))
    step = max(1, cld(b - a, 5))
    return 10.0 .^ (a:step:b)
end
powerLabels(values) = [Makie.rich("10", Makie.superscript(string(round(Int, log10(v))))) for v in values]

function smileDensityFigure(dataset, models; labels=Dict(), size=(480, 380), logscale=false, ylims=nothing, xlims=nothing,
    legendposition=:rt, offset=0, legend=true, colors=Dict())
    curves = readcsv("smile_$(dataset).csv")
    c = rows(curves, :Model, models...)
    logscale && (c = c[c.Density.>0, :])
    fig = bookFigure(size=size)
    ax = bookAxis(fig[1, 1], xlabel="Forward log-moneyness", ylabel="Probability density", yscale=logscale ? log10 : identity,
        yticks=logscale ? decadeTicks : Makie.automatic, ytickformat=logscale ? powerLabels : Makie.automatic)
    groupLines!(ax, c.Logmoneyness, c.Density, c.Model, order=collect(models), labels=labels, offset=offset, colors=colors)
    xlims === nothing || xlims!(ax, xlims...)
    ylims === nothing || ylims!(ax, ylims...)
    legend && bookLegend!(ax, position=legendposition)
    return fig
end
