#Andersen-Lake for an American put that may only be exercised on a union of intervals [a_i, b_i] (e.g. blocks of
#business days), with b_n the expiry. The volatility is constant within each interval, and the gaps (week-ends, holidays)
#may carry any variance, including none. Rates are constant.
#
#On [a_i, b_i], the put is an American option with the terminal payoff h_i(x) = max(K - x, C_i(x)) at b_i, where C_i(x) is
#the value of holding over the gap (b_i, a_{i+1}) and h_n(x) = (K - x)^+. The Kim representation reads
#   V(t,S) = D(t,b_i) E[h_i(S_b)] + ∫_t^b_i [r K D(t,u) Φ(-d2) - q S Q(t,u) Φ(-d1)] du,
#where the second term of h_i at b_i is the Bermudan jump of exercise at the Friday close. As C_i is needed over a range of
#spots, the intervals are solved backwards. On each interval, the boundary B(τ) (τ = b_i - t) is found with the fixed point
#of Andersen, Lake and Offengenden, and is represented by a Chebyshev polynomial of H = ln(B/X)² in sqrt(τ), where X is the
#boundary at the end of the interval. The strike is normalized to 1.

import AQFED.Math: GaussLegendre
import LinearAlgebra: qr, ColumnNorm
export AndersenLakeIntervalRepresentation

"Chebyshev interpolant of f in ln(x) on [xMin, xMax]."
struct LogChebyshev
    logMin::Float64
    logMax::Float64
    avec::Vector{Float64}
end

function LogChebyshev(f, xMin::Float64, xMax::Float64, n::Int)
    logMin, logMax = log(xMin), log(xMax)
    values = [f(exp(logMin + (logMax - logMin) * (1 + cos((i - 1) * pi / n)) / 2)) for i = 1:n+1]
    avec = zeros(n + 1)
    updateAvec!(avec, n, values)
    return LogChebyshev(logMin, logMax, avec)
end

(c::LogChebyshev)(x) = chebQck(c.avec, 2 * (log(x) - c.logMin) / (c.logMax - c.logMin) - 1)

"""Value C(x) of holding over a gap, for a spot x at the start of the gap: the European put to the expiry, plus the
remaining early-exercise value, which is smooth and is interpolated."""
struct HoldValue
    xMin::Float64 #below, exercise at the end of the gap is certain: C(x) = df - x * qf
    xMax::Float64 #above, the early-exercise value is negligible
    df::Float64 #discount factor over the gap
    qf::Float64 #dividend discount factor over the gap
    europeanForwardFactor::Float64 #to the expiry
    europeanVariance::Float64 #to the expiry
    europeanDf::Float64 #to the expiry
    earlyExercise::LogChebyshev
end

europeanValue(c::HoldValue, x) = blackScholesFormula(false, 1.0, x * c.europeanForwardFactor, c.europeanVariance, 1.0, c.europeanDf)

function (c::HoldValue)(x)
    if x <= c.xMin
        return c.df - x * c.qf
    elseif x >= c.xMax
        return europeanValue(c, x)
    end
    return europeanValue(c, x) + c.earlyExercise(x)
end

struct ExerciseInterval
    start::Float64
    finish::Float64
    vol::Float64 #constant volatility within the interval
    kink::Float64 #h(x) = 1 - x for x < kink and h(x) = C(x) above
    X::Float64 #boundary at the end of the interval
    hold::Union{Nothing,HoldValue} #nothing for the last interval
    avec::Vector{Float64} #Chebyshev coefficients of H(z) = ln(B/X)², z = 2 sqrt(τ/L) - 1
end

struct AndersenLakeIntervalRepresentation
    r::Float64
    q::Float64
    intervals::Vector{ExerciseInterval}
    quadrature::TanhSinhQuadrature #for the integral over the interval
    gl::GaussLegendre #for the expectations of the terminal payoff
end

function boundary(iv::ExerciseInterval, τ::Float64)
    L = iv.finish - iv.start
    H = chebQck(iv.avec, 2 * sqrt(max(τ, 0.0) / L) - 1)
    return iv.X * exp(-sqrt(max(H, 0.0)))
end

"E[(1 - S) 1{S < level}] for S = F exp(√v Z - v/2)."
function putBelow(F::Float64, level::Float64, v::Float64)
    if v <= 0.0
        return F <= level ? 1 - F : 0.0
    end
    d1 = (log(F / level) + v / 2) / sqrt(v)
    return normcdf(-(d1 - sqrt(v))) - F * normcdf(-d1)
end

"E[f(S) 1{S > level}] for S = F exp(√v Z - v/2), by Gauss-Legendre on the normal variable truncated at 9 deviations."
function expectationAbove(f, F::Float64, level::Float64, v::Float64, gl::GaussLegendre)
    if v <= 0.0
        return F > level ? f(F) : 0.0
    end
    sqrtv = sqrt(v)
    zMin, zMax = max((log(level / F) + v / 2) / sqrtv, -9.0), 9.0
    if zMin >= zMax
        return 0.0
    end
    scale, shift = (zMax - zMin) / 2, (zMax + zMin) / 2
    sum = 0.0
    for (x, w) in zip(gl.x, gl.w)
        z = scale * x + shift
        sum += w * f(F * exp(sqrtv * z - v / 2)) * normpdf(z)
    end
    return scale * sum
end

"discounted expectation of the terminal payoff h(S_b) from the spot S at the time to the end τ, split into the parts
proportional to the strike, to the spot, and the holding value: (e^{-rτ} Φ(-d2), e^{-qτ} Φ(-d1), e^{-rτ} E[C 1{S_b > kink}])."
function terminalParts(p::AndersenLakeIntervalRepresentation, iv::ExerciseInterval, τ::Float64, S::Float64)
    v = iv.vol^2 * τ
    F = S * exp((p.r - p.q) * τ)
    df = exp(-p.r * τ)
    strikePart, spotPart = if v <= 0.0
        F <= iv.kink ? (df, S * exp(-p.q * τ)) : (0.0, 0.0)
    else
        d1 = (log(F / iv.kink) + v / 2) / sqrt(v)
        df * normcdf(-(d1 - sqrt(v))), exp(-p.q * τ) * normcdf(-d1)
    end
    holdPart = isnothing(iv.hold) ? 0.0 : df * expectationAbove(iv.hold, F, iv.kink, v, p.gl)
    return strikePart, spotPart, holdPart
end

"""early-exercise integrals from the spot S at the time to the end τ: ∫_0^τ r e^{-rs} Φ(-d2) ds, ∫_0^τ q e^{-qs} Φ(-d1) ds, and their
smooth-pasting counterparts ∫_0^τ r e^{-rs} φ(d2)/(σ√s) ds, ∫_0^τ q e^{-qs} φ(d1)/(σ√s) ds."""
function premiumIntegrals(p::AndersenLakeIntervalRepresentation, iv::ExerciseInterval, τ::Float64, S::Float64)
    r, q, vol = p.r, p.q, iv.vol
    sum2, sum1, sumPdf2, sumPdf1 = 0.0, 0.0, 0.0, 0.0
    for (y, w) in zip(p.quadrature.yvec, p.quadrature.wvec)
        s = τ * (1 + y) / 2 #elapsed time
        if s > 0 && s < τ
            sqrtv = vol * sqrt(s)
            d1 = (log(S / boundary(iv, τ - s)) + (r - q) * s) / sqrtv + sqrtv / 2
            d2 = d1 - sqrtv
            sum2 += w * r * exp(-r * s) * normcdf(-d2)
            sum1 += w * q * exp(-q * s) * normcdf(-d1)
            sumPdf2 += w * r * exp(-r * s) * normpdf(d2) / sqrtv
            sumPdf1 += w * q * exp(-q * s) * normpdf(d1) / sqrtv
        end
    end
    return sum2 * τ / 2, sum1 * τ / 2, sumPdf2 * τ / 2, sumPdf1 * τ / 2
end

"put value (strike 1) at the time to the end τ of the interval."
function intervalValue(p::AndersenLakeIntervalRepresentation, iv::ExerciseInterval, τ::Float64, S::Float64)
    if S <= boundary(iv, τ)
        return 1 - S
    end
    strikePart, spotPart, holdPart = terminalParts(p, iv, τ, S)
    integral2, integral1, _, _ = premiumIntegrals(p, iv, τ, S)
    return strikePart - S * spotPart + holdPart + integral2 - S * integral1
end

"value of holding over the gap from gapStart to the start of the next interval, with the variance remainingVariance to the expiry T."
function makeHoldValue(p::AndersenLakeIntervalRepresentation, gapStart::Float64, next::ExerciseInterval, gapVariance::Float64,
    T::Float64, remainingVariance::Float64, xMax::Float64, nHold::Int)
    gap = next.start - gapStart
    df, qf = exp(-p.r * gap), exp(-p.q * gap)
    Lnext = next.finish - next.start
    nextBoundary = boundary(next, Lnext)
    continuation = y -> intervalValue(p, next, Lnext, y)
    xMin = nextBoundary / exp((p.r - p.q) * gap) * exp(-9 * sqrt(gapVariance))
    holdAt = function (x)
        F = x * exp((p.r - p.q) * gap)
        return df * (putBelow(F, nextBoundary, gapVariance) + expectationAbove(continuation, F, nextBoundary, gapVariance, p.gl))
    end
    hold = HoldValue(xMin, xMax, df, qf, exp((p.r - p.q) * (T - gapStart)), remainingVariance, exp(-p.r * (T - gapStart)), LogChebyshev(x -> 0.0, xMin, xMax, 1))
    return HoldValue(xMin, xMax, df, qf, hold.europeanForwardFactor, remainingVariance, hold.europeanDf,
        LogChebyshev(x -> holdAt(x) - europeanValue(hold, x), xMin, xMax, nHold))
end

"""updated boundary from the smooth-pasting condition ∂V/∂S(B) = -1, written as the fixed point B = N(B)/D(B) (the FP-B
system of Andersen, Lake and Offengenden, extended to the terminal payoff max(1 - x, C(x)), whose holding part contributes
e^{-rτ} E[C'(S_b) S_b 1{S_b > kink}] / B to D). Unlike the value-matching system FP-A, it is not stationary below the boundary."""
function fixedPointB(p::AndersenLakeIntervalRepresentation, iv::ExerciseInterval, τ::Float64, B::Float64)
    r, q = p.r, p.q
    sqrtv = iv.vol * sqrt(τ)
    d1 = (log(B / iv.kink) + (r - q) * τ) / sqrtv + sqrtv / 2
    d2 = d1 - sqrtv
    _, integral1, integralPdf2, integralPdf1 = premiumIntegrals(p, iv, τ, B)
    N = iv.kink * exp(-r * τ) * normpdf(d2) / sqrtv + integralPdf2
    D = 1 - exp(-q * τ) * normcdf(-d1) - integral1 + exp(-q * τ) * normpdf(d1) / sqrtv + integralPdf1
    if !isnothing(iv.hold)
        F = B * exp((r - q) * τ)
        D += exp(-r * τ) * expectationAbove(y -> holdDerivative(iv.hold, y) * y, F, iv.kink, sqrtv^2, p.gl) / B
    end
    return N / D
end

"C'(x), by central differences on the Chebyshev interpolant."
function holdDerivative(c::HoldValue, x)
    h = 1e-6 * x
    return (c(x + h) - c(x - h)) / (2h)
end

"""H at the collocation points for the initial guess: the QD+ boundary of a vanilla put (strike 1), shifted by the time to
expiry τ0 at which it equals X, so that it ends at X. For an interval followed by a zero gap, this is the exact continuation."""
function initialGuess(vol, r, q, X, L, zvec, atol)
    rGuess = max(r, 1e-4) #QD+ fails for r = 0; the guess only needs to be close
    model = ConstantBlackModel(vol, rGuess, q)
    qdBoundary = τ -> τ <= 0 ? min(1.0, rGuess / max(q, eps())) : americanBoundaryPutQDP(false, model, 1.0, 1.0, τ, atol)
    τ0 = qdBoundary(1e-8) <= X ? 0.0 : find_zero(τ -> qdBoundary(τ) - X, (1e-8, 100.0), A42())
    scale = X / qdBoundary(τ0)
    ratios = [scale * qdBoundary(τ0 + L / 4 * (1 + z)^2) / X for z in zvec]
    if !all(x -> isfinite(x) && x > 0, ratios) #QD+ may fail, e.g. for r = 0: start from B = X
        return zeros(length(zvec))
    end
    return @. log(min(ratios, 1.0))^2
end

"Anderson acceleration of the fixed-point iteration x = G(x), keeping the last m differences, until |G(x) - x| ≤ tol."
function andersonAcceleration(G, x::Vector{Float64}, nIter::Int, m::Int, tol::Float64)
    g = G(x)
    dF, dG = Vector{Float64}[], Vector{Float64}[]
    for iter = 2:nIter
        f = g - x
        if maximum(abs, f) <= tol
            break
        end
        xNew = if isempty(dF)
            g
        else
            γ = qr(reduce(hcat, dF), ColumnNorm()) \ f #least squares, robust to rank deficiency
            g - reduce(hcat, dG) * γ
        end
        gNew = G(xNew)
        push!(dF, (gNew - xNew) - f)
        push!(dG, gNew - g)
        if length(dF) > m
            popfirst!(dF)
            popfirst!(dG)
        end
        x, g = xNew, gNew
    end
    return g
end

"""
    AndersenLakeIntervalRepresentation(model, intervals; nC=7, nIter=64, tol=1e-10, andersonDepth=4, nTS=31, nGL=33, nHold=64, atol=1e-8)

American put (strike 1) exercisable on the intervals `[(a_1,b_1),...,(a_n,b_n)]`, sorted, with `a_1 = 0` and `b_n` the expiry.
The variance on each interval and on each gap is taken from the total variance of the `model`, the rates are constant.
"""
function AndersenLakeIntervalRepresentation(model, intervals::AbstractVector{Tuple{Float64,Float64}};
    nC::Int=7, nIter::Int=64, tol::Float64=1e-10, andersonDepth::Int=4, nTS::Int=31, nGL::Int=33, nHold::Int=64, atol::Float64=1e-8)
    T = last(intervals)[2]
    r = -logDiscountFactor(model, T) / T
    q = r - logForward(model, 0.0, T) / T
    totalVariance = t -> varianceByLogmoneyness(model, 0.0, t) * t
    p = AndersenLakeIntervalRepresentation(r, q, ExerciseInterval[], TanhSinhQuadrature(nTS), GaussLegendre(nGL))
    xMax = exp(abs(r - q) * T + 10 * sqrt(totalVariance(T))) #holding values above are negligible
    zvec = [cos((i - 1) * pi / nC) for i = 1:nC+1]
    for i = length(intervals):-1:1
        a, b = intervals[i]
        L = b - a
        vol = sqrt((totalVariance(b) - totalVariance(a)) / L)
        hold, kink = if i == length(intervals)
            nothing, 1.0
        else
            hold = makeHoldValue(p, b, p.intervals[1], max(totalVariance(intervals[i+1][1]) - totalVariance(b), 0.0),
                T, totalVariance(T) - totalVariance(b), xMax, nHold)
            exerciseGain = x -> 1 - x - hold(x) #exercise at b iff 1 - x ≥ C(x)
            gainMin = exerciseGain(hold.xMin)
            kink = if gainMin > 0
                find_zero(exerciseGain, (hold.xMin, 1.0), A42())
            elseif gainMin == 0 || hold.df == hold.qf #no gap, or no carry over the gap
                hold.xMin
            else #crossing on the linear part of C (possible for q > r)
                (1 - hold.df) / (1 - hold.qf)
            end
            hold, kink
        end
        X = q > r ? min(kink, r / q) : kink
        H = if i == length(intervals)
            initialGuess(vol, r, q, X, L, zvec, atol)
        else #the boundary relative to its end value X has nearly the same shape as in the next interval
            next = p.intervals[1]
            [log(boundary(next, min(L / 4 * (1 + z)^2, next.finish - next.start)) / next.X)^2 for z in zvec]
        end
        #fixed-point iteration on H (the last value, at the end of the interval, is 0), with Anderson acceleration of depth m:
        #the plain iteration contracts slowly on intervals followed by a gap.
        avec = zeros(nC + 1)
        fixedPointMap = function (H)
            updateAvec!(avec, nC, H)
            iv = ExerciseInterval(a, b, vol, kink, X, hold, avec)
            return vcat([log(min(fixedPointB(p, iv, L / 4 * (1 + zvec[j])^2, boundary(iv, L / 4 * (1 + zvec[j])^2)), X) / X)^2 for j = 1:nC], 0.0)
        end
        H = andersonAcceleration(fixedPointMap, H, nIter, andersonDepth, tol)
        updateAvec!(avec, nC, H)
        pushfirst!(p.intervals, ExerciseInterval(a, b, vol, kink, X, hold, avec))
    end
    if first(intervals)[1] != 0.0
        throw(DomainError(first(intervals)[1], "the first exercise interval must start at the valuation time 0"))
    end
    return p
end

function priceAmerican(p::AndersenLakeIntervalRepresentation, K::Float64, S::Float64)::Float64
    iv = first(p.intervals)
    return K * intervalValue(p, iv, iv.finish - iv.start, S / K)
end

"exercise boundary at the times t, NaN where exercise is not allowed."
function exerciseBoundary(p::AndersenLakeIntervalRepresentation, K::Float64, t::AbstractArray{Float64})
    return map(t) do ti
        i = findfirst(iv -> iv.start <= ti <= iv.finish, p.intervals)
        isnothing(i) ? NaN : K * boundary(p.intervals[i], p.intervals[i].finish - ti)
    end
end
