#ThiopheneIV / ThiopheneIV+ Black-Scholes implied volatility solver.
#Choi-Huh-Su L3 seed, then three cubic steps on the smaller tail-log objective:
#Euler-Chebyshev on log(c) for c <= 0.5 and Halley on log(1-c) for c > 0.5.
#ThiopheneIV+ adds one Newton polish in sqrt-forward beta space for c <= 0.5.
#Port of the Java reference in mypapers/iv3/thiophene-iv.
using SpecialFunctions: erf, erfc
#Cody's erfcx is within 2 ulp near zero (SpecialFunctions up to 8), which matters near the money where the
#Euler-Chebyshev step takes the difference of two erfcx values close to one.
const thiopheneErfcx = erfcxCody
export impliedVolatilityThiophene

const THIOPHENE_SQRT_TWO = sqrt(2.0)
const THIOPHENE_SQRT_TWO_PI = sqrt(2.0 * pi)
const THIOPHENE_SQRT_PI_OVER_TWO = sqrt(pi / 2.0)
const THIOPHENE_LOG2 = log(2.0)
const THIOPHENE_LOG_2PI = log(2.0 * pi)
const THIOPHENE_MIN_TOTAL_VOL = 1e-10
const THIOPHENE_POLISH_PRICE_CUTOFF = 0.5
const THIOPHENE_NORM_INV_LOW = 0.02425
const THIOPHENE_NORM_INV_HIGH = 1.0 - THIOPHENE_NORM_INV_LOW
const THIOPHENE_TINY_PRICE_CUTOFF = 1e-6 * (1.0 + 1e-12)
const THIOPHENE_TINY_X_CUTOFF = 1e-8
const THIOPHENE_TINY_TAIL_LOG_GAP = 20.0
const THIOPHENE_TINY_TAIL_RATIO_CUTOFF = 4.0

"""
    impliedVolatilityThiophene(isCall, price, forward, strike, tte, df; plus=false)

Implied Black volatility via ThiopheneIV (`plus=false`) or ThiopheneIV+ (`plus=true`).
`price` is the discounted option price and `df` the payment discount factor.
"""
function impliedVolatilityThiophene(isCall::Bool, price::Float64, forward::Float64, strike::Float64, tte::Float64, df::Float64; plus::Bool=false)::Float64
    c = price / forward / df
    ex = forward / strike
    if !isCall
        if ex <= 1.0
            c += 1.0 - 1.0 / ex
        else
            c *= ex
            ex = 1.0 / ex
        end
    elseif ex > 1.0
        c = (forward * (c - 1.0) + strike) / strike
        ex = 1.0 / ex
    end
    if c <= 0.0
        return eps(1.0)
    end
    if c >= 1.0
        throw(DomainError(c, "price out of range"))
    end
    x = log(ex)
    logC = log(c)
    vol = thiopheneUnpolished(x, ex, c, logC, tte)
    return (plus && c <= THIOPHENE_POLISH_PRICE_CUTOFF) ? thiopheneBetaNewtonPolish(x, ex, c, tte, vol) : vol
end

import ForwardDiff
"""
    impliedVolatilityThiophene(isCall, price::ForwardDiff.Dual, forward, strike, tte, df; plus=false)

Implied volatility of a price carrying ForwardDiff partials: the value by ThiopheneIV, the partials by the implicit
function theorem, ∂σ = ∂price / vega (zero where the vega underflows).
"""
function impliedVolatilityThiophene(isCall::Bool, price::ForwardDiff.Dual{Tg,V,N}, forward::Float64, strike::Float64, tte::Float64, df::Float64; plus::Bool=false) where {Tg,V,N}
    σ = impliedVolatilityThiophene(isCall, ForwardDiff.value(price), forward, strike, tte, df; plus=plus)
    sqrtT = sqrt(tte)
    d1 = (log(forward / strike) + σ^2 * tte / 2) / (σ * sqrtT)
    vega = df * forward * exp(-d1^2 / 2) / THIOPHENE_SQRT_TWO_PI * sqrtT
    dp = ForwardDiff.partials(price)
    return ForwardDiff.Dual{Tg}(σ, vega > 0 ? dp / vega : zero(dp))
end

"""
    thiopheneTotalVolatility(x, c, logC; plus=false)

Total volatility from the normalised out-of-the-money call price `c` at log-moneyness `x = log(F/K) <= 0`, given with its
logarithm `logC` (accurate even when `c` underflows the precision of the price).
"""
function thiopheneTotalVolatility(x::Float64, c::Float64, logC::Float64; plus::Bool=false)::Float64
    ex = exp(x)
    vol = thiopheneUnpolished(x, ex, c, logC, 1.0)
    return (plus && c <= THIOPHENE_POLISH_PRICE_CUTOFF) ? thiopheneBetaNewtonPolish(x, ex, c, 1.0, vol) : vol
end

function thiopheneUnpolished(x::Float64, ex::Float64, c::Float64, logC::Float64, tte::Float64)::Float64
    tinyNearAtm = thiopheneTinyNearAtmBachelierVolatility(x, c, logC, tte)
    if isfinite(tinyNearAtm)
        return tinyNearAtm
    end
    v = thiopheneChoiL3Guess(x, ex, c)
    if !(v > THIOPHENE_MIN_TOTAL_VOL) || !isfinite(v)
        v = thiopheneFallbackGuess(x, c, logC)
    end
    if !(v > THIOPHENE_MIN_TOTAL_VOL) || !isfinite(v)
        v = THIOPHENE_MIN_TOTAL_VOL
    end
    if c > 0.5
        gap = -expm1(logC)
        targetLogGap = (gap > 0.0 && isfinite(gap)) ? log(gap) : log1p(-c)
        for _ = 1:3
            v = thiopheneHalleyLogGapStep(x, v, targetLogGap)
        end
    else
        for _ = 1:3
            v = thiopheneChebyshevLogPriceStep(x, v, logC)
        end
    end
    return v / sqrt(tte)
end

function thiopheneChebyshevLogPriceStep(x::Float64, v::Float64, logC::Float64)::Float64
    h = x / v
    t = v / 2.0
    diff = thiopheneErfcx(-(h + t) / THIOPHENE_SQRT_TWO) - thiopheneErfcx(-(h - t) / THIOPHENE_SQRT_TWO)
    if !(diff > 0.0) || !isfinite(diff)
        return v
    end
    logCEst = x == 0.0 ? log(erf(t / THIOPHENE_SQRT_TWO)) : -(h * h + t * t) / 2.0 - THIOPHENE_LOG2 - x / 2.0 + log(diff)
    if !isfinite(logCEst)
        return v
    end
    logVega = (2.0 / THIOPHENE_SQRT_TWO_PI) / diff
    d2 = (h + t) * (h - t) / v - logVega
    step = -(logCEst - logC) / logVega
    next = v + step * (1.0 - 0.5 * d2 * step)
    return (next > 0.0 && isfinite(next)) ? next : v
end

function thiopheneHalleyLogGapStep(x::Float64, v::Float64, targetLogGap::Float64)::Float64
    d1 = x / v + 0.5 * v
    b1 = d1 / THIOPHENE_SQRT_TWO
    b2 = -(d1 - v) / THIOPHENE_SQRT_TWO
    erfcxSum = thiopheneErfcx(b1) + thiopheneErfcx(b2)
    if !(erfcxSum > 0.0) || !isfinite(erfcxSum)
        return v
    end
    logGap = -b1 * b1 - THIOPHENE_LOG2 + log(erfcxSum)
    if !isfinite(logGap)
        return v
    end
    newtonStep = 0.5 * THIOPHENE_SQRT_TWO_PI * erfcxSum * (logGap - targetLogGap)
    d1Prime = 0.5 - x / (v * v)
    secondOverFirst = -d1 * d1Prime + 2.0 / (THIOPHENE_SQRT_TWO_PI * erfcxSum)
    denominator = 1.0 + 0.5 * secondOverFirst * newtonStep
    if !(denominator > 0.0) || !isfinite(denominator)
        return v
    end
    next = v + newtonStep / denominator
    return (next > 0.0 && isfinite(next)) ? next : v
end

function thiopheneChoiL3Guess(x::Float64, ex::Float64, c::Float64)::Float64
    k = max(-x, 0.0)
    if k == 0.0
        return c < 1e-4 ? THIOPHENE_SQRT_TWO_PI * c * (1.0 + pi * c * c / 12.0) : 2.0 * thiopheneNormInv(0.5 * (1.0 + c))
    end
    ek = 1.0 / ex
    if !(ek > 0.0) || !isfinite(ek)
        ek = exp(k)
    end
    denominator = 2.0 * c + ek - 1.0
    if !(denominator > 0.0) || !isfinite(denominator)
        return NaN
    end
    z = thiopheneNormInv(c * (c + ek) / denominator)
    root = sqrt(z * z + 2.0 * k)
    return z >= 0.0 ? z + root : 2.0 * k / (root - z)
end

thiopheneClampProbability(p::Float64) = min(max(p, nextfloat(0.0)), prevfloat(1.0))

#Acklam's rational approximation: only used for the seed.
function thiopheneNormInv(probability::Float64)::Float64
    p = thiopheneClampProbability(probability)
    if p < THIOPHENE_NORM_INV_LOW
        return thiopheneTailNormInv(sqrt(-2.0 * log(p)))
    elseif p > THIOPHENE_NORM_INV_HIGH
        return -thiopheneTailNormInv(sqrt(-2.0 * log1p(-p)))
    end
    q = p - 0.5
    r = q * q
    num = (((((-39.69683028665376 * r + 220.9460984245205) * r - 275.9285104469687) * r + 138.3577518672690) * r - 30.66479806614716) * r + 2.506628277459239) * q
    den = (((((-54.47609879822406 * r + 161.5858368580409) * r - 155.6989798598866) * r + 66.80131188771972) * r - 13.28068155288572) * r + 1.0)
    return num / den
end

function thiopheneTailNormInv(q::Float64)::Float64
    return (((((-0.007784894002430293 * q - 0.3223964580411365) * q - 2.400758277161838) * q - 2.549732539343734) * q + 4.374664141464968) * q + 2.938163982698783) /
           ((((0.007784695709041462 * q + 0.3224671290700398) * q + 2.445134137142996) * q + 3.754408661907416) * q + 1.0)
end

function thiopheneFallbackGuess(x::Float64, c::Float64, logC::Float64)::Float64
    if c < 1e-4 && abs(x) < 0.01
        atmVol = THIOPHENE_SQRT_TWO_PI * c
        return sqrt(x * x + atmVol * atmVol)
    end
    d = sqrt(max(-2.0 * logC - THIOPHENE_LOG_2PI, 0.0))
    disc = d * d - 2.0 * x
    if disc > 0.0 && d > 0.0
        return -2.0 * x / (d + sqrt(disc))
    end
    return sqrt(2.0 * abs(x))
end

function thiopheneBetaNewtonPolish(x::Float64, ex::Float64, c::Float64, tte::Float64, vol::Float64)::Float64
    if !(vol > 0.0) || !(tte > 0.0) || !(ex > 0.0) || !(c > 0.0) || !isfinite(vol) || !isfinite(tte)
        return vol
    end
    sqrtEx = sqrt(ex)
    targetBeta = c * sqrtEx
    if !(targetBeta > 0.0) || !(targetBeta < sqrtEx) || !isfinite(targetBeta)
        return vol
    end
    sqrtTte = sqrt(tte)
    s = vol * sqrtTte
    beta = normalised_black_call(x, s)
    vega = normalised_vega(x, s)
    if !(beta > 0.0) || !(vega > 0.0) || !isfinite(beta) || !isfinite(vega)
        return vol
    end
    next = s - (beta - targetBeta) / vega
    return (next > 0.0 && isfinite(next)) ? next / sqrtTte : vol
end

#Tiny prices near the money: Black behaves like Bachelier in sqrt-forward units.
function thiopheneTinyNearAtmBachelierVolatility(x::Float64, c::Float64, logC::Float64, tte::Float64)::Float64
    if !(c > 0.0) || x > 0.0 || c > THIOPHENE_TINY_PRICE_CUTOFF || abs(x) > THIOPHENE_TINY_X_CUTOFF
        return NaN
    end
    m = -x
    beta = c * exp(0.5 * x)
    v = thiopheneTinyBachelierDeepTailVolatility(m, logC + 0.5 * x)
    if isfinite(v)
        return v / sqrt(tte)
    end
    v = thiopheneLfk2026BachelierVolatility(m, beta)
    if isfinite(v) && v > 0.0
        return thiopheneTinyBlackExpansionPolish(m, beta, v) / sqrt(tte)
    end
    return 0.0
end

function thiopheneTinyBlackExpansionPolish(m::Float64, beta::Float64, v::Float64)::Float64
    if !(beta > 0.0) || !(v > 0.0) || !isfinite(v)
        return v
    end
    if m > 0.0 && m / v > THIOPHENE_TINY_TAIL_RATIO_CUTOFF
        return v
    end
    for _ = 1:2
        a = m == 0.0 ? 0.0 : m / v
        derivative = exp(-0.5 * a * a) / THIOPHENE_SQRT_TWO_PI * exp(-0.125 * v * v)
        if !(derivative > 0.0) || !isfinite(derivative)
            break
        end
        step = (thiopheneTinyBlackSqrtNormalisedExpansion(m, v) - beta) / derivative
        v1 = v - step
        if !(v1 > 0.0) || !isfinite(v1)
            break
        end
        v = v1
        if abs(step) <= 2.0 * eps(v)
            break
        end
    end
    return v
end

function thiopheneTinyBlackSqrtNormalisedExpansion(m::Float64, v::Float64)::Float64
    a = m == 0.0 ? 0.0 : m / v
    phi = exp(-0.5 * a * a) / THIOPHENE_SQRT_TWO_PI
    q = m == 0.0 ? 0.5 : 0.5 * erfc(a / THIOPHENE_SQRT_TWO)
    i0 = v * phi - m * q
    v2 = v * v
    m2 = m * m
    i2 = (v2 * v * phi - m2 * i0) / 3.0
    i4 = (v2 * v2 * v * phi - m2 * i2) / 5.0
    return i0 - 0.125 * i2 + i4 / 128.0
end

function thiopheneTinyBachelierDeepTailVolatility(k::Float64, logC::Float64)::Float64
    if !(k > 0.0) || !isfinite(logC)
        return NaN
    end
    if log(k) - logC <= THIOPHENE_TINY_TAIL_LOG_GAP
        return NaN
    end
    targetLogRatio = logC - log(k)
    tailRatio = sqrt(max(1.0, -2.0 * (targetLogRatio + 0.5 * THIOPHENE_LOG_2PI)))
    for _ = 1:4
        millsRatio = THIOPHENE_SQRT_PI_OVER_TWO * thiopheneErfcx(tailRatio / THIOPHENE_SQRT_TWO)
        scaledGap = 1.0 / tailRatio - millsRatio
        if !(scaledGap > 0.0) || !isfinite(scaledGap)
            return NaN
        end
        residual = -0.5 * tailRatio * tailRatio - 0.5 * THIOPHENE_LOG_2PI + log(scaledGap) - targetLogRatio
        slope = -tailRatio + (1.0 - 1.0 / (tailRatio * tailRatio) - tailRatio * millsRatio) / scaledGap
        step = residual / slope
        next = tailRatio - step
        if !(next > 0.0) || !isfinite(next)
            return NaN
        end
        tailRatio = next
        if abs(step) <= 1e-14 * max(1.0, tailRatio)
            break
        end
    end
    v = k / tailRatio
    return k / v > THIOPHENE_TINY_TAIL_RATIO_CUTOFF ? v : NaN
end

const THIOPHENE_LFK_ALPHA = 0.15
const THIOPHENE_LFK_BETA_S = -log(THIOPHENE_LFK_ALPHA)
const THIOPHENE_LFK_BETA_E = 300.0 * log(10.0)
const THIOPHENE_LFK_INV_BETA_RANGE = 1.0 / (THIOPHENE_LFK_BETA_E - THIOPHENE_LFK_BETA_S)

function thiopheneLfk2026BachelierVolatility(k::Float64, c::Float64)::Float64
    if !(c > 0.0) || !(k > 0.0)
        return (k == 0.0 && c > 0.0) ? THIOPHENE_SQRT_TWO_PI * c : NaN
    end
    zPos = c / k
    if zPos > THIOPHENE_LFK_ALPHA
        u = k / c
        p = evalpoly(u, (2.50662827463100024e+00, 7.32071451154750097e+00, 8.38783649295939249e+00, 4.81532474104199881e+00, 1.47171929792136824e+00,
            2.36012991833862845e-01, 1.83954491376296485e-02, 5.87928493313025560e-04, 5.03803837400379678e-06, -2.70797042551215202e-09))
        q = evalpoly(u, (1.0, 3.42054254140456848e+00, 4.63611136038634353e+00, 3.18165296194859026e+00, 1.17534367470784673e+00,
            2.30574991341749191e-01, 2.23188636991579585e-02, 9.08750187647611803e-04, 1.06715338537575127e-05))
        return (c + k) * p / q
    end
    if zPos < 1e-300
        return 0.0
    end
    eta = -(log(zPos) + THIOPHENE_LFK_BETA_S) * THIOPHENE_LFK_INV_BETA_RANGE
    invAbsD = if eta < 0.01
        evalpoly(eta, (1.24910559446641112e+00, 6.82178299087715914e+02, 1.95717866058198881e+05, 3.07861092585915886e+07, 2.40533073130836535e+09,
            -5.10817469063371429e+10, -2.85331260987706523e+13, -2.49477863322856750e+15, -7.43714349945829120e+16, -4.02413716168378048e+17,
            1.10299370004290995e+18, -3.48308852288966554e+18)) /
        evalpoly(eta, (1.0, 8.31825318775392361e+02, 3.22084448691854079e+05, 7.18091511301956624e+07, 9.14684611768276405e+09,
            4.46201521087863831e+11, -4.73283686242051484e+13, -9.16374230029381000e+15, -5.43691399201888000e+17, -8.65868218927210598e+18))
    elseif eta < 0.09
        evalpoly(eta, (1.24860495497317925e+00, 8.82143141269435318e+02, 2.57450335132154607e+05, 4.12660599897888750e+07, 3.45074450506824827e+09,
            1.39249460078530365e+11, 2.58076177636243115e+12, 2.03806992766065625e+13, 5.73825498368429922e+13, 3.09426714906955234e+13,
            -8.76625536254249121e+12, 2.99147136389716260e+12)) /
        evalpoly(eta, (1.0, 9.91194200989211254e+02, 4.17679926778023364e+05, 9.56102882581782639e+07, 1.24794454367125645e+10,
            7.95484765720863647e+11, 2.31200993253629883e+13, 2.92823593402229500e+14, 1.45588799532258475e+15, 2.10719786854519525e+15))
    else
        evalpoly(eta, (1.61947615412173973e+00, 1.34977888075358550e+02, 1.53669112770474931e+03, -3.63727433825341577e+04, -6.78815730118479463e+05,
            -3.62091305895698769e+06, -7.65563194503842667e+06, -6.49374308713343088e+06, -1.90438009836981818e+06, -1.04999322668089051e+05,
            2.99524578014802091e+03, -1.01865471163877160e+02)) /
        evalpoly(eta, (1.0, 5.91576136257250369e+02, 2.27524257249606417e+04, -3.44836595974720476e+04, -6.85415582536362950e+06,
            -6.44309250732882321e+07, -2.16291778562989831e+08, -2.93510878971269727e+08, -1.51588857602825403e+08, -2.24196883894425444e+07))
    end
    return k * invAbsD
end
