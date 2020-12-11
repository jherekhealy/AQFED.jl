struct Householder <: SRSolver end

function impliedVolatilitySRHalley(
    isCall::Bool,
    price::T,
    f::T,
    strike::T,
    tte::T,
    df::T,
    ftolrel::T,
    maxEval::Int,
    solver::Householder,
)::T where {T}
    c, ex = normalizePrice(isCall, price, f, strike, df)
    if c >= 1 / ex || c <= 0
        throw(DomainError(c, string("Price out of range, must be < ", 1 / ex, " and > 0")))
    end
    x = log(ex)
    guess = T(impliedVolatilitySRGuessUndiscountedCall(c, ex,x))
    b = guess
    xtolrel = 32 * eps(T)
    xtol = T(0)
    ftolrel = max(ftolrel, eps(T))
    ftol = ftolrel * max(1, c)
    logc = log(c)
    fb, fbOverfpb, fp2bOverfpb, fp3bOverfpb = objectiveHouseholderLog(x, ex, b, logc)
    if abs(fb) < ftol
        return b / sqrt(tte)
    end
    for iteration = 1:maxEval
        x0 = b
        lf = (fbOverfpb) * (fp2bOverfpb)
        hn = -fbOverfpb
        num = (1 + fp2bOverfpb * hn / 2)
        denom = (1 + fp2bOverfpb * hn + fp3bOverfpb / 6 * hn * hn)
        x1 = x0 + hn * num / denom
        a = x0
        fb, fbOverfpb, fp2bOverfpb, fp3bOverfpb = objectiveHouseholderLog(x, ex, x1, logc)
        b = x1
        xtol_ = xtol + max(1, abs(b)) * xtolrel
        if abs(b - a) <= xtol_ || abs(fb) < ftol
            break
        end
    end
    return b / sqrt(tte)
end

function objectiveHouseholderLog(x::T, ex::T, v::T, logc::T) where {T}
    v = abs(v)
    h = x / v
    t = v / 2
    h2 = h^2
    t2 = t^2
    sqrt2 = sqrt(T(2))
    Np = erfcx(-(h + t) / sqrt2)
    Nm = erfcx(-(h - t) / sqrt2)
    eh2t2 = exp(-(h2 + t2) / 2)
    norm = 1 / (2 * sqrt(ex)) * eh2t2
    cEstimate = norm * (Np - Nm)
    logcEstimate = log(cEstimate)
    twopi = 2 * T(pi)
    logvega = (2 /sqrt(twopi)) / (Np - Nm)  #u'/u
    volgaOverVega = (h + t) * (h - t) / v
    logvolgaOverVega = volgaOverVega - logvega  #u'' u - u'2 /u^2
    h2mt2 = h2 - t2
    c3OverVega = (-3 * h2 - t2 + h2mt2 * h2mt2) / (v^2)   #
    logc3overVega = c3OverVega - 3 * logvega * volgaOverVega + 2 * logvega * logvega
    return logcEstimate - logc, (logcEstimate - logc) / logvega, logvolgaOverVega, logc3overVega
end


function objectiveHouseholder(x::T, ex::T, v::T, c::T) where {T}
    v = abs(v)
    h = x / v
    t = v / 2
    h2 = h^2
    t2 = t^2
    sqrt2 = sqrt(T(2))
    Np = erfcx(-(h + t) / sqrt2)
    Nm = erfcx(-(h - t) / sqrt2)
    eh2t2 = exp(-(h2 + t2) / 2)
    norm = 1 / (2 * sqrt(ex)) * eh2t2
    cEstimate = norm * (Np - Nm)
    eht = exp(-h * t)
    twopi = 2 * T(pi)
    vega = eh2t2 * eht / sqrt(twopi)    #normpdf(h + t)  #this is exp(-0.5* (h^2+t^2 +2*h*t)) = eh2t2*exp(-h*t)
    cEstimateOverVega = (Np - Nm) / (2 * eht * sqrt(ex / (twopi))) - c / vega
    volgaOverVega = (h + t) * (h - t) / v
    h2mt2 = h2 - t2

    c3OverVega = (-3 * h2 - t2 + h2mt2 * h2mt2) / (v^2)
    return cEstimate - c, cEstimateOverVega, volgaOverVega, c3OverVega
end
