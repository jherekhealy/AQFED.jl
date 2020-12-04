const polya_factor = 2/pi #polya is 2 / pi
#const polya_factor = sqrt(pi/8) #this is aludaat, 


function normalizePrice(isCall::Bool, price::T, f::T, strike::T, df::T) where {T}
    c = price / f / df
    ex = f / strike

    if !isCall
        if ex <= 1
            c = c + 1 - 1 / ex # put call parity
        else
            #else duality + put call parity
            c = ex * c
            ex = 1 / ex
        end
    else
        if ex > 1
            # use c(-x0, v)
            c = (f * (c - 1) + strike) / strike # in out duality, c = ex*c + 1 - ex  //not as good numericall
            ex = 1 / ex
        end
    end
    return c, ex
end

function impliedVolatilitySRGuess(isCall::Bool, price::Real, f::Real, strike::Real, tte::Real, df::Real)::Real
    c,ex = normalizePrice(isCall, price, f, strike, df)
    scaledVol = impliedVolatilitySRGuessUndiscountedCall(c, ex, log(ex))
    return scaledVol / sqrt(tte)
end

# f = 1.0, strike = 1/ex, ex=ey = 1.0/strike=f/strike, y = log(f/strike) = log(ey), alpha = price*ex
function impliedVolatilitySRGuessUndiscountedCall(price::Real, ey::Real, y::Real)::Real
    alpha = price *ey
    r = 2 * alpha - ey + 1
    em2piy = exp(-polya_factor * y)
    A = (ey * em2piy - 1.0 / (ey * em2piy))^2
    r2 = r * r
    B = 4 * (em2piy + 1.0 / em2piy) - 2 / ey * (ey * em2piy + 1.0 / (ey * em2piy)) * ((ey * ey) + 1 - r2)
    beta = 0.0
    if abs(alpha) < SqrtEpsilon
        C = -16 * (1 - 1.0 / ey) * alpha - 16 * (1 - 3 / ey + 1 / (ey * ey)) * alpha * alpha
        beta = C / B
        #    println(" alpha ",alpha, " C ",C," beta ",beta)
    elseif abs(ey - alpha) < SqrtEpsilon
        C = -16 * (1 + 1.0 / ey) * (alpha - ey) - 16 * (1 + 3 / ey + 1 / (ey * ey)) * (alpha - ey) * (alpha - ey)
        if C == 0
            beta = B / A #this is wrong sign but gives order of magnitude
        else
            beta = C / B
        end
        #    println(" alpha ",alpha, " C ",C," beta ",beta)
    elseif abs(y) < SqrtEpsilon
        a2 = alpha * alpha
        a4 = a2 * a2
        C = 16 * ((a2 - a4) + (2 * a4 + 2 * alpha * a2 - a2 - alpha) * y)
        B = 16 * (a2 - y * (alpha + a2))
        beta = C / B
    else
        eym1 = ey - 1
        eyp1 = ey + 1
        C = 1.0 / (ey * ey) * (r2 - eym1 * eym1) * (eyp1 * eyp1 - r2)
        beta = 2 * C / (B + sqrt(B * B + 4 * A * C))
        # println("beta ",beta, " ",C, " ",B, " ",A)
    end
    gamma = -log(beta) / polya_factor
    if y >= 0
        Asqrty = 0.5 * (1 + sqrt(1 - em2piy * em2piy))
        c0 = (Asqrty - 0.5/ey)
        gmy = gamma - y
        if gmy < 0
            #machine epsilon issues
            gmy = 0.0
        end
        if price <= c0
            return sqrt(gamma + y) - sqrt(gmy)
        end
        return (sqrt(gamma + y) + sqrt(gmy))
    end
    Asqrty = 0.5 * (1 - sqrt(1 - 1.0 / (em2piy * em2piy)))
    c0 = (0.5 - Asqrty/ey)
    gpy = gamma + y
    if gpy < 0
        #machine epsilon issues
        gpy = 0.0
    end
    if price <= c0
        return (-sqrt(gpy) + sqrt(gamma - y))
    end
    return (sqrt(gpy) + sqrt(gamma - y))
end

abstract type SRSolver end
struct Halley <: SRSolver end
struct SuperHalley <: SRSolver end
struct InverseQuadratic <: SRSolver end
struct CMethod <: SRSolver end

# 1 SuperHalley = 2 InverseQuadratic = 3 CMethod = 4
function impliedVolatilitySRHalley(
    isCall::Bool,
    price::T,
    f::T,
    strike::T,
    tte::T,
    df::T,
    ftolrel::T,
    maxEval::Int,
    solver::SRSolver,
)::T where {T}
    c, ex = normalizePrice(isCall, price, f, strike, df)
        #println(" c ",c, " 1/ex ",1/ex)
    if (c >= min(1, 1 / ex)) || (c <= 0)
        throw(DomainError(c, string("Price out of range, must be < ", min(1, 1 / ex), " and > 0")))
    end
    x = log(ex)
    guess = T(impliedVolatilitySRGuessUndiscountedCall(c, ex, x))
    b = guess
    xtolrel = 32 * eps(T) * max(1, guess)
    xtol = 0 * eps(T)
    ftolrel = max(ftolrel, eps(T))
    ftol_ = max(1, abs(c)) * ftolrel
    useLog = true
    local fb, fbOverfpb, fp2bOverfpb
    if useLog
        c = log(c)
        fb, fbOverfpb, fp2bOverfpb = objectiveLog(x, ex, b, c)
    else
        fb, fbOverfpb, fp2bOverfpb = objective(x, ex, b, c)
    end

    if abs(fb) < ftol_
        return b / sqrt(tte)
    end
    for iteration = 1:maxEval
        x0 = b
        lf = fbOverfpb * fp2bOverfpb
        x1 = x0 + srSolve(solver, fb, fbOverfpb, lf)
        a = x0
        if useLog
            fb, fbOverfpb, fp2bOverfpb = objectiveLog(x, ex, x1, c)
        else
            fb, fbOverfpb, fp2bOverfpb = objective(x, ex, b, c)
        end
        # println(iteration, " ", fb, " ", fbOverfpb, " ", fp2bOverfpb, " ", x1, " ", x0, " ", lf)
        b = x1
        xtol_ = xtol + max(1, abs(b)) * xtolrel

        if abs(b - a) <= xtol_ || abs(fb) <= ftol_
            #value = b
            break
        end
    end
    return b / sqrt(tte)
end

function srSolve(solver::Halley, fb::T, fbOverfpb::T, lf::T)::T where {T}
    return -(1 / (1 - lf / 2)) * fbOverfpb
end
function srSolve(solver::SuperHalley, fb::T, fbOverfpb::T, lf::T)::T where {T}
    return -(1 + lf / (2 * (1 - lf))) * fbOverfpb #super halley faster for med low vols/more OTM, almost similar for ATM, but unstable for very small fb
end
function srSolve(solver::InverseQuadratic, fb::T, fbOverfpb::T, lf::T)::T where {T}
    return -(1 + lf / 2) * fbOverfpb #inverse quadratic
end
function srSolve(solver::CMethod, fb::T, fbOverfpb::T, lf::T)::T where {T}
    return -(1 + lf / 2 + lf * lf / 2) * fbOverfpb  #C method seems more balanced perf
end

function objectiveLog(x::T, ex::T, v::T, logc::T) where {T}
    v = abs(v)
    h = x / v
    t = v / 2
    # naive implementation:
    # Np = normcdf(h+t)
    # Nm = normcdf(h-t)/ex
    # norm = T(1)
    # cEstimate = norm * (Np - Nm)
    # vega = normpdf(h+t)
    # volgaOverVega = (h + t) * (h - t) / v
    # logcEstimate = log(cEstimate)
    # logvega = vega/cEstimate #u'/u
    #logVolga = (volgaOverVega*logvega - logvega*logvega)#u'' u - u'2 /u^2

    # the following actually helps convergence and accuracy (up to a certain machine epsilon point):
    sqrt2 = sqrt(T(2))
    Np = erfcx(-(h + t) / sqrt2)
    Nm = erfcx(-(h - t) / sqrt2)
    eh2t2 = exp(-(h * h + t * t) / 2)
    norm = 1 / (2 * sqrt(ex)) * eh2t2
    cEstimate = norm * (Np - Nm)
    logcEstimate = log(cEstimate)
    twopi = 2 * T(pi)
    logvega = (2 /sqrt(twopi)) / (Np - Nm)  #u'/u
    volgaOverVega = (h + t) * (h - t) / v
    logvolgaOverVega = volgaOverVega - logvega
    return logcEstimate - logc, (logcEstimate - logc) / logvega, logvolgaOverVega
end
function objective(x::T, ex::T, v::T, c::T) where {T}
    v = abs(v)
    h = x / v
    t = v / 2
    # naive implementation:
    # Np = normcdf(h+t)
    # Nm = normcdf(h-t)/ex
    # norm = T(1)
    # cEstimate = norm * (Np - Nm)
    # vega = normpdf(h+t)
    # cEstimateOverVega = (cEstimate-c)/vega
    # volgaOverVega = (h + t) * (h - t) / v
    # the following actually helps convergence and accuracy (up to a certain machine epsilon point):
    sqrt2 = sqrt(T(2))
    Np = erfcx(-(h + t) / sqrt2)
    Nm = erfcx(-(h - t) / sqrt2)
    eh2t2 = exp(-(h * h + t * t) / 2)
    norm = 1 / (2 * sqrt(ex)) * eh2t2
    cEstimate = norm * (Np - Nm)
    eht = exp(-h * t)
    twopi = 2 * T(pi)
    vega = eh2t2 * eht / sqrt(twopi)    #normpdf(h + t)  #this is exp(-0.5* (h^2+t^2 +2*h*t)) = eh2t2*exp(-h*t)
    cEstimateOverVega = (Np - Nm) / (2 * eht * sqrt(ex / (twopi))) - c / vega
    volgaOverVega = (h + t) * (h - t) / v
    return cEstimate - c, cEstimateOverVega, volgaOverVega
end

function impliedVolatilitySRHouseholder(
    isCall::Bool,
    price::T,
    f::T,
    strike::T,
    tte::T,
    df::T,
    ftolrel::T = 0,
    maxEval::Int = 64,
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
