struct Halley <: SRSolver end
struct SuperHalley <: SRSolver end
struct InverseQuadratic <: SRSolver end
struct CMethod <: SRSolver end

# 1 SuperHalley = 2 InverseQuadratic = 3 CMethod = 4
function impliedVolatilitySRHalley(
    isCall::Bool,
    price::TP,
    f::T,
    strike::T,
    tte::T,
    df::T,
    ftolrel::T,
    maxEval::Int,
    solver::SRSolver,
)::TP where {T,TP}
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
