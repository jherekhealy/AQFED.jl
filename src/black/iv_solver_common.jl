#common, except for Jaeckel
abstract type SRSolver end

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
#polya is 2 / pi, polya_factor = sqrt(pi/8) is aludaat,
function impliedVolatilitySRGuessUndiscountedCall(price::Real, ey::Real, y::Real, polya_factor::Real = 2/pi)::Real
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
        gmy = max(gamma - y,0)      #machine epsilon issues
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
