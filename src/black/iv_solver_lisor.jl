const m00 = -0.00006103098165
const n00 = 1.0
const m01 = 5.33967643357688
const n01 = 22.96302109010794
const m10 = -0.40661990365427
const n10 = -0.48466536361620
const m02 = 3.25023425332360
const n02 = -0.77268824532468
const m11 = -36.19405221599028
const n11 = -1.34102279982050
const m20 = 0.08975394404851
const n20 = 0.43027619553168
const m03 = 83.84593224417796
const n03 = -5.70531500645109
const m12 = 41.21772632732834
const n12 = 2.45782574294244
const m21 = 3.83815885394565
const n21 = -0.04763802358853
const m30 = -0.21619763215668
const n30 = -0.03326944290044

function impliedVolatilitySqrtTimeRationalGuess(x, c)
    num = m00 + x * (m10 + x * (m20 + x * m30)) + c * (m01 + m11 * x + m21 * x^2) + c^2 * (m02 + m12 * x) + c^3 * m03
    den = n00 + x * (n10 + x * (n20 + x * n30)) + c * (n01 + n11 * x + n21 * x^2) + c^2 * (n02 + n12 * x) + c^3 * n03
    return num / den
end


function impliedVolatilityLiRationalGuess(
    isCall::Bool,
    price::Float64,
    forward::Float64,
    strike::Float64,
    timeToExpiry::Float64,
    df::Float64,
)::Float64
    c, ex = normalizePrice(isCall,price,forward,strike,df)
    if c >= 1 / ex || c >= 1
        throw(DomainError(c, string("price higher than intrinsic value ", 1 / ex)))
    elseif c <= 0
        throw(DomainError(c, "price is negative"))
    end
    x = log(ex)
    return impliedVolatilitySqrtTimeRationalGuess(x, c) / sqrt(timeToExpiry), nil
end

abstract type SORSolver end
struct SORTS <: SORSolver end
struct SORDR <: SORSolver end


function impliedVolatilityLiSOR(
    isCall::Bool,
    price::TP,
    forward::T,
    strike::T,
    timeToExpiry::T,
    df::T,
    impliedVolGuess::T,
    tolerance::T,
    maxIterations::Int,
    solver::SORSolver)::TP where {T,TP}
    c, ex = normalizePrice(isCall,price,forward,strike,df)
    if c > one(T) / ex
        return zero(TP)
    end
    x = log(ex)
    v0 = zero(TP)
    if impliedVolGuess == zero(T)
        v0 = impliedVolatilitySqrtTimeRationalGuess(x, c)
        # if abs(x) < 3 && c > 0.0005 && c < 0.9995 #domain of validity of the rational approximation
        #     v0 = impliedVolatilitySqrtTimeRationalGuess(x, c)
        #     #        println("guess R ",v0," ",v0/sqrtte)
        # else
            # v0 = TP(impliedVolatilitySRGuessUndiscountedCall(c, ex, x))
            #    println("guess SR ",v0," ",v0/sqrtte,x,c)
        # end
    else
        v0 = impliedVolGuess * sqrt(timeToExpiry)
    end
    return computeLiSOR(c, x, ex, v0, sqrt(timeToExpiry), tolerance, maxIterations, solver)
end


function computeLiSOR(
    c::TP,
    x::T,
    ex::T,
    v0::TP,
    sqrttte::T,
    tolerance::T,
    maxIterations::Int,
    solver::SORSolver,
)::TP where {T,TP}
    v = v0
    if maxIterations <= 0
        return v
    end
    exinvsqrt = 1 / sqrt(ex)
    h = x / v
    t = v / 2
    sqrt2 = sqrt(T(2))
    Np = erfcx(-(h + t)/sqrt2)
    Nm = erfcx(-(h - t)/sqrt2)
    norm = exinvsqrt /2 * exp(-(h^2 + t^2)/2)
    cEstimate = norm * (Np - Nm)
    if cEstimate < zero(TP)
        cEstimate = zero(TP)
    end

    phi = (v^2 + 2 * x) / (v^2 - 2 * x)
    omega = omegaSOR(solver, phi)
    iterations = 0
    vOld = zero(TP)
    vTolerance = 64 * eps(T) * sqrttte #accuracy of inversion is limited, do not loop forever
    while abs(v - vOld) > vTolerance && abs(c - cEstimate) > tolerance && iterations < maxIterations
        vOld = v
        phi = (v^2 + 2 * x) / (v^2 - 2 * x)
        alpha = (1 + omega) / (1 + phi)
        F = c + norm * (Nm + omega * Np)
        Fom = F / (1 + omega)
        if Fom >= 1
            Fom = F / 2
        end
        Nm1 = zero(T)
        #try
            Nm1 = norminv(Fom)
        #catch y
        #    #println(iterations, " ", v, " ", c, " ", F, " ", Fom, " ", omega)
        #    return v0 / sqrttte
        #end
        G = Nm1 + sqrt(Nm1^2 - 2 * x)
        v = iterateSOR(solver, v, G, alpha)
        omega = omegaSOR(solver, phi)
        h = x / v
        t = v / 2

        Np = erfcx(-(h + t)/sqrt2)
        Nm = erfcx(-(h - t)/sqrt2)
        norm = exinvsqrt * exp(-(h^2 + t^2)/2)/2
        # slower alternative:
        # Np = normcdf(h+t)
        # Nm = normcdf(h-t)/ex
        # norm = 1.0
        cEstimate = max(zero(TP),norm * (Np - Nm))
        iterations += 1
        #    println("SORTS ",iterations, " ",abs(v-vOld)," ",abs(c-cEstimate))
    end
    sigma = v / sqrttte
    return sigma
end

function iterateSOR(solver::SORTS, v, G, alpha)
    return alpha * G + (1 - alpha) * v
end
function iterateSOR(solver::SORDR, v, G, alpha) 
    return G
end

function omegaSOR(solver::SORTS, phi)
    return 1
end

function omegaSOR(solver::SORDR, phi)
    return phi
end
