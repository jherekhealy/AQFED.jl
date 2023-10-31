using CharFuncPricing
using AQFED.Math
import SpecialFunctions: gamma

export RoughHestonParams, PadeCharFunc, AdamsCharFunc

struct RoughHestonParams{T}
    H::T
    ρ::T
    ν::T
    varianceCurve # a function of time.
end


struct PadeCharFunc{MT,CR,ORDER} <: CharFuncPricing.CharFunc{MT,CR} #model type, return type (e.g. Complex or acb)
    model::MT
    q::Quadrature
end

PadeCharFunc{CR}(model::MT, q::Quadrature=Chebyshev{Float64,1}(24)) where {MT,CR} = PadeCharFunc{MT,CR,3}(model, q) #
PadeCharFunc(model::RoughHestonParams{Float64}) = PadeCharFunc{Complex}(model)


@inline CharFuncPricing.model(cf::PadeCharFunc) = cf.model

@inline function CharFuncPricing.evaluateCharFunc(p::CharFuncPricing.CharFunc{RoughHestonParams{T},CR}, z::CT, τ::T)::CR where {T,CR,CT}
    return exp(evaluateLogCharFunc(p, z, τ))
end
function CharFuncPricing.getControlVariance(p::PadeCharFunc{RoughHestonParams{T},CR,ORDER}, τ::T) where {T,CR,ORDER}
    return zero(T)
end

function CharFuncPricing.evaluateLogCharFunc(p::PadeCharFunc{RoughHestonParams{T},CR,3}, z::CT, τ::T) where {T,CR,CT}
    #integrate pade33.
    params = CharFuncPricing.model(p)
    numal = params.ν^(1 / (params.H + 0.5))
    integrand = function (u)
        x = u * numal
        pade(p, z, x)[2] * params.varianceCurve(τ - u)
    end

    return integrate(p.q, integrand, zero(T), τ)
end

function pade(p::PadeCharFunc{RoughHestonParams{T},CR,3}, a, x) where {T,CR}
    params = CharFuncPricing.model(p)
    H = params.H
    rho = params.ρ
    al = H + 0.5

    aa = sqrt(a * (a + 1im) - rho^2 * a^2)
    rm = -(1im) * rho * a - aa
    rp = -(1im) * rho * a + aa

    b1 = -a * (a + 1im) / (2 * gamma(1 + al))
    b2 = (1 - a * 1im) * a^2 * rho / (2 * gamma(1 + 2 * al))
    b3 = gamma(1 + 2 * al) / gamma(1 + 3 * al) * (a^2 * (1im + a)^2 / (8 * gamma(1 + al)^2) + (a + 1im) * a^3 * rho^2 / (2 * gamma(1 + 2 * al)))

    g0 = rm
    g1 = -rm / (aa * gamma(1 - al))
    g2 = rm / aa^2 / gamma(1 - 2 * al) * (1 + rm / (2 * aa) * gamma(1 - 2 * al) / gamma(1 - al)^2)

    den = g0^3 + 2 * b1 * g0 * g1 - b2 * g1^2 + b1^2 * g2 + b2 * g0 * g2

    p1 = b1
    p2 = (b1^2 * g0^2 + b2 * g0^3 + b1^3 * g1 + b1 * b2 * g0 * g1 - b2^2 * g1^2 + b1 * b3 * g1^2 + b2^2 * g0 * g2 - b1 * b3 * g0 * g2) / den
    q1 = (b1 * g0^2 + b1^2 * g1 - b2 * g0 * g1 + b3 * g1^2 - b1 * b2 * g2 - b3 * g0 * g2) / den
    q2 = (b1^2 * g0 + b2 * g0^2 - b1 * b2 * g1 - b3 * g0 * g1 + b2^2 * g2 - b1 * b3 * g2) / den
    q3 = (b1^3 + 2 * b1 * b2 * g0 + b3 * g0^2 - b2^2 * g1 + b1 * b3 * g1) / den
    p3 = g0 * q3

    y = x^al
    pade = (p1 * y + p2 * y^2 + p3 * y^3) / (1 + q1 * y + q2 * y^2 + q3 * y^3)
    return pade, (pade - rm) * (pade - rp) / 2
end



## Based on the R code from Jim Gatheral - the various sums would benefit from optimization
struct AdamsCharFunc{MT,CR} <: CharFuncPricing.CharFunc{MT,CR} #model type, return type (e.g. Complex or acb)
    model::MT
    n::Int
end

AdamsCharFunc{CR}(model::MT, n::Int) where {MT,CR} = AdamsCharFunc{MT,CR}(model, n) #
AdamsCharFunc(model::RoughHestonParams{Float64}, n::Int=200) = AdamsCharFunc{Complex}(model, n)

@inline CharFuncPricing.model(cf::AdamsCharFunc) = cf.model

RoughHestonCVCharFunc(delegate::Union{AdamsCharFunc{RoughHestonParams{T},CR},PadeCharFunc{RoughHestonParams{T},CR}}) where {T,CR} =
    CharFuncPricing.CVCharFunc{RoughHestonParams{T},BlackParams{T},CR}(
        delegate,
        CharFuncPricing.DefaultCharFunc{BlackParams{T},CR}(
            CharFuncPricing.BlackParams{T}(sqrt(CharFuncPricing.model(delegate).varianceCurve(zero(T))))
        ),
    )

@inline function CharFuncPricing.getControlVariance(
        cf::CVCharFunc{RoughHestonParams{T},BlackParams{T}},
        τ::T,
    )::T where {T}
      CharFuncPricing.model(cf).varianceCurve(zero(T))
end

function CharFuncPricing.evaluateCharFunc(
    p::CharFuncPricing.CVCharFunc{MAINT,CONTROLT,CR},
    z::CT,
    τ::T)::CR where {T,CR,CT,MAINT,CONTROLT}
    phi = CharFuncPricing.evaluateCharFunc(p.main, z, τ)
    phiB = CharFuncPricing.evaluateCharFunc(p.control, z, τ)
    return phi - phiB
end

struct ChebyshevCharFunc{CF,MT,CR} <:    CharFuncPricing.CharFunc{MT,CR} 
    delegate::CF      
    upperBound::Float64
    coeff::Vector{Complex}
end

function ChebyshevCharFunc(delegate::AdamsCharFunc{MT,CR} , upperBound::Float64, n::Int, τ::Float64) where {MT,CR} 
    coeff = zeros(Complex,n)
    fvalues = zeros(Complex,n)
    f = function(x)
        CharFuncPricing.evaluateLogCharFunc(delegate, x - 1im/2, τ)
    end
    a = 0.0
    b = upperBound
    nodes = (cheb2nodes(Float64, n) .* (b-a) .+ (a+b)) ./ 2
    @. fvalues =f(nodes) #works only with Lewis formula for now.
    cheb2coeff!(coeff, fvalues)
    return ChebyshevCharFunc{AdamsCharFunc{MT,CR},MT,CR}(delegate, upperBound, coeff)
end
function CharFuncPricing.evaluateLogCharFunc(p::ChebyshevCharFunc, z::CT, τ::T) where {T,CT}
    if (real(z) > p.upperBound || real(z) < 0.0) 
        return zero(Complex)
    end
    cheb2interp(p.coeff, real(z), 0.0, p.upperBound)
end
function CharFuncPricing.getControlVariance(p::ChebyshevCharFunc, τ::T) where {T}
    return zero(T)
end

@inline CharFuncPricing.model(cf:: ChebyshevCharFunc{ T} ) where {T} = CharFuncPricing.model(cf.delegate)

function CharFuncPricing.getControlVariance(p::AdamsCharFunc{RoughHestonParams{T},CR}, τ::T) where {T,CR}
    return zero(T)
end

function CharFuncPricing.evaluateLogCharFunc(p::AdamsCharFunc{RoughHestonParams{T},CR}, z::CT, τ::T) where {T,CR,CT}
    #integrate pade33.
    params = CharFuncPricing.model(p)
    #    numal = params.ν^(1/(params.H+0.5))
    dhA = adams(p, z, τ)
    n = p.n
    integral = sum(dhA .* [params.varianceCurve(τ * (1 - k / n)) for k = 0:n]) * τ / n
    return integral
end

function adams(p::AdamsCharFunc{RoughHestonParams{T},CR}, a, bigT) where {T,CR}
    params = p.model
    rho = params.ρ
    nu = params.ν
    obj = function (u, h)
        return -u / 2 * (u + 1im) + rho * nu * u * h * 1im + (nu * h)^2 / 2
    end
    n = p.n
    DELTA = bigT / n
    H = params.H
    alpha = H + 0.5
    hA = zeros(CR, n + 1)
    dhA = zeros(CR, n + 1)
    f = zeros(CR, n + 1)


    a01 = (DELTA^alpha) / gamma(alpha + 2) * alpha
    a11 = (DELTA^alpha) / gamma(alpha + 2)
    b01 = (DELTA^alpha) / gamma(alpha + 1)
    dhA[1] = obj(a, hA[1])
    # hA[1] is zero, boundary condition
    hP = b01 * obj(a, hA[1])
    hA[2] = a01 * obj(a, hA[1]) + a11 * obj(a, hP)
    dhA[2] = obj(a, hA[2])
    # ------

    k0 = 1:(n-1)
    a0 = @. (DELTA^alpha) / gamma(alpha + 2) * (k0^(alpha + 1) - (k0 - alpha) * (k0 + 1)^alpha)
    q = 0:(n-1)
    aj = @. (DELTA^alpha) / gamma(alpha + 2) * ((q + 2)^(alpha + 1) + (q)^(alpha + 1) - 2 * (q + 1)^(alpha + 1))
    r = 0:n
    bj = @. (DELTA^alpha) / gamma(alpha + 1) * ((r + 1)^alpha - (r)^alpha)

    akp1 = (DELTA^alpha) / gamma(alpha + 2)
    f[1] = obj(a, hA[1])

    for k = 1:(n-1)
        f[k+1] = obj(a, hA[k+1])
        hP = sum(f[1:k+1] .* bj[(k+1):-1:1]) 
        aA = vcat(a0[k], aj[k:-1:1])
        hA[k+2] = sum(f[1:k+1] .* aA) + akp1 * obj(a, hP) 
        dhA[k+2] = obj(a, hA[k+2])
    end
    #println(length(dhA), dhA)
    return dhA
end
