#Hagan approximation for Heston 
#using AQFED.Math
import CharFuncPricing: modsim
import AQFED.TermStructure: SABRParams, SABRSection, HaganEffective, varianceByLogmoneyness, HaganHestonPiecewiseParams,makeHestonTSParams,makeHaganHestonPiecewiseParams
import AQFED.Black: blackScholesFormula
import AQFED.Bachelier: bachelierFormula
export HaganHestonGenericApprox, HaganHestonTSApprox, makeHaganHestonPiecewiseParams, convertTime, makeHestonTSParams,priceEuropean
struct HaganHestonGenericApprox{T}
    v0::T
    κ::T
    θ::Function
    ρ::Function
    σ::Function
    startTime::Vector{T}
end


struct HaganHestonTSApprox{T,U}
    params::T
    tte::U
    sabrParams::SABRParams{U}
end

function HaganHestonTSApprox(params::T, tte::U) where {T<:HaganHestonPiecewiseParams, U <: Number}
    ρ = params.ρ
    κ = params.κ
    tteIndex = searchsortedfirst(params.startTime, tte)
    ts = vcat(params.startTime[1:tteIndex-1], tte)
    #println("ts ",ts)
    #In Hagan, sigma captures term structure and theta=v0=1
    # dS = sig * sqrt(V)* F*dW and dV = kappa(theta-V) dt + omega*sqrt(V)*dW
    # d(sig2 * V) = kappa(sig2*theta - (sig2*V)) + omega*sqrt(sig2*V)*sig*dW
    sig2 = params.leverage .^ 2
    sig = params.leverage
    ω = params.σ  #is quickly large as sigma is usually > sqrt(theta).
    #println("omega ",ω)
    #we have implicitly v0=theta(1)=sig2(1). may add at small dt upfront to allow for generic theta(1).
    t1 = tte
    τsum = zero(tte)
    I2sum = zero(tte)
    I5sum = zero(tte)
    I6sum = zero(tte)
    n = length(ts) - 1
    for i = n:-1:1
        t0 = ts[i]
        dt = (t1 - t0)
        τsum += sig2[i] * dt
        emkt = exp(-κ * dt)
        I2sum += sig[i]^3 * ρ[i] * ω[i] * (κ * dt - 1 + emkt) / κ^2
        if (i > 1)
            I2sumpart = sum([ρ[j] * ω[j] * sig[j] * exp(κ * ts[j+1]) * (1 - exp(-κ * (ts[j+1] - ts[j]))) / κ for j = 1:i-1])
            I2sum += sig2[i] * (1 - emkt) / κ * exp(-κ * t0) * I2sumpart
        end
        I5sum += ω[i]^2 * sig2[i]^2 * (1 + 2κ * dt - (2 - emkt)^2) / (2 * κ^3)
        if (i < n)
            I5sumpart1 = sum([sig2[j] * exp(-κ * ts[j]) * (1 - exp(-κ * (ts[j+1] - ts[j]))) / κ for j = i+1:n])
            I5sum += ω[i]^2 * sig2[i] * (1 - emkt)^2 / κ^2 * exp(κ * t1) * I5sumpart1
            I5sum += ω[i]^2 * (1 - emkt^2) / (2 * κ) * exp(2κ * t1) * I5sumpart1^2
        end
        I6sum += ρ[i]^2 * ω[i]^2 * sig2[i]^2 * (-2 + κ * dt + (2 + κ * dt) * emkt) / κ^3
        if (i > 1)
            I6sumpart1 = sum([ρ[j] * ω[j] * sig[j] * exp(κ * ts[j+1]) * (1 - exp(-κ * (ts[j+1] - ts[j]))) / κ for j = 1:i-1])
            I6sumpart2 = sum([ρ[j]^2 * sig2[j] * ω[j]^2 * exp(κ * ts[j+1]) * (1 - (1 + κ * (ts[j+1] - ts[j])) * exp(-κ * (ts[j+1] - ts[j]))) / κ^2 for j = 1:i-1])
            I6sum += ρ[i] * ω[i] * sig[i]^3 * (1 - (1 + κ * dt) * emkt) / κ^2 * exp(-κ * t0) * I6sumpart1
            I6sum += sig2[i] * (1 - emkt) / κ * exp(-κ * t0) * I6sumpart2
            if i > 2
                I6sumpart3 = zero(tte)
                for j = 2:i-1
                    for k = 1:j-1
                        I6sumpart3 += ρ[j] * ρ[k] * sig[j] * sig[k] * ω[j] * ω[k] * (ts[j+1] - ts[j]) * exp(κ * ts[k+1]) * (1 - exp(-κ * (ts[k+1] - ts[k]))) / κ
                    end
                end
                I6sum += sig2[i] * (1 - emkt) / κ * exp(-κ * t0) * I6sumpart3
            end
        end

        t1 = t0
    end
    Δ = sqrt(τsum / tte)
    b = I2sum / τsum^2
    #bConst = ρ[1]*ω[1]/(sig[1]) *(κ*tte- 1+exp(-κ*tte))/(κ*tte)^2
    #println("b ",b," ",bConst)
    c = 3I5sum / (4 * τsum^3) + 3I6sum / τsum^3 - 3b^2
    #cConst = 3*ω[1]^2/(sig[1]^2)*(1+2κ*tte-(2-exp(-κ*tte))^2)/(8κ^3 * tte^3)+ 3*ρ[1]^2*ω[1]^2/(sig[1]^2)*(κ^2*tte^2*exp(-κ*tte)-(1-exp(-κ*tte))^2)/(κ^4 * tte^4)
    #println("c ",c, " ",cConst)
    αstd = Δ * exp(-c / 4 * Δ^2 * tte)
    ρstd = max(-0.99,b / sqrt(c))
    νstd = Δ * sqrt(c)
    sabrParams = SABRParams(αstd, 1.0, ρstd, νstd)
    return HaganHestonTSApprox(params, tte, sabrParams)
end
function priceEuropean(p::HaganHestonTSApprox{T,U}, isCall::Bool, strike, forward, tte, df) where {T<:HaganHestonPiecewiseParams, U <: Number}
    if tte != p.tte
        throw(DomainError(tte,"Time to maturity does not match"))
    end
    section = SABRSection(HaganEffective(), p.sabrParams, tte, forward, 0.0)
    bvolSq = normalVarianceByMoneyness(section, strike - forward)
    bprice = bachelierFormula(isCall, strike, forward, sqrt(bvolSq), tte, df)
    #println("bprice ",bprice)
    #volSq = varianceByLogmoneyness(section, log(strike/forward)) #less accurate?
    #price = blackScholesFormula(isCall, strike, forward, volSq*tte, 1.0, df)
    return bprice
end

function priceEuropean(p::HaganHestonGenericApprox{T}, isCall::Bool, strike, forward, tte, df; tol=sqrt(eps(T))) where {T}
    v0 = p.params.v0
    θ = p.params.θ
    σ = p.params.σ
    ρ = p.params.ρ
    κ = p.params.κ
    ts = p.params.startTime

    function D(x, t)
        (1 - exp(-κ * (t - x))) / κ
    end
    function V(u, t)
        v0 * (exp(-κ * t) - exp(-κ * u)) + modsim(x -> κ * θ(x) * exp(-κ * (t - x)), u, t, tol)
    end
    function I1(u, t)
        modsim(x -> ρ(x) * σ(x) * V(x) * exp(-κ * (t - x)), u, t, tol)
    end
    function I2(u, t)
        modsim(x -> ρ(x) * σ(x) * V(x) * D(x, t), u, t, tol)
    end
    function I3(u, t)
        modsim(x -> σ^2(x) * V(x) * exp(-κ * (t - x)) * D(x, t), u, t, tol)
    end
    function I4part(x, t)
        modsim(u -> ρ(u) * σ(u), x, t, tol)
    end
    function I4(u, t)
        modsim(x -> ρ(x) * σ(x) * V(x) * exp(-κ * (t - x)) * I4part(x, t), u, t, tol)
    end
    function τ(u, t)
        modsim(x -> V(x), u, t, tol)
    end
    Vsum = zero(strike)
    I1sum = zero(strike)
    I2sum = zero(strike)
    I3sum = zero(strike)
    I4sum = zero(strike)
    τsum = zero(strike)
    t1 = tte
    for i = length(ts):-1:1
        t0 = ts[i]
        if (t0 < tte)
            t1 = min(t1, tte)
            Vsum += V(t0 + 1e-7, t1 - 1e-7)
            I1sum += I1(t0 + 1e-7, t1 - 1e-7)
            I2sum += I2(t0 + 1e-7, t1 - 1e-7)
            I3sum += I3(t0 + 1e-7, t1 - 1e-7)
            I4sum += I4(t0 + 1e-7, t1 - 1e-7)
            τsum += τ(t0 + 1e-7, t1 - 1e-7)
        end
        t1 = t0
    end
    b = I1sum / (2 * τsum * Vsum)
    c = I3sum / (2 * τsum^2 * Vsum) - 3 * b * I2sum / τsum^2 + I4sum / (τsum^2 * Vsum)
    Gamma0 = 1
    G = -(b * Gamma0 + c) * τSum
    Δ = 1
    αstd = Δ * exp(-c / 4 * Δ^2 * tte)
    ρstd = b / sqrt(c)
    νstd = Δ * sqrt(c)
    sabrParams = SABRParams(αstd, 1.0, ρstd, νstd)
    section = SABRSection(HaganEffective(), sabrParams, tte, forward, 0.0)
    volSq = varianceByLogmoneyness(section, log(strike / forward))
    price = blackScholesFormula(isCall, strike, forward, volSq * tte, 1.0, df)
    return price
end
