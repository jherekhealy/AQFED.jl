using AQFED.Math
using FastGaussQuadrature
export DeelstraBasketPricer, DeelstraLBBasketPricer, priceEuropean
using LinearAlgebra
using Roots

struct DeelstraBasketPricer{T}
    δIndex::Int
    fIndex::Int
    x::Vector{T} #quadrature abscissae in -1, 1
    w::Vector{T} #quadrature weights
end

function DeelstraBasketPricer(δIndex::Int, fIndex::Int)
    xG, wG = gausslegendre(33)
    DeelstraBasketPricer{Float64}(δIndex, fIndex, xG, wG)
end

function DeelstraLBBasketPricer(δIndex::Int, fIndex::Int)
    DeelstraBasketPricer{Float64}(δIndex, fIndex, Vector{Float64}(undef, 0), Vector{Float64}(undef, 0))
end

DeelstraBasketPricer() = DeelstraBasketPricer(3,3)
DeelstraLBBasketPricer() = DeelstraLBBasketPricer(3,3)

function priceEuropean(
    p::DeelstraBasketPricer{T},
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::Vector{T},
    forward::Vector{T}, #forward to option maturity
    totalVariance::Vector{T}, #vol^2 * τ
    weight::Vector{T},
    correlation::Matrix{T},
)::T where {T}
    n = length(spot)
    δ = zeros(T, n) #we store δ * S here.
    f = zero(T)
    u1 = zero(T)
    for (i, wi) in enumerate(weight)
        if p.δIndex == 1
            δ[i] = forward[i] * exp(-totalVariance[i] / 2)
        elseif p.δIndex == 2
            δ[i] = spot[i]
        else #p.δIndex == 3
            δ[i] = forward[i]
        end
        f += wi * δ[i]
        u1 += wi * forward[i]
    end
    sumwbd = zero(T)
    wsdv = zeros(T, n)
    for (i, wi) in enumerate(weight)
        wTildei = wi * δ[i] / f
        βi = log(forward[i]) - totalVariance[i] / 2
        sumwbd += wTildei * (βi - log(δ[i]))
        wsdv[i] = wi * δ[i] * sqrt(totalVariance[i])
    end
    dGamma = f * (log(strike / f) - sumwbd)

    varGamma = zero(T)

    r = zeros(T, n)
    for (i, wsdi) in enumerate(wsdv)
        for (j, wsdj) in enumerate(wsdv)
            covar = wsdj * correlation[i, j]
            varGamma += covar * wsdi
            r[i] += covar
        end
    end
    varGamma = abs(varGamma)
    volGamma = sqrt(varGamma)
    r ./= volGamma
    priceCall = zero(T)

    if length(p.x) == 0
        function objective(λ::T)::Tuple{T,T,T}
            #fΔxΔΔx
            eS = zero(T)
            DeS = zero(T)
            D2eS = zero(T)
            for (i, fi) in enumerate(forward)
                viT = totalVariance[i]
                eSi = weight[i] * fi * exp(-r[i]^2 * viT / 2 + r[i] * sqrt(viT) * (λ) / (volGamma))
                eS += eSi
                DeS += eSi * (r[i] * sqrt(viT))/volGamma
                D2eS += eSi * ((r[i] * sqrt(viT))/volGamma)^2
            end
            # println(λ," ", eS-strike," ", DeS, " ",D2eS)
            return (eS - strike, (eS-strike)/DeS, DeS/D2eS)
        end
        lambdaMax = dGamma * 2
        if lambdaMax < zero(T)
            lambdaMax = dGamma / 2
        end
        #init guess dGamma
        lambdaOpt = find_zero(objective, dGamma, Roots.Halley(), atol=1e-8)
        dGammaStar = (lambdaOpt) / volGamma

        i1 = zero(T)
        for (i, fi) in enumerate(forward)
            i1 += weight[i] * fi * normcdf(r[i] * sqrt(totalVariance[i]) - dGammaStar)
        end
        i1 -= strike * normcdf(-dGammaStar)
        priceCall = i1
    else
        function integrand(λ::T)::T
            #depends on forward, weight, tvar, r, volGamma,correl, f, sumwbd, strike
            eS = zero(T)
            eS2 = zero(T)
            for (i, fi) in enumerate(forward)
                tempi = weight[i] * fi
                viT = totalVariance[i]
                sqrti = sqrt(viT)
                eS += tempi * exp(-r[i]^2 * viT / 2 + r[i] * sqrti * (λ) / (volGamma))
                for j = 1:i-1
                    vjT = totalVariance[j]
                    sqrtj = sqrt(vjT)
                    sigmaij = sqrt(viT + vjT + 2 * sqrti * sqrtj * correlation[i, j])
                    rij = (r[i] * sqrti + r[j] * sqrtj) / sigmaij
                    eFactor = -viT / 2 - vjT / 2 + (1 - rij^2) / 2 * sigmaij^2 + rij * sigmaij * (λ) / (volGamma)
                    eS2 += 2 * tempi * weight[j] * forward[j] * exp(eFactor)
                end
                eS2 += tempi^2 * exp(-viT + 2 * (1 - r[i]^2) * viT + r[i] * 2 * sqrti * (λ) / (volGamma))
            end

            fS = zero(T)
            if p.fIndex == 1
                #zero
            elseif p.fIndex == 2
                lnGF = λ / f + sumwbd
                fS = f * (1 + lnGF)
            else #p.fIndex >= 3
                #most accurate according to Deelstra paper Fig.2
                lnGF = λ / f + sumwbd
                gF = exp(lnGF)
                fS = f * gF
            end
            euphs2 = eS - fS
            if euphs2 < eps(T)
                fS = 0
                euphs2 = eS
            end
            e2ups2 = eS2 - 2 * fS * eS + fS^2

            if e2ups2 < eps(T)
                e2ups2 = eps(T)
            end
            uphs2 = log(euphs2)
            ups2 = log(e2ups2) / 2
            muS = (uphs2 - ups2 / 2) * 2
            sigmaS2 = ups2 - muS
            if sigmaS2 < eps(T)
                sigmaS2 = eps(T)
            end
            varLambda = varGamma
            pdf = exp(-λ^2 / (2 * varLambda)) / sqrt(T(pi) * 2 * varLambda)

            if strike <= fS
                return euphs2 * pdf
            end
            d1 = (ups2 - log(strike - fS)) / sqrt(sigmaS2)
            d2 = d1 - sqrt(sigmaS2)
            return (euphs2 * normcdf(d1) - (strike - fS) * normcdf(d2)) * pdf
        end
        λMin = -3 * sqrt(varGamma)
        i2 = zero(T)
        if dGamma > λMin
            a = λMin
            b = dGamma
            bma2 = (b - a) / 2
            bpa2 = (b + a) / 2
            i2 = bma2 * dot(p.w, integrand.(bma2 .* p.x .+ bpa2))
        end
        dGammaStar = (dGamma) / sqrt(varGamma)

        i1 = zero(T)
        for (i, fi) in enumerate(forward)
            i1 += fi * weight[i] * normcdf(r[i] * sqrt(totalVariance[i]) - dGammaStar)
        end
        i1 -= strike * normcdf(-dGammaStar)
        priceCall = (i1 + i2)
    end

    pricePut = strike - u1 + priceCall
    if isCall
        return priceCall * discountFactor
    else
        return pricePut * discountFactor
    end
end
