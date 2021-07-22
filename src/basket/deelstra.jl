using AQFED.Math
using FastGaussQuadrature
export DeelstraBasketPricer, DeelstraLBBasketPricer, priceEuropean
using LinearAlgebra
using Roots

struct DeelstraBasketPricer
    δIndex::Int
    fIndex::Int
    x::Vector{Float64} #quadrature abscissae in -1, 1
    w::Vector{Float64} #quadrature weights
end

function DeelstraBasketPricer(δIndex::Int, fIndex::Int; N=33*4)
    xG, wG = gausslegendre(N)
    DeelstraBasketPricer(δIndex, fIndex, xG, wG)
end

function DeelstraLBBasketPricer(δIndex::Int, fIndex::Int)
    DeelstraBasketPricer(δIndex, fIndex, Vector{Float64}(undef, 0), Vector{Float64}(undef, 0))
end

DeelstraBasketPricer() = DeelstraBasketPricer(3,3)
DeelstraLBBasketPricer() = DeelstraLBBasketPricer(3,3)

function priceEuropean(
    p::DeelstraBasketPricer,
    isCall::Bool,
    strike::Number,
    discountFactor::Number, #discount factor to payment
    spot::AbstractArray{<:Number},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:Number}, #vol^2 * τ
    weight::AbstractArray{<:Number},
    correlation::Matrix{TV},
)::Number where {TV <: Number}
    n = length(spot)
    δ = zeros(TV, n) #we store δ * S here.
    f = zero(eltype(forward))
    u1 = zero(eltype(forward))
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
    sumwbd = zero(TV)
    wsdv = zeros(TV, n)

    for (i, wi) in enumerate(weight)
        wTildei = wi * δ[i] / f
        βi = log(forward[i]) - totalVariance[i] / 2
        sumwbd += wTildei * (βi - log(δ[i]))
        if totalVariance[i] > 0
            wsdv[i] = wi * δ[i] * sqrt(totalVariance[i])
        end
    end
    dGamma = f * (log(strike / f) - sumwbd)

    varGamma = zero(TV)
    r = zeros(TV, n)
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
    priceCall = zero(Float64)
    if length(p.x) == 0
        function objective(λ::Number)::Tuple{<:Number,<:Number,<:Number}
            #fΔxΔΔx
            eS = zero(typeof(λ))
            DeS = zero(typeof(λ))
            D2eS = zero(typeof(λ))
            for (i, fi) in enumerate(forward)
                viT = totalVariance[i]
                eSi = weight[i] * fi
                if viT > 0
                    sqrti = sqrt(viT)
                    eSi *= exp(-r[i]^2 * viT / 2 + r[i] * sqrti  * (λ) / (volGamma))
                    DeS += eSi * (r[i] * sqrti)/volGamma
                    D2eS += eSi * ((r[i] * sqrti)/volGamma)^2
                end
                eS += eSi
            end
            # println(λ," ", eS-strike," ", DeS, " ",D2eS)
            return (eS - strike, (eS-strike)/DeS, DeS/D2eS)
        end
        lambdaMax = dGamma * 2
        if lambdaMax < zero(Float64)
            lambdaMax = dGamma / 2
        end
        #init guess dGamma
        lambdaOpt = find_zero(objective, dGamma, Roots.Halley(), atol=1e-8)
        dGammaStar = (lambdaOpt) / volGamma
        #println("dG ",dGammaStar)
        i1 = zero(Float64)
        for (i, fi) in enumerate(forward)
            if totalVariance[i] > 0
                i1 += weight[i] * fi * normcdf(r[i] * sqrt(totalVariance[i]) - dGammaStar)
            else
                i1 += weight[i] * fi * normcdf(- dGammaStar)
            end
        end
        i1 -= strike * normcdf(-dGammaStar)
        priceCall = i1
    else
        function integrand(λ)
            #depends on forward, weight, tvar, r, volGamma,correl, f, sumwbd, strike
            eS = zero(Float64)
            eS2 = zero(Float64)
            for (i, fi) in enumerate(forward)
                tempi = weight[i] * fi
                eSi = tempi
                viT = totalVariance[i]
                sqrti = zero(Float64)
                if viT > 0
                    sqrti = sqrt(viT)
                    eSi *= exp(-r[i]^2 * viT / 2 + r[i] * sqrti * (λ) / (volGamma))
                end
                eS += eSi
                for j = 1:i-1
                    vjT = totalVariance[j]
                    sqrtj = zero(Float64)
                    if vjT > 0
                        sqrtj = sqrt(vjT)
                    end
                    sigmaij = sqrt(viT + vjT + 2 * sqrti * sqrtj * correlation[i, j])
                    rij = (r[i] * sqrti + r[j] * sqrtj) / sigmaij
                    eFactor = -viT / 2 - vjT / 2 + (1 - rij^2) / 2 * sigmaij^2 + rij * sigmaij * (λ) / (volGamma)
                    eS2 += 2 * tempi * weight[j] * forward[j] * exp(eFactor)
                end
                eS2 += tempi^2 * exp(-viT + 2 * (1 - r[i]^2) * viT + r[i] * 2 * sqrti * (λ) / (volGamma))
            end

            fS = zero(Float64)
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
            if euphs2 < eps(Float64)
                fS = 0
                euphs2 = eS
            end
            e2ups2 = eS2 - 2 * fS * eS + fS^2

            if e2ups2 < eps(Float64)
                e2ups2 = eps(Float64)
            end
            uphs2 = log(euphs2)
            ups2 = log(e2ups2) / 2
            muS = (uphs2 - ups2 / 2) * 2
            sigmaS2 = ups2 - muS
            if sigmaS2 < eps(Float64)
                sigmaS2 = eps(Float64)
            end
            varLambda = varGamma
            pdf = exp(-λ^2 / (2 * varLambda)) / sqrt(Float64(pi) * 2 * varLambda)

            if strike <= fS
                return euphs2 * pdf
            end
            d1 = (ups2 - log(strike - fS)) / sqrt(sigmaS2)
            d2 = d1 - sqrt(sigmaS2)
            return (euphs2 * normcdf(d1) - (strike - fS) * normcdf(d2)) * pdf
        end
        λMin = -3 * sqrt(varGamma)
        i2 = zero(Float64)
        if dGamma > λMin
            a = λMin
            b = dGamma
            bma2 = (b - a) / 2
            bpa2 = (b + a) / 2
            i2 = bma2 * dot(p.w, integrand.(bma2 .* p.x .+ bpa2))
        end
        dGammaStar = (dGamma) / sqrt(varGamma)

        i1 = zero(Float64)
        for (i, fi) in enumerate(forward)
            if totalVariance[i] > 0
                i1 += weight[i] * fi * normcdf(r[i] * sqrt(totalVariance[i]) - dGammaStar)
            else
                i1 += weight[i] * fi * normcdf(- dGammaStar)
            end
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
