using AQFED.Math
export BasketPricer, DeelstraBasketPricer, DeelstraLBBasketPricer, priceEuropean
using LinearAlgebra
using Roots

abstract type BasketPricer end

struct DeelstraBasketPricer <: BasketPricer
    δIndex::Int
    fIndex::Int
    q::Quadrature
    #number of conditional moments matched by the shifted lognormal: 2 (Deelstra et al., shift given by fIndex) or 3
    #(the shift is fitted to the conditional third moment, fIndex is then ignored). 3 is more accurate for negatively
    #correlated assets, where the conditional distribution is far from lognormal, at the cost of O(d³) operations.
    nMoments::Int
    #number of directions integrated out besides Λ (0 or 1). With negatively correlated assets, the basket given Λ still
    #depends on a long-short combination of the assets, along which it behaves like a hyperbolic cosine: its left tail is then poorly described by
    #a shifted lognormal, whatever the number of moments matched. With 1, the residual factor (the direction of largest variance given Λ) is
    #integrated (qz on fixed panels of width 2, the kinks of the intrinsic value being integrated exactly) and the shifted
    #lognormal approximation is applied to the basket given Λ and the residual factor.
    nDirections::Int
    qz::GaussLegendre
end
DeelstraBasketPricer(δIndex::Int, fIndex::Int, q::Quadrature, nMoments::Int=2) = DeelstraBasketPricer(δIndex, fIndex, q, nMoments, 0, GaussLegendre(4))
export LowerBound
struct LowerBound <: Quadrature
end
isSplit(q::T) where {T<:Quadrature} = false



function DeelstraBasketPricer(δIndex::Int, fIndex::Int; q::Quadrature=GaussLegendre(21), nMoments::Int=2, nDirections::Int=0, qz::GaussLegendre=GaussLegendre(4))
    DeelstraBasketPricer(δIndex, fIndex, q, nMoments, nDirections, qz)
end


function DeelstraLBBasketPricer(δIndex::Int, fIndex::Int)
    DeelstraBasketPricer(δIndex, fIndex, LowerBound())
end

DeelstraBasketPricer() = DeelstraBasketPricer(3, 3)
DeelstraLBBasketPricer() = DeelstraLBBasketPricer(3, 3)

#undiscounted call on the shifted lognormal fS + Y matching the conditional moments eS = E[B], eS2 = E[B²], with the
#shift fS given (Deelstra et al.)
shiftedLognormalCall(eS::T, eS2::T, fS::T, strike::T) where {T} = shiftedLognormalCentralCall(eS, eS2 - eS^2, fS, strike)

#same, from the mean eS and the variance of B, without cancellation when the variance is small
function shiftedLognormalCentralCall(eS::T, variance::T, fS::T, strike::T)::T where {T}
    euphs2 = eS - fS
    if euphs2 < eps(T)
        fS = zero(T)
        euphs2 = eS
    end
    e2ups2 = max(variance + euphs2^2, eps(T))
    uphs2 = log(euphs2)
    ups2 = log(e2ups2) / 2
    muS = (uphs2 - ups2 / 2) * 2
    sigmaS2 = max(ups2 - muS, eps(T))
    if strike <= fS #the shifted lognormal is always in the money
        return euphs2 + fS - strike
    end
    d1 = (ups2 - log(strike - fS)) / sqrt(sigmaS2)
    d2 = d1 - sqrt(sigmaS2)
    return euphs2 * normcdf(d1) - (strike - fS) * normcdf(d2)
end

#undiscounted call on c + sgn Y, Y lognormal, matching the mean eS, the variance and the third central moment m3 of B
#(sgn: sign of the skewness)
function shiftedLognormal3Call(eS::T, variance::T, m3::T, strike::T)::T where {T}
    if variance <= eps(T) * eS^2
        return max(eS - strike, zero(T))
    end
    skew = m3 / variance^1.5
    sgn = skew < 0 ? -1 : 1
    #skewness of a lognormal (w+2)√(w-1) with w = exp(s²): u = √(w-1) solves u³ + 3u = |skew|, in a form without
    #cancellation for small skewness
    u = 2 * sinh(asinh(abs(skew) / 2) / 3)
    if u < cbrt(eps(T)) #nearly Gaussian: first order Edgeworth correction in the skewness, continuous with the lognormal
        sd = sqrt(variance)
        d = (eS - strike) / sd
        return (eS - strike) * normcdf(d) + sd * normpdf(d) + skew * (strike - eS) * normpdf(d) / 6
    end
    s2 = log1p(u^2)
    s = sqrt(s2)
    meanY = sqrt(variance) / u
    c = eS - sgn * meanY
    if sgn > 0 #(c + Y - K)^+
        k = strike - c
        if k <= 0
            return eS - strike
        end
        d1 = (log(meanY / k) + s2 / 2) / s
        return meanY * normcdf(d1) - k * normcdf(d1 - s)
    else #(c - Y - K)^+ = (k - Y)^+
        k = c - strike
        if k <= 0
            return zero(T)
        end
        d1 = (log(meanY / k) + s2 / 2) / s
        return k * normcdf(-d1 + s) - meanY * normcdf(-d1)
    end
end


#g_i = Cov[X_i, Λ]/σ_Λ, M = exp(Σ - g gᵀ) - 1 and, with a residual factor, ℓ the direction of largest variance of the
#log-returns given Λ: given Λ and Z, the covariance is Σ - g gᵀ - ℓ ℓᵀ, and Mz = exp(Σ - g gᵀ - ℓ ℓᵀ) - 1. The
#variance of the log-returns, rather than the one of the basket (ConditionalBasketPricer), is used: the left tail of the
#conditional basket, which the shifted lognormal does not capture, is driven by the long-short combinations of the most volatile assets.
function conditionalCovariance(p::DeelstraBasketPricer, r, forward::AbstractArray{TV}, totalVariance, weight, correlation) where {TV}
    n = length(forward)
    sqrtVar = sqrt.(totalVariance)
    g = r .* sqrtVar
    Σ = correlation .* (sqrtVar * sqrtVar')
    R = Σ - g * g'
    E = expm1.(R)
    if p.nDirections > 0 && n > 1
        eig = eigen(Symmetric(R))
        if eig.values[end] > 1e-10 * tr(Σ)
            ℓ = eig.vectors[:, end] .* sqrt(eig.values[end])
            return g, E, ℓ, expm1.(R - ℓ * ℓ'), true
        end
    end
    return g, E, zeros(TV, n), E, false
end

function priceEuropean(
    p::DeelstraBasketPricer,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV};
    ndev=8.0, isLognormal=1, useM3=false #the call keeps a value for very negative λ when the conditional mean is large there (deep in the money calls, puts by the parity, U-shaped conditional mean): a truncation at 4 standard deviations misses up to Φ(-4)(E[B|Λ]-K), and gives negative puts
)::T where {T,TV}
    n = length(spot)
    δ = zeros(TV, n) #we store δ * S here.
    f = zero(T)
    u1 = zero(T)
    for (i, wi) in enumerate(weight)
        if p.δIndex == 1
            δ[i] = forward[i] * exp(-totalVariance[i] / 2)
        elseif p.δIndex == 2
            δ[i] = spot[i]
        elseif p.δIndex == 3
            δ[i] = forward[i]
        elseif p.δIndex == 4
            δ[i] = one(T)/spot[i]
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
   varGamma = max(varGamma,sqrt(eps(T)))
    volGamma = sqrt(varGamma)
    r ./= volGamma
    #conditional moments of the basket given Λ = λ, with x = λ/σ_Λ and g_i = r_i σ_i √T:
    #  E[Π_{i∈I} q_i S_i | Λ] = Π_{i∈I} A_i(x) Π_{i<j∈I} M_ij,  A_i(x) = q_i F_i exp(-g_i²/2 + g_i x),  M_ij = exp(c_ij - g_i g_j)
    #where c_ij = ρ_ij σ_i σ_j T (I is a multiset). Used for the third moment and for the residual factor.
    g, M, ℓ, Mz, useResidualFactor = conditionalCovariance(p, r, forward, totalVariance, weight, correlation)
    priceCall = zero(Float64)
    if typeof(p.q) == LowerBound
        function objective(λ::T)::Tuple{<:T,<:T,<:T}
            #fΔxΔΔx
            local eS = zero(typeof(λ))
            local DeS = zero(typeof(λ))
            local D2eS = zero(typeof(λ))
            for (i, fi) in enumerate(forward)
                viT = totalVariance[i]
                eSi = weight[i] * fi
                if viT > 0
                    sqrti = sqrt(viT)
                    eSi *= exp(-r[i]^2 * viT / 2 + r[i] * sqrti * (λ) / (volGamma))
                    DeS += eSi * (r[i] * sqrti) / volGamma
                    D2eS += eSi * ((r[i] * sqrti) / volGamma)^2
                end
                eS += eSi
            end
            value = log(eS)-log(strike)
            Dvalue = DeS/eS
            D2value = D2eS/eS - (DeS/eS)^2
            #println(λ," lambda ", eS-strike," ", DeS, " ",D2eS, " ",Dvalue, " ",D2value)
            return (value, value / Dvalue, Dvalue / D2value)
        end
        lambdaMax = dGamma * 2
        if lambdaMax < zero(T)
            lambdaMax = dGamma / 2
        end
        #init guess dGamma
        lambdaOpt = try 
            find_zero(objective, dGamma, Roots.QuadraticInverse(), atol=1e-8)
        catch e
            println("Error while finding lower bound starting with ",dGamma,": ",e)
            dGamma            
        end
        dGammaStar = (lambdaOpt) / volGamma
        #println("dG ",dGammaStar)
        i1 = zero(T)
        for (i, fi) in enumerate(forward)
            if totalVariance[i] > 0
                i1 += weight[i] * fi * normcdf(r[i] * sqrt(totalVariance[i]) - dGammaStar)
            else
                i1 += weight[i] * fi * normcdf(-dGammaStar)
            end
        end
        i1 -= strike * normcdf(-dGammaStar)
        priceCall = i1
    else
        function density(λ::T) where {T}
            varLambda = varGamma
            return exp(-λ^2 / (2 * varLambda)) / sqrt(T(pi) * 2 * varLambda)
        end
        #conditional central moments of the basket from its conditional forwards A and the matrix Ec = exp(conditional
        #covariance) - 1, without cancellation when the conditional variance is small: Var[B] = Σ A_i A_j Ec_ij,
        #E[(B - E[B])³] = Σ A_i A_j A_k t(Ec_ij, Ec_ik, Ec_jk) with t(a, b, c) = ab + ac + bc + abc. The three pairwise
        #terms sum to 3 Σ_i A_i (Ec A)_i², in O(d²): only the term abc is cubic, summed over i ≥ j ≥ k with the
        #multiplicities 1 (i = j = k), 3 (two equal) and 6. Ec is symmetric: the inner loops run over its columns.
        function conditionalCall(A, Ec, fS)
            nA = length(A)
            eS = sum(A)
            variance = zero(eS)
            m3 = zero(eS)
            @inbounds for i = 1:nA
                Ai = A[i]
                EAi = zero(eS)
                @simd for j = 1:nA
                    EAi += Ec[j, i] * A[j]
                end
                variance += Ai * EAi
                p.nMoments == 3 || continue
                Eii = Ec[i, i]
                m3 += 3 * Ai * EAi^2 + Ai^3 * Eii^3
                for j = 1:i-1
                    Aj, Eij = A[j], Ec[j, i]
                    m3 += 3 * Ai * Aj * Eij^2 * (Ai * Eii + Aj * Ec[j, j])
                    acc = zero(eS)
                    @simd for k = 1:j-1
                        acc += A[k] * Ec[k, i] * Ec[k, j]
                    end
                    m3 += 6 * Ai * Aj * Eij * acc
                end
            end
            variance = max(variance, zero(eS))
            if p.nMoments == 3
                return shiftedLognormal3Call(eS, variance, m3, strike)
            end
            return variance <= eps(eS) * eS^2 ? max(eS - strike, zero(eS)) : shiftedLognormalCentralCall(eS, variance, fS, strike)
        end
        #integral over the residual factor Z, given Λ: the conditional forwards are A0_i exp(ℓ_i z - ℓ_i²/2), and the conditional
        #mean m(z) is convex in z. When the variance left is small, the conditional price is close to the intrinsic value
        #(m(z) - K)^+, with a (smoothed) kink at the crossings z* of the strike. At each crossing, we subtract the proxy
        #α v ψ(m'(z*)(z - z*)/v), ψ(u) = uΦ(u) + φ(u), the price of the linearized conditional basket with the conditional
        #standard deviation v at z*, and add its integral E[(m'(Z - z*) + vN)^+] = s ψ(-m' z*/s), s² = m'² + v², exactly.
        #The remainder is smooth and integrated on fixed panels: no knot depends on the parameters, and the price is smooth.
        #The weight α = exp(-(v/m')²) vanishes with all its derivatives when the crossings appear (m' → 0). For two
        #assets (v = 0), the exact intrinsic value is subtracted instead, and the integral is exact.
        function residualFactorCall(A0, fS)
            zMax = T(6) #range of the numerical integration
            zRoot = T(12) #range of the search of the crossings of the strike
            meanZ(z) = sum(@. A0 * exp(ℓ * z - ℓ^2 / 2))
            zmin = all(ℓ .>= 0) ? -zRoot : (all(ℓ .<= 0) ? zRoot : zero(T))
            if zmin == 0 #Newton on the derivative of the convex conditional mean
                for _ = 1:50
                    e = @. A0 * exp(ℓ * zmin - ℓ^2 / 2)
                    step = dot(e, ℓ) / dot(e, ℓ .^ 2)
                    zmin = clamp(zmin - step, -zRoot, zRoot)
                    abs(step) < 1e-10 && break
                end
            end
            #crossings by Newton's method from the outside, monotone for a convex function; ±∞ without crossing
            function crossing(zEnd)
                meanZ(zEnd) > strike || return zEnd < 0 ? T(-Inf) : T(Inf)
                z = zEnd
                for _ = 1:50
                    e = @. A0 * exp(ℓ * z - ℓ^2 / 2)
                    step = (sum(e) - strike) / dot(e, ℓ)
                    z -= step
                    abs(step) < 1e-10 && break
                end
                return z
            end
            ψ(u) = u * normcdf(u) + normpdf(u)
            hasCrossing = meanZ(zmin) < strike
            #per crossing: (z*, m'(z*), v(z*), α); α = 0 without crossing
            proxies = NTuple{4,T}[]
            if hasCrossing
                for zEnd in (-zRoot, zRoot)
                    zc = crossing(zEnd)
                    isinf(zc) && continue
                    A = @. A0 * exp(ℓ * zc - ℓ^2 / 2)
                    slope = dot(A, ℓ)
                    sdc = sqrt(max(dot(A, Mz * A), zero(T)))
                    α = sdc <= sqrt(eps(T)) * strike ? one(T) : exp(-(sdc / slope)^2)
                    push!(proxies, (zc, slope, sdc, α))
                end
            end
            value = zero(T)
            for (zc, slope, sdc, α) in proxies
                if sdc <= sqrt(eps(T)) * strike #exact intrinsic value on the side of the crossing
                    value += zc < zmin ? sum(@. A0 * normcdf(zc - ℓ)) - strike * normcdf(zc) : sum(@. A0 * normcdf(ℓ - zc)) - strike * normcdf(-zc)
                else
                    sc = sqrt(slope^2 + sdc^2)
                    value += α * sc * ψ(-slope * zc / sc)
                end
            end
            function remainder(z)
                A = @. A0 * exp(ℓ * z - ℓ^2 / 2)
                rem = conditionalCall(A, Mz, fS)
                for (zc, slope, sdc, α) in proxies
                    if sdc <= sqrt(eps(T)) * strike
                        if (zc < zmin && z <= zc) || (zc > zmin && z >= zc)
                            rem -= max(sum(A) - strike, zero(T))
                        end
                    else
                        rem -= α * sdc * ψ(slope * (z - zc) / sdc)
                    end
                end
                return rem
            end
            #fixed panels of width 2 (the Gaussian weight is well integrated by a short Gauss-Legendre rule): knots that
            #follow the crossings would move as the square root of the distance to their birth, and spoil the greeks
            for zl = -zMax:2:zMax-2
                value += Math.integrate(p.qz, z -> normpdf(z) * remainder(z), T(zl), T(zl + 2))
            end
            return value
        end
        function expectationValue(λ::T) where {T}
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
            A = @. weight * forward * exp(-g^2 / 2 + g * λ / volGamma)
            return useResidualFactor ? residualFactorCall(A, fS) : conditionalCall(A, M, fS)
        end

        function integrand(λ::T) where {T}
            return expectationValue(λ) * density(λ)
        end

        function lognormalIntegrand(z::T) where {T}
            λ = f * (log(z / f) - sumwbd)
            return integrand(λ) * f / z
        end


        # function integrandM3(λ)
        #     #depends on forward, weight, tvar, r, volGamma,correl, f, sumwbd, strike
        #     eS = zero(Float64)
        #     eS2 = zero(Float64)
        #     eS3 = zero(Float64)
        #     for (i, fi) in enumerate(forward)
        #         tempi = weight[i] * fi
        #         eSi = tempi
        #         viT = totalVariance[i]
        #         sqrti = zero(Float64)
        #         if viT > 0
        #             sqrti = sqrt(viT)
        #             eSi *= exp(-r[i]^2 * viT / 2 + r[i] * sqrti * (λ) / (volGamma))
        #         end
        #         eS += eSi
        #         for j = 1:i-1
        #             vjT = totalVariance[j]
        #             sqrtj = zero(Float64)
        #             if vjT > 0
        #                 sqrtj = sqrt(vjT)
        #             end
        #             sigmaij = sqrt(viT + vjT + 2 * sqrti * sqrtj * correlation[i, j])
        #             rij = (r[i] * sqrti + r[j] * sqrtj) / sigmaij
        #             eFactor = -viT / 2 - vjT / 2 + (1 - rij^2) / 2 * sigmaij^2 + rij * sigmaij * (λ) / (volGamma)
        #             eS2 += 2 * tempi * weight[j] * forward[j] * exp(eFactor)
        #             for k=1:j-1
        #                 vkT = totalVariance[k]
        #                 sqrtk = zero(Float64)
        #                 if vkT > 0
        #                     sqrtk = sqrt(vkT)
        #                 end
        #                 sigmaijk= sqrt(viT + vjT + vkT + 2 * sqrti * sqrtj * correlation[i, j]+ 2 * sqrti * sqrtk * correlation[i, k]+ 2 * sqrtk * sqrtj * correlation[j, k])
        #                 rijk = (r[i] * sqrti + r[j] * sqrtj + r[k]*sqrtk) / sigmaijk

        #                 eFactork =  -viT / 2 - vjT / 2 - vkT / 2 + (1 - rijk^2) / 2 * sigmaijk^2 + rijk * sigmaijk * (λ) / (volGamma)
        #                 eS3 += 6 * weight[k] * forward[k] * exp(eFactork)
        #             end
        #             sigmaijj= sqrt(viT + 4*vjT + 4 * sqrti * sqrtj * correlation[i, j])
        #             rijj = (r[i] * sqrti + 2*r[j] * sqrtj) / sigmaijj                       
        #             eFactorj =  -viT / 2 - vjT + (1 - rijj^2) / 2 * sigmaijj^2 + rijj * sigmaijj * (λ) / (volGamma)
        #             eS3 += 3 * tempi * weight[j]^2 * forward[j]^2 * exp(eFactorj) #i;j=k
        #             sigmaiji= sqrt(4*viT + vjT + 4 * sqrti * sqrtj * correlation[i, j])
        #             riji = (2*r[i] * sqrti + r[j] * sqrtj) / sigmaiji                    
        #             eFactori =  -viT  - vjT/2 + (1 - riji^2) / 2 * sigmaiji^2 + riji * sigmaiji * (λ) / (volGamma)
        #             eS3 += 3 * tempi^2 * weight[j] * forward[j] * exp(eFactori) #j;i=k
        #         end
        #         eS2 += tempi^2 * exp(-viT +  2*(1 - r[i]^2) * viT + r[i] * 2 * sqrti * (λ) / (volGamma))
        #         eS3 += tempi^3 * exp(-3 * viT / 2 + 9 * (1 - r[i]^2) / 2 * viT + r[i] * 3 * sqrti * (λ) / (volGamma))  #i=j=k
        #     end

        #     fS = zero(Float64)
        #     if p.fIndex == 1
        #         #zero
        #     elseif p.fIndex == 2
        #         lnGF = λ / f + sumwbd
        #         fS = f * (1 + lnGF)
        #     else #p.fIndex >= 3
        #         #most accurate according to Deelstra paper Fig.2
        #         lnGF = λ / f + sumwbd
        #         gF = exp(lnGF)
        #         fS = f * gF
        #     end
        #     euphs2 = eS - fS
        #     if euphs2 < eps(Float64)
        #         fS = 0
        #         euphs2 = eS
        #     end
        #     e2ups2 = eS2 - 2 * fS * eS + fS^2

        #     if e2ups2 < eps(Float64)
        #         e2ups2 = eps(Float64)
        #     end

        #     e3ups3 = eS3 - 3 * fS * eS2 + 3*fS^2*eS - fS^3

        #     varLambda = varGamma
        #     pdf = exp(-λ^2 / (2 * varLambda)) / sqrt(Float64(pi) * 2 * varLambda)

        #     a0 = euphs2
        #     m3 = e3ups3 - euphs2 * e2ups2 - 2*euphs2*(e2ups2-euphs2*euphs2)
        #     a2sign = 1
        #     if m3  < 0
        #         a2sign = -1
        #     end
        #     #TODO verify third moment. find good criteria for a3.
        #     #a2Poly = Polynomials.Polynomial([-m3, 6*(e2ups2-a0^2), 0, -4, 16*a2sign/a0 ])
        #     #a2Roots = roots(a2Poly)
        #     #println(λ," roots a2 = ",a2Roots)
        #     #a2 = zero(Float64)
        #     #for a2r in a2Roots
        #     #    if imag(a2r) < eps(Float64) && sign(a2r) == a2sign 
        #     #        a2 = real(a2r)
        #     #        break
        #     #    end
        #     #end
        #     #a3 = sqrt(2*abs(a2)^3)/(36*a0)
        #     #a1 = sqrt(abs(e2ups2 - a0^2  - 2*a2^2- 6*a3^2))
        #     #println(λ," as = ",a0," ",a1," ",a2," ",a3)
        #     #a3 = 
        #     #a1,a2,a3
        #     a1 = sqrt(abs(e2ups2 - a0^2))
        #     a2 = 0.0
        #     a3 = 0.0
        #     b0 = a0 - a2
        #     b1 = a1 - 3*a3
        #     b2 = a2
        #     b3 = a3
        #     g = Polynomials.Polynomial([b0-strike+fS, b1, b2, b3])
        #     gRoots = roots(g)
        #     ck = roots(g)[1]
        #     for gr in gRoots 
        #         if imag(gr) < eps(Float64) 
        #             ck = real(gr)
        #             break
        #         end
        #     end
        #         m0 = normcdf(-ck)
        #     m1 = normpdf(ck)
        #     m2 = m0 + ck * m1
        #     m3 = 2*m1 + ck^2 * m1
        #     return (b0*m0+ b1*m1+b2*m2+b3*m3 - m0*(strike-fS)) * pdf
        # end

        λMin = -ndev * sqrt(varGamma)
        i2 = zero(T)
        if dGamma > λMin
            a = λMin
            b = dGamma
            if isLognormal == 1
                #we have dGamma = f * (log(strike / f) - sumwbd), thus
                za = f * exp(a / f + sumwbd)
                zb = f * exp(b / f + sumwbd)
                if isSplit(p.q)
                    i2 = zero(T)
                    nk = 1
                    for k = 1:nk
                        zak = za + (zb - za) * (k - 1) / nk
                        zbk = zak + (zb - za) / nk
                        expectationValue1 = function (y)
                            z = (zbk - zak) / 2 * y + (zbk + zak) / 2
                            λ = f * (log(z / f) - sumwbd)
                            expectationValue(λ)
                        end
                        if typeof(p.q) == Chebyshev{Float64,1}
                            n = Int(round(p.q.N / nk))
                            fValues = zeros(T, n)
                            chebnodevalues!(fValues, expectationValue1)
                            coeffs = zeros(T, n)
                            chebcoeff!(coeffs, fValues)
                            #TODO implement closed form forumula 
                            i2 += Math.integrate(GaussKronrod(1e-15), z -> chebinterp(coeffs, (2z - (zbk + zak)) / (zbk - zak)) * density(f * (log(z / f) - sumwbd)) * f / z, zak, zbk)
                        elseif typeof(p.q) == Chebyshev{Float64,2}
                            n = Int(round(p.q.N / nk))
                            fValues = zeros(T, n)
                            cheb2values!(fValues, expectationValue1)
                            coeffs = zeros(T, n)
                            cheb2coeff!(coeffs, fValues)
                            ## ta = collect(range(-1.0, stop=1.0, length=129))
                            ## println("ta=", ta)
                            ## println("fa=", [expectationValue1(z) for z in ta])
                            ## println("ga=", [chebinterp(coeffs, z) for z in ta])
                            i2 += Math.integrate(GaussKronrod(1e-15), z -> cheb2interp(coeffs, (2z - (zbk + zak)) / (zbk - zak)) * density(f * (log(z / f) - sumwbd)) * f / z, zak, zbk)
                        end
                    end
                else
                    i2 = Math.integrate(p.q, lognormalIntegrand, za, zb)
                end
            else
                if isSplit(p.q)
                    expectationValue1 = function (y)
                        λ = (b - a) / 2 * y + (b + a) / 2
                        expectationValue(λ)
                    end
                    if kind(p.q) == 1
                        n = p.q.N
                        fValues = zeros(T, n)
                        chebnodevalues!(fValues, expectationValue1)
                        coeffs = zeros(T, n)
                        chebcoeff!(coeffs, fValues)
                        #TODO implement closed form forumula 
                        i2 = Math.integrate(GaussKronrod(1e-15), z -> chebinterp(coeffs, (2z - (b + a)) / (b - a)) * density(z), a, b)
                    elseif kind(p.q) == 2
                        n = p.q.N
                        fValues = zeros(T, n)
                        cheb2values!(fValues, expectationValue1)
                        coeffs = zeros(T, n)
                        cheb2coeff!(coeffs, fValues)
                        ## ta = collect(range(-1.0, stop=1.0, length=129))
                        ## println("ta=", ta)
                        ## println("fa=", [expectationValue1(z) for z in ta])
                        ## println("ga=", [chebinterp(coeffs, z) for z in ta])
                        i2 = Math.integrate(GaussKronrod(1e-15), z -> cheb2interp(coeffs, (2z - (b + a)) / (b - a)) * density(z), a, b)
                    end
                else
                    i2 = Math.integrate(p.q, integrand, a, b)
                end
            end
        end
        dGammaStar = (dGamma) / sqrt(varGamma)

        i1 = zero(T)
        for (i, fi) in enumerate(forward)
            if totalVariance[i] > 0
                i1 += weight[i] * fi * normcdf(r[i] * sqrt(totalVariance[i]) - dGammaStar)
            else
                i1 += weight[i] * fi * normcdf(-dGammaStar)
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

