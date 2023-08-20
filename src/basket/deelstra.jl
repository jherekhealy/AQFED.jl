using AQFED.Math
export DeelstraBasketPricer, DeelstraLBBasketPricer, priceEuropean
using LinearAlgebra
using Roots
using Polynomials

abstract type DeelstraBasketQuadrature
end
struct DeelstraBasketPricer
    δIndex::Int
    fIndex::Int
    q::DeelstraBasketQuadrature
end

using FastGaussQuadrature
using QuadGK
import DoubleExponentialFormulas: quadde, quaddeo
export LowerBound, GaussLegendre, DoubleExponential, TanhSinh, GaussKronrod
struct LowerBound <: DeelstraBasketQuadrature
end
struct GaussLegendre <: DeelstraBasketQuadrature
    N::Int
    x::Vector{Float64} #quadrature abscissae in -1, 1
    w::Vector{Float64} #quadrature weights
    function GaussLegendre(N::Int=33) 
        if N==33
            xG=[0.9974246942464552, 0.9864557262306425, 0.9668229096899927, 0.9386943726111684, 0.9023167677434336, 0.8580096526765041, 0.8061623562741665, 0.7472304964495622, 0.6817319599697428, 0.610242345836379, 0.5333899047863476, 0.4518500172724507, 0.36633925774807335, 0.27760909715249704, 0.18643929882799157, 0.0936310658547334, 0, -0.0936310658547334, -0.18643929882799157, -0.27760909715249704, -0.36633925774807335, -0.4518500172724507, -0.5333899047863476, -0.610242345836379, -0.6817319599697428, -0.7472304964495622, -0.8061623562741665, -0.8580096526765041, -0.9023167677434336, -0.9386943726111684, -0.9668229096899927, -0.9864557262306425, -0.9974246942464552]
            wG = [0.0066062278475874535, 0.015321701512934681, 0.023915548101749465, 0.03230035863232906, 0.04040154133166957, 0.04814774281871171, 0.05547084663166358, 0.0623064825303174, 0.06859457281865682, 0.07427985484395423, 0.07931236479488682, 0.08364787606703872, 0.08724828761884422, 0.09008195866063856, 0.09212398664331695, 0.09335642606559616, 0.09376844616020999, 0.09335642606559616, 0.09212398664331695, 0.09008195866063856, 0.08724828761884422, 0.08364787606703872, 0.07931236479488682, 0.07427985484395423, 0.06859457281865682, 0.0623064825303174, 0.05547084663166358, 0.04814774281871171, 0.04040154133166957, 0.03230035863232906, 0.023915548101749465, 0.015321701512934681, 0.0066062278475874535]
        else
            xG, wG = gausslegendre(N)
        end
        new(N,xG,wG) 
    end
end

function integrate(q::GaussLegendre, integrand, a::T, b::T)::T where {T}
    bma2 = (b - a) / 2
    bpa2 = (b + a) / 2
    i2 = bma2 * dot(q.w, integrand.(bma2 .* q.x .+ bpa2))
    # i2 = zero(T)
    # @sync Threads.@threads for i=1:length(p.w)
    #     i2 += p.w[i] * integrand(bma2 * p.x[i] + bpa2)
    # end
    # i2*=bma2
    return i2
    end

    struct GaussLegendreParallel <: DeelstraBasketQuadrature
        N::Int
        x::Vector{Float64} #quadrature abscissae in -1, 1
        w::Vector{Float64} #quadrature weights
        function GaussLegendreParallel(N::Int=33) xG, wG = gausslegendre(N); new(N,xG,wG) end
    end
    
    function integrate(q::GaussLegendreParallel, integrand, a::Float64, b::Float64)::Float64
        bma2 = (b - a) / 2
        bpa2 = (b + a) / 2
        i2 = zero(T)
         @sync Threads.@threads for i=1:length(p.w)
             i2 += p.w[i] * integrand(bma2 * p.x[i] + bpa2)
        end
        i2*=bma2
        return i2
        end
    
struct GaussKronrod <: DeelstraBasketQuadrature
    rtol::Float64
    GaussKronrod(rtol::Float64=1e-8) = new(rtol)
end

function integrate(q::GaussKronrod, integrand, a::T, b::T)::T where {T} 
    i2,err = quadgk(integrand,a,b,rtol=q.rtol)
    return i2
end
struct DoubleExponential <: DeelstraBasketQuadrature
    rtol::Float64
    DoubleExponential(rtol::Float64=1e-8) = new(rtol)
end

function integrate(q::DoubleExponential, integrand, a::T, b::T)::T where {T} 
    i2,err = quadde(integrand,a,b,rtol=q.rtol)
    return i2
end

struct TanhSinh{T} <: DeelstraBasketQuadrature
    h::T
    y::Array{T,1}
    w::Array{T,1}
    tol::T
    isParallel::Bool
    function TanhSinh(n::Int, tol::T,isParallel::Bool=false) where {T}
        y = Vector{T}(undef, 2 * n + 1)
        w = Vector{T}(undef, 2 * n + 1)
        if n <= 0
            throw(DomainError(n, "the number of points must be > 0"))
        end
        h = convert(T, lambertW(Float64(2 * pi * n)) / n)
        for i = -n:n
            q = exp(-sinh(i * h)*pi)
            yi = 2 * q / (1 + q)
            y[n+i+1] = yi
            w[n+i+1] = yi * h * pi * cosh(i * h) / (1 + q)
            if isnan(w[n+i+1]) || w[n+i+1] < tol
                # overflow is expected to happen for large n
                # the correct way to address it is to set the weights to zero (ignore the points)
                w[n+i+1] = Base.zero(T)
                y[n+i+1] = one(T)
            end
        end
        return new{T}(h, y, w, tol, isParallel)
    end
end

function integrate(q::TanhSinh{T}, integrand,a::T,b::T)::T where {T}
    if b <= a 
		return zero(T)
	end
    n = trunc(Int, (length(q.w) - 1) / 2)
    I = Base.zero(T)
    if q.isParallel
        @sync Threads.@threads    for i = 1:2*n+1
            yi = q.y[i]-one(T)
            zi = (a+b)/2 + (b-a)/2*yi
            if q.w[i] != 0
                fyi = integrand(zi)
                if fyi == 0 || isnan(fyi)
                    break
                else
                    I += q.w[i] * fyi
                end
            end
        end
    else
    for i = 1:2*n+1
        yi = q.y[i]-one(T)
        zi = (a+b)/2 + (b-a)/2*yi
        if q.w[i] != 0
            fyi = integrand(zi)
            if fyi == 0 || isnan(fyi)
                break
            else
                I += q.w[i] * fyi
            end
        end
    end
end
    return I*(b - a) / 2
end



function DeelstraBasketPricer(δIndex::Int, fIndex::Int; q::DeelstraBasketQuadrature=GaussLegendre(33))
    DeelstraBasketPricer(δIndex, fIndex, q)
end


function DeelstraLBBasketPricer(δIndex::Int, fIndex::Int)
    DeelstraBasketPricer(δIndex, fIndex, LowerBound())
end

DeelstraBasketPricer() = DeelstraBasketPricer(3,3)
DeelstraLBBasketPricer() = DeelstraLBBasketPricer(3,3)


function priceEuropean(
    p::DeelstraBasketPricer,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}; useM3 = false
)::T where {T, TV}
    n = length(spot)
    δ = zeros(TV, n) #we store δ * S here.
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
    if typeof(p.q)  == LowerBound
        function objective(λ::T)::Tuple{<:T,<:T,<:T}
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
        if lambdaMax < zero(T)
            lambdaMax = dGamma / 2
        end
        #init guess dGamma
        lambdaOpt = find_zero(objective, dGamma, Roots.Halley(), atol=1e-8)
        dGammaStar = (lambdaOpt) / volGamma
        #println("dG ",dGammaStar)
        i1 = zero(T)
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
        function integrand(λ::T) where {T}
            #depends on forward, weight, tvar, r, volGamma,correl, f, sumwbd, strike
            eS = zero(T)
            eS2 = zero(T)
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
      
      
        λMin = -3 * sqrt(varGamma)
        i2 = zero(T)
        if dGamma > λMin
            a = λMin
            b = dGamma
            i2 = integrate(p.q,integrand,a,b)
        end
        dGammaStar = (dGamma) / sqrt(varGamma)

        i1 = zero(T)
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

