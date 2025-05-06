export VorstGeometricExpansion
struct VorstGeometricExpansion <: BasketPricer
    order::Int
end
function priceEuropean(
    p::VorstGeometricExpansion,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}
)::T where {T,TV}
    signc = if isCall
        1
    else
        -1
    end
    nAsset = length(totalVariance)
    # price = priceEuropean(p, isCall, strike, discountFactor, spots, forward, totalVariance, weight, correlation)
    A = weight' * forward
    a = @. (weight * forward) / A
    vtilde2 = zero(TV)
    covar = zeros(TV, size(correlation))
    for i = 1:nAsset
        vi = totalVariance[i]
        for j = 1:nAsset
            vj = totalVariance[j]
            covarij = sqrt(vi * vj) * correlation[i, j]
            covar[i, j] = covarij
            product = a[i] * a[j] * covarij
            vtilde2 += product
        end

    end
    if (A <= zero(T)) 
        return max(signc*(A-strike),zero(T))
    end
    #println("A ",A," ",vtilde2)
    mtilde = -vtilde2 / 2
    vtilde = sqrt(vtilde2)
    strikeScaled = strike / A
    logStrikeScaled = log(strikeScaled)
    #strikeAdjustment = 0.0; mtilde = - vtilde2/2 #lognormal with geom vol
    d1 = (mtilde - logStrikeScaled + vtilde2) / vtilde
    d2 = d1 - vtilde
    price = signc * A * (normcdf(signc * d1) - strikeScaled * normcdf(signc * d2))
    #above is order 0 price = Geometric approx with adjusted forward to match basket forward (instead of adjusting the strike as in Gentle)
    if p.order > 0
        tvi = zeros(TV, length(totalVariance))
        @inbounds for i = 1:nAsset
            @inbounds for l = 1:nAsset
                tvi[i] += a[l] * covar[i, l]
            end
        end
        mtildep = mtilde + vtilde2
        d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
        d2 = d1 - vtilde
        eTerm = -signc * (normcdf(signc * d2))
        for i = 1:nAsset
            tv = tvi[i]
            mtildei = mtilde + tv
            d1 = (mtildei - logStrikeScaled + vtilde2) / vtilde
            d2 = d1 - vtilde
            eTerm -= -a[i] * signc * (normcdf(signc * d2))
        end
        correction = A * eTerm
        price += correction
        if p.order > 1
            mtildep = mtilde + 2vtilde2
            d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
            d2 = d1 - vtilde
            eTerm = 0.5 /(strikeScaled * vtilde)*  (normpdf(d2))*exp(vtilde2) #term in product^2.
            
            #now term in sum squared
            for i = 1:nAsset
                factori = a[i] / (strikeScaled * vtilde)
                for j = 1:i-1
                    tv = tvi[i] + tvi[j] 
                    d2 = (mtilde + tv - logStrikeScaled) / vtilde
                    eTerm += factori * a[j] *(normpdf(d2)) * exp(covar[i, j])
                end
                d2 = (mtilde + 2tvi[i] - logStrikeScaled) / vtilde
                eTerm += factori * a[i] / 2 * (normpdf(d2)) *exp(totalVariance[i])

            end
            #finally term in product*sum.
            for i = 1:nAsset
                tv = tvi[i]
                mtildeij = mtilde + tv + vtilde2
                d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm -= a[i] * (normpdf(d2)) / (strikeScaled * vtilde) * exp(tv)
            end
            price += A * eTerm
            if p.order > 2
                #d3CallBS/dK = phi(d2)*(d2/vtilde - 1)/(K^2*vtilde)
                mtildep = mtilde + 3vtilde2
                d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm = -normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(3vtilde2) #term in product^3.
                eTermS3 = zero(TV)

                for i = 1:nAsset
                    @inbounds for j = 1:i-1
                        factorij = 6 * a[i] * a[j] / (strikeScaled^2 * vtilde)
                        @inbounds for l = 1:j-1
                            tv = tvi[i] + tvi[j] + tvi[l]
                            d2v = (mtilde + tv - logStrikeScaled) / vtilde2
                            eTermS3 += a[l] * factorij * normpdf(d2v * vtilde) * (d2v - 1) * exp(covar[i, j] + covar[i, l] + covar[l, j])
                        end
                        d2 = (mtilde + 2 * tvi[j] + tvi[i] - logStrikeScaled) / vtilde
                        eTermS3 += factorij / 2 * a[j] * normpdf(d2) * (d2 / vtilde - 1) * exp(2covar[i, j] + totalVariance[j])
                        d2 = (mtilde + 2 * tvi[i] + tvi[j] - logStrikeScaled) / vtilde
                        eTermS3 += factorij / 2 * a[i] * normpdf(d2) * (d2 / vtilde - 1) * exp(2covar[i, j] + totalVariance[i])
                    end
                    d2 = (mtilde + 3 * tvi[i] - logStrikeScaled) / vtilde
                    eTermS3 += a[i]^3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(3totalVariance[i])
                end
                eTerm += eTermS3
                #now term in sum^2* pi.
                for i = 1:nAsset
                    @inbounds for j = 1:i-1
                        tv = tvi[i] + tvi[j]
                        mtildeij = mtilde + tv + vtilde2
                        d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                        d2 = d1 - vtilde
                        eTerm -= a[i] * a[j] * 6 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(tv + covar[i, j])
                    end
                    tv = 2tvi[i]
                    mtildeij = mtilde + tv + vtilde2
                    d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                    d2 = d1 - vtilde
                    eTerm -= a[i]^2 * 3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(tv + totalVariance[i])
                end
                #now term in sum*pi^2
                for i = 1:nAsset
                    tv = tvi[i]
                    mtildeij = mtilde + tv + 2vtilde2
                    d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                    d2 = d1 - vtilde
                    eTerm += a[i] * 3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(2tv + vtilde2)
                end
                price -= A * eTerm / 6

            end
        end
    end
    return price * discountFactor
end

export VorstLevyExpansion
struct VorstLevyExpansion <: BasketPricer
    order::Int
end
function priceEuropean(
    p::VorstLevyExpansion,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}
)::T where {T,TV}
    signc = if isCall
        1
    else
        -1
    end
    nAsset = length(totalVariance)
    # price = priceEuropean(p, isCall, strike, discountFactor, spots, forward, totalVariance, weight, correlation)
    A = weight' * forward
    a = @. (weight * forward) / A
    evtilde2 = zero(TV)
    gvtilde2 = zero(TV)
    covar = zeros(TV, size(correlation))
    for i = 1:nAsset
        vi = totalVariance[i]
        for j = 1:nAsset
            vj = totalVariance[j]
            covarij = sqrt(vi * vj) * correlation[i, j]
            covar[i, j] = covarij
            product = a[i] * a[j] * exp(covarij)
            gvtilde2 += a[i] * a[j] * covarij
            evtilde2 += product
        end

    end
    #Note, if covariance is not PSD evtilde2 < 1. println("evtilde2 ",evtilde2)
    vtilde = sqrt(log(evtilde2))
    afactor = (vtilde / sqrt(gvtilde2))
    ag = a .* afactor
    vtilde2 = vtilde^2
    mtilde = -vtilde2 / 2
    strikeScaled = strike / A
    logStrikeScaled = log(strikeScaled)
    #strikeAdjustment = 0.0; mtilde = - vtilde2/2 #lognormal with geom vol
    d1 = (mtilde - logStrikeScaled + vtilde2) / vtilde
    d2 = d1 - vtilde
    price = signc * A * (normcdf(signc * d1) - strikeScaled * normcdf(signc * d2))
    #above is order 0 price = Geometric approx with adjusted forward to match basket forward (instead of adjusting the strike as in Gentle)
    if p.order > 0
        tvi = zeros(TV, length(totalVariance))
        @inbounds for i = 1:nAsset
            @inbounds for l = 1:nAsset
                tvi[i] += ag[l] * covar[i, l]
            end
        end
        #G*h(G)
        mtildep = mtilde + vtilde2
        d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
        d2 = d1 - vtilde
        eTerm = -signc * (normcdf(signc * d2))
        #Si*h(G)  =>  covar(log A,log Si). Levy process G = e^{vtilde*W_G} but we don't know corr(W_G,W_i)
        #solution: adjust ai part of G such that covar(GVorst)=covar(A) exactly.
        for i = 1:nAsset
            tv = tvi[i]
            mtildei = mtilde + tv
            d1 = (mtildei - logStrikeScaled + vtilde2) / vtilde
            d2 = d1 - vtilde
            eTerm -= -a[i] * signc * (normcdf(signc * d2))
        end
        correction = A * eTerm
        price += correction
        if p.order > 1
            mtildep = mtilde + 2vtilde2
            d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
            d2 = d1 - vtilde
            eTerm = 0.5 * normpdf(d2) / (strikeScaled * vtilde) * exp(vtilde2) #term in product^2.
            #now term in sum squared
            for i = 1:nAsset
                factori = a[i] / (strikeScaled * vtilde)
                for j = 1:i-1
                    tv = tvi[i] + tvi[j]
                    d2 = (mtilde + tv - logStrikeScaled) / vtilde
                    eTerm += factori * a[j] * normpdf(d2) * exp(covar[i, j])
                end
                d2 = (mtilde + 2tvi[i] - logStrikeScaled) / vtilde
                eTerm += factori * a[i] / 2 * normpdf(d2) * exp(totalVariance[i])

            end
            #finally term in product*sum.
            for i = 1:nAsset
                tv = (tvi[i])
                mtildeij = mtilde + tv + vtilde2
                d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm -= a[i] * normpdf(d2) / (strikeScaled * vtilde) * exp(tv)
            end
            price += A * eTerm
            if p.order > 2
                #d3CallBS/dK = phi(d2)*(d2/vtilde - 1)/(K^2*vtilde)
                mtildep = mtilde + 3vtilde2
                d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm = -normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(3vtilde2) #term in product^3.
                eTermS3 = zero(TV)

                for i = 1:nAsset
                    @inbounds for j = 1:i-1
                        factorij = 6 * a[i] * a[j] / (strikeScaled^2 * vtilde)
                        @inbounds for l = 1:j-1
                            tv = tvi[i] + tvi[j] + tvi[l]
                            d2v = (mtilde + tv - logStrikeScaled) / vtilde2
                            eTermS3 += a[l] * factorij * normpdf(d2v * vtilde) * (d2v - 1) * exp(covar[i, j] + covar[i, l] + covar[l, j])
                        end
                        d2 = (mtilde + 2 * tvi[j] + tvi[i] - logStrikeScaled) / vtilde
                        eTermS3 += factorij / 2 * a[j] * normpdf(d2) * (d2 / vtilde - 1) * exp(2covar[i, j] + totalVariance[j])
                        d2 = (mtilde + 2 * tvi[i] + tvi[j] - logStrikeScaled) / vtilde
                        eTermS3 += factorij / 2 * a[i] * normpdf(d2) * (d2 / vtilde - 1) * exp(2covar[i, j] + totalVariance[i])
                    end
                    d2 = (mtilde + 3 * tvi[i] - logStrikeScaled) / vtilde
                    eTermS3 += a[i]^3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(3totalVariance[i])
                end
                eTerm += eTermS3
                #now term in sum^2* pi.
                for i = 1:nAsset
                    @inbounds for j = 1:i-1
                        tv = tvi[i] + tvi[j]
                        mtildeij = mtilde + tv + vtilde2
                        d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                        d2 = d1 - vtilde
                        eTerm -= a[i] * a[j] * 6 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(tv + covar[i, j])
                    end
                    tv = 2tvi[i]
                    mtildeij = mtilde + tv + vtilde2
                    d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                    d2 = d1 - vtilde
                    eTerm -= a[i]^2 * 3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(tv + totalVariance[i])
                end
                #now term in sum*pi^2
                for i = 1:nAsset
                    tv = tvi[i]
                    mtildeij = mtilde + tv + 2vtilde2
                    d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                    d2 = d1 - vtilde
                    eTerm += a[i] * 3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(2tv + vtilde2)
                end
                price -= A * eTerm / 6

            end
        end
    end
    return price * discountFactor
end

export ShiftedGeometricExpansion
struct ShiftedGeometricExpansion <: BasketPricer
    order::Int
    nMoments::Int
end
#=
max(product(S_i^star(T)^ai + (1-E[G]) - K^star ),0)
may be interpreted as shifted lognormal but leads to log( (Kstar - (1e-EG)) / product)
what if we adjust ai first moment and then alpha from second moment? w
    E[G] = A*e^{-1/2sum a_i sigma_i^2 T  + 1/2 sum sum a_i a_j sigma_i sigma_j rho_ij T} = e^{  -mtilde}
    a_i = a_i +b or 
    ai=ai*b =>  b(- sum ai sigma_i^2 T + b*sumsum a_ia_j sisjT ) = 0 
=#

function priceEuropean(
    p::ShiftedGeometricExpansion,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}
)::T where {T,TV}
    signc = if isCall
        1
    else
        -1
    end
    nAsset = length(totalVariance)
    # price = priceEuropean(p, isCall, strike, discountFactor, spots, forward, totalVariance, weight, correlation)
    A = weight' * forward
    a = @. (weight * forward) / A
    strikeScaled = strike / A
    evtilde2 = zero(TV)
    gvtilde2 = zero(TV)
    covar = zeros(TV, size(correlation))
    for i = 1:nAsset
        vi = totalVariance[i]
        for j = 1:nAsset
            vj = totalVariance[j]
            covarij = sqrt(vi * vj) * correlation[i, j]
            covar[i, j] = covarij
            product = a[i] * a[j] * exp(covarij)
            gvtilde2 += a[i] * a[j] * covarij
            evtilde2 += product
        end
    end
    b, c1,c2 = if p.nMoments == 3
        sbar = a
        n = length(sbar)
        abar = zeros(T, n)
        rhobar = zeros(T, (n, n))
        sqrtVar = sqrt.(totalVariance)
        for j = 1:size(correlation, 2)
            for i = 1:size(correlation, 1)
                rhobar[i, j] = exp(correlation[i, j] * sqrtVar[i] * sqrtVar[j])
            end
        end
        abar = rhobar' * sbar
        u1 = one(T)
        u2 = zero(T)
        u3 = zero(T)
        for (i, si) = enumerate(sbar)
            #  u1 += si
            for (j, sj) = enumerate(sbar)
                temp = si * sj * rhobar[i, j]
                u2 += temp
                for (k, sk) = enumerate(sbar)
                    temp3 = temp * sk * rhobar[i, k] * rhobar[j, k]
                    u3 += temp3
                end
            end
        end
        u3n = u3 - 3u2 * u1 + 2u1^3
        ς = sqrt(u2 - u1^2)
        η = u3n / ς^3
        sqrt3 = (8 + 4η^2 + 4 * sqrt(4η^2 + η^4))^(1 / 3)
        w = sqrt3 / 2 + 2 / sqrt3 - 1
        b = 1 / sqrt(log(w))
        aa = b / 2 * log((w) * (w - 1) / ς^2)
        d = sign(η)
        c = d * u1 - exp((1 / (2b) - aa) / b)
        #Q = aa + b*log((strikeScaled-c)/d)
        #price = signc * ((u1 - c) * normcdf(-signc * (Q - 1 / b)) - (strikeScaled - c) * normcdf(-signc * Q))
        # c = shift.   Y = X-c is lognormal => E[X-K]+ = E[Y-(K-c)]+ = Black(Y(0),K-c)=Black(X(0)-c,K-c)
        # d2 = -Q, d1 = -Q + 1/b, vtilde = 1/b, mtilde=-vtilde^2 / 2, logStrikeScaled = -d1*vtilde+mtilde+vtilde2
        one(T) / b, c,c
    elseif p.nMoments == 2 #Levy
        b = sqrt(log(evtilde2))
        c = zero(T)
        b, c,c
    elseif p.nMoments == 1 #Vorst + strike shift
        b = sqrt(gvtilde2)
        elng = - (totalVariance' * a) /2
        c = (1-exp(elng + gvtilde2 / 2)) # PI + C - K => C + PI = 1
        # println("c ",c)
        b, c,c
    elseif p.nMoments == 0
        b = sqrt(gvtilde2)
        elng = - (totalVariance' * a) /2
        c1 = (1-exp(elng + gvtilde2 / 2)) # PI + C - K => C + PI = 1
        c2 = zero(T)
        b,c2 ,c2 
    end
    afactor = b / (sqrt(gvtilde2))
    ag = a .* afactor
    vtilde2 = gvtilde2 * afactor^2
    vtilde = sqrt(vtilde2)
    mtilde = -vtilde2 / 2
    logStrikeScaled = log((strikeScaled - c2) / (1 - c1))
    #strikeAdjustment = 0.0; mtilde = - vtilde2/2 #lognormal with geom vol
    d1 = (mtilde - logStrikeScaled + vtilde2) / vtilde
    d2 = d1 - vtilde
    price = signc * A * ((1 - c1) * normcdf(signc * d1) - (strikeScaled - c2) * normcdf(signc * d2))
    #above is order 0 price = Geometric approx with adjusted forward to match basket forward (instead of adjusting the strike as in Gentle)
    if p.order > 0
        tvi = zeros(TV, length(totalVariance))
        @inbounds for i = 1:nAsset
            @inbounds for l = 1:nAsset
                tvi[i] += ag[l] * covar[i, l]
            end
        end
        #G*h(G)
        eTerm = -signc * c2 * normcdf(signc * d2)
        mtildep = mtilde + vtilde2
        d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
        d2 = d1 - vtilde
        eTerm += -signc * ((1 - c1) * normcdf(signc * d2))
        #Si*h(G)  =>  covar(log A,log Si). Levy process G = e^{vtilde*W_G} but we don't know corr(W_G,W_i)
        #solution: adjust ai part of G such that covar(GVorst)=covar(A) exactly.
        for i = 1:nAsset
            tv = tvi[i]
            mtildei = mtilde + tv
            d1 = (mtildei - logStrikeScaled + vtilde2) / vtilde
            d2 = d1 - vtilde
            eTerm -= -a[i] * signc * (normcdf(signc * d2))
        end
        correction = A * eTerm
        price += correction
        if p.order > 1
            divisor = (strikeScaled - c2) * vtilde
            d1 = (mtilde - logStrikeScaled + vtilde2) / vtilde
            d2 = d1 - vtilde
            eTerm = 0.5 * c2^2 * normpdf(d2) / divisor
            mtildep = mtilde + 2vtilde2
            d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
            d2 = d1 - vtilde
            eTerm += 0.5 * (1 - c1)^2 * normpdf(d2) / divisor * exp(vtilde2) #term in product^2.
            #term in 2*c*(1-c)
            mtildep = mtilde + vtilde2
            d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
            d2 = d1 - vtilde
            eTerm += 0.5 * 2 * c2 * (1 - c1) * normpdf(d2) / divisor #term in product*c

            #now term in sum squared
            for i = 1:nAsset
                factori = a[i] / divisor
                for j = 1:i-1
                    tv = tvi[i] + tvi[j]
                    d2 = (mtilde + tv - logStrikeScaled) / vtilde
                    eTerm += factori * a[j] * normpdf(d2) * exp(covar[i, j])
                end
                d2 = (mtilde + 2tvi[i] - logStrikeScaled) / vtilde
                eTerm += factori * a[i] / 2 * normpdf(d2) * exp(totalVariance[i])

            end
            #finally term in product*sum.
            for i = 1:nAsset
                tv = (tvi[i])
                mtildeij = mtilde + tv + vtilde2
                d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm -= a[i] * (1 - c1) * normpdf(d2) / divisor * exp(tv)
                mtildeij = mtilde + tv
                d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm -= a[i] * c2 * normpdf(d2) / divisor
            end
            price += A * eTerm
            if p.order > 2
                divisor = (strikeScaled - c2)^2 * vtilde
                #d3CallBS/dK = phi(d2)*(d2/vtilde - 1)/(K^2*vtilde)
                mtildep = mtilde + 3vtilde2
                d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm = -(1 - c1)^3 * normpdf(d2) / divisor * (d2 / vtilde - 1) * exp(3vtilde2) #term in product^3.
                d1 = (mtilde - logStrikeScaled + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm += -c2^3 * normpdf(d2) / divisor * (d2 / vtilde - 1)
                mtildep = mtilde + vtilde2
                d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm += -3 * (1 - c1) * c2^2 * normpdf(d2) / divisor * (d2 / vtilde - 1)
                mtildep = mtilde + 2vtilde2
                d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm += -3 * (1 - c1)^2 * c2 * normpdf(d2) / divisor * (d2 / vtilde - 1) * exp(vtilde2)
                eTermS3 = zero(TV)
                for i = 1:nAsset
                    @inbounds for j = 1:i-1
                        factorij = 6 * a[i] * a[j] / divisor
                        @inbounds for l = 1:j-1
                            tv = tvi[i] + tvi[j] + tvi[l]
                            d2v = (mtilde + tv - logStrikeScaled) / vtilde2
                            eTermS3 += a[l] * factorij * normpdf(d2v * vtilde) * (d2v - 1) * exp(covar[i, j] + covar[i, l] + covar[l, j])
                        end
                        d2 = (mtilde + 2 * tvi[j] + tvi[i] - logStrikeScaled) / vtilde
                        eTermS3 += factorij / 2 * a[j] * normpdf(d2) * (d2 / vtilde - 1) * exp(2covar[i, j] + totalVariance[j])
                        d2 = (mtilde + 2 * tvi[i] + tvi[j] - logStrikeScaled) / vtilde
                        eTermS3 += factorij / 2 * a[i] * normpdf(d2) * (d2 / vtilde - 1) * exp(2covar[i, j] + totalVariance[i])
                    end
                    d2 = (mtilde + 3 * tvi[i] - logStrikeScaled) / vtilde
                    eTermS3 += a[i]^3 * normpdf(d2) / divisor * (d2 / vtilde - 1) * exp(3totalVariance[i])
                end
                eTerm += eTermS3
                #now term in sum^2* pi.
                for i = 1:nAsset
                    @inbounds for j = 1:i-1
                        tv = tvi[i] + tvi[j]
                        mtildeij = mtilde + tv + vtilde2
                        d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                        d2 = d1 - vtilde
                        eTerm -= a[i] * a[j] * 6 * (1 - c1) * normpdf(d2) / divisor * (d2 / vtilde - 1) * exp(tv + covar[i, j])
                        mtildeij = mtilde + tv
                        d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                        d2 = d1 - vtilde
                        eTerm -= a[i] * a[j] * 6 * c2 * normpdf(d2) / divisor * (d2 / vtilde - 1) * exp(covar[i, j])
                    end
                    tv = 2tvi[i]
                    mtildeij = mtilde + tv + vtilde2
                    d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                    d2 = d1 - vtilde
                    eTerm -= a[i]^2 * 3 * (1 - c1) * normpdf(d2) / divisor * (d2 / vtilde - 1) * exp(tv + totalVariance[i])
                    mtildeij = mtilde + tv
                    d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                    d2 = d1 - vtilde
                    eTerm -= a[i]^2 * 3 * c2 * normpdf(d2) / divisor * (d2 / vtilde - 1) * exp(totalVariance[i])
                end
                #now term in sum*pi^2
                for i = 1:nAsset
                    tv = tvi[i]
                    mtildeij = mtilde + tv + 2vtilde2
                    d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                    d2 = d1 - vtilde
                    eTerm += a[i] * 3 * (1 - c1)^2 * normpdf(d2) / divisor * (d2 / vtilde - 1) * exp(2tv + vtilde2)
                    mtildeij = mtilde + tv
                    d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                    d2 = d1 - vtilde
                    eTerm += a[i] * 3 * c2^2 * normpdf(d2) / divisor * (d2 / vtilde - 1)
                    mtildeij = mtilde + tv + vtilde2
                    d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
                    d2 = d1 - vtilde
                    eTerm += a[i] * 3 * (1 - c1) * c2 * 2 * normpdf(d2) / divisor * (d2 / vtilde - 1) * exp(tv)

                end
                price -= A * eTerm / 6

            end
        end
    end
    return price * discountFactor
end