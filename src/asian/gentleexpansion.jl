using AQFED.Math
export VorstAsianExpansion
struct VorstAsianExpansion
    order::Int
end


function priceAsianFixedStrike(
    p::VorstAsianExpansion,
    isCall::Bool,
    strike::AbstractFloat,
    discountFactor::AbstractFloat,
    spot::AbstractFloat,
    forward::AbstractArray{TV}, #forward to each Asian observation t_i
    totalVariance::AbstractArray{TV}, #vol^2 * t_i
    weight::AbstractArray{TV},
)::Number where {TV<:Number}
    sign = if isCall
        1
    else
        -1
    end
    nAsset = length(totalVariance)
    # price = priceEuropean(p, isCall, strike, discountFactor, spots, forward, totalVariance, weight, correlation)
    A = weight' * forward
    a = @. (weight * forward) / A
    vtilde2Diag = a' * totalVariance
    vtilde2 = zero(TV)
    for i = 1:nAsset
        vi = totalVariance[i]
        for j = 1:nAsset
            vj = totalVariance[j]
            product = a[i] * a[j] * min(vi, vj)
            vtilde2 += product
        end

    end
    mtilde = -vtilde2 / 2
    vtilde = sqrt(vtilde2)
    strikeScaled = strike / A
    #strikeAdjustment = 0.0; mtilde = - vtilde2/2 #lognormal with geom vol
    d1 = (mtilde - log(strikeScaled) + vtilde2) / vtilde
    d2 = d1 - vtilde
    price = sign * A * (normcdf(sign * d1) - strikeScaled * normcdf(sign * d2))
    #above is order 0 price = Geometric approx with adjusted forward to match basket forward (instead of adjusting the strike as in Gentle)
    if p.order > 0
        tvi = zeros(TV, length(totalVariance))
        @inbounds for i = 1:nAsset
            @inbounds for l = 1:nAsset
                tvi[i] += a[l] * min(totalVariance[i], totalVariance[l])
            end
        end
        mtildep = mtilde + vtilde2
        d1 = (mtildep - log(strikeScaled) + vtilde2) / vtilde
        d2 = d1 - vtilde
        eTerm = -sign * (normcdf(sign * d2))
        for i = 1:nAsset
            tv = tvi[i]
            mtildei = mtilde + tv
            d1 = (mtildei - log(strikeScaled) + vtilde2) / vtilde
            d2 = d1 - vtilde
            eTerm -= -a[i] * sign * (normcdf(sign * d2))
        end
        correction = A * eTerm
        price += correction
        if p.order > 1
            mtildep = mtilde + 2vtilde2
            d1 = (mtildep - log(strikeScaled) + vtilde2) / vtilde
            d2 = d1 - vtilde
            eTerm = 0.5 * normpdf(d2) / (strikeScaled * vtilde) * exp(vtilde2) #term in product^2.
            #now term in sum squared
            for i = 1:nAsset
                factori = a[i] / (strikeScaled * vtilde)
                for j = 1:i-1
                    tv = tvi[i] + tvi[j]
                    d2 = (mtilde + tv - log(strikeScaled)) / vtilde
                    eTerm += factori * a[j] * normpdf(d2) * exp(min(totalVariance[i], totalVariance[j]))
                end
                d2 = (mtilde + 2tvi[i] - log(strikeScaled)) / vtilde
                eTerm += factori * a[i] / 2 * normpdf(d2) * exp(totalVariance[i])

            end
            #finally term in product*sum.
            for i = 1:nAsset
                tv = tvi[i]
                mtildeij = mtilde + tv + vtilde2
                d1 = (mtildeij - log(strikeScaled) + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm -= a[i] * normpdf(d2) / (strikeScaled * vtilde) * exp(tv)
            end
            price += A * eTerm
            if p.order > 2
                #d3CallBS/dK = phi(d2)*(d2/vtilde - 1)/(K^2*vtilde)
                mtildep = mtilde + 3vtilde2
                d1 = (mtildep - log(strikeScaled) + vtilde2) / vtilde
                d2 = d1 - vtilde
                eTerm = -normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(3vtilde2) #term in product^3.
                eTermS3 = zero(TV)

                for i = 1:nAsset
                    @inbounds for j = 1:i-1
                        factorij = 6 * a[i] * a[j] / (strikeScaled^2 * vtilde)
                        @inbounds for l = 1:j-1
                            tv = tvi[i] + tvi[j] + tvi[l]
                            d2v = (mtilde + tv - log(strikeScaled)) / vtilde2
                            eTermS3 += a[l] * factorij * normpdf(d2v * vtilde) * (d2v - 1) * exp(min(totalVariance[i], totalVariance[j]) + min(totalVariance[i], totalVariance[l]) + min(totalVariance[l], totalVariance[j]))
                        end
                        d2 = (mtilde + 2 * tvi[j] + tvi[i] - log(strikeScaled)) / vtilde
                        eTermS3 += factorij / 2 * a[j] * normpdf(d2) * (d2 / vtilde - 1) * exp(2min(totalVariance[i], totalVariance[j]) + totalVariance[j])
                        d2 = (mtilde + 2 * tvi[i] + tvi[j] - log(strikeScaled)) / vtilde
                        eTermS3 += factorij / 2 * a[i] * normpdf(d2) * (d2 / vtilde - 1) * exp(2min(totalVariance[i], totalVariance[j]) + totalVariance[i])
                    end
                    d2 = (mtilde + 3 * tvi[i] - log(strikeScaled)) / vtilde
                    eTermS3 += a[i]^3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(3totalVariance[i])
                end
                eTerm += eTermS3
                #now term in sum^2* pi.
                for i = 1:nAsset
                    @inbounds for j = 1:i-1
                        tv = tvi[i] + tvi[j]
                        mtildeij = mtilde + tv + vtilde2
                        d1 = (mtildeij - log(strikeScaled) + vtilde2) / vtilde
                        d2 = d1 - vtilde
                        eTerm -= a[i] * a[j] * 6 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(tv + min(totalVariance[i], totalVariance[j]))
                    end
                    tv = 2tvi[i]
                    mtildeij = mtilde + tv + vtilde2
                    d1 = (mtildeij - log(strikeScaled) + vtilde2) / vtilde
                    d2 = d1 - vtilde
                    eTerm -= a[i]^2 * 3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(tv + totalVariance[i])
                end
                #now term in sum*pi^2
                for i = 1:nAsset
                    tv = tvi[i]
                    mtildeij = mtilde + tv + 2vtilde2
                    d1 = (mtildeij - log(strikeScaled) + vtilde2) / vtilde
                    d2 = d1 - vtilde
                    eTerm += a[i] * 3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(2tv + vtilde2)
                end
                price -= A * eTerm / 6

            end
        end
    end
    return price * discountFactor
end
#vorstlognormal(1) == gentle(1)