
function priceAsianFloatingStrikeWithSpread(
    p,
    isCall::Bool,
    strikeSpread::Number,
    discountFactor::Number,
    spot::Number,
    forward::Vector{TV}, #forward to each Asian observation t_i, last one is maturity
    totalVariance::Vector{TV}, #vol^2 * t_i
    weight::Vector{TV}, #last weight must be 0.
)::Number where {TV<:Number}
    if length(weight) == length(totalVariance)-1
        weight = vcat(weight,zero(TV))
    else 
        weight[end] = zero(TV)
    end
    forwardK = @. forward / forward[end] * spot
    totalVarianceK = @. totalVariance[end] - totalVariance

    sShift = 1e-4
    s = 1e-4
    weight = vcat(s/spot*one(TV), weight)
    forwardK = vcat(spot/forward[end]*spot, forwardK)
    totalVarianceK = vcat(zero(TV), totalVarianceK)
    price = priceAsianFixedStrike(p, !isCall, spot, discountFactor * forward[end] / spot, spot, forwardK, totalVarianceK, weight)
    weight[1] = (s+sShift)/spot*one(TV)
    priceUp = priceAsianFixedStrike(p, !isCall, spot, discountFactor * forward[end] / spot, spot, forwardK, totalVarianceK, weight)
    weight[1] = (s-sShift)/spot*one(TV)
    priceDown = priceAsianFixedStrike(p, !isCall, spot, discountFactor * forward[end] / spot, spot, forwardK, totalVarianceK, weight)
    
    return price + (strikeSpread-s)* (priceUp-priceDown)/2sShift  + (strikeSpread-s)^2 * (priceUp-2price+priceDown)/(2* sShift^2)
end

function priceAsianFloatingStrikeWithSpreadD(
    p,
    isCall::Bool,
    strikeSpread::Number,
    discountFactor::Number,
    spot::Number,
    forward::Vector{TV}, #forward to each Asian observation t_i, last one is maturity
    totalVariance::Vector{TV}, #vol^2 * t_i
    weight::Vector{TV}, #last weight must be 0.
)::Number where {TV<:Number}
    if length(weight) == length(totalVariance)-1
        weight = vcat(weight,zero(TV))
    else 
        weight[end] = zero(TV)
    end
    forwardK = @. forward / forward[end] * spot
    totalVarianceK = @. totalVariance[end] - totalVariance

    #work directly with fixed strike equiv. Vfloatcall(S,1) = Vfixedput(S,spot/1)
    #                                                  S-sum-K  --   (spot/1-K) - sum    
    sShift = 1e-4
    s = 0e-4
    s = strikeSpread*spot/forward[end]
    price = priceAsianFixedStrike(p, !isCall, spot-s, discountFactor * forward[end] / spot, spot, forwardK, totalVarianceK, weight)
    return price
    # priceUp = priceAsianFixedStrike(p, !isCall, spot-(s+sShift), discountFactor * forward[end] / spot, spot, forwardK, totalVarianceK, weight)
    # priceDown = priceAsianFixedStrike(p, !isCall, spot-(s-sShift), discountFactor * forward[end] / spot, spot, forwardK, totalVarianceK, weight)
    
    # return price + (strikeSpread-s)*(priceUp-priceDown)/2sShift  + (strikeSpread-s)^2 * (priceUp-2price+priceDown)/(2* sShift^2)
end