export priceAsianFloatingStrike

function priceAsianFloatingStrike(
    p,
    isCall::Bool,
    strikePercent::Number,
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
    return strikePercent * priceAsianFixedStrike(p, !isCall, spot / strikePercent, discountFactor * forward[end] / spot, spot, forwardK, totalVarianceK, weight)
end