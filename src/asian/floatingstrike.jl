export priceAsianFloatingStrike

function priceAsianFloatingStrike(
    p,
    isCall::Bool,
    strikePercent::Number,
    discountFactor::Number,
    spot::Number,
    forward::Vector{TV}, #forward to each Asian observation t_i
    totalVariance::Vector{TV}, #vol^2 * t_i
    weight::Vector{TV},
)::Number where {TV<:Number}
    forwardK = @. forward / forward[end] * spot
    totalVarianceK = @. totalVariance[end] - totalVariance
    return strikePercent * priceAsianFixedStrike(p, !isCall, spot / strikePercent, discountFactor * forward[end] / spot, spot, forwardK, totalVarianceK, weight)
end