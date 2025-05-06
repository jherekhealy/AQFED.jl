
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

#=
AQFED.Asian.priceAsianFloatingStrike(AQFED.Basket.VorstGeometricExpansion(3), true, 1.0, 1.0, 100.0,  [1.0,100.5,101.0,101.5,102.0,102.0], [0.0,0.4^2*0.1,0.4^2*0.2,0.4^2*0.3, 0.4^2*0.5-1e-8,0.4^2*0.5], [-1.0,0.05,0.05,0.4,0.5,0.0])
4.324535459383135

AQFED.Asian.priceAsianSpread(AQFED.Basket.MonteCarloEngine(true,1024*1024), true, [-1.0,0.5,1.0], 1.0, 100.0, [100.5,101.0,101.5,102.0,102.0], [0.4^2*0.1,0.4^2*0.2,0.4^2*0.3, 0.4^2*0.5,0.4^2*0.5], [0.05,0.05,0.4,0.5,1.0], 4)
4.324528859513569

AQFED.Spread.priceAsianSpread(PearsonPricer(128*128), true, 1.0, 1.0, 100.0, [100.5,101.0,101.5,102.0,102.0,1.0], [0.4^2*0.1,0.4^2*0.2,0.4^2*0.3, 0.4^2*0.5,0.4^2*0.5,1e-16], [-0.05,-0.05,-0.4,-0.5,1.0,1.0])
4.264103256375925

AQFED.Spread.priceAsianSpread(PearsonPricer(128*128), true, 1.0, 1.0, 100.0, [100.5,101.0,101.5,102.0,102.0,1.0], [0.4^2*0.1,0.4^2*0.2,0.4^2*0.3, 0.4^2*0.5,0.4^2*0.5,1e-16], [-0.05,-0.05,-0.4,-0.5,1.0,1.0],momentsPricer=AQFED.Basket.SLN3MMBasketPricer())
4.2729752610455884

#SLN3MM with neg weight much more accurate than pearson.
AQFED.Asian.priceAsianFloatingStrike(AQFED.Basket.SLN3MMBasketPricer(), true, 1.0, 1.0, 100.0,  [1.0,100.5,101.0,101.5,102.0,102.0], [0.0,0.4^2*0.1,0.4^2*0.2,0.4^2*0.3, 0.4^2*0.5-1e-8,0.4^2*0.5], [-1.0,0.05,0.05,0.4,0.5,0.0])
4.32457748187951

#2MM also vastly more accurate than pearson.
AQFED.Asian.priceAsianFloatingStrike(AQFED.Basket.VorstLevyExpansion(0), true, 1.0, 1.0, 100.0,  [1.0,100.5,101.0,101.5,102.0,102.0], [0.0,0.4^2*0.1,0.4^2*0.2,0.4^2*0.3, 0.4^2*0.5-1e-8,0.4^2*0.5], [-1.0,0.05,0.05,0.4,0.5,0.0])
4.326591462662002


#the regular asian (forward spread < 0) is pearson also worse than levy (looks like it is true because change of measure more precise than double logn approx)
AQFED.Asian.priceAsianFloatingStrike(AQFED.Basket.VorstLevyExpansion(0), true, 1.0, 1.0, 100.0,  [1.0,100.5,101.0,101.5,102.0,102.0], [0.0,0.4^2*0.1,0.4^2*0.2,0.4^2*0.3, 0.4^2*0.5-1e-8,0.4^2*0.5], [1.0,0.05,0.05,0.4,0.5,0.0])
3.4247613721805794

AQFED.Asian.priceAsianSpread(AQFED.Basket.VorstGeometricExpansion(3), true, 1.0, 1.0, 100.0, [1.0,100.5,101.0,101.5,102.0,102.0], [1e-16,0.4^2*0.1,0.4^2*0.2,0.4^2*0.3, 0.4^2*0.5,0.4^2*0.5], [-1.0,-0.05,-0.05,-0.4,-0.5,1.0])
3.3854733584265366


=#