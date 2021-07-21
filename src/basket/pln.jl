
import AQFED.Black: blackScholesFormula
import AQFED.TermStructure: CapitalizedDividend, futureValue
export priceEuropean

function priceEuropean(
    p::DeelstraBasketPricer,
    isCall::Bool,
    strike::Number,
    rawForward::Number, #The raw forward to τ (without cash dividends)
    variance::Number,
    τ::Number,
    discountDf::Number, #discount factor to payment date
    dividends::Vector{CapitalizedDividend},
)
    if length(dividends) == 0
        return blackScholesFormula(isCall, strike, spot, variance, spot / rawForward, discountDf)
    elseif length(dividends) == 1 && dividends[1].dividend.exDate < sqrt(eps(Float64))
        forward = rawForward - futureValue(dividends[1])
        return blackScholesFormula(isCall, strike, forward, variance, one(Float64), discountDf)
    end

    sumDiv = sum(futureValue(cd) for cd in dividends)
    n = length(dividends) + 1
    weight = ones(Number, n)
    spots = ones(Number, n)
    tvar = zeros(Number, n)
    dfAsset = discountDf
    forwards = zeros(Number, n)
    for i = 1:n-1
        spots[i] = dividends[i].dividend.amount
        forwards[i] = futureValue(dividends[i]) #dividends[i].dividend.amount/dividends[i].capitalizationFactor #-r
        tvar[i] = variance / τ * dividends[i].dividend.exDate
    end
    spots[n] = strike
    forwards[n] = strike
    tvar[n] = variance
    correlation = zeros(Number, (n, n))
    for (i, vi) in enumerate(tvar)
        for j = 1:(i-1)
            vj = tvar[j]
            if vi != 0 && vj != 0
                correlation[i, j] = min(vi, vj) / sqrt(vi * vj)
                # else is zero
            end
        end
        correlation[i, i] = one(Float64)
    end
    for i = 1:n
        for j = i+1:n
            correlation[i, j] = correlation[j, i]
        end
    end
    priceFixed = priceEuropean(p, !isCall, rawForward, dfAsset, spots, forwards, tvar, weight, correlation)

    if isCall
        return priceFixed
    else
        forward = priceEuropean(p, true, 0.0, rawForward, variance, τ, discountDf, dividends)
        return priceFixed - (forward - strike) * discountDf
    end
end
