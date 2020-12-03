using AQFED.Math
const SqrtEpsilon = sqrt(eps())
function blackScholesFormula(isCall::Bool, strike::T, spot::T, variance::T, driftDf::T, discountDf::T) where {T}

    sign = 1
    if !isCall
        sign = -1
    end
    forward = spot / driftDf
    if variance < eps()
        price = discountDf * max(sign * (forward - strike), 0)
        return price
    elseif spot < eps()
        if isCall
            return 0
        else
            return discountDf * strike
        end
    elseif strike < eps()
        if isCall
            return discountDf * forward
        else
            return 0
        end
    else
        sqrtVar = sqrt(variance)
        d1 = log(forward / strike)/sqrtVar + sqrtVar/2
        d2 = d1 - sqrtVar
        nd1 = normcdf(sign * d1)
        nd2 = normcdf(sign * d2)
        price = sign * discountDf * (forward * nd1 - strike * nd2)
        return price
    end
end
