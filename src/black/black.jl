using AQFED.Math
const SqrtEpsilon = sqrt(eps())
# totalVariance is vol^2*τ
export blackScholesFormula, blackScholesVega

function blackScholesFormula(isCall::Bool,   strike::Real, 
    spot::Real, 
    totalVariance::Real, 
    driftDf::Real, 
    discountDf::Real)
    sign = 1
    if !isCall
        sign = -1
    end
    forward = spot / driftDf
    if totalVariance < eps()
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
        sqrtVar = sqrt(totalVariance)
        d1 = log(forward / strike)/sqrtVar + sqrtVar/2
        d2 = d1 - sqrtVar
        nd1 = Math.normcdf(sign * d1)
        nd2 = Math.normcdf(sign * d2)
        price = sign * discountDf * (forward * nd1 - strike * nd2)
        return price
    end
end

function blackScholesVega(strike::Number,
	spot::Number,
	totalVariance::Number,
	driftDf::Number,
	discountDf::Number,
	tte::Number)::Number
	forward = spot / driftDf
	sqrtVar = sqrt(totalVariance)
	d1 = log(forward/strike)/sqrtVar + sqrtVar/2
	nd1 = Math.normpdf(d1)
	vega = discountDf * forward * nd1 * sqrt(tte)
	return vega
end
