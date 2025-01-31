using AQFED.Math
using PPInterpolation
import AQFED.TermStructure: VarianceSection, varianceByLogmoneyness


struct ContinuousVarianceSwapReplication
    q::Quadrature
end
#Continuous replication
function priceVarianceSwap(self::ContinuousVarianceSwapReplication, spot::Number, tte::Number, varianceSection::VarianceSection, driftDf::Number, discountDf::Number; ndev=6, u=100/sqrt(tte))
    integrand1 = function (k)
        vk = varianceByLogmoneyness(varianceSection, k) * tte
        sqrtvk = sqrt(vk)
        d1 = sqrtvk / 2 - k / sqrtvk
        d2 = d1 - sqrtvk
        normcdf(-d2) - exp(-k) * normcdf(-d1)
    end
    integrand2 = function (k)
        vk = varianceByLogmoneyness(varianceSection, k) * tte
        sqrtvk = sqrt(vk)
        d1 = sqrtvk / 2 - k / sqrtvk
        d2 = d1 - sqrtvk
        exp(-k) * normcdf(d1) - normcdf(d2)
    end
    #f = spot/driftDf
    atmSqrtv = sqrt(varianceByLogmoneyness(varianceSection, zero(spot)) * tte)
    kmin = -ndev * atmSqrtv
    kmax = ndev * atmSqrtv
    2 * u^2 * discountDf * (integrate(self.q, integrand1, kmin, zero(spot)) + integrate(self.q, integrand2, zero(spot), kmax))
end

struct FukasawaVarianceSwapReplication
    isLinear::Bool
end


function filterNonDecreasingData(key::AbstractArray{T}, value::AbstractArray{T}, startIndex::Int) where {T}
	km = zeros(T,1)
	vm =zeros(T,1)
	km[1] = key[startIndex]
	vm[1] = value[startIndex]
	for i = (startIndex+1):length(key)
		if key[i] < km[end] 
			append!(km, key[i])
			append!(vm, value[i])
        end
	end
	for i = startIndex - 1:-1:1
		if key[i] > km[1] 
			km = prepend!(km,key[i])
			vm = prepend!(vm, value[i])
        end
	end
    return km, vm
end
d2(y,v) = - y/sqrt(v) - sqrt(v)/2

#Fukasawa
function priceVarianceSwap(self::FukasawaVarianceSwapReplication,
    tte::Number,
	logmoneyness::AbstractArray{T},
	variance::AbstractArray{T},
    discountDf::Number;u=100) where {T}
    bracket = searchsorted(logmoneyness,zero(T))
    fIndex = if abs(logmoneyness[bracket.start]) < abs(logmoneyness[bracket.stop])
        bracket.start
    else
        bracket.stop
    end
    zs = [d2(y,v * tte) for (y, v) in zip(logmoneyness, variance)]
    zs, vs = filterNonDecreasingData(zs,variance,fIndex)
    reverse!(zs)
    reverse!(vs)
    spline = makeCubicPP(zs, vs, PPInterpolation.SECOND_DERIVATIVE,0.0, PPInterpolation.SECOND_DERIVATIVE,0.0, C2Hyman89())
    left = if self.isLinear
        PPInterpolation.LinearAutoExtrapolation()
    else
        PPInterpolation.ConstantAutoExtrapolation()
    end 
    right = if self.isLinear
        PPInterpolation.LinearAutoExtrapolation()
    else
        PPInterpolation.ConstantAutoExtrapolation()
    end 
    price = PPInterpolation.evaluateHermiteIntegral(spline, leftExtrapolation=left,rightExtrapolation=right)
    return discountDf * u^2*price
end