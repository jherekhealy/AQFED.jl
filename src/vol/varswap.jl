using AQFED.Math
using PPInterpolation
import SpecialFunctions: besselix
import AQFED.Black: blackScholesFormula
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

"""
    CarrLeeVolatilitySwapReplication(q::Quadrature)

Volatility swap by the replication of Carr and Lee (2008), "Robust replication of volatility derivatives", exact when the
volatility is independent of the asset Brownian motion: E[√⟨ln S⟩_T] = E[h(S_T)] with the payoff
h(S) = √(π/2)/F |S-F| + ∫ h''(K) (S-K)⁺ or (K-S)⁺ dK and h''(K) = η √(π/(8K³F)) (I₁(k) - I₀(k)), k = ln(K/F)/2.
"""
struct CarrLeeVolatilitySwapReplication
    q::Quadrature
end
CarrLeeVolatilitySwapReplication() = CarrLeeVolatilitySwapReplication(GaussKronrod(1e-10))

"""
    priceVolatilitySwap(self::CarrLeeVolatilitySwapReplication, forward, tte, varianceSection, discountDf; ndev=8, u=100/√tte)

Discounted fair volatility `u` E[√⟨ln S⟩_T], by default the annualized volatility in percent.
"""
function priceVolatilitySwap(self::CarrLeeVolatilitySwapReplication, forward::Number, tte::Number, varianceSection::VarianceSection, discountDf::Number; ndev=8, u=100 / sqrt(tte))
    function payoffSecond(strike)
        k = log(strike / forward) / 2
        η = strike < forward ? -1 : 1
        #scaled Bessel functions: I(k)e^{-|k|}, avoids the overflow for far strikes
        return η * sqrt(π / (8 * strike^3 * forward)) * (besselix(1, k) - besselix(0, k)) * exp(abs(k))
    end
    optionPrice(isCall, strike) = blackScholesFormula(isCall, strike, forward, varianceByLogmoneyness(varianceSection, log(strike / forward)) * tte, 1.0, 1.0)
    atmVariance = varianceByLogmoneyness(varianceSection, zero(forward)) * tte
    kMin = forward * exp(-ndev * sqrt(atmVariance))
    kMax = forward * exp(ndev * sqrt(atmVariance))
    putPart = integrate(self.q, strike -> optionPrice(false, strike) * payoffSecond(strike), kMin, forward)
    callPart = integrate(self.q, strike -> optionPrice(true, strike) * payoffSecond(strike), forward, kMax)
    straddle = optionPrice(true, forward) + optionPrice(false, forward)
    return u * discountDf * (sqrt(π / 2) / forward * straddle + putPart + callPart)
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
    #println("logm ",logmoneyness, " ", bracket)
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