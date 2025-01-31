using AQFED.Math
using AQFED.Black
export HHLPLNPricer, DividendPolicy, priceEuropean

@enum DividendPolicy LIQUIDATOR SURVIVOR SHIFT

struct HHLPLNPricer
    q::Quadrature
    policy::DividendPolicy
end

function priceEuropean(
    p::HHLPLNPricer,
    isCall::Bool,
    strike::T,
    rawForward::T, #The raw forward to τ (without cash dividends)
    variance::T,
    τ::T,
    discountDf::T, #discount factor to payment date
    dividends::AbstractArray{CapitalizedDividend{T}},
) where {T}
    if length(dividends) == 0
        return blackScholesFormula(isCall, strike, rawForward, variance, one(T), discountDf)
    end
    strikeAdj = strike
    price = zero(T)
    for i = length(dividends):-1:1
        Tdm1 = if (i == 1)
            0.0
        else
            dividends[i-1].dividend.exDate
        end
        varianceTd = variance / τ * dividends[i].dividend.exDate
        varianceTdm1 = variance / τ * Tdm1
        varianceTdTdm1 = varianceTd - varianceTdm1
        varianceTdT = variance - varianceTd
        sqrtVarianceTdTdm1 = sqrt(varianceTd - varianceTdm1)
        forwardD = futureValue(dividends[i])
        D = dividends[i].dividend.amount
        driftDfTdTdm1 = if (i == 1)
            D / forwardD
        else
            futureValue(dividends[i-1]) / forwardD
        end
        driftDfTdT = D/forwardD
        a = 0.0
        aMin = -12.0
        a = (log(forwardD / rawForward) + varianceTdTdm1 / 2) / sqrtVarianceTdTdm1
        b = 12.0
        #rawForward *exp(ax+b) - D /driftDfTDT= strike => x = log(strike+D/driftDfTdT  / rawForward)
        c = (log((strikeAdj + D / driftDfTdT) / rawForward) + varianceTdTdm1 / 2) / sqrtVarianceTdTdm1
        f = function (x)
            if x < a
                return zero(x)
            end
            #spot / driftDfTdTdm1 = rawForward*dfitDfTdT
            s = rawForward * driftDfTdTdm1 * exp(-(varianceTdTdm1) / 2 + sqrtVarianceTdTdm1 * x)
            option = blackScholesFormula(isCall, strikeAdj, s - D, varianceTdT, driftDfTdT, discountDf)
            phi = normpdf(x)
            return option * phi
        end
        price = if a < c && c < b
            priceA = integrate(p.q, f, a, c)
            priceB = integrate(p.q, f, c, b)
            # println("a b c",a ," ",b, " ",c," ",driftDfTdT," ",priceA, " ",priceB)
            priceA + priceB
        else
            integrate(p.q, f, a, b)
        end
        if p.policy == LIQUIDATOR
            zeroValue = blackScholesFormula(isCall, strikeAdj, 0, varianceTdT, driftDfTdT, discountDf)
            if abs(zeroValue) > eps(T)
                fTail = function (x)
                    phi = normpdf(x)
                    return phi
                end
                tailPrice = integrate(p.q, fTail, aMin, a)
                tailPrice *= zeroValue
                price += tailPrice
            end
        elseif p.policy == SURVIVOR
            fTail = function (x)
                s = spot / driftDfTd * exp(-0.5 * varianceTd + sqrtVarianceTd * x)
                option = blackScholesFormula(isCall, strikeAdj, s, varianceTdT, driftDfTdT, discountDf)
                phi = normpdf(x)
                return option * phi
            end
            tailPrice = integrate(p.q, fTail, aMin, a)
            price += tailPrice
        elseif p.policy == SHIFT
            fTail = function (x)
                s = spot / driftDfTd * exp(-0.5 * varianceTd + sqrtVarianceTd * x)
                option = blackScholesFormula(isCall, strikeAdj + D / driftDfTdT, s, varianceTdT, driftDfTdT, discountDf)
                phi = normpdf(x)
                return option * phi
            end
            tailPrice = integrate(p.q, fTail, aMin, a)
            price += tailPrice
        end
        strikeAdj = strikeAdj + D * driftDfTdT
        driftDfTdm1Td = 1.0
        volAdj = impliedVolatilityJaeckel(isCall,
            price,
            rawForward *driftDfTdm1Td*driftDfTdT, #fixme
            strikeAdj,
            τ - Tdm1,
            discountDf)
        variance = volAdj^2 * τ
    end
    return price
end
