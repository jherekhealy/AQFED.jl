import AQFED.Black: blackScholesFormula
import AQFED.TermStructure: CapitalizedDividend, futureValue
import AQFED.Math: OneOverSqrt2Pi, normpdf, normcdf
export EtoreGobetPLNPricer, LeFlochLehmanPLNPricer, priceEuropean

struct EtoreGobetPLNPricer
    order::Int
end


function priceEuropean(
    p::EtoreGobetPLNPricer,
    isCall::Bool,
    strike::Number,
    rawForward::Number, #The raw forward to τ (without cash dividends)
    variance::Number,
    τ::Number,
    discountDf::Number, #discount factor to payment date
    dividends::Vector{CapitalizedDividend},
)
    if length(dividends) == 0
        return blackScholesFormula(isCall, strike, rawForward, variance, one(Float64), discountDf)
    end
    if length(dividends) == 1 && dividends[1].dividend.exDate < sqrt(eps(Float64))
        forward = rawForward - futureValue(dividends[1])
        return blackScholesFormula(isCall, strike, forward, variance, one(Float64), discountDf)
    end
    s = one(Float64)
    if !isCall
        s = -s
    end
    sqrtVar = sqrt(variance)
    sumDiv = sum(futureValue(cd) for cd in dividends)
    kd = strike + sumDiv
    forward = rawForward
    if forward - sumDiv <= zero(Float64)
        if isCall
            return zero(Float64)
        else
            return discountDf * strike
        end
    end

    logfk = log(forward / kd)
    d1 = logfk / sqrtVar + sqrtVar / 2
    d2 = d1 - sqrtVar
    nd1 = normcdf(s * d1)
    nd2 = normcdf(s * d2)
    price = s * (forward * nd1 - kd * nd2)
    sum1 = zero(Float64)
    d1Call = -s * nd2
    for cd in dividends
        frac = one(Float64) - cd.dividend.exDate / τ
        d2i = d2 + sqrtVar * frac
        d1Calli = -s * normcdf(s * d2i)
        sum1 += futureValue(cd) * (d1Calli - d1Call)
    end
    sum2 = zero(Float64)
    sum3 = zero(Float64)

    n = length(dividends)
    if p.order > 1
        for i = 1:n
            divPVi = futureValue(dividends[i])
            fraci = 1.0 - dividends[i].dividend.exDate / τ
            for j = i:n
                divPVj = futureValue(dividends[j])
                fracj = 1.0 - dividends[j].dividend.exDate / τ
                fracij = fracj
                d2ij = d2 + sqrtVar * (fraci + fracj)
                d2Callij = exp(-d2ij^2 / 2 + fracij * variance) * OneOverSqrt2Pi / (kd * sqrtVar)
                a = divPVi * divPVj * d2Callij
                sum2 += a
                if i != j
                    sum2 += a
                end
                d3Callij = d2Callij / kd * (d2ij / sqrtVar - one(Float64))
                b = -sumDiv * divPVi * divPVj * d3Callij * 3
                sum3 += b
                if i != j
                    sum3 += b
                end
            end
            d2i = d2 + sqrtVar * fraci
            d2Calli = normpdf(d2i) / (kd * sqrtVar)
            sum2 += -2 * sumDiv * divPVi * d2Calli
            d3Calli = d2Calli / kd * (d2i / sqrtVar - 1)
            sum3 += sumDiv^2 * divPVi * d3Calli * 3
        end
        c = sumDiv^2 * normpdf(d2) / (kd * sqrtVar)
        sum2 += c
        sum3 -= sumDiv * c / kd * (d2 / sqrtVar - 1)
    end
    if p.order > 2
        for i = 1:n
            divPVi = futureValue(dividends[i])
            fraci = one(Float64) - dividends[i].dividend.exDate / τ
            for j = i:n
                divPVj = futureValue(dividends[j])
                fracj = one(Float64) - dividends[j].dividend.exDate / τ

                for l = j:n
                    divPVl = futureValue(dividends[l])
                    fracl = one(Float64) - dividends[l].dividend.exDate / τ
                    fracijl = 3.0 - (dividends[j].dividend.exDate + 2 * dividends[l].dividend.exDate) / τ
                    d2ijl = d2 + sqrtVar * (fraci + fracj + fracl)
                    d3Callijl =
                        exp(-d2ijl^2 / 2 + fracijl * variance) * OneOverSqrt2Pi / (kd^2 * sqrtVar) *
                        (d2ijl / sqrtVar - one(Float64))
                    a = divPVi * divPVj * divPVl * d3Callijl
                    sum3 += a
                    if i != j || j != l
                        if i != j && j != l
                            sum3 += 5 * a
                        else
                            sum3 += 2 * a
                        end
                    end
                end
            end
        end
    else
        sum3 = 0.0
    end


    price = discountDf * (price + sum1 + sum2 / 2 + sum3 / 6)
    return price
end


struct LeFlochLehmanPLNPricer
    order::Int
end



function priceEuropean(
    p::LeFlochLehmanPLNPricer,
    isCall::Bool,
    strike::T,
    rawForward::T, #The raw forward to τ (without cash dividends)
    variance::T,
    τ::T,
    discountDf::T, #discount factor to payment date
    dividends::Vector{CapitalizedDividend},
)::T where {T}
    if length(dividends) == 0
        return blackScholesFormula(isCall, strike, rawForward, variance, one(T), discountDf)
    end
    if length(dividends) == 1 && dividends[1].dividend.exDate < sqrt(eps(T))
        forward = rawForward - futureValue(dividends[1])
        return blackScholesFormula(isCall, strike, forward, variance, one(T), discountDf)
    end
    s = one(T)
    if !isCall
        s = -s
    end
    sqrtVar = sqrt(variance)
    xNear = sum(futureValue(cd) * (τ - cd.dividend.exDate) / τ for cd in dividends)
    xFar = sum(futureValue(cd) * cd.dividend.exDate / τ for cd in dividends)
    kd = strike + xFar
    forward = rawForward - xNear
    if forward - xFar <= zero(T)
        if isCall
            return zero(T)
        else
            return discountDf * strike
        end
    end
    if sqrtVar <= eps(T)
        return s * discountDf * max(forward - kd, 0)
    end


    logfk = log(forward / kd)
    d1 = logfk / sqrtVar + sqrtVar / 2
    d2 = d1 - sqrtVar
    nd1 = normcdf(s * d1)
    nd2 = normcdf(s * d2)
    price = s * (forward * nd1 - kd * nd2)
    sum1 = zero(T)
    d1CallS = s * nd1
    d1CallK = -s * nd2
    sum1 += xNear * d1CallS - xFar * d1CallK

    for cd in dividends
        frac = cd.dividend.exDate / τ
        d2i = d1 - sqrtVar * frac
        d1CalliK = -s * normcdf(s * d2i)
        sum1 += futureValue(cd) * d1CalliK
    end
    sum2 = zero(T)
    sum3 = zero(T)

    n = length(dividends)
    if p.order > 1
        for i = 1:n
            divPVi = futureValue(dividends[i])
            fraci = dividends[i].dividend.exDate / τ
            for j = i:n
                divPVj = futureValue(dividends[j])
                fracj = dividends[j].dividend.exDate / τ
                fracij = fracj
                d2ij = d2 + sqrtVar * (2 * one(T) - fraci - fracj)
                d2CallijK = exp(-d2ij^2 / 2 + (one(T) - fracij) * variance) * OneOverSqrt2Pi / (kd * sqrtVar)
                a = divPVi * divPVj * d2CallijK
                sum2 += a
                c = xFar * divPVi * divPVj * d2CallijK / kd * (d2ij / sqrtVar - one(T)) * 3
                c += xNear * divPVi * divPVj * d2CallijK / forward * (d2ij / sqrtVar) * 3
                sum3 -= c
                if i != j
                    sum2 += a
                    sum3 -= c
                end
            end
            d2iK = d1 - sqrtVar * fraci
            nd2iK = normpdf(d2iK)
            d2CalliS = nd2iK / (forward * sqrtVar)
            d2CalliK = nd2iK / (kd * sqrtVar)
            sum2 -= 2 * xNear * divPVi * d2CalliS + 2 * xFar * divPVi * d2CalliK
            sum3 += 3 * xFar^2 * divPVi * d2CalliK / kd * (d2iK / sqrtVar - one(T))

            d3Calli = nd2iK / (forward^2 * sqrtVar) * (d2iK / sqrtVar + one(T))
            sum3 += 3 * xNear^2 * divPVi * d3Calli
            d3Calli = nd2iK / (kd * forward * sqrtVar) * (d2iK / sqrtVar)
            sum3 += 6 * xNear * xFar * divPVi * d3Calli
        end
        ndd2 = normpdf(d2)
        d2i2K = d2 + sqrtVar * 2
        sum2 += xNear^2 * exp(-d2i2K^2 / 2 + variance) * OneOverSqrt2Pi / (kd * sqrtVar)
        sum2 += xFar^2 * ndd2 / (kd * sqrtVar)
        sum2 += 2 * xNear * xFar * ndd2 / (forward * sqrtVar)
        sum3 -= xFar^3 * ndd2 / (kd * kd * sqrtVar) * (d2 / sqrtVar - one(T))
        sum3 -= xNear^3 * ndd2 * kd / (forward^3 * sqrtVar) * (d2 / sqrtVar + 2)
        sum3 -= 3 * xNear * xFar^2 * ndd2 / (forward * kd * sqrtVar) * d2 / sqrtVar
        sum3 -= 3 * xNear^2 * xFar * ndd2 / (forward^2 * sqrtVar) * (d2 / sqrtVar + 1)
    end
    if p.order > 2
        for i = 1:n
            divPVi = futureValue(dividends[i])
            fraci = one(T) - dividends[i].dividend.exDate / τ
            for j = i:n
                divPVj = futureValue(dividends[j])
                fracj = one(T) - dividends[j].dividend.exDate / τ

                for l = j:n
                    divPVl = futureValue(dividends[l])
                    fracl = one(T) - dividends[l].dividend.exDate / τ
                    fracijl = 3.0 - (dividends[j].dividend.exDate + 2 * dividends[l].dividend.exDate) / τ
                    d2ijl = d2 + sqrtVar * (fraci + fracj + fracl)
                    d3Callijl =
                        exp(-d2ijl^2 / 2 + fracijl * variance) * OneOverSqrt2Pi / (kd^2 * sqrtVar) *
                        (d2ijl / sqrtVar - one(T))
                    a = divPVi * divPVj * divPVl * d3Callijl
                    sum3 += a
                    if i != j || j != l
                        if i != j && j != l
                            sum3 += 5 * a
                        else
                            sum3 += 2 * a
                        end
                    end
                end
            end
        end
    else
        sum3 = 0.0
    end


    price = discountDf * (price + sum1 + sum2 / 2 + sum3 / 6)
    return price
end
