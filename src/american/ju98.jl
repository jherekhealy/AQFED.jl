using AQFED.TermStructure
import AQFED.Math: normcdf, normpdf, ClosedTransformation, IdentityTransformation
import AQFED.Math:inv as invtransform
import AQFED.Black: blackScholesFormula
#import Roots: find_zero, Newton, A42
using GaussNewton
export Ju98Representation
using NLsolve

struct Ju98Representation{TM}
    isCall::Bool
    model::TM
    tauMax::Float64
    tauHat::Float64
    m::Int
    capX::Float64
    t::Vector{Float64}
    B::Vector{Float64}
    b::Vector{Float64}
    varianceA::Vector{Float64}
    discountDfA::Vector{Float64}
    forwardA::Vector{Float64}
end

function Ju98Representation(
    model::TSBlackModel,
    tauMax::Float64;
    isCall::Bool=false,
    atol::Float64=1e-8,
    m::Int,
    useSqrt=true,
    specialTimes=Float64[],
    isIterative::Bool=false,
    useNLsolve::Bool=false,
    isLower::Bool=false)
    K = 1.0
    rShort = -(logDiscountFactor(model, tauMax + 1e-7) - logDiscountFactor(model, tauMax)) / 1e-7
    qShort = rShort - (logForward(model, 0.0, tauMax + 1e-7) - logForward(model, 0.0, tauMax)) / 1e-7
    capX = isLower ? K * rShort / qShort : K

    if qShort > rShort
        capX = K * rShort / qShort
    end

    tauHat = tauMax
    logCapX = log(capX)
    vol = sqrt(varianceByLogmoneyness(model, 0.0, tauHat))
    r = -logDiscountFactor(model, tauHat) / tauHat
    q = r - logForward(model, 0.0, tauHat) / tauHat
    modelB = ConstantBlackModel(vol, r, q)

    t = zeros(Float64, m + 1)
    if (useSqrt)
        du = sqrt(tauMax) / m
        for i = m+1:-1:1
            ui = du * (m - (i - 1))
            t[i] = tauHat - ui^2
        end
    else
        du = tauMax / m
        for i = m+1:-1:1
            t[i] = du * (i - 1)
        end
    end
    iSpecial = 1
    tNew = Float64[t[1]]
    for i = 1:m
        while iSpecial <= length(specialTimes) && specialTimes[iSpecial] < t[i+1]
            append!(tNew, specialTimes[iSpecial])
            iSpecial += 1
        end
        if iSpecial <= length(specialTimes) && abs(specialTimes[iSpecial] - t[i+1]) > 1e-6 && abs(specialTimes[iSpecial] - tNew[end]) > 1e-6
            append!(tNew, t[i+1])
        end
        if iSpecial > length(specialTimes)
            append!(tNew, t[i+1])
        end
    end
    t = tNew

    m = length(t) - 1
    B = zeros(Float64, m + 1)
    b = zeros(Float64, m + 1)

    varianceA = zeros(Float64, m + 1)
    forwardA = zeros(Float64, m + 1)
    discountDfA = zeros(Float64, m + 1)

    for i = m+1:-1:1
        varianceA[i] = varianceByLogmoneyness(model, 0.0, t[i]) * t[i]
        forwardA[i] = forward(model, 1.0, t[i])
        discountDfA[i] = discountFactor(model, t[i])
    end
    local fprev = capX
    B[m+1] = log(capX) #normalized to capX?
    b[m+1] = 0.0
    transform = ClosedTransformation(-20.0, 20.0) #limit the slope, otherwise numerical explosion due to machine accuracy limits. Perhaps this hsould be dependent on tj-tj+1.
    if isIterative
        fi = americanBoundaryPutQDP(isLower, modelB, fprev, K, tauMax - t[m], atol)
        x0 = zeros(Float64, 2)
        rr = zeros(Float64, 2)
        sign = -1.0
        x0[1] = log(fi)
        #x0[2] = log(fprev / fi) / (t[m+1] - t[m])
        iSpecial = length(specialTimes)
        for i = m:-1:1
            variance = varianceA[end]
            forward = forwardA[end]
            discountDf = discountDfA[end]
            forwardi = forward / forwardA[i]
            variancei = variance - varianceA[i]
            discountDfi = discountDf / discountDfA[i]
            sqrtvi = sqrt(variancei)

            function obji!(rr, x)
                Bstar = exp(x[1])
                lnxy = log(Bstar * forwardi / K)
                d1i = (lnxy + variancei / 2) / sqrtvi
                d2i = d1i - sqrtvi
                valueS = 0.0
                valueK = 0.0
                i1 = 0.0
                i2 = 0.0
                cdf1 = normcdf(-sign * d1i)
                piS = sign * cdf1 * forwardi * discountDfi
                piK = 0.0
                i1b = 0.0
                i2b = 0.0

                #at time ti, we integrate 0 to ti. Bstar varies => need to recompute.
                for j = m:-1:i
                    Bj = (j == i) ? Bstar : exp(B[j] + b[j] * (t[i] - t[j]))
                    bj = (j == i) ? transform(x[2]) : b[j]
                    r = -log(discountDfA[j+1] / discountDfA[j]) / (t[j+1] - t[j])
                    q = r - log(forwardA[j+1] / forwardA[j]) / (t[j+1] - t[j])
                    sigma = sqrt((varianceA[j+1] - varianceA[j]) / (t[j+1] - t[j]))
                    i1part, i1spart = iFuncAndS(t[j] - t[i], t[j+1] - t[i], Bstar, Bj, bj, -1, r, r, q, sigma)
                    i1 += i1spart
                    i1b += i1part
                    i2part, i2spart = iFuncAndS(t[j] - t[i], t[j+1] - t[i], Bstar, Bj, bj, 1, q, r, q, sigma)
                    i2b += i2part
                    i2 += i2part
                    i2 += i2spart
                end
                if isinf(i1) || isnan(i1) || isinf(i2) || isnan(i2) || isinf(i1b) || isnan(i1b) || isinf(i2b) || isnan(i2b)
                    i1 = 0.0
                    i2 = 0.0
                    i1b = 0.0
                    i2b = 0.0
                end
                valueS = -piS + i2 + normpdf(d1i) / sqrtvi * discountDfi * forwardi
                valueK = -piK + i1 + normpdf(d2i) / sqrtvi * discountDfi
                # println(B[i]," ",valueS, " ",K, " ",valueK, " ",r)
                rr[2] = Bstar * valueS - K * valueK
                piS = sign * cdf1 * forwardi * discountDfi
                piK = sign * normcdf(-sign * d2i) * discountDfi
                valueS = -piS + i2b #we changed the sign in cdf. -Cdf(-x) = -(1-cdf(x)) +1
                valueK = -piK + i1b
                rr[1] = Bstar * valueS - K * valueK
            end
            x0[2] = invtransform(transform,0.0)
            if useNLsolve
                result = nlsolve(obji!, x0, autodiff=:finite, ftol=atol)
                x0[1] = result.zero[1]
                x0[2] = result.zero[2]
            else
                sqerr = optimize!(obji!, x0, rr, autodiff=:forward, abstol=atol, reltol=0.0, iscale=1)
                #println(sqerr)

            end
            B[i] = x0[1]
            b[i] = transform(x0[2])
        end
    else
        for i = m:-1:1
            fi = americanBoundaryPutQDP(isLower, modelB, fprev, K, tauMax - t[i], atol)
            B[i] = log(fi)
            b[i] = min(log(fprev / fi) / (t[i+1] - t[i]),2.0)
            fprev = fi
        end
        # println(B," ",b)
        #TODO try b[m] = b[m-1]

        x0 = zeros(Float64, 2 * m)
        x0[1:m] = B[1:m]
        @. x0[m+1:2m] = invtransform(transform, b[1:m])
        rr = zeros(Float64, 2 * m)
        sign = -1.0
        function obj!(rr, x)
            #  B[1:m] = x[1:m]
            #  b[1:m] = x[m+1:2m]
            for i = m:-1:1
                Bstar = exp(x[i])
                variance = varianceA[end]
                forward = forwardA[end]
                discountDf = discountDfA[end]
                forwardi = forward / forwardA[i]
                variancei = variance - varianceA[i]
                discountDfi = discountDf / discountDfA[i]
                sqrtvi = sqrt(variancei)
                lnxy = log(Bstar * forwardi / K)
                d1i = (lnxy + variancei / 2) / sqrtvi
                d2i = d1i - sqrtvi
                valueS = 0.0
                valueK = 0.0
                i1 = 0.0
                i2 = 0.0
                cdf1 = normcdf(-sign * d1i)
                piS = sign * cdf1 * forwardi * discountDfi
                piK = 0.0
                i1b = 0.0
                i2b = 0.0

                #at time ti, we integrate 0 to ti. Bstar varies => need to recompute.
                for j = m:-1:i
                    bj = transform(x[j+m])
                    Bj = exp(x[j] + bj * (t[i] - t[j]))
                    r = -log(discountDfA[j+1] / discountDfA[j]) / (t[j+1] - t[j])
                    q = r - log(forwardA[j+1] / forwardA[j]) / (t[j+1] - t[j])
                    sigma = sqrt((varianceA[j+1] - varianceA[j]) / (t[j+1] - t[j]))
                    i1part, i1spart = iFuncAndS(t[j] - t[i], t[j+1] - t[i], Bstar, Bj, bj, -1, r, r, q, sigma)
                    i1 += i1spart
                    i1b += i1part
                    i2part, i2spart = iFuncAndS(t[j] - t[i], t[j+1] - t[i], Bstar, Bj, bj, 1, q, r, q, sigma)
                    i2b += i2part
                    i2 += i2part
                    i2 += i2spart
                end
                valueS = -piS + i2 + normpdf(d1i) / sqrtvi * discountDfi * forwardi
                valueK = -piK + i1 + normpdf(d2i) / sqrtvi * discountDfi
                # println(B[i]," ",valueS, " ",K, " ",valueK, " ",r)
                rr[i+m] = Bstar * valueS - K * valueK
                piS = sign * cdf1 * forwardi * discountDfi
                piK = sign * normcdf(-sign * d2i) * discountDfi
                valueS = -piS + i2b #we changed the sign in cdf. -Cdf(-x) = -(1-cdf(x)) +1
                valueK = -piK + i1b
                rr[i] = Bstar * valueS - K * valueK
            end
        end
        if useNLsolve
            result = nlsolve(obj!, x0, autodiff=:forward, ftol=atol)
            x0[1:m] = result.zero[1:m]
            x0[m+1:2m] = result.zero[m+1:2m]
        else
            sqerr = optimize!(obj!, x0, rr, autodiff=:forward, abstol=atol, iscale=1)
        end
        B[1:m] = x0[1:m]
        @. b[1:m] = transform(x0[m+1:2m])
    end
    return Ju98Representation(isCall, model, tauMax, tauHat, m, capX, t, B, b, varianceA, discountDfA, forwardA)
end


function exerciseBoundary(p::Ju98Representation{TSBlackModel{TS,TC1,TC2}}, K::Float64, t::AbstractArray{Float64}) where {TS,TC1,TC2}
    capX = p.capX * K
    Bzk = zeros(Float64, length(t))
    iB = 2
    for i = eachindex(t)
        while p.t[iB] < t[i]
            iB += 1
        end
        Bzk[i] = K * exp(p.B[iB-1] + p.b[iB-1] * (t[i] - p.t[iB-1]))
    end
    return Bzk
end

function priceAmerican(p::Ju98Representation{TSBlackModel{TS,TC1,TC2}}, K::Float64, S::Float64)::Float64 where {TS,TC1,TC2}
    if p.isCall #use McDonald and Schroder symmetry 
        K, S = S, K
    end
    capX = p.capX * K

    B = p.B
    b = p.b
    t = p.t
    if S < exp(B[1]) * K
        return max(K - S, 0.0)
    end
    discountDfA = p.discountDfA
    forwardA = p.forwardA
    varianceA = p.varianceA
    i = 1
    i1 = 0.0
    i2 = 0.0
    for j = length(p.b)-1:-1:1
        Bj = K * exp(B[j] + b[j] * (t[i] - t[j]))
        r = -log(discountDfA[j+1] / discountDfA[j]) / (t[j+1] - t[j])
        q = r - log(forwardA[j+1] / forwardA[j]) / (t[j+1] - t[j])
        sigma = sqrt((varianceA[j+1] - varianceA[j]) / (t[j+1] - t[j]))

        i1 += iFuncc(t[j] - t[i], t[j+1] - t[i], S, Bj, b[j], -1, r, r, q, sigma)
        i2 += iFuncc(t[j] - t[i], t[j+1] - t[i], S, Bj, b[j], 1, q, r, q, sigma)
    end
    contPrice = K * (-i1) - S * (-i2)
    euro = blackScholesFormula(
        false,
        K,
        forward(p.model, S, p.tauMax),
        varianceByLogmoneyness(p.model, 0.0, p.tauMax) * p.tauMax,
        1.0,
        discountFactor(p.model, p.tauMax)
    )
    # println(euro, " ",contPrice)
    totalPrice = euro + contPrice
    totalPrice = max(K - S, totalPrice)
    return totalPrice
end


function iFuncAndS(t1, t2, x, y, z, phi, nu, r, q, sigma)
    if abs(t1) < eps(Float64)
        return jFuncAndS(t2, x, y, z, phi, nu, r, q, sigma)
    end
    z1 = (r - q - z + phi * sigma^2 / 2) / sigma
    z2 = log(x / y) / sigma
    sqrtt1 = sqrt(t1)
    sqrtt2 = sqrt(t2)
    z1s = z1 * sqrtt1 + z2 / sqrtt1
    z12s = z1 * sqrtt2 + z2 / sqrtt2
    Nz1 = normcdf(z1s)
    nz1 = normpdf(z1s)
    Nz12 = normcdf(z12s)
    nz12 = normpdf(z12s)
    enu1 = exp(-nu * t1)
    enu2 = exp(-nu * t2)
    dvalue = (enu1 * nz1 / sqrtt1 - enu2 * nz12 / sqrtt2) / sigma
    disc = z1^2 + 2 * nu
    if abs(disc) < eps(Float64)
        disc = 0.0
    end

    value = enu1 * Nz1 - enu2 * Nz12
    z3 = if disc < 0
        sqrt(Complex(disc))
    else
        sqrt(disc)
    end
    z3s = -(z3 * sqrtt1 + z2 / sqrtt1)
    z32s = -(z3 * sqrtt2 + z2 / sqrtt2)
    z3ms = -(z3 * sqrtt1 - z2 / sqrtt1)
    z32ms = -(z3 * sqrtt2 - z2 / sqrtt2)

    signz3 = 1.0
    if real(z3s) > 0.0
        z3s = -z3s
        z32s = -z32s
        signz3 = -1.0
    end
    signz3m = 1.0
    if real(z3ms) > 0.0
        z3ms = -z3ms
        z32ms = -z32ms
        signz3m = -1.0
    end
    Nz3 = normcdf(z3s)
    nz3 = normpdf(z3s)
    Nz32 = normcdf(z32s)
    nz32 = normpdf(z32s)
    Nz3m = normcdf(z3ms)
    nz3m = normpdf(z3ms)
    Nz32m = normcdf(z32ms)
    nz32m = normpdf(z32ms)
    ez = exp(z2 * (z3 - z1))
    ezm = exp(-z2 * (z3 + z1))
    nz32 /= sqrtt2
    nz3 /= sqrtt1
    nz32m /= sqrtt2
    nz3m /= sqrtt1
    if abs(Nz32 - Nz3) > 1e-300
        value += -0.5 * (z1 / z3 + 1) * ez * (Nz32 - Nz3) * signz3
        dvalue += (z3 - z1) * (1 + z1 / z3) / (sigma * 2) * ez * (Nz3 - Nz32) * signz3 #Ju paper has typos there - is actually(z3-z1)*I
        dvalue += ez * (1 + z1 / z3) / (sigma * 2) * (nz32 - nz3)
    end
    if abs(Nz32m - Nz3m) > 1e-300
        value -= 0.5 * (z1 / z3 - 1) * ezm * (Nz32m - Nz3m) * signz3m
        dvalue -= (z3 + z1) * (z1 / z3 - 1) / (sigma * 2) * ezm * (Nz3m - Nz32m) * signz3m #iS actually -(z3+z1)*I
        dvalue -= ezm * (z1 / z3 - 1) / (sigma * 2) * (nz32m - nz3m)
    end

    return (real(value), real(dvalue))
end

function jFuncAndS(t, x, y, z, phi, nu, r, q, sigma)
    z1 = (r - q - z + phi * sigma^2 / 2) / sigma
    z2 = log(x / y) / sigma
    sqrtt = sqrt(t)
    z12s = z1 * sqrtt + z2 / sqrtt
    Nz12 = normcdf(z12s)
    nz12 = normpdf(z12s)
    enu2 = exp(-nu * t)
    nz3 = 0.0
    Nz3m = 1.0
    Nz3 = 0.0
    nz3m = 0.0
    # #the following does not happen, except for neg rates
    # if z2 < 0 {
    # 	Nz3 = 1.0
    # 	Nz3m = 0.0
    # end
    Nz1 = 1.0
    # #the following does not happen, except for neg rates
    # if z2 < 0 
    # 	Nz1 = 0.0
    # end
    value = Nz1 - enu2 * Nz12

    dvalue = -enu2 * nz12 / (sqrtt * sigma)
    disc = z1 * z1 + 2 * nu
    if abs(disc) < eps(Float64)
        disc = 0.0
    end

    z3 = if disc < 0
        sqrt(Complex(disc))
    else
        sqrt(disc)
    end

    ez = exp(z2 * (z3 - z1))
    ezm = exp(-z2 * (z3 + z1))
    z32s = -(z3 * sqrtt + z2 / sqrtt)
    z32m = -z3 * sqrtt + z2 / sqrtt

    Nz32 = normcdf(z32s)
    nz32 = normpdf(z32s)
    Nz32m = normcdf(z32m)
    nz32m = normpdf(z32m)
    nz32 /= sqrtt
    nz32m /= sqrtt

    if abs(Nz32 - Nz3) > 1e-300
        value -= 0.5 * (z1 / z3 + 1) * ez * (Nz32 - Nz3)
        dvalue += (z3 - z1) * (1 + z1 / z3) / (sigma * 2) * ez * (Nz3 - Nz32) #Ju paper has typos there - is actually(z3-z1)*I
        dvalue += ez * (1 + z1 / z3) / (sigma * 2) * (nz32 - nz3)
    end
    if abs(Nz32m - Nz3m) > 1e-300
        value -= 0.5 * (z1 / z3 - 1) * ezm * (Nz32m - Nz3m)
        dvalue -= (z3 + z1) * (z1 / z3 - 1) / (sigma * 2) * ezm * (Nz3m - Nz32m) #iS actually -(z3+z1)*I
        dvalue -= ezm * (z1 / z3 - 1) / (sigma * 2) * (nz32m - nz3m)
    end
    return (real(value), real(dvalue))
end



function iFuncc(t1, t2, x, y, z, phi, nu, r, q, sigma)
    if abs(t1) < eps(Float64)
        return jFuncc(t2, x, y, z, phi, nu, r, q, sigma)
    end
    z1 = (r - q - z + phi * sigma^2 / 2) / sigma
    z2 = log(x / y) / sigma
    sqrtt1 = sqrt(t1)
    sqrtt2 = sqrt(t2)
    z1s = z1 * sqrtt1 + z2 / sqrtt1
    z12s = z1 * sqrtt2 + z2 / sqrtt2
    Nz1 = normcdf(-z1s)
    Nz12 = normcdf(-z12s)
    enu1 = exp(-nu * t1)
    enu2 = exp(-nu * t2)
    disc = z1^2 + 2 * nu
    if abs(disc) < eps(Float64)
        disc = 0.0
    end

    value = -enu1 * Nz1 + enu2 * Nz12
    z3 = if disc < 0
        sqrt(Complex(disc))
    else
        sqrt(disc)
    end

    z3s = -(z3 * sqrtt1 + z2 / sqrtt1)
    z32s = -(z3 * sqrtt2 + z2 / sqrtt2)
    z3ms = -(z3 * sqrtt1 - z2 / sqrtt1)
    z32ms = -(z3 * sqrtt2 - z2 / sqrtt2)

    signz3 = 1.0
    if real(z3s) > 0.0
        z3s = -z3s
        z32s = -z32s
        signz3 = -1.0
    end
    signz3m = 1.0
    if real(z3ms) > 0.0
        z3ms = -z3ms
        z32ms = -z32ms
        signz3m = -1.0
    end
    Nz3 = normcdf(z3s)
    Nz32 = normcdf(z32s)
    Nz3m = normcdf(z3ms)
    Nz32m = normcdf(z32ms)
    ez = exp(z2 * (z3 - z1))
    ezm = exp(-z2 * (z3 + z1))
    if abs(Nz32 - Nz3) > 1e-300
        value += -0.5 * (z1 / z3 + 1) * ez * (Nz32 - Nz3) * signz3
    end
    if abs(Nz32m - Nz3m) > 1e-300
        value -= 0.5 * (z1 / z3 - 1) * ezm * (Nz32m - Nz3m) * signz3m
    end

    return value
end

function jFuncc(t, x, y, z, phi, nu, r, q, sigma)
    z1 = (r - q - z + phi * sigma^2 / 2) / sigma
    z2 = log(x / y) / sigma
    sqrtt = sqrt(t)
    z12s = z1 * sqrtt + z2 / sqrtt
    Nz12 = normcdf(-z12s)
    enu2 = exp(-nu * t)
    Nz3m = 1.0
    Nz3 = 0.0
    # #the following does not happen, except for neg rates
    # if z2 < 0 {
    # 	Nz3 = 1.0
    # 	Nz3m = 0.0
    # end

    # #the following does not happen, except for neg rates
    # if z2 < 0 
    # 	Nz1 = 0.0
    # end
    value = enu2 * Nz12

    disc = z1 * z1 + 2 * nu
    if abs(disc) < eps(Float64)
        disc = 0.0
    end

    z3 = if disc < 0
        sqrt(Complex(disc))
    else
        sqrt(disc)
    end

    ez = exp(z2 * (z3 - z1))
    ezm = exp(-z2 * (z3 + z1))
    z32s = -(z3 * sqrtt + z2 / sqrtt)
    z32m = -z3 * sqrtt + z2 / sqrtt

    Nz32 = normcdf(z32s)
    Nz32m = normcdf(z32m)

    if abs(Nz32 - Nz3) > 1e-300
        value -= 0.5 * (z1 / z3 + 1) * ez * (Nz32 - Nz3)
    end
    if abs(Nz32m - Nz3m) > 1e-300
        value -= 0.5 * (z1 / z3 - 1) * ezm * (Nz32m - Nz3m)
    end
    return value
end
