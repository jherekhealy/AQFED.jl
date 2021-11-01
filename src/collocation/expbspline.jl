#Stochastic collocation towards an exponential bspline and the normal density
using Roots
import AQFED.Math: normcdf, normpdf, norminv, inv, ClosedTransformation
import AQFED.Black: blackScholesFormula, blackScholesVega, impliedVolatility
using LeastSquaresOptim
using BSplines
using SpecialFunctions


struct ExpBSplineCollocation{T,U}
    g::QuadraticPP{T,U} #a bspline, can I express price directly in terms of bspline basis?
    forward::U
end

function solvePositiveStrike(c::ExpBSplineCollocation{U,T}, expStrike::T)::Tuple{U,Int} where {U, T <: Number}
    #Note: specifying types above makes a drastic difference in performance
    pp = c.g
    n = length(pp.x)
    strike = log(expStrike)
    if strike <= pp.a[1] # is zero
        leftSlope = pp.b[1]
        return (strike - pp.a[1]) / leftSlope + pp.x[1], 0
    elseif strike > pp.a[n]
        rightSlope = pp.b[n] #linear extrapolation a + b x-xn = strike
        return (strike - pp.a[n]) / rightSlope + pp.x[n], n
    end
    i = searchsortedfirst(pp.a, strike)  # x[i-1]<z<=x[i]
    if strike == pp.a[i]
        return pp.x[i], i
    end
    if i > 1
        i -= 1
    end
    x0 = pp.x[i]
    c = pp.c[i]
    b = pp.b[i]
    a = pp.a[i]
    cc = a + x0 * (-b + x0 * c) - strike
    bb = b - 2 * x0 * c
    aa = c
    allck = quadRootsReal(aa, bb, cc)
    for cki in allck
        if cki > pp.x[i] - sqrt(eps(strike)) && cki <= pp.x[i+1] + sqrt(eps(strike))
            return cki, i
        end
    end
    println(allck, " ", pp.x[i], " ", pp.x[i+1], " ", a, " ", b, " ", c, " strike ", strike)
    throw(DomainError(strike, "strike not found"))
end

#QuadRootsReal returns the real roots of a*x^2+ b*x +c =0
function quadRootsReal(a::T, b::T, c::T)::AbstractArray{T} where {T}
    d = b^2 - 4a * c
    if iszero(d)# single root
        return [-b / (2a)]
    elseif d > zero(d)# two real roots
        d = b < zero(d) ? sqrt(d) - b : -sqrt(d) - b
        return [d / (2a), (2c) / d]
    else # two complex roots
        return []
    end
end


function priceEuropean(
    c::ExpBSplineCollocation{T,U},
    isCall::Bool,
    strike::U,
    forward::U,
    discountDf::U,
)::T where {T,U}
    ck, ckIndex = solvePositiveStrike(c, strike)

    useForward = true
    if useForward || isCall
        valuef = hermiteIntegralBounded(c, ck, ckIndex)
        valuek = normcdf(-ck)
        callPrice = valuef - strike * valuek
        putPrice = -(forward - strike) + callPrice
        if isCall
            return callPrice * discountDf
        else
            return putPrice * discountDf
        end
    else #put
        integral = zero(ck)
        pp = c.g
        if ck < pp.x[1]
            integral += firstMomentExtrapolationBounded(p, 1, -300.0, ck)
        else
            integral += firstMomentExtrapolationBounded(p, 1, -300.0, pp.x[1])
        end
        for j = 1:ckIndex-1
            integral += firstMomentBounded(p, j, pp.x[j], pp.x[j+1])
        end
        if ckIndex >= n
            integral += firstMomentExtrapolationBounded(p, n, pp.x[n], ck)
        else
            integral += firstMomentBounded(p, j, pp.x[ckIndex], ck)
        end
        return strike * normcdf(ck) - integral
    end
end


function density(c::ExpBSplineCollocation{T,U}, strike::U)::U where {T, U <: Number}
    ck, ckIndex = solvePositiveStrike(c, strike)
    dp = evaluateDerivative(c.g, ck)
    an = normpdf(ck) / (dp * strike)
    return an
end


function adjustForward(lsc::ExpBSplineCollocation)
    theoForward = hermiteIntegral(lsc)
    lnTheo = log(theoForward)
    if isnan(lnTheo) || isinf(lnTheo)
        println("inf forward ", theoForward, " ", lsc.g)
    else
        # println("for ",theoForward)
        lnf = log(lsc.forward)
        lsc.g.a .+= lnf - lnTheo
    end
end

function hermiteIntegral(p::ExpBSplineCollocation{T,U})::T where {T,U}
    return hermiteIntegralBounded(p, -300.0, 1)
end


function firstMomentExtrapolationBounded(p::ExpBSplineCollocation{T,U}, i::Int, x0t::Number, x1t::Number)::T where {T,U}
    pp = p.g
    # i = length(pp.x) or i = 1
    x0 = pp.x[i]
    a0 = pp.a[i] + x0 * (-pp.b[i])
    a1 = pp.b[i]
    e = exp(a0 + a1^2 / 2)
    if isinf(e) || e == 0 || isnan(e)
        return zero(a0)
    end
    s = (x0 >= zero(x0)) ? one(x0) : -one(x0)
    ndiff = (normcdf(s * (-x0t + a1)) - normcdf(s * (-x1t + a1)))
    if ndiff == zero(ndiff) #necessary for ForwardDiff
        return ndiff
    else
        return s * e * ndiff
    end
end

function firstMomentBounded(p::ExpBSplineCollocation{T,U}, i::Int, a::Number, b::Number)::T where {T,U}
    pp = p.g
    x0 = pp.x[i]
    x1 = pp.x[i+1]
    x0t = max(a, x0)
    x1t = min(b, x1)
    c0 = pp.c[i]
    a0 = pp.a[i] + x0 * (-pp.b[i] + x0 * c0)
    a1 = pp.b[i] - 2 * x0 * c0
    a2 = c0
    onec = 1 - 2 * a2
    lne = a0 + a1^2 / (2 * onec)
    e = exp(lne)
    if iszero(e)
        return e
    elseif isinf(e) || isnan(e)
        return zero(onec)
    elseif abs(onec) < eps(one(onec))
        return e * (x1t - x0t) / sqrt(2 * pi())
    elseif onec > zero(onec)
        sqrtonec = sqrt(onec)
        s = (x0 >= zero(x0)) ? one(x0) : -one(x0)
        ndiff = normcdf(s * (-x0t * sqrtonec + a1 / sqrtonec)) - normcdf(s * (-x1t * sqrtonec + a1 / sqrtonec))
        if ndiff == zero(ndiff) #necessary for ForwardDiff
            return ndiff
        else
            value = s * e * (ndiff) / sqrtonec
            return value
        end
    else
        sqrtonec = sqrt(-onec)
        fip = x1t * sqrtonec + a1 / sqrtonec
        fi = x0t * sqrtonec + a1 / sqrtonec
        s = (fi >= zero(fi)) ? one(x0) : -one(x0)
        useDawson = true
        if !useDawson
            ndiff = (erfi(-s * fi / sqrt(2)) - erfi(-s * fip / sqrt(2)))
            if ndiff == zero(ndiff) #necessary for ForwardDiff
                return ndiff
            else
                value = s * e * ndiff / (2 * sqrtonec)

                if isinf(value) || isnan(value)
                    println("naninfo line190 ", e, " ", fi, " ", fip, " ", sqrtonec)
                    #return zero(x0t)
                end
                return value
            end
        else
            ea = exp(-x0t^2 * onec / 2 + a1 * x0t + a0)
            if iszero(ea)
                return ea
            end
            ebma = exp(-(x1t^2 - x0t^2) * onec / 2 + a1 * (x1t - x0t))
            #  if isinf(ea) || isinf(ebma) || isnan(ea) || isnan(ebma)
            #     return zero(ea)
            # end
            erfb = dawson(-s * fip / sqrt(2)) * 2 / sqrt(pi)
            erfa = dawson(-s * fi / sqrt(2)) * 2 / sqrt(pi)
            ndiff = iszero(erfb) ? erfa : (erfa - ebma * erfb) #issue if ebma large and erfb small, derivative will be 0.
            if iszero(ndiff) #necessary for ForwardDiff
                return ndiff
            end
            fdif = s * ea * ndiff
            if iszero(fdif)
                return fdif
            end
            value = fdif / (2 * sqrtonec)
            if isnan(value) || isinf(value)
                println("naninf line213 ", onec, " ", -x1t^2 * onec / 2 + a1 * x1t + a0, " ", ea, " ", ebma, " dawson ", erfa, " ", erfb)
            end
            return value
        end
    end
end

function hermiteIntegralBounded(p::ExpBSplineCollocation{T,U}, ck::Number, ckIndex::Int)::T where {T,U}
    pp = p.g
    n = length(pp.x)
    i = ckIndex
    if i > length(pp.x)
        i -= 1
    end
    integral = zero(ck)
    if ck < pp.x[1]
        integral += firstMomentExtrapolationBounded(p, 1, ck, pp.x[1])
        i = 0
    elseif ck < pp.x[n]
        #include logck to x[i] with i-1 coeffs
        integral += firstMomentBounded(p, i, ck, pp.x[i+1])
    end
    for j = i+1:n-1
        integral += firstMomentBounded(p, j, pp.x[j], pp.x[j+1])
    end
    integral += firstMomentExtrapolationBounded(p, n, max(pp.x[n], ck), 300.0)
    if isnan(integral) || isinf(integral)
        println(ck, " ", ckIndex, " inf integral ", p.g)
        throw(DomainError("infinite integral"))
    end
    return integral
end


function makeExpBSplineCollocationGuess(
    strikes::Vector{T},
    callPrices::Vector{T},
    weights::Vector{T},
    τ::T,
    forward::T,
    discountDf::T;
    size = 0,
    minSlope = eps(Float64)
) where {T}
    b = fitExpBSplineBachelier(strikes, callPrices, weights, τ, forward, discountDf, size = size, minSlope = minSlope)
end

function makeExpBSplineCollocation(
    strikes::Vector{T},
    callPrices::Vector{T},
    weights::Vector{T},
    τ::T,
    forward::T,
    discountDf::T;
    minSlope = 1e-4,
    penalty = 0.0,
    size = 0,
    rawFit = false,
)::Tuple{ExpBSplineCollocation,Number} where {T} #return collocation and error measure
    strikesf, pricesf, weightsf = filterConvexPrices(
        strikes,
        callPrices ./ discountDf,
        weights,
        forward,
        tol = minSlope + sqrt(eps(one(minSlope))),
    )
    isoc = makeExpBSplineCollocationGuess(strikesf, pricesf, weightsf, τ, forward, 1.0, size = size, minSlope = minSlope)
    isoc, m =
        rawFit ? fit(isoc, strikes, callPrices, weights, forward, discountDf, minSlope = minSlope, penalty = penalty) :
        fit(isoc, strikesf, pricesf, weightsf, forward, discountDf, minSlope = minSlope, penalty = penalty)
    return isoc, m
end

using FastGaussQuadrature
function fitExpBSplineBachelier(strikes, prices, weights, τ, forward, discountDf; size::Int = 0, minSlope::Float64 = eps(Float64))
    m = length(strikes)
    i = findfirst(x -> x > forward, strikes)
    if i == nothing
        i = m
    elseif i == 1
        i = 2
    end
    price = (prices[i] * (forward - strikes[i-1]) + prices[i-1] * (strikes[i] - forward)) / (strikes[i] - strikes[i-1])
    strike = forward
    #bvol = Bachelier.bachelierImpliedVolatility(price, true, strike, τ, forward, discountDf)
    # isoc = IsotonicCollocation(Polynomials.Polynomial([sqrt(bvol * sqrt(τ))]), Polynomials.Polynomial([0.0]), forward)
    vol = impliedVolatility(true, price, forward, strike, τ, discountDf)
    σ = vol * sqrt(τ)
    # need to use black because ys need to be > 0. xs < 0 ok.
    strikesf, pif, xf = makeXFromUndiscountedPrices(strikes, prices, slopeTolerance = minSlope)
    mindx = minimum(xf[2:end] - xf[1:end-1])
    if mindx < zero(xf[1])
        throw(DomainError(mindx, "dx negative, x is decreasing"))
    end

    local x::Vector{Float64}
    local n::Int
    if size == 0
        n = length(xf)
        x = copy(xf)
        #@. x[1:n-1] = xf[1:n-1]+xf[2:n])/2
        # x[1:n-1] = xf[1:n-1]
        x[n] = max(xf[end] * 1.2, 2.0)
        #x[1] = xf[2] - (xf[2] - xf[1]) / 2
    else
        n = max(size, 3) + 1
        # x = collect(range(min(xf[1] * 1.2, -3.0), stop = max(xf[end] * 1.2, 3.0), length = n))
        x = gausshermite(n)[1]
        x = x .* (max(-xf[1] * 1.2, xf[end] * 1.2, 1.5) / x[end])
    end
    a = @. (-σ^2 / 2 + log(forward) + σ * x)
    b = @. (σ * ones(Float64, n))
    c = zeros(Float64, n)
    pp = QuadraticPP(a, b, c, x)
    isoc = ExpBSplineCollocation(pp, forward)
    # adjustForward(isoc)
    return isoc
end

Base.length(p::ExpBSplineCollocation) = Base.length(p.g.x)
Base.broadcastable(p::ExpBSplineCollocation) = Ref(p)

using ForwardDiff

function fit(isoc::ExpBSplineCollocation, strikes, prices, weights, forward, discountDf; minSlope = 1e-6, penalty = 0.0)
    iter = 0
    basis = BSplineBasis(3, isoc.g.x)
    t = BSplines.knots(basis)
    c = zeros(Float64, length(basis)) # = length x + 1
    spl = convert(BSplines.Spline, isoc.g)
    ct = zeros(Float64, length(spl.coeffs) - 1)
    minValue = max(1e-8 * (spl.coeffs[end] - spl.coeffs[1]), minSlope)
    maxValue = 4 * (spl.coeffs[end] - spl.coeffs[1])
    #transform = ExpMinTransformation(minValue)
    transform = ClosedTransformation(minValue, maxValue)
    for i = 1:length(ct)
        ct[i] = inv(transform, min(max(spl.coeffs[i+1] - spl.coeffs[i], minValue), maxValue))
        # println(i, " ", spl.coeffs[i+1] - spl.coeffs[i]," ct ",ct[i])
    end
    function obj!(fvec, ct0::AbstractArray{T}) where {T}

        ct = @. transform(ct0)
        α = zeros(T, length(basis)) # = length x + 1
        α[1] = -sum(ct) #balance out theoretical forward such that it does not explode
        for i = 2:length(α)
            α[i] = ct[i-1] + α[i-1]
        end
        # println("iteration ",iter," ", ForwardDiff.value.(ct0))
        spl = BSplines.Spline(basis, α)
        pp = convert(QuadraticPP, spl)
        lsc = ExpBSplineCollocation(pp, forward)
        # ck = solvePositiveStrike(lsc, strikes[1])[1]
        # verr = firstMomentBounded(lsc, 1, pp.x[1], pp.x[2])
        # println(verr)
        adjustForward(lsc)
        iter += 1
        n = length(strikes)
        @. fvec[1:n] = weights * (priceEuropean(lsc, true, strikes, forward, discountDf) - prices)
        if penalty > 0
            @. fvec[n+1:end] = ((1 / lsc.g.b[2:end] - 1 / lsc.g.b[1:end-1]) * penalty) #more appropriate if transform is unbounded > 0
            # vpen = @. (lsc.g.c[1:end] * penalty) #bounded transform?
            # return vcat(verr, vpen)
        else
            # return vcat(verr)
        end
        fvec
    end
    #     function obj!(fvec, x)
    #     fvec[:] = obj(x)
    #     fvec
    # end
#     cfg = ForwardDiff.JacobianConfig(obj, ct)
# function jac!(fvec, x)
#         fvec[:] = ForwardDiff.jacobian(obj, x, cfg)
#         #println("jac ",ForwardDiff.value.(fvec))
#         if isnan(ForwardDiff.value.(fvec[1]))
#             println("NaN in fvec for x ", x)
#         end
#         fvec
#     end
    # xerr = [0.39995341844904403, 0.5701199797586093, 0.5719276151194864, 0.5749687387719313, 0.5792870114145262, 0.5849459867261593, 0.5920318162549387, 0.6006571022625484, 0.6109662554578916, 0.6231428961116537, 0.637420114896599, 0.6540948511536501, 0.6735483718820394, 0.6962760710263824, 0.7229319988442942, 0.7543975847042282, 0.7918919014142847, 0.8373318965074066, 0.8941117058099329, 0.968128639742918, 1.0708664149833151, 1.2290358686740526, 1.528853236357424, 3.1467136183063698, 0.6495297531849857, 0.2032851901865146, 0.1649241311004164, 0.16488058052073906, 0.1945626540660544, 0.4238521533062266, 0.3181125312715509, 0.20542436305187858, 0.1820776017058426, 0.16554249062832802, 0.15359025795483686, 0.14744503512937399, 0.14343801501723907, 0.13812943948290135, 0.13187658168464514, 0.12548380500682077, 0.11896539357695363, 0.1132340096687979, 0.10887520969143545, 0.10616838922548384, 0.10466025306165908, 0.10356814434805951, 0.10255067548487518, 0.10182025744343283, 0.10145458270349705, 0.10093016161063079, 0.09983396460674955, 0.09814047452830171, 0.09601978776914528, 0.09376538827320721, 0.09175270079823829, 0.0903275648714271, 0.08961473802430946, 0.08937007223655123, 0.08926136226468968, 0.08902365495660793, 0.08855590378343575, 0.08796143487378127, 0.0874257083869379, 0.08698206005804007, 0.08658184258148705, 0.08616569498269007, 0.08573478298521706, 0.08531502458771066, 0.08496648156850824, 0.08477127394096373, 0.08483486149431202, 0.08511029161991776, 0.08549169345727906, 0.08587987096959282, 0.08622352366061321, 0.08651601223406952, 0.08677746695661273, 0.08701303156828352, 0.08722439396408083, 0.08741327232368615, 0.08758138688337708, 0.08773043542525312, 0.08786207262280761, 0.08797789320757941, 0.08807941878729295, 0.0881680880488771, 0.08824525001618294, 0.08831215999677243, 0.08836997783949553, 0.08841976812961458, 0.08846250196607737, 0.08849905999232086, 0.08853023638406257, 0.08855674353241154, 0.08857921719602688, 0.08859822193060779, 0.0886142566364226, 0.08862776009442899, 0.08863911638821935, 0.08864866013245495, 0.06267943175334381]
    # fvec = ForwardDiff.jacobian(obj, xerr, cfg)
    # if isnan(fvec[1])
    #     throw(DomainError(fvec[1]))
    # end
    # throw(DomainError(0))
    outlen = length(strikes)
    if penalty > 0
        outlen += length(isoc.g.b) - 1
    end
    #fit = optimize(obj, ct, LevenbergMarquardt(); autodiff = :forward, show_trace = false, iterations = 1024)
    fit = optimize!(
        LeastSquaresProblem(x = ct, f! = obj!, autodiff = :forward,
        # g! = jac!, #useful to debug issue with ForwardDiff NaNs
        output_length = outlen),
        LevenbergMarquardt();
        iterations = 1024,
    )
    fvec = zeros(Float64, outlen)
    # fit = fsolve(obj!, jac!, ct, outlen; show_trace=false, method=:lm, tol=1e-8) #fit.x
    println(iter, " fit ", fit, obj!(fvec,fit.minimizer)) #obj(fit.x))  #fit.f
    ct0 = fit.minimizer


    ct = @. transform(ct0)
    c[1] = -sum(ct)
    for i = 2:length(c)
        c[i] = ct[i-1] + c[i-1]
    end
    spl = BSplines.Spline(basis, c)
    pp = convert(QuadraticPP, spl)
    measure = fit.ssr #sqrt(sum(x -> x^2, fit.f)/length(fit.f)) #fit.ssr
    lsc = ExpBSplineCollocation(pp, forward)
    adjustForward(lsc)
    return lsc, measure
end
