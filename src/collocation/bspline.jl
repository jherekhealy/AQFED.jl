#Stochastic collocation towards an exponential bspline and the normal density
using Roots
import AQFED.Math: cubicRootsReal, quadRootsReal, normcdf, normpdf, norminv, inv, ClosedTransformation, FitResult,MQMinTransformation
import AQFED.Black: blackScholesFormula, blackScholesVega, impliedVolatility
using LeastSquaresOptim
using BSplines
#using PPInterpolation
import OffsetArrays: no_offset_view, LinearIndices
import PPInterpolation:evaluatePiece,evaluateDerivativePiece,evaluateSecondDerivativePiece, evaluateDerivative

struct BSplineCollocationCache{U}
    pdfCache::Vector{U}
    cdfCache::Vector{U}
end
 
 struct BSplineCollocation{N,T,U}
    g::FastPP{N,T,U} #we could use bspline repr for price, but this is not convenient for solveStrike
    forward::U
    cache::BSplineCollocationCache{U}
end

function BSplineCollocation(g::FastPP{N,T,U},forward::U) where {N,T,U}
    pdfCache = normpdf.(g.x)
    cdfCache = normcdf.(g.x)
    return BSplineCollocation(g,forward,BSplineCollocationCache(pdfCache,cdfCache))
end

function solveStrike(c::BSplineCollocation{N,U,T}, strike::T)::Tuple{U,Int} where {N, U, T <: Number}
    #Note: specifying types above makes a drastic difference in performance
    #  elapsed = @elapsed begin
        pp = c.g
    n = length(pp.x)
    #println(strike, " a ",pp.a)
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
    res = solveStrikeAt(pp,i,strike) 
# end
# println("solve el ",elapsed)
return res
end

function solveStrikeAt(pp::FastPP{2,T,TX},i::Int,strike::TX) where {T,TX}
    #TODO split impl below for quad,cubic, N with halley
    x0 = pp.x[i]
    c = pp.c[i,1]
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


function solveStrikeAt(pp::FastPP{3,T,TX},i::Int,strike::TX) where {T,TX}
    #TODO split impl below for quad,cubic, N with halley
    x0 = pp.x[i]
    d = pp.c[i,2]
    c = pp.c[i,1]
    b = pp.b[i]
    a = pp.a[i]
    cc = a + x0 * (-b + x0 * (c-d*x0)) - strike
    bb = b +  x0 * (-2c+x0*d*3)
    aa = c - 3d*x0
    allck = if d == zero(T) 
        quadRootsReal(aa, bb, cc)
    else
        aa /= d
		bb /= d
		cc /= d
		cubicRootsReal(aa, bb, cc) #x^3+ax^2+bx+c=0
    end
    for cki in allck
        if cki > pp.x[i] - sqrt(eps(strike)) && cki <= pp.x[i+1] + sqrt(eps(strike))
            return cki, i
        elseif abs(pp.x[i]^3+aa*pp.x[i]^2+bb*pp.x[i]+cc) < sqrt(eps(strike))
            return pp.x[i],i
        elseif abs(pp.x[i+1]^3+aa*pp.x[i+1]^2+bb*pp.x[i+1]+cc) < sqrt(eps(strike))
            return pp.x[i+1],i
        end
    end
    # println(allck, " ", pp.x[i], " ", pp.x[i+1], " ", aa, " ", bb, " ", cc, " strike ", strike)
    throw(DomainError(strike, "strike not found"))
end

function solveStrikeAt(pp::FastPP{N,T,TX},i::Int,strike::TX) where {N,T,TX}
    # function objHalley(x::W) where {W}
    #     u = evaluatePiece(pp,i,x)-strike
    #     du =  evaluateDerivativePiece(pp,i,x)
    #     d2u = evaluateSecondDerivativePiece(pp,i,x)
    #     return (u, u / du, d2u == 0 ? Inf : du / d2u)
    # end
  
    # return find_zero(
    #     objHalley,
    #     (pp.x[i]+pp.x[i+1])/2,
    #     Roots.SuperHalley(), #seems to be (much) more robust around the spike in the density.
    #     atol = 100 * eps(strike),
    #     maxevals = 32,
    #     verbose = false,
    # ),i
    return find_zero(x -> evaluatePiece(pp,i,x)-strike, (pp.x[i],pp.x[i+1]),Roots.A42()),i
end

function priceEuropean(
    c::BSplineCollocation{N,T,U},
    isCall::Bool,
    strike::U,
    forward::U,
    discountDf::U,
)::T where {N,T,U}
    ck, ckIndex = solveStrike(c, strike)

        valuef = hermiteIntegralBounded(c, ck, ckIndex)
        valuek = normcdf(-ck)
        callPrice = valuef - strike * valuek
        putPrice = -(forward - strike) + callPrice
        if isCall
            return callPrice * discountDf
        else
            return putPrice * discountDf
        end
    end


function density(c::BSplineCollocation{N,T,U}, strike::U)::U where {N,T, U <: Number}
    ck, ckIndex = solveStrike(c, strike)
    dp = evaluateDerivative(c.g, ck)
    an = normpdf(ck) / dp
    return an
end


function adjustForward(lsc::BSplineCollocation)
    theoForward = hermiteIntegral(lsc)
    if isnan(theoForward) || isinf(theoForward)
        println("inf forward ", theoForward, " ", lsc.g)
    else
        # println("for ",theoForward)
        lsc.g.a .+= lsc.forward - theoForward
    end
end

function hermiteIntegral(p::BSplineCollocation{N,T,U})::T where {N,T,U}
    return hermiteIntegralBounded(p, -300.0, 1)
end

#from ck to infinity
function hermiteIntegralBounded(p::BSplineCollocation{2,T,U}, ck::Number, ckIndex::Int)::T where {T,U}
   
    pp = p.g
    n = length(pp.x)
    i = ckIndex
    if i > length(pp.x)
        i -= 1
    end
    integral = zero(ck)
    l = ck
    pdfCache = p.cache.pdfCache
    cdfCache = p.cache.cdfCache
    pdfL,cdfL = if ck == pp.x[1]
        pdfCache[1],cdfCache[1]
    else normpdf(l), normcdf(l)    
    end
    if ck <= pp.x[1]
        #should be up to x[1]
        r = pp.x[1]
        pdfR = pdfCache[1]
        cdfR = cdfCache[1]
        slope = pp.b[1]
        moment = (pp.a[1]-pp.x[1]*slope)*(cdfR-cdfL) - slope*(pdfR-pdfL)
        integral += moment
        pdfL = pdfR
        cdfL = cdfR
        i = 0
    elseif ck < pp.x[n]
        #println(i," ",ck," ",pp.x)
        r = pp.x[i+1]
        pdfR = pdfCache[i+1]
        cdfR = cdfCache[i+1]
        moment = (pp.a[i] - pp.x[i]*(pp.b[i]) + pp.c[i,1]*(pp.x[i]^2+1)) * (cdfR-cdfL)
		moment += pdfL*(pp.b[i]-pp.c[i,1]*(2*pp.x[i]-l)) - pdfR*(pp.b[i]-pp.c[i,1]*(2*pp.x[i]-r))		
        integral += moment
        pdfL = pdfR
        cdfL = cdfR
    end
    for j = i+1:n-1
        l = pp.x[j]
        r = pp.x[j+1]
        pdfR =pdfCache[j+1]
        cdfR = cdfCache[j+1]
        moment = (pp.a[j] - pp.x[j]*(pp.b[j]) + pp.c[j,1]*(pp.x[j]^2+1)) * (cdfR-cdfL)
		moment += pdfL*(pp.b[j]-pp.c[j,1]*(2*pp.x[j]-l)) - pdfR*(pp.b[j]-pp.c[j,1]*(2*pp.x[j]-r))		
        integral += moment
        pdfL = pdfR
        cdfL = cdfR
    end
    l = max(pp.x[n], ck)
    slope = evaluateDerivativePiece(pp,n-1,pp.x[n])
    yn = evaluatePiece(pp,n-1,pp.x[n])
    moment = (yn-slope*pp.x[n])*(1-cdfL) + slope*pdfL
    integral += moment
    if isnan(integral) || isinf(integral)
        println(ck, " ", ckIndex, " inf integral ", p.g)
        throw(DomainError("infinite integral"))
    end

    return integral
end

function hermiteIntegralBounded(p::BSplineCollocation{3,T,U}, ck::Union{T,U}, ckIndex::Int)::T where {T,U}
    #  elapsed = @elapsed begin
        
           pp = p.g
    n = length(pp.x)
    i = ckIndex
    if i > length(pp.x)
        i -= 1
    end
    integral = zero(ck)
    l = ck
    pdfCache = p.cache.pdfCache
    cdfCache = p.cache.cdfCache
 
    pdfL,cdfL = if ck == pp.x[1]
        pdfCache[1],cdfCache[1]
    else normpdf(l), normcdf(l)    
    end
    if ck <= pp.x[1]
        #should be up to x[1]
        r = pp.x[1]
        pdfR = pdfCache[1]
        cdfR = cdfCache[1]
        slope = pp.b[1]
        moment = (pp.a[1]-pp.x[1]*slope)*(cdfR-cdfL) - slope*(pdfR-pdfL)
        integral += moment
        pdfL = pdfR
        cdfL = cdfR
        i = 0
    elseif ck < pp.x[n]
        #println(i," ",ck," ",pp.x)
        r = pp.x[i+1]
        pdfR = pdfCache[i+1]
        cdfR = cdfCache[i+1]
        moment = (pp.a[i] - pp.x[i]*(pp.b[i]+pp.c[i,2]*(pp.x[i]^2+3)) + pp.c[i,1]*(pp.x[i]^2+1)) * (cdfR-cdfL)
		moment += pdfL*(pp.b[i]-pp.c[i,1]*(2*pp.x[i]-l)+pp.c[i,2]*(3*pp.x[i]^2-3l*pp.x[i]+l^2+2)) - pdfR*(pp.b[i]-pp.c[i,1]*(2*pp.x[i]-r)+pp.c[i,2]*(3*pp.x[i]^2-3*pp.x[i]*r+r^2+2))		
        integral += moment
        pdfL = pdfR
        cdfL = cdfR
    end
    for j = i+1:n-1
        l = pp.x[j]
        r = pp.x[j+1]
        pdfR = pdfCache[j+1]
        cdfR = cdfCache[j+1]
        moment = (pp.a[j] - pp.x[j]*(pp.b[j]+pp.c[j,2]*(pp.x[j]^2+3)) + pp.c[j,1]*(pp.x[j]^2+1)) * (cdfR-cdfL)
		moment += pdfL*(pp.b[j]-pp.c[j,1]*(2*pp.x[j]-l)+pp.c[j,2]*(3*pp.x[j]^2-3l*pp.x[j]+l^2+2)) - pdfR*(pp.b[j]-pp.c[j,1]*(2*pp.x[j]-r)+pp.c[j,2]*(3*pp.x[j]^2-3*pp.x[j]*r+r^2+2))		
        integral += moment
        pdfL = pdfR
        cdfL = cdfR
    end
    l = max(pp.x[n], ck)
    slope = evaluateDerivativePiece(pp,n-1,pp.x[n])
    yn = evaluatePiece(pp,n-1,pp.x[n])
    moment = (yn-slope*pp.x[n])*(1-cdfL) + slope*pdfL
    integral += moment
    if isnan(integral) || isinf(integral)
        println(ck, " ", ckIndex, " inf integral ", p.g)
        throw(DomainError("infinite integral"))
    end
# end
# println("elapsed i ",elapsed)
    return integral
end


function makeBSplineCollocationGuess(
    strikes::Vector{T},
    callPrices::Vector{T},
    weights::Vector{T},
    τ::T,
    forward::T,
    discountDf::T;
    N=2,
    size = 0,
    extrapolationFactor=1.25,
    name="Bachelier", 
) where {T}
if name == "Bachelier"
    return fitBSplineBachelier(strikes, callPrices, weights, τ, forward, discountDf, size = size,N=N,extrapolationFactor=extrapolationFactor)
else 
    return fitBSplineGuess(strikes, callPrices, weights, τ, forward, discountDf, size = size,N=N)
end
end

function makeBSplineCollocation(
    strikes::Vector{T},
    callPrices::Vector{T},
    weights::Vector{T},
    τ::T,
    forward::T,
    discountDf::T;
    minSlope = 1e-4,
    penalty = 0.0,
    size = 0,
    N = 2, extrapolationFactor=1.25,
    rawFit = false, guess = "Bachelier",optimizerName="GaussNewton"
)::Tuple{BSplineCollocation,FitResult} where {T} #return collocation and error measure
    strikesf, pricesf, weightsf = filterConvexPrices(
        strikes,
        callPrices ./ discountDf,
        weights,
        forward,
        tol = minSlope + sqrt(eps(one(minSlope)))    )
   
    isoc = makeBSplineCollocationGuess(strikesf, pricesf, weightsf, τ, forward, 1.0, N=N, size = size, name=guess,extrapolationFactor=extrapolationFactor)
    isoc, m =
        rawFit ? fit(isoc, strikes, callPrices, weights, forward, discountDf, minSlope = minSlope, penalty = penalty,optimizerName=optimizerName) :
        fit(isoc, strikesf, pricesf, weightsf, forward, discountDf, minSlope = minSlope, penalty = penalty,optimizerName=optimizerName)
    return isoc, m
end

using FastGaussQuadrature
function fitBSplineBachelier(strikes, prices, weights, τ, forward, discountDf; size::Int = 0, extrapolationFactor=1.25,  N=2, slopeTolerance::Float64 = sqrt(eps(Float64)))
    m = length(strikes)
    i = findfirst(x -> x > forward, strikes)
    if i == nothing
        i = m
    elseif i == 1
        i = 2
    end
    price = (prices[i] * (forward - strikes[i-1]) + prices[i-1] * (strikes[i] - forward)) / (strikes[i] - strikes[i-1])
    strike = forward
    bvol = Bachelier.bachelierImpliedVolatility(price, true, strike, τ, forward, discountDf)
    # isoc = IsotonicCollocation(Polynomials.Polynomial([sqrt(bvol * sqrt(τ))]), Polynomials.Polynomial([0.0]), forward)
    σ = bvol * sqrt(τ)
    # need to use black because ys need to be > 0. xs < 0 ok.
    strikesf, pif, xf = makeXFromUndiscountedPrices(strikes, prices, slopeTolerance = slopeTolerance)
    #println("xf ",xf, strikesf)
    mindx = minimum(xf[2:end] - xf[1:end-1])
    if mindx < zero(xf[1])
        throw(DomainError(mindx, "dx negative, x is decreasing"))
    end

    local x::Vector{Float64}
    local n::Int
    if size == 0
        n = length(xf)
        x = copy(xf)
        #  @. x[1:n-1] = (xf[1:n-1]+xf[2:n])/2
         x[n] = max(xf[end] * extrapolationFactor, 2.0)
         x[1] = min(xf[1]*extrapolationFactor,-2.0)
         # x[1:n-1] = xf[1:n-1]
        # x[n] = max(xf[end] +(xf[end]-xf[end-1])/2, 2.0)
        #      x[1] =  xf[1] - (xf[2] - xf[1])/2
    else
        n = max(size, 3) + 1
          x = collect(range(min(xf[1]*extrapolationFactor, -3.0), stop = max(xf[end]*1.25, 3.0), length = n))
        #  for m=n:n*10
        #     xg = gausshermite(m)[1]
        #     x = filter(x -> x >= xf[1]*1.25 && x <= xf[end]*1.25, xg)
        #     if length(x) >= n
        #         break
        #     end
        # end
        n = length(x)
        # x = xf[round.(Int,range(1,stop=length(xf),length=n))] #if many obs two points close by may create artif non smooth dens?
     end
   # println("x ", x,"\n xf =",xf,"\n strikef =",strikesf, "\n forward=",forward,"\n sigma=",σ)
    # yi = sigma xi + F = sigma(x-xi)+F-sigmax
    a = @. (forward + σ*x)
    b = @. (σ * ones(Float64, n))
    c = zeros(Float64, (n,N-1))
    pp = FastPP{N,Float64,Float64}(a, b, c, x)
    # pp = FastPP(N,a, b, c, x)
    #println(pp)
    isoc = BSplineCollocation(pp, forward)
    adjustForward(isoc)
    return isoc
end


function fitBSplineGuess(strikes, prices, weights, τ, forward, discountDf; size::Int = 0,  N=2, slopeTolerance::Float64 = 1e-5)
    #quadprog for quad spline
    strikesf, pif, xf = makeXFromUndiscountedPrices(strikes, prices, slopeTolerance = sqrt(eps(Float64)))
    mindx = minimum(xf[2:end] - xf[1:end-1])
    if mindx < zero(xf[1])
        throw(DomainError(mindx, "dx negative, x is decreasing"))
    end
    local x::Vector{Float64}
    local n::Int
    if size == 0
        n = length(xf)
        x = copy(xf)
        #  @. x[1:n-1] = (xf[1:n-1]+xf[2:n])/2
         x[n] = xf[end] * 1.25
         x[1] = xf[1]*1.25
         # x[1:n-1] = xf[1:n-1]
        # x[n] = max(xf[end] +(xf[end]-xf[end-1])/2, 2.0)
        #      x[1] =  xf[1] - (xf[2] - xf[1])/2
    else
        n = max(size, 3) + 1
         x = collect(range(min(xf[1]*1.25, -2.0), stop = max(xf[end]*1.25, 2.0), length = n))
        # x = xf[round.(Int,range(1,stop=length(xf),length=n))] #if many obs two points close by may create artif non smooth dens?
     end
     pdfCache = normpdf.(x)
     cdfCache = normcdf.(x)
    
     #TODO right now don't include martingale constraint, only monotonicity.
    basis = BSplineBasis(N+1, x)
    t = BSplines.knots(basis)    
    α = zeros(Float64, length(basis)) # = length x + 1
    M = zeros(Float64, (length(xf),length(basis)))
    for (i,xi) in enumerate(xf)
        bvalues = bsplines(basis,xi)
        M[i,LinearIndices(bvalues)] = no_offset_view(bvalues)
    end
    #b[1]*(xInf-x1)+a[1]=strikeinf
    # xInf = -3.0
    # j=1
    # M[length(xf)+1,1] = -2/(t[j+3]-t[j+1]) *(xInf-x[1]) + (1 - (t[j+2]-t[j+1])/(t[j+3]-t[j+1]))
    # M[length(xf)+1,2] = 2/(t[j+3]-t[j+1]) *(xInf-x[1]) + (t[j+2]-t[j+1])/(t[j+3]-t[j+1])
    # dvec = M'*vcat(strikesf,0.0)
    dvec = M' * strikesf
    nConstraints = length(basis)
    G = spzeros(Float64, (nConstraints,length(basis)))
    q = zeros(Float64,nConstraints)
    #monotone conditions
    for i=2:length(basis)
        factor =  N / (t[i+N]-t[i])
        G[i,i-1] = -factor
        G[i,i] = factor
        q[i] = slopeTolerance
    end
    #martingale conditions for quad spline
    for i = 1:length(x)-1
        xi = x[i]
        phi0 = cdfCache[i]
        phi1 = cdfCache[i+1]
        phid0 = pdfCache[i]
        phid1 = pdfCache[i+1]
        j = i-1
        aFactor = (t[j+3] - t[j+2]) / (t[j+4] - t[j+2])
        bFactor = 2 / (t[j+4] - t[j+2])
        cFactor = 1 / ((t[j+4] - t[j+3]) * (t[j+4] - t[j+2]))
        cFactorP = 1 / ((t[j+4] - t[j+3]) * (t[j+5] - t[j+3]))
        # println(i," ",aFactor," ",bFactor," ",cFactor," ",cFactorP)
        c0 = (1-aFactor+xi*bFactor+(xi*xi+1)*cFactor)*(phi1-phi0) - (bFactor+xi*cFactor)*phid0 - (-bFactor+cFactor*(x[i+1]-2*xi))*phid1
        c1 = (aFactor-xi*bFactor-(xi*xi+1)*(cFactorP+cFactor))*(phi1-phi0) + (bFactor+xi*(cFactor+cFactorP))*phid0 - (bFactor-(cFactor+cFactorP)*(x[i+1]-2*xi))*phid1
        c2 = cFactorP*(xi*xi+1)*(phi1-phi0) - cFactorP*xi*phid0 - cFactorP*(x[i+1]-2*xi)*phid1
        G[1,i] += c0
        G[1,i+1] += c1
        G[1,i+2] += c2
    end
    i=1
    phi0 = cdfCache[i]
    phid0 = pdfCache[i]
    j = i-1
    aFactor = (t[j+3] - t[j+2]) / (t[j+4] - t[j+2])
    bFactor = 2 / (t[j+4] - t[j+2])
    c0 = (1-aFactor+x[i]*bFactor)*phi0 + bFactor*phid0
	c1 = (aFactor-x[i]*bFactor)*phi0 - bFactor*phid0
    G[1,i] += c0
    G[1,i+1] += c1
    i=length(x)
    phi1 = 1-cdfCache[i]
    phid1 = pdfCache[i]
    j = i-1
    aFactor = (t[j+3] - t[j+2]) / (t[j+4] - t[j+2])
    bFactor = 2 / (t[j+4] - t[j+2])
    c0 = (1-aFactor+x[i]*bFactor)*phi1 - bFactor*phid1
	c1 = (aFactor-x[i]*bFactor)*phi1 + bFactor*phid1
    G[1,i] += c0
    G[1,i+1] += c1
    q[1] = forward
    #call quadprog
    amat, aind = convertSparse(copy(G'))
  
    α, lagr, crval, iact, nact, iter = solveQPcompact(M'*M, dvec, amat,aind, q,meq= 1, factorized=false)		
    println(α)
    bspl = Spline(basis, α)
    isoc = BSplineCollocation(convert(FastPP{N,Float64,Float64},bspl),forward, BSplineCollocationCache(pdfCache,cdfCache))
    adjustForward(isoc)
    #we could adjust the knots eventually by inverse interpolation and recalculate the Bspl.
    return isoc
end
    
Base.length(p::BSplineCollocation) = Base.length(p.g.x)
Base.broadcastable(p::BSplineCollocation) = Ref(p)

using ForwardDiff

function fit(isoc::BSplineCollocation{N,T,U}, strikes::AbstractArray{U}, prices, weights::AbstractArray{U}, forward::U, discountDf::U=one(U); minSlope = 1e-8, penalty =zero(U),optimizerName="GaussNewton") where {N,T,U}
    iter = 0
    cache = isoc.cache
    basis = BSplineBasis(N+1, isoc.g.x)
    t = BSplines.knots(basis)
    c = zeros(Float64, length(basis)) # = length x + 1
    spl = convert(BSplines.Spline, isoc.g)
    ct = zeros(Float64, length(spl.coeffs) - 1)
    minValue = max(1e-8 * (spl.coeffs[end] - spl.coeffs[1]), minSlope)
    maxValue = 2 * (spl.coeffs[end] - spl.coeffs[1])
    # println("initial coeffs ",spl.coeffs)
    transforms = [MQMinTransformation(minSlope*(t[j+N]-t[j]),1.0) for j=1:length(basis)-1]
    # transform = ClosedTransformation(minValue, maxValue)
    for i = 1:length(ct)
        ct[i] = inv(transforms[i], min(max(spl.coeffs[i+1] - spl.coeffs[i], sqrt(eps(U))+minSlope*(t[i+N]-t[i])), maxValue))
        # println(i, " ", spl.coeffs[i+1] - spl.coeffs[i]," ct ",ct[i])
    end
    function obj!(fvec, ct0::AbstractArray{TC}) where {TC}

        α = zeros(TC, length(basis)) # = length x + 1
        for i=1:length(basis)-1
            α[i] = transforms[i](ct0[i])
        end
        suma = -sum(α)
        prev = α[1]
        α[1] = suma  #balance out theoretical forward such that it does not explode
        for i = 2:length(α)            
            next = α[i]
            α[i] = α[i-1] + prev
            prev = next
        end
        # println("iteration ",iter," ", ForwardDiff.value.(ct0))
        spl = BSplines.Spline(basis, α)
        pp = convert(FastPP{N,TC,U}, spl)
        lsc = BSplineCollocation(pp, forward,cache)
        # println("spl elapsed ",elapsed)
        # println("lsc ",α)
        adjustForward(lsc)
        iter += 1
        n = length(strikes)
        @. fvec[1:n] = weights * (priceEuropean(lsc, true, strikes, forward, discountDf) - prices)

        if penalty > 0
            #  @. fvec[n+1:end] = ((1/lsc.g.b[2:end] - 1/lsc.g.b[1:end-1]) * penalty) #more appropriate if transform is unbounded > 0
            #  t = BSplines.knots(spl.basis)
  
            #  for j = 3:length(basis)
            #     factor1j = 2 / (t[j+3-2] - t[j-1])
            #     factor1jm = 2 / (t[j-1+3-2] - t[j-2])
            #     factorj = 1 / (t[j+3-3] - t[j-1])
            #     d1d = factorj*factor1j*(α[j-1]-α[j-2]) + factorj*factor1jm*(α[j-3]-α[j-2])
            #     fvec[j-3+n+1] = penalty * (d1d) 
            #  end
             fvec[n+1:end] = @. (lsc.g.c[1:end,1] * penalty) 
            # fvec[n+1:end] = @. ((α[3:end+2-N]-2α[2:end-1+2-N]+α[1:end-2+2-N]) * penalty) 
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
    fvec = zeros(Float64, outlen)
    fr = if optimizerName == "GaussNewton"
        measure = GaussNewton.optimize!(obj!,ct,fvec,reltol=1e-8)
        ct0 = ct
        FitResult(measure, iter, ct, fvec, measure)
    else
        fit =  LeastSquaresOptim.optimize!(
        LeastSquaresProblem(x = ct, f! = obj!, autodiff = :forward,
        # g! = jac!, #useful to debug issue with ForwardDiff NaNs
        output_length = outlen),
        LevenbergMarquardt();
        iterations = 1024,
        # f_tol = 1e-7,g_tol=1e-9,
    # x_tol= 1e-4,
    )
    obj!(fvec,fit.minimizer)
    # fit = fsolve(obj!, jac!, ct, outlen; show_trace=false, method=:lm, tol=1e-8) #fit.x
    #println(iter, " fit ", fit,  #obj(fit.x))  #fit.f
    ct0 = fit.minimizer
    measure = fit.ssr #sqrt(sum(x -> x^2, fit.f)/length(fit.f)) #fit.ssr
    FitResult(fit.ssr, iter, fit.minimizer, fvec, fit)
end
   
    for i=1:length(basis)-1
        c[i] = transforms[i](ct0[i])
    end
    suma = -sum(c)
    prev = c[1]
    c[1] = suma  #balance out theoretical forward such that it does not explode
    for i = 2:length(c)            
        next = c[i]
        c[i] = c[i-1] + prev
        prev = next
    end
    spl = BSplines.Spline(basis, c)
    
    pp = convert(FastPP{N,T,U}, spl)
    lsc = BSplineCollocation(pp, forward,cache)
    adjustForward(lsc)
    return lsc, fr
end

