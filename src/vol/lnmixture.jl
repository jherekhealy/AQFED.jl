using LinearAlgebra
using LeastSquaresOptim
import Statistics: median
#using GaussNewton
import AQFED.Black: blackScholesFormula, impliedVolatilitySRHalley, Householder
import AQFED.Math: normpdf, normcdf, ClosedTransformation, IdentityTransformation, ExpMinTransformation, MQMinTransformation,    inv, toPositiveHypersphere!, fromPositiveHypersphere!
export LognormalKernel, priceEuropean, density, cumulativeDensity, calibrateLognormalMixture, calibrateLognormalMixtureSoftmax, calibrateLognormalMixtureGrid
struct LognormalKernel{TV,TX,TS}
    x::Vector{TX}
    σ::Vector{TS}
    α::Vector{TV}
end

Base.broadcastable(p::LognormalKernel) = Ref(p)


function priceEuropean(model::LognormalKernel, isCall::Bool, strike::T) where {T}
    sum = zero(T)
    for (xi, σi, αi) = zip(model.x, model.σ, model.α)
        pricei = blackScholesFormula(isCall, strike, xi, σi^2, one(T), one(T))
        sum += αi * pricei
    end
    return sum
end


function density(model::LognormalKernel, strike::T) where {T}
    sum = zero(T)
    for (xi, σi, αi) = zip(model.x, model.σ, model.α)
        pricei = normpdf(log(strike / xi) / σi + σi / 2) / (xi * σi)
        sum += αi * pricei
    end
    return sum
end

function cumulativeDensity(model::LognormalKernel,strike::T) where {T}
    sum = zero(T)
    for (xi, σi, αi) = zip(model.x, model.σ, model.α)
        pricei = normcdf(log(strike / xi) / σi + σi / 2) 
        sum += αi * pricei
    end
    return sum
end

function optimizeLognormalKernel(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; size::Int=length(strikes), logmoneynessFactor=1.25,solver="COSMO") where {T}
    x = forward.*exp.(range(log(strikes[1]/forward)*logmoneynessFactor,stop=log(strikes[end]/forward)*logmoneynessFactor,length=size))
    c = sqrt((log(x[2])-log(x[1]))^2+ (log(x[3])-log(x[1]))^2)/2
    obj = function (ψ)
        kernel,errorMeasure= calibrateLognormalKernel(tte, forward, strikes, callPrices, weights, x,ψ,solver=solver)
        errorMeasure
    end
    res = Optim.optimize(obj, c/2, c*2, method=Optim.Brent(), rel_tol=1e-4)
    return calibrateLognormalKernel(tte, forward, strikes, callPrices, weights, x,res.minimizer,solver=solver)
end

function calibrateLognormalKernel(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T};size::Int=length(strikes), atmVolFactor=1.0,solver="COSMO") where {T}
#fixed x = strikes (or hermite points) make sure first moment = forward == additional constraint on alpha_i
#fixed sigma = N*atm.
#optimize alpha only. linear problem.
s = searchsortedfirst(strikes, forward) #index of forward  
atmVol = impliedVolatilitySRHalley(true, callPrices[s], forward, forward, tte, 1.0, 0e-14, 64, Householder())
# x = strikes
x = forward.*exp.(range(log(strikes[1]/forward)*1.25,stop=log(strikes[end]/forward)*1.25,length=size))
c = atmVolFactor*atmVol*sqrt(tte)  
#c = sqrt((log(x[2])-log(x[1]))^2+ (log(x[3])-log(x[1]))^2)/2
return calibrateLognormalKernel(tte, forward, strikes, callPrices, weights, x,c,solver=solver,)
end

function calibrateLognormalKernel(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T},x::AbstractVector{T},c; solver="COSMO") where {T}
n = length(strikes)
m = length(x)
B = zeros(T,(n,m)) 
v= zeros(T,n)
for i=1:n
    isCall = strikes[i] >= forward
    for j=1:m
        B[i,j] = sqrt(weights[i])* blackScholesFormula(isCall, strikes[i], x[j], c^2, one(T), one(T))
    end
    if isCall
    v[i] = sqrt(weights[i])*callPrices[i]
    else
        v[i] = sqrt(weights[i])*(callPrices[i]-(forward-strikes[i]))
    end
end
A1 =spzeros(T,(2,m))
b1 = zeros(T,2)
A1[1,:] = ones(T,m)
b1[1] = -one(T)
A1[2,:] = x
b1[2] = -forward
b2 = zeros(T,m)
A2 = spzeros(T, (m,m))
for i=1:m
    A2[i,i] = one(T)
end
α = if solver == "COSMO"
    settings = COSMO.Settings(verbose=false,eps_abs=1e-16,eps_rel=1e-16)
    model = COSMO.Model()
    #println(size(A')," ", size(C), " ",n)
    constraint1 = COSMO.Constraint(A1, b1, COSMO.ZeroSet)
    constraint2 = COSMO.Constraint(A2, b2, COSMO.Nonnegatives)
    assemble!(model, B'*B, (-B'*v)', [constraint1; constraint2], settings=settings)
    res = COSMO.optimize!(model)
    res.x
else
    Aa = hcat(A1',A2')
    amat, aind = convertSparse(Aa)
    bb = vcat(-b1,-b2)
    #println(size(Aa)," ",size(bb))
    try
     sol, lagr, crval, iact, nact, iter = solveQPcompact(B'*B, v'*B, amat, aind, bb, meq=2,factorized=false)
     sol
    catch e
        println("error with c=",c)
        A = zeros(T,(m,m))
        b = zeros(T,m)
        b[1] = one(T)
        b[2] = forward
        A[1,:]  = ones(T,m)
        A[2,:] = x
        for i=3:m
            A[i,i] = one(T)
            A[i,i-1]= -one(T)
        end
        sol = A\b
        sol
    end
     #  sol, lagr, crval, iact, nact, iter = solveQPcompact(Base.inv(B), v' * B, amat, aind, bb, meq=2,factorized=true)
end
r = B * α - v
errorMeasure = r' * r
return LognormalKernel(x,ones(T,m)*c,α), errorMeasure
end

function calibrateLognormalMixture(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; size::Int=4, useVol=false,transformName="abs") where {T}
    x = zeros(T, size)
    width = (strikes[end] - strikes[1]) / (forward * 2size)
    for i = 1:size
        x[i] = forward * exp((i - (size + 1) / 2) * width)
    end
    s = searchsortedfirst(strikes, forward) #index of forward
    #initial guess
    vols = zeros(T, length(strikes))
    if useVol
        for i = eachindex(vols)
            vols[i] = impliedVolatilitySRHalley(true, callPrices[i], forward, strikes[i], tte, 1.0, 0e-14, 64, Householder())
        end
    end
    atmVol = impliedVolatilitySRHalley(true, callPrices[s], forward, forward, tte, 1.0, 0e-14, 64, Householder())
    σ = zeros(T, size)
   
    sumkx = zero(T)
    kx = zeros(T, size)
    # if size == 4
    #     α = [0.175, 0.325, 0.325, 0.175]
    # elseif size==6
    #     α = [0.085, 0.18, 0.235, 0.235, 0.18, 0.085]
    # else
    #     α = ones(T,size)/size
    # end    
    α = ones(T,size)/size
    for i = eachindex(σ)
        σ[i] = atmVol * sqrt(tte)
        kx[i] = α[i] * x[i]
        sumkx += kx[i]
    end
    kx[1] += forward - sumkx
    if kx[1] < zero(T)
        kx[1] = sqrt(eps(T))
    end

    xv = zeros(T, size * 3 - 2) #guess of minim
    fromPositiveHypersphere!(@view(xv[1:size-1]), one(T), α)
    fromPositiveHypersphere!(@view(xv[size:2size-2]), sqrt(forward), kx)
    transform = if transformName == "abs"
        IdentityTransformation{T}()
    elseif transformName == "exp"
        ExpMinTransformation(σ[1]/10)
    elseif transformName == "mq"
        MQMinTransformation(σ[1]/10,one(T))
    else
        ClosedTransformation(σ[1]/10, σ[1]*10)
    end
      @.  xv[2*size-1:size*3-2] = inv(transform,σ)
    
   # println("xv ",xv," ",α, " ",kx)
    function obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        αg = zeros(W, size)
        toPositiveHypersphere!(αg, one(T), @view(c[1:size-1]))
        kxg = zeros(W, size)
        toPositiveHypersphere!(kxg, sqrt(forward), @view(c[size:2size-2]))
        xg = @. (ifelse(αg == zero(W), kxg/1e-8,kxg / αg))
        #  σg = (@view(c[2size-1:end]) ).^ 2 .+ σmin
        σg = @. abs(transform(@view(c[2size-1:end]) ))
        # println("LognormalKernel ",xg," ",αg," ",σg)
        if useVol
            # println( θ)
            for (i, strike) = enumerate(strikes)
                mPrice = priceEuropean(LognormalKernel(xg, σg, αg), strike >= forward, strike)
                fvec[i] = impliedVolatilitySRHalley(strike >= forward, mPrice, forward, strike, tte, 1.0, 0e-14, 64, Householder()) - vols[i]
            end
        else
            for (i, strike) = enumerate(strikes)
                mPrice = priceEuropean(LognormalKernel(xg, σg, αg), strike >= forward, strike)
                # println(strike, " ",mPrice)
                otmPrice = callPrices[i]
                if strike < forward 
                    otmPrice -= forward-strike
                end
                fvec[i] = weights[i] * (mPrice - otmPrice) #FIXME what if forward not on list?
            end
        end
        fvec
    end
    fit = LeastSquaresOptim.optimize!(
        LeastSquaresProblem(x=xv, (f!)=obj!, autodiff=:forward, #:forward is 4x faster than :central
            output_length=length(callPrices)),
        LevenbergMarquardt();
        iterations=1000
    )
    # fvec = zeros(Float64, length(callPrices))
    # obj!(fvec, fit.minimizer)
    xv = fit.minimizer
    toPositiveHypersphere!(α, one(T), @view(xv[1:size-1]))
    toPositiveHypersphere!(kx, sqrt(forward), @view(xv[size:2size-2]))
    for i = eachindex(x)
        if α[i] != zero(T)
            x[i] = kx[i] / α[i]
        else
            x[i] = sqrt(eps(T))
        end
    end
    for i = eachindex(σ)
        # σ[i] = (xv[i+2size-2] )^2 + σmin
        σ[i] = abs(transform(xv[i+2size-2] ))
    end
    sumw = zero(T)
    for i = eachindex(α)
        sumw += α[i]
    end
    # println("sumw ",sumw," fit ",fit)
    return LognormalKernel(x, σ, α)
end

function calibrateLognormalMixture(tte::T, forward::T, axisTransforms::Vector{U}, xs::AbstractVector{T}, vols::AbstractVector{T}, weights::AbstractVector{T}; size::Int=4, useVol=true,transformName="abs") where {T, U <: AxisTransformation}
    strikes = map((trans, x,vol) -> forward*exp(convertToLogmoneyness(trans, x, vol)), axisTransforms, xs, vols)
    
    x = zeros(T, size)
    width = (strikes[end] - strikes[1]) / (forward * 2size)
    for i = 1:size
        x[i] = forward * exp((i - (size + 1) / 2) * width)
    end
    s = searchsortedfirst(strikes, forward) #index of forward
    #initial guess
    
    atmVol = vols[s]
    σ = zeros(T, size)
   
    sumkx = zero(T)
    kx = zeros(T, size)
    # if size == 4
    #     α = [0.175, 0.325, 0.325, 0.175]
    # elseif size==6
    #     α = [0.085, 0.18, 0.235, 0.235, 0.18, 0.085]
    # else
    #     α = ones(T,size)/size
    # end    
    α = ones(T,size)/size
    for i = eachindex(σ)
        σ[i] = atmVol * sqrt(tte)
        kx[i] = α[i] * x[i]
        sumkx += kx[i]
    end
    kx[1] += forward - sumkx
    if kx[1] < zero(T)
        kx[1] = sqrt(eps(T))
    end

    xv = zeros(T, size * 3 - 2) #guess of minim
    fromPositiveHypersphere!(@view(xv[1:size-1]), one(T), α)
    fromPositiveHypersphere!(@view(xv[size:2size-2]), sqrt(forward), kx)
    transform = if transformName == "abs"
        IdentityTransformation{T}()
    elseif transformName == "exp"
        ExpMinTransformation(σ[1]/10)
    elseif transformName == "mq"
        MQMinTransformation(σ[1]/10,one(T))
    else
        ClosedTransformation(σ[1]/10, σ[1]*10)
    end
      @.  xv[2*size-1:size*3-2] = inv(transform,σ)
    
   # println("xv ",xv," ",α, " ",kx)
    function obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        αg = zeros(W, size)
        toPositiveHypersphere!(αg, one(T), @view(c[1:size-1]))
        kxg = zeros(W, size)
        toPositiveHypersphere!(kxg, sqrt(forward), @view(c[size:2size-2]))
        xg = @. (ifelse(αg == zero(W), kxg/1e-8,kxg / αg))
        #  σg = (@view(c[2size-1:end]) ).^ 2 .+ σmin
        σg = @. abs(transform(@view(c[2size-1:end]) ))
        # println("LognormalKernel ",xg," ",αg," ",σg)
        varianceByLogmoneynessFunction = function(y)
            strike = exp(y)*forward
            mPrice = priceEuropean(LognormalKernel(xg, σg, αg), y >= 0, strike)
            return impliedVolatility(y >= 0, mPrice, forward, strike, tte, 1.0)^2
        end
        if useVol
            # println( θ)
            for (i, xi) = enumerate(xs)
                strike = forward*exp(solveLogmoneyness(axisTransforms[i], xi, varianceByLogmoneynessFunction))
                mPrice = priceEuropean(LognormalKernel(xg, σg, αg), strike >= forward, strike)
                fvec[i] = impliedVolatility(strike >= forward, mPrice, forward, strike, tte, 1.0) - vols[i]
            end
        else
            for (i, xi) = enumerate(xs)
                strike = forward*exp(solveLogmoneyness(axisTransforms[i], xi, varianceByLogmoneynessFunction))
                mPrice = priceEuropean(LognormalKernel(xg, σg, αg), strike >= forward, strike)
                # println(strike, " ",mPrice)
                otmPrice = callPrices[i]
                if strike < forward 
                    otmPrice -= forward-strike
                end
                fvec[i] = weights[i] * (mPrice - otmPrice) #FIXME what if forward not on list?
            end
        end
        fvec
    end
    fvec = zeros(Float64, length(vols))
    rmse, fit = GaussNewton.optimize!(obj!, xv, fvec)
    # fit = LeastSquaresOptim.optimize!(
    #     LeastSquaresProblem(x=xv, (f!)=obj!, autodiff=:forward, #:forward is 4x faster than :central
    #         output_length=length(vols)),
    #     LevenbergMarquardt();
    #     iterations=1000
    # )
    # fvec = zeros(Float64, length(callPrices))
    # obj!(fvec, fit.minimizer)
    xv = fit.minimizer
    toPositiveHypersphere!(α, one(T), @view(xv[1:size-1]))
    toPositiveHypersphere!(kx, sqrt(forward), @view(xv[size:2size-2]))
    for i = eachindex(x)
        if α[i] != zero(T)
            x[i] = kx[i] / α[i]
        else
            x[i] = sqrt(eps(T))
        end
    end
    for i = eachindex(σ)
        # σ[i] = (xv[i+2size-2] )^2 + σmin
        σ[i] = abs(transform(xv[i+2size-2] ))
    end
    sumw = zero(T)
    for i = eachindex(α)
        sumw += α[i]
    end
    # println("sumw ",sumw," fit ",fit)
    return LognormalKernel(x, σ, α)
end


function calibrateLognormalMixtureFX(tte::T, forward::T, strikes::AbstractVector{T}, vols::AbstractVector{T};useVol=true,transformName="mq") where {T}
    size=3
    x = zeros(T, size)
    width = (strikes[end] - strikes[1]) / (forward * 2size)
    for i = 1:size-1
        x[i] = forward * exp((i - (size ) / 2) * width)
    end
    x[end]=strikes[3]
    #initial guess
    σ = zeros(T, size)
    atmVol = vols[3]
    sumkx = zero(T)
    kx = zeros(T, size)
    α = ones(T,size)/size
    for i = eachindex(σ)
        σ[i] = atmVol * sqrt(tte)
        kx[i] = α[i] * x[i]
        sumkx += kx[i]
    end

    kx[1] += forward - sumkx 
    if kx[1] < zero(T)
        kx[1] = sqrt(eps(T))
    end

    xv = zeros(T, size * 3 - 2) #guess of minim
    fromPositiveHypersphere!(@view(xv[1:size-1]), one(T), α)
    fromPositiveHypersphere!(@view(xv[size:2size-2]), sqrt(forward), kx[1:end])
    transform = if transformName == "abs"
        IdentityTransformation{T}()
    elseif transformName == "exp"
        ExpMinTransformation(σ[1]/10)
    elseif transformName == "mq"
        MQMinTransformation(σ[1]/10,one(T))
    else
        ClosedTransformation(σ[1]/10, σ[1]*10)
    end
    @.  xv[2*size-1:size*3-2] = inv(transform,σ)
    fixedSigmaTrans = xv[end]
    fixedKx = xv[2*size-2]
    # println("xv ",xv," ",α, " ",kx)
    function obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        xv = vcat(c[1:size-1],c[size:2size-3],fixedKx,c[2*size-2:size*3-4],fixedSigmaTrans)
        αg = zeros(W, size)
        toPositiveHypersphere!(αg, one(T), @view(xv[1:size-1]))
        kxg = zeros(W, size)
        toPositiveHypersphere!(@view(kxg[1:end]), sqrt(forward), @view(xv[size:2size-2]))
        #kxg[end] = αg[end]*x[end]
        xg = @. (ifelse(αg == zero(W), kxg/1e-8,kxg / αg))
        #  σg = (@view(c[2size-1:end]) ).^ 2 .+ σmin
        σg = @. abs(transform(@view(xv[2size-1:end]) ))
        # println("LognormalKernel ",xg," ",αg," ",σg)
        if useVol
            # println( θ)
            for (i, strike) = enumerate(strikes)
                mPrice = priceEuropean(LognormalKernel(xg, σg, αg), strike >= forward, strike)
                fvec[i] = impliedVolatilitySRHalley(strike >= forward, mPrice, forward, strike, tte, 1.0, 0e-14, 64, Householder()) - vols[i]
            end
        else
            for (i, strike) = enumerate(strikes)
                mPrice = priceEuropean(LognormalKernel(xg, σg, αg), strike >= forward, strike)
                # println(strike, " ",mPrice)
                otmPrice = callPrices[i]
                if strike < forward 
                    otmPrice -= forward-strike
                end
                fvec[i] = weights[i] * (mPrice - otmPrice) #FIXME what if forward not on list?
            end
        end
        fvec
    end
    c=vcat(xv[1:size-1],xv[size+1:2size-2],xv[2size-2:3size-4])
    fit = LeastSquaresOptim.optimize!(
        LeastSquaresProblem(x=c, (f!)=obj!, autodiff=:forward, #:forward is 4x faster than :central
            output_length=length(strikes)),
        LevenbergMarquardt();
        iterations=1000
    )
    # fvec = zeros(Float64, length(callPrices))
    # obj!(fvec, fit.minimizer)
    c = fit.minimizer
    xv = vcat(c[1:size-1],c[size:2size-3],fixedKx,c[2*size-2:size*3-4],fixedSigmaTrans)
    toPositiveHypersphere!(α, one(T), @view(xv[1:size-1]))
    toPositiveHypersphere!(@view(kx[1:end]), sqrt(forward), @view(xv[size:2size-2]))
  #  kx[end] = α[end]*x[end]
  for i = eachindex(x)
        if α[i] != zero(T)
            x[i] = kx[i] / α[i]
        else
            x[i] = sqrt(eps(T))
        end
    end
    for i = eachindex(σ)
        # σ[i] = (xv[i+2size-2] )^2 + σmin
        σ[i] = abs(transform(xv[i+2size-2] ))
    end
    sumw = zero(T)
    for i = eachindex(α)
        sumw += α[i]
    end
    # println("sumw ",sumw," fit ",fit)
    return LognormalKernel(x, σ, α)
end

function calibrateLognormalMixtureFixedWeights(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T};  α = [0.05,0.45,0.45,0.05], useVol=false) where {T}
    size = length(α )
    x = zeros(T, size)
    width = (strikes[end] - strikes[1]) / (forward * 2size)
    for i = 1:size
        x[i] = forward * exp((i - (size + 1) / 2) * width)
    end
    s = searchsortedfirst(strikes, forward) #index of forward
    #initial guess
    vols = zeros(T, length(strikes))
    if useVol
        for i = eachindex(vols)
            vols[i] = impliedVolatilitySRHalley(true, callPrices[i], forward, strikes[i], tte, 1.0, 0e-14, 64, Householder())
        end
    end
    atmVol = impliedVolatilitySRHalley(true, callPrices[s], forward, forward, tte, 1.0, 0e-14, 64, Householder())
    σ = zeros(T, size)
   
    sumkx = zero(T)
    kx = zeros(T, size)
    for i = eachindex(σ)
        σ[i] = atmVol * sqrt(tte)
        kx[i] = α[i] * x[i]
        sumkx += kx[i]
    end
    kx[1] += forward - sumkx
    if kx[1] < zero(T)
        kx[1] = sqrt(eps(T))
    end

    xv = zeros(T, size * 2 - 1) #guess of minim
    fromPositiveHypersphere!(@view(xv[1:size-1]), sqrt(forward), kx)
     σmin = σ[1]/100
     xv[size:end] = sqrt.( σ .-  σmin)
    # xv[2size-1:end] = abs.(σ)
   # println("xv ",xv," ",α, " ",kx)
    function obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        kxg = zeros(W, size)
        toPositiveHypersphere!(kxg, sqrt(forward), @view(c[1:size-1]))
        xg = @. (ifelse(α == zero(W), kxg/1e-8,kxg / α))
         σg = (@view(c[size:end]) ).^ 2 .+ σmin
        # σg = abs.(@view(c[2size-1:end]) )
        # println("LognormalKernel ",xg," ",αg," ",σg)
        if useVol
            # println( θ)
            for (i, strike) = enumerate(strikes)
                mPrice = priceEuropean(LognormalKernel(xg, σg, α), strike >= forward, strike)
                fvec[i] = impliedVolatilitySRHalley(strike >= forward, mPrice, forward, strike, tte, 1.0, 0e-14, 64, Householder()) - vols[i]
            end
        else
            for (i, strike) = enumerate(strikes)
                mPrice = priceEuropean(LognormalKernel(xg, σg, α), strike >= forward, strike)
                # println(strike, " ",mPrice)
                otmPrice = callPrices[i]
                if strike < forward 
                    otmPrice -= forward-strike
                end
                fvec[i] = weights[i] * (mPrice - otmPrice) #FIXME what if forward not on list?
            end
        end
        fvec
    end
    fit = LeastSquaresOptim.optimize!(
        LeastSquaresProblem(x=xv, (f!)=obj!, autodiff=:forward, #:forward is 4x faster than :central
            output_length=length(callPrices)),
        LevenbergMarquardt();
        iterations=1000
    )
    # fvec = zeros(Float64, length(callPrices))
    # obj!(fvec, fit.minimizer)
    xv = fit.minimizer
    toPositiveHypersphere!(kx, sqrt(forward), @view(xv[1:size-1]))
    for i = eachindex(x)
        if α[i] != zero(T)
            x[i] = kx[i] / α[i]
        else
            x[i] = sqrt(eps(T))
        end
    end
    for i = eachindex(σ)
        σ[i] = (xv[i+size-1] )^2 + σmin
        # σ[i] = abs(xv[i+2size-2] )
    end
    sumw = zero(T)
    for i = eachindex(α)
        sumw += α[i]
    end
    # println("sumw ",sumw," fit ",fit)
    return LognormalKernel(x, σ, α)
end

#Option B: gauge-fixed softmax weights (w_0=0), global martingale shift of the log-centers and logistic bounded bandwidths.
#The martingale constraint sum α_k F_k = forward holds algebraically for any parameter value.
#The initial guess uses log-centers spread over ±centerSpread*σ_ATM√T around log(forward).
#centerSpread=0 (all components identical, as in the book) is a symmetric saddle point where the gradient w.r.t weights and centers
#vanishes (Proposition 1): it is omitted by default in favor of [0.5, 1.0, 2.0].
#An exact allocation-free analytical Jacobian is used by default for efficiency.
#solver=:LM uses the Levenberg-Marquardt algorithm of LeastSquaresOptim, solver=:GN the trust-region Gauss-Newton algorithm of GaussNewton
#(with the analytical Jacobian).
function calibrateLognormalMixtureSoftmax(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T};
    size::Int=4, useVol=false, λMinFactor=0.05, λMaxFactor=3.0, centerSpread=[0.5, 1.0, 2.0], iterations=1000, autodiff=false, solver=:LM, gnOptions=NamedTuple()) where {T}
    s = argmin(abs.(strikes .- forward)) #strike closest to the forward
    atmVol = impliedVolatilitySRHalley(true, callPrices[s], forward, strikes[s], tte, 1.0, 0e-14, 64, Householder())
    λAtm = atmVol * sqrt(tte)
    λMin = λMinFactor * λAtm
    λMax = λMaxFactor * λAtm
    vols = zeros(T, length(strikes))
    if useVol
        for i = eachindex(vols)
            vols[i] = impliedVolatilitySRHalley(true, callPrices[i], forward, strikes[i], tte, 1.0, 0e-14, 64, Householder())
        end
    end
    lnf = log(forward)
    K = size

    # Pre-allocated buffers for allocation-free function and Jacobian evaluation
    buf_F = zeros(T, K)
    buf_λ = zeros(T, K)
    buf_α = zeros(T, K)
    buf_p = zeros(T, K)
    buf_dλ = zeros(T, K)
    buf_v = zeros(T, K)
    buf_Δ = zeros(T, K)
    buf_D = zeros(T, K)

    function update_kernel!(c::AbstractArray{W}) where {W}
        wmax = zero(W)
        for j = 1:K-1
            wmax = max(wmax, c[j])
        end
        buf_α[1] = exp(-wmax)
        sum_e = buf_α[1]
        for j = 1:K-1
            buf_α[j+1] = exp(c[j] - wmax)
            sum_e += buf_α[j+1]
        end
        for k = 1:K
            buf_α[k] /= sum_e
        end
        zmax = -W(Inf)
        for k = 1:K
            zk = log(buf_α[k]) + c[K-1+k]
            buf_D[k] = zk
            zmax = max(zmax, zk)
        end
        sum_ez = zero(W)
        for k = 1:K
            sum_ez += exp(buf_D[k] - zmax)
        end
        lnFbar = zmax + log(sum_ez)
        for k = 1:K
            buf_F[k] = exp(c[K-1+k] - lnFbar + lnf)
            ωk = c[2K-1+k]
            e_neg_ω = exp(-ωk)
            buf_λ[k] = λMin + (λMax - λMin) / (one(W) + e_neg_ω)
            buf_p[k] = buf_α[k] * buf_F[k] / forward
            buf_dλ[k] = (buf_λ[k] - λMin) * (one(W) - (buf_λ[k] - λMin) / (λMax - λMin))
        end
    end

    toKernel = function (c::AbstractArray{W}) where {W}
        w = @view(c[1:size-1])
        xbar = @view(c[size:2size-1])
        ω = @view(c[2size:3size-1])
        wmax = max(zero(W), maximum(w; init=zero(W)))
        e = vcat(exp(-wmax), exp.(w .- wmax))
        α = e ./ sum(e)
        lα = log.(α)
        z = lα .+ xbar
        zmax = maximum(z)
        lnFbar = zmax + log(sum(exp.(z .- zmax)))
        F = @. exp(xbar - lnFbar + lnf)
        λ = @. λMin + (λMax - λMin) / (1 + exp(-ω))
        return LognormalKernel(F, λ, α)
    end

    function fast_obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        update_kernel!(c)
        for (i, strike) = enumerate(strikes)
            isCall = strike >= forward
            mPrice = zero(T)
            for k = 1:K
                mPrice += buf_α[k] * blackScholesFormula(isCall, strike, buf_F[k], buf_λ[k]^2, one(T), one(T))
            end
            otmPrice = callPrices[i]
            if !isCall
                otmPrice -= forward - strike
            end
            fvec[i] = weights[i] * (mPrice - otmPrice)
        end
        fvec
    end

    function fast_jac!(J, c)
        update_kernel!(c)
        for (i, strike) = enumerate(strikes)
            isCall = strike >= forward
            sum_v = zero(T)
            sum_D = zero(T)
            for k = 1:K
                buf_v[k] = blackScholesFormula(isCall, strike, buf_F[k], buf_λ[k]^2, one(T), one(T))
                sum_v += buf_α[k] * buf_v[k]
                d1 = (log(buf_F[k] / strike) + 0.5 * buf_λ[k]^2) / buf_λ[k]
                buf_Δ[k] = isCall ? normcdf(d1) : normcdf(d1) - one(T)
                buf_D[k] = buf_α[k] * buf_F[k] * buf_Δ[k]
                sum_D += buf_D[k]
            end
            for j = 1:K-1
                dV_dw = buf_α[j+1] * (buf_v[j+1] - sum_v) + (buf_α[j+1] - buf_p[j+1]) * sum_D
                J[i, j] = weights[i] * dV_dw
            end
            for m = 1:K
                dV_dxbar = buf_D[m] - buf_p[m] * sum_D
                J[i, K - 1 + m] = weights[i] * dV_dxbar
            end
            for k = 1:K
                d1 = (log(buf_F[k] / strike) + 0.5 * buf_λ[k]^2) / buf_λ[k]
                vega_k = buf_F[k] * normpdf(d1)
                dV_dω = buf_α[k] * vega_k * buf_dλ[k]
                J[i, 2K - 1 + k] = weights[i] * dV_dω
            end
        end
        return J
    end

    function obj_ad!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        kernel = toKernel(c)
        for (i, strike) = enumerate(strikes)
            mPrice = priceEuropean(kernel, strike >= forward, strike)
            if useVol
                fvec[i] = impliedVolatilitySRHalley(strike >= forward, mPrice, forward, strike, tte, 1.0, 0e-14, 64, Householder()) - vols[i]
            else
                otmPrice = callPrices[i]
                if strike < forward
                    otmPrice -= forward - strike
                end
                fvec[i] = weights[i] * (mPrice - otmPrice)
            end
        end
        fvec
    end

    initialGuess = function (spread)
        c0 = zeros(T, 3size - 1)
        half = (size - 1) / 2
        for k = 1:size
            c0[size+k-1] = lnf + (half == 0 ? zero(T) : spread * λAtm * (k - 1 - half) / half)
        end
        c0[2size:3size-1] .= -log((λMax - λMin) / (λAtm - λMin) - 1)
        return c0
    end

    starts = [initialGuess(spread) for spread in centerSpread]
    if solver == :GN && !(autodiff || useVol)
        fvec = zeros(T, length(callPrices))
        bestx, bestssr = starts[1], T(Inf)
        for start in starts
            _, result = GaussNewton.optimize!(fast_obj!, fast_jac!, copy(start), fvec; limit=10 * iterations, gnOptions...)
            x = result.minimizer
            fast_obj!(fvec, x)
            ssr = sum(abs2, fvec)
            if ssr < bestssr
                bestx, bestssr = x, ssr
            end
        end
        update_kernel!(bestx)
        return LognormalKernel(copy(buf_F), copy(buf_λ), copy(buf_α))
    end
    best = nothing
    for start in starts
        fit = if autodiff || useVol
            LeastSquaresOptim.optimize!(
                LeastSquaresProblem(x=start, (f!)=obj_ad!, autodiff=:forward, output_length=length(callPrices)),
                LevenbergMarquardt();
                iterations=iterations
            )
        else
            LeastSquaresOptim.optimize!(
                LeastSquaresProblem(x=start, (f!)=fast_obj!, (g!)=fast_jac!, output_length=length(callPrices)),
                LevenbergMarquardt();
                iterations=iterations
            )
        end
        if best === nothing || fit.ssr < best.ssr
            best = fit
        end
    end
    update_kernel!(best.minimizer)
    return LognormalKernel(copy(buf_F), copy(buf_λ), copy(buf_α))
end

#Option C: fixed grid of log-centers covering ±coverage*σ_ATM√T around the forward, extended to margin*σ_ATM√T beyond the extreme strikes:
#kernels of small bandwidth can not produce the tail mass of far out-of-the-money options otherwise.
#The bandwidth is overlap*Δx by default (book), or the given bandwidth (a total volatility, independent of the grid); with a given
#bandwidth and size=nothing, the grid spacing is at most the bandwidth. bandwidth=:auto is a quarter of the smallest total implied volatility
#σ_min√T of the quotes. wingMargin=true measures the margin with the total implied volatility of the extreme strikes instead of the
#at-the-money one: steep wings need centers much further out.
#The weights α solve the convex QP  min ‖W(Bα-V)‖² + μ αᵀRα  s.t. Σα=1, ΣαF=forward, α≥0,
#where R_kl = ∫ψ_k''(x)ψ_l''(x)dx penalizes the roughness of the log-density g, and μ = smoothing*median(diag(BᵀWB))/median(diag(R)).
#smoothing=:auto selects the largest μ in 1e-12..1e-2 whose RMSE stays within 25% (or 2e-4) of the smallest one.
#Without smoothing, BᵀWB is singular when size > length(strikes) and the solution is neither unique nor smooth.
#adaptive=true places the centers uniformly in z(x) = ∫dx/λ(x), where λ(x) is the total implied volatility of the quotes, linearly
#interpolated in log-strike and flat extrapolated, with the margin measured with the wing volatilities: the spacing and the bandwidth
#follow the local volatility, narrow kernels at the money and wide kernels in the wings, where quotes are sparse. The bandwidth is
#overlap times the local spacing, λ_k = overlap*λ(x_k)*Δz. Without size, Δz ≈ 1/resolution (λ_k ≈ overlap*λ(x_k)/4 ≈ λ(x_k)/3 by default with overlap=4/3).
#Returns the kernel, the weighted squared price error, and the smoothing factor used (the one selected with smoothing=:auto).
function calibrateLognormalMixtureGrid(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T};
    size::Union{Nothing,Int}=nothing, coverage=4.5, margin=2.0, overlap=4 / 3, bandwidth=nothing, smoothing=0.0, wingMargin=false,
    adaptive=false, resolution=4.0, solver="GI") where {T}
    s = argmin(abs.(strikes .- forward))
    atmVol = impliedVolatilitySRHalley(true, callPrices[s], forward, strikes[s], tte, 1.0, 0e-14, 64, Householder())
    λAtm = atmVol * sqrt(tte)
    #total volatility of the out-of-the-money option at strike index i
    wingλ(i) = impliedVolatilitySRHalley(strikes[i] >= forward, strikes[i] >= forward ? callPrices[i] : callPrices[i] - (forward - strikes[i]),
        forward, strikes[i], tte, 1.0, 0e-14, 64, Householder()) * sqrt(tte)
    λLeft, λRight = (wingMargin || adaptive) ? (wingλ(1), wingλ(length(strikes))) : (λAtm, λAtm)
    if bandwidth == :auto #a quarter of the smallest total implied volatility of the quotes
        bandwidth = minimum(wingλ, eachindex(strikes)) / 4
    end
    lnf = log(forward)
    xMin = min(lnf - coverage * λAtm, log(strikes[1]) - margin * λLeft)
    xMax = max(lnf + coverage * λAtm, log(strikes[end]) + margin * λRight)
    F, λ = if adaptive
        adaptiveLognormalGrid(log.(strikes), [wingλ(i) for i in eachindex(strikes)], xMin, xMax, size, overlap, resolution)
    else
        uniformLognormalGrid(xMin, xMax, size, overlap, bandwidth)
    end
    if smoothing == :auto
        factors = 10.0 .^ (-12:0.5:-2)
        fits = [calibrateLognormalMixtureFixedKernels(forward, strikes, callPrices, weights, F, λ, smoothing=μ, solver=solver) for μ in factors]
        rmse = [sqrt(err / length(strikes)) for (_, err) in fits]
        rmin = minimum(rmse)
        i = findlast(r -> r <= max(1.25rmin, rmin + 2e-4), rmse)
        return fits[i]..., factors[i]
    end
    return calibrateLognormalMixtureFixedKernels(forward, strikes, callPrices, weights, F, λ, smoothing=smoothing, solver=solver)..., smoothing
end

function uniformLognormalGrid(xMin::T, xMax::T, size, overlap, bandwidth) where {T}
    n = if size !== nothing
        size
    elseif bandwidth isa Number
        clamp(ceil(Int, (xMax - xMin) / bandwidth) + 1, 21, 801)
    else
        51
    end
    Δx = (xMax - xMin) / (n - 1)
    F = collect(exp.(range(xMin, stop=xMax, length=n)))
    λ = bandwidth === nothing ? fill(overlap * Δx, n) : (bandwidth isa Number ? fill(T(bandwidth), n) : collect(T, bandwidth))
    return F, λ
end

#centers uniform in z(x) = ∫dx/λ(x) for the piecewise linear total volatility λ(x) through the quotes (xq, λq), flat extrapolated.
#On a segment where λ(y) = λ0 + b(y-x0), z(x) - z(x0) = ln(λ(x)/λ0)/b, inverted as x = x0 + λ0(exp(b(z-z0)) - 1)/b (Δx/λ0 when b = 0).
function adaptiveLognormalGrid(xq::AbstractVector{T}, λq::AbstractVector{T}, xMin::T, xMax::T, size, overlap, resolution) where {T}
    #breakpoints of λ within [xMin, xMax], with the values of λ and z at each of them
    xb = vcat(xMin, filter(x -> xMin < x < xMax, xq), xMax)
    λb = map(xb) do x
        i = clamp(searchsortedlast(xq, x), 1, length(xq) - 1)
        t = clamp((x - xq[i]) / (xq[i+1] - xq[i]), zero(T), one(T))
        (1 - t) * λq[i] + t * λq[i+1]
    end
    slope(j) = (λb[j+1] - λb[j]) / (xb[j+1] - xb[j])
    Δzb = [abs(slope(j)) < sqrt(eps(T)) ? (xb[j+1] - xb[j]) / λb[j] : log(λb[j+1] / λb[j]) / slope(j) for j in 1:length(xb)-1]
    zb = vcat(zero(T), cumsum(Δzb))
    Z = zb[end]
    n = size !== nothing ? size : clamp(ceil(Int, resolution * Z) + 1, 21, 801)
    Δz = Z / (n - 1)
    x = zeros(T, n)
    λ = zeros(T, n)
    for k = 1:n
        zk = (k - 1) * Δz
        j = clamp(searchsortedlast(zb, zk), 1, length(xb) - 1)
        b = slope(j)
        x[k] = abs(b) < sqrt(eps(T)) ? xb[j] + λb[j] * (zk - zb[j]) : xb[j] + λb[j] * expm1(b * (zk - zb[j])) / b
        λ[k] = λb[j] + b * (x[k] - xb[j])
    end
    return exp.(x), overlap .* λ .* Δz
end

#roughness matrix R_kl = ∫ψ_k''ψ_l'' dx for Gaussian kernels of means m and standard deviations λ in log-space:
#the fourth derivative of the Gaussian density of variance λ_k²+λ_l² at m_k-m_l.
function lognormalMixtureRoughness(m::AbstractVector{T}, λ::AbstractVector{T}) where {T}
    n = length(m)
    R = zeros(T, n, n)
    for k = 1:n, l = 1:n
        v = λ[k]^2 + λ[l]^2
        d = m[k] - m[l]
        R[k, l] = normpdf(d / sqrt(v)) / sqrt(v) * (d^4 - 6v * d^2 + 3v^2) / v^4
    end
    return R
end

function calibrateLognormalMixtureFixedKernels(forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}, F::AbstractVector{T}, λ::AbstractVector{T}; smoothing=0.0, solver="GI") where {T}
    n = length(strikes)
    m = length(F)
    B = zeros(T, (n, m))
    v = zeros(T, n)
    for i = 1:n
        isCall = strikes[i] >= forward
        for j = 1:m
            B[i, j] = weights[i] * blackScholesFormula(isCall, strikes[i], F[j], λ[j]^2, one(T), one(T))
        end
        v[i] = weights[i] * (isCall ? callPrices[i] : callPrices[i] - (forward - strikes[i]))
    end
    P = B' * B
    q = -B' * v
    if smoothing > 0
        R = lognormalMixtureRoughness(log.(F) .- λ .^ 2 ./ 2, λ)
        #relative to the typical (median) diagonals: the far out-of-the-money columns with huge inverse vega weights must not drive the scale
        P += (smoothing * median(diag(P)) / median(diag(R))) .* R
    end
    #the inverse vega weights make P badly scaled (diagonal spanning 1e14): Jacobi scaling, solve for β = Dα with D = diag(√P)
    #so that the scaled matrix has a unit diagonal. β ≥ 0 is equivalent to α ≥ 0.
    D = sqrt.(max.(diag(P), floatmin(T)))
    P = (P + P') ./ (2 .* D .* D')
    q = q ./ D
    #unit mass and martingale constraints, the latter relative to the forward
    Aeq = vcat(ones(T, 1, m), F' ./ forward) ./ D'
    β = if solver == "COSMO"
        settings = COSMO.Settings(verbose=false, eps_abs=1e-12, eps_rel=1e-12, max_iter=100000)
        model = COSMO.Model()
        constraints = [COSMO.Constraint(sparse(Aeq), [-one(T), -one(T)], COSMO.ZeroSet), COSMO.Constraint(sparse(one(T) * I, m, m), zeros(T, m), COSMO.Nonnegatives)]
        assemble!(model, P, q, constraints, settings=settings)
        res = COSMO.optimize!(model)
        res.status == :Solved || @warn "COSMO did not converge" res.status res.iter
        res.x
    else
        #Goldfarb-Idnani requires a positive definite matrix: tiny ridge relative to the unit diagonal
        P += 1e-10 * I
        amat, aind = convertSparse(sparse(hcat(Aeq', Matrix{T}(I, m, m))))
        sol, _ = solveQPcompact(P, -q, amat, aind, vcat(one(T), one(T), zeros(T, m)), meq=2, factorized=false)
        sol
    end
    α = β ./ D
    α = max.(α, zero(T))
    α ./= sum(α)
    r = B * α - v
    return LognormalKernel(F, λ, α), r' * r
end
