using LinearAlgebra
using LeastSquaresOptim
#using GaussNewton
import AQFED.Black: blackScholesFormula, impliedVolatilitySRHalley, Householder
import AQFED.Math: normpdf, normcdf, ClosedTransformation, IdentityTransformation, ExpMinTransformation, MQMinTransformation,    inv, toPositiveHypersphere!, fromPositiveHypersphere!
export LognormalKernel, priceEuropean, density, cumulativeDensity, calibrateLognormalMixture
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