#Andreasen-Huge interpolation
import PPInterpolation: evaluatePiece, CubicSplineNatural, QuadraticLagrangePP

struct DiscreteLVG{TV,T}
    lvg::QuadraticLVG{TV,T}
    z::Vector{T} #discretization z[1] = lvg.x[1], z[end]=lvg.x[end]
    dt::T
    pp::QuadraticLagrangePP{T,TV}
end
function DiscreteLVG(lvg::QuadraticLVG{TV,T},n::Int) where {TV,T}
    z = collect(range(lvg.x[1],stop=lvg.x[end],length=n))
    pp = computeDiscretePrices(lvg,z,lvg.tte )
    return DiscreteLVG(lvg, z, lvg.tte, pp)
end

function computeDiscretePrices(lvg::QuadraticLVG{TV,T},z,dt) where {TV,T}
    tri = Tridiagonal(zeros(TV, length(z)-1),zeros(TV,length(z)),zeros(TV,length(z)-1))
    #tri = model.tri
    a = lvg.a
    b = lvg.b
    c = lvg.c
    n = length(tri.d)
    dz = z[2]-z[1]
    tri.d[1] = one(T)
    tri.d[n] = one(T)
    for i=2:n-1
        lvIndex = max(searchsortedlast(lvg.x, z[i]), 1)
        lv = (a[lvIndex]*z[i]+ b[lvIndex])*z[i] + c[lvIndex]
        factor = dt/lvg.tte*(lv/dz)^2
        tri.d[i] = one(T)+2factor
        tri.dl[i-1]= -factor
        tri.du[i] = -factor
    end
    forward = lvg.forward
    rhs = @. max(forward-z,zero(T))
    prices = tri\rhs
    # pp = CubicSplineNatural(z,prices) #z is uniform we could have a faster impl.
    pp = QuadraticLagrangePP(z,prices)
    return pp
end


struct DiscreteLogLVG{TV,T,TI}
    lvg::QuadraticLVG{TV,T}
    z::Vector{T} #discretization z[1] = lvg.x[1], z[end]=lvg.x[end]
    dt::T
    #pp::QuadraticLagrangePP{T,TV}
    pp::TI
end
Base.broadcastable(p::DiscreteLogLVG) = Ref(p)

function DiscreteLogLVG(lvg::QuadraticLVG{TV,T},n::Int) where {TV,T}
    z = exp.(collect(range(log(lvg.x[1]),stop=log(lvg.x[end]),length=n)))
    xIndices = [max(searchsortedlast(lvg.x, zi), 1) for zi in z]
    pp = computeDiscreteLogPrices(lvg,xIndices, z,lvg.tte )
    return DiscreteLogLVG(lvg, z, lvg.tte, pp)
end
function computeDiscreteLogPrices(lvg::QuadraticLVG{TV,T},xIndices,z,dt;  tri = Tridiagonal(zeros(TV, length(z)-1),zeros(TV,length(z)),zeros(TV,length(z)-1)), rhs = @. max(lvg.forward-z,zero(T))) where {TV,T}
   
    #tri = model.tri
    a = lvg.a
    b = lvg.b
    c = lvg.c
    n = length(tri.d)
    dz = log(z[2]/z[1])
    tri.d[1] = one(T)
    tri.d[n] = one(T)
    for i=2:n-1
        lvIndex = xIndices[i]
        lv = (a[lvIndex]*z[i]+ b[lvIndex])*z[i] + c[lvIndex]
        factor = dt/lvg.tte*(lv/(z[i]*dz))^2
        tri.d[i] = one(T)+2factor
        tri.dl[i-1]= -factor*(one(T)+dz/2)
        tri.du[i] = -factor*(one(T)-dz/2)
    end
    prices = tri\rhs
    pp = CubicSplineNatural(z,prices)
    # pp = QuadraticLagrangePP(z,prices)
     return pp
end

function priceEuropean(model::DiscreteLogLVG{TV,T,TI}, isCall::Bool, strike, t=model.lvg.tte) where {TV,T,TI}
    forward = model.lvg.forward
    x = model.z
    if strike <= x[1] || strike >= x[end]
        return priceEuropeanPiece(0, model, isCall, strike, t)
    end
    i = max(searchsortedlast(x, strike), 1)
    return priceEuropeanPiece(i, model, isCall, strike, t)
end

function priceEuropeanPiece(i::Int, model::DiscreteLogLVG{TV,T,TI}, isCall::Bool, strike, t=model.lvg.tte) where {TV,T,TI}
    forward = model.lvg.forward
    if i <= 0 || i > length(model.z)
        if isCall 
            return max(forward-strike,zero(T))
        else
            return max(strike-forward,zero(T))
        end
    end
    callPrice = evaluatePiece(model.pp,i,(strike))
    if !isCall
        return callPrice - (forward-strike)
    end
    return callPrice 
end

#same calibration as QuadraticLVG, only difference is pricceEuropean calls. Also cache index of forward in z
function calibrateDiscreteLogLVG(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; discreteSize=101, U::T=6 * forward, L::T=forward / 6, useVol=false, model::LVGKind=LinearBachelier(), penalty=zero(T), nRefine=3, location="Mid-XX", isC3=true, minCount=1, size=0, guess="Constant", optimizer="LM", previousLVG=Nothing) where {T}

    #if the forward is not part of the strikes, add the forward and interpolate prices with quadratic. Eventually set corresponding weight to 0
    if typeof(previousLVG) == QuadraticLVG
        L = previousLVG.x[1]
        U = previousLVG.x[end]
    else
        if U <= strikes[end]
            U = strikes[end] + 3 * (strikes[end] - forward)
            #U = strikes[end]*1.1
        end
        if L >= strikes[1]
            L = strikes[1] / 2
        end
    end
    s = searchsortedfirst(strikes, forward) #index of forward
    origStrikes = strikes
    useForwardInStrikes = false
    local priceAtm
    if strikes[s] != forward
        qStrikes = strikes[s-1:s+1]
        qPrices = callPrices[s-1:s+1]
        qvols = @. impliedVolatilitySRHalley(true, qPrices, forward, qStrikes, tte, 1.0, 0e-14, 64, Householder())
        lagrange = QuadraticLagrangePP(qStrikes, qvols, knotStyle=LEFT_KNOT)
        fVol = lagrange(forward)
        fPrice = blackScholesFormula(true, forward, forward, fVol^2 * tte, 1.0, 1.0)
        priceAtm = fPrice
        if useForwardInStrikes
            strikes = vcat(strikes[1:s-1], forward, strikes[s:end])
            callPrices = vcat(callPrices[1:s-1], fPrice, callPrices[s:end])
            weights = vcat(weights[1:s-1], weights[s], weights[s:end])
        end
    else
        priceAtm = callPrices[s]
    end
    local x
    if typeof(previousLVG) == QuadraticLVG
        x = previousLVG.x
        s = searchsortedfirst(x, forward) - 1
    else
        x = makeKnots(tte, forward, strikes, size, location=location, minCount=minCount, L=L, U=U)
        s = searchsortedfirst(x, forward)
        if !useForwardInStrikes && x[s] != forward
            x = sort(vcat(x, forward))
        end
        s = searchsortedfirst(x, forward) - 1
        adjustKnots!(x, s, priceAtm)
    end
    xBasis = makeBasis(model, x, forward, isC3)
    m = length(x) - 1
    #initial guess
    vols = zeros(T, length(strikes))
    if useVol || guess != "Constant"
        for i = eachindex(vols)
            vols[i] = impliedVolatilitySRHalley(true, callPrices[i], forward, strikes[i], tte, 1.0, 0e-14, 64, Householder())
        end
    end
    z = exp.(collect(range(log(L),stop=log(U),length=discreteSize)))
    lvIndices = [max(searchsortedlast(x, zi), 1) for zi in z]

    iter = 0
    basis = BSplineBasis(3, xBasis)
    #println("xBasis ", xBasis, " knots ", knots, "len knoths=", length(knots))
    #strikeIndices = indices of strike in x. (PP version) x will contain forward.
    strikeIndices = max.(searchsortedlast.(Ref(z), strikes), 1)  # x[i]<=z<=x[i+1]
    l = numberOfFreeParameters(model, length(x), length(callPrices), isC3)
    atmVol = impliedVolatilitySRHalley(true, priceAtm, forward, forward, tte, 1.0, 0e-14, 64, Householder())
    a0 = atmVol * sqrt(tte / 2)
    local minValueV
    local maxValueV
    if typeof(previousLVG) == QuadraticLVG
        aMinV = toFreeParameters(previousLVG.λ, model, m, l, s, forward, isC3=isC3)
        minValueV = max.(aMinV, a0 / 100)
        maxValueV = fill(a0 * 8000, l)
    else
        minValueV = fill(a0 / 10, l)
        maxValueV = fill(a0 * 800, l)
    end

    a0v =
        if guess == "Constant"
            fill(a0, l)
        else
            xIndices = unique([min(searchsortedfirst(strikes, xi), length(strikes)) for xi in x])
            varSpline = CubicSplineNatural(strikes[xIndices], vols[xIndices] .^ 2)
            epsDens = 1e-3
            #TODO compute xmid. 
            xPrices = [blackScholesFormula(true, strike, forward, varSpline(strike) * tte, 1.0, 1.0) for strike in x]
            xPricesUp = [blackScholesFormula(true, strike + epsDens, forward, varSpline(strike + epsDens) * tte, 1.0, 1.0) for strike in x]
            xPricesDown = [blackScholesFormula(true, strike - epsDens, forward, varSpline(strike - epsDens) * tte, 1.0, 1.0) for strike in x]
            xIntrinsic = [max(forward - strike, 0) for strike in x]
            #we divide by forward since we multiply by it later (black like scaling)
            @.(sqrt(min((a0 * 10)^2, max((a0 / 5)^2, (xPrices - xIntrinsic) / (forward^2 * (xPricesUp - 2xPrices + xPricesDown) / epsDens^2)))))[1:end]
        end
    transformV = [makeTransform(minValue, maxValue, optimizer) for (minValue, maxValue) = zip(minValueV, maxValueV)]


    #println(s, " forward found ", strikes[s], " min ", minValue, " max ", maxValue, " a0 ", a0v, " atm ", priceAtm, "m+1-l-2=", (m - l - 1))
    sumw = if useVol
        length(weights) / m
    else
        sum(weights) / m
    end
    discreteForwardIndex = searchsortedlast(z,forward)
    logForward = log(forward)

    ### TODO don't set sigma0=sigma1: one more variable to minimize on. don't set lambda2 = lambda3.
    #TODO split in separate types Linear,... and operate on those toQuadForm. instead of strings.
    function obj!(fvec::Z, coeff::AbstractArray{W})::Z where {Z,W}
        n = length(strikes)
        σ = zeros(W, l)
        for i = 1:l
            σ[i] = transformV[i](coeff[i])
        end
        γ = zeros(W, m)
        β = zeros(W, m)
        α = zeros(W, m)
        θ = zeros(W, 2m)
        λ = makeLambda(model,σ, m, l, s, forward, isC3=isC3)
        if isC3
            updateLambda!(λ, model, x, s + 1, priceAtm, basis=basis)
        end
        toQuadForm!(model, m, l, s, x, α, β, γ, λ, basis=basis)
        lvg = QuadraticLVG(x, α, β, γ, θ, λ, tte, forward)
        pp = computeDiscreteLogPrices(lvg,lvIndices,z,tte)        
        if isC3
            for _ = 1:nRefine
                modelPriceAtm = evaluatePiece(pp,discreteForwardIndex,forward)
                updateLambda!(λ, model, x, s + 1, modelPriceAtm, basis=basis)
                toQuadForm!(model, m, l, s, x, α, β, γ, λ, basis=basis)
                lvg = QuadraticLVG(x, α, β, γ, θ, λ, tte, forward)
                pp = computeDiscreteLogPrices(lvg,lvIndices,z,tte)               
            end
        end
        # println("θ ", θ)
        if useVol
            for i = 1:n
                isCall = strikes[i] >= forward
                strikeIndex = strikeIndices[i]
                mPrice = 
                    priceEuropeanPiece(strikeIndex, DiscreteLogLVG(lvg, z, lvg.tte, pp), isCall, strikes[i])
                fvec[i] = weights[i] * (impliedVolatilitySRHalley(isCall, max(mPrice, 1e-64), forward, strikes[i], tte, 1.0, 0e-14, 64, Householder()) - vols[i])
                # fvec[i] =  impliedVolatilityLiSOR(isCall, max(mPrice,1e-64), forward, strikes[i], tte, 1.0, 0.0, 0e-14, 64, SORTS()) - vols[i]
            end
        else
            for i = 1:n
                isCall = strikes[i] >= forward
                strikeIndex = strikeIndices[i]
                mPrice = 
                    priceEuropeanPiece(strikeIndex, DiscreteLogLVG(lvg, z, lvg.tte, pp), isCall, strikes[i])
                if (strikes[i] < forward)
                    mPrice += forward - strikes[i]
                end
                fvec[i] = weights[i] * (mPrice - callPrices[i]) #FIXME what if forward not on list?
                if isnan(fvec[i]) || isinf(fvec[i])
                    println("σ ", σ, "\n θ ", θ)
                    println("alpha ", α, "\n beta ", β, "\n gamma ", γ)
                    throw(DomainError(fvec, " fvec must be a number"))
                end
            end
        end
        if penalty > 0
            #perform penalty on the knots x, not on the strikes.
            # v = zeros(W, m-1)
            # for i = 1:m-1
            #     density = abs(θ[2i+2] / (α[i+1] * x[i]^2 + β[i+1] * x[i] + γ[i+1])^2)
            #     v[i] = log(density)
            # end
            for i = 2:l-1
                # fvec[i+n-1] = sumw * penalty * ((v[i+1] - v[i]) / (x[i+1] - x[i]) - (v[i] - v[i-1]) / (x[i] - x[i-1]))
                fvec[i+n-1] = sumw * penalty * (((σ[i+1] - σ[i]) / (x[i+1] - x[i])))
            end
        end
        #     ip = hermiteIntegral(derivative(p, 2)^2)
        #     pvalue = penalty * ip
        #     fvec[n+1] = pvalue
        # end
        # println(iter," ",fvec)
        iter += 1
        fvec
    end
    σ = zeros(T, l)
    Random.seed!(1)
    epsvec = rand(T, l) * sqrt(eps(T))
    for i = eachindex(σ)
        σ[i] = inv(transformV[i], a0v[i] * (1 + epsvec[i]))
    end
    x0 = σ[1:l]
    outlen = length(callPrices)
    if penalty > 0
        outlen += m - 1
    end
    fvec = zeros(Float64, outlen)

    x0 = if optimizer == "LM-Curve"
        fit = LeastSquaresOptim.optimize!(
            LeastSquaresProblem(x=x0, (f!)=obj!, autodiff=:forward, #:forward is 4x faster than :central
                output_length=outlen),
            LevenbergMarquardt();
            lower=minValueV, upper=maxValueV,
            iterations=1000        )
        #    println(iter, " fit ", fit)
        fit.minimizer
        # elseif optimizer == "LM-MQ" || optimizer == "LM-SIN" || optimizer == "LM-LOG" || optimizer == "LM-ALG"

        #     fit = LeastSquaresOptim.optimize!(
        #         LeastSquaresProblem(x=x0, (f!)=obj!, autodiff=:forward, #:forward is 4x faster than :central
        #             output_length=outlen),
        #         LevenbergMarquardt();
        #         iterations=1000
        #     )
        # #    println(iter, " fit ", fit)
        #     fit.minimizer
    elseif optimizer == "GN-MQ" || optimizer == "GN-SIN" || optimizer == "GN" || optimizer == "GN-LOG" || optimizer == "GN-ALG" || optimizer == "GN-ATAN" || optimizer == "GN-TANH"
        fit = GaussNewton.optimize!(obj!, x0, fvec, autodiff=:single, iscale=1, abstol=length(fvec) * eps())
        # println(iter, " fit ", fit)
        x0
    elseif optimizer == "LM"
        obj1 = OnceDifferentiable(obj!, x0, copy(fvec); autodiff=:forward, inplace=true)
        fit = LsqFit.levenberg_marquardt(obj1, x0,
            lower=minValueV, upper=maxValueV)
        
        # println(iter, " fit ", fit)
        fit.minimizer
    elseif optimizer == "LM-MQ" || optimizer == "LM-SIN" || optimizer == "LM-LOG" || optimizer == "LM-ALG" || optimizer == "LM-TANH" || optimizer == "LM-ATAN"
        obj1 = OnceDifferentiable(obj!, x0, copy(fvec); autodiff=:forward, inplace=true)
        fit = LsqFit.levenberg_marquardt(obj1, x0)
        # println(iter, " fit ", fit)
        fit.minimizer

    end
    #  
    obj!(fvec, x0)
    for i = 1:l
        σ[i] = transformV[i](x0[i])
    end
    #println("σ=", σ)
    γ = zeros(T, m)
    β = zeros(T, m)
    α = zeros(T, m)
    θ = zeros(T, 2m)
    λ = makeLambda(model,σ, m, l, s, forward, isC3=isC3)
    if isC3
        updateLambda!(λ, model, x, s + 1, priceAtm, basis=basis)
    end
    toQuadForm!(model, m, l, s, x, α, β, γ, λ, basis=basis)
    lvg = QuadraticLVG(x, α, β, γ, θ, λ, tte, forward)
    pp = computeDiscreteLogPrices(lvg,lvIndices,z,tte)        
    if isC3
        for _ = 1:nRefine
            modelPriceAtm = evaluatePiece(pp,discreteForwardIndex,forward)
            # println("atmPRices ",priceAtm," ",modelPriceAtm," ", pp(logForward))
            updateLambda!(λ, model, x, s + 1, modelPriceAtm, basis=basis)
            toQuadForm!(model, m, l, s, x, α, β, γ, λ, basis=basis)
            lvg = QuadraticLVG(x, α, β, γ, θ, λ, tte, forward)
            pp = computeDiscreteLogPrices(lvg,lvIndices,z,tte)               
        end
    end
    return DiscreteLogLVG(lvg, z, lvg.tte, pp)
end
