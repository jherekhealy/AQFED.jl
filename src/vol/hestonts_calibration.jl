using CharFuncPricing, Random123, LinearAlgebra, PPInterpolation, AQFED.TermStructure
using AQFED.Black, AQFED.GlobalOptimization, GaussNewton
import AQFED.Math:
    MQMinTransformation,
    LogisticTransformation,
    inv
using Optim

function priceVarianceSwap(p::HestonTSParams, tte::T) where {T}
    θ = p.θ
    κ = p.κ
    tv = zero(T)
    vPrev = p.v0
    t0 = p.startTime[1]
    indexTte = searchsortedfirst(p.startTime, tte)
    endTimes = vcat(p.startTime[2:indexTte-1], tte)
    #println(endTimes)
    for (i, t1) = enumerate(endTimes)
        dti = t1 - t0
        eFactor = (1 - exp(-κ[i] * dti)) / κ[i]
        tv += θ[i] * (dti - eFactor) + vPrev * eFactor
        vPrev = (vPrev - θ[i]) * exp(-κ[i] * dti) + θ[i]
        t0 = t1

    end
    #tv,_ = quadgk(t -> xsi(t) ,0.0,tte)
    tv / tte
end


function priceVarianceSwap(p::HaganHestonPiecewiseParams, tte::T) where {T}

    t0 = p.startTime[1]
    indexTte = searchsortedfirst(p.startTime, tte)
    endTimes = vcat(p.startTime[2:indexTte-1], tte)
    #println(endTimes)
    tv = zero(T)
    for (i, t1) = enumerate(endTimes)
        dti = t1 - t0
        tv += p.leverage[i]^2 * dti
        t0 = t1
    end
    #tv,_ = quadgk(t -> xsi(t) ,0.0,tte)
    tv / tte
end

function atmVolFromPrices(forward, t, strikes::AbstractVector{T}, prices::AbstractVector{T}, isCall::AbstractVector{Bool}) where {T}
    m = length(strikes)
    iAtm = findfirst(x -> x > forward, strikes)
    if iAtm == nothing
        iAtm = m - 1
    elseif iAtm == 1
        iAtm = 2
    else
        iAtm -= 1
    end
    # println(iAtm, " ", forward , " ",strikes)
    volsShort = map(index -> Black.impliedVolatility(isCall[index], prices[index], forward, strikes[index], t, 1.0), iAtm-1:iAtm+1)
    volAtm = PPInterpolation.evaluate(2, strikes[iAtm-1:iAtm+1], volsShort, forward)
    return volAtm
end



function makeSwapCurveConstant(atmVol)
    return t -> atmVol^2
end

function makeSwapCurveReplication(ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T}, vols::AbstractMatrix{T}
) where {T}
    swapPrices = map(i -> priceVarianceSwap(FukasawaVarianceSwapReplication(true), ts[i], log.(strikes[i, :] ./ forwards[i]), vols[i, :] .^ 2, 1.0) / 10000, 1:length(ts))
    spline = makeCubicPP(ts, swapPrices, PPInterpolation.SECOND_DERIVATIVE, 0.0, PPInterpolation.SECOND_DERIVATIVE, 0.0, HuynRational())
    return spline
end

priceEuropean(p::Union{CharFuncPricing.JoshiYangCharFuncPricer,CharFuncPricing.CosCharFuncPricer,CharFuncPricing.FlinnCharFuncPricer,CharFuncPricing.AdaptiveFlinnCharFuncPricer,CharFuncPricing.SwiftCharFuncPricer}, isCall::Bool, strike, forward, tte, df) = CharFuncPricing.priceEuropean(p, isCall, strike, forward, tte, df)

function calibrateHestonHaganFromPricesParam(
    ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    prices::AbstractMatrix{T},
    isCall::AbstractMatrix{Bool},
    weights::AbstractMatrix{T};
    κ=1.0,
    lower=[1e-4, -0.99, 1e-2], upper=[4.0, 0.5, 4.0], #leverage, rho, sigma
    isRelative=false,
    method="Joshi-Yang",
    isDebug=false, useGN=true,
    minimizer="DE", numberOfKnots=[4, 3, 4],
    swapCurve=makeSwapCurveConstant(atmVolFromPrices(forwards[1], ts[1], strikes[1, :], prices[1, :], isCall[1, :]))) where {T}


    # uPrices = zeros(T, length(ts),length(strikes))))
    # uWeights = zeros(T, length(ts),length(strikes))
    # for i  = eachindex(forwards)
    # 	@. uPrices[i,:] = prices[i,:] ./ forwards[i]
    # 	@. uWeights[i,:] = weights[i,:] .* forwards[i]
    # end
    n2 = size(prices, 2)
    uPrices = prices
    uWeights = weights
    # lower =  [range[1].θ, range[1].ρ, range[1].σ]
    # upper =  [range[2].θ, range[2].ρ, range[2].σ]
    transforms = map((low, high) -> LogisticTransformation(low, high), lower, upper)

    startTime = vcat(zero(T), ts[1:end-1])
    θAll = zeros(length(startTime))
    ρAll = zeros(length(startTime))
    σAll = zeros(length(startTime))
    rmse = zeros(length(startTime))
    θknots =
        if (numberOfKnots[1] < length(startTime))
            startTime[round.(Int, LinRange(1, length(startTime), numberOfKnots[1]))]
        else
            startTime
        end
    ρknots = if numberOfKnots[2] < length(startTime)
        startTime[round.(Int, LinRange(1, length(startTime), numberOfKnots[2]))]
    else
        startTime
    end
    σknots = if numberOfKnots[3] < length(startTime)
        startTime[round.(Int, LinRange(1, length(startTime), numberOfKnots[3]))]
    else
        startTime
    end
    function interpolateKnots(out, knots, values)
        #interpolate knots to the startTime.
        if length(knots) < length(startTime)
            spline = makeCubicPP(knots, values, PPInterpolation.SECOND_DERIVATIVE, 0.0, PPInterpolation.SECOND_DERIVATIVE, 0.0, HuynRational())
            for (i, t) = enumerate(startTime)
                ti = (t + ts[i]) / 2
                if ti > startTime[end]
                    ti = startTime[end] #spline not defined
                end
                out[i] = spline(ti)
            end
        else
            out .= values
        end
    end
    #println(θknots, " ",ρknots, " ",σknots)
    function objectiveT(F, xr, hParams::HaganHestonPiecewiseParams)
        #xr[1] = theta, xr[2] = rho, xr[3] = sigma.
        #x = map( (transformi, xri) -> transformi(xri), transforms, xr)
        θp = xr[1:length(θknots)]
        ρp = xr[length(θknots)+1:length(θknots)+length(ρknots)]
        σp = xr[length(θknots)+length(ρknots)+1:length(θknots)+length(ρknots)+length(σknots)]
        #println(θp, " ",ρp, " ",σp)
        θ = hParams.leverage
        interpolateKnots(θ, θknots, θp)
        ρ = hParams.ρ
        interpolateKnots(ρ, ρknots, ρp)
        σ = hParams.σ
        interpolateKnots(σ, σknots, σp)
        lastTime = ts[end]
        hestonTSParams, lastτ = makeHestonTSParams(hParams, tte=lastTime)
        cf = DefaultCharFunc(hestonTSParams)
        for i = eachindex(ts)
            t = if i < length(ts)
                hestonTSParams.startTime[i+1]
            else
                lastτ
            end
            # pricer =JoshiYangCharFuncPricer(cf, t, n=64)
            pricer = if method == "Andersen-Lake"
                CharFuncPricing.ALCharFuncPricer(cf, n=64)
            elseif method == "Joshi-Yang"
                JoshiYangCharFuncPricer(cf, t, n=64)
            elseif method == "Cos-128"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, 128, 12)
            elseif method == "Cos"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, 256, 16)
            elseif method == "Cos-Junike"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, tol=1e-4, maxM=4096)
            elseif method == "Flinn"
                CharFuncPricing.FlinnCharFuncPricer(cf, t, tTol=1e-4, qTol=1e-8)
            elseif method == "Flinn-Transformed"
                #CharFuncPricing.makeCVCharFunc(cf,t,CharFuncPricing.InitialControlVariance())
                CharFuncPricing.AdaptiveFlinnCharFuncPricer(cf, t)
            elseif method == "Swift"
                m, _ = CharFuncPricing.findSwiftScaling(cf, t)
                CharFuncPricing.makeSwiftCharFuncPricer(cf, t, m, 3)
            elseif method == "BGM"
                BGMApprox(hestonTSParams)
            elseif method == "SABR"
                t = ts[i]
                HaganHestonTSApprox(hParams, t)
            end
            for (j, strike) ∈ enumerate(strikes[i, :])
                F[(i-1)*n2+j] = if iszero(uWeights[i, j])
                    uWeights[i, j]
                else
                    mPrice = priceEuropean(pricer, isCall[i, j], strike / forwards[i], 1.0, t, 1.0) * forwards[i]
                    if isRelative
                        uWeights[i, j] * (mPrice / uPrices[i, j] - 1)
                    else
                        uWeights[i, j] * (mPrice - uPrices[i, j])
                    end
                end
            end
        end
        F
    end

    hParams = HaganHestonPiecewiseParams(θAll, κ, ρAll, σAll, startTime)

    totalSize = n2 * length(ts)
    out = zeros(totalSize)
    function objective1(x)
        objectiveT(out, x, hParams)
        norm(out) / norm(uWeights)
    end
    #int_0^T lev^2 du = swapCurve(T)*T
    lev0 = sqrt.(computeForwardVariance(swapCurve, θknots))
    x0 = vcat(lev0, ones(T, length(ρknots)) .* (-0.5), ones(T, length(σknots)) .* (0.5))
    if isDebug
        println("init guess ", x0)
    end

    rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
    lowern = vcat(ones(T, length(θknots)) .* lower[1], ones(T, length(ρknots)) .* lower[2], ones(T, length(σknots)) .* lower[3])
    uppern = vcat(ones(T, length(θknots)) .* upper[1], ones(T, length(ρknots)) .* upper[2], ones(T, length(σknots)) .* upper[3])
    problem = GlobalOptimization.Problem(length(x0), objective1, lowern, uppern)
    if minimizer == "DE"
        optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(50),
            problem,
            rng,
            GlobalOptimization.Best1Bin(0.9, 0.5),
            #GlobalOptimization.Best2Bin(0.7, 0.9),
            x0=x0)
        result = GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(500, 100, 1e-5, 1e-4))
        if isDebug
            println("Result DE ", result, " ", GlobalOptimization.minimizer(optim))
        end
        x0 = GlobalOptimization.minimizer(optim) #calibrated params.
    elseif minimizer == "SA"
        optim = GlobalOptimization.SimulatedAnnealing(problem, rng)
        result = GlobalOptimization.optimize(optim, x0, maxevl=length(x0) * 30 * 10)
        if isDebug
            println("Result SA ", result, " ", GlobalOptimization.minimizer(optim))
        end
        x0 = GlobalOptimization.minimizer(optim) #calibrated params.
    else
        #predefined guess
    end



    function objectiveTConstrained(F, xr)
        x = copy(xr)
        x[1:length(θknots)] = map(xri -> transforms[1](xri), xr[1:length(θknots)])
        x[length(θknots)+1:length(θknots)+length(ρknots)] = map(xri -> transforms[2](xri), xr[length(θknots)+1:length(θknots)+length(ρknots)])
        x[length(θknots)+length(ρknots)+1:length(θknots)+length(ρknots)+length(σknots)] = map(xri -> transforms[3](xri), xr[length(θknots)+length(ρknots)+1:length(θknots)+length(ρknots)+length(σknots)])
        #println("objectiveX ",x)
        return objectiveT(F, x, hParams)
    end
    function objective1Constrained(xr)
        objectiveTConstrained(out, xr)
        norm(out) / norm(uWeights)
    end
    x = x0
    xr = copy(x)
    xr[1:length(θknots)] = map(xri -> inv(transforms[1], xri), x[1:length(θknots)])
    xr[length(θknots)+1:length(θknots)+length(ρknots)] = map(xri -> inv(transforms[2], xri), x[length(θknots)+1:length(θknots)+length(ρknots)])
    xr[length(θknots)+length(ρknots)+1:length(θknots)+length(ρknots)+length(σknots)] = map(xri -> inv(transforms[3], xri), x[length(θknots)+length(ρknots)+1:length(θknots)+length(ρknots)+length(σknots)])
    if useGN
        gnRmse = GaussNewton.optimize!(objectiveTConstrained, xr, out, autodiff=:single)
        if isDebug
            println("Result GN ", gnRmse)
        end
    else
        result = Optim.optimize(objective1Constrained, xr, BFGS(), Optim.Options(f_abstol=1e-6))
        if isDebug
            println("Result Optim ", result, " ", Optim.minimizer(result))
        end
        gnRmse = Optim.minimum(result)
        xr = Optim.minimizer(result)
    end
    objectiveTConstrained(out, xr)
    return hParams, gnRmse
end



function computeForwardVariance(swapCurve, ts::Array{T}) where {T}
    #ts does not include zero.
    lev0 = zeros(T, length(ts))
    lev0[1] = swapCurve(ts[1])
    for i = 2:length(lev0)
        lev0[i] = (max(1e-4, swapCurve(ts[i]) * ts[i] - swapCurve(ts[i-1]) * ts[i-1])) / (ts[i] - ts[i-1])
    end
    lev0
end
function calibrateHestonHaganFromPrices(
    ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    prices::AbstractMatrix{T},
    isCall::AbstractMatrix{Bool},
    weights::AbstractMatrix{T};
    κ=1.0,
    lower=[1e-4, -0.99, 1e-2], upper=[4.0, 0.5, 4.0], #leverage, rho, sigma
    isRelative=false,
    method="Joshi-Yang",
    minimizer="DE",
    isGlobal=false,
    isDebug=false,
    penalty=0.0,
    swapCurve=makeSwapCurveConstant(atmVolFromPrices(forwards[1], ts[1], strikes[1, :], prices[1, :], isCall[1, :]))
) where {T}
    # uPrices = zeros(T, length(ts),length(strikes))))
    # uWeights = zeros(T, length(ts),length(strikes))
    # for i  = eachindex(forwards)
    # 	@. uPrices[i,:] = prices[i,:] ./ forwards[i]
    # 	@. uWeights[i,:] = weights[i,:] .* forwards[i]
    # end
    n2 = size(prices, 2)
    uPrices = prices
    uWeights = weights
    volAtm = atmVolFromPrices(forwards[1], ts[1], strikes[1, :], prices[1, :], isCall[1, :])
    v0 = volAtm^2
    # lower =  [range[1].θ, range[1].ρ, range[1].σ]
    # upper =  [range[2].θ, range[2].ρ, range[2].σ]
    transforms = map((low, high) -> LogisticTransformation(low, high), lower, upper)

    startTime = vcat(zero(T), ts[1:end-1])
    θAll = zeros(length(startTime))
    ρAll = zeros(length(startTime))
    σAll = zeros(length(startTime))
    rmse = zeros(length(startTime))

    function objectiveTi(F, xr, hParams::HaganHestonPiecewiseParams, i)
        #xr[1] = theta, xr[2] = rho, xr[3] = sigma.
        #x = map( (transformi, xri) -> transformi(xri), transforms, xr)
        θ = hParams.leverage
        ρ = hParams.ρ
        σ = hParams.σ
        t = ts[i]
        θ[i] = xr[1]
        ρ[i] = xr[2]
        σ[i] = xr[3]
        lastTime = t #ts[end]
        hestonTSParams, lastτ = makeHestonTSParams(hParams, tte=lastTime)

        cf = DefaultCharFunc(hestonTSParams)

        # pricer =JoshiYangCharFuncPricer(cf, t, n=64)
        pricer = if method == "Andersen-Lake"
            CharFuncPricing.ALCharFuncPricer(cf, n=64)
        elseif method == "Joshi-Yang"
            JoshiYangCharFuncPricer(cf, lastτ, n=64)
        elseif method == "Cos-128"
            CharFuncPricing.makeCosCharFuncPricer(cf, lastτ, 128, 12)
        elseif method == "Cos"
            CharFuncPricing.makeCosCharFuncPricer(cf, lastτ, 256, 16)
        elseif method == "Cos-Junike"
            CharFuncPricing.makeCosCharFuncPricer(cf, lastτ, tol=1e-4, maxM=4096)
        elseif method == "Flinn"
            CharFuncPricing.FlinnCharFuncPricer(cf, lastτ, tTol=1e-4, qTol=1e-8)
        elseif method == "Flinn-Transformed"
            #CharFuncPricing.makeCVCharFunc(cf,t,CharFuncPricing.InitialControlVariance())
            CharFuncPricing.AdaptiveFlinnCharFuncPricer(cf, lastτ)
        elseif method == "Swift"
            m, _ = CharFuncPricing.findSwiftScaling(cf, lastτ)
            CharFuncPricing.makeSwiftCharFuncPricer(cf, lastτ, m, 3)
        end
        for (j, strike) ∈ enumerate(strikes[i, :])
            if iszero(uWeights[i, j])
                F[j] = uWeights[i, j]
            else
                mPrice = CharFuncPricing.priceEuropean(pricer, isCall[i, j], strike / forwards[i], 1.0, lastτ, 1.0) * forwards[i]
                F[j] = if isRelative
                    uWeights[i, j] * (mPrice / uPrices[i, j] - 1)
                else
                    uWeights[i, j] * (mPrice - uPrices[i, j])
                end
            end
        end
        if penalty > 0.0
           if i>1
                F[n2+1] = penalty * (σ[i]-σ[i-1])
                F[n2+2] = penalty * (θ[i]-θ[i-1]) #penalty for large parameters.
                F[n2+3] = penalty * (ρ[i]-ρ[i-1]) #penalty for large parameters.
           end
        end
        F
    end

    lev0 = zeros(T, length(θAll))
    lev0[1] = sqrt(swapCurve(ts[1]))
    for i = 2:length(lev0)
        lev0[i] = sqrt((max(1e-4, swapCurve(ts[i]) * ts[i] - swapCurve(ts[i-1]) * ts[i-1])) / (ts[i] - ts[i-1]))
    end


    for (i, t) ∈ enumerate(ts)
        θ = θAll[1:i]
        ρ = ρAll[1:i]
        σ = σAll[1:i]

        hParams = HaganHestonPiecewiseParams(θ, κ, ρ, σ, startTime[1:i])

        objectiveT(F, xr) = objectiveTi(F, xr, hParams, i)

        totalSize = n2
        if penalty > 0.0
            totalSize += 3 #penalty term
        end
        out = zeros(totalSize)
        function objective1(x)
            objectiveT(out, x)
            norm(out) / norm(uWeights)
        end

        function objectiveTiConstrained(F, xr)
            x = map((transformi, xri) -> transformi(xri), transforms, xr)
            return objectiveT(F, x)
        end
        function objective1Constrained(xr)
            objectiveTiConstrained(out, xr)
            norm(out) / norm(uWeights)
        end
        x0 = if i > 0
            [lev0[i], -0.5, 0.5]
        else
            [lev0[i], ρ[i-1], σ[i-1]]
        end
        isGlobalGuess = true
        x = if isGlobalGuess
            result = if minimizer == "BFGS"
                rx0 = map((transformi, xri) -> inv(transformi, xri), transforms, x0)
                result = Optim.optimize(objective1Constrained, rx0, BFGS(), Optim.Options(f_abstol=1e-9))
                if isDebug
                    println("Result Optim ", result, " ", Optim.minimizer(result))
                end
                xr = Optim.minimizer(result)
                [transforms[1](xr[1]), transforms[2](xr[2]), transforms[3](xr[3])]
            elseif minimizer == "NM"
                rx0 = map((transformi, xri) -> inv(transformi, xri), transforms, x0)
                result = Optim.optimize(objective1Constrained, rx0, NelderMead(), Optim.Options(f_abstol=1e-9))
                if isDebug
                    println("Result NM ", result, " ", Optim.minimizer(result))
                end
                xr = Optim.minimizer(result)
                [transforms[1](xr[1]), transforms[2](xr[2]), transforms[3](xr[3])]
            elseif minimizer == "FMB"
                result = Optim.optimize(objective1, lower, upper, x0, Fminbox(GradientDescent()))
                Optim.minimizer(result)
            elseif minimizer == "DE"
                rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
                problem = GlobalOptimization.Problem(length(x0), objective1, lower, upper)
                optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(9),
                    problem,
                    rng, GlobalOptimization.Best2Bin(0.9, 0.5),
                    x0=x0)
                GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(1000, 100, 1e-6, 1e-5))
                if isDebug
                    println("Result DE ", result, " ", GlobalOptimization.minimizer(optim))
                end
                GlobalOptimization.minimizer(optim) #calibrated params.

            elseif minimizer == "SA"#SA works better
                rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
                problem = GlobalOptimization.Problem(length(x0), objective1, lower, upper)
                optim = GlobalOptimization.SimulatedAnnealing(problem, rng)
                GlobalOptimization.optimize(optim, x0)
                if isDebug
                    println("Result SA ", result, " ", GlobalOptimization.minimizer(optim))
                end
                GlobalOptimization.minimizer(optim) #calibrated params.
            else
                x0
            end
        else
            x0
        end

        isLM = minimizer == "DE" || minimizer == "SA" || minimizer == "NM" || minimizer == "User"
        if isLM
            rx = map((transformi, xri) -> inv(transformi, xri), transforms, x)
            gnRmse = GaussNewton.optimize!(objectiveTiConstrained, rx, out, autodiff=:single)
            if isDebug
                println("Result GN ", gnRmse)
            end
            θ[i] = transforms[1](rx[1])
            ρ[i] = transforms[2](rx[2])
            σ[i] = transforms[3](rx[3])
        else
            θ[i] = x[1]
            ρ[i] = x[2]
            σ[i] = x[3]
        end
        rmse[i] = objective1([θ[i], ρ[i], σ[i]])
        θAll[i] = θ[i]
        ρAll[i] = ρ[i]
        σAll[i] = σ[i]
    end
    if isGlobal
        # additional global calib

        hParams = HaganHestonPiecewiseParams(θAll, κ, ρAll, σAll, startTime)

        function objectiveTConstrained(F, xr)
            for i = eachindex(ts)
                xri = xr[3*(i-1)+1:3*(i-1)+3]
                x = map((transformi, xri) -> transformi(xri), transforms, xri)
                θAll[i] = x[1]
                ρAll[i] = x[2]
                σAll[i] = x[3]
            end
            for i = eachindex(ts)
                xri = xr[3*(i-1)+1:3*(i-1)+3]
                x = map((transformi, xri) -> transformi(xri), transforms, xri)
                #println("objectiveTConstrained x=",x," i=",i)
                objectiveTi(@view(F[(i-1)*n2+1:i*n2]), x, hParams, i)
            end
            F
        end
        rx = zeros(3 * length(startTime))
        for i = 1:length(startTime)
            x = [θAll[i], ρAll[i], σAll[i]]
            rx[3*(i-1)+1:3*(i-1)+3] = map((transformi, xri) -> inv(transformi, xri), transforms, x)
        end
        out = zeros(length(ts) * n2)
        gnRmse = GaussNewton.optimize!(objectiveTConstrained, rx, out, autodiff=:single)
        if isDebug
            println("Result GN ", gnRmse)
        end
        for i = 1:length(startTime)
            rxi = rx[3*(i-1)+1:3*(i-1)+3]
            θAll[i] = transforms[1](rxi[1])
            ρAll[i] = transforms[2](rxi[2])
            σAll[i] = transforms[3](rxi[3])
        end
        # for i = 1:length(ts)
        #     rmse[i] = objective1([θ[i], ρ[i], σ[i]])
        # end
    end

    hParams = HaganHestonPiecewiseParams(θAll, κ, ρAll, σAll, startTime)
    return hParams, rmse
end



function calibrateHestonTSFromPricesParam(
    ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    prices::AbstractMatrix{T},
    isCall::AbstractMatrix{Bool},
    weights::AbstractMatrix{T};
    κ=ones(length(ts)) .* 1.0,
    lower=[1e-4, -0.99, 1e-2], upper=[1.0, 0.5, 2.0],
    isRelative=false,
    method="Joshi-Yang",
    minimizer="DE", numberOfKnots=[4, 3, 4]) where {T}
    # uPrices = zeros(T, length(ts),length(strikes))))
    # uWeights = zeros(T, length(ts),length(strikes))
    # for i  = eachindex(forwards)
    # 	@. uPrices[i,:] = prices[i,:] ./ forwards[i]
    # 	@. uWeights[i,:] = weights[i,:] .* forwards[i]
    # end
    uPrices = prices
    uWeights = weights
    volAtm = atmVolFromPrices(forwards[1], ts[1], strikes[1, :], prices[1, :], isCall[1, :])
    v0 = volAtm^2
    # lower =  [range[1].θ, range[1].ρ, range[1].σ]
    # upper =  [range[2].θ, range[2].ρ, range[2].σ]
    transforms = map((low, high) -> LogisticTransformation(low, high), lower, upper)
    n2 = size(prices, 2)
    startTime = vcat(zero(T), ts[1:end-1])
    θAll = zeros(length(startTime))
    ρAll = zeros(length(startTime))
    σAll = zeros(length(startTime))
    rmse = zeros(length(startTime))
    θknots = startTime[round.(Int, LinRange(1, length(startTime), numberOfKnots[1]))]
    ρknots = startTime[round.(Int, LinRange(1, length(startTime), numberOfKnots[2]))]
    σknots = startTime[round.(Int, LinRange(1, length(startTime), numberOfKnots[3]))]
    println(θknots, " ", ρknots, " ", σknots)
    function objectiveT(F, xr, hParams::HestonTSParams)
        #xr[1] = theta, xr[2] = rho, xr[3] = sigma.
        #x = map( (transformi, xri) -> transformi(xri), transforms, xr)
        θp = xr[1:length(θknots)]
        ρp = xr[length(θknots)+1:length(θknots)+length(ρknots)]
        σp = xr[length(θknots)+length(ρknots)+1:length(θknots)+length(ρknots)+length(σknots)]
        #println(θp, " ",ρp, " ",σp)
        splineθ = makeCubicPP(θknots, θp, PPInterpolation.SECOND_DERIVATIVE, 0.0, PPInterpolation.SECOND_DERIVATIVE, 0.0, HuynRational())
        splineρ = makeCubicPP(ρknots, ρp, PPInterpolation.SECOND_DERIVATIVE, 0.0, PPInterpolation.SECOND_DERIVATIVE, 0.0, HuynRational())
        splineσ = makeCubicPP(σknots, σp, PPInterpolation.SECOND_DERIVATIVE, 0.0, PPInterpolation.SECOND_DERIVATIVE, 0.0, HuynRational())


        θ = hParams.θ
        ρ = hParams.ρ
        σ = hParams.σ
        for (i, t) = enumerate(startTime)
            ti = (t + startTime[i+1]) / 2 #same as t
            θ[i] = splineθ(ti)
            ρ[i] = splineρ(ti)
            σ[i] = splineσ(ti)
        end
        cf = DefaultCharFunc(hParams)

        for (i, t) = enumerate(ts)
            # pricer =JoshiYangCharFuncPricer(cf, t, n=64)
            pricer = if method == "Andersen-Lake"
                CharFuncPricing.ALCharFuncPricer(cf, n=64)
            elseif method == "Joshi-Yang"
                JoshiYangCharFuncPricer(cf, t, n=64)
            elseif method == "Cos-128"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, 128, 12)
            elseif method == "Cos"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, 256, 16)
            elseif method == "Cos-Junike"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, tol=1e-4, maxM=4096)
            elseif method == "Flinn"
                CharFuncPricing.FlinnCharFuncPricer(cf, t, tTol=1e-4, qTol=1e-8)
            elseif method == "Flinn-Transformed"
                #CharFuncPricing.makeCVCharFunc(cf,t,CharFuncPricing.InitialControlVariance())
                CharFuncPricing.AdaptiveFlinnCharFuncPricer(cf, t)
            elseif method == "Swift"
                m, _ = CharFuncPricing.findSwiftScaling(cf, t)
                CharFuncPricing.makeSwiftCharFuncPricer(cf, t, m, 3)
            end
            for (j, strike) ∈ enumerate(strikes[i, :])
                F[(i-1)*n2+j] = if iszero(uWeights[i, j])
                    uWeights[i, j]
                else
                    mPrice = CharFuncPricing.priceEuropean(pricer, isCall[i, j], strike / forwards[i], 1.0, t, 1.0) * forwards[i]
                    if isRelative
                        uWeights[i, j] * (mPrice / uPrices[i, j] - 1)
                    else
                        uWeights[i, j] * (mPrice - uPrices[i, j])
                    end
                end
            end
        end
        F
    end

    hParams = HestonTSParams(v0, κ, θAll, ρAll, σAll, startTime)

    totalSize = n2 * length(ts)
    out = zeros(totalSize)
    function objective1(x)
        objectiveT(out, x, hParams)
        norm(out) / norm(uWeights)
    end
    x0 = vcat(ones(T, length(θknots)) .* v0, ones(T, length(ρknots)) .* (-0.75), ones(T, length(σknots)) .* 0.25)
    rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
    lowern = vcat(ones(T, length(θknots)) .* lower[1], ones(T, length(ρknots)) .* lower[2], ones(T, length(σknots)) .* lower[3])
    uppern = vcat(ones(T, length(θknots)) .* upper[1], ones(T, length(ρknots)) .* upper[2], ones(T, length(σknots)) .* upper[3])
    problem = GlobalOptimization.Problem(length(x0), objective1, lowern, uppern)
    result = if minimizer == "DE"
        optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(15),
            problem,
            rng, GlobalOptimization.Best1Bin(0.9, 0.5))
        GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(1000, 100, 1e-6, 1e-5))
    else
        optim = GlobalOptimization.SimulatedAnnealing(problem, rng)
        GlobalOptimization.optimize(optim, x0)
    end
    println("Result DE ", result, " ", GlobalOptimization.minimizer(optim))
    x0 = GlobalOptimization.minimizer(optim) #calibrated params.

    function objectiveTConstrained(F, xr)
        x = copy(xr)
        x[1:length(θknots)] = map(xri -> transforms[1](xri), xr[1:length(θknots)])
        x[length(θknots)+1:length(θknots)+length(ρknots)] = map(xri -> transforms[2](xri), xr[length(θknots)+1:length(θknots)+length(ρknots)])
        x[length(θknots)+length(ρknots)+1:length(θknots)+length(ρknots)+length(σknots)] = map(xri -> transforms[3](xri), xr[length(θknots)+length(ρknots)+1:length(θknots)+length(ρknots)+length(σknots)])
        #println("objectiveX ",x)
        return objectiveT(F, x, hParams)
    end
    x = x0
    xr = copy(x)
    xr[1:length(θknots)] = map(xri -> inv(transforms[1], xri), x[1:length(θknots)])
    xr[length(θknots)+1:length(θknots)+length(ρknots)] = map(xri -> inv(transforms[2], xri), x[length(θknots)+1:length(θknots)+length(ρknots)])
    xr[length(θknots)+length(ρknots)+1:length(θknots)+length(ρknots)+length(σknots)] = map(xri -> inv(transforms[3], xri), x[length(θknots)+length(ρknots)+1:length(θknots)+length(ρknots)+length(σknots)])
    gnRmse = GaussNewton.optimize!(objectiveTConstrained, xr, out, autodiff=:single)
    println("Result GN ", gnRmse)
    objectiveTConstrained(out, xr)
    return hParams, gnRmse
end



function calibrateHestonTSFromPrices(
    ts::AbstractVector{T},
    forwards::AbstractVector{T},
    strikes::AbstractMatrix{T},
    prices::AbstractMatrix{T},
    isCall::AbstractMatrix{Bool},
    weights::AbstractMatrix{T};
    κ=ones(length(ts)) .* 1.0,
    lower=[1e-4, -0.99, 1e-2], upper=[1.0, 0.5, 2.0],
    isRelative=false,
    method="Joshi-Yang",
    minimizer="DE",) where {T}
    # uPrices = zeros(T, length(ts),length(strikes))))
    # uWeights = zeros(T, length(ts),length(strikes))
    # for i  = eachindex(forwards)
    # 	@. uPrices[i,:] = prices[i,:] ./ forwards[i]
    # 	@. uWeights[i,:] = weights[i,:] .* forwards[i]
    # end
    n2 = size(prices, 2)
    uPrices = prices
    uWeights = weights
    volAtm = atmVolFromPrices(forwards[1], ts[1], strikes[1, :], prices[1, :], isCall[1, :])
    v0 = volAtm^2
    # lower =  [range[1].θ, range[1].ρ, range[1].σ]
    # upper =  [range[2].θ, range[2].ρ, range[2].σ]
    transforms = map((low, high) -> LogisticTransformation(low, high), lower, upper)

    startTime = vcat(zero(T), ts[1:end-1])
    θAll = zeros(length(startTime))
    ρAll = zeros(length(startTime))
    σAll = zeros(length(startTime))
    rmse = zeros(length(startTime))

    function objectiveTi(F, xr, hParams::HestonTSParams, i)
        #xr[1] = theta, xr[2] = rho, xr[3] = sigma.
        #x = map( (transformi, xri) -> transformi(xri), transforms, xr)
        θ = hParams.θ
        ρ = hParams.ρ
        σ = hParams.σ
        t = ts[i]
        θ[i] = xr[1]
        ρ[i] = xr[2]
        σ[i] = xr[3]
        cf = DefaultCharFunc(hParams)

        # pricer =JoshiYangCharFuncPricer(cf, t, n=64)
        pricer = if method == "Andersen-Lake"
            CharFuncPricing.ALCharFuncPricer(cf, n=64)
        elseif method == "Joshi-Yang"
            JoshiYangCharFuncPricer(cf, t, n=64)
        elseif method == "Cos-128"
            CharFuncPricing.makeCosCharFuncPricer(cf, t, 128, 12)
        elseif method == "Cos"
            CharFuncPricing.makeCosCharFuncPricer(cf, t, 256, 16)
        elseif method == "Cos-Junike"
            CharFuncPricing.makeCosCharFuncPricer(cf, t, tol=1e-4, maxM=4096)
        elseif method == "Flinn"
            CharFuncPricing.FlinnCharFuncPricer(cf, t, tTol=1e-4, qTol=1e-8)
        elseif method == "Flinn-Transformed"
            #CharFuncPricing.makeCVCharFunc(cf,t,CharFuncPricing.InitialControlVariance())
            CharFuncPricing.AdaptiveFlinnCharFuncPricer(cf, t)
        elseif method == "Swift"
            m, _ = CharFuncPricing.findSwiftScaling(cf, t)
            CharFuncPricing.makeSwiftCharFuncPricer(cf, t, m, 3)
        end
        for (j, strike) ∈ enumerate(strikes[i, :])
            if iszero(uWeights[i, j])
                F[j] = uWeights[i, j]
            else
                mPrice = CharFuncPricing.priceEuropean(pricer, isCall[i, j], strike / forwards[i], 1.0, t, 1.0) * forwards[i]
                F[j] = if isRelative
                    uWeights[i, j] * (mPrice / uPrices[i, j] - 1)
                else
                    uWeights[i, j] * (mPrice - uPrices[i, j])
                end
            end
        end
        F
    end
    for (i, t) ∈ enumerate(ts)
        θ = θAll[1:i]
        ρ = ρAll[1:i]
        σ = σAll[1:i]

        hParams = HestonTSParams(v0, κ[1:i], θ, ρ, σ, startTime[1:i])

        objectiveT(F, xr) = objectiveTi(F, xr, hParams, i)

        totalSize = n2
        out = zeros(totalSize)
        function objective1(x)
            objectiveT(out, x)
            norm(out) / norm(uWeights)
        end
        x0 = if i > 0
            [v0, -0.75, 0.25]
        else
            [θ[i-1], ρ[i-1], σ[i-1]]
        end
        x = if i == 1
            rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
            problem = GlobalOptimization.Problem(length(x0), objective1, lower, upper)
            result = if minimizer == "DE"
                optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(15),
                    problem,
                    rng, GlobalOptimization.Best1Bin(0.9, 0.5))
                GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(1000, 100, 1e-6, 1e-5))
            else
                optim = GlobalOptimization.SimulatedAnnealing(problem, rng)
                GlobalOptimization.optimize(optim, x0)
            end
            println("Result DE ", result, " ", GlobalOptimization.minimizer(optim))
            GlobalOptimization.minimizer(optim) #calibrated params.
        else
            x0
        end

        function objectiveTiConstrained(F, xr)
            x = map((transformi, xri) -> transformi(xri), transforms, xr)
            return objectiveT(F, x)
        end
        rx = map((transformi, xri) -> inv(transformi, xri), transforms, x)
        gnRmse = GaussNewton.optimize!(objectiveTiConstrained, rx, out, autodiff=:single)
        println("Result GN ", gnRmse)
        θ[i] = transforms[1](rx[1])
        ρ[i] = transforms[2](rx[2])
        σ[i] = transforms[3](rx[3])
        rmse[i] = objective1([θ[i], ρ[i], σ[i]])
        θAll[i] = θ[i]
        ρAll[i] = ρ[i]
        σAll[i] = σ[i]
    end
    isGlobal = true
    if isGlobal
        # additional global calib

        hParams = HestonTSParams(v0, κ, θAll, ρAll, σAll, startTime)

        function objectiveTConstrained(F, xr)
            for i = eachindex(ts)
                xri = xr[3*(i-1)+1:3*(i-1)+3]
                x = map((transformi, xri) -> transformi(xri), transforms, xri)
                θAll[i] = x[1]
                ρAll[i] = x[2]
                σAll[i] = x[3]
            end
            for i = eachindex(ts)
                xri = xr[3*(i-1)+1:3*(i-1)+3]
                x = map((transformi, xri) -> transformi(xri), transforms, xri)
                #println("objectiveTConstrained x=",x," i=",i)
                objectiveTi(@view(F[(i-1)*n2+1:i*n2]), x, hParams, i)
            end
            F
        end
        rx = zeros(3 * length(startTime))
        for i = 1:length(startTime)
            x = [θAll[i], ρAll[i], σAll[i]]
            rx[3*(i-1)+1:3*(i-1)+3] = map((transformi, xri) -> inv(transformi, xri), transforms, x)
        end
        out = zeros(length(ts) * n2)
        gnRmse = GaussNewton.optimize!(objectiveTConstrained, rx, out, autodiff=:single)
        println("Result GN ", gnRmse)
        for i = 1:length(startTime)
            rxi = rx[3*(i-1)+1:3*(i-1)+3]
            θAll[i] = transforms[1](rxi[1])
            ρAll[i] = transforms[2](rxi[2])
            σAll[i] = transforms[3](rxi[3])
        end
        # for i = 1:length(ts)
        #     rmse[i] = objective1([θ[i], ρ[i], σ[i]])
        # end
    end
    hParams = HestonTSParams(v0, κ, θAll, ρAll, σAll, startTime)
    return hParams, rmse
end


function calibrateHestonTSFromVolsVS(ts::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractMatrix{T}, vols::AbstractMatrix{T}; weights::AbstractMatrix{T}=ones(T, length(ts), length(strikes)), vegaFloor=1e-2, minimizer="DE", isRelative=false,
    κ=ones(length(ts)) .* 1.0, method="Joshi-Yang", truncationDev=2.0,
    lower=[1e-4, -0.99, 1e-2], upper=[1.0, 0.5, 2.0], swapCurve=makeSwapCurveReplication(ts, forwards, strikes, vols)
) where {T}
    uPrices, isCall, uWeights = convertVolsToPricesOTMWeights(ts, forwards, strikes, vols, weights=weights, vegaFloor=vegaFloor, truncationDev=truncationDev)

    #println(swapPrices)
    v0 = swapCurve(0.0)


    θ = zeros(length(ts))
    startTimes = vcat(zero(T), ts[1:end-1])
    prevTotalVar = zero(T)
    vPrev = v0
    for i = 1:length(ts)
        swapPricesi = swapCurve(ts[i])
        dti = ts[i] - startTimes[i]
        eFactor = (1 - exp(-κ[i] * dti)) / κ[i]
        θ[i] = (swapPricesi * ts[i] - prevTotalVar - vPrev * eFactor) / (dti - eFactor)
        θ[i] = min(max(θ[i], lower[1]), upper[1])
        vPrev = (vPrev - θ[i]) * exp(-κ[i] * dti) + θ[i]
        prevTotalVar = swapPricesi * ts[i]
    end

    #println(θ)

    transforms = map((low, high) -> LogisticTransformation(low, high), lower[2:end], upper[2:end])
    n2 = size(vols, 2)
    startTime = vcat(zero(T), ts[1:end-1])
    θAll = copy(θ)
    ρAll = zeros(length(startTime))
    σAll = zeros(length(startTime))
    rmse = zeros(length(startTime))
    for (i, t) ∈ enumerate(ts)
        #θ = θAll[1:i]
        ρ = ρAll[1:i]
        σ = σAll[1:i]

        hParams = HestonTSParams(v0, κ[1:i], θ[1:i], ρ, σ, startTime[1:i])


        function objectiveT(F, xr)
            #xr[1] = theta, xr[2] = rho, xr[3] = sigma.
            #x = map( (transformi, xri) -> transformi(xri), transforms, xr)

            #θ[i] = xr[1]
            ρ[i] = xr[1]
            σ[i] = xr[2]
            cf = DefaultCharFunc(hParams)

            # pricer =JoshiYangCharFuncPricer(cf, t, n=64)
            pricer = if method == "Andersen-Lake"
                CharFuncPricing.ALCharFuncPricer(cf, n=64)
            elseif method == "Joshi-Yang"
                JoshiYangCharFuncPricer(cf, t, n=64)
            elseif method == "Cos-128"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, 128, 12)
            elseif method == "Cos"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, 256, 16)
            elseif method == "Cos-Junike"
                CharFuncPricing.makeCosCharFuncPricer(cf, t, tol=1e-4, maxM=4096)
            elseif method == "Flinn"
                CharFuncPricing.FlinnCharFuncPricer(cf, t, tTol=1e-4, qTol=1e-8)
            elseif method == "Flinn-Transformed"
                #CharFuncPricing.makeCVCharFunc(cf,t,CharFuncPricing.InitialControlVariance())
                CharFuncPricing.AdaptiveFlinnCharFuncPricer(cf, t)
            elseif method == "Swift"
                m, _ = CharFuncPricing.findSwiftScaling(cf, t)
                CharFuncPricing.makeSwiftCharFuncPricer(cf, t, m, 3)
            end
            for (j, strike) ∈ enumerate(strikes[i, :])
                if iszero(uWeights[i, j])
                    F[j] = uWeights[i, j]
                else
                    mPrice = CharFuncPricing.priceEuropean(pricer, isCall[i, j], strike / forwards[i], 1.0, t, 1.0) * forwards[i]
                    F[j] = if isRelative
                        uWeights[i, j] * (mPrice / uPrices[i, j] - 1)
                    else
                        uWeights[i, j] * (mPrice - uPrices[i, j])
                    end
                end
            end
            F
        end
        totalSize = n2
        out = zeros(totalSize)
        function objective1(x)
            objectiveT(out, x)
            norm(out) / norm(uWeights)
        end
        x0 = if i == 1
            [-0.5, 0.25]
        else
            [ρ[i-1], σ[i-1]]
        end
        rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
        problem = GlobalOptimization.Problem(length(x0), objective1, lower[2:end], upper[2:end])
        result = if minimizer == "DE"
            optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(15),
                problem,
                rng, GlobalOptimization.Best1Bin(0.9, 0.5))
            GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(1000, 100, 1e-6, 1e-5))
        else
            optim = GlobalOptimization.SimulatedAnnealing(problem, rng)
            GlobalOptimization.optimize(optim, x0)
        end
        println("Result DE ", result, " ", GlobalOptimization.minimizer(optim))
        x = GlobalOptimization.minimizer(optim) #calibrated params.
        function objectiveConstrained(F, xr)
            x = map((transformi, xri) -> transformi(xri), transforms, xr)
            return objectiveT(F, x)
        end
        rx = map((transformi, xri) -> inv(transformi, xri), transforms, x)
        gnRmse = GaussNewton.optimize!(objectiveConstrained, rx, out, autodiff=:single)
        println("Result GN ", gnRmse)
        #θ[i] = transforms[1](rx[1])
        ρ[i] = transforms[1](rx[1])
        σ[i] = transforms[2](rx[2])
        x = map((transformi, xri) -> transformi(xri), transforms, rx)
        rmse[i] = objective1(x)
        #θAll[i] = θ[i]
        ρAll[i] = ρ[i]
        σAll[i] = σ[i]
    end
    hParams = HestonTSParams(v0, κ, θ, ρAll, σAll, startTime)
    volError, rmseVol = estimateVolError(hParams, ts, forwards, strikes, vols, weights=weights)
    #println("RMSEVOL ",volError)
    return hParams, rmse, volError
    #return params, rmseVol
end


function estimateVolError(hagParams::HaganHestonPiecewiseParams, ts::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractMatrix{T}, vols; weights=ones(T, length(ts), length(strikes))) where {T}
    params, lastτ = makeHestonTSParams(hagParams, tte=ts[end])
    τs = vcat(params.startTime[2:end], lastτ)

    volError = zeros(size(vols))
    for (i, t) ∈ enumerate(ts)
        cf = DefaultCharFunc(params)
        #pricer = CharFuncPricing.ALCharFuncPricer(cf, n=128)
        τ = convertTime(hagParams, t)
        pricer = CharFuncPricing.JoshiYangCharFuncPricer(cf, τ)
        for (j, strike) ∈ enumerate(strikes[i, :])
            isCall = strike > forwards[i]
            mPrice = CharFuncPricing.priceEuropean(pricer, isCall, strike / forwards[i], 1.0, τ, 1.0)
            try
                volError[i, j] = weights[i, j] * (Black.impliedVolatility(isCall, mPrice, 1.0, strike / forwards[i], t, 1.0) - vols[i, j])
            catch e
                println(e)
                volError[i, j] = NaN
            end
        end
    end
    rmseVol = norm(volError) / sqrt(size(vols, 2) * length(ts)) #around 0.015
    return volError, rmseVol
end
