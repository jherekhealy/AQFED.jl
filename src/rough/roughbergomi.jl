#flat fwd variance curve, iv skew compared to approx.
import AQFED.Black

import AQFED.Math: norminv
using Statistics
using HypergeometricFunctions
using SpecialFunctions
using LinearAlgebra
import AQFED.Random: next!, nextn!, skipTo
using AQFED.Random
using AQFED.MonteCarlo
import DSP: conv

struct FlatRoughExp
    σ::Float64
    H::Float64
    η::Float64
end

struct FlatRoughBergomi
    σ::Float64 #σ^2 is initial forward variance
    H::Float64
    η::Float64
    ρ::Float64
end

function logForward(model::FlatRoughBergomi, lnspot, t0)
    return lnspot
end

function ndims(model::FlatRoughBergomi, specificTimes::Vector{Float64}, timestepSize::Float64)
    genTimes = pathgenTimes(model, specificTimes, timestepSize)
    return (length(genTimes) - 1) * 2 #0.0 does not count
end

function covariance(model::FlatRoughBergomi, ti, tj)
    γ = 0.5 - model.H
    return ti^(model.H * 2) * (1 - 2 * γ) / (1 - γ) * (ti / tj)^γ * _₂F₁(γ, 1, 2 - γ, ti / tj) #rough bergomi
end

function covariance(model::FlatRoughExp, ti, tj)
    H = model.H
    return (ti^(2H) + tj^(2H) - abs(ti - tj)^(2H)) / 2
end

function instantVariance(model::FlatRoughBergomi, x, t)
    return model.σ^2 * exp((-model.η^2 * t^(2 * model.H)) / 2 + model.η * x)
end

function instantVariance(model::FlatRoughExp, x, t)
    return model.σ^2 * exp(2 * model.η * x)
end

abstract type IntegralDiscretization end

struct LeftRectangular <: IntegralDiscretization
end

struct RightRectangular <: IntegralDiscretization
end

struct Trapezoidal <: IntegralDiscretization
end

function computeVarianceIntegral(disc::Trapezoidal, model, futureTimes, wz1, timestepSize, i, lagIndex)
    value = (instantVariance(model, wz1[i+lagIndex], futureTimes[i+lagIndex]) + instantVariance(model, wz1[i], futureTimes[i])) / 2
    if lagIndex > 1
        value += sum(instantVariance(model, x, t) for (x, t) in zip(wz1[i+1:i+lagIndex-1], futureTimes[i+1:i+lagIndex-1])) #or should it be from i to i+lagIndex-1?
    end
    return value * timestepSize
end

function computeVarianceIntegral(disc::RightRectangular, model, futureTimes, wz1, timestepSize, i, lagIndex)
    return timestepSize * sum(instantVariance(model, x, t) for (x, t) in zip(wz1[i+1:i+lagIndex], futureTimes[i+1:i+lagIndex])) #or should it be from i to i+lagIndex-1?
end

function computeVarianceIntegral(disc::LeftRectangular, model, futureTimes, wz1, timestepSize, i, lagIndex)
    return timestepSize * sum(instantVariance(model, x, t) for (x, t) in zip(wz1[i:i+lagIndex-1], futureTimes[i:i+lagIndex-1])) #or should it be from i to i+lagIndex-1?
end




dimensions(optionTte, timestepSize) = round(Int, optionTte / timestepSize)
makeGenTimes(tte, timestepSize) = collect(range(0.0, stop=tte, step=timestepSize))

function simulateImpliedVolatility(prng, model::Union{FlatRoughExp,FlatRoughBergomi}, tte::Float64, optionTte::Float64, nSim::Int, timestepSize::Float64; qrng=prng, reuseNumbers=true, useAntithetic::Bool=false, disc::IntegralDiscretization=Trapezoidal)
    #simulate option X days implied vol from 0 to tte.
    genTimes = makeGenTimes(tte, timestepSize)
    lagIndex = dimensions(optionTte, timestepSize)
    #println(genTimes, " ",lagIndex," ", genTimes[1]," ", genTimes[lagIndex]," ", genTimes[lagIndex+1]-genTimes[1])
    dimt = (length(genTimes) - 1)
    dimz = dimt
    covar = Array{Float64}(undef, (dimz, dimz))

    for (i, ti) in enumerate(genTimes[2:end])
        for (j, tj) in enumerate(genTimes[2:end])
            if (ti <= tj)
                covar[i, j] = covariance(model, ti, tj)
            end
        end
    end
    for (i, ti) in enumerate(genTimes[2:end])
        for (j, tj) in enumerate(genTimes[2:end])
            if (ti > tj)
                covar[i, j] = covar[j, i]
            end
        end
    end
    sqrtCovar = cholesky(covar).L
    t0 = genTimes[1]
    #Simulate one path
    z0 = Vector{Float64}(undef, dimz)
    z1 = Vector{Float64}(undef, lagIndex)
    z = zeros((lagIndex, nSim))
    nPaths = nSim
    if useAntithetic
        nPaths *= 2
    end
    v = zeros((dimz, nPaths))
    nextn!(prng, z0)
    wz0 = zeros(dimz)
    wz1 = zeros(dimz)
    wz1a = zeros(dimz)

    price = zeros(dimt - lagIndex)
    iv = Vector{Float64}(undef, length(price))
    #store nSimxgenTimes instantvariance
    #expectation for vanilla option at each time  step does not need to be independent. We can use the same paths.
    futureTimes = genTimes[2:end]
    wzKnown = zero(dimz)

    #could use low disc seq if this does not introduce a bias. size=lagIndex+nTimes. Or lagIndex and use PRNG for nTimes? Makes sense since only 1 time.
    if reuseNumbers
        for sim = 1:nSim
            nextn!(qrng, z1)
            z[:, sim] .= z1
        end
    end
    elapsed = @elapsed for (i, ti) = enumerate(futureTimes[1:end-lagIndex])
        if !reuseNumbers
            for sim = 1:nSim
                nextn!(qrng, z1)
                z[:, sim] .= z1
            end
        end
        @inbounds @simd for j = i:i+lagIndex
            wz0[j] = sum(sqrtCovar[j, p] * z0[p] for p = 1:i)
        end
        for sim = 1:nSim
            #wz0 .= sqrtCovar * vec(z1)  #wz0[row] .= sqrtCovar[row,:] * vec(z1)
            wz1[i] = wz0[i]
            @inbounds @simd for j = i+1:i+lagIndex
                wz1[j] = wz0[j] + sum(sqrtCovar[j, p] * z[p-i, sim] for p = i+1:j)
            end
            if useAntithetic
                wz1a[i] = wz0[i]
                @inbounds @simd for j = i+1:i+lagIndex
                    wz1a[j] = 2 * wz0[j] - wz1[j]  #zAnti = -z
                end
            end
            v[i, sim] = computeVarianceIntegral(disc, model, futureTimes, wz1, timestepSize, i, lagIndex)
            if useAntithetic
                v[i, sim+nSim] = computeVarianceIntegral(disc, model, futureTimes, wz1a, timestepSize, i, lagIndex)
            end
            #v[i,nSim, tteIndex] = ... in for loop from i+1 to i+lagIndex!
        end
    end
    println("elapsed ", elapsed)

    for (i, ti) = enumerate(futureTimes[1:end-lagIndex])
        @inbounds price[i] = sum(Black.blackScholesFormula(true, 1.0, 1.0, v[i, sim], 1.0, 1.0) for sim = 1:nPaths)
        price[i] /= nPaths
        iv[i] = Black.impliedVolatilitySRHalley(true, price[i], 1.0, 1.0, optionTte, 1.0, 1e-8, 64, Black.CMethod())
    end
    #	println(i, " ",ti," ", totalVariance," ",sim)
    return futureTimes[1:end-lagIndex], iv, v
end

function simulateImpliedVolatilityOld(prng, model::FlatRoughBergomi, tte, optionTte, nSim::Int, timestepSize::Float64)
    #simulate option X days implied vol from 0 to tte.
    genTimes = pathgenTimes(model, [tte], timestepSize)
    lagIndex = round(Int, optionTte / timestepSize)
    #println(genTimes, " ",lagIndex," ", genTimes[1]," ", genTimes[lagIndex]," ", genTimes[lagIndex+1]-genTimes[1])
    # genTimes[1] = 0
    dimt = (length(genTimes) - 1)
    dimz = dimt
    z = Vector{Float64}(undef, dimz)
    covar = Array{Float64}(undef, (dimz, dimz))
    H = model.H
    η = model.η
    σ = model.σ
    γ = 0.5 - model.H
    DH = sqrt(2 * model.H) / (model.H + 0.5)
    for (i, ti) in enumerate(genTimes[2:end])
        for (j, tj) in enumerate(genTimes[2:end])
            if (ti <= tj)
                #covar[i, j] = (ti^(2H)+tj^(2H)-abs(ti-tj)^(2H))/2
                covar[i, j] = ti^(model.H * 2) * (1 - 2 * γ) / (1 - γ) * (ti / tj)^γ * _₂F₁(γ, 1, 2 - γ, ti / tj) #rough bergomi
            end
        end
    end
    for (i, ti) in enumerate(genTimes[2:end])
        for (j, tj) in enumerate(genTimes[2:end])
            if (ti > tj)
                covar[i, j] = covar[j, i]
            end
        end
    end
    sqrtCovar = cholesky(covar).L
    t0 = genTimes[1]
    #Simulate one path
    nextn!(prng, z)

    price = zeros(dimt - lagIndex)
    iv = Vector{Float64}(undef, length(price))
    for (i, ti) = Iterators.reverse(enumerate(genTimes[2:end-lagIndex]))
        for sim = 1:nSim
            #on the path we are at ti
            #wzi = wz[:]
            nextn!(prng, @view z[i+1:end])
            wzi = (sqrtCovar) * vec(z) #TODO could truncate as we need only up to lagIndex
            #totalVariance =  exp(2*( -η^2 * genTimes[i+1+lagIndex]^(2H) / 2 + η*wzi[i+lagIndex])) - exp(2*( -η^2 * genTimes[i+2]^(2H) / 2 + η*wzi[i+1]))
            totalVariance = timestepSize * sum(σ^2 * exp((-η^2 * t^(2H) / 2 + η * x)) for (x, t) in zip(wzi[i+1:i+lagIndex], genTimes[i+2:i+1+lagIndex])) #Bergomi
            #totalVariance =  timestepSize*sum( σ^2*exp(2*η*x) for x  in wzi[i+2:i+lagIndex])
            price[i] += Black.blackScholesFormula(true, 1.0, 1.0, totalVariance, 1.0, 1.0)
            #	println(i, " ",ti," ", totalVariance," ",sim)
            #TODO why isn't last price more stable? because the path changes!
        end
    end
    for (i, ti) = enumerate(genTimes[2:end-lagIndex])
        price[i] /= nSim
        iv[i] = Black.impliedVolatilitySRHalley(true, price[i], 1.0, 1.0, optionTte, 1.0, 1e-8, 64, Black.CMethod())
    end
    return genTimes[2:end-lagIndex], iv
end

function simulate(
    rng,
    model::FlatRoughBergomi,
    spot::Float64,
    payoff::MonteCarlo.VanillaOption,
    start::Int,
    nSim::Int,
    timestepSize::Float64
)
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = 1.0
    genTimes = MonteCarlo.pathgenTimes(model, specTimes, timestepSize)

    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, tte)
    dimt = (length(genTimes) - 1)
    dimz = dimt * 2
    z = Vector{Float64}(undef, dimz)
    u = Array{Float64}(undef, (2, dimt))
    covar = Array{Float64}(undef, (dimz, dimz))
    payoffValues = Vector{Float64}(undef, nSim)
    specValues = Vector{Float64}(undef, length(specTimes))
    γ = 0.5 - model.H
    DH = sqrt(2 * model.H) / (model.H + 0.5)
    #println("genTimes ",genTimes)
    for (i, ti) in enumerate(genTimes[2:end])
        for (j, tj) in enumerate(genTimes[2:end])
            if (ti <= tj)
                covar[i*2-1, j*2-1] = min(ti, tj) #Zu,Zu
                # F(x) = 1/x^gamma*F(1/x) = 1/x^gamma * 1/ (x-s)(1-s)
                # 
                #covar[i*2,j*2] = ti^(model.H*2) * (1-2*γ)/(1 - γ) * _₂F₁(1, γ, 2-γ, ti/tj)
                #1-s * x-s   => b=1, c= 2-gamma, a = gamma z = 1/x   factor= 1/x^gamma
                covar[i*2, j*2] = ti^(model.H * 2) * (1 - 2 * γ) / (1 - γ) * (ti / tj)^γ * _₂F₁(γ, 1, 2 - γ, ti / tj)
                covar[i*2, j*2-1] = model.ρ * DH * ti^(model.H + 0.5)
                covar[i*2-1, j*2] = model.ρ * DH * (tj^(model.H + 0.5) - (tj - ti)^(model.H + 0.5))
                # covar[i,j] = min(ti,tj) #Zu,Zu
                # covar[i+dimt-1,j] = ti^(model.H*2) * (1-2*γ)/(1 - γ) * _₂F₁(1, γ, 2-γ, tj/ti)
                # covar[i,j+dimt-1] = tj^(model.H*2) * (1-2*γ)/(1 - γ) * _₂F₁(1, γ, 2-γ, ti/tj)
                # covar[i+dimt-1,j+dimt-1] = model.ρ * DH *( tj^(model.H+0.5) - (tj-min(ti,tj))^(model.H+0.5) )
            end
        end
    end
    for (i, ti) in enumerate(genTimes[2:end])
        for (j, tj) in enumerate(genTimes[2:end])
            if (ti > tj)
                covar[i*2-1, j*2-1] = covar[j*2-1, i*2-1]
                covar[i*2, j*2] = covar[j*2, i*2]
                covar[i*2, j*2-1] = covar[j*2-1, i*2]
                covar[i*2-1, j*2] = covar[j*2, i*2-1]
            end
        end
    end
    sqrtCovar = cholesky(covar).L
    #skipTo(rng, start)
    for sim = 1:nSim
        specIndex = 1
        t0 = genTimes[1]
        nextn!(rng, z)
        t0 = genTimes[1]
        @inbounds for i = 1:dimt
            dim = 2 * i - 1
            t1 = genTimes[i+1]
            u[1, i] = z[dim]
            u[2, i] = z[dim+1]
            t0 = t1
        end

        wz = sqrtCovar * vec(u) #vector of columns
        u = reshape(wz, 2, dimt)
        #now compute dz, dw from u = Z, W).
        t0 = genTimes[1]
        logpathValue = lnf0
        lnv = 0.0
        local sqrtmv = model.σ
        z0 = 0.0
        w0 = 0.0
        @inbounds for i = 1:dimt
            t1 = genTimes[i+1]
            # dim = 2 * i - 1
            dZ = u[1, i] - z0
            z0 = u[1, i]
            w1 = u[2, i]

            h = t1 - t0
            lnf1 = logForward(model, lnspot, t1)

            logpathValue += lnf1 - lnf0 - sqrtmv^2 / 2 * h + sqrtmv * dZ
            #lnv += -model.η^2 * (t1^(2*model.H) - t0^(2*model.H) )/ 2  + model.η*dW  
            lnv = -model.η^2 * t1^(2 * model.H) / 2 + model.η * w1
            sqrtmv = model.σ * exp(lnv / 2)

            if specIndex <= length(specTimes) && t1 == specTimes[specIndex]
                specValues[specIndex] = exp(logpathValue)
                if specIndex == length(specTimes)
                    payoffValues[sim] = evaluatePayoffOnPath(payoff, specValues, df)
                else
                    specIndex += 1
                end
            end
            t0 = t1
            w0 = w1
            lnf0 = lnf1
        end

    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))

end

slowconvcost(m,n) = 2e-9*m*n+1e-6
fastconvcost(m,n) = 7e-8*(m+n)*log(m+n)+1e-4
function sebconv(a::Array{T}, b::Array{T}) where {T}
    m = length(a)
    n = length(b)
    if(fastconvcost(m,n)<slowconvcost(m,n)) return conv(a,b); end
    c = zeros(T,m+n-1)
    @inbounds for j=1:m
        for k=1:n
            c[j+k-1] += a[j]*b[k]
        end
    end
    return c
end

function simulateHybrid(
    rng,
    model::FlatRoughBergomi,
    spot::Float64,
    payoff::MonteCarlo.VanillaOption,
    start::Int,
    nSim::Int,
    n::Int #steps per year
)
    specTimes = specificTimes(payoff)
    tte = specTimes[end]
    df = 1.0
    dt = 1.0 / n

    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, tte)
    dimt = Int(floor(n * tte))
    genTimes = collect(range(0.0, stop=tte, length=(1 + dimt)))

    dimz = dimt * 2
    z = Vector{Float64}(undef, dimz)
    payoffValues = Vector{Float64}(undef, nSim)
    specValues = Vector{Float64}(undef, length(specTimes))
    γ = 0.5 - model.H
    a = -γ #n = dimt

    Y1 = zeros(dimt + 1)
    Y = zeros(dimt + 1)

    G = zeros(dimt + 1)
    V = zeros(dimt + 1)
    g = function (x, a)
        return x^a
    end
    b = function (k, a)
        return ((k^(a + 1) - (k - 1)^(a + 1)) / (a + 1))^(1 / a)
    end
    c = [1.0/n 1.0/((a+1)*n^(a+1)); 0.0 1.0/((2a+1)*n^(2a+1))]
    c[2, 1] = c[1, 2]
    dW = zeros((dimt,2))
    dW2 = zeros(dimt)
    dB = zeros(dimt)
    sqrtc = cholesky(c).L
    for k = 2:dimt
        G[k+1] = g(b(k, a) / n, a)
    end
    ta = @. (model.η^2 * genTimes^(2a + 1)) / 2
    for sim = 1:nSim
        specIndex = 1
        t0 = genTimes[1]
        nextn!(rng, z)
        nextn!(rng, dW2)
        @inbounds for i = 1:dimt
            dW[i, 1] = sqrtc[1,1] * z[2i-1] + sqrtc[1,2]*z[2i]
            dW[i,2] = sqrtc[2,1] * z[2i-1] + sqrtc[2,2]*z[2i]
        end
        Y1[2:dimt+1] .= dW[:,2]
        X = dW[:,1]
        Y2 = sebconv(G, X)[1:1+dimt]
        @. Y = sqrt(2a + 1) * (Y1 + Y2)
        @. V = model.σ^2 * exp(model.η * Y - ta)
        dW2 .*= sqrt(dt)
        @. dB = model.ρ * dW[:,1] + sqrt(1 - model.ρ^2) * dW2

        logpathValue = lnf0

        for i = 1:dimt
            t1 = genTimes[i+1]
            # dim = 2 * i - 1           

            h = t1 - t0
            lnf1 = logForward(model, lnspot, t1)

            logpathValue += lnf1 - lnf0 - V[i] * h / 2+  sqrt(V[i]) * dB[i]

            if specIndex <= length(specTimes) && t1 >= specTimes[specIndex] - 1e-8
                specValues[specIndex] = exp(logpathValue)
                if specIndex == length(specTimes)
                    payoffValues[sim] = evaluatePayoffOnPath(payoff, specValues, df)
                else
                    specIndex += 1
                end
            end
            t0 = t1
            lnf0 = lnf1
        end

    end
    payoffMean = mean(payoffValues)
    return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))

end



function atmSkew(model::FlatRoughBergomi, T::Float64)
    γ = 0.5 - model.H
    DH = sqrt(2 * model.H) / (model.H + 0.5)
    EH = DH / (model.H + 1.5)
    return model.ρ * model.η / 2 * EH / T^γ + model.ρ^2 * model.η^2 * model.σ * T^(2 * model.H) / 4 * (DH^2 / (1 + model.H) * (1 + gamma(model.H + 1.5)^2 / gamma(model.H * 2 + 3) - 1.5 * EH^2))
end

#=
model = AQFED.MonteCarlo.FlatRoughBergomi(0.025,0.05,0.4,-0.65)
payoff = AQFED.MonteCarlo.VanillaOption(true,1.0,AQFED.MonteCarlo.BulletCashFlow(1.0,1.0,false,0.0),0.0)
AQFED.MonteCarlo.simulateHybrid(AQFED.Random.ZRNGSeq(AQFED.Random.MRG32k3a(),1), model, 1.0, payoff, 1, 1024*64, 365)
=#