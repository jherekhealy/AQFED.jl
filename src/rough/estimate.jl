using Roots
using Statistics
using Base.Threads
function wStatA(ts, vs, K0, L, step, p)
    bvs = vs # big.(vs)
    bts = ts # big.(ts)
    value = sum(abs(log(bvs[k+K0]) - log(bvs[k]))^p / sum(abs(log(bvs[l+step]) - log(bvs[l]))^p for l in k:step:k+K0-step) * abs((bts[k+K0] - bts[k])) for k = 1:K0:L-K0+1)
    #println("wStat ", p, " ",value)
    return value
end

function meanRoughness(tm, ivm, K0, L)
    cm = zeros(length(tm) - L)
    Threads.@threads for i = eachindex(cm)
        local ivi = ivm[i:i+L]
        local ti = tm[i:i+L]
        T = abs((ti[end] - ti[1]))
        try
            cm[i] = 1.0 / find_zero(p -> wStatA(ti, ivi, K0, L, 1, p) - T, (1.0, 10.0))
        catch e
            if isa(e, ArgumentError)
                cm[i] = 0.0
            else
                throw(e)
            end
        end
    end
    meanValue = mean(filter(function (x)
            x > 0
        end, cm))
    stdValue = std(filter(function (x)
            x > 0
        end, cm))
    return meanValue, stdValue, cm
end

function estimateRHanSchied(tm, ivm)
    m = length(ivm)
    nFinal = Int(trunc(log2(m - 1))) - 2
    println(nFinal)
    ϑ = zeros(2^nFinal)
    r = zeros(nFinal)
    for n = 1:nFinal
        for k = 0:(2^n-1)
            npow = 2^(nFinal - n)
            ϑ[k+1] = (ivm[Int(4k * npow)+1] - 2ivm[Int((4k + 1) * npow)+1] + 2ivm[Int((4k + 3) * npow)+1] - ivm[Int((4k + 4) * npow)+1])
            ϑ[k+1] *= 2^(3n / 2 + 3)
        end
        r[n] = 1 - log2(sqrt(sum(ϑ[1:2^n] .^ 2))) / n
    end
    return r
end
function estimateRSHanSchied(r; α=ones(Float64, Int(trunc(length(r) / 2))))
    m = length(α) - 1 #starts at 0
    n = length(r) - 1
    c = sum([α[n-k+1] / (k^2 * (k - 1)^2) for k = (n-m):n])
    rs = 0.0
    for k = (n-m-1):n
        β = if k == n
            1 + α[1] / (c * n^2 * (n - 1))
        elseif k == n - m - 1
            -α[m+1] / (c * n * (n - m) * (n - m - 1))
        else
            (α[n-k+1] / (k - 1) - α[n-k] / (k + 1)) / (c * n * k)
        end
        rs += β * r[k]
    end
    return rs
end

function realizedVolatility(t, x, freq, durationMult; start=1)
    duration = freq * durationMult
    freqi = Int(trunc(length(t) / Int(trunc(t[end] / freq))))
    smap = start:freqi:length(t)
    p = x[smap]
    #println("smap ", smap, " ",p," ",length(p))
    imap = collect(1:durationMult:length(p)-durationMult-1)
    r = zeros(length(imap))
    for (ii, i) = enumerate(imap)
        r[ii] = sum((@view(p[i+1:i+1+durationMult]) - @view(p[i:i+durationMult])) .^ 2) / duration
    end

    return t[smap][imap], r
end


function parkinsonVolatility(t, x, freq, durationMult; start=1)
    duration = freq * durationMult * 4 * log(2)
    freqi = Int(trunc(length(t) / Int(trunc(t[end] / freq))))
    smap = start:freqi:length(t)
    p = x[smap]
    #println("smap ", smap, " ",p," ",length(p))
    imap = collect(1:durationMult:length(p)-durationMult)
    r = zeros(length(imap))
    for (ii, i) = enumerate(imap)
        r[ii] = ((maximum(@view(p[i:i+durationMult])) - minimum(@view(p[i:i+durationMult]))) .^ 2) / duration
    end

    return t[smap][imap], r
end


function realizedVolatilityF(t, x, freq, durationMult; start=1)
    duration = freq * durationMult
    freqi = Int(trunc(length(t) / Int(trunc(t[end] / freq))))
    smap = start:freqi:length(t)
    p = x[smap]
    ts = t[smap]
    #println("smap ", smap, " ",p," ",length(p))
    r = zeros(Int(trunc(length(p) / durationMult)))
    tr = zeros(length(r))
    for j = eachindex(r)
        rv = 0.0
        for i = 1:durationMult
            rv += (p[(j-1)*durationMult+i+1] - p[(j-1)*durationMult+i])^2
        end
        r[j] = rv / duration
        tr[j] = ts[j*durationMult]
    end
    return tr, r
end

function realizedVolatilityCum(t, x, freq, durationMult; start=1)
    duration = freq * durationMult
    freqi = Int(trunc(length(t) / Int(trunc(t[end] / freq))))
    println(freqi)
    smap = start:freqi:length(t)
    p = x[smap]
    ts = t[smap]
    #println("smap ", smap, " ",p," ",length(p))
    r = zeros(Int(trunc(length(p) / durationMult)) + 1)
    tr = zeros(length(r))
    rv = 0.0
    tr[1] = t[1]
    r[1] = rv
    for j = 1:length(r)-1
        for i = 1:durationMult
            rv += (p[(j-1)*durationMult+i+1] - p[(j-1)*durationMult+i])^2
        end
        tr[j+1] = t[j*durationMult+1]
        r[j+1] = rv / duration
    end
    return tr, r
end


struct AbsoluteBrownianSV
end

logForward(model::AbsoluteBrownianSV, lnspot, tte) = lnspot

function simulatePath(
    rng,
    model::AbsoluteBrownianSV,
    spot::Float64,
    tte::Float64,
    n::Int #steps per year
)
    dt = 1.0 / n

    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, tte)
    dimt = Int(floor(n * tte))
    genTimes = collect(range(0.0, stop=tte, length=(1 + dimt)))

    dimz = dimt * 2
    z = Vector{Float64}(undef, dimz)

    logpathValues = zeros(dimt + 1)
    sqrtdt = sqrt(dt)
    t0 = genTimes[1]
    nextn!(rng, z)
    logpathValues[1] = lnf0
    vi = 0.0
    for i = 1:dimt
        t1 = genTimes[i+1]
        h = t1 - t0
        lnf1 = logForward(model, lnspot, t1)
        logpathValues[i+1] = logpathValues[i] + lnf1 - lnf0 - vi^2 * h / 2 + vi * z[2i-1] * sqrtdt
        vi = abs.(z[2i]) * sqrt(t1)
        t0 = t1
        lnf0 = lnf1
    end

    return genTimes, logpathValues

end


struct ExpOUSV
    y0::Float64
    σ::Float64
    γ::Float64
    θ::Float64
end

logForward(model::ExpOUSV, lnspot, tte) = lnspot

function simulatePath(
    rng,
    model::ExpOUSV,
    spot::Float64,
    tte::Float64,
    n::Int #steps per year
)
    dt = 1.0 / n

    lnspot = log(spot)
    lnf0 = logForward(model, lnspot, tte)
    dimt = Int(floor(n * tte))
    genTimes = collect(range(0.0, stop=tte, length=(1 + dimt)))

    dimz = dimt * 2
    z = Vector{Float64}(undef, dimz)

    logpathValues = zeros(dimt + 1)
    sqrtdt = sqrt(dt)
    t0 = genTimes[1]
    nextn!(rng, z)
    logpathValues[1] = lnf0
    yi = model.y0
    for i = 1:dimt
        t1 = genTimes[i+1]
        h = t1 - t0
        lnf1 = logForward(model, lnspot, t1)
        vi = model.σ * exp(yi)
        logpathValues[i+1] = logpathValues[i] + lnf1 - lnf0 - vi^2 * h / 2 + vi * z[2i-1] * sqrt(h)
        yi += -model.γ * yi * h + model.θ * z[2i] * sqrt(h)
        t0 = t1
        lnf0 = lnf1
    end

    return genTimes, logpathValues

end

using AQFED.TermStructure
function simulatePath(
    rng,
    model::TermStructure.HestonModel{Float64},
    spot::Float64,
    tte::Float64,
    n::Int
)
    dt = 1.0 / n
    lnspot = log(spot)
    lnf0 = MonteCarlo.logForward(model, lnspot, tte)
    dimt = Int(floor(n * tte))
    genTimes = collect(range(0.0, stop=tte, length=(1 + dimt)))
    xsi = Vector{Float64}(undef, dimt)
    u2 = Vector{Float64}(undef, dimt)
    logpathValues = zeros(dimt + 1)
    v = zeros(dimt + 1)

    t0 = genTimes[1]
    lnspot = log(spot)
    lnf0 = MonteCarlo.logForward(model, lnspot, t0)
    logpathValues[1] = lnf0
    v[1] = model.v0
    ρBar = sqrt(1 - model.ρ^2)
    nextn!(rng, xsi)
    next!(rng, u2)
    for i = 1:dimt
        t1 = genTimes[i+1]
        h = t1 - t0
        ekdth = exp(-model.κ * h / 2)
        c0 = model.θ * h / 4
        c1 = (ekdth - 1) / (model.κ * 2)
        c2 = model.θ * (1 - ekdth)

        vTmp = (v[i] * ekdth + c2) / model.σ^2
        yr = if vTmp > 1.896 * h  #limit for y1 > 0 => 1.89564392373896
            MonteCarlo.dvss2_case1(vTmp, u2[i], h)
        else
            MonteCarlo.dvss2_case2(vTmp, u2[i], h)
        end
        xb = logpathValues[i] - c0 + c1 * (v[i] - model.θ)
        xb += model.σ * (ρBar * xsi[i] * sqrt(h) * sqrt((vTmp + yr) / 2) + model.ρ * (yr - vTmp))
        yr = model.σ^2 * yr
        lnf1 = MonteCarlo.logForward(model, lnspot, t1)
        logpathValues[i+1] = lnf1 - lnf0 + xb - c0 + c1 * (yr - model.θ)
        v[i+1] = yr * ekdth + c2
        t0 = t1
        lnf0 = lnf1
    end
    return genTimes, logpathValues, v

end

using CharFuncPricing
function simulatePath(rng, params::CharFuncPricing.CGMYParams, spot::Float64, tte::Float64, n::Int)
    dt = 1.0 / n
    lnspot = log(spot)
    dimt = Int(floor(n * tte))
    genTimes = collect(range(0.0, stop=tte, length=(1 + dimt)))
    u = zeros(Float64, dimt)

    cf = DefaultCharFunc(params)
    cfPricer = makeCosCharFuncPricer(cf, dt, 1024 * 32, 16)
    logxmin = -sqrt(dt) * 12
    logxmax = -logxmin
    x = collect(range(logxmin, stop=logxmax, length=1024*4))
    y = exp.(x)
    cumDist = map(xi -> priceDigital(cfPricer, false, xi, 1.0, dt, 1.0), y)
    sort!(cumDist) #just in case there is some numerical error
    println(cumDist)
    lp = zeros(dimt + 1)
    lp[1] = lnspot
    next!(rng, u)
    logpathValue = lnspot
    for i = 1:dimt
        l = searchsortedlast(cumDist, u[i])
        increment = if l <= 1
            x[1]
        elseif l >= length(x)
            x[end]
        else
            (u[i] * (x[l+1] - x[l]) + x[l] * cumDist[l+1] - x[l+1] * cumDist[l]) / (cumDist[l+1] - cumDist[l])
        end
        lp[i+1] = lp[i] + increment
    end
    return genTimes, lp
end

struct SVCJModel
    heston::TermStructure.HestonModel{Float64}
    jumpIntensity::Float64
    μBar::Float64
    sjumpVariance::Float64
    vjumpMean::Float64
    jumpCorrelation::Float64
end

using Distributions
function simulatePath(
    rng,
    jmodel::SVCJModel,
    spot::Float64,
    tte::Float64,
    n::Int
)
    model = jmodel.heston
    dt = 1.0 / n
    lnspot = log(spot)
    lnf0 = MonteCarlo.logForward(model, lnspot, tte)
    dimt = Int(floor(n * tte))
    genTimes = collect(range(0.0, stop=tte, length=(1 + dimt)))
    # xsi = Vector{Float64}(undef, dimt)
    # u2 = Vector{Float64}(undef, dimt)
    logpathValues = zeros(dimt + 1)
    v = zeros(dimt + 1)

    t0 = genTimes[1]
    lnspot = log(spot)
    lnf0 = MonteCarlo.logForward(model, lnspot, t0)
    logpathValues[1] = lnf0
    v[1] = model.v0
    v0 = v[1]
    ρBar = sqrt(1 - model.ρ^2)
    sjumpMean = log((1 + jmodel.μBar) * (1 - jmodel.jumpCorrelation * jmodel.vjumpMean)) - jmodel.sjumpVariance / 2
    # sjumpMean = exp(jmodel.μBar + jmodel.sjumpVariance/2) / (1- jmodel.jumpCorrelation*jmodel.vjumpMean) -1
    # nextn!(rng, xsi)
    # next!(rng, u2)
    #pdist = Poisson(jmodel.jumpIntensity*dt)
    #edist = Exponential(1/jmodel.jumpIntensity)
    #evdist = Exponential(jmodel.vjumpMean)
    #simulate Yi, => jump at time Y(0), then jump at time Y(0)+Y(1),...
    lp0 = logpathValues[1]
    i = 1
    tNextJump = 0.0# quantile(edist, rand(rng,Float64))
    while i < dimt
        tNextJump += -log(1 - rand(rng, Float64)) / (jmodel.jumpIntensity)
        while i <= dimt && tNextJump > genTimes[i+1]
            xsii = norminv(rand(rng, Float64))
            u2i = rand(rng, Float64)
            t1 = genTimes[i+1]
            h = t1 - t0
            ekdth = exp(-model.κ * h / 2)
            c0 = model.θ * h / 4
            c1 = (ekdth - 1) / (model.κ * 2)
            c2 = model.θ * (1 - ekdth)

            vTmp = (v0 * ekdth + c2) / model.σ^2
            yr = if vTmp > 1.896 * h  #limit for y1 > 0 => 1.89564392373896
                MonteCarlo.dvss2_case1(vTmp, u2i, h)
            else
                MonteCarlo.dvss2_case2(vTmp, u2i, h)
            end
            xb = lp0 - c0 + c1 * (v0 - model.θ)
            xb += model.σ * (ρBar * xsii * sqrt(h) * sqrt((vTmp + yr) / 2) + model.ρ * (yr - vTmp))
            yr = model.σ^2 * yr
            lnf1 = MonteCarlo.logForward(model, lnspot, t1)

            lp1 = lnf1 - lnf0 + xb - c0 + c1 * (yr - model.θ)
            lp1 -= jmodel.jumpIntensity * jmodel.μBar * h
            v1 = yr * ekdth + c2
            v[i+1] = v1
            logpathValues[i+1] = lp1 #- jmodel.jumpIntensity*jmodel.μBar*genTimes[i+1]
            # println("ti+1 ",t1, " ",i, " ",dimt, " ",length(genTimes))
            lp0 = lp1
            v0 = v1
            t0 = t1
            lnf0 = lnf1
            i += 1
        end
        if i == dimt + 1
            break
        end
        # println("tnextjump ",tNextJump)
        # we have tNextJump < genTimes[i+1]       
        #same as above but until tNextJump
        xsii = norminv(rand(rng, Float64))
        u2i = rand(rng, Float64)
        t1 = tNextJump
        h = t1 - t0
        ekdth = exp(-model.κ * h / 2)
        c0 = model.θ * h / 4
        c1 = (ekdth - 1) / (model.κ * 2)
        c2 = model.θ * (1 - ekdth)

        vTmp = (v0 * ekdth + c2) / model.σ^2
        yr = if vTmp > 1.896 * h  #limit for y1 > 0 => 1.89564392373896
            MonteCarlo.dvss2_case1(vTmp, u2i, h)
        else
            MonteCarlo.dvss2_case2(vTmp, u2i, h)
        end
        xb = lp0 - c0 + c1 * (v0 - model.θ)
        xb += model.σ * (ρBar * xsii * sqrt(h) * sqrt((vTmp + yr) / 2) + model.ρ * (yr - vTmp))
        yr = model.σ^2 * yr
        lnf1 = MonteCarlo.logForward(model, lnspot, t1)
        lp1 = lnf1 - lnf0 + xb - c0 + c1 * (yr - model.θ) - jmodel.jumpIntensity * jmodel.μBar * h
        v1 = yr * ekdth + c2
        #now jump
        ξv = -log(1 - rand(rng, Float64)) * jmodel.vjumpMean
        v1 += ξv
        σξ = sqrt(jmodel.sjumpVariance)
        logξs = sjumpMean + jmodel.jumpCorrelation * ξv + σξ * norminv(rand(rng, Float64))
        lp1 += logξs
        lp0 = lp1
        v0 = v1
        t0 = t1
        lnf0 = lnf1
    end
    return genTimes, logpathValues, v

end


## blog post: roughness of jumps. SVCJ model. Not so trivial to price with charfunc, even though charfunc is known. Duffie pricing formula has issues with oscillating integrals, price is wrong. Cos leads to bad prices, regardless of L. AdaptFilon works (although auto interval has some issues with very low strikes).
##            roughness measure on V.
##            issue measuring realized vol roughness. (other blog post)
# note to me: sim still has some issue with martingality, not sure why. Does not seem realted to dvss but to jump process.
####
#  cf = AQFED.Rough.SVCJCharFunc(modelj)
# pricer1k = CharFuncPricing.AdaptiveFilonCharFuncPricer(cf,1.0,qTol=1e-8)
#AdaptiveFilonCharFuncPricer{Float64}(1.0, [0.0 0.00196078431372549 … 383.0 511.0; 3.928136154635811 3.928075606707497 … -7.449574726414927e-28 -6.280996469727867e-36; 0.0 0.0002466959296861222 … -7.107415457954845e-28 4.250325240225424e-36], 0.0, 3.141592653589793)
#
#julia> priceEuropean(pricer1k, true, 100.0, 100.0,1.0,1.0)
#6.86187552821022
# ,myTrans=CharFuncPricing.IdentityTransformation(0.0,160.0))   #=> price ok with strike = 0.1, 1.0, 5.0. Otherwise default trans => bad price. Look at why!
#
#yK+1-yK  = sum(1:2^(N-n-2)*k) - sum(1:2^(N-n-2)*(k-1))
#=
n=16
N=n+6
dw = randn(2^N) ./ sqrt(2^N)
w = zeros(2^N+1)
for k=1:length(w)-1
    w[k+1]=w[k]+dw[k]
end
y = zeros(2^(n+2)+1)
for k=1:length(y)-1
    y[k+1] = y[k] + sum(w[(k-1)*2^(N-n-2)+1:k*2^(N-n-2)])
end
y ./= 2^N

 t,lp,vh = AQFED.Rough.simulateHybridPath(prng, modelRB, 1.0, 1.0,2^24)
vhi = zeros(Int(trunc(length(vh)/16)));
for i=1:length(vhi)-1
vhi[i+1] = vhi[i]+sum(vh[(i-1)*16+1:i*16])
end
r = AQFED.Rough.estimateRHanSchied(vhi,vhi)
rs = AQFED.Rough.estimateRSHanSchied(r,α=ones(Float64,5)) #recovers vh roughness

tr,rv = AQFED.Rough.realizedVolatilityCum(t,lp,1.0/2^(24),16)
lpp = zeros(Int(trunc(length(lp))));
rvv =  zeros(Int(trunc(length(rv))));
ns=16
for s=1:ns
 t,lp,vh = AQFED.Rough.simulateHybridPath(prng, modelRB, 1.0, 1.0,2^24)
 AQFED.Rough.realizedVolatilityCum(t,lp,1.0/2^(24),16)
 lpp .+= lp
  tr,rv = AQFED.Rough.realizedVolatilityCum(t,lp,1.0/2^(24),16)
  rvv .+= rv
end
lpp ./= ns
rvv ./= ns
=#

#=

plot(tr[1:16:end],vhr[1:16:end],label="Rough Heston")
plot!(th[1:16:end],vhh[1:16:end],xlab="Time",ylab="Instantaneous variance",label="Heston")
plot!(t[1:16:end],vh[1:16:end],label="SVCJ")
=#