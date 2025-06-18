#Benhamou Gobet Miri approximation for Heston 
import CharFuncPricing: HestonParams, HestonTSParams
using ForwardDiff
struct BGMApprox{T}
    params::T
end


function priceEuropean(p::BGMApprox{T}, isCall::Bool, strike, forward, tte, df) where {T<:HestonTSParams}
    ρ = p.params.ρ
    κ = p.params.κ[1]
    σ = p.params.σ
    θ = p.params.θ
    tteIndex = searchsortedfirst(p.params.startTime, tte)
    ts = vcat(p.params.startTime[1:tteIndex-1], tte)
    NT = length(ts) - 1
	wmk = zeros(NT)
    w0mk = zeros(NT)
    wmkk = zeros(NT)
    for i in 1:NT
        wmk[i] = -(exp(-κ * ts[i+1]) - exp(-κ * ts[i])) / κ
        w0mk[i] = (exp(-κ * ts[i+1]) * (ts[i] * κ - ts[i+1] * κ - 1) + exp(-κ * ts[i])) / (κ^2)
        wmkk[i] = (exp(-κ * ts[i]) - exp(-κ * ts[i+1]))^2 / (2 * κ^2)
    end

    a1T = 0.0
    a2T = 0.0
    b0T = 0.0
    w1T = 0.0
    w2T = 0.0
	alphaT = 0.0
	betaT = 0.0
    v0T = p.params.v0

    for i in 1:NT
        # f0T
        f0T = exp(-2.0 * κ * ts[i+1]) * (exp(2.0 * κ * ts[i]) * (θ[i] - 2.0 * v0T) +
                                         exp(2.0 * κ * ts[i+1]) * ((-2.0 * κ * ts[i] + 2.0 * κ * ts[i+1] - 5.0) * θ[i] + 2.0 * v0T) +
                                         4.0 * exp(κ * (ts[i] + ts[i+1])) * ((-κ * ts[i] + κ * ts[i+1] + 1.0) * θ[i] - κ * (ts[i+1] - ts[i]) * v0T)) / (4 * κ^3)
        # f1T
        f1T = exp(-κ * ts[i+1]) * (exp(κ * ts[i+1]) * ((-κ * ts[i] + κ * ts[i+1] - 2.0) * θ[i] + v0T) -
                                   exp(κ * ts[i]) * ((κ * ts[i] - κ * ts[i+1] - 2.0) * θ[i] - κ * ts[i] * v0T +
                                                     κ * ts[i+1] * v0T + v0T)) / (κ^2)
        # h1T
        h1T = (exp(κ * ts[i+1]) * θ[i] +
               exp(κ * ts[i]) * ((κ * ts[i] - κ * ts[i+1] - 1.0) * θ[i] + κ * (ts[i+1] - ts[i]) * v0T)) / κ
        # f2T
        f2T = exp(-κ * (ts[i] + 3 * ts[i+1])) * (2.0 * exp(κ * (ts[i] + 3 * ts[i+1])) * ((κ * (ts[i+1] - ts[i]) - 3.0) * θ[i] + v0T) +
                                                 exp(2.0 * κ * (ts[i] + ts[i+1])) * ((κ * (κ * (ts[i] - ts[i+1]) - 4.0) * (ts[i] - ts[i+1]) + 6.0) * θ[i] -
                                                                                     (κ * (κ * (ts[i] - ts[i+1]) - 2.0) * (ts[i] - ts[i+1]) + 2.0) * v0T)) / (2 * κ^3)
        # g1T
        g1T = (2.0 * exp(κ * ts[i+1]) * θ[i] +
               exp(κ * ts[i]) * ((κ^2 * (ts[i] - ts[i+1])^2 * v0T - (κ * (κ * (ts[i] - ts[i+1]) - 2.0) * (ts[i] - ts[i+1]) + 2.0) * θ[i]))) / (2 * κ^2)
        # g2T
        g2T = exp(-κ * ts[i+1]) * (exp(2.0 * κ * ts[i+1]) * θ[i] -
                                   exp(2.0 * κ * ts[i]) * (θ[i] - 2 * v0T) + 2.0 * exp(κ * (ts[i] + ts[i+1])) * ((κ * (ts[i] - ts[i+1]) * (θ[i] - v0T) - v0T))) / (2 * κ^2)
        # h2T
        h2T = (exp(κ * ts[i]) - exp(κ * ts[i+1])) * (exp(κ * ts[i]) * (θ[i] - 2.0 * v0T) - exp(κ * ts[i+1]) * θ[i]) / (2 * κ)

        a1T += wmk[i] * w1T + ρ[i] * σ[i] * f1T
        a2T += wmk[i] * alphaT + ρ[i] * σ[i] * w0mk[i] * w1T + ρ[i]^2 * σ[i]^2 * f2T
        b0T += wmk[i] * betaT + wmkk[i] * w2T + σ[i]^2 * f0T
        alphaT += ρ[i] * σ[i] * (ts[i+1] - ts[i]) * w1T + ρ[i]^2 * σ[i]^2 * g1T
        betaT += wmk[i] * w2T + σ[i]^2 * g2T
        w1T += ρ[i] * σ[i] * h1T
        w2T += σ[i]^2 * h2T
        v0T = exp(-κ * (ts[i+1] - ts[i])) * (v0T - θ[i]) + θ[i]
    end
	b2T = a1T^2 / 2
    varT = totalVariance(p.params.v0, κ, θ, ts)
    x = log(forward / strike)
    y = varT
    g = -x / sqrt(y) - sqrt(y) / 2 #g=-d1
    f = -x / sqrt(y) + sqrt(y) / 2 #f = -d2
    PHIf = normcdf(f)
    PHIg = normcdf(g)
    bsPut = strike * PHIf - forward * PHIg

    phif = normpdf(f)
    phig = normpdf(g)

    # Derivatives of f and g
    fx = -1 / sqrt(y)
    fy = -g / (2y)
    gx = fx
    gy = -f / (2y)

    # Derivatives of the pdf phi(f)
    phifx = f * phif / sqrt(y)
    phify = f * g * phif / (2y) # -fy* f*phif = g*f*phif/2y

    # Derivatives of the cdf PHI(f)
    PHIfxy = y^(-1.5) * phif * (1 - f * g) / 2
    PHIfx2y = y^(-2) * phif * (2 * f + g - f * f * g) / 2
    PHIfy2 = y^(-2) * phif * (g + f / 2 - f * g * g / 2) / 2
    PHIfx2y2 = ((y^(-2) * phify - 2 * y^(-3) * phif) * (2 * f + g - f * f * g) + y^(-2) * phif * (2 * fy + gy - 2 * f * fy * g - f * f * gy)) / 2

    # Derivatives of the pdf phi(g)
    phigx = g * phig / sqrt(y)
    phigy = f * g * phig / (2y)

    # Derivatives of cdf PHI(g)
    PHIgx = -phig / sqrt(y)
    PHIgy = -f * phig / y / 2
    PHIgxy = y^(-1.5) * phig * (1 - f * g) / 2
    PHIgx2y = y^(-2) * phig * (2 * g + f - g * g * f) / 2
    PHIgy2 = y^(-2) * phig * (f + g / 2 - g * f * f / 2) / 2
    PHIgxy2 = -1.5 * y^(-2.5) * phig * (1 - f * g) / 2 + y^(-1.5) * phigy * (1 - f * g) / 2 - y^(-1.5) * phig * fy * g / 2 - y^(-1.5) * phig * f * gy / 2
    #this has errors in Rouah's code: y^(-2) * (phigx * (f + g / 2 - f * f * g / 2) + phig * (fx + gx / 2 - f * fx * g / 2 - f * f * gx / 2)) / 2
    PHIgx2y2 = ((y^(-2) * phigy - 2 * y^(-3) * phig) * (2 * g + f - g * g * f) + y^(-2) * phig * (2 * gy + fy - 2 * g * gy * f - g * g * fy)) / 2

    # Derivatives of Black-Scholes Put
    #dpdx = - PHIg * dforwarddx = -PHIg * forward  #forward = strike*ex
    dPdxdy = strike * PHIfxy - forward * (PHIgy + PHIgxy) #
    dPdx2dy = strike * PHIfx2y - forward * (PHIgy + 2 * PHIgxy + PHIgx2y)
    dPdy2 = strike * PHIfy2 - forward * PHIgy2
    dPdx2dy2 = strike * PHIfx2y2 - forward * (PHIgy2 + 2 * PHIgxy2 + PHIgx2y2) #Also fixed compared to Rouah's code.

    # Benhamou, Gobet, Miri expansion
    putPrice = bsPut + a1T * dPdxdy + a2T * dPdx2dy + b0T * dPdy2 + b2T * dPdx2dy2

    return if isCall
        df * (putPrice - strike + forward)
    else
        df * putPrice
    end
end

function totalVariance(v0, κ, θ, ts)
    NT = length(ts) - 1
    sum = v0 * (1 - exp(-κ * ts[NT+1])) / κ
    for i in 1:NT
        t0 = ts[i]
        t1 = ts[i+1]
        for j=1:(i-1)
			sum +=  θ[j] * (exp(κ * ts[j+1]) - exp(κ * ts[j])) * (-exp(-κ * ts[i+1]) + exp(-κ * ts[i])) / κ 
		end
        sum += θ[i] * (t1 - t0 - (1 - exp(-κ * (t1 - t0))) / κ)
    end
    return sum
end

function priceEuropean(p::BGMApprox{T}, isCall::Bool, strike, forward, tte, df) where {T<:HestonParams}
    v0 = p.params.v0
    θ = p.params.θ
    σ = p.params.σ
    ρ = p.params.ρ
    κ = p.params.κ
    kt = κ * tte
    emkt = exp(-kt)
    ekt = 1 / emkt
    m0 = emkt * (-1 + ekt) / κ
    m1 = tte - m0
    p0 = emkt * (-kt + ekt - 1) / κ^2
    p1 = emkt * (kt + ekt * (kt - 2) + 2) / κ^2
    q0 = emkt * (-kt * (kt + 2) + 2ekt - 2) / (2 * κ^3)
    q1 = emkt * (2ekt * (kt - 3) + kt * (kt + 4) + 6) / (2 * κ^3)
    r0 = emkt^2 * (-4ekt * kt + 2 * ekt^2 - 2) / (4 * κ^3)
    r1 = emkt^2 * (4ekt * (kt + 1) + (2kt - 5) * ekt^2 + 1) / (4 * κ^3)
    varT = m0 * v0 + m1 * θ
    a1T = ρ * σ * (p0 * v0 + p1 * θ)
    a2T = (ρ * σ)^2 * (q0 * v0 + q1 * θ)
    b0T = σ^2 * (r0 * v0 + r1 * θ)
    b2T = a1T^2 / 2


    x = log(forward / strike)
    y = varT
    g = -x / sqrt(y) - sqrt(y) / 2 #g=-d1
    f = -x / sqrt(y) + sqrt(y) / 2 #f = -d2
    PHIf = normcdf(f)
    PHIg = normcdf(g)
    bsPut = strike * PHIf - forward * PHIg

    phif = normpdf(f)
    phig = normpdf(g)

    # Derivatives of f and g
    fx = -1 / sqrt(y)
    fy = -g / (2y)
    gx = fx
    gy = -f / (2y)

    # Derivatives of the pdf phi(f)
    phifx = f * phif / sqrt(y)
    phify = f * g * phif / (2y) # -fy* f*phif = g*f*phif/2y

    # Derivatives of the cdf PHI(f)
    PHIfxy = y^(-1.5) * phif * (1 - f * g) / 2
    PHIfx2y = y^(-2) * phif * (2 * f + g - f * f * g) / 2
    PHIfy2 = y^(-2) * phif * (g + f / 2 - f * g * g / 2) / 2
    PHIfx2y2 = ((y^(-2) * phify - 2 * y^(-3) * phif) * (2 * f + g - f * f * g) + y^(-2) * phif * (2 * fy + gy - 2 * f * fy * g - f * f * gy)) / 2

    # Derivatives of the pdf phi(g)
    phigx = g * phig / sqrt(y)
    phigy = f * g * phig / (2y)

    # Derivatives of cdf PHI(g)
    PHIgx = -phig / sqrt(y)
    PHIgy = -f * phig / y / 2
    PHIgxy = y^(-1.5) * phig * (1 - f * g) / 2
    PHIgx2y = y^(-2) * phig * (2 * g + f - g * g * f) / 2
    PHIgy2 = y^(-2) * phig * (f + g / 2 - g * f * f / 2) / 2
    PHIgxy2 = -1.5 * y^(-2.5) * phig * (1 - f * g) / 2 + y^(-1.5) * phigy * (1 - f * g) / 2 - y^(-1.5) * phig * fy * g / 2 - y^(-1.5) * phig * f * gy / 2
    #this has errors in Rouah's code: y^(-2) * (phigx * (f + g / 2 - f * f * g / 2) + phig * (fx + gx / 2 - f * fx * g / 2 - f * f * gx / 2)) / 2
    PHIgx2y2 = ((y^(-2) * phigy - 2 * y^(-3) * phig) * (2 * g + f - g * g * f) + y^(-2) * phig * (2 * gy + fy - 2 * g * gy * f - g * g * fy)) / 2

    # Derivatives of Black-Scholes Put
    #dpdx = - PHIg * dforwarddx = -PHIg * forward  #forward = strike*ex
    dPdxdy = strike * PHIfxy - forward * (PHIgy + PHIgxy) #
    dPdx2dy = strike * PHIfx2y - forward * (PHIgy + 2 * PHIgxy + PHIgx2y)
    dPdy2 = strike * PHIfy2 - forward * PHIgy2
    dPdx2dy2 = strike * PHIfx2y2 - forward * (PHIgy2 + 2 * PHIgxy2 + PHIgx2y2) #Also fixed compared to Rouah's code.

    #     function Put(x,y)   
    #         g = -x / sqrt(y) -  sqrt(y)/2 #g=-d1
    # 	    f = -x / sqrt(y) + sqrt(y)/2 #f = -d2
    # 	    PHIf = normcdf(f)
    # 	    PHIg = normcdf(g)
    #         strike * (PHIf - exp(x) * PHIg)
    #     end
    #     dPdxdyFD = ForwardDiff.derivative(y -> ForwardDiff.derivative(x -> Put(x,y),x),y)
    #     dPdx2dyFD =ForwardDiff.derivative(y -> ForwardDiff.derivative(x ->
    #     ForwardDiff.derivative(x2-> Put(x2,y),x),x),y)
    #     dPdxdy2FD = ForwardDiff.derivative(y -> ForwardDiff.derivative(y2 -> ForwardDiff.derivative(x -> Put(x,y2),x),y),y)
    #     dPdy2FD =  ForwardDiff.derivative(y -> ForwardDiff.derivative(y2 -> Put(x,y2),y),y)
    #     dPdx2dy2FD =ForwardDiff.derivative(y -> ForwardDiff.derivative(y2 -> ForwardDiff.derivative(x ->
    #     ForwardDiff.derivative(x2-> Put(x2,y2),x),x),y),y)
    #  println(dPdx2dy2, " ", dPdx2dy2FD, " ",dPdx2dy2 - dPdx2dy2FD)
    # Benhamou, Gobet, Miri expansion
    putPrice = bsPut + a1T * dPdxdy + a2T * dPdx2dy + b0T * dPdy2 + b2T * dPdx2dy2

    return if isCall
        df * (putPrice - strike + forward)
    else
        df * putPrice
    end
end
