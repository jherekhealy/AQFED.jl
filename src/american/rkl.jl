using PPInterpolation
import AQFED.TermStructure:
    TSBlackModel, varianceByLogmoneyness, discountFactor, logForward, forward

#Runge-Kutta-Legendre FDM for the American option.

function makeFDMPriceInterpolation(isCall, isEuropean, model, T, strike, N, M; method = "RKL2", sDisc = "Sinh", useExponentialFitting = false, smoothing = "Kreiss", lambdaS = 0.25, Xdev = 4, Smax = 0, rklStages = 0, epsilonRKL = 0, useSqrt = true)
    upwindingThreshold = 1.0
    v0 = varianceByLogmoneyness(model, 0.0, T)
    Xspan = Xdev * sqrt(v0 * T)
    lnfK = logForward(model, log(strike), T)
    Xmin = lnfK - Xspan - 0.5 * v0 * T
    Xmax = lnfK + Xspan - 0.5 * v0 * T
    Smin = exp(Xmin)
    if Smax == 0
        Smax = exp(Xmax)
    end
    X = collect(range(Xmin, stop = Xmax, length = M))
    hm = X[2] - X[1]
    Sscale = strike * lambdaS
    if sDisc == "Exp"
        S = exp.(X)
        J = exp.(X)
        Jm = @. exp(X - hm / 2)
    elseif sDisc == "Sinh"
        u = collect(range(0.0, stop = 1.0, length = M))
        Smin = 0.0
        c1 = asinh((Smin - strike) / Sscale)
        c2 = asinh((Smax - strike) / Sscale)
        S = @. strike + Sscale * sinh((c2 - c1) * u + c1)
        hm = u[2] - u[1]
        J = @. Sscale * (c2 - c1) * cosh((c2 - c1) * u + c1)
        Jm = @. Sscale * (c2 - c1) * cosh((c2 - c1) * (u - hm / 2) + c1)
    else #if sDisc == "Linear"
        S = collect(range(0.0, stop = Smax, length = M))
        X = S
        hm = X[2] - X[1]
        J = ones(M)
        Jm = ones(M)
    end
    Smin = S[1]
    sign = 1
    if !isCall
        sign = -1
    end

    F0 = zeros(M)
    if isCall
        F0 = @. max(S - strike, 0.0)
    else
        F0 = @. max(strike - S, 0.0)
    end
    iStrike = searchsortedfirst(S, strike)

    if smoothing == "Averaging"
        if strike < (S[iStrike] + S[iStrike-1]) / 2
            iStrike -= 1
        end
        a = (S[iStrike] + S[iStrike+1]) / 2
        if !isCall
            a = (S[iStrike] + S[iStrike-1]) / 2   # int a,lnK K-eX dX = K(a-lnK)+ea-K
        end
        value = (a - strike) * (a - strike) * 0.5
        F0[iStrike] = value / (S[iStrike+1] - S[iStrike-1]) * 2
    elseif smoothing == "Kreiss"
        xmk = S[iStrike] - strike
        h = (S[iStrike+1] - S[iStrike-1]) / 2
        # F0smooth[iStrike] = 0.5*xmk + h/6 + 0.5*xmk*xmk/h*(1-xmk/(3*h))
        F0[iStrike] = 0.5 * xmk + h / 6 + 0.5 * xmk * xmk / h * (1 - xmk / (3 * h))
        if !isCall
            F0[iStrike] -= xmk # C-P = f-K
        end
        iStrike -= 1
        xmk = S[iStrike] - strike
        h = (S[iStrike+1] - S[iStrike-1]) / 2
        F0[iStrike] = 0.5 * xmk + h / 6 + 0.5 * xmk * xmk / h * (1 + xmk / (3 * h))
        if !isCall
            F0[iStrike] -= xmk # C-P = f-K
        end
    end
    F = zeros(M)
    F[1:M] = F0

    lowerBoundA = zeros(M)
    if !isEuropean
        lowerBoundA .= F
    end
    ti = T
    dt = ti / N
    if useSqrt
        ti = 0.0
        dt = sqrt(T) / N
    end
    useDirichlet = false
    lbValue = computeLowerBoundary(isCall, strike, useDirichlet, 0.0, Smin, useSqrt ? ti^2 : ti, model)
    updatePayoffExplicitTrans(F, useDirichlet, lbValue, M)
    A1ilj = zeros(M)
    A1ij = zeros(M)
    A1iuj = zeros(M)
    if method == "RKL2"
        makeSystem(model, A1ilj, A1ij, A1iuj, useExponentialFitting, upwindingThreshold, dt, S, J, Jm, hm, useDirichlet, M)
        if useSqrt
            tih = ti + dt / 2
            @. A1ij *= 2tih
            @. A1ilj *= 2tih
            @. A1iuj *= 2tih
        end
        s, a, b, w0, w1 = initRKLCoeffs(dt, A1ij, epsilonRKL = epsilonRKL, rklStages = rklStages)
        Y0 = zeros(M)
        Y1 = zeros(M)
        Y2 = zeros(M)
        for n = 1:N
            tih = useSqrt ? ti + dt / 2 : ti - dt / 2
            lbValue = computeLowerBoundary(isCall, strike, useDirichlet, 0.0, Smin, useSqrt ? T - tih^2 : tih, model)
            F .= RKLStep(s, a, b, w0, w1, A1ilj, A1ij, A1iuj, F, Y0, Y1, Y2, useDirichlet, lbValue, lowerBoundA, M)
            if useSqrt
                ti += dt
                #  println(ti^2, " ", dt)
            else
                ti -= dt
            end

            if n < N
                makeSystem(model, A1ilj, A1ij, A1iuj, useExponentialFitting, upwindingThreshold, dt, S, J, Jm, hm, useDirichlet, M)
                if useSqrt
                    tih = ti + dt / 2
                    @. A1ij *= 2tih
                    @. A1ilj *= 2tih
                    @. A1iuj *= 2tih
                    s, a, b, w0, w1 = initRKLCoeffs(dt, A1ij, epsilonRKL = epsilonRKL, rklStages = rklStages)
                end
            end
        end
    end
    spl = makeCubicPP(S, F, PPInterpolation.SECOND_DERIVATIVE, 0.0, PPInterpolation.SECOND_DERIVATIVE, 0.0, PPInterpolation.VanLeer())
    return spl
end


function initRKLCoeffs(dt, A1ij; epsilonRKL = 0.0, rklStages = 0)
    dtexplicit = dt / max(maximum(A1ij))
    dtexplicit /= 2 #lambdaS        
    s = 0.0
    delta = 1 + 4 * (2 + 4 * dt / dtexplicit)
    s = ceil(Int, (-1 + sqrt(delta)) / 2)
    if s % 2 == 0
        s += 1
    end
    if epsilonRKL > 0
        s = computeRKLStages(dtexplicit, dt, epsilonRKL)
    end
    if rklStages > 0
        s = rklStages
    end
    # println("s=",s)
    a = zeros(s)
    b = zeros(s)
    w0 = 1.0
    w1 = 0.0
    if epsilonRKL == 0
        w1 = 4 / (s^2 + s - 2)
        b[1] = 1.0 / 3
        b[2] = 1.0 / 3
        a[1] = 1.0 - b[1]
        a[2] = 1.0 - b[2]
        for i = 3:s
            b[i] = (i^2 + i - 2.0) / (2 * i * (i + 1.0))
            a[i] = 1.0 - b[i]
        end
    else
        w0 = 1 + epsilonRKL / s^2
        _, tw0p, tw0p2 = legPoly(s, w0)
        w1 = tw0p / tw0p2
        b = zeros(s)
        for jj = 2:s
            _, tw0p, tw0p2 = legPoly(jj, w0)
            b[jj] = tw0p2 / tw0p^2
        end
        b[1] = b[2]
        a = zeros(s)
        for jj = 2:s
            tw0, _, _ = legPoly(jj - 1, w0)
            a[jj-1] = (1 - b[jj-1] * tw0)
        end
    end
    return s, a, b, w0, w1
end

function makeSystem(model, A1ilj::AbstractArray{T}, A1ij::AbstractArray{T}, A1iuj::AbstractArray{T}, useExponentialFitting::Bool, upwindingThreshold::Real, dt::Real, S::Vector{T}, J::Vector{T}, Jm::Vector{T}, hm::Real, useDirichlet::Bool, M::Int) where {T}
    if useDirichlet
        A1ij[1] = 0
    else
        drifti = (model.r - model.q) * S[1]
        A1iuj[1] = -dt * drifti / (Jm[2] * hm)
        A1ij[1] = -dt * (-model.r * 0.5) - A1iuj[1]
    end
    drifti = (model.r - model.q) * S[M]
    A1ilj[M] = dt * drifti / (Jm[M] * hm)
    A1ij[M] = -dt * (-model.r * 0.5) - A1ilj[M]

    @inbounds @simd for i = 2:M-1
        svi = S[i]^2 * model.vol^2 / J[i]
        drifti = (model.r - model.q) * S[i]
        if useExponentialFitting
            if abs(drifti * hm / svi) > upwindingThreshold
                svi = drifti * hm / tanh(drifti * hm / svi)
            end
        end
        svi /= hm^2
        drifti /= (2 * J[i] * hm)
        A1iuj[i] = -dt * (svi / (2Jm[i+1]) + drifti)
        A1ij[i] = -dt * (-svi / 2 * (1 / Jm[i+1] + 1 / Jm[i]) - model.r)
        A1ilj[i] = -dt * (svi / (2Jm[i]) - drifti)
    end
end

function explicitStep(a::Real, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, F::Vector{T}, Y0::Vector{T}, Y1::Vector{T}, M::Int) where {T}
    index = 1
    Y1[index] = Y0[index] - a * (A1ij[index] * F[index] + A1iuj[index] * F[index+1])
    index = M
    Y1[index] = Y0[index] - a * (A1ij[index] * F[index] + A1ilj[index] * F[index-1])
    @inbounds @simd for index = 2:M-1
        Y1[index] = Y0[index] - a * (A1ij[index] * F[index] + A1iuj[index] * F[index+1] + A1ilj[index] * F[index-1])
    end
end

function RKLStep(s::Int, a::Vector{T}, b::Vector{T}, w0::Real, w1::Real, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, F::Vector{T}, Yjm2::Vector{T}, Yjm::Vector{T}, Yj::Vector{T}, useDirichlet::Bool, lbValue::Real, lowerBound::Vector{T}, M::Int) where {T}
    mu1b = b[1] * w1
    explicitStep(mu1b, A1ilj, A1ij, A1iuj, F, F, Yjm, M)
    updatePayoffExplicitTrans(Yjm, useDirichlet, lbValue, M)
    MY0 = (Yjm - F) / mu1b
    enforceLowerBound(Yjm, lowerBound, M)
    Yjm2 .= F
    for j = 2:s
        muu = (2 * j - 1) * b[j] / (j * b[j-1])
        muj = muu * w0
        mujb = muu * w1
        gammajb = -a[j-1] * mujb
        nuj = -1.0 * b[2] / (2.0 * b[1]) #b0 = b[1]
        if j > 2
            nuj = -(j - 1) * b[j] / (j * b[j-2])
        end
        @. Yj = muj * Yjm + nuj * Yjm2 + (1 - nuj - muj) * F + gammajb * MY0 # + mujb*MYjm
        explicitStep(mujb, A1ilj, A1ij, A1iuj, Yjm, Yj, Yj, M)
        updatePayoffExplicitTrans(Yj, useDirichlet, lbValue, M)
        enforceLowerBound(Yj, lowerBound, M)
        Yjm2, Yjm = Yjm, Yjm2
        Yjm, Yj = Yj, Yjm
    end
    return Yjm
end


function enforceLowerBound(F::Vector{T}, lowerBound::Vector{T}, M::Int) where {T}
    if length(lowerBound) > 0
        @. F = max(F, lowerBound)
    end
end



function computeLowerBoundary(isCall::Bool, strike::Real, useDirichlet::Bool, B::Real, Smin::Real, ti::Real, model)
    lbValue = 0.0
    if useDirichlet && B == 0
        if !isCall
            lbValue = (strike - forward(model, Smin, ti) * discountFactor(model, ti))
        end
    end
    return lbValue
end

function updatePayoffExplicitTrans(F::Vector{T}, useDirichlet::Bool, lbValue::Real, M::Int) where {T}
    if useDirichlet
        F[1] = lbValue
    end
end

function computeRKLStages(dtexplicit, dt, ep)
    s = 1
    betaFunc = function (s::Int)
        w0 = 1 + ep / s^2
        _, tw0p, tw0p2 = legPoly(s, w0)
        beta = (w0 + 1) * tw0p2 / tw0p
        return beta - 2 * dt / (dtexplicit)
    end
    while s < 10000 && betaFunc(s) < 0
        s += 1
    end
    #s += Int(ceil(s/10))
    return s
end

function legPoly(s::Int, w0::Real)
    tjm = 1.0
    tj = w0
    if s == 1
        return tj, 1.0, 0.0
    end
    dtjm = 0.0
    dtj = 1.0
    d2tjm = 0.0
    d2tj = 0.0

    for j = 2:s
        onej = 1.0 / j
        tjp = (2 - onej) * w0 * tj - (1 - onej) * tjm
        dtjp = (2 - onej) * (tj + w0 * dtj) - (1 - onej) * dtjm
        d2tjp = (2 - onej) * (dtj * 2 + w0 * d2tj) - (1 - onej) * d2tjm
        tjm = tj
        dtjm = dtj
        d2tjm = d2tj
        tj = tjp
        dtj = dtjp
        d2tj = d2tjp
    end
    return tj, dtj, d2tj
end
