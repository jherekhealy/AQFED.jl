using LinearAlgebra
using LeastSquaresOptim
using GaussNewton
import AQFED.Black: blackScholesFormula, impliedVolatilitySRHalley, Householder
import AQFED.Math: ClosedTransformation, inv
import PPInterpolation: PP, evaluatePiece, evaluateDerivativePiece, evaluateSecondDerivativePiece

struct EQuadraticLVG{TV,T,TC}
    x::Vector{T}
    α::Vector{TV} #αx^2+βx+γ = (a+bx)(1+cx) #encompasses constant, black, linear, linearblack.
    β::Vector{TV}
    γ::Vector{TV}
    θ::Vector{TC}
    tte::T
    forward::T
end

Base.broadcastable(p::EQuadraticLVG) = Ref(p)


function priceEuropean(model::EQuadraticLVG{TV,T,TC}, isCall::Bool, strike::T) where {TV,T,TC}
    x = model.x
    if strike <= x[1] || strike >= x[end]
        return priceEuropeanPiece(0, model, isCall, strike)
    end
    i = max(searchsortedlast(x, strike), 1)
    return priceEuropeanPiece(i, model, isCall, strike)
end

function priceEuropeanPiece(i::Int, model::EQuadraticLVG{TV,T,TC}, isCall::Bool, strike::T) where {TV,T,TC}
    if i <= 0 || i > length(model.α)
        return isCall ? max(model.forward - strike, 0.0) : max(strike - model.forward, 0.0)
    end
    β = model.β[i]
    α = model.α[i]
    γ = model.γ[i]
    xi = model.x[i]
    value = if α == zero(TV) && β == zero(TV)
        zzi = strike - xi
        χ = one(T)
        ω = one(T) / γ
        u = ω * zzi
        v = χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))
        abs(v)
    elseif α == zero(TV)
        δ = β^2
        x1 = -γ / β  # beta (x-x1) = beta x + gamma
        zzi = (strike - x1) / (xi - x1)
        χ = sqrt((strike - x1) / (xi - x1))#sqrt((x-x1)*(x-x2))
        Δ = δ # α^2*(x1-x2)^2
        ω = sqrt(4 / Δ + 1) / 2
        u = ω * log(zzi)
        v = χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))
        abs(v)
    else
        δ = complex(β^2 - 4α * γ)
        x1 = (-β - sqrt(δ)) / (2α)
        x2 = (-β + sqrt(δ)) / (2α)
        zzi = (strike - x1) / (strike - x2) * (xi - x2) / (xi - x1)
        χ = sqrt(complex((strike^2 + β / α * strike + γ / α) / (xi^2 + β / α * xi + γ / α)))#sqrt((x-x1)*(x-x2))
        Δ = δ # α^2*(x1-x2)^2
        ω = sqrt(4 / Δ + 1) / 2
        u = ω * log(zzi)
        v = χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))
        abs(v)
    end
    #println(i," ",x[i]," ",strike," ",χ," ",u)
    if isCall && strike < model.forward
        value += model.forward - strike
    elseif !isCall && strike > model.forward
        value += strike - model.forward
    end
    return value
end

function derivativePrice(model::EQuadraticLVG{TV,T,TC}, isCall::Bool, strike::T) where {TV,T,TC}
    x = model.x
    if strike <= x[1] || strike >= x[end]
        return if isCall && model.forward > strike
            -one(T)
        elseif !isCall && model.forward < strike
            one(T)
        else
            zero(T)
        end
    end
    i = searchsortedfirst(x, strike)  # x[i-1]<z<=x[i]
    if strike != x[i] && i > 1
        i -= 1
    end
    # println(i," ",x[i]," ",strike)
    β = model.β[i]
    α = model.α[i]
    γ = model.γ[i]
    value = value = if α == zero(TV) && β == zero(TV)
        zzi = strike - x[i]
        #χ = one(T)
        ω = one(T) / γ
        u = ω * zzi
        sh, ch = sinh(u), cosh(u)
        real(ω * ((model.θ[2i]) * sh + (model.θ[2i-1]) * ch))
    elseif α == zero(TV)
        δ = β^2
        x1 = -γ / β  # beta (x-x1) = beta x + gamma
        zzi = (strike - x1) / (x[i] - x1)
        χ = sqrt((strike - x1) / (x[i] - x1))#sqrt((x-x1)*(x-x2))
        Δ = δ # α^2*(x1-x2)^2
        ω = sqrt(4 / Δ + 1) / 2
        u = ω * log(zzi)
        dz = one(T) / (strike - x1)
        κ = one(TV) / 2
        sh, ch = sinh(u), cosh(u)
        real(χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) + ω * (model.θ[2i-1])) * ch))
    else
        δ = complex(β^2 - 4α * γ)
        x1 = (-β - sqrt(δ)) / (2α)
        x2 = (-β + sqrt(δ)) / (2α)
        zzi = (strike - x1) / (strike - x2) * (x[i] - x2) / (x[i] - x1)
        χ = sqrt(complex((strike^2 + β / α * strike + γ / α) / (x[i]^2 + β / α * x[i] + γ / α)))#sqrt((x-x1)*(x-x2))
        Δ = δ # α^2*(x1-x2)^2
        ω = sqrt(4 / Δ + 1) / 2
        u = ω * log(zzi)
        #αx^2+βx+γ = (a+bx)(1+cx), α(x-x1)(x-x2)
        #dz = b / (a+b*strike) - c / (1+c*strike)
        dz = one(T) / (strike - x1) - one(T) / (strike - x2)
        # dχ = (b)/sqrt(a+bstrike) * sqrt(1+cstrike) + c/sqrt(1+cstrike)*sqrt(a+bstrike) = X * (b/(a+bstrike)+c/(1+cstrike))
        #  κ = dχ/(χ*dz)
        #κ = (b / (a+b*strike) + c / (1+c*strike)) / (2 * dz)
        κ = (one(T) / (strike - x1) + one(T) / (strike - x2)) / (2dz)
        sh, ch = sinh(u), cosh(u)
        # println(ω ," zzi",zzi," u ",u, " ",ch," ",sh)
        real(χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) + ω * (model.θ[2i-1])) * ch))
    end
    if isCall && strike < model.forward
        value -= 1
    elseif !isCall && strike > model.forward
        value += 1
    end
    return value
end

function computeCoeffsEQuadratic(s::Int, tte::T, strikes::AbstractVector{T}, α::AbstractVector{TC}, β::AbstractVector{TC}, γ::AbstractVector{TC}, θ::AbstractVector{TA};) where {T,TA,TC}
    #in order to price, we must solve a tridiag like system. This can be done once for all strikes. Hence outside price
    TAC = complex(TC)
    tri = Tridiagonal(zeros(TAC, 2(length(strikes) - 1) - 1), zeros(TAC, 2(length(strikes) - 1)), zeros(TAC, 2(length(strikes) - 1) - 1))
    m = length(strikes) - 1
    lhsd = tri.d
    lhsdl = tri.dl
    lhsdu = tri.du
    rhs = zeros(T, 2m)

    #theta0_c = 0
    lhsd[1] = zero(TAC)
    lhsdu[1] = one(TAC)
    rhs[1] = zero(T)
    #a[i] between zi and zip , a[i+1]at zip
    for i = 1:m-1
        local χi, zdi, κi, ωi, omegazipmzi
        if α[i] == zero(TC) && β[i] == zero(TC)
            χi = one(TAC)
            ωi = one(TAC) / γ[i]
            κi = zero(TAC)
            zzi = strikes[i+1] - strikes[i]
            zdi = one(TAC)
            omegazipmzi = ωi * zzi
        elseif α[i] == zero(TC)
            δ = β[i]^2
            x1 = -γ[i] / β[i]  # beta (x-x1) = beta x + gamma
            χi = sqrt((strikes[i+1] - x1) / (strikes[i] - x1))#sqrt((x-x1)*(x-x2))
            zdi = one(TAC) / (strikes[i+1] - x1)
            κi = one(TAC) / 2
            ωi = sqrt(4 / δ + 1) / 2
            zzi = (strikes[i+1] - x1) / (strikes[i] - x1)
            omegazipmzi = ωi * log(zzi)
        else
            #line 2(i-1)+1
            χi = sqrt(complex((strikes[i+1]^2 + β[i] / α[i] * strikes[i+1] + γ[i] / α[i]) / (strikes[i]^2 + β[i] / α[i] * strikes[i] + γ[i] / α[i])))#sqrt((x-x1)*(x-x2))
            δ = complex(β[i]^2 - 4α[i] * γ[i])
            x1 = (-β[i] - sqrt(δ)) / (2α[i])
            x2 = (-β[i] + sqrt(δ)) / (2α[i])
            zdi = one(TAC) / (strikes[i+1] - x1) - one(TAC) / (strikes[i+1] - x2)
            κi = (one(TAC) / (strikes[i+1] - x1) + one(TAC) / (strikes[i+1] - x2)) / (2zdi)
            ωi = sqrt(4 / δ + 1) / 2
            zzi = (strikes[i+1] - x1) / (strikes[i+1] - x2) * (strikes[i] - x2) / (strikes[i] - x1)
            omegazipmzi = ωi * log(zzi)
        end
        local zdip, κip, ωip
        if α[i+1] == zero(TC) && β[i+1] == zero(TC)
            ωip = one(TAC) / γ[i+1]
            κip = zero(TAC)
            zdip = one(TAC)
        elseif α[i+1] == zero(TC)
            δp = β[i+1]^2
            x1p = -γ[i+1] / β[i+1]  # beta (x-x1) = beta x + gamma
            zdip = one(TAC) / (strikes[i+1] - x1p)
            κip = one(TAC) / 2
            ωip = sqrt(4 / δp + 1) / 2
        else
            δp = complex(β[i+1]^2 - 4α[i+1] * γ[i+1])
            x1p = (-β[i+1] - sqrt(δp)) / (2α[i+1])
            x2p = (-β[i+1] + sqrt(δp)) / (2α[i+1])
            zdip = one(TAC) / (strikes[i+1] - x1p) - one(TAC) / (strikes[i+1] - x2p)
            κip = (one(TAC) / (strikes[i+1] - x1p) + one(TAC) / (strikes[i+1] - x2p)) / (2zdip)
            ωip = sqrt(4 / δp + 1) / 2
        end

        coshi = cosh(omegazipmzi)
        sinhi = sinh(omegazipmzi)
        lhsdv = (κi * coshi + ωi * sinhi) * χi * zdi - κip * coshi * χi * zdip 
        lhsd[2i] = lhsdv  #2i
        lhsdl[2i-1] = (ωi * coshi + κi * sinhi) * χi * zdi - κip * sinhi * χi * zdip   #2i-1
        lhsdu[2i] = -ωip * zdip                                      #2i+1
        #line 2(i-1)+2    
        lhsd[2i+1] = -ωip * zdip
        lhsdl[2i] = (κi * coshi + ωi * sinhi - (ωi * coshi + κi * sinhi) / sinhi * coshi) * χi * zdi
        lhsdu[2i+1] = zdi * (ωi * coshi + κi * sinhi) / (sinhi) - κip * zdip
        # tanhi = tanh(omegazipmzi)
        # coshi = cosh(omegazipmzi)
        # lhsd[2i] = (κi  + ωi * tanhi) * χi * zdi - κip  * χi * zdip   #2i
        # lhsdl[2i-1] = (ωi  + κi * tanhi) * χi * zdi - κip * tanhi * χi * zdip   #2i-1
        # lhsdu[2i] = -ωip * zdip  / coshi                                    #2i+1
        # #line 2(i-1)+2    
        # lhsd[2i+1] = -ωip * zdip / coshi
        # lhsdl[2i] = (κi  + ωi * tanhi - (ωi  + κi * tanhi) /tanhi) * χi * zdi 
        # lhsdu[2i+1] = zdi * (ωi/tanhi  + κi )/ coshi  - κip * zdip/ coshi

        # if !isfinite(lhsd[2i]) || !isfinite(lhsdl[2i-1]) || !isfinite(lhsdl[2i]) || !isfinite(lhsdu[2i+1]) 
        #     println(" Na N ",ωi*(zip-zi), " at ",i)
        # end
    end
    rhs[2s] = one(T)
    rhs[2s+1] = one(T)
    local ωm, zzi
    if α[m] == zero(TC) && β[m] == zero(TC)
        ωm = one(TC) / γ[m]
        zzi = strikes[m+1] - strikes[m]
    elseif α[m] == zero(TC)
        δ = β[m]^2
        x1 = -γ[m] / β[m]  # beta (x-x1) = beta x + gamma
        ωm = sqrt(4 / δ + 1) / 2
        zzi = log((strikes[m+1] - x1) / (strikes[m] - x1))
    else
        δ = complex(β[m]^2 - 4α[m] * γ[m])
        x1 = (-β[m] - sqrt(δ)) / (2α[m])
        x2 = (-β[m] + sqrt(δ)) / (2α[m])
        ωm = sqrt(4 / δ + 1) / 2
        zzi = log((strikes[m+1] - x1) / (strikes[m+1] - x2) * (strikes[m] - x2) / (strikes[m] - x1))
    end
    # coshm = cosh(ωm * zzi)
    tanhm = tanh(ωm * zzi)
    # u1=thetaOs, u2=theta0c, u3=theta1s, u4=theta1c    ...
    lhsd[2m] = one(T)
    lhsdl[2m-1] = tanhm
    if isinf(tanhm) 
        println(ωm ," ", zzi," ",β[m]," ",α[m] )
        throw(DomainError(sinhm,"sinhm must be finite"))
    end
    rhs[2m] = zero(T)

    #    println("solving with a=",γ," lhsdl=",lhsdl," lhsd=",lhsd," lhsdu",lhsdu)
    lhsf = lu!(tri)
    ##  lhsf = factorize(Tridiagonal(lhsdl, lhsd, lhsdu))
    ldiv!(θ, lhsf, rhs)
    # println("found solution ",θ) #theta is either full real for full im. Simplif are possible
    # θ[:] = Tridiagonal(lhsdl, lhsd, lhsdu) \ rhs
end

function makeKnots(tte::T, forward::T, strikes::AbstractVector{T}, size::Int; L=strikes[1] / 2, U=strikes[end] * 2, location="Mid-XX", minCount=1) where {T}
    size = min(size, length(strikes))
    if size <= 0 || size == length(strikes)
        s = searchsortedfirst(strikes, forward) #index of forward

        return if location == "Strikes"
            vcat(L, strikes, U) #almost ok, small osclla
        elseif location == "Mid-X"
            vcat(L, (strikes[1:s-2] + strikes[2:s-1]) / 2, (strikes[s:end-1] + strikes[s+1:end]) / 2, U) #good dens worse fit 
        elseif location == "Mid-Strikes"
            vcat(L, (strikes[1:end-1] + strikes[2:end]) / 2, U)
        elseif location == "Mid-XX"
            vcat(L, (strikes[1] * 3 - strikes[2]) / 2, (strikes[1:s-2] + strikes[2:s-1]) / 2, (strikes[s:end-1] + strikes[s+1:end]) / 2, (-strikes[end-1] + 3 * strikes[end]) / 2, U) #good dens worse fit 
        #        strikesWithMidF = sort(vcat(L,strikes,U, (forward+strikes[s-1])/2,(forward+strikes[s])/2))
        elseif location == "Uniform"
            uStrikes = range(strikes[1], stop=strikes[end], length=length(strikes) + 1)
            us = searchsortedfirst(uStrikes, forward)
            vcat(L, uStrikes .+ (forward - uStrikes[us]), U)
        end
    else

        uStrikes = if location == "Uniform"
            range(start=strikes[1], stop=strikes[end], length=size)
        elseif location == "Equidistributed"
            indices = round.(Int, range(1, stop=length(strikes), length=size))
            strikes[indices]
        end
        origStrikes = if size >= 0 && size != length(strikes)
            #if bucket is empty remove 
            count = 1
            newStrikes = [uStrikes[1]]
            index = 2
            for strike = strikes
                if strike < uStrikes[index]
                    count += 1
                else
                    if count >= minCount
                        push!(newStrikes, uStrikes[index])
                        count = 0
                    else
                        if index == length(uStrikes)
                            newStrikes[end] = uStrikes[index]
                        end
                    end
                    if index == length(uStrikes)
                        break
                    end
                    index += 1
                end
            end
            newStrikes
        else
            uStrikes
        end
        s = searchsortedfirst(origStrikes, forward) #index of forward
        x = vcat(L, (origStrikes[1] * 3 - origStrikes[2]) / 2, (origStrikes[1:s-2] + origStrikes[2:s-1]) / 2, (origStrikes[s:end-1] + origStrikes[s+1:end]) / 2, (-origStrikes[end-1] + 3 * origStrikes[end]) / 2, U) #good dens worse fit 
        return x
    end

end

using BSplines
function calibrateEQuadraticLVG(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; U::T=6 * forward, L::T=forward / 6, useVol=false, model="Linear", penalty=zero(T), nRefine=3, location="Mid-XX", isC3=true, minCount=1, size=0) where {T}

    #if the forward is not part of the strikes, add the forward and interpolate prices with quadratic. Eventually set corresponding weight to 0
    if U <= strikes[end]
        U = 2 * strikes[end] - strikes[end-1]
        #U = strikes[end]*1.1
    end
    if L >= strikes[1]
        L = strikes[1] / 2
    end
    s = searchsortedfirst(strikes, forward) #index of forward
    origStrikes = strikes
    useForwardInStrikes = false
    local priceAtm
    if strikes[s] != forward
        qStrikes = strikes[s-1:s+1]
        qPrices = callPrices[s-1:s+1]
        qvols = @. impliedVolatilitySRHalley(true, qPrices, forward, qStrikes, tte, 1.0, 0e-14, 64, Householder())
        lagrange = QuadraticLagrangePP(qStrikes, qvols, knotStyle=PPInterpolation.LEFT_KNOT)
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
    x = makeKnots(tte, forward, strikes, size, location=location, minCount=minCount, L=L, U=U)

    s = searchsortedfirst(x, forward)
    if !useForwardInStrikes && x[s] != forward
        x = sort(vcat(x, forward))
    end
    #  
    # if !useForwardInStrikes && strikes[s] != forward
    # end
    #shift approach to place forward.
    # x = vcat(L,strikes,U) .+ (forward-strikes[s]) #good dens, worse fit. Would need one more point? what if two strikes in same bucket?
    #x = vcat(L,range(strikes[1],stop=strikes[end],length=length(strikes)),U)
    #sx = searchsortedfirst(x, forward) 
    # x .+= forward-x[sx]
    s = searchsortedfirst(x, forward) - 1
    xBasis = if model == "Quadratic" && isC3
        sort(vcat(x, forward))
        # vcat(x[1:s+1],x[s+1:end]) #double knot at forward
    else
        x
    end
    m = length(x) - 1
    #initial guess
    vols = zeros(T, length(strikes))
    if useVol
        for i = eachindex(vols)
            vols[i] = impliedVolatilitySRHalley(true, callPrices[i], forward, strikes[i], tte, 1.0, 0e-14, 64, Householder())
        end
    end
    atmVol = impliedVolatilitySRHalley(true, priceAtm, forward, forward, tte, 1.0, 0e-14, 64, Householder())
    a0 = atmVol * sqrt(tte / 2)

    # minValue = sqrt(2 ) / 400
    minValue = a0 / 10
    maxValue = a0 * 1000
    # if model == "LinearBlack"
    #     a0*=forward
    #     # minValue*=10
    #     maxValue*=forward
    # end
    println(s, " forward found ", strikes[s], " min ", minValue, " max ", maxValue, " a0 ", a0)

    transform = ClosedTransformation(minValue, maxValue)
    iter = 0
    basis = BSplineBasis(3, xBasis)
    knots = BSplines.knots(basis)
    println("xBasis ", xBasis, " knots ", knots, "len knoths=", length(knots))
    #strikeIndices = indices of strike in x. (PP version) x will contain forward.
    strikeIndices = max.(searchsortedlast.(Ref(x), strikes), 1)  # x[i]<=z<=x[i+1]
    l = min(m + 1, length(callPrices))
    ### TODO don't set sigma0=sigma1: one more variable to minimize on. don't set lambda2 = lambda3.
    function toQuadForm!(α::AbstractArray{W}, β::AbstractArray{W}, γ::AbstractArray{W}, σ::AbstractArray{W}, priceAtm; model=model) where {W}

        if model == "Linear"
            if m + 1 == l + 2
                λ = vcat(σ[1], σ[1:l], σ[l])
            else
                λ = vcat(σ[1], σ[1:s], σ[s:l], σ[l])
                sp = s + 1
                λ[sp] = -((λ[sp+1] / (x[sp+1] - x[sp]) + λ[sp-1] / (x[sp] - x[sp-1]))) / (1 / (2priceAtm) - (1 / (x[sp] - x[sp-1]) + 1 / (x[sp+1] - x[sp])))
            end
            for i = 1:m
                β[i] = (λ[i+1] * forward - λ[i] * forward) / (x[i+1] - x[i])
                γ[i] = -x[i] * β[i] + λ[i] * forward
            end
        elseif model == "LinearBlack" #αx^2+βx+γ = (a+bx)(1+cx) 
            if m + 1 == l + 2
                λ = vcat(σ[1], σ[1:l], σ[l])
            else
                λ = vcat(σ[1], σ[1:s], σ[s:l], σ[l])
                sp = s + 1
                λ[sp] = -((λ[sp+1] / (x[sp+1] - x[sp]) + λ[sp-1] / (x[sp] - x[sp-1]))) / (1 / (2priceAtm) - (1 / (x[sp] - x[sp-1]) + 1 / (x[sp+1] - x[sp])))
                #     σ[sp]/=x[sp]
            end
            for i = 1:m
                bi = (λ[i+1] - λ[i]) / (x[i+1] - x[i])
                β[i] = -x[i] * bi + λ[i]
                α[i] = bi
            end
        elseif model == "Quadratic"
            #sigma is param from 2 to m;
            # println("sigma=",σ)vcat(σ.*x,σ[end]*x[end]))#
            λ = if isC3
                vcat(σ[1:s], σ[s:l]) .* forward
            else
                σ[1:l] .* forward
            end
            # λ = vcat(λ[1],λ,λ[end])
            if length(λ) < m + 3
                λ = vcat(λ[1], λ)
            end
            if length(λ) < m + 3
                λ = vcat(λ, λ[end])
            end
            if length(λ) < m + 3
                λ = vcat(λ, λ[end])
            end
            if length(λ) < m + 3
                λ = vcat(λ[1], λ)
            end
            if isC3
                sp = s + 1
                #λ[sp+1] = -((λ[sp+2]/(x[sp+1]-x[sp])+λ[sp]/(x[sp]-x[sp-1])))/(1/(4priceAtm)-(1/(x[sp]-x[sp-1])+1/(x[sp+1]-x[sp])))
                
                #what if initial priceATm too far off ? or even NaN
                denom = 1 / (4priceAtm) - (1 / (knots[sp+4] - knots[sp+2]) + 1 / (knots[sp+3] - knots[sp+1]))
                if denom == zero(W) || isnan(denom) || isinf(denom)
                    λ[sp+1] = λ[sp]
                else
                    λ[sp+1] = -((λ[sp+2] / (knots[sp+4] - knots[sp+2]) + λ[sp] / (knots[sp+3] - knots[sp+1]))) / denom
                end
            end
            # if priceAtm == 0
            #     λ[sp+1] =  λ[sp]      
            # else
            #     λ[sp+1] = -((λ[sp+2] / (knots[s+5] - knots[s+3]) + λ[sp] / (knots[s+4] - knots[s+2]))) / (1 / (4priceAtm) - (1 / (knots[s+5] - knots[s+3]) + 1 / (knots[s+4] - knots[s+2])))
            # end
           # println("lambda=", λ)
            spl = BSplines.Spline(basis, λ)
            for i = 1:m  #αx^2+βx+γ = a + b (x-xi) + c (x-xi)^2
                α[i] = spl(x[i], Derivative(2)) / 2
                # if abs(α[i]) < sqrt(eps(W))
                #     α[i] = sqrt(eps(W))
                # end
                β[i] = spl(x[i], Derivative(1)) - 2α[i] * x[i]
                γ[i] = spl(x[i]) - (α[i] * x[i] + β[i]) * x[i]
            end
        end
        # println("abc ",α ," ",β," ",γ)
        return α, β, γ
    end
    function obj!(fvec::Z, coeff::AbstractArray{W})::Z where {Z,W}
        n = length(strikes)
        σ = zeros(W, m + 1)
        @. σ[1:l] = transform(coeff)
        γ = zeros(W, m)
        β = zeros(W, m)
        α = zeros(W, m)
        θ = zeros(complex(W), 2m)
        toQuadForm!(α, β, γ, σ, priceAtm, model=model)
        computeCoeffsEQuadratic(s, tte, x, α, β, γ, θ)
        if isC3
            for k = 1:nRefine
                toQuadForm!(α, β, γ, σ, abs(θ[2s+2]), model=model)
                computeCoeffsEQuadratic(s, tte, x, α, β, γ, θ)
            end
        end
        # println("θ ", θ)
        if useVol
            for i = 1:n
                isCall = strikes[i] >= forward
                strikeIndex = strikeIndices[i]
                mPrice = if x[strikeIndex] == strikes[i]
                    abs(θ[2strikeIndex])
                else
                    priceEuropeanPiece(strikeIndex, EQuadraticLVG(x, α, β, γ, θ, tte, forward), isCall, strikes[i])
                end
                fvec[i] = impliedVolatilitySRHalley(isCall, mPrice, forward, strikes[i], tte, 1.0, 0e-14, 64, Householder()) - vols[i]
            end
        else
            for i = 1:n
                isCall = strikes[i] >= forward
                strikeIndex = strikeIndices[i]
                mPrice = if x[strikeIndex] == strikes[i]
                    abs(θ[2strikeIndex])
                else
                    priceEuropeanPiece(strikeIndex, EQuadraticLVG(x, α, β, γ, θ, tte, forward), isCall, strikes[i])
                end
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
            sumw = zero(T)
            v = zeros(W, m)
            for i = 1:m
                density = abs(θ[2i+2] / (α[i+1] * x[i]^2 + β[i+1] * x[i] + γ[i+1])^2)
                v[i] = log(density)
                sumw += weights[i]
            end
            for i = 2:m
                fvec[i+m-1] = sumw * penalty * ((v[i+1] - v[i]) / (x[i+1] - x[i]) - (v[i] - v[i-1]) / (x[i] - x[i-1]))
                # fvec[i+n-1] = sumw * penalty *( (σ[i+1]-σ[i])/(x[i+1]-x[i]) - (σ[i]-σ[i-1])/(x[i]-x[i-1]))
            end
        end
        #     ip = hermiteIntegral(derivative(p, 2)^2)
        #     pvalue = penalty * ip
        #     fvec[n+1] = pvalue
        # end
        iter += 1
        fvec
    end
    σ = zeros(T, l)
    for i = eachindex(σ)
        σ[i] = inv(transform, a0)
    end
    x0 = σ[1:l]
    outlen = length(callPrices)
    if penalty > 0
        outlen += m - 1
    end
    fit = LeastSquaresOptim.optimize!(
        LeastSquaresProblem(x=x0, (f!)=obj!, autodiff=:central, #:forward is 4x faster than :central
            output_length=outlen),
        LevenbergMarquardt();
        iterations=1000
    )
    fvec = zeros(Float64, outlen)
    # fit = GaussNewton.optimize!(obj!,x0,fvec,autodiff=:forward)
    obj!(fvec, fit.minimizer)
    #println(iter, " fit ", fit)
    @. σ[1:l] = transform(fit.minimizer)
    println("σ=", σ)
    γ = zeros(T, m)
    β = zeros(T, m)
    α = zeros(T, m)
    θ = zeros(complex(T), 2m)
    toQuadForm!(α, β, γ, σ, priceAtm, model=model)
    computeCoeffsEQuadratic(s, tte, x, α, β, γ, θ)
    if isC3
        for k = 1:nRefine
            toQuadForm!(α, β, γ, σ, abs(θ[2s+2]), model=model)
            computeCoeffsEQuadratic(s, tte, x, α, β, γ, θ)
        end
    end
    return EQuadraticLVG(x, α, β, γ, θ, tte, forward)
end

