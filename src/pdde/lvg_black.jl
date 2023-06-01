using LinearAlgebra
using LeastSquaresOptim
using GaussNewton
import AQFED.Black: impliedVolatilitySRHalley, Householder
import AQFED.Math: ClosedTransformation, inv, IdentityTransformation
export ConstantBlackLVG, LinearBlackLVG, priceEuropeanCall
struct ConstantBlackLVG{TV,T}
    x::Vector{T}
    a::Vector{TV}
    θ::Vector{T}
    tte::T
    forward::T
end

struct LinearBlackLVG{TV,T}
    x::Vector{T}
    α::Vector{TV}
    θ::Vector{T}
    tte::T
    forward::T
end

Base.broadcastable(p::LinearBlackLVG) = Ref(p)


function priceEuropean(model::LinearBlackLVG{TV,T}, isCall::Bool, strike) where {TV,T}
    x = model.x
    if strike <= x[1] || strike >= x[end]
        return isCall ? max(model.forward - strike, 0.0) : max(strike - model.forward, 0.0)
    end
    i = searchsortedfirst(x, strike)  # x[i-1]<z<=x[i]
    if strike != x[i] && i > 1
        i -= 1
    end
    αi = model.α[i] * sqrt(model.tte)
    αip = model.α[i+1] * sqrt(model.tte)
    b = (αip - αi) / (x[i+1] - x[i])
    a = -x[i] * b + αi
    axb = abs(b * (strike - x[i]) + αi)
    χ = sqrt(strike * axb / (x[i] * αi))
    ω = sqrt(4 / a^2 + 1) / 2
    u = ω * log(strike * αi / (x[i] * axb))   #log(strike+1/strike)*1/sqrt(tte*a))
    value = χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))
    if isCall && strike < model.forward
        value += model.forward - strike
    elseif !isCall && strike > model.forward
        value += strike - model.forward
    end
    return value
end

function derivativePrice(model::LinearBlackLVG{TV,T}, isCall::Bool, strike::T) where {TV,T}
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
    αi = model.α[i] * sqrt(model.tte)
    αip = model.α[i+1] * sqrt(model.tte)
    b = (αip - αi) / (x[i+1] - x[i])
    a = -x[i] * b + αi
    axb = abs(b * strike + a)
    χ = sqrt(strike * axb / (x[i] * αi))
    ω = sqrt(4 / a^2 + 1) / 2
    z = log(strike / axb) # log(strike * α[i] * sqrt(model.tte) / (x[i] * axb))
    zi = log(x[i] / αi)
    # a = (α[i]-x[i]*(α[i+1]-α[i])/(x[i+1]-x[i]))*sqrt(tte)
    dz = one(T) / strike - b / axb
    # dχ = (sqrt(axb)/sqrt(strike)+b*sartstrike/sqrt(axb))/2= X/2*(1/strike+b/axb)
    #  κ = dχ/(χ*dz)
    # κ = (sqrt(axb / strike) + b * sqrt(strike / axb)) / (2 * (sqrt(axb / strike) - b * sqrt(strike / axb)))
    κ = (b / (axb) + one(T) /strike ) / (2 * dz)
  
    sh, ch = sinh(ω * (z - zi)), cosh(ω * (z - zi))

    value = χ * dz * ((κ * model.θ[2i-1] + ω * model.θ[2i]) * sh + (κ * model.θ[2i] + ω * model.θ[2i-1]) * ch)
    if isCall && strike < model.forward
        value -= 1
    elseif !isCall && strike > model.forward
        value += 1
    end
    return value
end

function computeCoeffsLinearBlack(s::Int, tte::T, strikes::AbstractVector{T}, α::AbstractVector{TA}, θ::AbstractVector{TA}; tri=Tridiagonal(zeros(TA, 2(length(strikes) - 1) - 1), zeros(TA, 2(length(strikes) - 1)), zeros(TA, 2(length(strikes) - 1) - 1))) where {T,TA}
    #in order to price, we must solve a tridiag like system. This can be done once for all strikes. Hence outside price
    m = length(strikes) - 1
    lhsd = tri.d
    lhsdl = tri.dl
    lhsdu = tri.du
    rhs = zeros(T, 2m)

    #theta0_c = 0
    lhsd[1] = zero(TA)
    lhsdu[1] = one(TA)
    rhs[1] = zero(T)
    #a[i] between zi and zip , a[i+1]at zip
    for i = 1:m-1
        #line 2(i-1)+1
        αip = α[i+1] * sqrt(tte)
        αi = α[i] * sqrt(tte)
        αip2 = α[i+2] * sqrt(tte)
        bi = (αip - αi) / (strikes[i+1] - strikes[i])
        ai = -strikes[i] * bi + αi
        #ai= (α[i]-strikes[i]*ai)
        axbip = αip
        bip = (αip2 - αip) / (strikes[i+2] - strikes[i+1])
        aip = -strikes[i+1] * bip + αip
        sqrtas = sqrt(axbip / strikes[i+1])
        κi = (sqrtas + bi / sqrtas) / (2 * (sqrtas - bi / sqrtas))
        κip = (sqrtas + bip / sqrtas) / (2 * (sqrtas - bip / sqrtas))
        χi = sqrt(strikes[i+1] * axbip / (strikes[i] * αi))
        zdi = one(T) / strikes[i+1] - bi / axbip
        ωi = sqrt(4 / ai^2 + 1) / 2
        ωip = sqrt(4 / aip^2 + 1) / 2
        zdip = one(T) / strikes[i+1] - bip / axbip
        omegazipmzi = ωi * log(strikes[i+1] * αi / (strikes[i] * αip)) # log(strike * α[i] * sqrt(model.tte) / (x[i] * axb))
        coshi = cosh(omegazipmzi)
        sinhi = sinh(omegazipmzi)
        lhsd[2i] = (κi * coshi + ωi * sinhi) * χi * zdi - κip * coshi * χi * zdip   #2i
        lhsdl[2i-1] = (ωi * coshi + κi * sinhi) * χi * zdi - κip * sinhi * χi * zdip   #2i-1
        lhsdu[2i] = -ωip * zdip                                      #2i+1
        #line 2(i-1)+2    
        lhsd[2i+1] = -ωip * zdip
        lhsdl[2i] = (κi * coshi + ωi * sinhi - (ωi * coshi + κi * sinhi) / sinhi * coshi) * χi * zdi
        lhsdu[2i+1] = zdi * (ωi * coshi + κi * sinhi) / (sinhi) - κip * zdip
        # if !isfinite(lhsd[2i]) || !isfinite(lhsdl[2i-1]) || !isfinite(lhsdl[2i]) || !isfinite(lhsdu[2i+1]) 
        #     println(" Na N ",ωi*(zip-zi), " at ",i)
        # end
    end
    rhs[2s] = one(T)
    rhs[2s+1] = one(T)
    αm = α[m] * sqrt(tte)
    αmp = α[m+1] * sqrt(tte)
    zi = log(strikes[m] / αm)
    zip = log(strikes[m+1] / αmp)
    bi = (αmp - αm) / (strikes[m+1] - strikes[m])
    ai = -strikes[m] * bi + αm
    ωm = sqrt(4 / ai^2 + 1) / 2
    tanhm = tanh(ωm * (zip - zi))
    # u1=thetaOs, u2=theta0c, u3=theta1s, u4=theta1c    ...
    lhsd[2m] = one(T)
    lhsdl[2m-1] = tanhm
    rhs[2m] = zero(T)

    #    println("solving with a=",a," lhsdl=",lhsdl," lhsd=",lhsd," lhsdu",lhsdu)
    lhsf = lu!(tri)
    #  lhsf = factorize(Tridiagonal(lhsdl, lhsd, lhsdu))
    ldiv!(θ, lhsf, rhs)
    # θ[:] = Tridiagonal(lhsdl, lhsd, lhsdu) \ rhs
end

function calibrateLinearBlackLVG(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; U::T=6 * forward, L::T=forward / 6, useVol=false, penalty=zero(T)) where {T}

    #if the forward is not part of the strikes, add the forward and interpolate prices with quadratic. Eventually set corresponding weight to 0
    if U <= strikes[end]
        U = 2 * strikes[end] - strikes[end-1]
        #U = strikes[end]*1.1
    end
    if L >= strikes[1]
        L = strikes[1] / 2
    end
    s = searchsortedfirst(strikes, forward) #index of forward
    isForwardOnGrid = (strikes[s] == forward)
    x = if isForwardOnGrid
        vcat(L, strikes, U)
    else
        vcat(L,strikes[1:s-1],forward,strikes[s:end],U)                
    end
    m = length(x) - 1
    # println(x[s]," ",x[s+1])
    #initial guess
    vols = zeros(T, length(strikes))
    if useVol
        for i = eachindex(vols)
            vols[i] = impliedVolatilitySRHalley(true, callPrices[i], forward, strikes[i], tte, 1.0, 0e-14, 64, Householder())
        end
    end
    atmVol = impliedVolatilitySRHalley(true, callPrices[s], forward, forward, tte, 1.0, 0e-14, 64, Householder())
    a0 = atmVol/sqrt(2)
    priceAtm = callPrices[s]
    if !isForwardOnGrid
        priceAtm = (callPrices[s]*(strikes[s+1]-forward)+callPrices[s+1]*(forward-strikes[s]))/(strikes[s+1]-strikes[s])
    end
    minValue = sqrt(2 / tte) / 2000
    maxValue = 1.0 / (4minValue)
  # println("forward found ", strikes[s], " min ", minValue, " max ", maxValue, " a0 ",a0," ",s)

    transform = ClosedTransformation(minValue, maxValue)
    # transform = IdentityTransformation{T}()
    iter = 0
    function fillAWithParams!(a,θ::AbstractArray{W}, c) where {W}
        if isForwardOnGrid
            @. a[2:m] = transform(c)
            a[1] = a[2]
            a[m+1] = a[m] #/log(x[2]/x[1])*log(x[3]/x[2])
        else
            @. a[2:s] = transform(c[1:s-1])
            @. a[s+2:end-1] = transform(c[s:end])
            a[1] = a[2]
            a[end] = a[m] #/log(x[2]/x[1])*log(x[3]/x[2])
            sp=s+1
             a[sp] = -((a[sp+1]/(x[sp+1]-x[sp])+a[sp-1]/(x[sp]-x[sp-1])))/(1/(2priceAtm)-(1/(x[sp]-x[sp-1])+1/(x[sp+1]-x[sp])))
            # println("a[s-1] ",a[sp-1:sp+1])
              computeCoeffsLinearBlack(s, tte, x, a, θ)
              a[sp] = -((a[sp+1]/(x[sp+1]-x[sp])+a[sp-1]/(x[sp]-x[sp-1])))/(one(W)/(2θ[2sp])-(1/(x[sp]-x[sp-1])+1/(x[sp+1]-x[sp])))
            #  computeCoeffsLinearBlack(s, tte, x, a, θ)
            #  a[sp] = -((a[sp+1]/(x[sp+1]-x[sp])+a[sp-1]/(x[sp]-x[sp-1])))/(one(W)/(2θ[2sp])-(1/(x[sp]-x[sp-1])+1/(x[sp+1]-x[sp])))

        end
        computeCoeffsLinearBlack(s, tte, x, a, θ)
        #  println("a=",a,"\n c=",c)
    end
    function obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        n = length(strikes)
        a = zeros(W, m + 1)
        θ = zeros(W, 2m)
        fillAWithParams!(a, θ, c)
        if useVol
            # println( θ)
            if isForwardOnGrid
            for i = 1:n
                fvec[i] = impliedVolatilitySRHalley(strikes[i] >= forward, abs(θ[2i+2]), forward, strikes[i], tte, 1.0, 0e-14, 64, Householder()) - vols[i]
            end
        else
            for i = 1:n
                ip = i+1
                if i>=s
                    ip+=1
                end
                fvec[i] = impliedVolatilitySRHalley(strikes[i] >= forward, abs(θ[2ip]), forward, strikes[i], tte, 1.0, 0e-14, 64, Householder()) - vols[i]
            end
        end
        else
            if isForwardOnGrid
            for i = 1:n
                mPrice = θ[2i+2]
                if (strikes[i] < forward)
                    mPrice += forward - strikes[i]
                end
                fvec[i] = weights[i] * (mPrice - callPrices[i]) #FIXME what if forward not on list?
            end
        else
            for i = 1:n
                ip = i+1
                if i>=s
                    ip+=1
                end
                mPrice = θ[2ip]
                if (strikes[i] < forward)
                    mPrice += forward - strikes[i]
                end
                fvec[i] = weights[i] * (mPrice - callPrices[i]) #FIXME what if forward not on list?
            end
        end
        end
        if penalty > 0
            sumw = zero(W)
            for i = 1:n
                sumw += weights[i]
            end
            for i = 3:length(x)-2
                if θ[2i-2] != zero(W) && θ[2i] != zero(W) && θ[2i+2] != zero(W)

                vim = log(abs(θ[2i-2]) / ((a[i-1]*x[i-1])^2 * tte))
                vi = log(abs(θ[2i]) / ((a[i]*x[i])^2 * tte))
                vip = log(abs(θ[2i+2]) / ((a[i+1]*x[i+1])^2 * tte))
          
                fvec[i+n-2] = sumw * penalty * ((vip-vi)/(x[i+1]-x[i]) - (vi-vim)/(x[i]-x[i-1]))
                else
                    fvec[i+n-2] = zero(W)
                end
            end
        end   
        iter += 1
        fvec
    end
    a = zeros(T, m + 1)
    for i = eachindex(a)
        a[i] = inv(transform, a0)
    end
    x0 = if isForwardOnGrid
        a[2:m]
    else
        vcat(a[2:s],a[s+2:end-1])
    end
    outlen = length(callPrices)
    if penalty > 0
        outlen += length(x)-3
    end
    fit = LeastSquaresOptim.optimize!(
        LeastSquaresProblem(x=x0, (f!)=obj!, autodiff=:forward, #:forward is 4x faster than :central
            output_length=outlen), 
        LevenbergMarquardt();    
        #   lower=fill(minValue,length(x0)),upper=fill(maxValue,length(x0))  ,
        iterations=1000
    )
    x0 = fit.minimizer
    fvec = zeros(Float64, outlen)
    # fit = GaussNewton.optimize!(obj!,x0,fvec,autodiff=:forward)
    obj!(fvec, x0)
    # println(iter, " fit ", fit)
    θ = zeros(T, 2m)
    fillAWithParams!(a,θ, x0)
    return LinearBlackLVG(x, a, θ, tte, forward)
end


function priceEuropean(model::ConstantBlackLVG{TV,T}, isCall::Bool, strike::T) where {TV,T}
    if strike <= model.x[1] || strike >= model.x[end]
        return isCall ? max(model.forward - strike, 0.0) : max(strike - model.forward, 0.0)
    end
    i = searchsortedfirst(model.x, strike)  # x[i-1]<z<=x[i]
    if strike != model.x[i] && i > 1
        i -= 1
    end
    # println(strike-model.x[i]," ",model.x[i+1]-strike)
    b = model.a[i] * sqrt(model.tte)
    χ = sqrt(strike / model.x[i])
    u = log(strike / model.x[i]) * (sqrt(4 / b^2 + 1) / 2)
    value = χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))
    if isCall && strike < model.forward
        value += model.forward - strike
    elseif !isCall && strike > model.forward
        value += strike - model.forward
    end
    return value
end

function derivativePrice(model::ConstantBlackLVG{TV,T}, isCall::Bool, strike::T) where {TV,T}
    if strike <= model.x[1] || strike >= model.x[end]
        return isCall ? max(model.forward - strike, 0.0) : max(strike - model.forward, 0.0)
    end
    i = searchsortedfirst(model.x, strike)  # x[i-1]<z<=x[i]
    if strike != model.x[i] && i > 1
        i -= 1
    end
    b = model.a[i] * sqrt(model.tte)
    χ = sqrt(strike / model.x[i])
    z = log(strike / model.x[i])
    dz = one(T) / strike
    ω = sqrt(4 / b^2 + 1) / 2
    #dχ = 1/sqrt(x) = 1/2*χ*dz
    κ = 1 / 2
    sh, ch = sinh(ω * z), cosh(ω * z)

    value = χ * dz * ((κ * model.θ[2i-1] + ω * model.θ[2i]) * sh + (κ * model.θ[2i] + ω * model.θ[2i-1]) * ch)
    if isCall && strike < model.forward
        value -= 1
    elseif !isCall && strike > model.forward
        value += 1
    end
    return value
end

function computeCoeffsBlack(s::Int, tte::T, strikes::AbstractVector{T}, a::AbstractVector{TA}, θ::AbstractVector{TA}) where {T,TA}
    #in order to price, we must solve a tridiag like system. This can be done once for all strikes. Hence outside price
    m = length(strikes) - 1
    lhsd = zeros(TA, 2m)
    lhsdl = zeros(TA, 2m - 1)
    lhsdu = zeros(TA, 2m - 1)
    rhs = zeros(T, 2m)

    #theta0_c = 0
    lhsd[1] = zero(TA)
    lhsdu[1] = one(TA)
    rhs[1] = zero(T)
    #a[i] between zi and zip , a[i+1]at zip
    for i = 1:m-1
        #line 2(i-1)+1
        κi = one(T) / 2
        κip = one(T) / 2
        zi = log(strikes[i])
        zip = log(strikes[i+1])
        χi = sqrt(strikes[i+1] / strikes[i])
        zdi = 1 / strikes[i+1]
        ωi = sqrt(4 / (tte * a[i]^2) + 1) / 2
        ωip = sqrt(4 / (tte * a[i+1]^2) + 1) / 2
        zdip = 1 / strikes[i+1]
        coshi = cosh(ωi * (zip - zi))
        sinhi = sinh(ωi * (zip - zi))
        if i == s
            lhsd[2i] = (κi * coshi + ωi * sinhi) * χi * zdi - κip * zdip * χi * coshi
            lhsdl[2i-1] = (ωi * coshi + κi * sinhi) * χi * zdi - κip * sinhi * χi * zdip
            lhsdu[2i] = -ωip * zdip
            rhs[2i] = one(T)
            #line 2(i-1)+2    
            lhsd[2i+1] = -ωip * zdip
            lhsdl[2i] = (κi * coshi + ωi * sinhi - (ωi * coshi + κi * sinhi) / sinhi * coshi) * χi * zdi
            lhsdu[2i+1] = zdi * (ωi * coshi + κi * sinhi) / (sinhi) - κip * zdip
            rhs[2i+1] = one(T)

        else
            lhsd[2i] = (κi * coshi + ωi * sinhi) * χi * zdi - κip * coshi * χi * zdip   #2i
            lhsdl[2i-1] = (ωi * coshi + κi * sinhi) * χi * zdi - κip * sinhi * χi * zdip   #2i-1
            lhsdu[2i] = -ωip * zdip                                      #2i+1
            #line 2(i-1)+2    
            lhsd[2i+1] = -ωip * zdip
            lhsdl[2i] = (κi * coshi + ωi * sinhi - (ωi * coshi + κi * sinhi) / sinhi * coshi) * χi * zdi
            lhsdu[2i+1] = zdi * (ωi * coshi + κi * sinhi) / (sinhi) - κip * zdip
            # if !isfinite(lhsd[2i]) || !isfinite(lhsdl[2i-1]) || !isfinite(lhsdl[2i]) || !isfinite(lhsdu[2i+1]) 
            #     println(" Na N ",ωi*(zip-zi), " at ",i)
            # end
        end
    end
    zi = log(strikes[m])
    zip = log(strikes[m+1])
    ωm = sqrt(4 / (tte * a[m]^2) + 1) / 2
    coshm = cosh(ωm * (zip - zi))
    sinhm = sinh(ωm * (zip - zi))
    # u1=thetaOs, u2=theta0c, u3=theta1s, u4=theta1c    ...
    lhsd[2m] = coshm
    lhsdl[2m-1] = sinhm
    rhs[2m] = zero(T)
    #    println("solving with a=",a," lhsdl=",lhsdl," lhsd=",lhsd," lhsdu",lhsdu)
    θ[:] = Tridiagonal(lhsdl, lhsd, lhsdu) \ rhs

end

#TODO most of this function is in common wth LVG, could use model as param.
function calibrateConstantBlackLVG(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; U::T=6 * forward, L::T=forward / 6, useVol=false) where {T}

    #if the forward is not part of the strikes, add the forward and interpolate prices with quadratic. Eventually set corresponding weight to 0
    if U <= strikes[end]
        U = 2 * strikes[end] - strikes[end-1]
        #U = strikes[end]*1.1
    end
    if L >= strikes[1]
        L = strikes[1] / 2
    end
    x = vcat(L, strikes, U)
    m = length(x) - 1
    s = searchsortedfirst(strikes, forward) #index of forward
    #initial guess
    vols = zeros(T, length(strikes))
    if useVol
        for i = eachindex(vols)
            vols[i] = impliedVolatilitySRHalley(true, callPrices[i], forward, strikes[i], tte, 1.0, 0e-14, 64, Householder())
        end
    end
    atmVol = impliedVolatilitySRHalley(true, callPrices[s], forward, forward, tte, 1.0, 0e-14, 64, Householder())
    a0 = atmVol
    minValue = sqrt(2 / tte) / 50
    maxValue = 1.0 / minValue
    println("forward found ", strikes[s], " min ", minValue, " max ", maxValue)

    transform = ClosedTransformation(minValue, maxValue)

    function obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        n = length(strikes)
        a = zeros(typeof(c[1]), m)
        @. a[2:m] = transform(c[1:end])
        a[1] = a[2] / log(x[2] / x[1]) * log(x[3] / x[2])
        θ = zeros(typeof(c[1]), 2m)
        computeCoeffsBlack(s, tte, x, a, θ)
        if useVol
            # println( θ)
            for i = 1:n
                fvec[i] = impliedVolatilitySRHalley(strikes[i] >= forward, abs(θ[2i+2]), forward, strikes[i], tte, 1.0, 0e-14, 64, Householder()) - vols[i]
            end
        else
            for i = 1:n
                mPrice = θ[2i+2]
                if (strikes[i] < forward)
                    mPrice += forward - strikes[i]
                end
                fvec[i] = weights[i] * (mPrice - callPrices[i]) #FIXME what if forward not on list?
            end
        end
        # if penalty > 0
        #     ip = hermiteIntegral(derivative(p, 2)^2)
        #     pvalue = penalty * ip
        #     fvec[n+1] = pvalue
        # end
        fvec
    end
    a = zeros(T, m)
    for i = eachindex(a)
        a[i] = inv(transform, a0)
    end
    fit = optimize!(
        LeastSquaresProblem(x=a[2:end], (f!)=obj!, autodiff=:forward, #:forward is 4x faster than :central
            output_length=length(callPrices)),
        LevenbergMarquardt();
        iterations=1000
    )
    fvec = zeros(Float64, length(callPrices))
    obj!(fvec, fit.minimizer)
    #println(iter, " fit ", fit)
    @. a[2:end] = transform(fit.minimizer)
    a[1] = a[2] / log(x[2] / x[1]) * log(x[3] / x[2])
    θ = zeros(T, 2m)

    computeCoeffsBlack(s, tte, x, a, θ)
    return ConstantBlackLVG(x, a, θ, tte, forward)
end
