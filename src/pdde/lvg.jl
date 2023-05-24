using LinearAlgebra
using LeastSquaresOptim
import AQFED.Black:impliedVolatilitySRHalley,Householder
import AQFED.Math:ClosedTransformation,inv
export ConstantLVG, priceEuropeanCall
struct ConstantLVG{TV,T}
    x::Vector{T}
    a::Vector{TV}
    θ::Vector{T}    
    tte::T
    forward::T
end

function priceEuropean(model::ConstantLVG{TV,T}, isCall::Bool, strike::T) where {TV,T}
    if strike <= model.x[1] || strike >= model.x[end]
        return isCall ? max(model.forward-strike,0.0) : max(strike-model.forward,0.0)
    end
    i = searchsortedfirst(model.x, strike)  # x[i-1]<z<=x[i]
    if strike != model.x[i] && i > 1
        i -= 1
    end
    # println(strike-model.x[i]," ",model.x[i+1]-strike)
    u = (strike-model.x[i])/(model.x[i+1]-model.x[i])*sqrt(2/(model.tte*model.a[i]^2))
    value = model.θ[2i]*cosh(u)+model.θ[2i-1]*sinh(u)
    if isCall && strike < model.forward
        value += model.forward-strike
    elseif !isCall && strike > model.forward
        value += strike-model.forward
    end
    return value
end

function derivativePrice(model::ConstantLVG{TV,T}, isCall::Bool, strike::T) where {TV,T}
    if strike <= model.x[1] || strike >= model.x[end]
        return isCall ? max(model.forward-strike,0.0) : max(strike-model.forward,0.0)
    end
    i = searchsortedfirst(model.x, strike)  # x[i-1]<z<=x[i]
    if strike != model.x[i] && i > 1
        i -= 1
    end
    ω=sqrt(2/(model.tte*model.a[i]^2))/(model.x[i+1]-model.x[i])
    u = (strike-model.x[i])* ω

    value = ω*model.θ[2i]*sinh(u)+  ω*model.θ[2i-1]*cosh(u)
    if isCall && strike < model.forward
        value -= 1
    elseif !isCall && strike > model.forward
        value += 1
    end
    return value
end

function computeCoeffs(s::Int,tte::T,strikes::AbstractVector{T},a::AbstractVector{TA},θ::AbstractVector{TA}) where {T,TA}
    #in order to price, we must solve a tridiag like system. This can be done once for all strikes. Hence outside price
    m = length(strikes)-1
    lhsd = zeros(TA, 2m)
    lhsdl = zeros(TA, 2m-1)
    lhsdu = zeros(TA, 2m -1)
    rhs = zeros(T, 2m)

    #theta0_c = 0
    lhsd[1] = zero(TA)
    lhsdu[1] = one(TA)
    rhs[1] = zero(T)
    #a[i] between zi and zip , a[i+1]at zip
    for i=1:m-1
        #line 2(i-1)+1
        κi = zero(T)
        κip = zero(T)
        zi = strikes[i]
        zip = strikes[i+1]
        χi = one(T)
        zdi = one(T)
        ωi = sqrt(2/(tte*a[i]^2))/(zip-zi)
        ωip = sqrt(2/(tte*a[i+1]^2))/(strikes[i+2]-strikes[i+1])
        zdip = one(T)
        coshi = cosh(ωi*(zip-zi))
        sinhi = sinh(ωi*(zip-zi))
        if i==s
            lhsd[2i] = (κi*coshi + ωi*sinhi)*χi*zdi - κip*zdip*χi*coshi
            lhsdl[2i-1] = (ωi*coshi+κi*sinhi)*χi*zdi - κip*sinhi*χi*zdip
            lhsdu[2i] = -ωip*zdip
            rhs[2i] = one(T)
            #line 2(i-1)+2    
            lhsd[2i+1] = -ωip*zdip
            lhsdl[2i]= (κi*coshi + ωi*sinhi - (ωi*coshi+κi*sinhi)/sinhi*coshi)*χi*zdi
            lhsdu[2i+1] =  zdi*(ωi*coshi+κi*sinhi)/(sinhi) - κip*zdip
            rhs[2i+1] = one(T)

        else 
        lhsd[2i] = (κi*coshi + ωi*sinhi)*χi*zdi - κip*coshi*χi*zdip   #2i
        lhsdl[2i-1] = (ωi*coshi+κi*sinhi)*χi*zdi - κip*sinhi*χi*zdip   #2i-1
        lhsdu[2i] = -ωip*zdip                                      #2i+1
        #line 2(i-1)+2    
        lhsd[2i+1] = -ωip*zdip
        lhsdl[2i]= (κi*coshi + ωi*sinhi - (ωi*coshi+κi*sinhi)/sinhi*coshi)*χi*zdi
        lhsdu[2i+1] =  zdi*(ωi*coshi+κi*sinhi)/(sinhi) - κip*zdip 
        # if !isfinite(lhsd[2i]) || !isfinite(lhsdl[2i-1]) || !isfinite(lhsdl[2i]) || !isfinite(lhsdu[2i+1]) 
        #     println(" Na N ",ωi*(zip-zi), " at ",i)
        # end
        end             
    end
    zi = strikes[m]
    zip = strikes[m+1]
    ωm = sqrt(2/(tte*a[m]^2))/(zip-zi)
    coshm = cosh(ωm*(zip-zi))
    sinhm = sinh(ωm*(zip-zi))
    # u1=thetaOs, u2=theta0c, u3=theta1s, u4=theta1c    ...
    lhsd[2m] = coshm
    lhsdl[2m-1] = sinhm
    rhs[2m] = zero(T)
#    println("solving with a=",a," lhsdl=",lhsdl," lhsd=",lhsd," lhsdu",lhsdu)
    θ[:] = Tridiagonal(lhsdl,lhsd,lhsdu) \ rhs
    
end

function calibrateConstantLVG(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T};U::T=6*forward, L::T=zero(forward),useVol=false ) where {T}

    #if the forward is not part of the strikes, add the forward and interpolate prices with quadratic. Eventually set corresponding weight to 0
    if U <= strikes[end]
        U = 2*strikes[end] - strikes[end-1]
        #U = strikes[end]*1.1
    end
    x = vcat(L,strikes,U)
    m = length(x)-1
    s = searchsortedfirst(strikes, forward) #index of forward
    #initial guess
    vols = zeros(T,length(strikes))
    if useVol
        for i=eachindex(vols)
            vols[i] =  impliedVolatilitySRHalley(true, callPrices[i], forward, strikes[i], tte, 1.0, 0e-14, 64,Householder())
        end
    end
    atmVol = impliedVolatilitySRHalley(true, callPrices[s], forward, forward, tte, 1.0, 0e-14, 64,Householder())
    a0 = atmVol*forward
    minValue = sqrt(2/tte)/50 # ch(sqrt(2/(tte*a[i]^2))*deltastrike) => divide by deltastrike
    maxValue = 1.0/minValue
    println("forward found ",strikes[s]," min ",minValue, " max ",maxValue)
    
    transform = ClosedTransformation(minValue, maxValue)
   
    function obj!(fvec::Z, c::AbstractArray{W})::Z where {Z,W}
        n = length(strikes)
        a = zeros(typeof(c[1]),m)
        @. a[2:m] = transform(c[1:end])
        a[1] = a[2] #*(x[2]-x[1])/(x[3]-x[2])
        θ  = zeros(typeof(c[1]),2m)
        computeCoeffs(s,tte,x,a,θ) 
        if useVol
            # println( θ)
            for i=1:n
                fvec[i] =  impliedVolatilitySRHalley(strikes[i] >= forward,abs( θ[2i+2]), forward, strikes[i], tte, 1.0, 0e-14, 64,Householder())-vols[i]
            end
        else   
        for i=1:n
            mPrice = θ[2i+2]
            if (strikes[i] < forward)
            mPrice += forward-strikes[i]
            end
            fvec[i] = weights[i]*(mPrice-callPrices[i]) #FIXME what if forward not on list?
        end
    end
        # if penalty > 0
        #     ip = hermiteIntegral(derivative(p, 2)^2)
        #     pvalue = penalty * ip
        #     fvec[n+1] = pvalue
        # end
        fvec
    end
    a = zeros(T,m)
    for i=eachindex(a)
        a[i] = inv(transform, a0)
    end
    fit = optimize!(
        LeastSquaresProblem(x = a[2:end], f! = obj!, autodiff = :forward, #:forward is 4x faster than :central
        output_length = length(callPrices)),
        LevenbergMarquardt();
        iterations = 1000,
    )
    fvec = zeros(Float64, length(callPrices))
    obj!(fvec,fit.minimizer)
    #println(iter, " fit ", fit)
    @. a[2:end] = transform(fit.minimizer)
    a[1] = a[2]
    θ = zeros(T,2m)

    computeCoeffs(s,tte,x,a,θ)    
    return ConstantLVG(x,a,θ,tte,forward)
end
