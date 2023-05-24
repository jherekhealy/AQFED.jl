using LinearAlgebra
using LeastSquaresOptim
using GaussNewton
import AQFED.Black: blackScholesFormula, impliedVolatilitySRHalley, Householder
import AQFED.Math: ClosedTransformation, inv

struct QuadraticLVG{TV,T}
    x::Vector{T}
    a::Vector{TV} #(a+bx)(1+cx) #encompasses constant, black, linear, linearblack.
    b::Vector{TV}
    c::Vector{TV}
    θ::Vector{T}
    tte::T
    forward::T
end

Base.broadcastable(p::QuadraticLVG) = Ref(p)


function priceEuropean(model::QuadraticLVG{TV,T}, isCall::Bool, strike::T) where {TV,T}
    x = model.x
    if strike <= x[1] || strike >= x[end]
        return isCall ? max(model.forward - strike, 0.0) : max(strike - model.forward, 0.0)
    end
    i = searchsortedfirst(x, strike)  # x[i-1]<z<=x[i]
    if strike != x[i] && i > 1
        i -= 1
    end
    b = model.b[i]
    a = model.a[i]
    c = model.c[i]
    value = if c==zero(TV) && b==zero(TV)
        zzi = strike-x[i]
        χ = one(T)
        ω = sqrt(1/a)    
        u = ω * zzi
        χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))
    else
        zzi = abs( (a + b*(strike))*(c*x[i]+1)/((a+b*x[i])*(c*(strike)+1)))
        χ = sqrt(abs((a + b*strike)*(c*strike+1)/((a+b*x[i])*(c*x[i]+1))))
        Δ = (b-a*c)^2
        ω = sqrt(4 / Δ + 1) / 2
        u = ω * log(zzi) 
        χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))
    end
    #println(i," ",x[i]," ",strike," ",χ," ",u)
    if isCall && strike < model.forward
        value += model.forward - strike
    elseif !isCall && strike > model.forward
        value += strike - model.forward
    end
    return value
end

function derivativePrice(model::QuadraticLVG{TV,T}, isCall::Bool, strike::T) where {TV,T}
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
    b = model.b[i]
    a = model.a[i]
    c = model.c[i]
    zzi = abs( (a + b*(strike))*(c*x[i]+1)/((a+b*x[i])*(c*(strike)+1)))
    χ = sqrt(abs((a + b*strike)*(c*strike+1)/((a+b*x[i])*(c*x[i]+1))))
    Δ = (b-a*c)^2
    ω = sqrt(4 / Δ + 1) / 2
    u = ω * log(zzi) 
    dz = b / (a+b*strike) - c / (1+c*strike)
    # dχ = (b)/sqrt(a+bstrike) * sqrt(1+cstrike) + c/sqrt(1+cstrike)*sqrt(a+bstrike) = X * (b/(a+bstrike)+c/(1+cstrike))
    #  κ = dχ/(χ*dz)
    κ = (b / (a+b*strike) + c / (1+c*strike)) / (2 * dz)
    sh, ch = sinh(u), cosh(u)

    value = χ * dz * ((κ * model.θ[2i-1] + ω * model.θ[2i]) * sh + (κ * model.θ[2i] + ω * model.θ[2i-1]) * ch)
    if isCall && strike < model.forward
        value -= 1
    elseif !isCall && strike > model.forward
        value += 1
    end
    return value
end

function computeCoeffsQuadratic(s::Int, tte::T, strikes::AbstractVector{T}, a::AbstractVector{TC},b::AbstractVector{TC},c::AbstractVector{TC}, θ::AbstractVector{TA}; tri=Tridiagonal(zeros(TA, 2(length(strikes) - 1) - 1), zeros(TA, 2(length(strikes) - 1)), zeros(TA, 2(length(strikes) - 1) - 1))) where {T,TA,TC}
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
        χi = sqrt(abs((a[i] + b[i]*strikes[i+1])*(c[i]*strikes[i+1]+1)/((a[i]+b[i]*strikes[i])*(c[i]*strikes[i]+1))))
        zdi = b[i] / (a[i]+b[i]*strikes[i+1]) - c[i] / (1+c[i]*strikes[i+1])
        zdip = b[i+1] / (a[i+1]+b[i+1]*strikes[i+1]) - c[i+1] / (1+c[i+1]*strikes[i+1]) 
        κi = (b[i] / (a[i]+b[i]*strikes[i+1]) + c[i] / (1+c[i]*strikes[i+1])) / (2 * zdi)
        κip = (b[i+1] / (a[i+1]+b[i+1]*strikes[i+1]) + c[i+1] / (1+c[i+1]*strikes[i+1])) / (2 * zdip)
        Δ = (b[i]-a[i]*c[i])^2
        Δp = (b[i+1]-a[i+1]*c[i+1])^2
        ωi = sqrt(4 / Δ + 1) / 2
        ωip = sqrt(4 / Δp + 1) / 2
        zzi = abs( (a[i] + b[i]*(strikes[i+1]))*(c[i]*strikes[i]+1)/((a[i]+b[i]*strikes[i])*(c[i]*(strikes[i+1])+1)))
    
        omegazipmzi = ωi * log(zzi)
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
    Δ = (b[m]-a[m]*c[m])^2
    ωm = sqrt(4 / Δ + 1) / 2
    zzi = abs( (a[m] + b[m]*(strikes[m+1]))*(c[m]*strikes[m]+1)/((a[m]+b[m]*strikes[m])*(c[m]*(strikes[m+1])+1)))
    coshm = cosh(ωm * zzi)
    sinhm = sinh(ωm * zzi)
    # u1=thetaOs, u2=theta0c, u3=theta1s, u4=theta1c    ...
    lhsd[2m] = coshm
    lhsdl[2m-1] = sinhm
    rhs[2m] = zero(T)

    #    println("solving with a=",a," lhsdl=",lhsdl," lhsd=",lhsd," lhsdu",lhsdu)
    lhsf = lu!(tri)
    #  lhsf = factorize(Tridiagonal(lhsdl, lhsd, lhsdu))
    ldiv!(θ, lhsf, rhs)
    # θ[:] = Tridiagonal(lhsdl, lhsd, lhsdu) \ rhs
end
function calibrateQuadraticLVG(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; U::T=6 * forward, L::T=forward / 6, useVol=false, model="Linear",penalty=zero(T)) where {T}

    #if the forward is not part of the strikes, add the forward and interpolate prices with quadratic. Eventually set corresponding weight to 0
    if U <= strikes[end]
        U = 2 * strikes[end] - strikes[end-1]
        #U = strikes[end]*1.1
    end
    if L >= strikes[1]
        L = strikes[1] / 2
    end
    s = searchsortedfirst(strikes, forward) #index of forward
    if strikes[s] != forward
        qStrikes = strikes[s-1:s+1]
        qPrices = callPrices[s-1:s+1]
        qvols = @. impliedVolatilitySRHalley(true, qPrices, forward, qStrikes, tte, 1.0, 0e-14, 64, Householder())
        lagrange = QuadraticLagrangePP(qStrikes,qvols, knotStyle=PPInterpolation.LEFT_KNOT)
        fVol = lagrange(forward)
        fPrice = blackScholesFormula(true, forward, forward, fVol^2*tte,1.0,1.0)
        strikes = vcat(strikes[1:s-1],forward,strikes[s:end])
        callPrices = vcat(callPrices[1:s-1],fPrice,callPrices[s:end])
        weights = vcat(weights[1:s-1],weights[s],weights[s:end])
    end
    x = vcat(L, strikes, U)
    m = length(x) - 1
    #initial guess
    vols = zeros(T, length(strikes))
    if useVol
        for i = eachindex(vols)
            vols[i] = impliedVolatilitySRHalley(true, callPrices[i], forward, strikes[i], tte, 1.0, 0e-14, 64, Householder())
        end
    end
    atmVol = impliedVolatilitySRHalley(true, callPrices[s], forward, forward, tte, 1.0, 0e-14, 64, Householder())
    a0 = atmVol*sqrt(tte/2)
    minValue = sqrt(2 ) / 400
    maxValue = 1.0 / (4minValue)
    #println("forward found ", strikes[s], " min ", minValue, " max ", maxValue)

    transform = ClosedTransformation(minValue, maxValue)
    iter = 0
    function toQuadForm(σ::AbstractArray{W}; model=model) where {W}
        a = zeros(W,m)
        b= zeros(W,m)
        c = zeros(W, m)
       if model == "Linear"
            for i = 1:m
                # b[i] = (σ[i+1] - σ[i]) / (x[i+1] - x[i])
                # a[i] = -x[i] * b[i] + σ[i]
                b[i] = (σ[i+1]*x[i+1] - σ[i]*x[i]) / (x[i+1] - x[i])
                a[i] = -x[i] * b[i] + σ[i]*x[i]
            end
        elseif model == "LinearBlack"
            for i = 1:m
                bi =  (σ[i+1] - σ[i]) / (x[i+1] - x[i])
                b[i] = -x[i] * bi + σ[i]
                # b[i] = one(W)
                c[i] = bi/b[i]
            end
        elseif model=="Quadratic"
            pp = makeQuadraticSpline(x,σ.*x)
            for i=1:m
                #r = Polynomials.roots(Polynomials.Polynomial([pp.a[i]-pp.b[i]*x[i]+pp.c[i]*x[i]^2,pp.b[i]-2*pp.c[i]*x[i],pp.c[i]]))
                CC = pp.a[i]-pp.b[i]*x[i]+pp.c[i]*x[i]^2
                BB = pp.b[i]-2*pp.c[i]*x[i]
                AA = pp.c[i]
                if AA == zero(W)
                    #BB x + CC
                    a[i] = CC
                    b[i] = BB
                    c[i] = zero(W)
                else
                Δ = Complex(BB^2 - 4*AA*CC)
                r1 = (-BB - sqrt(Δ))/(2AA)
                r2 = (-BB + sqrt(Δ))/(2AA)
                #println(i, " roots ",r1, " ",r2)
                a[i] = r1*r2*pp.c[i]
                b[i] = -r1*pp.c[i]
                c[i] = -one(W)/r1
                end
            end
        #TODO compute quadratic spline of σ, transform to (a+bx)*(1+cx) form.
        end 
        #println("abc",a," ",b," ",c)
        return a,b,c
    end
    function obj!(fvec::Z, coeff::AbstractArray{W})::Z where {Z,W}
        n = length(strikes)
        σ = zeros(W, m + 1)
        @. σ[2:m] = transform(coeff)
        σ[1] = σ[2]
        σ[m+1] = σ[m] #/log(x[2]/x[1])*log(x[3]/x[2])               
        a,b,c = toQuadForm(σ,model=model)
        θ = zeros(W, 2m)
        # print("abc ",a," ",b," ",c)
        computeCoeffsQuadratic(s, tte, x, a,b,c, θ)
        # println("θ ", θ)
        if useVol            
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
         if penalty > 0
            sumw = zero(T)
            v = zeros(W,n)
            for i = 1:n
                density = θ[2i+2] / ((a[i+1]+b[i+1]*strikes[i])*(1+c[i+1]*strikes[i]) )
                v[i] = log(density)
                sumw += weights[i]
            end
            for i = 2:n-1
                fvec[i+n-1] = sumw * penalty * ((v[i+1]-v[i])/(strikes[i+1]-strikes[i]) - (v[i]-v[i-1])/(strikes[i]-strikes[i-1]))
			end
        end     
        #     ip = hermiteIntegral(derivative(p, 2)^2)
        #     pvalue = penalty * ip
        #     fvec[n+1] = pvalue
        # end
        iter += 1
        fvec
    end
    σ = zeros(T, m + 1)
    for i = eachindex(σ)
        σ[i] = inv(transform, a0)
    end
    x0 = σ[2:m]
    outlen = length(callPrices)
    if penalty > 0
        outlen += length(strikes)-2
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
    @. σ[2:m] = transform(fit.minimizer)
    σ[1] = σ[2] #/log(x[2]/x[1])*log(x[3]/x[2])
    σ[m+1] = σ[m]
    a,b,c = toQuadForm(σ)
    θ = zeros(T, 2m)
    computeCoeffsQuadratic(s, tte, x, a,b,c, θ)
    return QuadraticLVG(x, a,b,c, θ, tte, forward)
end

using PPInterpolation

function makeQuadraticSpline(x::AbstractArray{TX},y::AbstractArray{T};leftDerivative::T=zero(T)) where {T, TX}
    pp = PP(2,T,TX,length(y))
    pp.x .= x
    pp.a .= y    
    pp.b[1] = leftDerivative
    for i=1:length(x)-1
        pp.b[i+1] = (y[i+1]-y[i])*2/(x[i+1]-x[i]) -pp.b[i]
        pp.c[i,1] = (pp.b[i+1]-pp.b[i])/((x[i+1]-x[i])*2)
    end
    return pp
end
