using LinearAlgebra
using LeastSquaresOptim
using GaussNewton, LsqFit, Optim, Random
import AQFED.Black: blackScholesFormula, impliedVolatilitySRHalley, Householder, impliedVolatilityLiSOR, SORTS
import AQFED.Math: ClosedTransformation, inv, MQMinTransformation, IdentityTransformation, LogisticTransformation, AlgebraicTransformation, TanhTransformation,AtanTransformation
import PPInterpolation: PP, evaluatePiece, evaluateDerivativePiece, evaluateSecondDerivativePiece, LEFT_KNOT, QuadraticLagrangePP, CubicSplineNatural

struct QuadraticLVG{TV,T}
    x::Vector{T}
    a::Vector{TV} #ax^2 + bx + c
    b::Vector{TV}
    c::Vector{TV}
    θ::Vector{TV}
    tte::T
    forward::T
end

abstract type LVGKind end
struct LinearBachelier <: LVGKind end
struct LinearBlack <: LVGKind end
struct Quadratic <: LVGKind end

Base.broadcastable(p::QuadraticLVG) = Ref(p)


function priceEuropean(model::QuadraticLVG{TV,T}, isCall::Bool, strike) where {TV,T}
    x = model.x
    if strike <= x[1] || strike >= x[end]
        return priceEuropeanPiece(0, model, isCall, strike)
    end
    i = max(searchsortedlast(x, strike), 1)
    return priceEuropeanPiece(i, model, isCall, strike)
end

function imlog(zzi::Complex{T})::T where {T}
    # imzzi = asin(imag(zzi))
    # if real(zzi) < zero(T)
    #     imzzi = -imzzi-pi
    # end
    # return imzzi
    return imag(log(zzi))
end

function priceEuropeanPiece(i::Int, model::QuadraticLVG{TV,T}, isCall::Bool, strike) where {TV,T}
    if i <= 0 || i > length(model.a)
        return isCall ? max(model.forward - strike, 0.0) : max(strike - model.forward, 0.0)
    end
    β = model.b[i]
    α = model.a[i]
    γ = model.c[i]
    xi = model.x[i]
    value = if iszero(α) && iszero(β)
        zzi = strike - xi
        χ = one(T)
        ω = one(T) / γ
        u = ω * zzi
        # println(u," ",model.θ[2i]," ", model.θ[2i-1], " ",cosh(u))    
        # χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))
        χ * (model.θ[2i] + model.θ[2i-1] - model.θ[2i-1] * 2 / (1 + exp(2u))) * cosh(u)
    elseif iszero(α)
        δ = β^2
        x1 = -γ / β  # beta (x-x1) = beta x + gamma
        zzi = (strike - x1) / (xi - x1)
        χ = sqrt(abs(zzi))#sqrt((x-x1)*(x-x2))
        Δ = δ # α^2*(x1-x2)^2
        ω = sqrt(4 / Δ + 1) / 2
        u = ω * log(abs(zzi)) # zi^w        
        #  χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))         
        χ * (model.θ[2i] + model.θ[2i-1] - model.θ[2i-1] * 2 / (1 + exp(2u))) * cosh(u)
    else
        δ = β^2 - 4α * γ
        ω2 = one(T) / δ + one(T) / 4
        ω = sqrt(abs(ω2))
        χ = sqrt(abs((strike^2 + β / α * strike + γ / α) / (xi^2 + β / α * xi + γ / α)))#sqrt((x-x1)*(x-x2))
        #println(i," ",δ, " ",ω)        
        v = if δ >= 0
            x1 = (-β - sqrt(abs(δ))) / (2α)
            x2 = (-β + sqrt(abs(δ))) / (2α)
            zzi = (strike - x1) / (strike - x2) * (xi - x2) / (xi - x1)
            u = ω * log(abs(zzi))
            # χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))
            χ * (model.θ[2i] - model.θ[2i-1] + model.θ[2i-1] * 2 / (1 + exp(-2u))) * cosh(u)
        else
            x1 = (-β - sqrt(complex(δ))) / (2α)
            x2 = (-β + sqrt(complex(δ))) / (2α)
            zzi = (strike - x1) / (strike - x2) * (xi - x2) / (xi - x1)
            imzzi = imlog(zzi)
            u = ω * imzzi
            if ω2 >= 0
                χ * (model.θ[2i] * cos(u) + model.θ[2i-1] * sin(-u))
            else
                # println(u," ",model.θ[2i]," ", model.θ[2i-1], " ",cosh(u))    
                # χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))
                # χ * (model.θ[2i]- model.θ[2i-1] + model.θ[2i-1](1+tanh(u))*cosh(u)
                χ * (model.θ[2i] - model.θ[2i-1] + model.θ[2i-1] * 2 / (1 + exp(-2u))) * cosh(u)
            end
        end
        v
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
    β = model.b[i]
    α = model.a[i]
    γ = model.c[i]
    xi = x[i]
    value = if α == zero(TV) && β == zero(TV)
        zzi = strike - xi
        #χ = one(T)
        ω = one(T) / γ
        u = ω * zzi
        sh, ch = sinh(u), cosh(u)
        ω * ((model.θ[2i]) * sh + (model.θ[2i-1]) * ch)
    elseif α == zero(TV)
        δ = β^2
        x1 = -γ / β  # beta (x-x1) = beta x + gamma
        zzi = abs((strike - x1) / (xi - x1))
        χ = sqrt(zzi)#sqrt((x-x1)*(x-x2))
        ω = sqrt(4 / δ + 1) / 2
        u = ω * log(zzi)
        dz = one(T) / (strike - x1)
        κ = one(TV) / 2
        sh, ch = sinh(u), cosh(u)
        χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) + ω * (model.θ[2i-1])) * ch)
    else
        δ = β^2 - 4α * γ
        ω2 = one(T) / δ + one(T) / 4
        ω = sqrt(abs(ω2))
        χ = sqrt(abs((strike^2 + β / α * strike + γ / α) / (xi^2 + β / α * xi + γ / α)))#sqrt((x-x1)*(x-x2))
        # println(i, " ", δ, " ", ω2)
        if δ >= 0

            x1 = (-β - sqrt(abs(δ))) / (2α)
            x2 = (-β + sqrt(abs(δ))) / (2α)
            zzi = abs((strike - x1) / (strike - x2) * (xi - x2) / (xi - x1))
            dz = one(T) / (strike - x1) - one(T) / (strike - x2)
            κ = (one(T) / (strike - x1) + one(T) / (strike - x2)) / (2dz)
            u = ω * log(zzi)
            sh, ch = sinh(u), cosh(u)
            χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) + ω * (model.θ[2i-1])) * ch)
        else
            x1 = (-β - sqrt(complex(δ))) / (2α)
            x2 = (-β + sqrt(complex(δ))) / (2α)
            zzi = (strike - x1) / (strike - x2) * (xi - x2) / (xi - x1)
            # println("dz ",(one(T) / (strike - x1) - one(T) / (strike - x2)))
            dz = imag(one(T) / (strike - x1) - one(T) / (strike - x2))
            κ = real((one(T) / (strike - x1) + one(T) / (strike - x2))) / 2dz
            # kappa = strike-x2 + strike - X1 / 2(strike-x2 - strike-x1)= (2strike+2beta/2alpha) / 2(-2sqrt(delta)/2alpha)
            imzzi = imlog(zzi)
            u = ω * imzzi
            v = if ω2 >= 0
                sh, ch = sin(-u), cos(u)
                χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) - ω * (model.θ[2i-1])) * ch)
            else
                sh, ch = sinh(u), cosh(u)
                χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) + ω * (model.θ[2i-1])) * ch)
            end
            # println(v, " " ,u," ",zzi," COSH ",ch," ",sh," ",κ, " ",dz)
            v
        end
    end
    if isCall && strike < model.forward
        value -= 1
    elseif !isCall && strike > model.forward
        value += 1
    end
    return value
end

function computeCoeffsQuadratic(s::Int, tte::T, strikes::AbstractVector{T}, α::AbstractVector{TC}, β::AbstractVector{TC}, γ::AbstractVector{TC}, θ::AbstractVector{TA}; tri=Tridiagonal(zeros(TA, 2(length(strikes) - 1) - 1), zeros(TA, 2(length(strikes) - 1)), zeros(TA, 2(length(strikes) - 1) - 1))) where {T,TA,TC}
    #tri = Tridiagonal(zeros(TC, 2(length(strikes) - 1) - 1), zeros(TC, 2(length(strikes) - 1)), zeros(TC, 2(length(strikes) - 1) - 1))
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
        local χi, zdi, κi, ωi, si, coshi, sinhi
        if iszero(α[i]) && iszero(β[i])
            χi = one(T)
            ωi = one(T) / γ[i]
            κi = zero(T)
            zzi = strikes[i+1] - strikes[i]
            zdi = one(T)
            si = one(T)
            u = ωi * zzi
            sinhi, coshi = sinh(u), cosh(u)
        elseif iszero(α[i])
            δ = β[i]^2
            x1 = -γ[i] / β[i]  # beta (x-x1) = beta x + gamma
            χi = sqrt(abs((strikes[i+1] - x1) / (strikes[i] - x1)))#sqrt((x-x1)*(x-x2))
            zdi = one(T) / (strikes[i+1] - x1)
            κi = one(T) / 2
            ωi = sqrt(4 / δ + 1) / 2
            si = one(T)
            zzi = abs((strikes[i+1] - x1) / (strikes[i] - x1))
            u = ωi * log(zzi)
            sinhi, coshi = sinh(u), cosh(u)
        else

            δ = β[i]^2 - 4α[i] * γ[i]
            ω2 = one(T) / δ + one(T) / 4
            ωi = sqrt(abs(ω2))
            χi = sqrt(abs((strikes[i+1]^2 + β[i] / α[i] * strikes[i+1] + γ[i] / α[i]) / (strikes[i]^2 + β[i] / α[i] * strikes[i] + γ[i] / α[i])))#sqrt((x-x1)*(x-x2))
            if δ >= 0
                x1 = (-β[i] - sqrt(abs(δ))) / (2α[i])
                x2 = (-β[i] + sqrt(abs(δ))) / (2α[i])
                zzi = abs((strikes[i+1] - x1) / (strikes[i+1] - x2) * (strikes[i] - x2) / (strikes[i] - x1))
                zdi = one(T) / (strikes[i+1] - x1) - one(T) / (strikes[i+1] - x2)
                κi = (one(T) / (strikes[i+1] - x1) + one(T) / (strikes[i+1] - x2)) / (2zdi)
                u = ωi * log(zzi)
                si = one(T)
                sinhi, coshi = sinh(u), cosh(u)
                #                χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) + ω * (model.θ[2i-1])) * ch)
            else
                x1 = (-β[i] - sqrt(complex(δ))) / (2α[i])
                x2 = (-β[i] + sqrt(complex(δ))) / (2α[i])
                zzi = (strikes[i+1] - x1) / (strikes[i+1] - x2) * (strikes[i] - x2) / (strikes[i] - x1)
                # println("dz ",(one(T) / (strike - x1) - one(T) / (strike - x2)))
                zdi = imag(one(T) / (strikes[i+1] - x1) - one(T) / (strikes[i+1] - x2))
                κi = real((one(T) / (strikes[i+1] - x1) + one(T) / (strikes[i+1] - x2))) / 2zdi
                # kappa = strike-x2 + strike - X1 / 2(strike-x2 - strike-x1)= (2strike+2beta/2alpha) / 2(-2sqrt(delta)/2alpha)
                imzzi = imlog(zzi)
                u = ωi * imzzi
                if ω2 >= 0
                    si = -one(T)
                    sinhi, coshi = sin(-u), cos(u)
                    #χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) - ω * (model.θ[2i-1])) * ch)
                else
                    si = one(T)
                    sinhi, coshi = sinh(u), cosh(u)
                    # χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) + ω * (model.θ[2i-1])) * ch)
                end
                #println(ch," ",sh," ",κdz," ",((κdz * (model.θ[2i-1]) + dz *ω * (model.θ[2i])) * sh))
            end
        end
        local zdip, κip, ωip, sip
        if iszero(α[i+1]) && iszero(β[i+1])
            ωip = one(T) / γ[i+1]
            κip = zero(T)
            zdip = one(T)
            sip = one(T)
        elseif iszero(α[i+1])
            δp = β[i+1]^2
            x1p = -γ[i+1] / β[i+1]  # beta (x-x1) = beta x + gamma
            zdip = one(T) / (strikes[i+1] - x1p)
            κip = one(T) / 2
            ωip = sqrt(4 / δp + 1) / 2
            sip = one(T)
        else
            δp = β[i+1]^2 - 4α[i+1] * γ[i+1]
            ω2 = one(T) / δp + one(T) / 4
            ωip = sqrt(abs(ω2))
            if δp >= 0
                x1 = (-β[i+1] - sqrt(abs(δp))) / (2α[i+1])
                x2 = (-β[i+1] + sqrt(abs(δp))) / (2α[i+1])
                zdip = one(T) / (strikes[i+1] - x1) - one(T) / (strikes[i+1] - x2)
                κip = (one(T) / (strikes[i+1] - x1) + one(T) / (strikes[i+1] - x2)) / (2zdip)
                sip = one(T)
            else
                x1 = (-β[i+1] - sqrt(complex(δp))) / (2α[i+1])
                x2 = (-β[i+1] + sqrt(complex(δp))) / (2α[i+1])
                zdip = imag(one(T) / (strikes[i+1] - x1) - one(T) / (strikes[i+1] - x2))
                κip = real((one(T) / (strikes[i+1] - x1) + one(T) / (strikes[i+1] - x2))) / 2zdip
                if ω2 >= 0
                    sip = -one(T)
                else
                    sip = one(T)
                end
                #println(ch," ",sh," ",κdz," ",((κdz * (model.θ[2i-1]) + dz *ω * (model.θ[2i])) * sh))
            end
        end
#    χ * (model.θ[2i] + model.θ[2i-1] - model.θ[2i-1] * 2 / (1 + exp(2u))) * cosh(u)
    
        lhsdv = (κi * coshi + ωi * sinhi) * χi * zdi - κip * coshi * χi * zdip
        lhsd[2i] = lhsdv  #2i
        lhsdl[2i-1] = (si * ωi * coshi + κi * sinhi) * χi * zdi - κip * sinhi * χi * zdip   #2i-1
        lhsdu[2i] = -sip * ωip * zdip                                      #2i+1
        #line 2(i-1)+2    
        lhsd[2i+1] = -sip * ωip * zdip
        lhsdl[2i] = (κi * coshi + ωi * sinhi - (si * ωi * coshi + κi * sinhi) / sinhi * coshi) * χi * zdi
        lhsdu[2i+1] = zdi * (si * ωi * coshi + κi * sinhi) / (sinhi) - κip * zdip
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
    local tanhm
    if iszero(α[m]) && iszero(β[m])
        #  println("constant extrap ",γ[m])
        ωm = one(T) / γ[m]
        zzi = strikes[m+1] - strikes[m]
        tanhm = tanh(ωm * zzi)
    elseif iszero(α[m])
        #println("constant extrap ", γ[m], " ", β[m])
        δ = β[m]^2
        x1 = -γ[m] / β[m]  # beta (x-x1) = beta x + gamma
        ωm = sqrt(4 / δ + 1) / 2
        zzi = log(abs((strikes[m+1] - x1) / (strikes[m] - x1)))
        tanhm = tanh(ωm * zzi)
    else
        δ = β[m]^2 - 4α[m] * γ[m]
        ω2 = one(T) / δ + one(T) / 4
        ωm = sqrt(abs(ω2))
        tanhm = if δ >= 0
            x1 = (-β[m] - sqrt(abs(δ))) / (2α[m])
            x2 = (-β[m] + sqrt(abs(δ))) / (2α[m])
            zzi = abs((strikes[m+1] - x1) / (strikes[m+1] - x2) * (strikes[m] - x2) / (strikes[m] - x1))
            u = ωm * log(zzi)
            tanh(u)
        else
            x1 = (-β[m] - sqrt(complex(δ))) / (2α[m])
            x2 = (-β[m] + sqrt(complex(δ))) / (2α[m])
            zzi = (strikes[m+1] - x1) / (strikes[m+1] - x2) * (strikes[m] - x2) / (strikes[m] - x1)
            imzzi = imlog(zzi)
            u = ωm * imzzi
            if ω2 >= 0
                tan(-u)
            else
                tanh(u)
            end
            #println(ch," ",sh," ",κdz," ",((κdz * (model.θ[2i-1]) + dz *ω * (model.θ[2i])) * sh))
        end
    end
    lhsd[2m] = one(TA)
    lhsdl[2m-1] = tanhm
    if isinf(tanhm)
        println(ωm, " ", zzi, " ", β[m], " ", α[m])
        throw(DomainError(sinhm, "sinhm must be finite"))
    end
    rhs[2m] = zero(T)

    #    println("solving with a=",γ," lhsdl=",lhsdl," lhsd=",lhsd," lhsdu",lhsdu)
    lhsf = lu!(tri)
    ##  lhsf = factorize(Tridiagonal(lhsdl, lhsd, lhsdu))
    ldiv!(θ, lhsf, rhs)
    # println("found solution ",θ) #theta is either full real for full im. Simplif are possible
    # θ[:] = Tridiagonal(lhsdl, lhsd, lhsdu) \ rhs
end

function computeCoeffsQuadraticDirect(s::Int, tte::T, strikes::AbstractVector{T}, α::AbstractVector{TC}, β::AbstractVector{TC}, γ::AbstractVector{TC}, θ::AbstractVector{TA}; tri=Tridiagonal(zeros(TA, 2(length(strikes) - 1) - 1), zeros(TA, 2(length(strikes) - 1)), zeros(TA, 2(length(strikes) - 1) - 1))) where {T,TA,TC}
    #tri = Tridiagonal(zeros(TC, 2(length(strikes) - 1) - 1), zeros(TC, 2(length(strikes) - 1)), zeros(TC, 2(length(strikes) - 1) - 1))
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
        local χi, zdi, κi, ωi, si, coshi, sinhi
        if iszero(α[i]) && iszero(β[i])
            χi = one(T)
            ωi = one(T) / γ[i]
            κi = zero(T)
            zzi = strikes[i+1] - strikes[i]
            zdi = one(T)
            si = one(T)
            u = ωi * zzi
            sinhi, coshi = sinh(u), cosh(u)
        elseif iszero(α[i])
            δ = β[i]^2
            x1 = -γ[i] / β[i]  # beta (x-x1) = beta x + gamma
            χi = sqrt(abs((strikes[i+1] - x1) / (strikes[i] - x1)))#sqrt((x-x1)*(x-x2))
            zdi = one(T) / (strikes[i+1] - x1)
            κi = one(T) / 2
            ωi = sqrt(4 / δ + 1) / 2
            si = one(T)
            zzi = abs((strikes[i+1] - x1) / (strikes[i] - x1))
            u = ωi * log(zzi)
            sinhi, coshi = sinh(u), cosh(u)
        else

            δ = β[i]^2 - 4α[i] * γ[i]
            ω2 = one(T) / δ + one(T) / 4
            ωi = sqrt(abs(ω2))
            χi = sqrt(abs((strikes[i+1]^2 + β[i] / α[i] * strikes[i+1] + γ[i] / α[i]) / (strikes[i]^2 + β[i] / α[i] * strikes[i] + γ[i] / α[i])))#sqrt((x-x1)*(x-x2))
            if δ >= 0
                x1 = (-β[i] - sqrt(abs(δ))) / (2α[i])
                x2 = (-β[i] + sqrt(abs(δ))) / (2α[i])
                zzi = abs((strikes[i+1] - x1) / (strikes[i+1] - x2) * (strikes[i] - x2) / (strikes[i] - x1))
                zdi = one(T) / (strikes[i+1] - x1) - one(T) / (strikes[i+1] - x2)
                κi = (one(T) / (strikes[i+1] - x1) + one(T) / (strikes[i+1] - x2)) / (2zdi)
                u = ωi * log(zzi)
                si = one(T)
                sinhi, coshi = sinh(u), cosh(u)
                #                χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) + ω * (model.θ[2i-1])) * ch)
            else
                x1 = (-β[i] - sqrt(complex(δ))) / (2α[i])
                x2 = (-β[i] + sqrt(complex(δ))) / (2α[i])
                zzi = (strikes[i+1] - x1) / (strikes[i+1] - x2) * (strikes[i] - x2) / (strikes[i] - x1)
                # println("dz ",(one(T) / (strike - x1) - one(T) / (strike - x2)))
                zdi = imag(one(T) / (strikes[i+1] - x1) - one(T) / (strikes[i+1] - x2))
                κi = real((one(T) / (strikes[i+1] - x1) + one(T) / (strikes[i+1] - x2))) / 2zdi
                # kappa = strike-x2 + strike - X1 / 2(strike-x2 - strike-x1)= (2strike+2beta/2alpha) / 2(-2sqrt(delta)/2alpha)
                imzzi = imlog(zzi)
                u = ωi * imzzi
                if ω2 >= 0
                    si = -one(T)
                    sinhi, coshi = sin(-u), cos(u)
                    #χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) - ω * (model.θ[2i-1])) * ch)
                else
                    si = one(T)
                    sinhi, coshi = sinh(u), cosh(u)
                    # χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) + ω * (model.θ[2i-1])) * ch)
                end
                #println(ch," ",sh," ",κdz," ",((κdz * (model.θ[2i-1]) + dz *ω * (model.θ[2i])) * sh))
            end
        end
        local zdip, κip, ωip, sip
        if iszero(α[i+1]) && iszero(β[i+1])
            ωip = one(T) / γ[i+1]
            κip = zero(T)
            zdip = one(T)
            sip = one(T)
        elseif iszero(α[i+1])
            δp = β[i+1]^2
            x1p = -γ[i+1] / β[i+1]  # beta (x-x1) = beta x + gamma
            zdip = one(T) / (strikes[i+1] - x1p)
            κip = one(T) / 2
            ωip = sqrt(4 / δp + 1) / 2
            sip = one(T)
        else
            δp = β[i+1]^2 - 4α[i+1] * γ[i+1]
            ω2 = one(T) / δp + one(T) / 4
            ωip = sqrt(abs(ω2))
            if δp >= 0
                x1 = (-β[i+1] - sqrt(abs(δp))) / (2α[i+1])
                x2 = (-β[i+1] + sqrt(abs(δp))) / (2α[i+1])
                zdip = one(T) / (strikes[i+1] - x1) - one(T) / (strikes[i+1] - x2)
                κip = (one(T) / (strikes[i+1] - x1) + one(T) / (strikes[i+1] - x2)) / (2zdip)
                sip = one(T)
            else
                x1 = (-β[i+1] - sqrt(complex(δp))) / (2α[i+1])
                x2 = (-β[i+1] + sqrt(complex(δp))) / (2α[i+1])
                zdip = imag(one(T) / (strikes[i+1] - x1) - one(T) / (strikes[i+1] - x2))
                κip = real((one(T) / (strikes[i+1] - x1) + one(T) / (strikes[i+1] - x2))) / 2zdip
                if ω2 >= 0
                    sip = -one(T)
                else
                    sip = one(T)
                end
                #println(ch," ",sh," ",κdz," ",((κdz * (model.θ[2i-1]) + dz *ω * (model.θ[2i])) * sh))
            end
        end

        lhsdv = (κi * coshi + ωi * sinhi) * χi * zdi - κip * coshi * χi * zdip
        lhsd[2i] = lhsdv  #2i
        lhsdl[2i-1] = (si * ωi * coshi + κi * sinhi) * χi * zdi - κip * sinhi * χi * zdip   #2i-1
        lhsdu[2i] = -sip * ωip * zdip                                      #2i+1
        #line 2(i-1)+2    
        lhsd[2i+1] = -sip * ωip * zdip
        lhsdl[2i] = (κi * coshi + ωi * sinhi - (si * ωi * coshi + κi * sinhi) / sinhi * coshi) * χi * zdi
        lhsdu[2i+1] = zdi * (si * ωi * coshi + κi * sinhi) / (sinhi) - κip * zdip
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
    local tanhm
    if iszero(α[m]) && iszero(β[m])
        #  println("constant extrap ",γ[m])
        ωm = one(T) / γ[m]
        zzi = strikes[m+1] - strikes[m]
        tanhm = tanh(ωm * zzi)
    elseif iszero(α[m])
        println("constant extrap ", γ[m], " ", β[m])
        δ = β[m]^2
        x1 = -γ[m] / β[m]  # beta (x-x1) = beta x + gamma
        ωm = sqrt(4 / δ + 1) / 2
        zzi = log(abs((strikes[m+1] - x1) / (strikes[m] - x1)))
        tanhm = tanh(ωm * zzi)
    else
        δ = β[m]^2 - 4α[m] * γ[m]
        ω2 = one(T) / δ + one(T) / 4
        ωm = sqrt(abs(ω2))
        tanhm = if δ >= 0
            x1 = (-β[m] - sqrt(abs(δ))) / (2α[m])
            x2 = (-β[m] + sqrt(abs(δ))) / (2α[m])
            zzi = abs((strikes[m+1] - x1) / (strikes[m+1] - x2) * (strikes[m] - x2) / (strikes[m] - x1))
            u = ωm * log(zzi)
            tanh(u)
        else
            x1 = (-β[m] - sqrt(complex(δ))) / (2α[m])
            x2 = (-β[m] + sqrt(complex(δ))) / (2α[m])
            zzi = (strikes[m+1] - x1) / (strikes[m+1] - x2) * (strikes[m] - x2) / (strikes[m] - x1)
            imzzi = imlog(zzi)
            u = ωm * imzzi
            if ω2 >= 0
                tan(-u)
            else
                tanh(u)
            end
            #println(ch," ",sh," ",κdz," ",((κdz * (model.θ[2i-1]) + dz *ω * (model.θ[2i])) * sh))
        end
    end
    lhsd[2m] = one(TA)
    lhsdl[2m-1] = tanhm
    if isinf(tanhm)
        println(ωm, " ", zzi, " ", β[m], " ", α[m])
        throw(DomainError(sinhm, "sinhm must be finite"))
    end
    rhs[2m] = zero(T)

    #    println("solving with a=",γ," lhsdl=",lhsdl," lhsd=",lhsd," lhsdu",lhsdu)
    lhsf = lu!(tri)
    ##  lhsf = factorize(Tridiagonal(lhsdl, lhsd, lhsdu))
    ldiv!(θ, lhsf, rhs)
    # println("found solution ",θ) #theta is either full real for full im. Simplif are possible
    # θ[:] = Tridiagonal(lhsdl, lhsd, lhsdu) \ rhs
end


using BSplines
function makeBasis(model, x, forward, isC3)
    x
end
function makeBasis(model::Quadratic, x, forward, isC3)
    if isC3
        sort(vcat(x, forward))
    else
        x
    end
end

function toQuadForm!(model::LinearBachelier, m, l, s, forward, x, α::AbstractArray{W}, β::AbstractArray{W}, γ::AbstractArray{W}, σ::AbstractArray{W}, priceAtm; isC3=false, basis=x) where {W}
    if m + 1 == l + 2
        λ = vcat(σ[1], σ[1:l], σ[l])
    else
        λ = vcat(σ[1], σ[1:s], σ[s:l], σ[l])
        sp = s + 1
        λ[sp] = ((λ[sp+1] / (x[sp+1] - x[sp]) + λ[sp-1] / (x[sp] - x[sp-1]))) / (-1 / (2priceAtm) + (1 / (x[sp] - x[sp-1]) + 1 / (x[sp+1] - x[sp])))
    end
    # println("linearbach lambda ",λ," ", priceAtm)
    for i = 1:m
        β[i] = (λ[i+1] * forward - λ[i] * forward) / (x[i+1] - x[i])
        γ[i] = -x[i] * β[i] + λ[i] * forward
    end
    return α, β, γ
end
function toQuadForm!(model::LinearBlack, m, l, s, forward, x, α::AbstractArray{W}, β::AbstractArray{W}, γ::AbstractArray{W}, σ::AbstractArray{W}, priceAtm; isC3=false, basis=x) where {W}
    if m + 1 == l + 2
        λ = vcat(σ[1], σ[1:l], σ[l])
    else
        λ = vcat(σ[1], σ[1:s], σ[s:l], σ[l])
        sp = s + 1
        λ[sp] = -((λ[sp+1] / (x[sp+1] - x[sp]) + λ[sp-1] / (x[sp] - x[sp-1]))) / (1 / (2priceAtm) - (1 / (x[sp] - x[sp-1]) + 1 / (x[sp+1] - x[sp])))
       # println("overriding with ", λ[sp])
        #     σ[sp]/=x[sp]
    end
    for i = 1:m
        bi = (λ[i+1] - λ[i]) / (x[i+1] - x[i])
        β[i] = -x[i] * bi + λ[i]
        α[i] = bi
    end
    return α, β, γ
end
function toQuadForm!(model::Quadratic, m, l, s, forward, x, α::AbstractArray{W}, β::AbstractArray{W}, γ::AbstractArray{W}, σ::AbstractArray{W}, priceAtm; isC3=false, basis=x) where {W}
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
        knots = BSplines.knots(basis)
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
    # println("abc ",α ," ",β," ",γ)
    return α, β, γ
end

function numberOfFreeParameters(model::LVGKind, xLen::Int, pricesLen::Int, isC3::Bool)
    if isC3
        min(xLen - 3, pricesLen)
    else
        min(xLen - 2, pricesLen)
    end
end

function numberOfFreeParameters(model::Quadratic, xLen::Int, pricesLen::Int, isC3::Bool)
    min(xLen - 1, pricesLen)
end

function adjustKnots!(x, s, priceAtm; factor=2)
    sp = s + 1
    if -1 / (factor * priceAtm) + (1 / (x[sp] - x[sp-1]) + 1 / (x[sp+1] - x[sp])) <= 0
        h = min(x[sp] - x[sp-1], x[sp+1] - x[sp])
        #try out if h is ok
         println("adjusted knots before ",x, " ",h)
        if -1 / (factor * priceAtm) + (2 / h) <= 0
            # 2/h < 1/priceAtm == h > 2*priceAtm
            h = 1.5 * factor * priceAtm
            # println("adjusted knots before ",x, " ",h)
        end
        x[sp+1] = x[sp] + h
        x[sp-1] = x[sp] - h
        # println("adjusted knots ",x)
        #adjust x[sp-1] and x[sp+1]
    end
    return x
end


function calibrateQuadraticLVG(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; U::T=6 * forward, L::T=forward / 6, useVol=false, model::LVGKind=LinearBachelier(), penalty=zero(T), nRefine=3, location="Mid-XX", isC3=true, minCount=1, size=0, guess="Constant", optimizer="LM") where {T}

    #if the forward is not part of the strikes, add the forward and interpolate prices with quadratic. Eventually set corresponding weight to 0
    if U <= strikes[end]
        U = strikes[end] + 3 * (strikes[end] - forward)
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
    x = makeKnots(tte, forward, strikes, size, location=location, minCount=minCount, L=L, U=U)
    s = searchsortedfirst(x, forward)
    if !useForwardInStrikes && x[s] != forward
        x = sort(vcat(x, forward))
    end
    s = searchsortedfirst(x, forward) - 1
    adjustKnots!(x, s, priceAtm)
    xBasis = makeBasis(model, x, forward, isC3)
    m = length(x) - 1
    #initial guess
    vols = zeros(T, length(strikes))
    if useVol || guess != "Constant"
        for i = eachindex(vols)
            vols[i] = impliedVolatilitySRHalley(true, callPrices[i], forward, strikes[i], tte, 1.0, 0e-14, 64, Householder())
        end
    end


    # minValue = sqrt(2 ) / 400
    # minValue = sqrt(2)/2000
    # maxValue = 1/(4minValue*tte)
    # minValue = 0.0039706054061558775 *sqrt(tte)
    # maxValue = 397.0605406155877*sqrt(tte)
    # if model == "LinearBlack"
    #     a0*=forward
    #     # minValue*=10
    #     maxValue*=forward
    # end
    #    
    iter = 0
    basis = BSplineBasis(3, xBasis)
    #println("xBasis ", xBasis, " knots ", knots, "len knoths=", length(knots))
    #strikeIndices = indices of strike in x. (PP version) x will contain forward.
    strikeIndices = max.(searchsortedlast.(Ref(x), strikes), 1)  # x[i]<=z<=x[i+1]
    l = numberOfFreeParameters(model, length(x), length(callPrices), isC3)

    atmVol = impliedVolatilitySRHalley(true, priceAtm, forward, forward, tte, 1.0, 0e-14, 64, Householder())
    a0 = atmVol * sqrt(tte / 2)
    minValue = a0 / 10#FIXME TODO better initial guess (from cubic spline density) - does it improve over constant?
    maxValue = a0 * 800
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
            @.(sqrt(min((a0*10)^2 , max((a0/5)^2, (xPrices - xIntrinsic) / (forward^2 * (xPricesUp - 2xPrices + xPricesDown) / epsDens^2)))))[1:end] 
        end
    transform =
        if optimizer == "LM-MQ" || optimizer == "GN-MQ"
            MQMinTransformation(minValue, 1.0) #does not maex the iter, but more iter in general than raw.
        elseif optimizer == "LM-SIN" || optimizer == "GN-SIN" || optimizer == "GN" 
            ClosedTransformation(minValue, maxValue) #leads to best result, but often max iterations
        elseif optimizer == "LM" || optimizer == "LM-Curve"
            IdentityTransformation{T}()
        elseif optimizer == "LM-LOG" || optimizer == "GN-LOG"
            LogisticTransformation(minValue, maxValue)
        elseif optimizer == "LM-ALG" || optimizer == "GN-ALG"
            AlgebraicTransformation(minValue, maxValue)
        elseif optimizer == "LM-TANH" || optimizer == "GN-TANH"
            TanhTransformation(minValue,maxValue)
        elseif optimizer == "LM-ATAN" || optimizer == "GN-ATAN"
            AtanTransformation(minValue,maxValue)
        else 
            throw(DomainError(optimizer))
        end
    #println(s, " forward found ", strikes[s], " min ", minValue, " max ", maxValue, " a0 ", a0v, " atm ", priceAtm, "m+1-l-2=", (m - l - 1))
        sumw = if useVol 
            length(weights)/m
        else 
            sum(weights)/m
        end
    ### TODO don't set sigma0=sigma1: one more variable to minimize on. don't set lambda2 = lambda3.
    #TODO split in separate types Linear,... and operate on those toQuadForm. instead of strings.
    function obj!(fvec::Z, coeff::AbstractArray{W})::Z where {Z,W}
        n = length(strikes)
        σ = zeros(W, m + 1)
        @. σ[1:l] = transform(coeff)
        γ = zeros(W, m)
        β = zeros(W, m)
        α = zeros(W, m)
        θ = zeros(W, 2m)
        toQuadForm!(model, m, l, s, forward, x, α, β, γ, σ, priceAtm, isC3=isC3, basis=basis)
        computeCoeffsQuadratic(s, tte, x, α, β, γ, θ)
        if isC3
            for k = 1:nRefine
                toQuadForm!(model, m, l, s, forward, x, α, β, γ, σ, θ[2s+2], isC3=isC3, basis=basis)
                computeCoeffsQuadratic(s, tte, x, α, β, γ, θ)
            end
        end
        # println("θ ", θ)
        if useVol
            for i = 1:n
                isCall = strikes[i] >= forward
                strikeIndex = strikeIndices[i]
                mPrice = if x[strikeIndex] == strikes[i]
                    θ[2strikeIndex]
                else
                    priceEuropeanPiece(strikeIndex, QuadraticLVG(x, α, β, γ, θ, tte, forward), isCall, strikes[i])
                end
                fvec[i] = weights[i] * (impliedVolatilitySRHalley(isCall, max(mPrice, 1e-64), forward, strikes[i], tte, 1.0, 0e-14, 64, Householder()) - vols[i])
                # fvec[i] =  impliedVolatilityLiSOR(isCall, max(mPrice,1e-64), forward, strikes[i], tte, 1.0, 0.0, 0e-14, 64, SORTS()) - vols[i]
            end
        else
            for i = 1:n
                isCall = strikes[i] >= forward
                strikeIndex = strikeIndices[i]
                mPrice = if x[strikeIndex] == strikes[i]
                    θ[2strikeIndex]
                else
                    priceEuropeanPiece(strikeIndex, QuadraticLVG(x, α, β, γ, θ, tte, forward), isCall, strikes[i])
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
            # v = zeros(W, m-1)
            # for i = 1:m-1
            #     density = abs(θ[2i+2] / (α[i+1] * x[i]^2 + β[i+1] * x[i] + γ[i+1])^2)
            #     v[i] = log(density)
            # end
            for i = 2:l-1
                # fvec[i+n-1] = sumw * penalty * ((v[i+1] - v[i]) / (x[i+1] - x[i]) - (v[i] - v[i-1]) / (x[i] - x[i-1]))
                fvec[i+n-1] = sumw * penalty *(  ((σ[i+1]-σ[i])/(x[i+1]-x[i])) )
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
        σ[i] = inv(transform, a0v[i] * (1 + epsvec[i]))
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
            lower=fill(minValue, length(x0)), upper=fill(maxValue, length(x0)),
            iterations=1000
        )
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
    elseif optimizer == "GN-MQ" || optimizer == "GN-SIN" || optimizer == "GN" || optimizer == "GN-LOG" || optimizer == "GN-ALG"  ||optimizer =="GN-ATAN" || optimizer=="GN-TANH"
        fit = GaussNewton.optimize!(obj!, x0, fvec, autodiff=:single, iscale=1, abstol=length(fvec) * eps())
        # println(iter, " fit ", fit)
        x0
    elseif optimizer == "LM"
        obj1 = OnceDifferentiable(obj!, x0, copy(fvec); autodiff=:forward, inplace=true)
        fit = LsqFit.levenberg_marquardt(obj1, x0,
            lower=fill(minValue, length(x0)), upper=fill(maxValue, length(x0))
        )
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
    @. σ[1:l] = transform(x0)
    #println("σ=", σ)
    γ = zeros(T, m)
    β = zeros(T, m)
    α = zeros(T, m)
    θ = zeros(T, 2m)
    toQuadForm!(model, m, l, s, forward, x, α, β, γ, σ, priceAtm, isC3=isC3, basis=basis)
    computeCoeffsQuadratic(s, tte, x, α, β, γ, θ)
    if isC3
        for k = 1:nRefine
            toQuadForm!(model, m, l, s, forward, x, α, β, γ, σ, θ[2s+2], isC3=isC3, basis=basis)
            computeCoeffsQuadratic(s, tte, x, α, β, γ, θ)
        end
    end
    return QuadraticLVG(x, α, β, γ, θ, tte, forward)
end

#number of iterations vs LinearBlack direct? .
#how can we reuse the arrays in the autodiff optim?
