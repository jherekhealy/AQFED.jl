using LinearAlgebra
using LeastSquaresOptim
using GaussNewton, LsqFit, Optim, Random
import AQFED.Black: blackScholesFormula, impliedVolatilitySRHalley, Householder, impliedVolatilityLiSOR, SORTS, impliedVolatilityThiophene
import AQFED.Math: ClosedTransformation, inv, MQMinTransformation, IdentityTransformation, LogisticTransformation, AlgebraicTransformation, TanhTransformation, AtanTransformation
import PPInterpolation: PP, evaluatePiece, evaluateDerivativePiece, evaluateSecondDerivativePiece, LEFT_KNOT, QuadraticLagrangePP, CubicSplineNatural

struct QuadraticLVG{TV,T}
    x::Vector{T}
    a::Vector{TV} #ax^2 + bx + c
    b::Vector{TV}
    c::Vector{TV}
    θ::Vector{TV}
    λ::Vector{TV} #Bspline version
    tte::T
    forward::T
end

abstract type LVGKind end
struct LinearBachelier <: LVGKind end
struct LinearBlack <: LVGKind end
struct Quadratic <: LVGKind end

Base.broadcastable(p::QuadraticLVG) = Ref(p)


function priceEuropean(model::QuadraticLVG{TV,T}, isCall::Bool, strike, t=model.tte) where {TV,T}
    x = model.x
    if strike <= x[1] || strike >= x[end]
        return priceEuropeanPiece(0, model, isCall, strike, t)
    end
    i = max(searchsortedlast(x, strike), 1)
    return priceEuropeanPiece(i, model, isCall, strike, t)
end

#Im(z(b) - z(a)) with z(x) = log((x-x1)/(x-x2)) for complex conjugate roots x1, x2: twice the angle between a-x1 and
#b-x1, which lie in the same half plane. Exact, unlike the principal log of the ratio, which is wrong when the angle
#exceeds π/2 (possible when a B-spline coefficient is negative, e.g. the one set by the C3 condition).
imlogRatio(b, a, x1) = 2 * angle((b - x1) * conj(a - x1))



#sinh(a)/sinh(b) for a and b of the same sign, |a| <= |b|, without overflow
function sinhRatio(a, b)
    (iszero(b) || iszero(a)) && return iszero(b) ? one(a) : zero(a)
    aa, bb = abs(a), abs(b)
    return exp(aa - bb) * (-expm1(-2aa)) / (-expm1(-2bb))
end

#time value on interval i from the time values at both ends: on the hyperbolic pieces the representation
#θc cosh(u) + θs sinh(u) cancels catastrophically when the solution decays over a long interval (large u, θc ≈ -θs);
#V = χ (Vᵢ sinh(u_E - u) + V_{i+1}/χ_E sinh(u)) / sinh(u_E) does not.
function stableTimeValue(model::QuadraticLVG, i, χ, u, χE, uE)
    Vi = model.θ[2i]
    Vn = i < length(model.a) ? model.θ[2i+2] : zero(Vi)
    return χ * (Vi * sinhRatio(uE - u, uE) + Vn / χE * sinhRatio(u, uE))
end

function priceEuropeanPiece(i::Int, model::QuadraticLVG{TV,T}, isCall::Bool, strike, t=model.tte) where {TV,T}
    if i <= 0 || i > length(model.a)
        return isCall ? max(model.forward - strike, 0.0) : max(strike - model.forward, 0.0)
    end
    stable = t == model.tte
    β = model.b[i]
    α = model.a[i]
    γ = model.c[i]
    α *= sqrt(t / model.tte)
    β *= sqrt(t / model.tte)
    γ *= sqrt(t / model.tte)
    xi = model.x[i]
    value = if iszero(α) && iszero(β)
        zzi = strike - xi
        χ = one(T)
        ω = one(T) / γ
        u = ω * zzi
        # println(u," ",model.θ[2i]," ", model.θ[2i-1], " ",cosh(u))    
        # χ * (model.θ[2i] * cosh(u) + model.θ[2i-1] * sinh(u))
        stable ? stableTimeValue(model, i, χ, u, one(T), ω * (model.x[i+1] - xi)) :
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
        if stable
            zE = (model.x[i+1] - x1) / (xi - x1)
            stableTimeValue(model, i, χ, u, sqrt(abs(zE)), ω * log(abs(zE)))
        else
            χ * (model.θ[2i] + model.θ[2i-1] - model.θ[2i-1] * 2 / (1 + exp(2u))) * cosh(u)
        end
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
            if stable
                xn = model.x[i+1]
                zE = (xn - x1) / (xn - x2) * (xi - x2) / (xi - x1)
                χE = sqrt(abs((xn^2 + β / α * xn + γ / α) / (xi^2 + β / α * xi + γ / α)))
                stableTimeValue(model, i, χ, u, χE, ω * log(abs(zE)))
            else
                χ * (model.θ[2i] - model.θ[2i-1] + model.θ[2i-1] * 2 / (1 + exp(-2u))) * cosh(u)
            end
        else
            x1 = (-β - sqrt(complex(δ))) / (2α)
            x2 = (-β + sqrt(complex(δ))) / (2α)
            zzi = (strike - x1) / (strike - x2) * (xi - x2) / (xi - x1)
            imzzi = imlogRatio(strike, xi, x1)
            u = ω * imzzi
            if ω2 >= 0
                χ * (model.θ[2i] * cos(u) + model.θ[2i-1] * sin(-u))
            elseif stable
                xn = model.x[i+1]
                zE = (xn - x1) / (xn - x2) * (xi - x2) / (xi - x1)
                χE = sqrt(abs((xn^2 + β / α * xn + γ / α) / (xi^2 + β / α * xi + γ / α)))
                stableTimeValue(model, i, χ, u, χE, ω * imlogRatio(xn, xi, x1))
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
            imzzi = imlogRatio(strike, xi, x1)
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
    rhs = zeros(TA, 2m)
    # on the intervals where the solution is hyperbolic, the first continuity row is divided by cosh(u) (coefficients in
    # tanh(u) and sech(u)): cosh(u) overflows for u > 710 (small local variance on a wide interval, very short expiries)
    sechS = one(TA)

    #theta0_c = 0
    lhsd[1] = zero(TA)
    lhsdu[1] = one(TA)
    rhs[1] = zero(T)
    #a[i] between zi and zip , a[i+1]at zip
    for i = 1:m-1
        local χi, zdi, κi, ωi, si, coshi, sinhi, sechi
        if iszero(α[i]) && iszero(β[i])
            χi = one(T)
            ωi = one(T) / γ[i]
            κi = zero(T)
            zzi = strikes[i+1] - strikes[i]
            zdi = one(T)
            si = one(T)
            u = ωi * zzi
            sinhi, coshi, sechi = tanh(u), one(u), one(u) / cosh(u)
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
            sinhi, coshi, sechi = tanh(u), one(u), one(u) / cosh(u)
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
                sinhi, coshi, sechi = tanh(u), one(u), one(u) / cosh(u)
                #                χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) + ω * (model.θ[2i-1])) * ch)
            else
                x1 = (-β[i] - sqrt(complex(δ))) / (2α[i])
                x2 = (-β[i] + sqrt(complex(δ))) / (2α[i])
                zzi = (strikes[i+1] - x1) / (strikes[i+1] - x2) * (strikes[i] - x2) / (strikes[i] - x1)
                # println("dz ",(one(T) / (strike - x1) - one(T) / (strike - x2)))
                zdi = imag(one(T) / (strikes[i+1] - x1) - one(T) / (strikes[i+1] - x2))
                κi = real((one(T) / (strikes[i+1] - x1) + one(T) / (strikes[i+1] - x2))) / 2zdi
                # kappa = strike-x2 + strike - X1 / 2(strike-x2 - strike-x1)= (2strike+2beta/2alpha) / 2(-2sqrt(delta)/2alpha)
                imzzi = imlogRatio(strikes[i+1], strikes[i], x1)
                u = ωi * imzzi
                if ω2 >= 0
                    si = -one(T)
                    sinhi, coshi, sechi = sin(-u), cos(u), one(u)
                    #χ * dz * ((κ * (model.θ[2i-1]) + ω * (model.θ[2i])) * sh + (κ * (model.θ[2i]) - ω * (model.θ[2i-1])) * ch)
                else
                    si = one(T)
                    sinhi, coshi, sechi = tanh(u), one(u), one(u) / cosh(u)
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
        lhsdu[2i] = -sip * ωip * zdip * sechi                                     #2i+1
        #line 2(i-1)+2    
        lhsd[2i+1] = -sip * ωip * zdip
        # (κi coshi + ωi sinhi - (si ωi coshi + κi sinhi) coshi / sinhi) simplifies to -si ωi / sinh(u): exponentially small
        # when the solution decays (V(x_i) barely influences x_{i+1}), and no Inf/Inf
        lhsdl[2i] = -si * ωi * sechi / sinhi * χi * zdi
        lhsdu[2i+1] = zdi * (si * ωi * coshi + κi * sinhi) / (sinhi) - κip * zdip
        i == s && (sechS = sechi)

        # if !isfinite(lhsd[2i]) || !isfinite(lhsdl[2i-1]) || !isfinite(lhsdl[2i]) || !isfinite(lhsdu[2i+1]) 
        #     println(" Na N ",ωi*(zip-zi), " at ",i)
        # end
    end
    rhs[2s] = sechS
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
            imzzi = imlogRatio(strikes[m+1], strikes[m], x1)
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

function updateLambda!(λ::AbstractArray{W}, model::Union{LinearBachelier,LinearBlack}, x, sp, priceAtm; basis=x) where {W}
    λ[sp] = ((λ[sp+1] / (x[sp+1] - x[sp]) + λ[sp-1] / (x[sp] - x[sp-1]))) / (-1 / (2priceAtm) + (1 / (x[sp] - x[sp-1]) + 1 / (x[sp+1] - x[sp])))
end

function updateLambda!(λ::AbstractArray{W}, model::Quadratic, x, sp, priceAtm; basis=x) where {W}
    #λ[sp+1] = -((λ[sp+2]/(x[sp+1]-x[sp])+λ[sp]/(x[sp]-x[sp-1])))/(1/(4priceAtm)-(1/(x[sp]-x[sp-1])+1/(x[sp+1]-x[sp])))
    knots = BSplines.knots(basis)
    denom = 1 / (4priceAtm) - (1 / (knots[sp+4] - knots[sp+2]) + 1 / (knots[sp+3] - knots[sp+1]))
    if denom == zero(W) || isnan(denom) || isinf(denom)
        λ[sp+1] = λ[sp]
    else
        λ[sp+1] = -((λ[sp+2] / (knots[sp+4] - knots[sp+2]) + λ[sp] / (knots[sp+3] - knots[sp+1]))) / denom
    end
end



function makeLambda(model::LinearBachelier,σ, m, l, s, forward; isC3=false)
    if m + 1 == l + 2
        λ = vcat(σ[1], σ[1:l], σ[l])
    else
        λ = vcat(σ[1], σ[1:s], σ[s:l], σ[l])
    end
    λ .*= forward
    return λ
end
function toFreeParameters(λ, model::LinearBachelier,σ, m, l, s, forward; isC3=false)
    if m + 1 == l + 2
        σ = λ[2:end-1]
    else
        σ = λ[2:end-1]
        deleteat!(σ, s)
    end
    return σ ./ forward
end

function makeLambda(model::LinearBlack,σ, m, l, s, forward; isC3=false)
    if m + 1 == l + 2
        λ = vcat(σ[1], σ[1:l], σ[l])
    else
        λ = vcat(σ[1], σ[1:s], σ[s:l], σ[l])
    end
    return λ
end

function toFreeParameters(λ, model::LinearBlack, m, l, s, forward; isC3=false)
    if m + 1 == l + 2
        σ = λ[2:end-1]
    else
        σ = λ[2:end-1]
        deleteat!(σ, s)
    end
    return σ
end


function makeLambda(model::Quadratic,σ, m, l, s, forward; isC3=true)
    λ = if isC3
        vcat(σ[1:s], σ[s:l]) .* forward
    else
        σ[1:l] .* forward
    end
    λl = if isC3 m+3 else m+2 end
    if length(λ) < λl
        λ = vcat(λ[1], λ)
    end
    if length(λ) < λl
        λ = vcat(λ, λ[end])
    end
    if length(λ) < λl
        λ = vcat(λ[1], λ) # could be 2lambda[1]-lambda[2] or weighted by the x
    end
    if length(λ) < λl
        λ = vcat(λ, λ[end]) #could be 2lambda[end]-lambda[end-1] or weighted by the x
    end
    return λ
end

function toFreeParameters(λ, model::Quadratic, m, l, s, forward; isC3=true)
    if isC3
        #len lambda = m+3; l = len(sigma).
        diffLen = length(λ) - l - 1
        trimLen = trunc(Int, diffLen / 2)
        σ = λ[1+trimLen:end-trimLen]
        deleteat!(σ, s)
    else
        diffLen = length(λ) - l
        trimLen = trunc(Int, diffLen / 2)
        σ = λ[1+trimLen:end-trimLen]
    end
    return σ[end-l+1:end] ./ forward
end


function toQuadForm!(model::LinearBachelier, m, l, s, x, α::AbstractArray{W}, β::AbstractArray{W}, γ::AbstractArray{W}, λ::AbstractArray{W}; basis=x) where {W}
    # println("linearbach lambda ",λ," ", priceAtm)
    for i = 1:m
        β[i] = (λ[i+1] - λ[i]) / (x[i+1] - x[i])
        γ[i] = -x[i] * β[i] + λ[i]
    end
    return α, β, γ
end
function toQuadForm!(model::LinearBlack, m, l, s, x, α::AbstractArray{W}, β::AbstractArray{W}, γ::AbstractArray{W}, λ::AbstractArray{W}; basis=x) where {W}
    for i = 1:m
        bi = (λ[i+1] - λ[i]) / (x[i+1] - x[i])
        β[i] = -x[i] * bi + λ[i]
        α[i] = bi
    end
    return α, β, γ
end
function toQuadForm!(model::Quadratic, m, l, s, x, α::AbstractArray{W}, β::AbstractArray{W}, γ::AbstractArray{W}, λ::AbstractArray{W}; basis=x) where {W}
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
    if isC3
        min(xLen - 1, pricesLen)
    else
        min(xLen-1, pricesLen)
    end
end

function adjustKnots!(x, s, priceAtm; factor=2)
    sp = s + 1
    if -1 / (factor * priceAtm) + (1 / (x[sp] - x[sp-1]) + 1 / (x[sp+1] - x[sp])) <= 0
        h = min(x[sp] - x[sp-1], x[sp+1] - x[sp])
        #try out if h is ok
        # println("adjusted knots before ", x, " ", h)
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

function makeTransform(minValue::T, maxValue::T, optimizer) where {T}
    return if optimizer == "LM-MQ" || optimizer == "GN-MQ"
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
        TanhTransformation(minValue, maxValue)
    elseif optimizer == "LM-ATAN" || optimizer == "GN-ATAN"
        AtanTransformation(minValue, maxValue)
    else
        throw(DomainError(optimizer))
    end
end

function calibrateQuadraticLVG(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; U::T=6 * forward, L::T=forward / 6, useVol=false, model::LVGKind=LinearBachelier(), penalty=zero(T), nRefine=3, location="Mid-XX", isC3=true, minCount=1, size=0, guess="Constant", optimizer="LM", previousLVG=Nothing, knots=nothing, nextLVG=Nothing, flatWings=false, tieUnsupported=false, boundSupportedOnly=false, calendarWeight=10.0, minRatio=0.1, calendar=:bounds, calendarPoints=100) where {T}

    # weights normalized to a mean of one: the raw (e.g. inverse vega) weights may carry a scale such as 1/forward, which
    # makes the residuals and gradients tiny and stalls the optimizer (it then returns the initial guess)
    weights = weights ./ (sum(weights) / length(weights))
    #if the forward is not part of the strikes, add the forward and interpolate prices with quadratic. Eventually set corresponding weight to 0
    if previousLVG isa QuadraticLVG
        L = previousLVG.x[1]
        U = previousLVG.x[end]
    else
        if U <= strikes[end]
            U = strikes[end] + 3 * (strikes[end] - forward)
            #U = strikes[end]*1.1
        end
        if L >= strikes[1]
            L = strikes[1] / 2
        end
    end
    s = searchsortedfirst(strikes, forward) #index of forward
    s = max(1, min(length(strikes),s))
    origStrikes = strikes
    useForwardInStrikes = false
    local priceAtm
    if strikes[s] != forward
        qStrikes = strikes[s-1:s+1]
        qPrices = callPrices[s-1:s+1]
        qvols = @. impliedVolatilityThiophene(true, qPrices, forward, qStrikes, tte, 1.0)
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
    local x
    if previousLVG isa QuadraticLVG
        # the knots of the previous expiry, or a refinement of them (`knots` must contain them): the previous local
        # variance is then a spline on the new knots too, and its coefficients bound the new ones from below.
        x = knots === nothing ? previousLVG.x : collect(knots)
        knots === nothing || all(xi -> any(isapprox.(x, xi; atol=1e-12)), previousLVG.x) ||
            throw(ArgumentError("knots must contain the knots of previousLVG"))
        s = searchsortedfirst(x, forward) - 1
    else
        x = knots === nothing ? makeKnots(tte, forward, strikes, size, location=location, minCount=minCount, L=L, U=U) : collect(knots)
        if flatWings && knots === nothing && length(x) > 3
            # one knot beyond the outer strikes on each side (one mean log-spacing out): otherwise the last coefficient
            # covers both the outer quotes and the whole wing up to L or U, and there is nothing beyond the quotes to tie
            δ = (log(x[end-1]) - log(x[2])) / (length(x) - 3)
            xl, xu = min(x[2], strikes[1]) * exp(-δ), max(x[end-1], strikes[end]) * exp(δ)
            x = sort(vcat(x, filter(c -> L < c < U, [xl, xu])))
        end
        s = searchsortedfirst(x, forward)
        if !useForwardInStrikes && x[s] != forward
            x = sort(vcat(x, forward))
        end
        s = searchsortedfirst(x, forward) - 1
        # given knots are kept as they are (they may be shared with another expiry)
        knots === nothing && adjustKnots!(x, s, priceAtm)
    end
    xBasis = makeBasis(model, x, forward, isC3)
    m = length(x) - 1
    #initial guess
    vols = zeros(T, length(strikes))
    if useVol || guess != "Constant"
        for i = eachindex(vols)
            vols[i] = impliedVolatilityThiophene(true, callPrices[i], forward, strikes[i], tte, 1.0)
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
    # with given knots (shared with other expiries) every coefficient is free, even with fewer quotes than knots:
    # the damping of Levenberg-Marquardt handles the underdetermined directions
    l = knots === nothing ? numberOfFreeParameters(model, length(x), length(callPrices), isC3) : length(x) - 1

    atmVol = impliedVolatilityThiophene(true, priceAtm, forward, forward, tte, 1.0)
    a0 = atmVol * sqrt(tte / 2)
    local minValueV
    local maxValueV
    λPrev = nothing
    λNext = nothing
    #calendar=:bounds: the sufficient condition of Falck on the coefficients (the local variance times the expiry does not
    #decrease); calendar=:spreads: the exact condition at the expiries, the total implied variance does not decrease at
    #any strike, as one-sided residuals at `calendarPoints` strikes log-uniform over the knots
    boundPrev = calendar == :bounds ? previousLVG : Nothing
    boundNext = calendar == :bounds ? nextLVG : Nothing
    if boundPrev isa QuadraticLVG
        λPrev = x == previousLVG.x ? previousLVG.λ : refineLambda(previousLVG, x, model, forward, isC3)
        aMinV = toFreeParameters(λPrev, model, m, l, s, forward, isC3=isC3)
        minValueV = max.(aMinV, a0 * minRatio / 10)
        maxValueV = fill(a0 * 800, l)
    else
        minValueV = fill(a0 * minRatio, l)
        maxValueV = fill(a0 * 800, l)
    end
    if boundNext isa QuadraticLVG
        # calibration from the last expiry to the first: the next (longer) expiry bounds the coefficients from above.
        # `knots` must contain the knots of nextLVG.
        λNext = x == nextLVG.x ? nextLVG.λ : refineLambda(nextLVG, x, model, forward, isC3)
        aMaxV = toFreeParameters(λNext, model, m, l, s, forward, isC3=isC3)
        maxValueV = min.(maxValueV, aMaxV)
        minValueV = min.(minValueV, maxValueV .* 0.9)
    end
    # flat wings: the coefficients whose B-splines lie beyond the quoted strikes are not fitted but tied to the nearest
    # quoted coefficient on the same side (flat local variance beyond the quotes), within their bounds.
    tie = collect(1:l)
    if flatWings
        marker = makeLambda(model, T.(1:l), m, l, s, one(T), isC3=isC3)
        tk = BSplines.knots(basis)
        ξ = [(tk[k+1] + tk[k+2]) / 2 for k = 1:length(basis)]
        inside = falses(l)
        for k = 1:min(length(ξ), length(marker))
            i = round(Int, marker[k])
            (1 <= i <= l && strikes[1] <= ξ[k] <= strikes[end]) && (inside[i] = true)
        end
        if any(inside)
            firstIn, lastIn = findfirst(inside), findlast(inside)
            for i = 1:l
                i < firstIn && (tie[i] = firstIn)
                i > lastIn && (tie[i] = lastIn)
            end
        end
    end
    # too many knots for the quotes: a coefficient whose B-spline support contains no quoted strike is not fitted but
    # tied to the adjacent supported coefficient (the nearest one, the left one on a tie), within its bounds
    if tieUnsupported
        marker = makeLambda(model, T.(1:l), m, l, s, one(T), isC3=isC3)
        tk = BSplines.knots(basis)
        supported = falses(l)
        for k = 1:min(length(basis), length(marker))
            i = round(Int, marker[k])
            (1 <= i <= l && any(tk[k] < K < tk[k+3] for K in strikes)) && (supported[i] = true)
        end
        idx = findall(supported)
        if !isempty(idx)
            for i = 1:l
                supported[i] && continue
                tie[i] = idx[argmin(abs.(idx .- i) .+ 0.1 .* (idx .> i))]
                if boundSupportedOnly
                    # no quote under this coefficient: the calendar bounds of the neighbouring expiry, a sufficient
                    # condition only, are not imposed here (check the calendar spreads of the surface afterwards)
                    minValueV[i] = a0 * minRatio
                    maxValueV[i] = a0 * 800
                end
            end
        end
    end
    tieSigma!(σv) = (for i = 1:l
        tie[i] != i && (σv[i] = min(max(σv[tie[i]], minValueV[i]), maxValueV[i]))
    end; σv)

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
            @.(sqrt(min((a0 * 10)^2, max((a0 / 5)^2, (xPrices - xIntrinsic) / (forward^2 * (xPricesUp - 2xPrices + xPricesDown) / epsDens^2)))))[1:end]
        end
    # the initial guess must lie within the bounds (the previous expiry may bound it from below)
    a0v = min.(max.(a0v, minValueV .* 1.01), maxValueV .* 0.99)
    # bounds too close for the margins above: start in the middle
    a0v = [lo < v < hi ? v : (lo + hi) / 2 for (v, lo, hi) in zip(a0v, minValueV, maxValueV)]
    transformV = [makeTransform(minValue, maxValue, optimizer) for (minValue, maxValue) = zip(minValueV, maxValueV)]
    #calendar=:spreads: bounds on the total implied standard deviation at the check strikes, from the neighbouring expiries
    #(the out-of-the-money prices below 1e-12 of the forward are skipped, their implied volatility being meaningless)
    calK = T[]; calLo = T[]; calHi = T[]
    if calendar == :spreads && (previousLVG isa QuadraticLVG || nextLVG isa QuadraticLVG)
        stdOf(sec, K) = (isCall = K >= forward; p = priceEuropean(sec, isCall, K); p > 1e-12 * forward ? impliedVolatilityThiophene(isCall, p, forward, K, sec.tte, 1.0) * sqrt(sec.tte) : T(NaN))
        for K in exp.(range(log(x[1]), log(x[end]), length=calendarPoints + 2)[2:end-1])
            lo = previousLVG isa QuadraticLVG ? stdOf(previousLVG, K) : T(NaN)
            hi = nextLVG isa QuadraticLVG ? stdOf(nextLVG, K) : T(NaN)
            (isnan(lo) && isnan(hi)) && continue
            push!(calK, K); push!(calLo, isnan(lo) ? zero(T) : lo); push!(calHi, isnan(hi) ? T(Inf) : hi)
        end
    end
    calOffset = length(callPrices) + (penalty > 0 ? m - 1 : 0)


    # println(s, " forward found ", strikes[s], " min ", minValueV, " max ", maxValueV, " a0 ", a0v, " atm ", priceAtm, "m+1-l-2=", (m - l - 1))
    sumw = if useVol
        length(weights) / m
    else
        sum(weights) / m
    end
    ### TODO don't set sigma0=sigma1: one more variable to minimize on. don't set lambda2 = lambda3.
    #TODO split in separate types Linear,... and operate on those toQuadForm. instead of strings.
    function obj!(fvec::Z, coeff::AbstractArray{W})::Z where {Z,W}
        n = length(strikes)
        σ = zeros(W, m + 1)
        for i = 1:l
            σ[i] = transformV[i](coeff[i])
        end
        tieSigma!(σ)
        γ = zeros(W, m)
        β = zeros(W, m)
        α = zeros(W, m)
        θ = zeros(W, 2m)
        λ = makeLambda(model,σ, m, l, s, forward, isC3=isC3)
        if isC3
            updateLambda!(λ, model, x, s + 1, priceAtm, basis=basis)
        end
        toQuadForm!(model, m, l, s, x, α, β, γ, λ, basis=basis)
        computeCoeffsQuadratic(s, tte, x, α, β, γ, θ)
        if isC3
            for _ = 1:nRefine
                updateLambda!(λ, model, x, s + 1, θ[2s+2], basis=basis)
                toQuadForm!(model, m, l, s, x, α, β, γ, λ, basis=basis)
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
                    priceEuropeanPiece(strikeIndex, QuadraticLVG(x, α, β, γ, θ, λ, tte, forward), isCall, strikes[i])
                end
                fvec[i] = weights[i] * (impliedVolatilityThiophene(isCall, max(mPrice, 1e-64), forward, strikes[i], tte, 1.0) - vols[i])
                # fvec[i] =  impliedVolatilityLiSOR(isCall, max(mPrice,1e-64), forward, strikes[i], tte, 1.0, 0.0, 0e-14, 64, SORTS()) - vols[i]
            end
        else
            for i = 1:n
                isCall = strikes[i] >= forward
                strikeIndex = strikeIndices[i]
                mPrice = if x[strikeIndex] == strikes[i]
                    θ[2strikeIndex]
                else
                    priceEuropeanPiece(strikeIndex, QuadraticLVG(x, α, β, γ, θ, λ, tte, forward), isCall, strikes[i])
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
        if isC3 && (λPrev !== nothing || λNext !== nothing)
            # the coefficient at the forward is not free (set by the C3 condition), so the calendar bounds of the other
            # coefficients do not apply to it: a stiff one-sided residual keeps it within the bounds of the neighbouring expiries
            v = zero(W)
            λPrev === nothing || (v += max(λPrev[s+2] - λ[s+2], zero(W)))
            λNext === nothing || (v += max(λ[s+2] - λNext[s+2], zero(W)))
            fvec[end] = sumw * calendarWeight * v / forward
        end
        if !isempty(calK)
            sec = QuadraticLVG(x, α, β, γ, θ, λ, tte, forward)
            for (r, K) in enumerate(calK)
                isCall = K >= forward
                p = priceEuropeanPiece(max(searchsortedlast(x, K), 1), sec, isCall, K)
                sd = impliedVolatilityThiophene(isCall, max(p, 1e-64), forward, K, tte, 1.0) * sqrt(tte)
                #in implied volatility, the weight shared by the check strikes
                fvec[calOffset+r] = calendarWeight * (max(calLo[r] - sd, zero(W)) + max(sd - calHi[r], zero(W))) / sqrt(tte * length(calK))
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
                fvec[i+n-1] = sumw * penalty * (((σ[i+1] - σ[i]) / (x[i+1] - x[i])))
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
        σ[i] = inv(transformV[i], a0v[i] * (1 + epsvec[i]))
    end
    x0 = σ[1:l]
    outlen = length(callPrices)
    if penalty > 0
        outlen += m - 1
    end
    outlen += length(calK)
    isC3 && (λPrev !== nothing || λNext !== nothing) && (outlen += 1)
    fvec = zeros(Float64, outlen)

    function runOptimizer(x0)
        if optimizer == "LM-Curve"
            fit = LeastSquaresOptim.optimize!(
                LeastSquaresProblem(x=x0, (f!)=obj!, autodiff=:forward, #:forward is 4x faster than :central
                    output_length=outlen),
                LevenbergMarquardt();
                lower=minValueV, upper=maxValueV,
                iterations=1000        )
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
        elseif optimizer == "GN-MQ" || optimizer == "GN-SIN" || optimizer == "GN" || optimizer == "GN-LOG" || optimizer == "GN-ALG" || optimizer == "GN-ATAN" || optimizer == "GN-TANH"
            fit = GaussNewton.optimize!(obj!, x0, fvec, autodiff=:single, iscale=1, abstol=length(fvec) * eps())
            #  println(iter, " fit ", fit)
            x0
        elseif optimizer == "LM"
            obj1 = OnceDifferentiable(obj!, x0, copy(fvec); autodiff=LsqFit.AutoForwardDiff(), inplace=true)
            fit = LsqFit.levenberg_marquardt(obj1, x0,
                lower=minValueV, upper=maxValueV)
            
            # println(iter, " fit ", fit)
            fit.minimizer
        elseif optimizer == "LM-MQ" || optimizer == "LM-SIN" || optimizer == "LM-LOG" || optimizer == "LM-ALG" || optimizer == "LM-TANH" || optimizer == "LM-ATAN"
            obj1 = OnceDifferentiable(obj!, x0, copy(fvec); autodiff=LsqFit.AutoForwardDiff(), inplace=true)
            fit = LsqFit.levenberg_marquardt(obj1, x0)
            # println(iter, " fit ", fit)
            fit.minimizer

        end
        #  
    end
    x0 = runOptimizer(x0)
    obj!(fvec, x0)
    for i = 1:l
        σ[i] = transformV[i](x0[i])
    end
    tieSigma!(σ)
    #println("σ=", σ)
    γ = zeros(T, m)
    β = zeros(T, m)
    α = zeros(T, m)
    θ = zeros(T, 2m)
    λ = makeLambda(model,σ, m, l, s, forward, isC3=isC3)
    if isC3
        updateLambda!(λ, model, x, s + 1, priceAtm, basis=basis)
    end
    toQuadForm!(model, m, l, s, x, α, β, γ, λ, basis=basis)
    computeCoeffsQuadratic(s, tte, x, α, β, γ, θ)
    if isC3
        for _ = 1:nRefine
            updateLambda!(λ, model, x, s + 1, θ[2s+2], basis=basis)
            toQuadForm!(model, m, l, s, x, α, β, γ, λ, basis=basis)
            computeCoeffsQuadratic(s, tte, x, α, β, γ, θ)
        end
        # the residual on the coefficient at the forward is soft: if it still breaks the calendar bounds of the
        # neighbouring expiries, clamp it (the density is then no longer C1 at the forward, but the calendar spreads are
        # non-negative, the sufficient condition holding for every coefficient)
        if λPrev !== nothing || λNext !== nothing
            c = s + 2
            λc = λ[c]
            λPrev === nothing || (λc = max(λc, λPrev[c]))
            λNext === nothing || (λc = min(λc, λNext[c]))
            if λc != λ[c]
                λ[c] = λc
                toQuadForm!(model, m, l, s, x, α, β, γ, λ, basis=basis)
                computeCoeffsQuadratic(s, tte, x, α, β, γ, θ)
            end
        end
    end
    return QuadraticLVG(x, α, β, γ, θ, λ, tte, forward)
end

#number of iterations vs LinearBlack direct? .
#how can we reuse the arrays in the autodiff optim?




"""
    refineLambda(previous, x, model, forward, isC3)

B-spline coefficients, on the basis of the knots `x`, of the local variance spline of `previous`, whose knots are a
subset of `x` (knot insertion): the quadratic spline on the coarser knots is also a spline on the finer ones, and its
coefficients are recovered exactly by interpolation at the Greville abscissae of the finer basis.
"""
function refineLambda(previous::QuadraticLVG, x, model::LVGKind, forward, isC3)
    spPrev = BSplines.Spline(BSplineBasis(3, makeBasis(model, previous.x, forward, isC3)), previous.λ)
    basis = BSplineBasis(3, makeBasis(model, x, forward, isC3))
    t = BSplines.knots(basis)
    n = length(basis)
    ξ = [(t[i+1] + t[i+2]) / 2 for i = 1:n]
    A = [basis[j](ξi) for ξi in ξ, j = 1:n]
    return A \ spPrev.(ξ)
end

struct LVGSurface{T}
    expiries::Vector{T}
    forwards::Vector{T}
    sections::Vector{QuadraticLVG}
end
Base.broadcastable(p::LVGSurface) = Ref(p)


function calibrateLVGSurface(ttes::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractMatrix{T}, callPrices::AbstractMatrix{T}, weights::AbstractMatrix{T};useVol=false, model=PDDE.Quadratic(),location="Mid-XX",optimizer="LM-ALG") where {T}
    #starts with first slice
    slices = Vector{QuadraticLVG}(undef, length(ttes))
    forward = forwards[1]
    mstrikes = strikes[1, :] ./ forward
    mprices = callPrices[1, :] ./ forward
    slices[1] = calibrateQuadraticLVG(ttes[1], one(T), mstrikes, mprices, weights[1, :],useVol=useVol,model=model,location=location,optimizer=optimizer)
    #println("calibrated first slice ",slices[1])
    for i = 2:length(ttes)
        nextPrices = callPrices[i, :] ./ forwards[i]
        nextStrikes = strikes[i, :] ./ forwards[i]
        #we need to use existing knots, and ensure the Bslp-bsplPrev > 0
        slices[i] = calibrateQuadraticLVG(ttes[i], one(T), nextStrikes, nextPrices, weights[i, :], previousLVG=slices[i-1],useVol=useVol,model=model,location=location,optimizer=optimizer)
        #println("calibrated slice ",i," ",slices[i])
    end
    #returns a surface in prices.
    return LVGSurface(ttes, forwards, slices)
end

"""
    calibrateLVGSurface(ttes, forwards, strikes::Vector{Vector}, callPrices::Vector{Vector}, weights::Vector{Vector};
        size=10, location="Equidistributed", L=nothing, U=nothing, ...)

Calibrate the quadratic LVG slices from the shortest expiry to the longest, with a local variance that does not decrease
with the expiry (the B-spline coefficients of each expiry bounded below by those of the previous one, on shared knots).
Each expiry has its own strikes. The knots are nested: the first expiry uses `size` knots placed by `location` between
the global bounds `L` and `U`; each later expiry keeps all the knots of the previous one and adds its own knots beyond
the outermost interior knots so far, so that wider long-dated quotes get knots in their wings.
The lower bound of the coefficients of an expiry not bounded by another one is `minRatio` times its at-the-money
value (`minRatio/10` times when bounded below by the previous expiry). `flatWings=true` ties the coefficients beyond
the quotes of each expiry to the last quoted one on the same side; `L`, `U` default to half the lowest and twice the
highest strike of the surface, which may be too close for long expiries.
`calendar=:spreads` replaces the bounds between the coefficients of consecutive expiries (sufficient only) by the exact
condition at the expiries: the total implied variance of an expiry does not fall below that of the previous one (or rise
above that of the next one, order=:backward) at `calendarPoints` strikes log-uniform over the knots, as one-sided
residuals in implied volatility of total weight `calendarWeight` (each `calendarWeight/√calendarPoints`; a stiffer
penalty, from 100 per strike, stalls the optimizer far from the quotes).
"""
function calibrateLVGSurface(ttes::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractVector{<:AbstractVector{T}},
    callPrices::AbstractVector{<:AbstractVector{T}}, weights::AbstractVector{<:AbstractVector{T}}; size=10, location="Equidistributed",
    L=nothing, U=nothing, useVol=false, model=PDDE.Quadratic(), optimizer="LM-ALG", sharedKnots=nothing, order=:forward,
    flatWings=false, tieUnsupported=false, boundSupportedOnly=false, pivot=nothing, minRatio=0.1, calendar=:bounds, calendarPoints=100, calendarWeight=10.0) where {T}
    ks = [strikes[i] ./ forwards[i] for i in eachindex(ttes)]
    if sharedKnots !== nothing
        # one knot set for all expiries, in forward-normalised strikes (it must contain the forward 1)
        slices = Vector{QuadraticLVG}(undef, length(ttes))
        n = length(ttes)
        for i in (order == :backward ? (n:-1:1) : (1:n))
            prev = order == :backward || i == 1 ? Nothing : slices[i-1]
            next = order == :backward && i < n ? slices[i+1] : Nothing
            slices[i] = calibrateQuadraticLVG(ttes[i], one(T), ks[i], callPrices[i] ./ forwards[i], weights[i]; knots=sharedKnots,
                previousLVG=prev, nextLVG=next, useVol=useVol, model=model, optimizer=optimizer, minRatio=minRatio, calendar=calendar, calendarPoints=calendarPoints, calendarWeight=calendarWeight)
        end
        return LVGSurface(collect(ttes), collect(forwards), slices)
    end
    L = L === nothing ? minimum(first.(ks)) / 2 : L
    U = U === nothing ? 2maximum(last.(ks)) : U
    slices = Vector{QuadraticLVG}(undef, length(ttes))
    local x
    if order == :middle
        # from the expiry `pivot` (by default the closest to one year), downwards to the first expiry (each shorter one
        # bounded above by the next) and upwards to the last (each longer one bounded below by the previous); the knots are
        # nested in each direction
        n = length(ttes)
        p = pivot === nothing ? argmin(abs.(ttes .- 1)) : pivot
        slices[p] = calibrateQuadraticLVG(ttes[p], one(T), ks[p], callPrices[p] ./ forwards[p], weights[p]; size=size, location=location, L=L, U=U,
            useVol=useVol, model=model, optimizer=optimizer, minRatio=minRatio, calendar=calendar, calendarPoints=calendarPoints, calendarWeight=calendarWeight)
        xp = slices[p].x
        x = xp
        for i in p-1:-1:1
            cand = filter(c -> L < c < U, makeKnots(ttes[i], one(T), ks[i], size, location=location, L=L, U=U))
            gap = length(cand) < 2 ? zero(T) : minimum(diff(sort(cand))) / 2
            x = sort(vcat(x, filter(c -> minimum(abs.(x .- c)) > gap, cand)))
            slices[i] = calibrateQuadraticLVG(ttes[i], one(T), ks[i], callPrices[i] ./ forwards[i], weights[i]; nextLVG=slices[i+1], knots=x, L=L, U=U,
                useVol=useVol, model=model, optimizer=optimizer, minRatio=minRatio, calendar=calendar, calendarPoints=calendarPoints, calendarWeight=calendarWeight)
        end
        x = xp
        for i in p+1:n
            cand = filter(c -> L < c < U, makeKnots(ttes[i], one(T), ks[i], size, location=location, L=L, U=U))
            inner = x[2:end-1]
            gap = length(cand) < 2 ? zero(T) : minimum(diff(sort(cand))) / 2
            x = sort(vcat(x, filter(c -> c < inner[1] - gap || c > inner[end] + gap, cand)))
            slices[i] = calibrateQuadraticLVG(ttes[i], one(T), ks[i], callPrices[i] ./ forwards[i], weights[i]; previousLVG=slices[i-1], knots=x,
                useVol=useVol, model=model, optimizer=optimizer, minRatio=minRatio, calendar=calendar, calendarPoints=calendarPoints, calendarWeight=calendarWeight)
        end
        return LVGSurface(collect(ttes), collect(forwards), slices)
    end
    if order == :backward
        # from the last expiry to the first: each shorter expiry keeps the knots of the longer one and adds its own where
        # they are not already covered; the longer expiry bounds its local variance from above
        n = length(ttes)
        for i in n:-1:1
            prices = callPrices[i] ./ forwards[i]
            if i == n
                slices[n] = calibrateQuadraticLVG(ttes[n], one(T), ks[n], prices, weights[n]; size=size, location=location, L=L, U=U,
                    useVol=useVol, model=model, optimizer=optimizer, flatWings=flatWings, tieUnsupported=tieUnsupported, boundSupportedOnly=boundSupportedOnly, minRatio=minRatio, calendar=calendar, calendarPoints=calendarPoints, calendarWeight=calendarWeight)
                x = slices[n].x
            else
                cand = filter(c -> L < c < U, makeKnots(ttes[i], one(T), ks[i], size, location=location, L=L, U=U))
                gap = length(cand) < 2 ? zero(T) : minimum(diff(sort(cand))) / 2
                added = filter(c -> minimum(abs.(x .- c)) > gap, cand)
                x = sort(vcat(x, added))
                slices[i] = calibrateQuadraticLVG(ttes[i], one(T), ks[i], prices, weights[i]; nextLVG=slices[i+1], knots=x, L=L, U=U,
                    useVol=useVol, model=model, optimizer=optimizer, flatWings=flatWings, tieUnsupported=tieUnsupported, boundSupportedOnly=boundSupportedOnly, minRatio=minRatio, calendar=calendar, calendarPoints=calendarPoints, calendarWeight=calendarWeight)
            end
        end
        return LVGSurface(collect(ttes), collect(forwards), slices)
    end
    for i in eachindex(ttes)
        prices = callPrices[i] ./ forwards[i]
        if i == 1
            slices[1] = calibrateQuadraticLVG(ttes[1], one(T), ks[1], prices, weights[1]; size=size, location=location, L=L, U=U,
                useVol=useVol, model=model, optimizer=optimizer, flatWings=flatWings, tieUnsupported=tieUnsupported, boundSupportedOnly=boundSupportedOnly, minRatio=minRatio, calendar=calendar, calendarPoints=calendarPoints, calendarWeight=calendarWeight)
            x = slices[1].x
        else
            cand = makeKnots(ttes[i], one(T), ks[i], size, location=location, L=L, U=U)
            cand = filter(c -> L < c < U, cand)
            inner = x[2:end-1]
            gap = length(cand) < 2 ? zero(T) : minimum(diff(sort(cand))) / 2
            added = filter(c -> c < inner[1] - gap || c > inner[end] + gap, cand)
            x = sort(vcat(x, added))
            slices[i] = calibrateQuadraticLVG(ttes[i], one(T), ks[i], prices, weights[i]; previousLVG=slices[i-1], knots=x,
                useVol=useVol, model=model, optimizer=optimizer, flatWings=flatWings, tieUnsupported=tieUnsupported, boundSupportedOnly=boundSupportedOnly, minRatio=minRatio, calendar=calendar, calendarPoints=calendarPoints, calendarWeight=calendarWeight)
        end
    end
    return LVGSurface(collect(ttes), collect(forwards), slices)
end

function price(s::LVGSurface, y, t, indexTime::Int=0)
    if t <= s.expiries[1]
        #TODO specific impl with updated spl coeffs.
        return priceEuropean(s.sections[1], true, exp(y), t) * s.forwards[1]
    elseif t >= s.expiries[end]
        return priceEuropean(s.sections[end], true, exp(y), t) * s.forwards[end]
    else
        if indexTime == 0
            indexTime = searchsortedlast(s.expiries, t)
        end
        t0 = s.expiries[indexTime]
        var0 = priceEuropean(s.sections[indexTime], true, exp(y)) * s.forwards[indexTime]
        t1 = s.expiries[indexTime+1]
        var1 = priceEuropean(s.sections[indexTime+1], true, exp(y)) * s.forwards[indexTime+1]
        #linear interpolation in total variance along same logmoneyness
        v = (var1 * (t - t0) + var0 * (t1 - t)) / (t1 - t0)
        return v
    end
end

function varianceByLogmoneyness(section::QuadraticLVG, y, forward, tte)
    strike = exp(y)
    f1 = one(typeof(strike))
    isCall = strike >= f1
    price = priceEuropean(section, isCall, strike, tte)
    if price <= 0
        price = 1e-16
    end
    vol = impliedVolatilityThiophene(isCall, price, f1, strike, tte, 1.0)
    return vol^2
end

function varianceByLogmoneyness(s::LVGSurface, y, t, indexTime::Int=0)
    if t <= s.expiries[1]
        return varianceByLogmoneyness(s.sections[1], y, s.forwards[1], s.expiries[1])
    elseif t >= s.expiries[end]
        return varianceByLogmoneyness(s.sections[end], y, s.forwards[end], s.expiries[end])
    else
        if indexTime == 0
            indexTime = searchsortedlast(s.expiries, t)
        end
        t0 = s.expiries[indexTime]
        var0 = varianceByLogmoneyness(s.sections[indexTime], y, s.forwards[indexTime], s.expiries[indexTime])
        t1 = s.expiries[indexTime+1]
        var1 = varianceByLogmoneyness(s.sections[indexTime+1], y, s.forwards[indexTime+1], s.expiries[indexTime+1])
        #linear interpolation in total variance along same logmoneyness
        v = (var1 * t1 * (t - t0) + var0 * t0 * (t1 - t)) / (t1 - t0)
        return v / t
    end
end
"""
    calibrateLVGSurfaceGlobal(surface::LVGSurface, strikes::Vector{Vector}, callPrices::Vector{Vector},
        weights::Vector{Vector}; useVol=false, model=Quadratic(), isC3=true, nRefine=3, optimizer="LM-ALG", calendarWeight=10.0)

Global pass on a quadratic LVG surface bootstrapped on shared knots (`calibrateLVGSurface` with the strike matrix): all
the expiries are refitted together, starting from the bootstrap. The free coefficients are those of the first expiry and
the increments μᵢ = σᵢ - σᵢ₋₁ ≥ 0 of the later ones, so that the sufficient condition λᵢ ≥ λᵢ₋₁ against calendar spread
arbitrage is a fixed bound; the coefficient at the forward (set by the C3 condition) is kept above the previous one by a
one-sided residual (`calendarWeight`). Strikes and prices as given to the bootstrap (divided by the forwards here).
"""
function calibrateLVGSurfaceGlobal(surface::LVGSurface, strikes::AbstractVector{<:AbstractVector{T}}, callPrices::AbstractVector{<:AbstractVector{T}},
    weights::AbstractVector{<:AbstractVector{T}}; useVol=false, model=Quadratic(), isC3=true, nRefine=3, optimizer="LM-ALG", calendarWeight=10.0) where {T}
    secs = surface.sections
    n = length(secs)
    x = secs[1].x
    all(sec -> sec.x == x, secs) || throw(ArgumentError("the sections must share their knots"))
    ks = [strikes[i] ./ surface.forwards[i] for i = 1:n]
    cs = [callPrices[i] ./ surface.forwards[i] for i = 1:n]
    ws = [weights[i] ./ (sum(weights[i]) / length(weights[i])) for i = 1:n]
    m = length(x) - 1
    s = searchsortedfirst(x, one(T)) - 1
    basis = BSplineBasis(3, makeBasis(model, x, one(T), isC3))
    ls = [numberOfFreeParameters(model, length(x), length(cs[i]), isC3) for i = 1:n]
    all(==(ls[1]), ls) || throw(ArgumentError("the expiries must have the same number of free parameters"))
    l = ls[1]
    σs = [toFreeParameters(sec.λ, model, m, l, s, one(T), isC3=isC3) for sec in secs]
    priceAtm = [priceEuropean(sec, true, one(T)) for sec in secs]
    vols = [useVol ? [impliedVolatilityThiophene(true, cs[i][j], one(T), ks[i][j], secs[i].tte, one(T)) for j in eachindex(ks[i])] : T[] for i = 1:n]
    strikeIndices = [max.(searchsortedlast.(Ref(x), ks[i]), 1) for i = 1:n]
    a0 = maximum(σs[1])
    tr1 = makeTransform(minimum(σs[1]) / 10, a0 * 800, optimizer)
    trμ = makeTransform(zero(T), a0 * 800, optimizer)
    p0 = vcat([inv(tr1, v) for v in σs[1]], [inv(trμ, max(σs[i][k] - σs[i-1][k], a0 * 1e-6)) for i = 2:n for k = 1:l])
    function section(σ, i, W)
        α = zeros(W, m); β = zeros(W, m); γ = zeros(W, m); θ = zeros(W, 2m)
        λ = makeLambda(model, σ, m, l, s, one(T), isC3=isC3)
        isC3 && updateLambda!(λ, model, x, s + 1, priceAtm[i], basis=basis)
        toQuadForm!(model, m, l, s, x, α, β, γ, λ, basis=basis)
        computeCoeffsQuadratic(s, secs[i].tte, x, α, β, γ, θ)
        if isC3
            for _ = 1:nRefine
                updateLambda!(λ, model, x, s + 1, θ[2s+2], basis=basis)
                toQuadForm!(model, m, l, s, x, α, β, γ, λ, basis=basis)
                computeCoeffsQuadratic(s, secs[i].tte, x, α, β, γ, θ)
            end
        end
        return QuadraticLVG(x, α, β, γ, θ, λ, secs[i].tte, one(T))
    end
    sigmas(p) = (σ = [[tr1(p[k]) for k = 1:l]]; for i = 2:n
        push!(σ, σ[i-1] .+ [trμ(p[l*(i-1)+k]) for k = 1:l])
    end; σ)
    nres = sum(length.(cs)) + (isC3 ? n - 1 : 0)
    function obj!(fvec, p)
        W = eltype(p)
        σ = sigmas(p)
        r = 0
        λPrevF = zero(W)
        for i = 1:n
            sec = section(σ[i], i, W)
            for (j, k) in enumerate(ks[i])
                isCall = k >= one(T)
                si = strikeIndices[i][j]
                mPrice = priceEuropeanPiece(si, sec, isCall, k)
                k < one(T) && (mPrice += one(T) - k)
                r += 1
                fvec[r] = useVol ? ws[i][j] * (impliedVolatilityThiophene(true, max(mPrice, 1e-64), one(T), k, sec.tte, one(T)) - vols[i][j]) :
                          ws[i][j] * (mPrice - cs[i][j])
            end
            if isC3 && i > 1
                r += 1
                fvec[r] = calendarWeight * max(λPrevF - sec.λ[s+2], zero(W))
            end
            λPrevF = sec.λ[s+2]
        end
        fvec
    end
    fvec = zeros(T, nres)
    obj1 = OnceDifferentiable(obj!, p0, copy(fvec); autodiff=LsqFit.AutoForwardDiff(), inplace=true)
    p = LsqFit.levenberg_marquardt(obj1, p0).minimizer
    σ = sigmas(p)
    return LVGSurface(surface.expiries, surface.forwards, Vector{QuadraticLVG}([section(σ[i], i, T) for i = 1:n]))
end
