using COSMO
using SparseArrays, LinearAlgebra
using GoldfarbIdnaniSolver
import AQFED.TermStructure:varianceByLogmoneyness
import AQFED.Black:impliedVolatility

"C2 Cubic Spline representation in terms of second derivative values"
struct C2Spline{VX,VY}
    x::VX
    y::VY
    y2::VY
end

Base.broadcastable(p::C2Spline) = Ref(p)

(spl::C2Spline)(x) = evaluate(spl, x)

function evaluate(self::C2Spline{VX,VY}, z::TZ) where {VX,VY,TZ}
    if z <= self.x[1]
        h = (self.x[2] - self.x[1])
        gPrime = (self.y[2] - self.y[1]) / h - self.y2[2] * h / 6
        return self.y[1] - (self.x[1] - z) * gPrime
    elseif z >= self.x[end]
        h = (self.x[end] - self.x[end-1])
        gPrime = (self.y[end] - self.y[end-1]) / h - self.y2[end-1] * h / 6
        return self.y[end] + (self.x[end] - z) * gPrime
    end
    i = searchsortedfirst(self.x, z)  # x[i-1]<z<=x[i]
    if z != self.x[i] && i > 1
        i -= 1
    end
    return evaluatePiece(self, i, z)
end


@inline function evaluatePiece(self::C2Spline{VX,VY}, i::Int, z::TZ) where {TZ,VX,VY}
    h = self.x[i+1] - self.x[i]
    return ((z - self.x[i]) * self.y[i+1] + (self.x[i+1] - z) * self.y[i]) / h + (z - self.x[i]) * (z - self.x[i+1]) / 6 * (((z - self.x[i]) / h + 1) * self.y2[i+1] + ((self.x[i+1] - z) / h + 1) * self.y2[i])
end

function evaluateSecondDerivative(self::C2Spline{VX,VY}, z::TZ) where {VX,VY,TZ}
    if z <= self.x[1]
        return zero(TZ)
    elseif z >= self.x[end]
        return zero(TZ)
    end
    i = searchsortedfirst(self.x, z)  # x[i-1]<z<=x[i]
    if z != self.x[i] && i > 1
        i -= 1
    end
    return evaluateSecondDerivativePiece(self, i, z)
end


@inline function evaluateSecondDerivativePiece(self::C2Spline{VX,VY}, i::Int, z::TZ) where {TZ,VX,VY}
    h = self.x[i+1] - self.x[i]
    return ((z - self.x[i]) / h) * self.y2[i+1] + ((self.x[i+1] - z) / h) * self.y2[i]
end

#prior = cubic spline on vols, or prior = thin plate spline on prices 
struct FenglerSurface{T}
    expiries::Vector{T}
    forwards::Vector{T}
    sections::Vector{C2Spline}
end
Base.broadcastable(p::FenglerSurface) = Ref(p)


function calibrateFenglerSurface(ttes::AbstractVector{T},forwards::AbstractVector{T},strikes::AbstractMatrix{T},callPrices::AbstractMatrix{T}, weights::AbstractMatrix{T}; λ=1e-4, solver="COSMO",eps=1e-7) where {T}
#starts with last slice
slices = Vector{C2Spline}(undef,length(ttes))
slices[end] = calibrateFenglerSlice(ttes[end],forwards[end],strikes[end,:],callPrices[end,:],weights[end,:], λ=λ, solver=solver,eps=eps)
#println("calibrated last slice ",slices[end])
for i=length(ttes)-1:-1:1
    ttei = ttes[i]
    forwardi = forwards[i]
    strikesi = strikes[i,:]
    callPricesi = callPrices[i,:]
    weightsi = weights[i,:]
    nextPrices = zeros(T,length(strikesi))
    for j=eachindex(strikesi)
        strikej = strikesi[j]/forwardi*forwards[i+1]
        nextPrices[j] = slices[i+1](strikej)*forwardi/forwards[i+1]
    end
    #println("callPrices ",callPrices[i,:], " nextPrices ",nextPrices)
    slices[i] = calibrateFenglerSlice(ttes[i],forwards[i],strikes[i,:],callPrices[i,:],weights[i,:], λ=λ*ttei, solver=solver,eps=eps,nextPrices=nextPrices)
    #println("calibrated slice ",i," ",slices[i])
end
#returns a surface in prices.
return FenglerSurface(ttes,forwards,slices)
end

function price(s::FenglerSurface, y, t,indexTime::Int = 0)
    if t <= s.expiries[1]
        return s.sections[1]( exp(y)*s.forwards[1])
    elseif t >= s.expiries[end]
        return s.sections[end](exp(y)* s.forwards[end])
    else
        if indexTime == 0
            indexTime = searchsortedlast(s.expiries, t)
        end
        t0 = s.expiries[indexTime]
        var0 = s.sections[indexTime](exp(y)*s.forwards[indexTime])
        t1 = s.expiries[indexTime+1]
        var1 = s.sections[indexTime+1](exp(y)*s.forwards[indexTime+1])
        #linear interpolation in total variance along same logmoneyness
        v = (var1 *  (t - t0) + var0 * (t1 - t)) / (t1 - t0)
        return v 
    end
end
  
function varianceByLogmoneyness(section::C2Spline, y, forward,tte)
    strike = forward*exp(y)
    strike = min(max(strike,section.x[1]),section.x[end])
    price = section(strike)
    if price <= 0
        price = 1e-16
    end
    vol = impliedVolatility(true, price, forward, strike, tte, 1.0)   
    return vol^2
end

function varianceByLogmoneyness(s::FenglerSurface, y, t,indexTime::Int = 0)
    if t <= s.expiries[1]
        return varianceByLogmoneyness(s.sections[1], y, s.forwards[1],s.expiries[1])
    elseif t >= s.expiries[end]
        return varianceByLogmoneyness(s.sections[end], y, s.forwards[end],s.expiries[end] )
    else
        if indexTime == 0
            indexTime = searchsortedlast(s.expiries, t)
        end
        t0 = s.expiries[indexTime]
        var0 = varianceByLogmoneyness(s.sections[indexTime], y, s.forwards[indexTime],s.expiries[indexTime])
        t1 = s.expiries[indexTime+1]
        var1 = varianceByLogmoneyness(s.sections[indexTime+1], y, s.forwards[indexTime+1],s.expiries[indexTime+1])
        #linear interpolation in total variance along same logmoneyness
        v = (var1 * t1 * (t - t0) + var0 * t0 * (t1 - t)) / (t1 - t0)
        return v / t
    end
end

#the next prices are  undiscounted, at constant moneyness regarding the strikes at ti.
function calibrateFenglerSlice(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; λ=1e-4, solver="GI",eps=1e-7, nextPrices::AbstractVector{T}=T[]) where {T}
    n = length(callPrices)
    h = strikes[2:end] - strikes[1:end-1]
    #build Q and R
    Q = spzeros(T, n, n - 2)
    R = spzeros(T, n - 2, n - 2)
    for j = 2:n-1
        a = one(T) / h[j-1]
        b = one(T) / h[j]
        Q[j-1, j-1] = a
        Q[j+1, j-1] = b
        Q[j, j-1] = -a - b
        R[j-1, j-1] = (h[j] + h[j-1]) / 3
    end
    for j = 2:n-2
        R[j-1, j+1-1] = h[j] / 6
        R[j+1-1, j-1] = R[j-1, j+1-1]
    end
    #build A
    A = vcat(Q, -R')
    y = vcat(callPrices.*weights.^2, zeros(T, n - 2))
    mw = sum(weights.^2)/length(weights)
    λ *= mw
    #build B
    B = spzeros(T, 2n - 2, 2n - 2)
    for j = 1:n
        B[j, j] = weights[j]^2
    end
    B[n+1:end, n+1:end] = λ .* R
    #ineq constraints in C
    nextSize = 1
    if length(nextPrices) > 0
        nextSize = length(nextPrices)
    end
    C = spzeros(n - 2 + 4+nextSize, 2n - 2)
    cv = zeros(T, n - 2 + 4+nextSize)
    for j=eachindex(cv)
        cv[j] = -eps
    end
    for j = n+1:2n-2
        C[j-n, j] = one(T)
    end
    C[n-1, 1] = -one(T)
    C[n-1, 2] = one(T)
    cv[n-1] = (strikes[2] - strikes[1])
    C[n, n-1] = one(T)
    C[n, n] = -one(T)
    C[n+1, 1] = one(T)
    cv[n+1] = -(forward - strikes[1])-eps
    C[n+2, n] = one(T)
    if nextSize == 1
        C[n+3, 1] = -one(T)
        cv[n+3] = forward
    else
        for j=1:nextSize
            cv[n+2+j] = nextPrices[j]
            C[n+2+j,j] = -one(T)
        end
    end
    x = if solver == "COSMO"
        settings = COSMO.Settings(verbose=false,eps_abs=1e-16,eps_rel=1e-16)
        model = COSMO.Model()
        #println(size(A')," ", size(C), " ",n)
        Aa = A'
        ba = zeros(T, n - 2)
        constraint1 = COSMO.Constraint(Aa, ba, COSMO.ZeroSet)
        constraint2 = COSMO.Constraint(C, cv, COSMO.Nonnegatives)
        assemble!(model, B, -y, [constraint1; constraint2], settings=settings)
        res = COSMO.optimize!(model)
        res.x
    else
        Aa = hcat(A,C')
        amat, aind = convertSparse(Aa)
        bb = vcat(zeros(T, n - 2), -cv)
        #println(size(Aa)," ",size(bb))
        # sol, lagr, crval, iact, nact, iter = solveQPcompact(B, y, amat, aind, bb, meq=n - 2,factorized=false)
        #  Binvsqrt = inv(Matrix(sparse(cholesky(B).L))) #not sure why is wrong
       #Binvsqrt = sqrt(inv(Matrix(B)))
    #    Binvsqrt = inv(sqrt(Matrix(B)))
    #    Binvsqrt = Matrix(B)
    #    Binvsqrt[n+1:end,n+1:end] = inv(sqrt(@view(Binvsqrt[n+1:end, n+1:end])))
         sol, lagr, crval, iact, nact, iter = solveQPcompact(Matrix(B), y, amat, aind, bb, meq=n - 2,factorized=false)
        sol
    end
    #println("COSMO res ",res)
    spline = C2Spline(strikes, x[1:n], vcat(zero(T), x[n+1:end], zero(T)))
    return spline
end

