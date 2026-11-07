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


"""
    calibrateFenglerSurface(ttes, forwards, strikes, callPrices, weights; λ, solver, eps, calendarMargin=0, calendarRefine=0)

Fengler surface, from the longest expiry to the shortest, with the calendar constraint against the next expiry at the knots
(row i of `strikes`). `eps` is a lower bound of the density: a number, or a vector per expiry (a number or a vector at the
interior knots). `calendarMargin` δ requires the calendar spread to be at least min(δ, `calendarFraction` times the time
value of the next expiry) (in price; deep out of the money, both prices vanish). With `calendarRefine` = r > 0,
the calendar constraint is also imposed at the middle of each interval between the knots inside the range of the next
expiry, and up to r times at the points of a grid of `calendarPoints` points per interval where the spread is still below δ
(one per interval, the most violated), so that the spread is at least δ on that grid.
"""
function calibrateFenglerSurface(ttes::AbstractVector{T},forwards::AbstractVector{T},strikes::AbstractMatrix{T},callPrices::AbstractMatrix{T}, weights::AbstractMatrix{T}; λ=1e-4, solver="COSMO",eps=1e-7,
    calendarMargin=zero(T), calendarFraction=T(0.01), calendarRefine::Int=0, calendarPoints::Int=8) where {T}
epsOf(i) = eps isa Real ? eps : eps[i]
#starts with last slice
slices = Vector{C2Spline}(undef,length(ttes))
slices[end] = calibrateFenglerSlice(ttes[end],forwards[end],strikes[end,:],callPrices[end,:],weights[end,:], λ=λ, solver=solver,eps=epsOf(length(ttes)))
for i=length(ttes)-1:-1:1
    ttei = ttes[i]
    forwardi = forwards[i]
    strikesi = strikes[i,:]
    nextRaw(x) = slices[i+1](x / forwardi * forwards[i+1]) * forwardi / forwards[i+1]
    nextAt(x) = (p = nextRaw(x); p - min(calendarMargin, calendarFraction * max(p - max(forwardi - x, zero(T)), zero(T))))
    inNext(x) = slices[i+1].x[1] <= x / forwardi * forwards[i+1] <= slices[i+1].x[end]
    #outside the knots of the next expiry its spline is only extended linearly: no calendar constraint there (the bound is
    #the forward, which a call price never exceeds)
    nextPrices = [inNext(x) ? nextAt(x) : forwardi for x in strikesi]
    extra = Tuple{T,T}[]
    if calendarRefine > 0
        for j = 1:length(strikesi)-1
            xm = (strikesi[j] + strikesi[j+1]) / 2
            inNext(xm) && push!(extra, (xm, nextAt(xm)))
        end
    end
    slices[i] = calibrateFenglerSlice(ttes[i],forwards[i],strikesi,callPrices[i,:],weights[i,:], λ=λ*ttei, solver=solver,eps=epsOf(i),nextPrices=nextPrices, extraCalendar=extra)
    for _ = 1:calendarRefine
        added = 0
        for j = 1:length(strikesi)-1
            xs = range(strikesi[j], strikesi[j+1], length=calendarPoints + 2)[2:end-1]
            worst, xw = zero(T), zero(T)
            for x in xs
                inNext(x) || continue
                v = nextAt(x) - slices[i](x)
                if v < worst
                    worst, xw = v, x
                end
            end
            if worst < -1e-14
                push!(extra, (xw, nextAt(xw)))
                added += 1
            end
        end
        added == 0 && break
        slices[i] = calibrateFenglerSlice(ttes[i],forwards[i],strikesi,callPrices[i,:],weights[i,:], λ=λ*ttei, solver=solver,eps=epsOf(i),nextPrices=nextPrices, extraCalendar=extra)
    end
end
#returns a surface in prices.
return FenglerSurface(ttes,forwards,slices)
end

"""
    calibrateFenglerSurfaceJoint(ttes, forwards, strikes, callPrices, weights; λ, eps, calendarMargin=0, calendarFraction=0.01, calendarRefine=3)

Fengler surface solved as a single quadratic program over all the expiries (Clarabel), instead of expiry by expiry from the
longest: the objective is the sum of the objectives of the slices, and the calendar constraints link the variables of
consecutive expiries (the spline of an expiry at any strike is linear in its values and second derivatives at the knots).
A calendar arbitrage of the inputs is then shared between the expiries in the least-squares sense, instead of being taken
out of the shorter expiry. Calendar constraints at the knots and at the middle of the intervals that lie within the knots
of the next expiry, with a margin min(δ, `calendarFraction` times the time value of the input price of the next expiry), and
up to `calendarRefine` times at the most violated of 8 points per interval. `eps`: lower bound of the density, a number or
a vector per expiry (a number or a vector at the interior knots).
"""
function calibrateFenglerSurfaceJoint(ttes::AbstractVector{T}, forwards::AbstractVector{T}, strikes::AbstractMatrix{T}, callPrices::AbstractMatrix{T}, weights::AbstractMatrix{T};
    λ=zero(T), eps=zero(T), calendarMargin=zero(T), calendarFraction=T(0.01), calendarRefine::Int=3) where {T}
    m, n = size(strikes)
    nv = 2n - 2
    off(j) = (j - 1) * nv
    epsOf(j) = eps isa Real ? fill(T(eps), n - 2) : (eps[j] isa Real ? fill(T(eps[j]), n - 2) : eps[j])
    #objective and per-slice constraints (they do not change with the refinement)
    PI, PJ, PV = Int[], Int[], T[]
    q = zeros(T, m * nv)
    EI, EJ, EV = Int[], Int[], T[]
    nEq = 0
    II, IJ, IV = Int[], Int[], T[]  #inequalities C x + cv >= 0
    cv = T[]
    addIneq!(cols, vals, c) = (r = length(cv) + 1; for (cc, vv) in zip(cols, vals); push!(II, r); push!(IJ, cc); push!(IV, vv); end; push!(cv, c))
    for j = 1:m
        K = strikes[j, :]; h = diff(K); w2 = weights[j, :] .^ 2; o = off(j)
        mw = sum(w2) / n
        for i = 1:n
            push!(PI, o + i); push!(PJ, o + i); push!(PV, w2[i])
            q[o+i] = -callPrices[j, i] * w2[i]
        end
        #R (in γ indices 1..n-2) and Q
        for k = 2:n-1
            r = k - 1 #γ_k at column o + n + r
            push!(PI, o + n + r); push!(PJ, o + n + r); push!(PV, λ * ttes[j] * mw * (h[k-1] + h[k]) / 3)
            if k < n - 1
                push!(PI, o + n + r); push!(PJ, o + n + r + 1); push!(PV, λ * ttes[j] * mw * h[k] / 6)
            end
            #equality Qᵀz - Rγ = 0, row for interior knot k
            nEq += 1
            push!(EI, nEq); push!(EJ, o + k - 1); push!(EV, 1 / h[k-1])
            push!(EI, nEq); push!(EJ, o + k); push!(EV, -1 / h[k-1] - 1 / h[k])
            push!(EI, nEq); push!(EJ, o + k + 1); push!(EV, 1 / h[k])
            push!(EI, nEq); push!(EJ, o + n + r); push!(EV, -(h[k-1] + h[k]) / 3)
            k > 2 && (push!(EI, nEq); push!(EJ, o + n + r - 1); push!(EV, -h[k-1] / 6))
            k < n - 1 && (push!(EI, nEq); push!(EJ, o + n + r + 1); push!(EV, -h[k] / 6))
        end
        ev = epsOf(j)
        for r = 1:n-2
            addIneq!([o + n + r], [one(T)], -ev[r])  #γ >= eps
        end
        F = forwards[j]
        addIneq!([o + 1, o + 2], [-one(T), one(T)], h[1])          #slope at the first knot >= -1 (as the slice solver)
        addIneq!([o + n - 1, o + n], [one(T), -one(T)], zero(T))   #decreasing at the last knot
        addIneq!([o + 1], [one(T)], -(F - K[1]))                   #above the intrinsic value
        addIneq!([o + n], [one(T)], zero(T))                        #non-negative
        j == m && addIneq!([o + 1], [-one(T)], F)                   #below the forward
    end
    nBase = length(cv)
    #calendar points: (j, x) with x a strike of expiry j (the next expiry at the same forward moneyness)
    pts = [Tuple{Int,T}[] for _ = 1:m-1]
    inNext(j, x) = strikes[j+1, 1] <= x / forwards[j] * forwards[j+1] <= strikes[j+1, end]
    for j = 1:m-1, i = 1:n
        inNext(j, strikes[j, i]) && push!(pts[j], (j, strikes[j, i]))
        i < n && (xm = (strikes[j, i] + strikes[j, i+1]) / 2; inNext(j, xm) && push!(pts[j], (j, xm)))
    end
    rowj, rown = zeros(T, nv), zeros(T, nv)
    #input price of the next expiry at x (linear interpolation of the inputs), for the margin
    function nextInput(j, x)
        xs = x / forwards[j] * forwards[j+1]
        K = strikes[j+1, :]; i = clamp(searchsortedlast(K, xs), 1, n - 1)
        a = (xs - K[i]) / (K[i+1] - K[i])
        return ((1 - a) * callPrices[j+1, i] + a * callPrices[j+1, i+1]) * forwards[j] / forwards[j+1]
    end
    P = sparse(PI, PJ, PV, m * nv, m * nv)
    P = P + transpose(triu(P, 1)) #symmetric
    P = P + T(1e-14) * maximum(abs.(PV)) * I
    Eq = sparse(EI, EJ, EV, nEq, m * nv)
    local sol
    slices = Vector{C2Spline}(undef, m)
    for iter = 0:calendarRefine
        cI, cJ, cV = copy(II), copy(IJ), copy(IV); c = copy(cv)
        for j = 1:m-1, (_, x) in pts[j]
            fenglerEvaluationRow!(rowj, strikes[j, :], x)
            fenglerEvaluationRow!(rown, strikes[j+1, :], x / forwards[j] * forwards[j+1])
            r = length(c) + 1
            ratio = forwards[j] / forwards[j+1]
            for (k, v) in enumerate(rown); v != 0 && (push!(cI, r); push!(cJ, off(j + 1) + k); push!(cV, v * ratio)); end
            for (k, v) in enumerate(rowj); v != 0 && (push!(cI, r); push!(cJ, off(j) + k); push!(cV, -v)); end
            pin = nextInput(j, x)
            push!(c, -min(calendarMargin, calendarFraction * max(pin - max(forwards[j] - x, zero(T)), zero(T))))
        end
        Cm = sparse(cI, cJ, cV, length(c), m * nv)
        G = vcat(Eq, -Cm)
        hv = vcat(zeros(T, nEq), c)
        sol = fenglerClarabel(sparse(triu(P)), q, G, hv, nEq, length(c))
        for j = 1:m
            o = off(j)
            slices[j] = C2Spline(strikes[j, :], sol[o+1:o+n], vcat(zero(T), sol[o+n+1:o+nv], zero(T)))
        end
        iter == calendarRefine && break
        added = 0
        for j = 1:m-1
            K = strikes[j, :]
            for i = 1:n-1
                worst, xw = zero(T), zero(T)
                for x in range(K[i], K[i+1], length=10)[2:end-1]
                    inNext(j, x) || continue
                    v = slices[j+1](x / forwards[j] * forwards[j+1]) * forwards[j] / forwards[j+1] - slices[j](x)
                    v < worst && ((worst, xw) = (v, x))
                end
                if worst < -1e-14
                    push!(pts[j], (j, xw)); added += 1
                end
            end
        end
        added == 0 && break
    end
    return FenglerSurface(collect(ttes), collect(forwards), slices)
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
"""
    calibrateFenglerSlice(tte, forward, strikes, callPrices, weights; λ, solver, eps, nextPrices, knots=nothing)

Arbitrage-free cubic spline in prices of Fengler. Without `knots`, the knots are the strikes and the spline smooths the
prices at the strikes. With `knots` (fewer than the strikes, covering them), the spline lives on the knots and the quotes
are fitted by least squares, through the matrix of the values of the spline at the strikes; the constraints and the
penalty are on the knots, and `nextPrices` (calendar) are at the knots.
"""
function calibrateFenglerSlice(tte::T, forward::T, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; λ=1e-4, solver="GI",eps=1e-7, nextPrices::AbstractVector{T}=T[], knots=nothing,
    calendarMargin=zero(T), extraCalendar::AbstractVector{Tuple{T,T}}=Tuple{T,T}[]) where {T}
    if knots !== nothing
        return calibrateFenglerSliceKnots(tte, forward, collect(T, knots), strikes, callPrices, weights; λ=λ, solver=solver, eps=eps, nextPrices=nextPrices)
    end
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
    nExtra = length(extraCalendar)
    C = spzeros(n - 2 + 4+nextSize+nExtra, 2n - 2)
    cv = zeros(T, n - 2 + 4+nextSize+nExtra)
    #eps: a lower bound of the density, a number, or a vector at the interior knots (the other bounds are then exact)
    epsScalar = eps isa Real ? eps : zero(T)
    for j=eachindex(cv)
        cv[j] = -epsScalar
    end
    if !(eps isa Real)
        for j = 1:n-2
            cv[j] = -eps[j]
        end
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
    cv[n+1] = -(forward - strikes[1])-epsScalar
    C[n+2, n] = one(T)
    if nextSize == 1 && length(nextPrices) == 0
        C[n+3, 1] = -one(T)
        cv[n+3] = forward
    else
        for j=1:nextSize
            cv[n+2+j] = nextPrices[j] - calendarMargin
            C[n+2+j,j] = -one(T)
        end
    end
    #calendar spreads between the knots: the spline at x is linear in (z, γ), at most the next price minus the margin
    row = zeros(T, 2n - 2)
    for (e, (xe, pe)) in enumerate(extraCalendar)
        fenglerEvaluationRow!(row, strikes, xe)
        r = n + 2 + nextSize + e
        for (c, v) in enumerate(row)
            v != 0 && (C[r, c] = -v)
        end
        cv[r] = pe - calendarMargin
    end
    x = if solver == "Clarabel"
        #min xᵀBx/2 - yᵀx s.t. Aᵀx = 0 (the spline relations) and C x + cv ≥ 0: Clarabel form G x + s = h with s in the
        #zero cone for the equalities and in the non-negative cone for the inequalities
        G = vcat(sparse(transpose(A)), -sparse(C))
        h = vcat(zeros(T, n - 2), cv)
        fenglerClarabel(sparse(triu(sparse(B))), -Vector(y), G, h, n - 2, length(cv))
    elseif solver == "COSMO"
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




#the least-squares variant on given knots: the values z at the knots and the second derivatives γ at the interior knots,
#x = (z, γ); the spline at a strike s in [k_i, k_{i+1}] is linear in x
function fenglerEvaluationRow!(row, knots::AbstractVector{T}, s) where {T}
    n = length(knots)
    row .= zero(T)
    i = clamp(searchsortedlast(knots, s), 1, n - 1)
    h = knots[i+1] - knots[i]
    a, b = s - knots[i], knots[i+1] - s
    row[i] += b / h
    row[i+1] += a / h
    #γ_i, γ_{i+1} at the interior knots only (zero at the ends: natural spline)
    i > 1 && (row[n+i-1] += -a * b / 6 * (1 + b / h))
    i + 1 < n && (row[n+i] += -a * b / 6 * (1 + a / h))
    return row
end

function calibrateFenglerSliceKnots(tte::T, forward::T, knots::AbstractVector{T}, strikes::AbstractVector{T}, callPrices::AbstractVector{T}, weights::AbstractVector{T}; λ=1e-4, solver="Clarabel", eps=1e-7, nextPrices::AbstractVector{T}=T[]) where {T}
    n = length(knots)
    h = knots[2:end] - knots[1:end-1]
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
        R[j-1, j] = h[j] / 6
        R[j, j-1] = R[j-1, j]
    end
    A = vcat(Q, -R')
    #least squares on the quotes: E x ≈ c, with E the values of the spline at the strikes
    E = zeros(T, length(strikes), 2n - 2)
    row = zeros(T, 2n - 2)
    for (q, s) in enumerate(strikes)
        E[q, :] = fenglerEvaluationRow!(row, knots, s)
    end
    W2 = weights .^ 2
    mw = sum(W2) / length(W2)
    B = Matrix(E' * (W2 .* E))
    B[n+1:end, n+1:end] .+= λ * mw .* Matrix(R)
    y = E' * (W2 .* callPrices)
    nextSize = length(nextPrices) > 0 ? length(nextPrices) : 1
    C = spzeros(T, n - 2 + 4 + nextSize, 2n - 2)
    cv = fill(-eps, n - 2 + 4 + nextSize)
    for j = n+1:2n-2
        C[j-n, j] = one(T)
    end
    C[n-1, 1] = -one(T)
    C[n-1, 2] = one(T)
    cv[n-1] = h[1]
    C[n, n-1] = one(T)
    C[n, n] = -one(T)
    C[n+1, 1] = one(T)
    cv[n+1] = -(forward - knots[1]) - eps
    C[n+2, n] = one(T)
    if nextSize == 1 && length(nextPrices) == 0
        C[n+3, 1] = -one(T)
        cv[n+3] = forward
    else
        for j = 1:nextSize
            cv[n+2+j] = nextPrices[j]
            C[n+2+j, j] = -one(T)
        end
    end
    G = vcat(sparse(transpose(A)), -sparse(C))
    hv = vcat(zeros(T, n - 2), cv)
    Bs = sparse(B + 1e-14 * maximum(abs.(B)) * I)
    x = fenglerClarabel(sparse(triu(Bs)), -Vector(y), G, hv, n - 2, length(cv))
    return C2Spline(knots, x[1:n], vcat(zero(T), x[n+1:end], zero(T)))
end

const FENGLER_LAST_TOLERANCE = Ref(0.0) #tolerance at which the last Clarabel solve succeeded
const FENGLER_CLARABEL_SETTINGS = Ref{Any}((;)) #additional Clarabel settings (named tuple)
#min xᵀPx/2 + qᵀx s.t. G x + s = h, s in the zero cone (first nEq rows) and the non-negative cone; the tolerance is relaxed
#from 1e-10 when the interior point method does not progress (prices spanning many orders of magnitude)
function fenglerClarabel(P, q, G, h, nEq, nIneq)
    local result
    for tol in (1e-10, 1e-8, 1e-6)
        settings = Clarabel.Settings(; verbose=false, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol, max_iter=500, FENGLER_CLARABEL_SETTINGS[]...)
        model = Clarabel.Solver()
        Clarabel.setup!(model, P, q, G, h, [Clarabel.ZeroConeT(nEq), Clarabel.NonnegativeConeT(nIneq)], settings)
        result = Clarabel.solve!(model)
        if result.status == Clarabel.SOLVED || result.status == Clarabel.ALMOST_SOLVED
            FENGLER_LAST_TOLERANCE[] = tol
            return result.x
        end
    end
    throw(error("Clarabel: ", result.status))
end
