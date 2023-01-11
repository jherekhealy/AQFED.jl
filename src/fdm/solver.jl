using LinearAlgebra

abstract type FDMSolverAlgo end

#FIXME LBSolver has only lb, m, lbActive . algo is not mutable.
#       algo  implements decompose(solver, algo, m?) + solve(solver, algo, x, d )
mutable struct LowerBoundSolver{T}
    algo::FDMSolverAlgo
    isLowerBoundActive::Bool
    m::Tridiagonal{T,Vector{T}}
    lowerBound::Vector{T}
end

function decompose(solver::LowerBoundSolver{T}, m::Tridiagonal{T,Vector{T}}) where {T}
    solver.m = m
    decompose(solver, solver.algo, m)
end

function setLowerBoundActive(solver::LowerBoundSolver{T}, isLowerBoundActive::Bool) where {T}
    solver.isLowerBoundActive = isLowerBoundActive
end

function solve!(solver::LowerBoundSolver{T}, x::AbstractArray{T}, d::AbstractArray) where {T}
    if solver.isLowerBoundActive
        solve!(solver, solver.algo, x, d)
    else
        solve!(solver.m, x, d)
    end
end

struct TDMA{T} <: FDMSolverAlgo
end

"""  Thomas Algorithm. `x` contains the output. `d` is modified
 Vectors a,c of length(b)-1    
  A*x = D
  [b[1]   c[1]               ] [ x[1] ]   [ D[1] ]
  [a[1]   b[2]   c[2]        ] [ x[2] ]   [ D[2] ]
  [    ...   ...   ...       ] [ ...  ] = [ ...  ]
  [      a[n-2] b[n-1] c[n-1]] [x[n-1]]   [D[n-1]]
  [             a[n-1]  b[n] ] [ x[n] ]   [ D[n] ]"""
function solve!(solver::TDMA{T}, x::AbstractArray{T}, a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, d::AbstractArray{T}) where {T}
    n = length(d)
    b = copy(b)
    @inbounds for i = 2:n
        wi = a[i-1] / b[i-1]
        b[i] -= wi * c[i-1]
        d[i] -= wi * d[i-1]
    end
    x[n] = d[n] / b[n]
    @inbounds for k = n-1:-1:1
        x[k] = (d[k] - c[k] * x[k+1]) / b[k]
    end
    x
end

function solveIndirect!(solver::TDMA{T}, x::AbstractArray{T}, d::AbstractArray{T}) where {T}
    a = solver.m.dl
    b = solver.m.d
    c = solver.m.du
    solve!(solver, x, a, b, c, d)
end

function decompose(ssolver::LowerBoundSolver{T}, solver::TDMA{T}, m::Tridiagonal{T,Vector{T}}) where {T}
end

function solve!(m::Tridiagonal{T,Vector{T}}, x::AbstractArray{T}, d::AbstractArray{T}) where {T}
    n = length(d)
    b = diag(m)
    @inbounds for i = 2:n
        wi = m[i, i-1] / b[i-1]
        b[i] -= wi * m[i-1, i]
        d[i] -= wi * d[i-1]
    end
    x[n] = d[n] / b[n]
    @inbounds for k = n-1:-1:1
        x[k] = (d[k] - m[k, k+1] * x[k+1]) / b[k]
    end
    x
end

function solve!(ssolver::LowerBoundSolver{T}, solver::TDMA{T}, x::AbstractArray{T}, d::AbstractArray{T}) where {T}
    solve!(ssolver.m, x, d)
end

struct TDMAMax{T} <: FDMSolverAlgo
end

function decompose(ssolver::LowerBoundSolver{T}, solver::TDMAMax{T}, m::Tridiagonal{T,Vector{T}}) where {T}
end
function solve!(ssolver::LowerBoundSolver{T}, solver::TDMAMax{T}, x::AbstractArray{T}, d::AbstractArray{T}) where {T}
    solve!(ssolver.m, x, d)
    @. x = max(x, ssolver.lowerBound)
end



struct BrennanSchwartz{T} <: FDMSolverAlgo
    modifiedLowerBound::Vector{T}
    BrennanSchwartz{T}(size::Int) where {T} = new{T}(zeros(T, size))
end

function decompose(ssolver::LowerBoundSolver{T}, solver::BrennanSchwartz{T}, m::Tridiagonal{T,Vector{T}}) where {T}
end

function solve!(ssolver::LowerBoundSolver{T}, solver::BrennanSchwartz{T}, x::AbstractArray{T}, d::AbstractArray{T}) where {T}
    mul!(solver.modifiedLowerBound, ssolver.m, ssolver.lowerBound)
    n = length(d)
    m = ssolver.m
    b = diag(m)
    d -= solver.modifiedLowerBound
    y = solver.modifiedLowerBound
    if d[1] >= zero(T)
        y[1] = b[1]
        x[1] = d[1]
        @inbounds for i = 2:n
            wi = m[i, i-1] / y[i-1]
            y[i] = b[i] - wi * m[i-1, i]
            x[i] = d[i] - wi * x[i-1]
        end
        x[n] /= y[n]
        x[n] = max(x[n], 0)
        @inbounds for k = n-1:-1:1
            x[k] = max(zero(T), (x[k] - m[k, k+1] * x[k+1]) / y[k])
        end
    else
        y[n] = b[n]
        x[n] = d[n]
        @inbounds for i = n-1:-1:1
            wi = m[i, i+1] / y[i+1]
            y[i] = b[i] - wi * m[i+1, i]
            x[i] = d[i] - wi * x[i+1]
        end
        x[1] /= y[1]
        x[1] = max(x[1], zero(T))
        @inbounds for k = 2:n
            x[k] = max(zero(T), (x[k] - m[k, k-1] * x[k-1]) / y[k])
        end
    end
    @. x += ssolver.lowerBound
    x

end

struct DoubleSweep{T} <: FDMSolverAlgo
    modifiedLowerBound::Vector{T}
    DoubleSweep{T}(size::Int) where {T} = new{T}(zeros(T, size))
end

function decompose(ssolver::LowerBoundSolver{T}, solver::DoubleSweep{T}, m::Tridiagonal{T,Vector{T}}) where {T}
end

function solve!(ssolver::LowerBoundSolver{T}, solver::DoubleSweep{T}, x::AbstractArray{T}, d::AbstractArray{T}) where {T}
    mul!(solver.modifiedLowerBound, ssolver.m, ssolver.lowerBound)
    n = length(d)
    m = ssolver.m
    b = diag(m)
    d -= solver.modifiedLowerBound
    y = solver.modifiedLowerBound
    y[1] = b[1]
    x[1] = d[1]
    @inbounds for i = 2:n
        wi = m[i, i-1] / y[i-1]
        y[i] = b[i] - wi * m[i-1, i]
        x[i] = d[i] - wi * x[i-1]
    end
    x[n] /= y[n]
    x[n] = max(x[n], 0)
    @inbounds for k = n-1:-1:1
        x[k] = max(zero(T), (x[k] - m[k, k+1] * x[k+1]) / y[k])
    end
    y[n] = b[n]
    xb = copy(d)
    #xb[n] = d[n]
    @inbounds for i = n-1:-1:1
        wi = m[i, i+1] / y[i+1]
        y[i] = b[i] - wi * m[i+1, i]
        xb[i] -= wi * xb[i+1]
    end
    xb[1] /= y[1]
    x[1] = max(xb[1], x[1])
    @inbounds for k = 2:n
        x[k] = max(x[k], (xb[k] - m[k, k-1] * x[k-1]) / y[k])
    end
    @. x += ssolver.lowerBound
    x
end



struct LUUL{T} <: FDMSolverAlgo
    modifiedLowerBound::Vector{T}
    l::Vector{T}
    ll::Vector{T}
    uu::Vector{T}
    llb::Vector{T}
    ub::Vector{T}
    uub::Vector{T}
    LUUL{T}(size::Int) where {T} = new{T}(zeros(T, size), zeros(T, size), zeros(T, size), zeros(T, size), zeros(T, size), zeros(T, size), zeros(T, size))
end

function decompose(ssolver::LowerBoundSolver{T}, solver::LUUL{T}, m::Tridiagonal{T,Vector{T}}) where {T}
    d = solver.l
    l = solver.ll
    u = solver.uu
    n = length(solver.l)
    d[1] = m[1, 1]
    u[1] = m[1, 2] / d[1]
    for i = 2:n-1
        l[i-1] = m[i, i-1]
        d[i] = m[i, i] - l[i-1] * u[i-1]
        u[i] = m[i, i+1] / d[i]
    end
    l[n-1] = m[n, n-1]
    d[n] = m[n, n] - l[n-1] * u[n-1]
    l̄ = solver.llb
    d̄ = solver.ub
    ū = solver.uub
    d̄[n] = m[n, n]
    l̄[n-1] = m[n, n-1] / d̄[n]
    for i = n-1:-1:2
        ū[i] = m[i, i+1]
        d̄[i] = m[i, i] - ū[i] * l̄[i]
        l̄[i-1] = m[i, i-1] / d̄[i]
    end
    ū[1] = m[1, 2]
    d̄[1] = m[1, 1] - ū[1] * l̄[1]
end

function solve!(ssolver::LowerBoundSolver{T}, solver::LUUL{T}, x::AbstractArray{T}, r::AbstractArray{T}) where {T}
    n = length(r)
    m = ssolver.m
    mul!(solver.modifiedLowerBound, ssolver.m, ssolver.lowerBound)
    r -= solver.modifiedLowerBound
    y = solver.modifiedLowerBound
    d = solver.l
    l = solver.ll
    u = solver.uu
    y[1] = r[1] / l[1]
    @inbounds for i = 2:n
        y[i] = (r[i] - l[i-1] * y[i-1]) / d[i]
    end
    x[n] = y[n]
    x[n] = max(x[n], 0)
    @inbounds for k = n-1:-1:1
        x[k] = max(zero(T), y[k] - u[k] * x[k+1])
    end
    l̄ = solver.llb
    d̄ = solver.ub
    ū = solver.uub
   
    y[n] = r[n] / d̄[n]
    @inbounds for i = n-1:-1:1
        y[i] = (r[i] - ū[i] * y[i+1]) / d̄[i]
    end
    x[1] = max(y[1], x[1])
    @inbounds for k = 2:n
        x[k] = max(x[k], y[k] - l̄[k-1] * x[k-1])
    end
    @. x += ssolver.lowerBound
    x
end



@enum SolverGuessType begin
    GuessTDMA
    GuessRHS
    GuessLowerBound
end
mutable struct TDMAPolicyIteration{T} <: FDMSolverAlgo
    tolerance::T
    maxIterations::Int
    guessType::SolverGuessType
    callCount::Int
    averageLoop::Float64
    tn::Tridiagonal{T,Vector{T}}
    bn::Vector{T}
    xn::Vector{T}
    TDMAPolicyIteration{T}(size::Int; tolerance=eps(T), maxIterations=10000, guessType=GuessRHS) where {T} = new{T}(tolerance, maxIterations, guessType, 0, 0.0, Tridiagonal(zeros(T, size - 1), zeros(T, size), zeros(T, size - 1)), zeros(T, size), zeros(T, size))
end


function decompose(ssolver::LowerBoundSolver{T}, solver::TDMAPolicyIteration{T}, m::Tridiagonal{T,Vector{T}}) where {T}
end

function solve!(ssolver::LowerBoundSolver{T}, s::TDMAPolicyIteration{T}, x::AbstractArray{T}, d::AbstractArray{T}) where {T}
    sorLoops = 0
    if s.guessType == GuessTDMA
        s.bn[1:end] = d
        solve!(ssolver.m, s.xn, s.bn)
        @. s.xn = max(s.xn, ssolver.lowerBound)

    elseif s.guessType == GuessRHS
        @. s.xn = d
    else
        s.xn[1:end] = ssolver.lowerBound
    end

    errorSq = s.tolerance + one(T)
    upperIndex = length(d)
    lowerIndex = 1
    while errorSq > s.tolerance && sorLoops < s.maxIterations
        s.bn[1:end] = d
        #println(sorLoops, " policy iteration algorithm, error=", sqrt(errorSq)," ",s.xn," ",s.bn, " ")
        copy!(s.tn, ssolver.m)
        i = lowerIndex
        if s.xn[i] * s.tn[i, i] + s.xn[i+1] * s.tn[i, i+1] - s.bn[i] <= s.xn[i] - ssolver.lowerBound[i]
            # lambda = zero(T)
            # s.tn[i,i] -= lambda
            # s.bn[i] -= lambda
        else
            s.tn[i, i] = one(T)
            s.tn[i, i+1] = zero(T)
            s.bn[i] = ssolver.lowerBound[i]
        end
        for i = lowerIndex+1:upperIndex-1
            if s.xn[i-1] * s.tn[i, i-1] + s.xn[i] * s.tn[i, i] + s.xn[i+1] * s.tn[i, i+1] - s.bn[i] <= s.xn[i] - ssolver.lowerBound[i]
                lambda = s.tn[i, i-1] / s.tn[i-1, i-1]
                s.tn[i, i] -= lambda * s.tn[i-1, i]
                s.bn[i] -= lambda * s.bn[i-1]
            else
                s.tn[i, i] = one(T)
                s.tn[i, i+1] = zero(T)
                s.bn[i] = ssolver.lowerBound[i]
            end
        end
        i = upperIndex
        if s.xn[i-1] * s.tn[i, i-1] + s.xn[i] * s.tn[i, i] - s.bn[i] <= s.xn[i] - ssolver.lowerBound[i]
            lambda = s.tn[i, i-1] / s.tn[i-1, i-1]
            s.tn[i, i] -= lambda * s.tn[i-1, i]
            s.bn[i] -= lambda * s.bn[i-1]
        else
            s.tn[i, i] = one(T)
            #s.tn[i,i+1] = zero(T)
            s.bn[i] = ssolver.lowerBound[i]
        end
        x[i] = s.bn[i] / s.tn[i, i]
        for i = upperIndex-1:-1:lowerIndex
            x[i] = (s.bn[i] - s.tn[i, i+1] * x[i+1]) / s.tn[i, i]
        end
        errorSq = zero(T)
        for k = lowerIndex:upperIndex
            errorSq += (s.xn[k] - x[k])^2
            s.xn[k] = x[k]
        end
        sorLoops += 1
    end
    s.callCount += 1
    s.averageLoop += (sorLoops - s.averageLoop) / s.callCount
    if (sorLoops == s.maxIterations)
        println("policy iteration algorithm did not converge, error=", sqrt(errorSq))
    end
end


mutable struct PSOR{T} <: FDMSolverAlgo
    tolerance::T
    maxIterations::Int
    Ω::Float64
    guessType::SolverGuessType
    callCount::Int
    averageLoop::Float64
    rhs::Vector{T}
    PSOR{T}(size::Int; tolerance=16*eps(T)^2, maxIterations=4000, guessType=GuessTDMA) where {T} = new{T}(tolerance, maxIterations, 0.0, guessType, 0, 0.0, zeros(T, size))
end


function decompose(ss::LowerBoundSolver{T}, solver::PSOR{T}, m::Tridiagonal{T,Vector{T}}) where {T}
end

function solve!(ss::LowerBoundSolver{T}, s::PSOR{T}, x::AbstractArray{T}, d::AbstractArray{T}) where {T}
    upperIndex = length(d)
    lowerIndex = 1
    if s.Ω == 0.0
        rhoG = 0.0
        for i = lowerIndex+1:upperIndex-1
            rhoG = max(rhoG, (abs(ss.m[i, i-1]) + abs(ss.m[i, i+1])) / abs(ss.m[i, i]))
        end
        rhoG = max(0.04, min(0.998, rhoG))
        s.Ω = 2.0 / (1.0 + sqrt(1 - rhoG^2))
        println("using Ω=",s.Ω, " ",ss.m[end,end-1]," ",ss.m[end,end]," ",ss.m[end-1,end-1])
    end
    sorLoops = 0
    if s.guessType == GuessTDMA
        s.rhs[1:end] = d
        solve!(ss.m, x, s.rhs)
        @. x = max(x, ss.lowerBound)
    elseif s.guessType == GuessRHS
        @. x = max(ss.lowerBound, d)
    else
        x[1:end] = ss.lowerBound
    end
    tol = s.tolerance*length(x)^2
    errorSq = tol + one(T)
    while errorSq > tol && sorLoops < s.maxIterations
        errorSq = zero(T)
        for k = lowerIndex:upperIndex
            back = if k == lowerIndex
                ss.m[k, k+1] * x[k+1]
            elseif k == upperIndex
                ss.m[k, k-1] * x[k-1]
            else
                ss.m[k, k+1] * x[k+1] + ss.m[k, k-1] * x[k-1]
            end
            y = (d[k] - back) / ss.m[k, k] 
            if  k == upperIndex
                y = max(ss.lowerBound[k], (d[k]-ss.m[k, k-1] * x[k-1])/ ss.m[k, k] )
            else
              y = max(ss.lowerBound[k], x[k] + s.Ω * (y - x[k]))
            end           
            errorSq += (y - x[k])^2
            x[k] = y
        end
        sorLoops += 1
    end
    s.callCount += 1
    s.averageLoop += (sorLoops - s.averageLoop) / s.callCount
    if (sorLoops == s.maxIterations)
        println("PSOR algorithm did not converge, error=", sqrt(errorSq))
    end
end