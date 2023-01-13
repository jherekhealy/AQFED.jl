export UniformGrid, LogGrid, ShiftedGrid, CubicGrid, SinhGrid, makeArray, SmoothlyDeformedGrid
using PolynomialRoots
using PPInterpolation

abstract type Grid end

struct CubicGrid <: Grid
    α::Float64
end


# xi must have points in [0,1], typically uniform.
function makeArray(grid::CubicGrid, xi::AbstractArray{T}, Smin::T, Smax::T, starPoints::AbstractArray{T}, isMiddle::AbstractArray{Bool}) where {T}
    αScaled = grid.α * (Smax - Smin)
    coeff = one(T) / 6
    starMid = zeros(T, length(starPoints) + 1)
    starMid[1] = Smin
    starMid[2:end-1] = (starPoints[1:end-1] + starPoints[2:end]) / 2
    starMid[end] = Smax
    c1 = zeros(T, length(starPoints))
    c2 = zeros(T, length(starPoints))
    for i = 1:length(starPoints)
        local r = filter(isreal, PolynomialRoots.roots([(starPoints[i] - starMid[i]) / αScaled, one(T), zero(T), coeff]))
        c1[i] = real(sort(r)[1])
        local r = filter(isreal, PolynomialRoots.roots([(starPoints[i] - starMid[i+1]) / αScaled, one(T), zero(T), coeff]))
        c2[i] = real(sort(r)[1])
    end
    dd = zeros(T, length(starPoints) + 1)
    dl = zeros(T, length(starPoints))
    dr = zeros(T, length(starPoints))
    @. dl[1:end-1] = -αScaled * (3 * coeff * (c2[2:end] - c1[2:end]) * c1[2:end]^2 + c2[2:end] - c1[2:end])
    @. dr[2:end] = -αScaled * (3 * coeff * (c2[1:end-1] - c1[1:end-1]) * c2[1:end-1]^2 + c2[1:end-1] - c1[1:end-1])
    dd[2:end-1] = -dl[1:end-1] - dr[2:end]
    dd[1] = one(T)
    dd[end] = one(T)
    rhs = zeros(Float64, length(dd))
    rhs[end] = one(T)
    lhs = Tridiagonal(dl, dd, dr)
    local d = lhs \ rhs
    #  println("d ",d)
    @. c1 /= d[2:end] - d[1:end-1]
    @. c2 /= d[2:end] - d[1:end-1]
    #now transform
    dIndex = 2
    Sip = zeros(length(xi))
    for i = 2:length(xi)-1
        ui = xi[i]
        while (dIndex <= length(d) && d[dIndex] < ui)
            dIndex += 1
        end
        dIndex = min(dIndex, length(d))
        t = c2[dIndex-1] * (ui - d[dIndex-1]) + c1[dIndex-1] * (d[dIndex] - ui)
        Sip[i] = starPoints[dIndex-1] + αScaled * t * (coeff * t^2 + 1)
    end    
    pinGrid(Sip,Smin,Smax)
Sip
end

struct SinhGrid <: Grid
    α::Float64
   
end
function makeArray(grid::SinhGrid, x::AbstractArray{T}, min::T, max::T, starPoints::AbstractArray{T}, isMiddle::AbstractArray{Bool}) where {T}
    specialPoint = starPoints[end]
    α =   grid.α * (max - min)  
    c1 = asinh((min - specialPoint) / α)
	c2 = asinh((max - specialPoint) / α)
    return @. specialPoint + + α*sinh(c2*x+c1*(1-x))
 end

 # to place boundaries exactly on grid
function pinGrid(Sip::AbstractArray{T},Smin::T,Smax::T) where {T}
    Sip[1] = Smin
    Sip[end] = Smax
end

#to place boundaries in the middle
function shiftGrid(Sip::AbstractArray{T},Smin::T,Smax::T) where {T}
    Sip[1] -= (Sip[2] - Smin)
    Sip[end] += (Smax - Sip[end-1])
    return Sip
end

struct UniformGrid <: Grid
    withZero::Bool
end

function makeArray(grid::UniformGrid, x::AbstractArray{T}, min::T, max::T, starPoints::AbstractArray{T}, isMiddle::AbstractArray{Bool}) where {T}
    Li = if isnan(min) 
        zero(T)
    else
        min
    end
    Si = @. Li + x *(max - Li)
    if Li != zero(T) && grid.withZero
        prepend!(Si, zero(T))
    end
    return Si
end

#A shifted grid such that the first starPoint is in the middle
struct ShiftedGrid{TGrid} <: Grid
    delegate::TGrid
end

function makeArray(grid::ShiftedGrid{TGrid}, x::AbstractArray{T}, min::T, max::T, starPoints::AbstractArray{T}, isMiddle::AbstractArray{Bool}) where {T,TGrid}
    Si = makeArray(grid.delegate, x,min,max,starPoints,isMiddle)
    if !isempty(starPoints)
    strikeIndex = searchsortedlast(Si, starPoints[end]) 
    diff = starPoints[end] - (Si[strikeIndex]+Si[strikeIndex+1])/2
    if diff^2 > eps(T)
        @. Si += diff        
    end
    if min == zero(T)
        Si[1] = min
    end
    end
    Si
end


struct SmoothlyDeformedGrid{TGrid} <: Grid
    delegate::TGrid
end
function makeArray(grid::SmoothlyDeformedGrid{TGrid}, x::AbstractArray{T}, min::T, max::T, starPoints::AbstractArray{T}, isMiddle::AbstractArray{Bool}) where {TGrid, T}    
    Si = makeArray(grid.delegate, x, min, max, starPoints,isMiddle)
    n = length(Si)-1
    u = zeros(T,length(starPoints)+2)
    v = zeros(T,length(starPoints)+2)
    v[1] = x[1]
    u[1] = 0.0
    for (i,point) = enumerate(starPoints)
        k = searchsortedlast(Si,point )        #Sk between Si and Si+1
        ustar = x[k] + (x[k + 1] - x[k]) * (point - Si[k]) / (Si[k + 1] - Si[k])
        u[i+1] = floor(ustar*n+0.5)/n
        if isMiddle[i]
            u[i+1] += 0.5/n
        end
        if u[i+1] <= u[i] #simple but not optimal ways to deal with the case where points are too close-by
			u[i+1] = u[i] + 1.0/n
		end
        v[i+1] = ustar
    end
    threshold = 1.0-0.5/n
    if u[end-1] > threshold #may happen because we shifted the values for close-by points
        @. u /= (u[end-1]/threshold)
    end
    u[end] = 1.0
    v[end] = x[end]
    pp = makeCubicPP(u,v,PPInterpolation.SECOND_DERIVATIVE,zero(T),PPInterpolation.SECOND_DERIVATIVE,zero(T),PPInterpolation.C2Hyman89())
    ppinv = PPInterpolation.makeLinearPP(collect(x),Si)
    S = similar(x)
    evaluateSorted!(pp,S,x)
    Si = similar(x)
    evaluateSorted!(ppinv,Si,S)

    Si
end

function makeArray(grid::SmoothlyDeformedGrid{UniformGrid}, x::AbstractArray{T}, min::T, max::T, starPoints::AbstractArray{T}, isMiddle::AbstractArray{Bool}) where {T}    
    n = length(x)-1
    u = zeros(T,length(starPoints)+2)
    v = zeros(T,length(starPoints)+2)
    v[1] = min
    u[1] = 0.0
    for (i,point) = enumerate(starPoints)
        u[i+1] = floor((point - min)/(max-min)*n+0.5)/n
        if isMiddle[i]
            u[i+1] += 0.5/n
        end
        if u[i+1] <= u[i] #simple but not optimal ways to deal with the case where points are too close-by
			u[i+1] = u[i] + 1.0/n
		end
        v[i+1] = point
    end
    threshold = 1.0-0.5/n
    if u[end-1] > threshold #may happen because we shifted the values for close-by points
        @. u /= (u[end-1]/threshold)
    end
    u[end] = 1.0
    v[end] = max
    # println("u ",u," v ",v)
    pp = makeCubicPP(u,v,PPInterpolation.SECOND_DERIVATIVE,zero(T),PPInterpolation.SECOND_DERIVATIVE,zero(T),PPInterpolation.C2Hyman89())
    S = similar(x)
    evaluateSorted!(pp,S,x)
    return S
    #return makeArray(grid.delegate,x,min,max,starPoints,isMiddle) #@. pp(x)
end


struct LogGrid <: Grid
    withZero::Bool
end

function makeArray(grid::LogGrid, x::AbstractArray{T}, min::T, max::T, starPoints::AbstractArray{T}, isMiddle::AbstractArray{Bool}) where {T}
    Si = @. exp( log(min) + x * (log(max) - log(min)))
    if grid.withZero
        prepend!(Si, zero(T))
    end
    return Si
end       


function makeArray(grid::ShiftedGrid{LogGrid}, x::AbstractArray{T}, min::T, max::T, starPoints::AbstractArray{T}, isMiddle::AbstractArray{Bool}) where {T}
    Si = @. exp( log(min) + x * (log(max) - log(min)))
    if !isempty(starPoints)
        strikeIndex = searchsortedlast(Si, starPoints[end]) #FIXME handle strikeIndex=end
        println("strikkeKIndex ",strikeIndex, " in ",Si)
        diff = exp(log(starPoints[end]) - log((Si[strikeIndex]+Si[strikeIndex+1])/2))
        if diff^2 > eps(T)
            @. Si *= diff            
        end
    end
    if grid.delegate.withZero
        prepend!(Si, zero(T))
    end
    Si
end
