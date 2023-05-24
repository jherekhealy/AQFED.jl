using LinearAlgebra

struct Multiquadric{TX,TV,T}
    knots::TX
    α::TV
    c::T
end

struct ThinPlateSpline{TX,TV}
    knots::TX
    α::TV
end



function calibrateMultiquadric(ttes::AbstractArray{T}, forward::T, ys::AbstractMatrix{T}, vols::AbstractMatrix{T};cy=(ys[2]-ys[1])*4,cx=(ttes[end]-ttes[end-1])*2) where {T}
    ψ = x -> quadric(x,[cx,cy])
    #knots = matrix nStrikes*nTTE value x,y
    tys = zeros(T,(size(ys,1)*size(ys,2),2))
    variances = zeros(T,size(ys,1)*size(ys,2))
    for i=1:size(ys,1)
        for j=1:size(ys,2)
            tys[i+(j-1)*size(ys,1),:] = [ttes[i],ys[i,j]]
            variances[i+(j-1)*size(ys,1)] = vols[i,j]^2
        end
    end
    knots = copy(tys)
    α = calibrateRBF(ψ, tys, variances, knots=knots)

    return Multiquadric(knots,α,(cx,cy))
end


function calibrateRBF(ψ, tys::AbstractMatrix{T}, variances::AbstractArray{T}; knots) where {T}
        n = size(tys,1)
        m = size(knots,1)
        A = zeros(T,(n,m))
        for i=1:n
            for j=1:m
                A[i,j] =  ψ(tys[i,:].-knots[j,:])
            end
        end
        α = A \ variances
        return α
    end

function calibrateMultiquadric(tte::T, forward::T, ys::AbstractArray{T}, vols::AbstractArray{T}, weights::AbstractArray{T}; knots=[ys[1],ys[floor(Int,end/4)],ys[floor(Int,end/2)],ys[floor(Int,3*end/4)],ys[end]], c=2*maximum(ys[2:end]-ys[1:end-1]), noarbFactor = 2*one(T)) where {T}
    ψ = x -> quadric(x,c)
    α = calibrateRBF(ψ, ys, vols, weights, knots=knots, noarbFactor=noarbFactor)
    return Multiquadric(knots,α,c)
end


function calibrateThinPlateSpline(tte::T, forward::T, ys::AbstractArray{T}, vols::AbstractArray{T}, weights::AbstractArray{T}; knots=[ys[1],ys[floor(Int,end/4)],ys[floor(Int,end/2)],ys[floor(Int,3*end/4)],ys[end]], c=2*maximum(ys[2:end]-ys[1:end-1]), noarbFactor = 2*one(T)) where {T}
    ψ = x -> thinplate(x)
    α = calibrateRBF(ψ, ys, vols, weights, knots=knots, noarbFactor=noarbFactor)
    return ThinPlateSpline(knots,α)
end

function calibrateRBF(ψ, ys::AbstractArray{T}, vols::AbstractArray{T}, weights::AbstractArray{T}; knots=[ys[1],ys[floor(Int,end/4)],ys[floor(Int,end/2)],ys[floor(Int,3*end/4)],ys[end]],  noarbFactor = 2*one(T)) where {T}
variances = vols.^2
    n = length(ys)
    m = length(knots)
    A = zeros(T,(n,m))
    v= zeros(T,n)
    for i=1:n
        for j=1:m
            A[i,j] = sqrt(weights[i])* ψ(ys[i]-knots[j])
        end
        v[i] = sqrt(weights[i])*variances[i]
    end
    α = A \ v
    return α
end

function evaluate(m::Multiquadric{TV},x)  where {TV}
    sum=zero(m.c)
    for (xi,αi) in zip(m.knots,m.α)
        sum += αi*quadric(x-xi,m.c)
    end
    return sum
end

function evaluate(m::ThinPlateSpline{TV},x)  where {TV}
    sum=zero(x)
    for (xi,αi) in zip(m.knots,m.α)
        sum += αi*thinplate(x-xi)
    end
    return sum
end

function quadric(x::TX,c::TC) where {TX,TC}
    sqrt(one(TX)+norm(x./c)^2)
end

function quadric(x::AbstractArray{TX},c::AbstractArray{TC}) where {TX,TC}
    sqrt(one(TX)+norm(x./c)^2)
end

function thinplate(x::TX) where {TX}
    return x == zero(TX) ? zero(TX) :  x^2*log(x^2)
end

Base.broadcastable(p::Multiquadric) = Ref(p)

(spl::Multiquadric)(x) = evaluate(spl, x)

Base.broadcastable(p::ThinPlateSpline) = Ref(p)

(spl::ThinPlateSpline)(x) = evaluate(spl, x)
