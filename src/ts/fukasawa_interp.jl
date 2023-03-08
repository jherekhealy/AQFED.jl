using PPInterpolation
using Roots

d2(y,v) = - y/sqrt(v) - sqrt(v)/2

struct FukasawaInterpolator 
    spline::PPInterpolation.PP
    ys::Vector{Float64}
    tte::Float64
    forward::Float64
end

Base.broadcastable(p::FukasawaInterpolator) = Ref(p)


function makeFukasawaInterpolator(tte, forward, ys, vols) #eventually provide askvols as well    
    vs = [σ^2 for σ in vols]    
    zs = [-d2(y,v * tte) for (y, v) in zip(ys, vs)]
    spline = makeCubicPP(zs, vs, PPInterpolation.SECOND_DERIVATIVE,0.0, PPInterpolation.SECOND_DERIVATIVE,0.0, C2Hyman89())
    #Note: Fukasawa does not preserve monotonicity. Should it? There is a parallel with exponential spline collocation: same formula for var swap pricing.
    return FukasawaInterpolator(spline,ys,tte,forward)
end

function varianceByLogmoneyness(s::FukasawaInterpolator, y)
    #solve d2(y)
    bracket = searchsorted(s.ys, y)
    # println(y," ",bracket," ",length(s.ys))
    if bracket.start == bracket.stop
        return s.spline.a[bracket.start]
    elseif bracket.stop == 0
        #flat extrapolation
        return s.spline.a[1]
    elseif bracket.stop == length(s.ys)
        return s.spline.a[end]
    else
        obj(z) = z + d2(y,s.spline(z)*s.tte) 
        # dobj(z) = 1 + (y/(2v)- 1/4)/sqrt(v)*evaluateDerivative(s.spline,z)
        # println(obj(s.spline.x[1])," ",s.spline.x[end])
        z0 = find_zero(obj,(s.spline.x[bracket.stop],s.spline.x[bracket.start]),Roots.A42())        
        return s.spline(z0)
    end
end
