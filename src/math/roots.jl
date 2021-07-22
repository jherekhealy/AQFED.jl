#an extension to the Roots package
using Roots
export InverseQuadraticMethod

struct InverseQuadraticMethod <: Roots.AbstractHalleyLikeMethod
end

function Roots.update_state(method::InverseQuadraticMethod, fs, o::Roots.UnivariateZeroState{T,S}, options::Roots.UnivariateZeroOptions) where {T,S}
    xn = o.xn1
    fxn = o.fxn1
    r1, r2 = o.m

    xn1::T = xn - (1+ r1/(r2*2))*r1   #r1/r2 = L  1/(2- r1)*r1

    tmp = Roots.fΔxΔΔx(fs, xn1)
    fxn1::S, r1::T, r2::T = tmp[1], tmp[2], tmp[3]
    Roots.incfn(o,3)

    o.xn0, o.xn1 = xn, xn1
    o.fxn0, o.fxn1 = fxn, fxn1
    empty!(o.m); append!(o.m, (r1, r2))
end
