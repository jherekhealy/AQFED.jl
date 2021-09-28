#an extension to the Roots package
using Roots
using Setfield
export SuperHalley

struct SuperHalley <: Roots.AbstractHalleyLikeMethod
end

function Roots.update_state(method::SuperHalley, F, o::Roots.HalleyState{T,S}, options::Roots.UnivariateZeroOptions, l=Roots.NullTracks()) where {T,S}
    xn = o.xn1
    fxn = o.fxn1
    r1, r2 = o.Δ, o.ΔΔ

    Δ = (1 + r1/(2r2 - 2r1))*r1
    if Roots.isissue(Δ)
        Roots.log_message(l, "Issue with computing `Δ`")
        return (o, true)
    end

    xn1::T = xn - Δ
    fxn1::S, (r1::T, r2::T) = F(xn1)
    Roots.incfn(l,3)

    @set! o.xn0 = xn
    @set! o.xn1 = xn1
    @set! o.fxn0 = fxn
    @set! o.fxn1 = fxn1
    @set! o.Δ = r1
    @set! o.ΔΔ = r2
    return o, false
end
