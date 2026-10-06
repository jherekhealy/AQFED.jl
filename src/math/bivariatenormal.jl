export bivariateNormcdf

# Gauss-Legendre half nodes (negative abscissae) and weights for n = 6, 12, 20 as in Genz (2004).
const genzBVNX = (
    (-0.9324695142031522, -0.6612093864662647, -0.2386191860831970),
    (-0.9815606342467191, -0.9041172563704750, -0.7699026741943050, -0.5873179542866171, -0.3678314989981802, -0.1252334085114692),
    (-0.9931285991850949, -0.9639719272779138, -0.9122344282513259, -0.8391169718222188, -0.7463319064601508,
        -0.6360536807265150, -0.5108670019508271, -0.3737060887154196, -0.2277858511416451, -0.07652652113349733))
const genzBVNW = (
    (0.1713244923791705, 0.3607615730481384, 0.4679139345726904),
    (0.04717533638651177, 0.1069393259953183, 0.1600783285433464, 0.2031674267230659, 0.2334925365383547, 0.2491470458134029),
    (0.01761400713915212, 0.04060142980038694, 0.06267204833410906, 0.08327674157670475, 0.1019301198172404,
        0.1181945319615184, 0.1316886384491766, 0.1420961093183821, 0.1491729864726037, 0.1527533871307259))

#bivariateNormcdf(a, b, ρ) = P(X ≤ a, Y ≤ b) for a standard bivariate normal of correlation ρ.
#Port of BVND from A. Genz, "Numerical computation of rectangular bivariate and trivariate normal and t probabilities" (2004). Accuracy ~1e-15.
bivariateNormcdf(a::Real, b::Real, ρ::Real) = genzBVND(-a, -b, ρ)

#P(X > h, Y > k)
function genzBVND(h::T, k::T, r::T) where {T<:AbstractFloat}
    #literal indices keep the quadrature tuples type-stable.
    absr = abs(r)
    if absr < 0.3
        return genzBVND(h, k, r, genzBVNX[1], genzBVNW[1])
    elseif absr < 0.75
        return genzBVND(h, k, r, genzBVNX[2], genzBVNW[2])
    else
        return genzBVND(h, k, r, genzBVNX[3], genzBVNW[3])
    end
end

function genzBVND(h::T, k::T, r::T, xg::NTuple{N,Float64}, wg::NTuple{N,Float64}) where {T<:AbstractFloat,N}
    twoPi = 2 * T(pi)
    absr = abs(r)
    hk = h * k
    bvn = zero(T)
    if absr < 0.925
        hs = (h * h + k * k) / 2
        asr = asin(r)
        for i = eachindex(xg)
            sn = sin(asr * (xg[i] + 1) / 2)
            bvn += wg[i] * exp((sn * hk - hs) / (1 - sn * sn))
            sn = sin(asr * (-xg[i] + 1) / 2)
            bvn += wg[i] * exp((sn * hk - hs) / (1 - sn * sn))
        end
        return bvn * asr / (2 * twoPi) + normcdf(-h) * normcdf(-k)
    end
    if r < 0
        k = -k
        hk = -hk
    end
    if absr < 1
        as = (1 - r) * (1 + r)
        a = sqrt(as)
        bs = (h - k)^2
        c = (4 - hk) / 8
        d = (12 - hk) / 16
        bvn = a * exp(-(bs / as + hk) / 2) * (1 - c * (bs - as) * (1 - d * bs / 5) / 3 + c * d * as * as / 5)
        if hk > -160
            b = sqrt(bs)
            bvn -= exp(-hk / 2) * sqrt(twoPi) * normcdf(-b / a) * b * (1 - c * bs * (1 - d * bs / 5) / 3)
        end
        a /= 2
        for i = eachindex(xg)
            xs = (a * (xg[i] + 1))^2
            rs = sqrt(1 - xs)
            bvn += a * wg[i] * (exp(-bs / (2 * xs) - hk / (1 + rs)) / rs - exp(-(bs / xs + hk) / 2) * (1 + c * xs * (1 + d * xs)))
            xs = as * (-xg[i] + 1)^2 / 4
            rs = sqrt(1 - xs)
            bvn += a * wg[i] * exp(-(bs / xs + hk) / 2) * (exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs - (1 + c * xs * (1 + d * xs)))
        end
        bvn = -bvn / twoPi
    end
    if r > 0
        return bvn + normcdf(-max(h, k))
    else
        return -bvn + max(zero(T), normcdf(-h) - normcdf(-k))
    end
end

genzBVND(h::Real, k::Real, r::Real) = genzBVND(promote(float(h), float(k), float(r))...)

export BivariateNormcdf

#Bivariate normal cumulative distribution for a fixed correlation ρ: for |ρ| < 0.925 the Gauss-Legendre nodes
#sin(asin(ρ)(1±x_i)/2) of Genz (2004) do not depend on the arguments and are precomputed (with 20 points).
#Use as M = BivariateNormcdf(ρ); M(a, b) == bivariateNormcdf(a, b, ρ).
struct BivariateNormcdf
    ρ::Float64
    isTabulated::Bool
    sn::NTuple{20,Float64}
    invOneMinusSn2::NTuple{20,Float64}
    w::NTuple{20,Float64}
end

function BivariateNormcdf(ρ::Float64)
    isTabulated = abs(ρ) < 0.925
    asr = asin(ρ)
    xg, wg = genzBVNX[3], genzBVNW[3]
    sn = ntuple(i -> i <= 10 ? sin(asr * (xg[i] + 1) / 2) : sin(asr * (-xg[i-10] + 1) / 2), 20)
    invOneMinusSn2 = map(s -> 1 / (1 - s * s), sn)
    w = ntuple(i -> wg[mod1(i, 10)] * asr / (4 * pi), 20)
    return BivariateNormcdf(ρ, isTabulated, sn, invOneMinusSn2, w)
end

function (m::BivariateNormcdf)(a::Float64, b::Float64)
    if !m.isTabulated
        return bivariateNormcdf(a, b, m.ρ)
    end
    hk = a * b
    hs = (a * a + b * b) / 2
    bvn = 0.0
    for i = 1:20
        bvn += m.w[i] * exp((m.sn[i] * hk - hs) * m.invOneMinusSn2[i])
    end
    return bvn + normcdf(a) * normcdf(b)
end
