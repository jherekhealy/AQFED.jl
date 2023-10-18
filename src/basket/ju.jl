export JuBasketPricer
struct JuBasketPricer
end
function priceEuropean(
    p::JuBasketPricer,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}
)::T where {T,TV}
    n = length(forward)
    y = log(strike)
    sbar = @. forward * weight
    abar = zeros(T, n)
    rhobar = zeros(T, (n, n))
    sqrtVar = sqrt.(totalVariance)
     for j = 1:size(correlation, 2)
         for i = 1:size(correlation, 1)
            rhobar[i, j] = correlation[i, j] * sqrtVar[i] * sqrtVar[j]
        end
    end
    abar = rhobar' * sbar


    u1 = zero(T)
    u2 = zero(T)
    u21 = zero(T)
    up2 = zero(T)
    upp2 = zero(T)
    uppp2 = zero(T)
    eap2app = zero(T)
    eap2app2 = zero(T)
    eap3appp = zero(T)
    eapappappp = zero(T)
    eapp3 = zero(T)
    for (i, si) = enumerate(sbar)
        u1 += si
        eap2app += 2 * si * abar[i]^2
        eap3appp += 6 * si * abar[i]^3
        for (j, sj) = enumerate(sbar)
            temp = si * sj
            u2 += temp
            u21 += temp * exp(rhobar[i, j])
            temp *= rhobar[i,j]
            up2 += temp
            temp *= rhobar[i,j]
            upp2 += temp
            temp *= rhobar[i,j]
            uppp2 += temp
            eap2app2 += abar[i]^2 * si * sj * rhobar[i, j]
            eapappappp += 6 * si * sj * rhobar[i, j]^2 * abar[j]
            for (k, sk) = enumerate(sbar)
                s3 = si * sj * sk
                eapp3 += 8 * s3 * rhobar[i, k] * rhobar[j, k] * rhobar[i, j]
            end
        end
    end
    eap2app2 = 8 * eap2app2 + 2 * up2 * upp2

    a1 = -up2 / (2 * u2)
    a2 = 2 * a1^2 - upp2 / (2 * u2)
    a3 = 6 * a1 * a2 - 4 * a1^3 - uppp2 / (2 * u2)
    b1 = eap2app / (4 * u1^3)
    b2 = a1^2 - 0.5 * a2
    c1 = -a1 * b1
    c2 = (9 * eap2app2 + 4 * eap3appp) / (144 * u1^4)
    c3 = (4 * eapappappp + eapp3) / (48 * u1^3)
    c4 = a1 * a2 - 2 * a1^3 / 3 - a3 / 6
    d2 = 0.5 * (10 * a1^2 + a2 - 6 * b1 + 2 * b2) - (128 * a1^3 / 3 - a3 / 6 + 2 * a1 * b1 - a1 * b2 + 50 * c1 - 11 * c2 + 3 * c3 - c4)
    d3 = 2 * a1^2 - b1 - (88 * a1^3 + 3 * a1 * (5 * b1 - 2 * b2) + 3 * (35 * c1 - 6 * c2 + c3)) / 3
    d4 = -20 * a1^3 / 3 + a1 * (-4 * b1 + b2) - 10 * c1 + c2
    lnU2 = log(u21)
    lnU1 = log(u1)
    m1 = 2 * lnU1 - 0.5 * lnU2
    v1 = max(lnU2 - 2 * lnU1, eps(T))
    sqrtv1 = sqrt(v1)
    y1 = (m1 - y) / sqrtv1 + sqrtv1
    y2 = y1 - sqrtv1
    py = exp(-(y - m1)^2 / (2v1)) / sqrt(2π * v1)
    dpy = -(y - m1) / v1 * py
    d2py = -py / v1 - (y - m1) / v1 * dpy

    z1 = d2 - d3 + d4
    z2 = d3 - d4
    z3 = d4
    priceCall = u1 * normcdf(y1) - strike * normcdf(y2) + strike * (z1 * py + z2 * dpy + z3 * d2py)
    pricePut = strike - u1 + priceCall
    if isCall
        return priceCall * discountFactor
    end
    return pricePut * discountFactor
end