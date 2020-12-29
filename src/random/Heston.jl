using Nemo

export HestonParams, evaluateCharFunc, evaluateLogCharFunc, computeCumulants

struct HestonParams{T}
    v0::T
    κ::T
    θ::T
    ρ::T
    σ::T
end

#CC the Nemo arbField or Complex type. z: Nemo complex number or Complex number.
function evaluateCharFunc(CC, p::HestonParams, z::CT, τ::T) where {T,CT}
    return exp(evaluateLogCharFunc(CC, p, z, τ))
end

function evaluateLogCharFunc(CC, p::HestonParams, z::CT, τ::T) where {T,CT}
    v0 = p.v0
    κ = p.κ
    θ = p.θ
    ρ = p.ρ
    σ = p.σ
    cc1 = CC(0, 1)
    α = -(z * (z + cc1)) * σ^2
    β = κ - cc1 * σ * ρ * z
    D = sqrt(β^2 - α)
    G = (β - D) / (β + D)
    eDT = exp(-D * τ)
    l = log((1 - G * eDT) / (1 - G))
    A = ((β - D) * τ - 2 * l) * κ * θ / σ^2
    B = (1 - eDT) / (1 - G * eDT) * (β - D) / σ^2
    return A + B * v0
end

function computeCumulants(p::HestonParams{T}, τ::T) where {T}
    lambda = p.κ
    ubar = p.θ
    u0 = p.v0
    eta = p.σ
    rho = p.ρ
    term = τ
    if lambda == 0 && eta == 0
        c1 = -u0 * term / 2
        c2 = u0 * term
        c4 = 0
        return c1, c2, c4
    elseif lambda == 0
        c1 = -u0 * term / 2
        c2 = u0 * term * (1 + term * term * eta * (eta * term / 12 - rho / 2))
        c4 =
            u0 *
            eta^2 *
            term^3 *
            (
                2 * rho^2 - 2 * eta * term * rho + eta^4 * term^4 * 17 / 1680 + eta^4 * term^5 * rho^2 * 11 / 20 -
                eta * term * rho^3 / 2 + eta^2 * term^2 * 3 / 10 - eta^3 * term^3 * rho * 17 / 120 + 1
            )
        return c1, c2, c4
    end
    elt = exp(-lambda * term)
    c1 = (1 - elt) * (ubar - u0) / (2 * lambda) - ubar * term / 2
    lambda2 = lambda * lambda
    lambda3 = lambda * lambda2
    eta2 = eta * eta
    elt2 = elt * elt
    # correct c2
    c2 =
        1 / (8 * lambda3) * (
            2 *
            u0 *
            (
                lambda2 * 4 * (elt * term * rho * eta + 1 - elt) +
                lambda * (4 * eta * rho * (elt - 1) - 2 * elt * term * eta2) +
                eta2 * (1 - elt2)
            ) +
            ubar * (
                8 * lambda3 * term +
                lambda2 * 8 * (-eta * rho * term * (1 + elt) + elt - 1) +
                lambda * (2 * eta2 * term * (1 + 2 * elt) + eta * rho * 16 * (1 - elt)) +
                eta2 * (-5 + 4 * elt + elt2)
            )
        )
    # paper c2
    # c2 = 1 / (8 * lambda3) * (eta*term*lambda*elt*(u0-ubar)*(8*lambda*rho-4*eta) + lambda*rho*eta*(1-elt)*(16*ubar-8*u0) + 2*ubar*lambda*term*(-4*lambda*rho*eta+eta2+4*lambda2) + eta2*((ubar-2*u0)*elt2+ubar*(6*elt-7)+2*u0) + 8*lambda2*(u0-ubar)*(1-elt))
    lambda4 = lambda2 * lambda2
    lambda5 = lambda4 * lambda
    lambda6 = lambda4 * lambda2
    lambda7 = lambda3 * lambda4
    eta3 = eta * eta2
    term2 = term * term
    term3 = term2 * term
    rho2 = rho * rho
    eta4 = eta2 * eta2
    elt3 = elt2 * elt
    elt4 = elt2 * elt2
    rho3 = rho2 * rho
    c4 =
        -1 / lambda7 *
        2 *
        ubar *
        eta2 *
        (
            (
                (term3 * rho3 * eta - 3 * term2 * rho2) * lambda6 -
                3 * term * (term2 * rho2 * eta2 - 4 * term * rho * (rho2 + 1) * eta + 8 * rho2 + 2) * lambda5 / 2 +
                (
                    3 * term3 * rho * eta3 / 4 - 21 * term2 * (rho2 + 3 / 14) * eta2 / 2 +
                    (18 * term * rho3 + 24 * term * rho) * eta - 18 * rho2 - 3
                ) * lambda4 -
                (
                    term3 * eta3 - 42 * term2 * rho * eta2 + (240 * term * rho2 + 54 * term) * eta - 192 * rho3 -
                    192 * rho
                ) *
                eta *
                lambda3 / 8 - 3 * eta2 * (term2 * eta2 - 35 * term / 2 * rho * eta + 40 * rho2 + 15 / 2) * lambda2 / 4 -
                27 * eta3 * (term * eta - 20 * rho / 3) * lambda / 16 - 21 * eta4 / 16
            ) * elt +
            (
                (-3 / 4 + 3 * term * rho * eta - 3 * term2 * rho2 * eta2 / 2) * lambda4 +
                3 * eta * (term2 * rho * eta2 + (-4 * term * rho2 - 3 * term / 2) * eta + 4 * rho) * lambda3 / 2 -
                3 * eta2 * (term2 * eta2 - 14 * term * rho * eta + 20 * rho2 + 6) * lambda2 / 8 +
                (-15 / 16 * term * eta4 + 9 * eta3 * rho / 2) * lambda - 21 * eta4 / 32
            ) * elt2 +
            3 *
            eta2 *
            ((term * rho * eta - 1) * lambda2 + (-term / 2 * eta2 + 2 * rho * eta) * lambda - eta2 / 2) *
            elt3 / 8 - 3 * eta4 * elt4 / 128 +
            (-6 * term * rho2 - 3 * term / 2) * lambda5 +
            ((6 * term * rho3 + 9 * term * rho) * eta + 18 * rho2 + 15 / 4) * lambda4 -
            9 * eta * ((rho2 + 0.25) * term * eta + 8 * rho3 / 3 + 10 * rho / 3) * lambda3 +
            15 * eta2 * (term * rho * eta + 10 * rho2 + 11 / 5) * lambda2 / 4 +
            (-33 / 2 * eta3 * rho - 15 / 32 * term * eta4) * lambda +
            279 * eta4 / 128
        )
    c4 +=
        u0 / lambda7 * (
            2 *
            eta2 *
            (
                (
                    (term3 * rho3 * eta - 3 * term2 * rho2) * lambda6 -
                    3 * term * (term2 * rho2 * eta2 - 2 * term * rho * (rho2 + 2) * eta + 4 * rho2 + 2) * lambda5 / 2 +
                    (
                        3 * term3 * rho * eta3 / 4 - 6 * (rho2 + 3 / 8) * term2 * eta2 +
                        6 * term * rho * (rho2 + 2) * eta - 6 * rho2
                    ) * lambda4 -
                    eta *
                    (term3 * eta3 - 24 * term2 * rho * eta2 + (72 * term * rho2 + 18 * term) * eta - 48 * rho3) *
                    lambda3 / 8 - 3 * eta2 * (term2 * eta2 - 7 * term * rho * eta - 3) * lambda2 / 8 -
                    3 * eta3 * (term * eta + 10 * rho) * lambda / 16 + 3 * eta4 / 8
                ) * elt +
                (
                    (6 * term * rho * eta - 3 * term2 * rho2 * eta2 - 3 / 2) * lambda4 +
                    3 * (term2 * rho * eta2 + (-3 * term * rho2 - 3 * term / 2) * eta + 3 * rho) * eta * lambda3 -
                    3 * eta2 * (term2 * eta2 - 10 * term * rho * eta + 12 * rho2 + 3) * lambda2 / 4 -
                    9 * eta3 * (term * eta - 10 * rho / 3) * lambda / 8 - 3 * eta4 / 8
                ) * elt2 +
                9 *
                eta2 *
                ((term * rho * eta - 1) * lambda2 + (-term / 2 * eta2 + 5 / 3 * rho * eta) * lambda - eta2 / 3) *
                elt3 / 8 - 3 * eta4 * elt4 / 32 -
                6 *
                ((rho2 + 1 / 4) * lambda2 - 5 * lambda * rho * eta / 4 + 5 * eta2 / 16) *
                (lambda * rho * eta - eta2 / 4 - lambda2)
            )
        )
    return c1, c2, c4
end
