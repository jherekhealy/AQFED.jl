using AQFED.Math
import Roots: find_zero, Newton, A42

struct QuadBasketPricer
    q::Quadrature
end


function priceEuropean(
    p::QuadBasketPricer,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV};
    ndev=10.0, isSplit=false, isLognormal=false
)::T where {T,TV}
    if length(spot) != 2
        throw(DomainError(length(spot), "The quad method is only implemented for baskets of 2 assets."))
    end
    s0 = spot[1]
    s1 = spot[2]
    w0 = weight[1]
    w1 = weight[2]
    f0 = forward[1]
    f1 = forward[2]
    var0 = totalVariance[1]
    var1 = totalVariance[2]
    rho = correlation[1, 2]
    mu0 = -var0 / 2
    sig0 = sqrt(var0)
    mu1 = -var1 / 2
    sig1 = sqrt(var1)

    q1Value = if isSplit
        integrand0 = function (z0::T) where {T}
            es0 = w0 * f0 * exp(z0 + mu0)
            z1k = -(10 + mu1) * sig1
            if es0 < strike
                z1k = log((strike - es0) / (w1 * f1)) - mu1
            end
            pdf2 = normpdf(z0, zero(T), sig0)
            i0 = (es0 - strike) * (1 - normcdf(z1k, rho * sig1 / sig0 * (z0), sig1 * sqrt(1 - rho^2))) * pdf2
            i1 = w1 * f1 * (1 - normcdf(z1k, rho * sig1 / sig0 * (z0) + (1 - rho^2) * var1, sig1 * sqrt(1 - rho^2))) * pdf2
            efactor = exp((1 - rho^2) * var1 / 2 + mu1 + (rho * sig1 / sig0 * (z0)))
            i1 *= efactor
            return i0+i1
        end
        #z such that z1k = rho * sig1 / sig0 * (z0) <=> log((strike - w0 * f0 * exp(z0+mu0)) / (w1 * f1)) - mu1 = rho * sig1 / sig0 * (z0)
        #                                                    strike - w0 * f0 * exp(z0+mu0)) = (w1 * f1)*exp(mu1 + rho * sig1 / sig0 * (z0))
        split0 = function (z0)
            return strike - w0 * f0 * exp(z0 + mu0) - (w1 * f1) * exp(mu1 + rho * sig1 / sig0 * (z0))
        end
        split1 = function (z0)
            return strike - w0 * f0 * exp(z0 + mu0) - (w1 * f1) * exp(mu1 + rho * sig1 / sig0 * (z0) + (1 - rho^2) * var1)
        end
        zd0 =try
             find_zero(split0, (-ndev * sig0, ndev * sig0), A42())
        catch
            zero(T)
        end        
        zd1 = try
            find_zero(split1, (-ndev * sig0, ndev * sig0), A42())      
        catch
            zero(T)
        end   
        if abs(zd0-zd1) > sqrt(eps(T))
         zds =    sort([-ndev*sig0,zd0,zd1,ndev*sig0])
         sum = integrate(p.q, integrand0, zds[1], zds[2]) + integrate(p.q, integrand0, zds[2], zds[3])
         sum + integrate(p.q, integrand0, zds[3], zds[4])
        else 
           zds =  [-ndev*sig0,zd0,ndev*sig0]
           integrate(p.q, integrand0, zds[1], zds[2]) + integrate(p.q, integrand0, zds[2], zds[3])
        end
        # sum + integrate(p.q, integrand1, -ndev * sig0,ndev*sig0)
    else
        integrand = function (z0::T) where {T}
            es0 = w0 * f0 * exp(z0 + mu0)
            z1k = -(10 + mu1) * sig1
            if es0 < strike
                z1k = log((strike - es0) / (w1 * f1)) - mu1
            end
            pdf2 = normpdf(z0, zero(T), sig0)
            i0 = (es0 - strike) * (1 - normcdf(z1k, rho * sig1 / sig0 * (z0), sig1 * sqrt(1 - rho^2))) * pdf2
            i1 = w1 * f1 * (1 - normcdf(z1k, rho * sig1 / sig0 * (z0) + (1 - rho^2) * var1, sig1 * sqrt(1 - rho^2))) * pdf2
            efactor = exp((1 - rho^2) * var1 / 2 + mu1 + (rho * sig1 / sig0 * (z0)))
            i1 *= efactor
            value = i0 + i1
            return value
        end
        if isLognormal
            integrandLN = function (x::T) where {T}
                z = log(x)
                return integrand(z) / x
            end
            integrate(p.q, integrandLN, exp(-ndev * sig0), exp(ndev * sig0))
        else
            integrate(p.q, integrand, -ndev * sig0, ndev * sig0)
        end
    end
    if !isCall
        q1Value = q1Value + strike - (w0 * f0 + w1 * f1)
    end
    q1Value *= discountFactor

    return q1Value
end
