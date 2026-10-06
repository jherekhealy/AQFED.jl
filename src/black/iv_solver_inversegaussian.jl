#Black implied volatility based on the Inverse Gaussian distribution
#based on Jaehyuk Choi (2016) https://github.com/PyFE/FE-R/blob/main/pkg/R/blackscholes_impvol.R 
using Distributions
export impliedVolatilityIG
const N = Normal()
abstract type InverseGaussianCQuantileMethod end
struct DistributionsCQuantileMethod <: InverseGaussianCQuantileMethod end
struct NaiveStatModCQuantileMethod <: InverseGaussianCQuantileMethod end
struct StatModCQuantileMethod <: InverseGaussianCQuantileMethod end
function impliedVolatilityIG(isCall::Bool,
    price::TP,
    forward::T,
    strike::T,
    timeToExpiry::T,
    df::T; method::InverseGaussianCQuantileMethod = DistributionsCQuantileMethod()
)::TP where {T,TP}
    c, ex = normalizePrice(isCall, price, forward, strike, df)
    if c >= one(T) / ex || c >= one(T)
        throw(DomainError(c, string("price higher than intrinsic value ", 1 / ex)))
    elseif c <= zero(T)
        throw(DomainError(c, "price is negative"))
    end
    #timeval = (c/f - max(cp*(1-strike/f), zero(T)))/min(1, strike/f)
    #timeval = (c - max(1 - ex, zero(T))) / min(ex, 1)
    x = log(ex)
    # we use inverse CDF of inversegaussian distribution
    return if abs(x) < 4 * eps(T)
        #println("ex=",ex," timeval=",c)
        2 * quantile(N, (c + one(T)) / 2) / sqrt(timeToExpiry)
    else
        mu = 2 / abs(x)
        #u = qinvgauss(timeval, mean=mu, lower.tail=F)
        #println("ex=",ex," mu=",mu, ", timeval=",c)
        u = cquantile(method, c, mu)
        2 / sqrt(u * timeToExpiry)
        
    end

end

function cquantile(method::DistributionsCQuantileMethod, c, mu)
    return Distributions.cquantile(InverseGaussian(mu), c)
end
function cquantile(method::NaiveStatModCQuantileMethod, c::T, mu) where {T}
   qinvgauss_upper(c, mu, one(T))
end

function cquantile(method::StatModCQuantileMethod, c::T, mu) where {T}
   return qinvgauss(c, mu, lower_tail=false, log_p=false,                  maxit=400, tol=8*eps(T))
end

# --- log complementary CDF ---
function logccdf_invgauss(x, μ, λ)
    if x <= 0
        return 0.0  # log(1)
    end

    z1 = sqrt(λ / x) * (x / μ - 1)
    z2 = -sqrt(λ / x) * (x / μ + 1)

    a = logccdf(N, z1)              # log Φ(-z1)
    b = 2λ/μ + logcdf(N, z2)        # log( exp(2λ/μ) * Φ(z2) )

    # log(exp(a) - exp(b)) safely
    if b > a
        return -Inf
    end

    return a + log1p(-exp(b - a))
end


"""
Complementary quantile for inverse Gaussian
Find x such that P(X > x) = p

p can be given either as:
- probability (0 < p < 1)
- or log-probability if logp=true
"""
function qinvgauss_upper(p, μ, λ; logp=false, tol=1e-15, maxiter=100)

    # --- normalize input ---
    logp_val = logp ? p : log(p)

    if logp_val >= 0
        return 0.0   # p ≥ 1
    elseif logp_val == -Inf
        return Inf   # p = 0
    end

    # --- initial guess (right tail aware) ---
    # use normal approximation but flipped for upper tail
    z = quantile(N, exp(logp_val))  # small p → large negative z
    z = -z  # flip for upper tail

    x = μ * (1 + (μ * z) / (2λ) +
             sqrt((μ * z / (2λ))^2 + (μ / λ)))

    x = max(x, eps())

    # --- bracketing ---
    x_low = 0.0
    x_high = Inf

    for iter in 1:maxiter
        logF = logccdf_invgauss(x, μ, λ)
        F = exp(logF)

        # Update bracket
        if logF > logp_val
            x_low = x
        else
            x_high = x
        end

        # --- derivative (still in normal space) ---
        fx = sqrt(λ / (2π * x^3)) * exp(-λ * (x - μ)^2 / (2μ^2 * x))

        # Newton step in probability space
        # if fx > 0 && isfinite(F)
        #     step = (F - exp(logp_val)) / fx
        #     x_new = x - step
        # else
        #     x_new = NaN
        # end

        #Log-Newton step
        Δ = logF - logp_val
        ratio = fx / exp(logF)

        x_new = x + Δ / ratio

        # --- safeguard ---
        if !isfinite(x_new) || x_new <= x_low || x_new >= x_high
            if isfinite(x_high)
                x_new = (x_low + x_high) / 2
            else
                x_new = x * 2
            end
        end

        # --- convergence ---
        if abs(x_new - x) < tol * (1 + x)
            return x_new
        end

        x = x_new
    end

    error("qinvgauss_upper did not converge")
end

###### real statmod port below ####
using Distributions
using StatsFuns: normlogcdf, normlogccdf, norminvlogcdf, gammalogcdf, gammalogccdf, gammainvcdf, gammainvccdf, gammainvlogcdf, gammainvlogccdf, chisqlogcdf, chisqlogccdf, chisqinvcdf, chisqinvccdf #not normcdf: conflicts with AQFED.Math

"""
    pinvgauss(q, mean=1.0; shape=nothing, dispersion=1.0, lower_tail=true, log_p=false)
"""
function pinvgauss(q::Real, mean::Real=1.0; shape=nothing, dispersion=1.0, lower_tail::Bool=true, log_p::Bool=false)
    phi = isnothing(shape) ? dispersion : 1.0/shape
    
    if isnan(q) || isnan(mean) || isnan(phi) return NaN end
    if mean <= 0 || phi <= 0 return NaN end
    if q <= 0 return lower_tail ? (log_p ? -Inf : 0.0) : (log_p ? 0.0 : 1.0) end

    # Handle Infinite Mean
    if isinf(mean)
        val = 1.0 / (q * phi)
        # R: pchisq(val, df=1, lower.tail=!lower.tail, log.p=log.p)
        lp = lower_tail ? chisqlogccdf(1.0, val) : chisqlogcdf(1.0, val)
        return log_p ? lp : exp(lp)
    end

    # Standardize
    q_std = q / mean
    phi_std = phi * mean

    # Small CV case
    if phi_std < 1e-14
        lp = lower_tail ? gammalogcdf(1.0/phi_std, phi_std, q_std) : gammalogccdf(1.0/phi_std, phi_std, q_std)
        return log_p ? lp : exp(lp)
    end

    sq_phiq = sqrt(phi_std * q_std)
    
    # a <- pnorm(q-1, sd=sq_phiq, lower.tail=lower.tail, log.p=TRUE)
    log_a = lower_tail ? normlogcdf(0.0, sq_phiq, q_std - 1.0) : normlogccdf(0.0, sq_phiq, q_std - 1.0)
    # b <- 2/phi + pnorm(-q-1, sd=sq_phiq, log.p=TRUE)
    log_b = (2.0 / phi_std) + normlogcdf(0.0, sq_phiq, -q_std - 1.0)

    if lower_tail
        logp = log_a + log1p(exp(log_b - log_a))
    else
        logp = log_a + log1p(-exp(log_b - log_a))
    end

    # Asymptotic right tail
    if !lower_tail && q_std > 1e6 && (q_std / phi_std > 1e6)
        logp = 1.0/phi_std - 0.5*log(π) - log(2.0*phi_std) - 1.5*log(q_std / (2.0*phi_std)) - q_std / (2.0*phi_std)
    end

    return log_p ? logp : exp(logp)
end

"""
    qinvgauss(p, mean=1.0; shape=nothing, dispersion=1.0, lower_tail=true, log_p=false, maxit=100, tol=1e-14)
"""
function qinvgauss(p::Real, mean::Real=1.0; shape=nothing, dispersion=1.0, lower_tail::Bool=true, log_p::Bool=false, maxit::Int=100, tol::Real=1e-14)
    phi = isnothing(shape) ? dispersion : 1.0/shape
    phi_std = phi * mean

    # R code probability logic:
    # logp <- if(log.p) p else log(p)
    logp_orig = log_p ? p : log(p)
    
    # R code logic to create a "lower tail" logp for starting values:
    # if(!lower.tail) logp <- log1p(-exp(logp))
    logp_lower = lower_tail ? logp_orig : log1p(-exp(logp_orig))
    p_lower = exp(logp_lower)
    
    if isnan(p) || isnan(mean) || isnan(phi) return NaN end
    if p_lower < 0 || p_lower > 1 || mean <= 0 || phi <= 0 return NaN end
    if p_lower == 0 return 0.0 end
    if logp_lower == 0
        return Inf 
    end

    # Infinite Mean
    if isinf(mean)
        # 1 / (qchisq(p, df=1, lower.tail=!lower.tail, log.p=log.p) * phi)
        q_chi = lower_tail ? (log_p ? chisqinvccdf(1.0, p) : chisqinvccdf(1.0, exp(p))) : 
                             (log_p ? chisqinvcdf(1.0, p) : chisqinvcdf(1.0, exp(p)))
        return 1.0 / (q_chi * phi)
    end

    # Small phi case (matches R's 1e-08)
    if phi_std < 1e-08
        α = 1.0 / phi_std
        θ = phi_std
        q_gam = lower_tail ? (log_p ? gammainvlogcdf(α, θ, p) : gammainvcdf(α, θ, exp(p))) :
                             (log_p ? gammainvlogccdf(α, θ, p) : gammainvccdf(α, θ, exp(p)))
        return q_gam * mean
    end

    # --- STARTING VALUE x (mean=1 scale) ---
    kappa = 1.5 * phi_std
    x = sqrt(1.0 + kappa^2) - kappa
    if kappa > 1000.0
        x = 0.5/kappa - 0.125/kappa^3
    end

    # R: small <- logp < -11.51 (which is approx log(1e-5))
    if logp_lower < -11.512925464970228 # log(1e-5)
        # x[small] <- 1 / (phi[small] * qnorm(logp[small], log.p=TRUE)^2)
        z = norminvlogcdf(logp_lower)
        x = 1.0 / (phi_std * z^2)
    end
        
    # R: large <- logp > -1e-05 (which is approx log(1 - 1e-5))
    if logp_lower > -1e-05
        # x[large] <- pmax(x[large], qgamma(logp[large], shape=1/phi[large], scale=phi[large], log.p=TRUE))
        α = 1.0 / phi_std
        θ = phi_std
        q_gam_start = gammainvlogcdf(α, θ, logp_lower)
        x = max(x, q_gam_start)
    end

    # --- NEWTON ITERATION ---
    p_target = logp_orig
    p_val = exp(p_target)
    sdx = 0.0

    for iter in 1:maxit
        # logF <- pinvgauss(x, phi=phi, lower.tail=lower.tail, log.p=TRUE)
        logF = pinvgauss(x, 1.0; dispersion=phi_std, lower_tail=lower_tail, log_p=true)       
        dlogp = p_target - logF
        # Faithful R dp calculation (stability check at 1e-5)
        if abs(dlogp) < 1e-05
            dp = exp(p_target + log1p(-dlogp / 2.0)) * dlogp
        else
            dp = p_val - exp(logF)
        end
        
        # log density at mean=1
        logf = 0.5 * (-log(phi_std) - log(2.0*π) - 3.0*log(x) - (z_val = x - 1.0; z_val*z_val / (phi_std * x)))
        dx = dp / exp(logf)
        
        # R's oscillation check
        if iter > 1 && (dx * sdx < 0)
            dx = 0.0
        end
        sdx = sign(dx)
        
        if lower_tail
            x += dx
        else
            x -= dx
        end

        if abs(dx) < tol * x
            break
        end
    end
    return x * mean
end