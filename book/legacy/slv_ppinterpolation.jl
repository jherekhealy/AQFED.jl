# slv_ppinterpolation.jl
#
# SLV / Heston-QE implementation using PPInterpolation.
#
# This file is intended to be included in the existing Julia pricing code
# shown in the question. It reuses your existing:
#
#   HestonModel
#   GridConfig{QE}
#   BrownianBridgeConstruction
#   skipTo, next!, norminv
#   _obs_indices, _record_observation!, _store_payoff!, _check_nan
#   _mc_summary, dimensionOut
#   discountFactor, logForward, varianceByLogmoneyness
#
# Package:
#
#   using Pkg
#   Pkg.add("PPInterpolation")
#
# Interpolation policy:
#
#   total variance w(y,T)       -> C2 cubic PP in y
#   w_T                         -> finite difference / linear in TOTAL variance
#   w_y, w_yy                   -> exact derivatives of the PP polynomial
#   local variance              -> C2 cubic PP of log(local variance)
#   E[V | log S]                -> monotone cubic Hermite PP (HuynRational)
#   L²                          -> monotone cubic Hermite PP of log(L²)
#
# There is no linear interpolation in the spot/log-moneyness dimension.
# The only linear operation is the time derivative of TOTAL variance.

using PPInterpolation

const SLVCubicPP64 =
    PPInterpolation.PP{
        3,
        Float64,
        Float64,
        Vector{Float64},
        Matrix{Float64},
        Vector{Float64},
    }


Base.@kwdef struct SLVConfig
    nSimExpectation::Int = 50_000

    # Particle/KDE grid.
    gridSize::Int = 32
    maxGridSize::Int = 128
    bandwidthFactor::Float64 = 1.5
    threshold::Float64 = 1e-4
    nDev::Float64 = 3.0

    # Dupire log-moneyness grid.
    yMin::Float64 = -4.0
    yMax::Float64 = 4.0
    nY::Int = 257

    # varianceByLogmoneyness() in your lv.jl returns Black variance
    # sigma_imp^2. Therefore total variance is T*sigma_imp^2.
    # Set true only if your surface already returns total variance.
    surfaceReturnsTotalVariance::Bool = false

    # Numerical floors.
    totalVarianceFloor::Float64 = 1e-10
    localVarianceFloor::Float64 = 1e-10
    conditionalVarianceFloor::Float64 = 1e-10
    leverageFloor::Float64 = 1e-10
    denominatorFloor::Float64 = 1e-10
    bandwidthFloor::Float64 = 1e-5
end


struct SLVPrepared
    times::Vector{Float64}
    logForward::Vector{Float64}
    yGrid::Vector{Float64}

    # w(y,T) PP.
    totalVariancePP::Vector{SLVCubicPP64}

    # log(local variance) PP.
    localVarianceLogPP::Vector{SLVCubicPP64}

    # E[V | log S] PP.
    conditionalVariancePP::Vector{SLVCubicPP64}

    # log(L²) PP.
    leverageLogSquaredPP::Vector{SLVCubicPP64}

    # Diagnostics.
    localVarianceGrid::Matrix{Float64}
    leverageSquaredGrid::Matrix{Float64}

    config::SLVConfig
end


# ============================================================================
# PPInterpolation helpers
# ============================================================================

@inline function _slv_pp_flat(pp::SLVCubicPP64, x::Float64)
    return PPInterpolation.evaluate(
        pp,
        x;
        extrapolation = PPInterpolation.ConstantAutoExtrapolation(),
    )
end

@inline function _slv_pp_derivative(pp::SLVCubicPP64, x::Float64)
    return PPInterpolation.evaluateDerivative(
        pp,
        x;
        extrapolation = PPInterpolation.ConstantAutoExtrapolation(),
    )
end

@inline function _slv_pp_second_derivative(pp::SLVCubicPP64, x::Float64)
    return PPInterpolation.evaluateSecondDerivative(
        pp,
        x;
        extrapolation = PPInterpolation.ConstantAutoExtrapolation(),
    )
end


function _slv_c2_spline(
    x::Vector{Float64},
    y::Vector{Float64},
)
    length(x) >= 4 ||
        throw(ArgumentError("Cubic PP requires at least four nodes"))

    return PPInterpolation.makeCubicPP(
        x,
        y,
        PPInterpolation.SECOND_DERIVATIVE,
        0.0,
        PPInterpolation.SECOND_DERIVATIVE,
        0.0,
        PPInterpolation.C2(),
    )
end


function _slv_pchip(
    x::Vector{Float64},
    y::Vector{Float64},
)
    length(x) >= 4 ||
        throw(ArgumentError("PCHIP PP requires at least four nodes"))

    return PPInterpolation.makeCubicPP(
        x,
        y,
        PPInterpolation.FIRST_DERIVATIVE,
        0.0,
        PPInterpolation.FIRST_DERIVATIVE,
        0.0,
        PPInterpolation.HuynRational(),
    )
end


# ============================================================================
# Dupire / total variance
# ============================================================================

@inline function _slv_total_variance(
    lvModel,
    y::Float64,
    t::Float64,
    cfg::SLVConfig,
)
    t <= 0.0 && return 0.0

    raw = Float64(
        varianceByLogmoneyness(
            lvModel,
            y,
            t,
        ),
    )

    w =
        cfg.surfaceReturnsTotalVariance ?
        raw :
        t * raw

    return max(
        w,
        cfg.totalVarianceFloor,
    )
end


@inline function _slv_dupire_denominator(
    pp::SLVCubicPP64,
    y::Float64,
    cfg::SLVConfig,
)
    w = max(
        _slv_pp_flat(pp, y),
        cfg.totalVarianceFloor,
    )

    wy = _slv_pp_derivative(pp, y)
    wyy = _slv_pp_second_derivative(pp, y)

    return (
        1.0 -
        y / w * wy +
        0.25 *
        (
            -0.25 -
            1.0 / w +
            y * y / (w * w)
        ) *
        wy * wy +
        0.5 * wyy
    )
end


function _slv_precompute_total_variance(
    lvModel,
    times::Vector{Float64},
    yGrid::Vector{Float64},
    cfg::SLVConfig,
)
    pps = Vector{SLVCubicPP64}(undef, length(times))
    values = Vector{Float64}(undef, length(yGrid))

    for i in eachindex(times)
        t = times[i]

        @inbounds for j in eachindex(yGrid)
            values[j] =
                _slv_total_variance(
                    lvModel,
                    yGrid[j],
                    t,
                    cfg,
                )
        end

        pps[i] =
            _slv_c2_spline(
                yGrid,
                copy(values),
            )
    end

    return pps
end


function _slv_precompute_local_variance(
    totalVariancePP::Vector{SLVCubicPP64},
    times::Vector{Float64},
    yGrid::Vector{Float64},
    cfg::SLVConfig,
)
    nY = length(yGrid)
    nT = length(times)

    localVarianceLogPP =
        Vector{SLVCubicPP64}(undef, nT)

    localVarianceGrid =
        Matrix{Float64}(undef, nY, nT)

    logLv = Vector{Float64}(undef, nY)

    for i in 1:nT
        t1 = times[i]
        t0 = i == 1 ? 0.0 : times[i - 1]
        dt = t1 - t0

        dt > 0.0 ||
            throw(ArgumentError("Times must be strictly increasing"))

        pp1 = totalVariancePP[i]

        @inbounds for j in 1:nY
            y = yGrid[j]

            w1 =
                _slv_pp_flat(
                    pp1,
                    y,
                )

            w0 =
                i == 1 ?
                0.0 :
                _slv_pp_flat(
                    totalVariancePP[i - 1],
                    y,
                )

            # This is the only linear-in-time interpolation/differencing:
            # it is performed on TOTAL VARIANCE, not on volatility.
            wT =
                (w1 - w0) / dt

            denom =
                _slv_dupire_denominator(
                    pp1,
                    y,
                    cfg,
                )

            lv =
                if wT > 0.0 &&
                   denom > cfg.denominatorFloor
                    wT / denom
                else
                    cfg.localVarianceFloor
                end

            if !isfinite(lv) ||
               lv < cfg.localVarianceFloor
                lv = cfg.localVarianceFloor
            end

            localVarianceGrid[j, i] = lv
            logLv[j] = log(lv)
        end

        # Positive local variance is represented by a cubic PP of log(LV).
        localVarianceLogPP[i] =
            _slv_c2_spline(
                yGrid,
                copy(logLv),
            )
    end

    return localVarianceLogPP, localVarianceGrid
end


@inline function _slv_local_variance(
    prepared::SLVPrepared,
    timeIndex::Int,
    y::Float64,
)
    return exp(
        _slv_pp_flat(
            prepared.localVarianceLogPP[timeIndex],
            y,
        ),
    )
end


# ============================================================================
# QE step
# ============================================================================

"""
QE variance transition plus the original Heston QE martingale correction.

The function returns:

    vNew, martingaleCorrection

The SLV leverage is applied only in the log-price update, exactly as in the
Go HestonLVQE implementation.
"""
@inline function _slv_qe_variance_step(
    model::HestonModel,
    v::Float64,
    dt::Float64,
    uV::Float64,
    withMartingale::Bool,
)
    ekdt =
        exp(
            -model.κ * dt,
        )

    m =
        model.θ +
        (v - model.θ) *
        ekdt

    s2 =
        v *
        model.σ^2 *
        ekdt /
        model.κ *
        (1.0 - ekdt) +
        (
            model.θ *
            model.σ^2
        ) /
        (2.0 * model.κ) *
        (1.0 - ekdt)^2

    m = max(m, eps(Float64))

    psi =
        s2 /
        (m * m)

    psiC = 1.5

    vNew = 0.0
    martingaleCorrection = 0.0

    if psi <= psiC
        psiInv =
            1.0 /
            max(
                psi,
                eps(Float64),
            )

        b2 =
            2.0 * psiInv -
            1.0 +
            sqrt(
                max(
                    0.0,
                    2.0 *
                    psiInv *
                    (2.0 * psiInv - 1.0),
                ),
            )

        a =
            m /
            (1.0 + b2)

        b =
            sqrt(
                max(
                    b2,
                    0.0,
                ),
            )

        zv =
            norminv(uV)

        vNew =
            a *
            (b + zv)^2

        if withMartingale
            A =
                dt / 2.0 *
                (
                    model.κ *
                    model.ρ /
                    model.σ -
                    0.5
                ) +
                model.ρ /
                model.σ +
                dt / 4.0 *
                (
                    1.0 -
                    model.ρ^2
                )

            B =
                dt / 2.0 *
                (
                    model.κ *
                    model.ρ /
                    model.σ -
                    0.5
                ) -
                model.ρ /
                model.σ +
                dt / 4.0 *
                (
                    1.0 -
                    model.ρ^2
                )

            C =
                model.ρ /
                model.σ *
                model.κ *
                model.θ *
                dt

            denomA =
                1.0 -
                2.0 *
                A *
                a

            if denomA > eps(Float64)
                martingaleCorrection =
                    C -
                    A *
                    b2 *
                    a /
                    denomA +
                    0.5 *
                    log(denomA) -
                    B *
                    v
            end
        end
    else
        p =
            (psi - 1.0) /
            (psi + 1.0)

        invBeta =
            m /
            (1.0 - p)

        if uV <= p
            vNew = 0.0
        else
            vNew =
                log(
                    (1.0 - p) /
                    max(
                        1.0 - uV,
                        eps(Float64),
                    ),
                ) *
                invBeta
        end

        if withMartingale
            A =
                dt / 2.0 *
                (
                    model.κ *
                    model.ρ /
                    model.σ -
                    0.5
                ) +
                model.ρ /
                model.σ +
                dt / 4.0 *
                (
                    1.0 -
                    model.ρ^2
                )

            B =
                dt / 2.0 *
                (
                    model.κ *
                    model.ρ /
                    model.σ -
                    0.5
                ) -
                model.ρ /
                model.σ +
                dt / 4.0 *
                (
                    1.0 -
                    model.ρ^2
                )

            C =
                model.ρ /
                model.σ *
                model.κ *
                model.θ *
                dt

            arg =
                p +
                (1.0 - p) /
                max(
                    1.0 -
                    invBeta * A,
                    eps(Float64),
                )

            martingaleCorrection =
                C -
                log(
                    max(
                        arg,
                        eps(Float64),
                    ),
                ) -
                B *
                v
        end
    end

    return max(vNew, 0.0), martingaleCorrection
end


# ============================================================================
# Quartic particle KDE
# ============================================================================

@inline function _slv_quartic_kernel(u::Float64)
    a = 1.0 - u * u
    return a <= 0.0 ? 0.0 : (15.0 / 16.0) * a * a
end


@inline function _slv_inverse_quartic_radius(x::Float64)
    b =
        1.0 -
        sqrt(
            (16.0 / 15.0) * x,
        )

    return b <= 0.0 ? 1.0 : sqrt(b)
end


@inline function _slv_lower_bound_from(
    a::Vector{Float64},
    x::Float64,
    lo::Int,
)
    n = length(a)

    lo > n && return n + 1

    left = lo
    right = n + 1

    while left < right
        mid = (left + right) >>> 1

        if mid <= n && a[mid] < x
            left = mid + 1
        else
            right = mid
        end
    end

    return left
end


function _slv_conditional_variance_pp(
    logS::Vector{Float64},
    variance::Vector{Float64},
    time::Float64,
    localVarianceAtATM::Float64,
    cfg::SLVConfig,
)
    n = length(logS)

    n >= 16 ||
        throw(
            ArgumentError(
                "nSimExpectation must be at least 16",
            ),
        )

    # Sort the particle cloud by log spot.
    perm = sortperm(logS)
    xs = logS[perm]
    vs = variance[perm]

    minIndex = n ÷ 1000
    maxIndex = n - 1 - minIndex

    xMin = xs[minIndex + 1]
    xMax = xs[maxIndex + 1]

    # Degenerate cloud fallback.
    if !(xMax > xMin)
        meanV =
            max(
                sum(max(v, 0.0) for v in vs) / n,
                cfg.conditionalVarianceFloor,
            )

        δ = 1e-6

        xNodes =
            [
                xMin - 2δ,
                xMin - δ,
                xMin + δ,
                xMin + 2δ,
            ]

        return _slv_pchip(
            xNodes,
            fill(meanV, 4),
        )
    end

    # Same time-dependent grid-size idea as the Go implementation, but
    # bounded sensibly by maxGridSize.
    nGrid =
        max(
            4,
            min(
                cfg.maxGridSize,
                Int(
                    ceil(
                        max(
                            cfg.gridSize,
                            cfg.gridSize *
                            sqrt(
                                max(
                                    time,
                                    0.0,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    h =
        cfg.bandwidthFactor *
        sqrt(
            max(
                localVarianceAtATM *
                max(time, 0.25),
                cfg.localVarianceFloor,
            ),
        ) *
        n^(-0.2)

    h =
        max(
            h,
            cfg.bandwidthFloor,
        )

    supportRadius =
        _slv_inverse_quartic_radius(
            h * cfg.threshold,
        ) *
        h

    dx =
        (xMax - xMin) /
        (nGrid - 1)

    xNodes =
        Vector{Float64}(
            undef,
            nGrid,
        )

    eNodes =
        Vector{Float64}(
            undef,
            nGrid,
        )

    firstParticle = 1
    effective = 0

    for j in 1:nGrid
        xj =
            xMin +
            dx * (j - 1)

        xLow =
            xj -
            supportRadius

        xHigh =
            xj +
            supportRadius

        firstParticle =
            max(
                firstParticle,
                _slv_lower_bound_from(
                    xs,
                    xLow,
                    firstParticle,
                ),
            )

        lastParticle =
            _slv_lower_bound_from(
                xs,
                xHigh,
                firstParticle,
            ) - 1

        num = 0.0
        denom = 0.0

        if firstParticle <= lastParticle
            @inbounds for k in firstParticle:lastParticle
                u =
                    (xs[k] - xj) /
                    h

                if abs(u) < 1.0
                    kernel =
                        _slv_quartic_kernel(u) /
                        h

                    num += kernel

                    vk = vs[k]

                    if isfinite(vk) && vk > 0.0
                        denom +=
                            kernel *
                            vk
                    end
                end
            end
        end

        if num > 0.0 && denom > 0.0
            effective += 1
            xNodes[effective] = xj
            eNodes[effective] =
                max(
                    denom / num,
                    cfg.conditionalVarianceFloor,
                )
        end
    end

    # If the kernel bandwidth is too small, recover with a flat positive
    # conditional variance rather than introducing linear interpolation.
    if effective < 4
        meanV =
            max(
                sum(max(v, 0.0) for v in vs) / n,
                cfg.conditionalVarianceFloor,
            )

        xNodes =
            collect(
                range(
                    xMin,
                    xMax,
                    length = 4,
                ),
            )

        return _slv_pchip(
            xNodes,
            fill(meanV, 4),
        )
    end

    resize!(xNodes, effective)
    resize!(eNodes, effective)

    return _slv_pchip(
        xNodes,
        eNodes,
    )
end


# ============================================================================
# Sequential particle calibration
# ============================================================================

function _slv_calibrate_conditional_variance!(
    rngCalibration,
    model::HestonModel,
    spot::Float64,
    times::Vector{Float64},
    logForward::Vector{Float64},
    localVarianceLogPP::Vector{SLVCubicPP64},
    cfg::SLVConfig;
    withMartingale::Bool,
)
    nT = length(times)
    nP = cfg.nSimExpectation

    conditionalPP =
        Vector{SLVCubicPP64}(undef, nT)

    logS =
        fill(
            log(spot),
            nP,
        )

    variance =
        fill(
            model.v0,
            nP,
        )

    uniforms =
        zeros(
            Float64,
            2,
            nP,
        )

    rhoBar =
        sqrt(
            1.0 -
            model.ρ^2,
        )

    for i in 1:nT
        t1 = times[i]
        t0 = i == 1 ? 0.0 : times[i - 1]
        dt = t1 - t0
        sqrtDt = sqrt(dt)

        next!(
            rngCalibration,
            uniforms,
        )

        previousConditional =
            i == 1 ?
            nothing :
            conditionalPP[i - 1]

        @inbounds for p in 1:nP
            x = logS[p]
            v = variance[p]

            # At the first interval E[V|S] = V0.
            lvc =
                i == 1 ?
                model.v0 :
                max(
                    _slv_pp_flat(
                        previousConditional::SLVCubicPP64,
                        x,
                    ),
                    cfg.conditionalVarianceFloor,
                )

            y =
                x -
                logForward[i]

            lv =
                exp(
                    _slv_pp_flat(
                        localVarianceLogPP[i],
                        y,
                    ),
                )

            leverageSquared =
                max(
                    lv / lvc,
                    cfg.leverageFloor,
                )

            sqrtLV =
                sqrt(
                    leverageSquared,
                )

            vNew, martingaleCorrection =
                _slv_qe_variance_step(
                    model,
                    v,
                    dt,
                    uniforms[2, p],
                    withMartingale,
                )

            vIntegral =
                0.5 *
                (vNew + v)

            sqrtVIntegral =
                sqrt(
                    max(
                        vIntegral,
                        0.0,
                    ),
                )

            zS =
                norminv(
                    uniforms[1, p],
                )

            # HestonLVQE log-spot update from the supplied Go implementation.
            logS[p] =
                x +
                (
                    logForward[i] -
                    (
                        i == 1 ?
                        log(spot) :
                        logForward[i - 1]
                    )
                ) +
                model.ρ /
                model.σ *
                sqrtLV *
                (
                    vNew -
                    v -
                    model.κ *
                    model.θ *
                    dt
                ) +
                (
                    model.κ *
                    model.ρ /
                    model.σ -
                    0.5 *
                    sqrtLV
                ) *
                sqrtLV *
                vIntegral *
                dt +
                rhoBar *
                sqrtVIntegral *
                sqrtLV *
                zS *
                sqrtDt +
                martingaleCorrection

            variance[p] =
                max(
                    vNew,
                    0.0,
                )
        end

        lvATM =
            exp(
                _slv_pp_flat(
                    localVarianceLogPP[i],
                    0.0,
                ),
            )

        conditionalPP[i] =
            _slv_conditional_variance_pp(
                logS,
                variance,
                t1,
                lvATM,
                cfg,
            )
    end

    return conditionalPP
end


# ============================================================================
# Leverage L² = LV / E[V|S]
# ============================================================================

function _slv_precompute_leverage(
    localVarianceLogPP::Vector{SLVCubicPP64},
    conditionalPP::Vector{SLVCubicPP64},
    times::Vector{Float64},
    logForward::Vector{Float64},
    yGrid::Vector{Float64},
    cfg::SLVConfig,
)
    nT = length(times)
    nY = length(yGrid)

    leverageLogSquaredPP =
        Vector{SLVCubicPP64}(undef, nT)

    leverageSquaredGrid =
        Matrix{Float64}(undef, nY, nT)

    logL2 =
        Vector{Float64}(undef, nY)

    for i in 1:nT
        lvPP =
            localVarianceLogPP[i]

        evPP =
            conditionalPP[i]

        @inbounds for j in 1:nY
            y = yGrid[j]

            lv =
                exp(
                    _slv_pp_flat(
                        lvPP,
                        y,
                    ),
                )

            # S = F * exp(y), so log S = log F + y.
            x =
                logForward[i] +
                y

            ev =
                max(
                    _slv_pp_flat(
                        evPP,
                        x,
                    ),
                    cfg.conditionalVarianceFloor,
                )

            l2 =
                max(
                    lv / ev,
                    cfg.leverageFloor,
                )

            leverageSquaredGrid[j, i] = l2
            logL2[j] = log(l2)
        end

        # Positive leverage through cubic interpolation of log(L²).
        leverageLogSquaredPP[i] =
            _slv_pchip(
                yGrid,
                copy(logL2),
            )
    end

    return (
        leverageLogSquaredPP,
        leverageSquaredGrid,
    )
end


@inline function slvLeverageSquared(
    prepared::SLVPrepared,
    timeIndex::Int,
    y::Float64,
)
    return exp(
        _slv_pp_flat(
            prepared.leverageLogSquaredPP[timeIndex],
            y,
        ),
    )
end


@inline function slvLocalVariance(
    prepared::SLVPrepared,
    timeIndex::Int,
    y::Float64,
)
    return _slv_local_variance(
        prepared,
        timeIndex,
        y,
    )
end


@inline function slvConditionalVariance(
    prepared::SLVPrepared,
    timeIndex::Int,
    logS::Float64,
)
    return _slv_pp_flat(
        prepared.conditionalVariancePP[timeIndex],
        logS,
    )
end


# ============================================================================
# Complete precomputation
# ============================================================================

"""
    prepareSLV(
        rngCalibration,
        model,
        lvModel,
        spot,
        genTimes;
        config = SLVConfig(),
        withMartingale = true,
    )

Precompute the complete SLV leverage surface.

The sequence is:

    implied surface
        -> total variance PP in y
        -> Dupire local variance
        -> particle E[V|S]
        -> L² = LV/E[V|S]
        -> PP(log L²)

The production MC never evaluates Dupire or KDE.
"""
function prepareSLV(
    rngCalibration,
    model::HestonModel,
    lvModel,
    spot::Float64,
    genTimes;
    config::SLVConfig = SLVConfig(),
    withMartingale::Bool = true,
)
    gtimes =
        Float64.(genTimes)

    length(gtimes) >= 2 ||
        throw(
            ArgumentError(
                "genTimes must contain 0 and at least one positive time",
            ),
        )

    gtimes[1] == 0.0 ||
        throw(
            ArgumentError(
                "genTimes[1] must be 0",
            ),
        )

    all(diff(gtimes) .> 0.0) ||
        throw(
            ArgumentError(
                "genTimes must be strictly increasing",
            ),
        )

    config.nY >= 4 ||
        throw(
            ArgumentError(
                "SLVConfig.nY must be at least 4",
            ),
        )

    config.nSimExpectation >= 16 ||
        throw(
            ArgumentError(
                "nSimExpectation must be at least 16",
            ),
        )

    times =
        gtimes[2:end]

    yGrid =
        collect(
            range(
                config.yMin,
                config.yMax,
                length = config.nY,
            ),
        )

    lnspot =
        log(spot)

    # Forward associated with the implied-volatility surface.
    logForward =
        [
            logForward(
                lvModel,
                lnspot,
                t,
            )
            for t in times
        ]

    # 1. Cubic PP in log-moneyness for total variance.
    totalVariancePP =
        _slv_precompute_total_variance(
            lvModel,
            times,
            yGrid,
            config,
        )

    # 2. Dupire local variance.
    localVarianceLogPP,
    localVarianceGrid =
        _slv_precompute_local_variance(
            totalVariancePP,
            times,
            yGrid,
            config,
        )

    # 3. Sequential particle calibration of E[V|S].
    conditionalPP =
        _slv_calibrate_conditional_variance!(
            rngCalibration,
            model,
            spot,
            times,
            logForward,
            localVarianceLogPP,
            config;
            withMartingale = withMartingale,
        )

    # 4. Precompute leverage L² = LV / E[V|S].
    leverageLogSquaredPP,
    leverageSquaredGrid =
        _slv_precompute_leverage(
            localVarianceLogPP,
            conditionalPP,
            times,
            logForward,
            yGrid,
            config,
        )

    return SLVPrepared(
        times,
        logForward,
        yGrid,
        totalVariancePP,
        localVarianceLogPP,
        conditionalPP,
        leverageLogSquaredPP,
        localVarianceGrid,
        leverageSquaredGrid,
        config,
    )
end


# ============================================================================
# Production SLV QE Monte Carlo
# ============================================================================

"""
    simulateIterSLV(
        rng,
        model::HestonModel,
        lvModel,
        spot,
        payoff,
        start,
        nSim,
        config::GridConfig{QE};
        prepared,
        withBB = config.method.withBB,
        withMartingale = config.method.withMartingale,
    )

Production SLV QE simulation.

The only SLV operation inside the MC loop is:

    y  = log(S) - log(F)
    L² = exp(PP(log(L²))(y))

There is no linear spot/moneyness interpolation.
"""
function simulateIterSLV(
    rng,
    model::HestonModel{T},
    lvModel,
    spot::Float64,
    payoff::MCPayoff,
    start::Int,
    nSim::Int,
    config::GridConfig{QE};
    prepared::SLVPrepared,
    withBB = config.method.withBB,
    withMartingale = config.method.withMartingale,
) where {T}
    specTimes =
        config.specTimes

    tte =
        specTimes[end]

    df =
        discountFactor(
            model,
            tte,
        )

    genTimes =
        config.genTimes

    obsIdx =
        _obs_indices(
            genTimes,
            specTimes,
        )

    length(genTimes) - 1 ==
    length(prepared.times) ||
        throw(
            ArgumentError(
                "prepared SLV time grid does not match config.genTimes",
            ),
        )

    local bb

    if withBB
        bb =
            BrownianBridgeConstruction(
                genTimes[2:end],
            )
    end

    skipTo(
        rng,
        start,
    )

    lnspot =
        log(spot)

    rhoBar =
        sqrt(
            1.0 -
            model.ρ^2,
        )

    dimOut =
        dimensionOut(payoff)[1]

    payoffValues =
        zeros(
            T,
            (dimOut, nSim),
        )

    h =
        genTimes[2:end] -
        genTimes[1:end-1]

    nSteps =
        length(h)

    ekdt =
        @. exp(
            -model.κ * h,
        )

    u =
        zeros(
            Float64,
            2,
            nSteps,
        )

    u1 =
        @view u[1, :]

    u2 =
        @view u[2, :]

    z1 =
        zeros(Float64, nSteps)

    obsPath =
        zeros(
            T,
            length(specTimes),
        )

    # Existing Heston forward for the drift term.
    lnf =
        map(
            t -> logForward(
                model,
                lnspot,
                t,
            ),
            genTimes,
        )

    # Forward used by the local-volatility surface.
    # Normally this is identical to lnf.
    slvLogForward =
        vcat(
            lnspot,
            prepared.logForward,
        )

    for sim in 1:nSim
        next!(
            rng,
            u,
        )

        @. u1 =
            norminv(u1)

        if withBB
            transform!(
                bb,
                u1,
                z1,
            )
        else
            @. z1 =
                u1 *
                sqrt(h)
        end

        logpathValue =
            lnspot

        v =
            model.v0

        nextObs =
            _record_observation!(
                obsPath,
                1,
                obsIdx,
                0,
                logpathValue,
            )

        @inbounds for i in 1:nSteps
            dt =
                h[i]

            # QE variance step.
            vNew,
            martingaleCorrection =
                _slv_qe_variance_step(
                    model,
                    v,
                    dt,
                    u2[i],
                    withMartingale,
                )

            # Current log-moneyness relative to the LV forward.
            y =
                logpathValue -
                slvLogForward[i + 1]

            # Single production SLV lookup.
            leverageSquared =
                max(
                    slvLeverageSquared(
                        prepared,
                        i,
                        y,
                    ),
                    prepared.config.leverageFloor,
                )

            sqrtLV =
                sqrt(
                    leverageSquared,
                )

            vIntegral =
                0.5 *
                (
                    vNew +
                    v
                )

            sqrtVIntegral =
                sqrt(
                    max(
                        vIntegral,
                        0.0,
                    ),
                )

            # Same SLV QE log-price update as the supplied Go path generator:
            #
            # X_new = X
            #       + drift
            #       + rho/sigma * sqrt(L²)
            #           * (V_new - V - kappa*theta*dt)
            #       + (kappa*rho/sigma - 0.5*sqrt(L²))
            #           * sqrt(L²) * Vbar * dt
            #       + rhoBar * sqrt(Vbar) * sqrt(L²) * dW
            #       + martingale correction
            logpathValue +=
                lnf[i + 1] -
                lnf[i] +
                model.ρ /
                model.σ *
                sqrtLV *
                (
                    vNew -
                    v -
                    model.κ *
                    model.θ *
                    dt
                ) +
                (
                    model.κ *
                    model.ρ /
                    model.σ -
                    0.5 *
                    sqrtLV
                ) *
                sqrtLV *
                vIntegral *
                dt +
                rhoBar *
                sqrtVIntegral *
                sqrtLV *
                z1[i] +
                martingaleCorrection

            v =
                max(
                    vNew,
                    0.0,
                )

            nextObs =
                _record_observation!(
                    obsPath,
                    nextObs,
                    obsIdx,
                    i,
                    logpathValue,
                )
        end

        _store_payoff!(
            payoffValues,
            sim,
            payoff,
            obsPath,
            df,
        )

        _check_nan(
            payoffValues,
            sim,
            obsPath,
            u,
        )
    end

    return _mc_summary(
        payoffValues,
    )
end
