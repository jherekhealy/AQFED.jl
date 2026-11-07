# Chapter 1: EFFR curve of 6 November 2019, calibrated by Gauss-Newton to the monthly futures and to the OIS par rates, with a natural
# cubic spline on the logarithm of the discount factors. From the script sofr_curve.jl. The figure reads the CSV files of book/data.

module FedFundsCurve
using Dates, LinearAlgebra, Statistics, Printf, PPInterpolation, GaussNewton


# ==============================================================================
# Configuration
# ==============================================================================

const VALUATION_DATE = Date(2019, 11, 6)

const ACT365 = 365.0
const ACT360 = 360.0

yearfrac_act365(a::Date, b::Date) =
    Dates.value(b - a) / ACT365

yearfrac_act360(a::Date, b::Date) =
    Dates.value(b - a) / ACT360

time_from_date(d::Date) =
    yearfrac_act365(VALUATION_DATE, d)

# ==============================================================================
# Instruments
# ==============================================================================

abstract type Instrument end

struct SR1Future <: Instrument
    name::String
    start_date::Date
    end_date::Date
    myquote::Float64       # futures price
end

struct OISSwap <: Instrument
    name::String
    maturity::Date
    payment_dates::Vector{Date}
    accrual_fractions::Vector{Float64}
    myquote::Float64       # decimal par rate
end

# ==============================================================================
# Market data
# ==============================================================================

function build_market_data()
    instruments = Instrument[]

    # Published monthly Fed Funds futures rates from Healy (2020),
    # Appendix A. The paper reports maturity dates and rates.
    #
    # Here the averaging windows are reconstructed for the simplified
    # calendar-day-weighted monthly-average implementation below.

    futures = [
        (
            "Future Z19",
            Date(2019, 12, 1),
            Date(2020, 1, 2),
            0.01560
        ),
        (
            "Future F20",
            Date(2020, 1, 1),
            Date(2020, 2, 3),
            0.01535
        ),
        (
            "Future G20",
            Date(2020, 2, 1),
            Date(2020, 3, 2),
            0.01495
        ),
        (
            "Future H20",
            Date(2020, 3, 1),
            Date(2020, 4, 1),
            0.01475
        ),
        (
            "Future J20",
            Date(2020, 4, 1),
            Date(2020, 5, 1),
            0.01440
        ),
        (
            "Future K20",
            Date(2020, 5, 1),
            Date(2020, 6, 1),
            0.01425
        ),
        (
            "Future M20",
            Date(2020, 6, 1),
            Date(2020, 7, 1),
            0.01405
        ),
        (
            "Future N20",
            Date(2020, 7, 1),
            Date(2020, 8, 3),
            0.01385
        ),
        (
            "Future Q20",
            Date(2020, 8, 1),
            Date(2020, 9, 1),
            0.01370
        ),
        (
            "Future U20",
            Date(2020, 9, 1),
            Date(2020, 10, 1),
            0.01360
        )
    ]

    for (name, start_date, end_date, rate) in futures
        futures_price = 100.0 - 100.0 * rate

        push!(
            instruments,
            SR1Future(
                name,
                start_date,
                end_date,
                futures_price
            )
        )
    end

    # Published OIS par rates from Healy (2020), Appendix A.
    ois_quotes = [
        (Date(2020, 11, 9),  0.01455),
        (Date(2021, 11, 8),  0.01373),
        (Date(2022, 11, 8),  0.01354),
        (Date(2023, 11, 8),  0.01347),
        (Date(2024, 11, 8),  0.01355),
        (Date(2025, 11, 10), 0.01375),
        (Date(2026, 11, 9),  0.01398),
        (Date(2027, 11, 8),  0.01429),
        (Date(2028, 11, 8),  0.01451),
        (Date(2029, 11, 8),  0.01484),
        (Date(2031, 11, 10), 0.01534),
        (Date(2034, 11, 8), 0.01591),
        (Date(2039, 11, 8), 0.01645),
        (Date(2044, 11, 8), 0.01662),
        (Date(2049, 11, 8), 0.01672),
        (Date(2059, 11, 10), 0.01650),
        (Date(2069, 11, 8), 0.01617)
    ]

    # Short-dated OIS instruments.
    push!(
        instruments,
        OISSwap(
            "OIS 1D",
            Date(2019, 11, 7),
            [Date(2019, 11, 7)],
            [1.0 / ACT360],
            0.01560
        )
    )

    push!(
        instruments,
        OISSwap(
            "OIS 2D",
            Date(2019, 11, 8),
            [Date(2019, 11, 8)],
            [2.0 / ACT360],
            0.01560
        )
    )

    for (maturity, myquote) in ois_quotes
        payment_dates = annual_payment_dates(maturity)
        accruals = annual_accruals(payment_dates)

        push!(
            instruments,
            OISSwap(
                "OIS " * string(maturity),
                maturity,
                payment_dates,
                accruals,
                myquote
            )
        )
    end

    return instruments
end

# ==============================================================================
# OIS schedules
# ==============================================================================

function annual_payment_dates(maturity::Date)
    dates = Date[]
    previous = VALUATION_DATE

    while true
        candidate = previous + Year(1)

        if candidate >= maturity
            push!(dates, maturity)
            break
        end

        push!(dates, candidate)
        previous = candidate
    end

    return dates
end

function annual_accruals(payment_dates::Vector{Date})
    accruals = Float64[]
    previous = VALUATION_DATE

    for payment_date in payment_dates
        push!(
            accruals,
            yearfrac_act365(previous, payment_date)
        )
        previous = payment_date
    end

    return accruals
end

# ==============================================================================
# Curve
# ==============================================================================

struct LogDiscountCurve{S}
    pillars::Vector{Float64}
    logdfs::Vector{Float64}
    spline::S
end

function LogDiscountCurve(
    pillars::Vector{Float64},
    logdfs::Vector{Float64}
)
    @assert length(pillars) == length(logdfs)
    @assert length(pillars) >= 2
    @assert iszero(first(pillars))
    @assert iszero(first(logdfs))
    @assert all(diff(pillars) .> 0.0)

    spline = CubicSplineNatural(
        pillars,
        logdfs)

    return LogDiscountCurve(
        pillars,
        logdfs,
        spline
    )
end

function left_slope(curve::LogDiscountCurve)
    x = curve.pillars
    z = curve.logdfs

    return (z[2] - z[1]) / (x[2] - x[1])
end

function right_slope(curve::LogDiscountCurve)
    x = curve.pillars
    z = curve.logdfs

    return (z[end] - z[end - 1]) /
           (x[end] - x[end - 1])
end

function logdf(curve::LogDiscountCurve, t::Real)
    x = curve.pillars
    z = curve.logdfs

    if t <= x[1]
        return z[1] + left_slope(curve) * (t - x[1])
    elseif t >= x[end]
        return z[end] + right_slope(curve) * (t - x[end])
    else
        return evaluate(curve.spline, t)
    end
end

function instantaneous_forward(
    curve::LogDiscountCurve,
    t::Real
)
    x = curve.pillars

    if t <= x[1]
        return -left_slope(curve)
    elseif t >= x[end]
        return -right_slope(curve)
    else
        return -evaluateDerivative(curve.spline, t)
    end
end

discount_factor(curve::LogDiscountCurve, t::Real) =
    exp(logdf(curve, t))

discount_factor(curve::LogDiscountCurve, d::Date) =
    discount_factor(curve, time_from_date(d))

function zero_rate(curve::LogDiscountCurve, t::Real)
    if t <= 0.0
        return instantaneous_forward(curve, 0.0)
    end

    return -logdf(curve, t) / t
end

# ==============================================================================
# Pricing
# ==============================================================================

function model_quote(
    instrument::OISSwap,
    curve::LogDiscountCurve
)
    p_end = discount_factor(
        curve,
        instrument.maturity
    )

    annuity = sum(
        α * discount_factor(curve, d)
        for (α, d) in zip(
            instrument.accrual_fractions,
            instrument.payment_dates
        )
    )

    return (1.0 - p_end) / annuity
end

# Simplified monthly arithmetic-average rate:
# the fixing rate is weighted by the number of calendar days for which
# it applies. Friday carries through Saturday and Sunday.
function monthly_average_rate(
    instrument::SR1Future,
    curve::LogDiscountCurve
)
    dates = collect(
        instrument.start_date:
        Day(1):
        instrument.end_date - Day(1)
    )

    numerator = 0.0
    denominator = 0.0

    for date in dates
        weight = dayofweek(date) == 5 ? 3.0 : 1.0

        rate = instantaneous_forward(
            curve,
            time_from_date(date)
        )

        numerator += weight * rate
        denominator += weight
    end

    return numerator / denominator
end

function model_quote(
    instrument::SR1Future,
    curve::LogDiscountCurve
)
    rate = monthly_average_rate(instrument, curve)

    return 100.0 - 100.0 * rate
end

market_quote(instrument::Instrument) =
    instrument.myquote

# ==============================================================================
# Calibration pillars
# ==============================================================================

# function calibration_pillars(instruments)
#     pillars = Float64[0.0]

#     # Only OIS maturities are spline pillars.
#     # Futures are pricing constraints, not additional knots.
#     for instrument in instruments
#         if instrument isa OISSwap
#             push!(
#                 pillars,
#                 time_from_date(instrument.maturity)
#             )
#         end
#     end

#     return sort(unique(pillars))
# end

function calibration_pillars(instruments)
    pillars = Float64[0.0]

    for instrument in instruments
        if instrument isa OISSwap           

            push!(
                pillars,
                time_from_date(instrument.maturity)
            )
        elseif instrument isa SR1Future          
            push!(
                pillars,
                time_from_date(instrument.end_date)
            )
        end
    end

    return sort(unique(pillars))
end

function initial_logdfs(
    pillars;
    flat_rate = 0.015
)
    return -flat_rate .* pillars
end

# ==============================================================================
# Residuals
# ==============================================================================

function residuals(
    params::Vector{Float64},
    pillars::Vector{Float64},
    instruments::Vector{Instrument};
    futures_weight = 1.0,
    ois_weight = 100.0
)
    logdfs = vcat(0.0, params)
    curve = LogDiscountCurve(pillars, logdfs)

    result = Float64[]

    for instrument in instruments
        model = model_quote(instrument, curve)
        market = market_quote(instrument)

        if instrument isa OISSwap
            # Decimal rate residual scaled by 100.
            # Approximately one unit equals one basis point.
            push!(
                result,
                ois_weight * (model - market)
            )
        elseif instrument isa SR1Future
            # Price points converted to rate units.
            push!(
                result,
                futures_weight * (model - market) / 100.0
            )
        end
    end

    return result
end
# ==============================================================================
# Calibration
# ==============================================================================
function calibrate(
    instruments;
    futures_weight = 1.0,
    ois_weight = 100.0,
    maxiter = 2_000,
    tolerance = 1e-12,
    verbose = true
)
    pillars = calibration_pillars(instruments)

    initial = initial_logdfs(
        pillars;
        flat_rate = 0.015
    )

    x0 = collect(Float64, initial[2:end])

    residual_function = params -> residuals(
        params,
        pillars,
        instruments;
        futures_weight = futures_weight,
        ois_weight = ois_weight
    )

    result = GaussNewton.optimize(
        residual_function,
        x0
    )

    params = x0 # minimizer is in x0.

    @assert length(params) == length(pillars) - 1

    curve = LogDiscountCurve(
        pillars,
        vcat(0.0, params)
    )

    return curve, result
end
# ==============================================================================
# Diagnostics
# ==============================================================================

function objective_value(
    curve::LogDiscountCurve,
    instruments
)
    result = Float64[]

    for instrument in instruments
        model = model_quote(instrument, curve)
        market = market_quote(instrument)

        if instrument isa OISSwap
            push!(result, 100.0 * (model - market))
        elseif instrument isa SR1Future
            push!(result, (model - market) / 100.0)
        end
    end

    return dot(result, result)
end

function print_calibration_report(
    curve,
    instruments,
    result
)
    println("="^90)
    println("CALIBRATED CURVE")
    println("="^90)
    println("Result type: ", typeof(result))
    println("Residual objective: ", objective_value(curve, instruments))
    println()

    @printf(
        "%-20s %14s %14s %14s\n",
        "Instrument",
        "Market",
        "Model",
        "Error"
    )

    println("-"^90)

    for instrument in instruments
        market = market_quote(instrument)
        model = model_quote(instrument, curve)

        if instrument isa OISSwap
            @printf(
                "%-20s %13.8f%% %13.8f%% %10.3f bp\n",
                instrument.name,
                100.0 * market,
                100.0 * model,
                10_000.0 * (model - market)
            )
        else
            @printf(
                "%-20s %14.8f %14.8f %10.3f bp\n",
                instrument.name,
                market,
                model,
                100.0 * (model - market)
            )
        end
    end

    grid = range(0.001, 50.0; length = 20_000)

    forwards = [
        instantaneous_forward(curve, t)
        for t in grid
    ]

    dfs = [
        discount_factor(curve, t)
        for t in grid
    ]

    println()
    println("Curve diagnostics")

    @printf(
        "Minimum instantaneous forward: %.8f%%\n",
        100.0 * minimum(forwards)
    )

    @printf(
        "Maximum instantaneous forward: %.8f%%\n",
        100.0 * maximum(forwards)
    )

    @printf(
        "Minimum discount factor: %.10f\n",
        minimum(dfs)
    )

    @printf(
        "Maximum discount factor: %.10f\n",
        maximum(dfs)
    )
end

# ==============================================================================
# Plot
# ==============================================================================
end

@compute "fedfunds_curve_gauss_newton" begin
    F = FedFundsCurve
    instruments = F.build_market_data()
    curve, result = F.calibrate(instruments; futures_weight=1.0, ois_weight=100.0, maxiter=2_000, tolerance=1e-12, verbose=false)
    F.print_calibration_report(curve, instruments, result)
    positive = curve.pillars[curve.pillars.>0.0]
    grid = collect(range(isempty(positive) ? 1.0 / 365.0 : minimum(positive), maximum(curve.pillars); length=3_000))
    writetable("fedfunds_curve.csv", DataFrame(t=grid, zero=[100.0 * F.zero_rate(curve, t) for t in grid], forward=[100.0 * F.instantaneous_forward(curve, t) for t in grid]))
    ois = filter(i -> i isa F.OISSwap, instruments)
    writetable("fedfunds_ois_quotes.csv", DataFrame(t=[F.time_from_date(i.maturity) for i in ois], quote_percent=[100.0 * i.myquote for i in ois]))
end

@figure "fedfunds_curve_gauss_newton" begin
    curve, quotes = readcsv("fedfunds_curve.csv"), readcsv("fedfunds_ois_quotes.csv")
    fig = bookFigure(size=(1000, 560))
    ax = bookAxis(fig[1, 1], xlabel="Maturity (years)", ylabel="Rate (%)")
    lines!(ax, curve.t, curve.zero, color=PALETTE[1], label="Continuous zero rate")
    lines!(ax, curve.t, curve.forward, color=PALETTE[2], linestyle=:dash, label="Instantaneous forward rate")
    referenceScatter!(ax, quotes.t, quotes.quote_percent, label="OIS market quotes", dense=false)
    Legend(fig[2, 1], ax, orientation=:horizontal, framevisible=false)
    fig
end
