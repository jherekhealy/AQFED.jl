# Chapter 9: stochastic volatility. Fourier integrands and COS errors, Heston discretization schemes,
# quasi Monte-Carlo convergence, stochastic local volatility calibration.

@figure "integrand_heston_angle0" begin
    fig = bookFigure(size=(800, 400))
    ax = bookAxis(fig[1, 1], xlabel="u", ylabel="Integrand", xscale=log10)
    for (file, label) in (("integrand_heston_angle0.txt", "0"), ("integrand_heston_angle1.txt", "π/12"))
        df = readtable(file)
        idx = sortperm(df.u)
        lines!(ax, df.u[idx], df.f[idx], label=label)
    end
    limits!(ax, 0.01, 1e6, -1, 2)
    bookLegend!(ax, title="Angle shift")
    fig
end

@figure "kahale_closed_8" begin
    df = readtable("kahale_closed_8.txt")
    fig = bookFigure(size=(1000, 800))
    panels = [("κ", df.Kappa, identity), ("σ", df.Sigma, identity), ("√θ", sqrt.(df.Theta), identity), ("√V₀", sqrt.(df.V0), identity), ("RMSE", df.RMSE, log10)]
    axes = [bookAxis(fig[i, 1], ylabel=label, xlabel=i == 5 ? "Iterations" : "", yscale=scale) for (i, (label, _, scale)) in enumerate(panels)]
    for (i, (_, y, _)) in enumerate(panels)
        lines!(axes[i], df.Iteration, y, color=PALETTE[i])
        i < 5 && hidexdecorations!(axes[i], grid=false)
    end
    linkxaxes!(axes...)
    rowgap!(fig.layout, 4)
    fig
end

#absolute error of the COS method on a grid of expiries (rows) and strikes (columns)
function cosErrorSurface(file; columns=12)
    m = readmatrix(file)[:, 1:columns]
    expiries = [0.057534247, 0.153424658, 0.230136986, 0.479452055, 0.728767123, 1.22739726, 1.726027397, 2.243835616, 2.742465753, 3.24109589, 4.238356164]
    strikes = [1008, 1209.6, 1411.2, 1612.8, 1814.4, 1915.2, 2016, 2116.8, 2217.6, 2419.2, 2620.8, 3024]
    fig = bookFigure(size=(600, 600))
    surface3D!(fig[1, 1], expiries, strikes, abs.(m), xlabel="Expiry", ylabel="Strike", zlabel="Error", azimuth=1.25π, elevation=0.2π, zrange=(0, 0.02))
    return fig
end
@figure "heston_cos3d" cosErrorSurface("heston_cos3d.txt")
@figure "bates_cos3d" cosErrorSurface("bates_cos3d.txt")
@figure "double_heston_cos3d" cosErrorSurface("double_heston_cos3d.txt")
@figure "schobelzhu_cos3d" cosErrorSurface("schobelzhu_cos3d.txt")

## Heston discretization schemes: price against the time-step with a 95% confidence band

const DT_TICKS = ([1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1], ["1/32", "1/16", "1/8", "1/4", "1/2", "1"])
const SCHEME_COLORS = Dict("Exact" => :black, "QE" => PALETTE[1], "QELong" => PALETTE[2], "DVSS2" => PALETTE[2], "DVSS2X" => PALETTE[2], "Alfonsi2" => PALETTE[3])

#the exact price is dashed so that it stays visible under a converged scheme, markers show the time steps computed
function schemeLines!(ax, d, methods)
    for (i, m) in enumerate(methods)
        dm = sort(d[d.Method.==m, :], :dt)
        isempty(dm) && continue
        c = get(SCHEME_COLORS, m, PALETTE[i])
        m == "Exact" ? lines!(ax, dm.dt, dm.Price, color=c, linestyle=:dash, label=m) :
        scatterlines!(ax, dm.dt, dm.Price, color=c, markersize=7, label=m)
    end
end

const DT_LABEL = rich("Time-step size ", rich("Δt", font=:italic))

#price against the time-step size for the strikes 70, 100, 140 stacked, the strike as panel title and the legend on top
function schemePriceFigure(file, methods; maxdt=0.25, size=(300, 600))
    df = readtable(file)
    df = rows(df[df.dt.<=maxdt, :], :Method, methods...)
    fig = bookFigure(size=size)
    axes = Axis[]
    for (i, strike) in enumerate((70, 100, 140))
        d = df[df.Strike.==strike, :]
        ax = bookAxis(fig[i+1, 1], xlabel=i == 3 ? DT_LABEL : "", xscale=log10, xticks=DT_TICKS,
            title=rich(rich("K", font=:italic), " = $strike"), titlealign=:left, titlegap=2)
        groupBands!(ax, d.dt, d.Price, 1.96 .* d.StdErr, d.Method, order=collect(methods), colors=SCHEME_COLORS)
        schemeLines!(ax, d, methods)
        i < 3 && hidexdecorations!(ax, grid=false)
        push!(axes, ax)
    end
    linkxaxes!(axes...)
    #one row of legend entries when they fit in the width, two otherwise
    ncols = length(methods) * 85 > size[1] ? 2 : length(methods)
    Legend(fig[1, 1], axes[1], orientation=:vertical, nbanks=ncols,
        tellheight=true, tellwidth=false, colgap=10, padding=(0, 0, 0, 0))
    rowgap!(fig.layout, 6)
    return fig
end

#price against the time-step size for a single strike, with the 95% Monte-Carlo confidence band
function schemePriceStrikeFigure(file, methods, strike; maxdt=0.5)
    df = readtable(file)
    d = rows(df[(df.dt.<=maxdt).&(df.Strike.==strike), :], :Method, methods...)
    fig = bookFigure(size=(400, 300))
    ax = bookAxis(fig[1, 1], xlabel=DT_LABEL, ylabel="Price", xscale=log10, xticks=DT_TICKS)
    groupBands!(ax, d.dt, d.Price, 1.96 .* d.StdErr, d.Method, order=collect(methods), colors=SCHEME_COLORS)
    schemeLines!(ax, d, methods)
    bookLegend!(ax, position=:lt)
    fig
end
@figure "heston_ft_price_andersen1_100" schemePriceStrikeFigure("heston_qe_case1.txt", ("Exact", "QE", "Euler-FT"), 100)
@figure "heston_ft_price_andersen1_140" schemePriceStrikeFigure("heston_qe_case1.txt", ("Exact", "QE", "Euler-FT"), 140)
@figure "heston_qel_price_andersen1" schemePriceFigure("heston_qe_case1.txt", ("Exact", "QE", "QELong"), maxdt=0.5)
@figure "heston_qel_price_andersen2" schemePriceFigure("heston_qe_case2.txt", ("Exact", "QE", "QELong"))
@figure "heston_qel_price_andersen3" schemePriceFigure("heston_qe_case3.txt", ("Exact", "QE", "QELong"))
@figure "heston_qe_price_andersen1" schemePriceFigure("heston_qe_case1.txt", ("Exact", "QE", "DVSS2", "Alfonsi2"))
@figure "heston_qe_price_andersen2" schemePriceFigure("heston_qe_case2.txt", ("Exact", "QE", "DVSS2"))
@figure "heston_qe_price_andersen3" schemePriceFigure("heston_qe_case3.txt", ("Exact", "QE", "DVSS2"))
@figure "heston_forward_start_dvss2" schemePriceFigure("heston_qe_case_forward_start.txt", ("Exact", "QE", "DVSS2", "Alfonsi2"), maxdt=1.0, size=(400, 800))
@figure "heston_forward_start_dvss2x" schemePriceFigure("heston_qe_case_forward_start.txt", ("Exact", "QE", "DVSS2X", "Alfonsi2"), maxdt=1.0, size=(400, 800))

function correlationFigure(case)
    df = vcat(readtable("heston_qe_correlation_andersen$case.txt"), readtable("heston_dvss2_correlation_andersen$case.txt"))
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="T", ylabel="Correlation")
    groupLines!(ax, df.T, df.Correlation, df.Method)
    bookLegend!(ax)
    return fig
end
@figure "heston_qe_correlation_andersen1" correlationFigure(1)
@figure "heston_qe_correlation_andersen2" correlationFigure(2)

function digitalBySpotFigure(file)
    df = readtable(file)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="Spot", ylabel="Digital option price")
    groupLines!(ax, df.Spot, df.Price, df.Scheme)
    bookLegend!(ax)
    return fig
end
@figure "heston_dvss2_lowvolvol" digitalBySpotFigure("heston_dvss2_lowvolvol.txt")
@figure "black_discrete_sampling" digitalBySpotFigure("black_discrete_sampling.txt")

## quasi Monte-Carlo convergence: error against the number of paths, with the standard error as band

function qmcConvergenceFigure(df, methods; bands=true, ylims=(-0.02, 0.02), stderr=:StdErr)
    d = rows(df, :Method, methods...)
    fig = bookFigure()
    ax = bookAxis(fig[1, 1], xlabel="N", ylabel=bands ? "Error in price" : "Error", xscale=log10)
    bands && groupBands!(ax, d.N, d.Error, d[!, stderr], d.Method, order=collect(methods))
    groupLines!(ax, d.N, d.Error, d.Method, order=collect(methods))
    ylims === nothing || ylims!(ax, ylims...)
    bookLegend!(ax)
    return fig
end
@figure "schobelzhu_eae_qmc_ft" qmcConvergenceFigure(readtable("schobelzhu_eae_qmc"), ("Memt", "FaureTezuka2", "FaureTezuka"))
@figure "schobelzhu_eae_qmc_ft_bb" qmcConvergenceFigure(readtable("schobelzhu_eae_qmc"), ("Memt", "FaureTezuka2-BB", "FaureTezuka-BB"))
@figure "schobelzhulv_sobol_convergence" qmcConvergenceFigure(readtable("schobelzhulv_sobol_convergence.txt"), ("Sobol", "Sobol-BB", "Memt"), bands=false, ylims=nothing)
@figure "schobelzhulv_sobolft_convergence" qmcConvergenceFigure(readtable("schobelzhulv_sobol_convergence.txt"), ("Sobol", "Sobol-FaureTezuka-BB", "Sobol-FaureTezuka"), bands=false, ylims=nothing)

## stochastic local volatility: leverage on the (F, V) plane

function slvFigure(methods)
    df = readtable("slv_comp8.txt")
    ref = rows(df, :Method, "Reference")
    d = rows(df, :Method, methods...)
    fig = bookFigure(size=(400, 400))
    ax = bookAxis(fig[1, 1], xlabel="F", ylabel="V")
    scatter!(ax, exp.(ref.X), ref.V, label="Reference", color=(:black, 0.15), markersize=6)
    groupLines!(ax, exp.(d.X), d.V, d.Method, order=collect(methods))
    limits!(ax, 0, 1.8, 0, 0.25)
    bookLegend!(ax)
    return fig
end
@figure "slv_comp8_bin" slvFigure(("Bin", "BinSpline"))
@figure "slv_comp8_nc" slvFigure(("Bin", "NonConform"))
@figure "slv_comp8_particle15" slvFigure(("Bin", "Particle"))
@figure "slv_comp8_particle06" slvFigure(("Bin", "Particle-6"))

using Random

## sample paths of the volatility with the same random numbers: constant Heston and piecewise-constant Heston of Hagan, calibrated to SPX
## options in 2010 with κ̃ = 2 and κ̃ = 5 (parameters of book/legacy/mc_study.jl). Euler full truncation with daily steps, on the variance
## v = σ̃(t)² z, with dz = κ̃(1 - z)dt + α̃(t)√z dW (eqn:hestonhagan, the fields leverage and σ hold σ̃ and α̃).
#An earlier version used a loop which never left the first period (α̃ = 2.2 and 2.6 instead of up to 6.0 and 14.0) with the seed 2: the columns
#*_first_period of the CSV file keep this loop, set HESTONTD_PATH_SUFFIX to "_first_period" to plot them. All the sample paths use the same seed.
const HESTONTD_PATH_SUFFIX = ""
const VOLATILITY_PATH_SEED = 7

const HESTON_CONSTANT_2010 = (v0=0.07093860933433274, κ=0.8284272454251295, θ=0.08423535683329449, σ=0.4293947711854393)
const HESTONTD_KAPPA2_2010 = AQFED.TermStructure.HaganHestonPiecewiseParams{Float64}([0.27461541542162726, 0.2595950348073077, 0.26808982574600515, 0.26934167408666293, 0.2752669801070057, 0.2856987038294999, 0.2840274979524871, 0.2911137777444947, 0.28676432470283714, 0.2982736748189721, 0.30305460760572717], 2.0, [-0.33046713301768027, -0.2919570421871561, -0.6426589435056318, -0.9899995772821046, -0.3352689399265528, -0.9899996250510502, -0.29234198474847417, -0.6456793493511088, -0.29098389878173125, -0.548249267604537, -0.407609077835016], [2.1977068491208476, 2.3119563733959128, 2.006099144794511, 0.3924131843056486, 3.647379368661464, 0.9360405392243931, 4.322969674240923, 2.415051257134871, 5.356052999395499, 4.045436371729674, 5.963020006124786], [0.0, 0.057534247, 0.153424658, 0.230136986, 0.479452055, 0.728767123, 1.22739726, 1.726027397, 2.243835616, 2.742465753, 3.24109589])
const HESTONTD_KAPPA5_2010 = AQFED.TermStructure.HaganHestonPiecewiseParams{Float64}([0.27450249506335317, 0.25977235437080537, 0.26782880425875416, 0.26902117368130685, 0.27537328968572994, 0.2863900567537211, 0.2848359412033214, 0.2909692590786829, 0.28852980357513586, 0.2986757658620479, 0.30574713843468165], 5.0, [-0.3049836598089387, -0.33507082481344685, -0.32640906322878227, -0.989997588383315, -0.33225882447091715, -0.46706764129394485, -0.40829333086320274, -0.4297992996849096, -0.4112784949897249, -0.43032374305429066, -0.4425848812559059], [2.5881625681414024, 2.543601337019924, 4.9985363916080665, 0.9688728320645006, 5.82025957219315, 4.944006573955143, 7.165629741910108, 8.411076713057739, 10.162128575566362, 11.514938427839637, 14.027108114664067], [0.0, 0.057534247, 0.153424658, 0.230136986, 0.479452055, 0.728767123, 1.22739726, 1.726027397, 2.243835616, 2.742465753, 3.24109589])

#dt keeps the rounding of the former code: with a large α̃, a path close to zero amplifies it, and the sample path changes after a few years
function hestonVariancePath(p, n, seed=nothing)
    seed === nothing || Random.seed!(seed)
    t = range(0, 10.0, length=n)
    dt = t[2:end] - t[1:end-1]
    dw = randn(length(dt)) .* sqrt.(dt)
    v = zeros(n)
    v[1] = p.v0
    for i in 1:n-1
        s = sqrt(max(v[i], 0))
        v[i+1] = v[i] + p.κ * (p.θ - s^2) * dt[i] + p.σ * s * dw[i]
    end
    return t, v
end

function hestonPiecewiseVariancePath(p, n, seed; firstPeriodOnly=false)
    Random.seed!(seed)
    t = range(0, 10.0, length=n)
    dt = t[2:end] - t[1:end-1]
    dw = randn(length(dt)) .* sqrt.(dt)
    v, z = zeros(n), ones(n)
    v[1] = p.leverage[1]^2
    for i in 1:n-1
        k = firstPeriodOnly ? 1 : searchsortedlast(p.startTime, t[i])
        s = sqrt(max(z[i], 0))
        z[i+1] = z[i] + p.κ * (1 - s^2) * dt[i] + p.σ[k] * s * dw[i]
        v[i+1] = p.leverage[k]^2 * z[i+1]
    end
    return t, v
end

@compute "hestontd_volatility_paths" begin
    n, seed = 365 * 10, VOLATILITY_PATH_SEED
    percent(v) = 100 .* sqrt.(max.(v, 0.0))
    t, v = hestonVariancePath(HESTON_CONSTANT_2010, n, seed)
    path(p; kwargs...) = percent(last(hestonPiecewiseVariancePath(p, n, seed; kwargs...)))
    writetable("hestontd_volatility_paths.csv", DataFrame(t=collect(t), heston=percent(v), td2=path(HESTONTD_KAPPA2_2010), td5=path(HESTONTD_KAPPA5_2010),
        td2_first_period=path(HESTONTD_KAPPA2_2010, firstPeriodOnly=true), td5_first_period=path(HESTONTD_KAPPA5_2010, firstPeriodOnly=true)))
end

function hestonVolatilityPathFigure(series; file="hestontd_volatility_paths.csv")
    df = readcsv(file)
    fig = bookFigure(size=(750, 350))
    ax = bookAxis(fig[1, 1], xlabel="Time in years", ylabel="Instantaneous volatility (%)", xticks=0:2:10)
    for (i, (column, label)) in enumerate(series)
        lines!(ax, df.t, df[!, column], color=PALETTE[i], label=label)
    end
    Legend(fig[2, 1], ax, orientation=:horizontal, framevisible=false)
    fig
end
hestonVolatilityPathFigure(file, series) = hestonVolatilityPathFigure(series; file=file)
@figure "hestontd2_volatility_path" hestonVolatilityPathFigure([("heston", "Heston"), ("td2" * HESTONTD_PATH_SUFFIX, "Time-dependent Heston")])
@figure "hestontd5_volatility_path" hestonVolatilityPathFigure([("td2" * HESTONTD_PATH_SUFFIX, "κ̃ = 2"), ("td5" * HESTONTD_PATH_SUFFIX, "κ̃ = 5")])

## the same with the calibrated Schöbel-Zhu model, whose volatility goes negative, and with the two variances of the calibrated Double Heston model

const SCHOBELZHU_2010 = (v0=0.26239704055254287, κ=0.7386193638419278, θ=0.21511046752047921, σ=0.21189212665310628)
const DOUBLE_HESTON_1 = (v0=0.029, κ=4.847, θ=0.027, σ=1.082)
const DOUBLE_HESTON_2 = (v0=0.044, κ=0.009, θ=0.840, σ=0.259)

function schobelZhuVolatilityPath(p, n, seed)
    Random.seed!(seed)
    t = range(0, 10.0, length=n)
    dt = t[2:end] - t[1:end-1]
    dw = randn(length(dt)) .* sqrt.(dt)
    v = zeros(n)
    v[1] = p.v0
    for i in 1:n-1
        v[i+1] = v[i] + p.κ * (p.θ - v[i]) * dt[i] + p.σ * dw[i]
    end
    return t, v
end

@compute "volatility_paths_sz_double_heston" begin
    n, seed = 365 * 10, VOLATILITY_PATH_SEED
    percent(v) = 100 .* sqrt.(max.(v, 0.0))
    t, heston = hestonVariancePath(HESTON_CONSTANT_2010, n, seed)
    _, schobelZhu = schobelZhuVolatilityPath(SCHOBELZHU_2010, n, seed)
    writetable("hestonsz_volatility_path.csv", DataFrame(t=collect(t), heston=percent(heston), schobelzhu=100 .* schobelZhu))
    #two independent drivers: the second path carries on with the random numbers of the first
    _, v1 = hestonVariancePath(DOUBLE_HESTON_1, n, seed)
    _, v2 = hestonVariancePath(DOUBLE_HESTON_2, n)
    writetable("double_heston_variance_paths.csv", DataFrame(t=collect(t), v1=percent(v1), v2=percent(v2), total=percent(v1 .+ v2)))
end

@figure "hestonsz_volatility_path" hestonVolatilityPathFigure("hestonsz_volatility_path.csv", [("heston", "Heston"), ("schobelzhu", "Schöbel-Zhu")])
@figure "double_heston_variance_paths" hestonVolatilityPathFigure("double_heston_variance_paths.csv", [("v1", "√v₁"), ("v2", "√v₂"), ("total", "√(v₁ + v₂)")])
