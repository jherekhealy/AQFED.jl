# Chapter 7: calibration of the discrete local volatility of Andreasen and Huge, one implicit step per expiry on a grid
# uniform in log-moneyness with the forward on the grid (exact forward, src/pdde/ah_bspline.jl), the quotes priced by the
# linear interpolation of the grid prices. The local volatility between the quotes is piecewise constant, piecewise
# linear or a Hyman non-negative cubic spline of model volatilities attached to the quoted strikes, flat beyond, and is
# calibrated expiry by expiry in implied volatility by Levenberg-Marquardt, the bounds by the algebraic transformation.
# Kahale surface (first expiry plotted) and the SPX500 options expiring on March 29, 2018 as of March 24, 2017.

using AQFED, AQFED.PDDE, AQFED.Black, PPInterpolation, Printf
isdefined(@__MODULE__, :SmileData) || include(joinpath(@__DIR__, "..", "test", "smile_data.jl"))

const DAH_VOL_BOUNDS = (0.01, 2.0) #σ_L and σ_U

"""
grid uniform in z = ln(K/F) on `m` points, the forward on the grid: n_dev = 4 standard deviations of the longest expiry,
at least a factor 2 beyond the quoted strikes (the boundaries absorb the probability), at most 4 away from the forward
"""
function dahGrid(tLast, σAtm, ks, m; ndev=4)
    zmin = max(min(-σAtm^2 * tLast / 2 - ndev * σAtm * sqrt(tLast), log(minimum(minimum.(ks)) / 2)), -4.0)
    zmax = min(max(σAtm^2 * tLast / 2 + ndev * σAtm * sqrt(tLast), log(2maximum(maximum.(ks)))), 4.0)
    j0 = floor(Int, -zmin / ((zmax - zmin) / (m - 1)))
    h = -zmin / j0
    return collect(zmin .+ h .* (0:m-1))
end

"local volatility on the grid z from the model volatilities ν attached to the log-strikes zq"
function dahLocalVol(kind, z, zq, ν)
    h = z[2] - z[1]
    if kind == :constant
        #ν_l from the middle of the grid indices below the strikes l-1 and l to the middle of those of l and l+1
        i = [floor(Int, (y - z[1]) / h) for y in zq]
        thresholds = [(i[l-1] + i[l]) / 2 for l = 2:length(i)]
        return [ν[1+count(t -> j - 1 >= t, thresholds)] for j in eachindex(z)]
    end
    #nodes at the grid points closest to the strikes
    zn = [z[round(Int, (y - z[1]) / h)+1] for y in zq]
    if kind == :linear
        return map(z) do y
            y <= zn[1] ? ν[1] : y >= zn[end] ? ν[end] : (l = searchsortedlast(zn, y); ν[l] + (ν[l+1] - ν[l]) * (y - zn[l]) / (zn[l+1] - zn[l]))
        end
    end
    T = eltype(ν)
    pp = makeCubicPP(zn, collect(ν), PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2HymanNonNegative())
    return [y <= zn[1] ? ν[1] : y >= zn[end] ? ν[end] : PPInterpolation.evaluate(pp, y) for y in z]
end

"normalized call price of strike k from the probabilities P of the grid points x: the linear interpolation of the grid prices"
dahPrice(x, P, k) = sum(P[j] * (x[j] - k) for j in eachindex(x) if x[j] > k; init=zero(eltype(P)))

dahVol(c, k, t) = k >= 1 ? Black.impliedVolatilityThiophene(true, max(c, 1e-64), 1.0, k, t, 1.0) :
                  Black.impliedVolatilityThiophene(false, max(c - (1 - k), 1e-64), 1.0, k, t, 1.0)

"""
calibration expiry by expiry: `ks[i]` quoted K/F and `vols[i]` implied vols of the expiry `ttes[i]`, on `m` grid points.
Returns the grid x, the local volatilities, the errors in implied vol at the quotes and the iterations of each expiry.
"""
function calibrateDiscreteAH(ttes, ks, vols, m, kind; σAtm=vols[end][argmin(abs.(log.(ks[end])))])
    z = dahGrid(ttes[end], σAtm, ks, m)
    x = exp.(z)
    P = [j == argmin(abs.(z)) ? 1.0 : 0.0 for j in eachindex(z)] #all the probability at the forward
    tr = PDDE.makeTransform(DAH_VOL_BOUNDS..., "LM-ALG")
    lvs, errs, iters = Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), Int[]
    ν0 = nothing
    for (i, t) in enumerate(ttes)
        dt = t - (i == 1 ? 0.0 : ttes[i-1])
        zq = log.(ks[i])
        #initial guess: the at-the-money vol, then the calibrated volatilities of the previous expiry
        guess = ν0 !== nothing && length(ν0) == length(zq) ? ν0 : fill(vols[i][argmin(abs.(zq))], length(zq))
        p0 = [AQFED.Math.inv(tr, clamp(g, 1.1DAH_VOL_BOUNDS[1], 0.9DAH_VOL_BOUNDS[2])) for g in guess]
        step(p) = PDDE.ahAdjointStep(x, P, dahLocalVol(kind, z, zq, tr.(p)), dt)
        function obj!(f, p)
            Pi = step(p)
            for (l, k) in enumerate(ks[i])
                f[l] = dahVol(dahPrice(x, Pi, k), k, t) - vols[i][l]
            end
            f
        end
        od = PDDE.OnceDifferentiable(obj!, p0, zeros(length(zq)); autodiff=PDDE.LsqFit.AutoForwardDiff(), inplace=true)
        fit = PDDE.LsqFit.levenberg_marquardt(od, p0)
        ν0 = tr.(fit.minimizer)
        push!(lvs, dahLocalVol(kind, z, zq, ν0))
        push!(errs, obj!(zeros(length(zq)), fit.minimizer)) #before P moves to this expiry (captured by step)
        P = step(fit.minimizer)
        push!(iters, fit.iterations)
    end
    return x, lvs, errs, iters
end

@compute "kahale_discrete_ah" begin
    s = loadKahaleSurface()
    forwards = s.spot .* exp.((s.r - s.q) .* s.expiries)
    ks = [s.strikes ./ f for f in forwards]
    vols = [s.vols[i, :] for i in eachindex(s.expiries)]
    lv, checks = DataFrame(), DataFrame()
    for kind in (:constant, :linear), m in (200, 300, 400, 2000)
        x, lvs, errs, iters = calibrateDiscreteAH(s.expiries, ks, vols, m, kind)
        for (i, t) in enumerate(s.expiries)
            append!(checks, DataFrame(Kind=string(kind), Points=m, Expiry=t, RMSE=sqrt(sum(abs2, errs[i]) / length(errs[i])),
                MaxError=maximum(abs, errs[i]), Iterations=iters[i]))
        end
        keep = 400 .<= x .* forwards[1] .<= 900
        append!(lv, DataFrame(Kind=string(kind), Points=m, Strike=x[keep] .* forwards[1], LocalVol=lvs[1][keep]))
        @printf("%-8s %5d  max error %.2e  iterations %d\n", kind, m, maximum(checks.MaxError[(checks.Kind .== string(kind)).&(checks.Points .== m)]), sum(iters))
    end
    writetable("kahale_discrete_ah_lv.csv", lv)
    writetable("kahale_discrete_ah_checks.csv", checks)
end

@compute "spx500_1y_2017_discrete_ah" begin
    q = readcsv("spx500_20170324_1y.csv")
    tte, forward = 1.013699, 2334.312
    lv, checks = DataFrame(), DataFrame()
    for kind in (:linear, :hyman)
        x, lvs, errs, iters = calibrateDiscreteAH([tte], [q.Strike ./ forward], [Float64.(q.Vol)], 800, kind)
        append!(lv, DataFrame(Kind=string(kind), LogSpot=log.(x .* forward), LocalVol=lvs[1]))
        append!(checks, DataFrame(Kind=string(kind), RMSE=sqrt(sum(abs2, errs[1]) / length(errs[1])), MaxError=maximum(abs, errs[1]), Iterations=iters[1]))
    end
    println(checks)
    writetable("spx500_1y_2017_discrete_ah_lv.csv", lv)
    writetable("spx500_1y_2017_discrete_ah_checks.csv", checks)
end
