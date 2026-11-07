using AQFED, Test, StatsBase
using AQFED.PDDE, AQFED.Black, AQFED.Collocation
#the studies and the plots formerly in this file are in book/legacy/lvg_study.jl

@testset "lvg-flatblack-c3" begin
	strikes = [0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4]
	vols = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
	sigmas = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
	l = 0.25
	u = 4.0
	tte = 0.25
	forward = 1.025
	#weights = ones(length(vols))
	prices, weights = Collocation.weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor = 1e-7)
	lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, weights, useVol = false, L = l, U = u)
	ivkLVG = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvg, true, strikes), forward, strikes, tte, 1.0)
	rmseLVG = StatsBase.rmsd(vols, ivkLVG)
	@test isapprox(0.0, rmseLVG, atol = 1e-8)
	lvgq = PDDE.calibrateEQuadraticLVG(tte, forward, strikes, prices, weights, useVol = false, L = l, U = u, model = "Quadratic")
	ivkLVGq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
	rmseLVGq = StatsBase.rmsd(vols, ivkLVGq)
	@test isapprox(0.0, rmseLVGq, atol = 1e-8)
	lvgqc3 = PDDE.calibrateEQuadraticLVG(tte, forward, strikes, prices, weights, useVol = false, L = l, U = u, model = "Quadratic", location = "Mid-XX")
	ivkLVGq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgqc3, true, strikes), forward, strikes, tte, 1.0)
	rmseLVGq = StatsBase.rmsd(vols, ivkLVGq)
	@test isapprox(0.0, rmseLVGq, atol = 1e-8)
	lvgqnotc3 = PDDE.calibrateEQuadraticLVG(tte, forward, strikes, prices, weights, useVol = false, L = l, U = u, model = "Quadratic", isC3 = false, location = "Mid-XX")
	ivkLVGq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgqnotc3, true, strikes), forward, strikes, tte, 1.0)
	rmseLVGq = StatsBase.rmsd(vols, ivkLVGq)
	@test isapprox(0.0, rmseLVGq, atol = 1e-8)

	strikes = round.([95.25466637597646,
			99.74155408701951,
			108.83682736543783,
			116.84297587845171,
			121.51717213620066,
			122.68847999646283,
			126.01033716036022,
			133.4592851856828,
			133.82546157127993,
			134.5683605635301], digits = 2)
	forward = 101.0
	#weights = ones(length(vols))
	prices, weights = Collocation.weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor = 1e-7)
	lvgq = PDDE.calibrateQuadraticLVG(tte, forward, strikes, prices, weights, useVol = false, L = forward / 3, U = forward * 3, model = PDDE.Quadratic(), location = "Strikes")
	ivkLVGq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
	rmseLVGq = StatsBase.rmsd(vols, ivkLVGq)
	@test isapprox(0.0, rmseLVGq, atol = 1e-8)
end

@testset "lvg-quadratic-robustness" begin
	using BSplines, LinearAlgebra
	#quadratic LVG of given B-spline coefficients λ on the knots x, forward 1
	function makeLVG(x, λ)
		m = length(x) - 1
		α, β, γ = zeros(m), zeros(m), zeros(m)
		PDDE.toQuadForm!(PDDE.Quadratic(), m, length(λ), 0, x, α, β, γ, λ, basis = BSplineBasis(3, x))
		θ = zeros(2m)
		PDDE.computeCoeffsQuadratic(searchsortedfirst(x, 1.0) - 1, 1.0, x, α, β, γ, θ)
		PDDE.QuadraticLVG(x, α, β, γ, θ, λ, 1.0, 1.0)
	end
	#reference: finite differences on c - a(x)^2 c'' = (1-x)^+, c(L) = 1-L, c(U) = 0
	function referencePrices(lvg; n = 100_001)
		xs = range(lvg.x[1], lvg.x[end], length = n)
		h = step(xs)
		a2 = [(i = min(searchsortedlast(lvg.x, x), length(lvg.a)); (lvg.a[i] * x^2 + lvg.b[i] * x + lvg.c[i])^2) for x in xs]
		d, dl, du = 1 .+ 2a2 ./ h^2, -a2[2:end] ./ h^2, -a2[1:end-1] ./ h^2
		d[1] = d[end] = 1; du[1] = dl[end] = 0
		rhs = max.(1 .- xs, 0.0); rhs[1] = 1 - xs[1]; rhs[end] = 0
		xs, Tridiagonal(dl, d, du) \ rhs
	end
	x = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0]
	#a small or negative coefficient (as may be set by the C3 condition) with a positive local variance: complex roots and
	#an angle beyond π on the interval, where the principal log of the ratio was wrong
	for λmid in (0.02, -0.02, -0.05)
		lvg = makeLVG(x, [0.2, 0.2, 0.2, 0.2, λmid, 0.2, 0.2, 0.2, 0.2])
		xs, c = referencePrices(lvg)
		@test maximum(abs(PDDE.priceEuropean(lvg, true, xs[j]) - c[j]) for j = 1:100:length(xs)) < 1e-8
	end
	#very small local variance on a wide first interval: cosh overflows in the tridiagonal system
	lvg = makeLVG([1 / 6, 0.9, 0.95, 1.0, 1.05, 1.1, 6.0], fill(0.001, 8))
	@test all(isfinite, lvg.θ)
	@test isapprox(PDDE.priceEuropean(lvg, true, 1.0), 0.0005, atol = 1e-12)
end
