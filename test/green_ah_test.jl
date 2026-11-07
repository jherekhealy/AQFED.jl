using AQFED, Test
using AQFED.PDDE, AQFED.Black
using AQFED.PDDE.GreenAH
const GH = AQFED.PDDE.GreenAH

@testset "green-ah-flat-closed-form" begin
	#with a flat local volatility, φ± = x^r∓ with r± the roots of r(r-1) = 2/(Δσ²), W = r₊ - r₋, and the first step is the
	#local variance gamma closed form c(x) - (1-x)⁺ = G(x, 1) = min(x^r₊, x^r₋)/W
	knots = exp.(range(-0.5, 0.5, length=8)); σ = 0.25; dt = 0.3
	st = GH.greenStep(knots, fill(σ, length(knots) + 1), dt; N=12, sub=1)
	rp, rm = GH.roots(2 / (dt * σ^2))
	@test isapprox(st.W, rp - rm, rtol=1e-12)
	for y in (-2.0, -0.6, -0.1, 0.0, 0.2, 0.45, 1.5)
		@test isapprox(GH.psi(st, y, 1), rp * y, atol=1e-11)
		@test isapprox(GH.psi(st, y, 2), rm * y, atol=1e-11)
	end
end

@testset "green-ah-kernel-mass" begin
	#∫ G(x, ξ)/S(x) dx = 1 and ∫ x G(x, ξ)/S(x) dx = ξ: the densities of the steps have mass and mean one, for a local
	#volatility that is not flat (the first step from the Dirac at the forward, the second from the first)
	knots = exp.(range(-0.6, 0.5, length=10)); coeffs = [0.45, 0.45, 0.38, 0.3, 0.24, 0.2, 0.18, 0.17, 0.19, 0.22, 0.22]
	breaks = sort(unique(vcat(0.0, log.(knots))))
	st1 = GH.greenStep(knots, coeffs, 0.1; N=16, sub=2)
	rate = GH.rateOf([GH.stepRate(knots, coeffs, 0.1)])
	edges = GH.makeSegments(breaks, -12.0, 8.0, rate; τ=6.0, hmax=0.03)
	d1 = GH.firstDensity(st1, edges, 16)
	m0, m1 = GH.moments(d1)
	@test isapprox(m0, 1.0, atol=1e-10)
	@test isapprox(m1, 1.0, atol=1e-10)
	st2 = GH.greenStep(knots, 1.2 .* coeffs, 0.25; N=16, sub=2)
	d2 = GH.nextDensity(st2, GH.stepTable(d1, edges, 16), d1)
	m0, m1 = GH.moments(d2)
	@test isapprox(m0, 1.0, atol=1e-10)
	@test isapprox(m1, 1.0, atol=1e-10)
	#positive everywhere, and the same density on the same segments by the fixed part matrices
	@test all(isfinite, reduce(vcat, d2.lp))
	d2s = GH.nextDensitySame(st2, d1)
	@test maximum(abs.(reduce(vcat, d2s.lp) .- reduce(vcat, d2.lp))[abs.(reduce(vcat, d2.nodes)).<2]) < 1e-6
end

@testset "green-ah-calibration-flatblack" begin
	ttes = [0.1, 0.25, 0.5, 1.0, 2.0]; F = [100.0, 101.0, 102.0, 103.0, 105.0]
	strikes = [F[i] .* exp.(range(-3, 3, length=21) .* 0.2 .* sqrt(ttes[i])) for i in eachindex(ttes)]
	prices = [[blackScholesFormula(true, K, F[i], 0.04 * ttes[i], 1.0, 1.0) for K in strikes[i]] for i in eachindex(ttes)]
	w = [ones(length(k)) for k in strikes]
	s = calibrateGreenAHSurface(ttes, F, strikes, prices, w)
	err(s) = [1e4 * (sqrt(PDDE.varianceByLogmoneyness(s, log(K / F[i]), ttes[i])) - 0.2) for i in eachindex(ttes) for K in strikes[i]]
	@test maximum(abs, err(s)) < 0.2
	#no arbitrage: positive density, prices increasing in time at fixed moneyness (also between the expiries)
	xs = exp.(range(-1.5, 1.5, length=61))
	@test all(PDDE.densityNormalized(s, x, i) > 0 for x in xs, i in eachindex(ttes))
	ts = sort(vcat(ttes, [0.05, 0.17, 0.4, 0.8, 1.5]))
	for k in 1:length(ts)-1
		@test all(PDDE.priceNormalized(s, x, ts[k+1]) >= PDDE.priceNormalized(s, x, ts[k]) for x in xs)
	end
	#the analytic Jacobian and automatic differentiation give the same fit
	sad = calibrateGreenAHSurface(ttes, F, strikes, prices, w; jacobian=:ad)
	@test maximum(abs, err(s) .- err(sad)) < 1e-3
	#the Dupire local volatility between the expiries: σᵢ just after Tᵢ₋₁, that of greenDupireAtExpiry at Tᵢ
	yq = log.(strikes[3] ./ F[3])
	@test isapprox(greenDupireLocalVolatility(s, ttes[2] + 1e-4, yq), [PDDE.localVolatility(s, exp(y), ttes[3]) for y in yq], rtol=2e-3)
	@test isapprox(greenDupireLocalVolatility(s, ttes[3], yq), greenDupireAtExpiry(s, 3, yq), rtol=1e-12)
	@test all(v -> 0 < v < 0.3, greenDupireLocalVolatility(s, (ttes[2] + ttes[3]) / 2, yq))
	#substeps: the Dupire local volatility at the end of the intervals closer to the calibrated one
	s4 = calibrateGreenAHSurface(ttes, F, strikes, prices, w; substeps=4)
	@test maximum(abs, err(s4)) < 0.2
	gap(s, i) = maximum(abs.(greenDupireAtExpiry(s, i, log.(strikes[i] ./ F[i])) .- [PDDE.localVolatility(s, K / F[i], ttes[i]) for K in strikes[i]]))
	@test all(gap(s4, i) < gap(s, i) for i in 2:length(ttes))
end
