using AQFED, Test, AQFED.Basket, AQFED.Asian, Printf, DataFrames, StatsBase

@testset "BorovykhTable1" begin
	spot = 100.0
	strike = 100.0
	r = 0.0
	q = 0.0
	vols = [0.2, 0.4, 0.6, 0.8, 1.0]
	tte = 1.0
	n = 10
	weights = ones(Float64, n) / n
	tvar = zeros(Float64, n)
	forward = ones(Float64, n) * spot
	discountFactor = 1.0
	p = DeelstraBasketPricer(1, 3)
	pg0 = VorstGeometricExpansion(0)
	pg1 = VorstGeometricExpansion(1)
	pv0 = VorstLevyExpansion(0)
	pv1 = VorstLevyExpansion(1)
	pv2 = VorstLevyExpansion(2)
	pv3 = VorstLevyExpansion(3)
	ps0 = ShiftedGeometricExpansion(0, 3)
	ps1 = ShiftedGeometricExpansion(1, 3)
	ps2 = ShiftedGeometricExpansion(2, 3)

	gentlePrices = [4.9312, 9.7466, 14.3348, 18.5934, 22.4331]
	sophisPrices = [4.9476, 9.8761, 14.7669, 19.6015, 24.3623]
	refPrices = [4.9452, 9.8655, 14.7219, 19.4762, 24.1248]
	tols = [0.0013, 0.0057, 0.0156, 0.0289, 0.0523] * 2
	for (refPrice, tol, sophisPrice, gentlePrice, vol) in zip(refPrices, tols, sophisPrices, gentlePrices, vols)
		for i ∈ 1:n
			ti = (i) / (n) * tte
			tvar[i] = vol^2 * ti
			forward[i] = spot * exp((r - q) * ti)
		end
		price = priceAsianFixedStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
		@test isapprox(refPrice, price, atol = tol)
		price = priceAsianFixedStrike(pg0, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VGE-0 %.4f %.2e\n", strike, price, price - refPrice)
		@test isapprox(sophisPrice, price, atol = 1e-4)
		price = priceAsianFixedStrike(pg1, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VGE-1 %.4f %.2e\n", strike, price, price - refPrice)
		price = priceAsianFixedStrike(pv0, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VLE-0 %.4f %.2e\n", strike, price, price - refPrice)
		price = priceAsianFixedStrike(pv1, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VLE-1 %.4f %.2e\n", strike, price, price - refPrice)
		price = priceAsianFixedStrike(pv2, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VLE-2 %.4f %.2e\n", strike, price, price - refPrice)
		price = priceAsianFixedStrike(pv3, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VLE-3 %.4f %.2e\n", strike, price, price - refPrice)
		price = priceAsianFixedStrike(ps0, true, strike, discountFactor, spot, forward, tvar, weights)
		@test isapprox(refPrice, price, atol = tol)
		@printf("%6.2f VSE-0 %.4f %.2e\n", strike, price, price - refPrice)
		price = priceAsianFixedStrike(ps1, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VSE-1 %.4f %.2e\n", strike, price, price - refPrice)
		price = priceAsianFixedStrike(ps2, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VSE-2 %.4f %.2e\n", strike, price, price - refPrice)
		@test isapprox(refPrice, price, atol = tol)
	end

end

@testset "ZhouTable1" begin
	spot = 30.78
	strikes = spot .* [0.8, 1.0, 1.2]
	vol = 0.4133
	q = 0.0097
	r = 0.06
	tte = 1.0
	freq = 52 #12 or 52
	nWeights = Int(tte * freq)
	weights = zeros(Float64, nWeights)
	tvar = zeros(Float64, nWeights)
	forward = zeros(Float64, nWeights)
	for i ∈ eachindex(weights)
		weights[i] = 1.0 / (nWeights)
		ti = (i) / (nWeights) * tte
		tvar[i] = vol^2 * ti
		forward[i] = spot * exp((r - q) * ti)
	end
	discountFactor = exp(-r * tte)
	refPrices = [7.0243, 3.2284, 1.2588] #MC ref values from paper  
	mcPrices = [7.024276919490905,
		3.228397973658813,
		1.2585822156032385]
	#mcPrices = map(strike -> priceAsianFixedStrike(MonteCarloEngine(true,1024*1024*32),true,strike,discountFactor,spot,forward,tvar,weights), strikes)
	pricers = Dict(
		"Ju" => JuBasketPricer(),
		"LB" => DeelstraLBBasketPricer(1, 3),
		"VG2" => VorstGeometricExpansion(2),
		"Deelstra" => DeelstraBasketPricer(1, 3),
		"VG3" => VorstGeometricExpansion(3),
		"VG1" => VorstGeometricExpansion(1),
		"VL2" => VorstLevyExpansion(2),
		"VL3" => VorstLevyExpansion(3),
	)
	priceMap = Dict()
	for (pricerName, pricer) ∈ pricers
		prices = map(strike -> priceAsianFixedStrike(pricer, true, strike, discountFactor, spot, forward, tvar, weights), strikes)
		priceMap[pricerName] = prices
	end
	@test isapprox(mcPrices, priceMap["Deelstra"], atol = 1e-4)
	@test isapprox(mcPrices, priceMap["VG3"], atol = 1e-4)
	@test isapprox(mcPrices, priceMap["VG2"], atol = 1e-4)
	@test isapprox(mcPrices, priceMap["VL3"], atol = 1e-4)
	@test isapprox(mcPrices, priceMap["VL2"], atol = 2e-4)
	@test isapprox(mcPrices, priceMap["Ju"], atol = 1e-2)

end
@testset "JuAsianTable4" begin
	#Table 4 of Ju "Pricing Asian and Basket Options via Taylor Expansions"
	spot = 100.0
	r = 0.09
	q = 0.0
	tte = 3.0
	vols = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
	vols = [0.5]
	strikes = [80.0, 95.0, 100.0, 105.0, 120.0]
	nWeights = Int(tte * 52)
	weights = zeros(Float64, nWeights + 1)
	tvar = zeros(Float64, nWeights + 1)
	forward = zeros(Float64, nWeights + 1)
	pricers = Dict(
		"Ju" => JuBasketPricer(),
		"Deelstra-LB" => DeelstraLBBasketPricer(2, 3),
		"VG2" => VorstGeometricExpansion(2),
		"Deelstra" => DeelstraBasketPricer(2, 3, AQFED.Math.DoubleExponential()),
		"VG3" => VorstGeometricExpansion(3),
		"VG1" => VorstGeometricExpansion(1),
		"VL3" => AQFED.Basket.VorstLevyExpansion(3),
		"VL2" => AQFED.Basket.VorstLevyExpansion(2),
		"VS3" => AQFED.Basket.ShiftedGeometricExpansion(3, 3),
		"VS2" => AQFED.Basket.ShiftedGeometricExpansion(2, 3),)
	priceVolArray = Vector()
	for vol ∈ vols
		priceMap = Dict()
		for i ∈ 1:length(weights)
			weights[i] = 1.0 / (nWeights + 1)
			ti = (i - 1) / (nWeights) * tte
			tvar[i] = vol^2 * ti
			forward[i] = spot * exp((r - q) * ti)
		end
		discountFactor = exp(-r * tte)
		refPrices = [19.0144, 16.5766, 14.3830] #TS values
		pricesMC = priceAsianFixedStrike(MonteCarloEngine(true, 1024 * 1024), true, strikes, discountFactor, spot, forward, tvar, weights)
		priceMap["MC"] = pricesMC
		for (pricerName, pricer) ∈ pricers
			prices = map(strike -> priceAsianFixedStrike(pricer, true, strike, discountFactor, spot, forward, tvar, weights), strikes)
			priceMap[pricerName] = prices
		end
		push!(priceVolArray, priceMap)
		println("vol=", vol, " done")
	end
	for (ivol, vol) ∈ enumerate(vols)
		priceMap = priceVolArray[ivol]
		for (i, strike) ∈ enumerate(strikes)
			@printf(
				"(%1.2f, %d) & %2.4f & %2.4f & %2.4f & %2.4f &  %2.4f & %2.4f & %2.4f & %2.4f\\\\\n",
				vol,
				strike,
				priceMap["MC"][i],
				priceMap["Ju"][i],
				priceMap["Deelstra-LB"][i],
				priceMap["Deelstra"][i],
				priceMap["VG1"][i],
				priceMap["VG2"][i],
				priceMap["VG3"][i],
				priceMap["VL3"][i]
			)
		end
	end
	mcPrices = Vector{Float64}()
	for (ivol, vol) ∈ enumerate(vols)
		for p ∈ priceVolArray[ivol]["MC"]
			push!(mcPrices, p)
		end
	end
	for (name, pricer) ∈ pricers
		allPrices = Vector{Float64}()
		for (ivol, vol) ∈ enumerate(vols)
			for p ∈ priceVolArray[ivol][name]
				push!(allPrices, p)
			end
		end
		@printf("%s %1.4f %1.4f\n", name, rmsd(allPrices, mcPrices), maximum(abs.(allPrices - mcPrices)))
	end
	#Deelstra is much more accurate than TE6, as expected with vols relatively high
	@test rmsd(mcPrices, priceVolArray[1]["Deelstra"]) < 2e-4
	@test rmsd(mcPrices, priceVolArray[1]["VG3"]) < 4e-4
	@test rmsd(mcPrices, priceVolArray[1]["VL3"]) < 6e-4
	@test rmsd(mcPrices, priceVolArray[1]["Ju"]) > 3e-2
end

@testset "ZeliadeAsian" begin
	sigmas = vec(
		[0.22452963452712216 0.2245023704804817 0.2242841387952672 0.2250352559059984 0.22330701093809505 0.2221473781593173 0.2213294490509932 0.22101073165314433 0.22076252222523274 0.22054642268935865 0.2199906839589841 0.21952649365957014 0.21913294923525112 0.2187950620415034 0.21850180382746803 0.2182825326072634 0.2189384161145551 0.21951977853973867 0.2200386429294136 0.2205045769889864 0.22092528964093286 0.22133123380174013 0.22225449101108727 0.22309745370531386 0.22387017603730977 0.2245810983855645 0.22523735909225653 0.22584034246215315 0.22629176612789786 0.22671228390970152 0.22710496659616297 0.22747249096026378 0.22781720102687417 0.22808871489332277 0.22807706916558076 0.22806606987668573 0.2280556646556824 0.22804580663994675 0.2280364537695389 0.22813619458109619 0.2287863693983155 0.2294038703291732 0.22999110624500166 0.23055025430924814 0.23108328725249241 0.2315919968738603 0.23207801436568642 0.2325428279525193],
	)
	times = vec(
		[0.08333333333333333 0.16666666666666666 0.25 0.3333333333333333 0.4166666666666667 0.5 0.5833333333333334 0.6666666666666666 0.75 0.8333333333333334 0.9166666666666666 1 1.0833333333333333 1.1666666666666667 1.25 1.3333333333333333 1.4166666666666667 1.5 1.5833333333333333 1.6666666666666667 1.75 1.8333333333333333 1.9166666666666667 2 2.0833333333333335 2.1666666666666665 2.25 2.3333333333333335 2.4166666666666665 2.5 2.5833333333333335 2.6666666666666665 2.75 2.8333333333333335 2.9166666666666665 3 3.0833333333333335 3.1666666666666665 3.25 3.3333333333333335 3.4166666666666665 3.5 3.5833333333333335 3.6666666666666665 3.75 3.8333333333333335 3.9166666666666665 4],
	)
	tte = 4.0
	r = 0.0
	q = 0.0
	spot = 100.0
	strikes = [80.0, 95.0, 100.0, 105.0, 120.0]
	nWeights = Int(tte * 12)
	weights = zeros(Float64, nWeights)
	tvar = zeros(Float64, nWeights)
	forward = zeros(Float64, nWeights)
	pricers = Dict(
		"Ju" => JuBasketPricer(),
		"Deelstra-LB" => DeelstraLBBasketPricer(2, 3),
		"VGE2" => VorstGeometricExpansion(2),
		"Deelstra" => DeelstraBasketPricer(2, 3, AQFED.Math.DoubleExponential()),
		"VG3" => VorstGeometricExpansion(3),
		"VG2" => VorstGeometricExpansion(2),
		"VG1" => VorstGeometricExpansion(1),
		"VL3" => AQFED.Basket.VorstLevyExpansion(3),
		"VL2" => AQFED.Basket.VorstLevyExpansion(2),)
	priceMap = Dict()
	for i ∈ 1:length(weights)
		weights[i] = 1.0 / (nWeights)
		ti = times[i] # (i - 1) / (nWeights) * tte
		tvar[i] = sigmas[i]^2 * ti
		forward[i] = spot * exp((r - q) * ti)
	end
	discountFactor = exp(-r * tte)
	refPrices = [22.412453, 12.806853, 10.424364, 8.420118, 4.272546]
	#note , 1024*1024*16 recommended for better estimate of errors.
	pricesMC = priceAsianFixedStrike(MonteCarloEngine(true, 1024 * 1024), true, strikes, discountFactor, spot, forward, tvar, weights)
	priceMap["MC"] = pricesMC
	for (pricerName, pricer) ∈ pricers
		prices = map(strike -> priceAsianFixedStrike(pricer, true, strike, discountFactor, spot, forward, tvar, weights), strikes)
		priceMap[pricerName] = prices
	end
	for (i, strike) ∈ enumerate(strikes)
		@printf(
			"(%d) & %2.4f & %2.4f & %2.4f & %2.4f &  %2.4f & %2.4f & %2.4f & %2.4f\\\\\n",
			strike,
			priceMap["MC"][i],
			priceMap["Ju"][i],
			priceMap["Deelstra-LB"][i],
			priceMap["Deelstra"][i],
			priceMap["VG1"][i],
			priceMap["VG2"][i],
			priceMap["VG3"][i],
			priceMap["VL3"][i]
		)
	end

end

@testset "LordAsianTable4" begin
	#Table 4 of Lord "Partially exact and bounded approximations for arithmetic Asian options"
	spot = 100.0
	r = 0.05
	q = 0.0
	tte = 5.0
	vol = 0.5
	moneyness = [0.5, 1.0, 1.5]
	strikes = [58.2370, 116.4741, 174.7111]
	nWeights = Int(tte)
	weights = zeros(Float64, nWeights)
	tvar = zeros(Float64, nWeights)
	forward = zeros(Float64, nWeights)
	for i ∈ eachindex(weights)
		weights[i] = 1.0 / (nWeights)
		ti = (i) / (nWeights) * tte
		tvar[i] = vol^2 * ti
		forward[i] = spot * exp((r - q) * ti)
	end
	discountFactor = exp(-r * tte)
	refPrices = [49.3944, 26.5780, 15.5342]
	priceMap = Dict()
	pricers = Dict(
		"Ju" => JuBasketPricer(),
		"LB(1,3)" => DeelstraLBBasketPricer(1, 3), #2,3 matches paper
		"LB(2,3)" => DeelstraLBBasketPricer(2, 3), #2,3 matches paper
		"LB(3,3)" => DeelstraLBBasketPricer(3, 3), #2,3 matches paper
		"VG2" => VorstGeometricExpansion(2),
		"D(1,3)" => DeelstraBasketPricer(1, 3, AQFED.Math.DoubleExponential()),
		"D(2,3)" => DeelstraBasketPricer(2, 3, AQFED.Math.DoubleExponential()),
		"D(3,3)" => DeelstraBasketPricer(3, 3, AQFED.Math.DoubleExponential()),
		"VG3" => VorstGeometricExpansion(3),
		"VG1" => VorstGeometricExpansion(1),
		"VL2" => VorstLevyExpansion(2),
		"VL3" => VorstLevyExpansion(3),
		"VS3" => AQFED.Basket.ShiftedGeometricExpansion(3, 3),
		"VS2" => AQFED.Basket.ShiftedGeometricExpansion(2, 3),
	)
	pricesMC = [49.39441452225911, 26.57802196472295, 15.534169470167514]
	#pricesMC = priceAsianFixedStrike(MonteCarloEngine(true, 1024 * 1024 * 32), true, strikes, discountFactor, spot, forward, tvar, weights)
	priceMap["MC"] = pricesMC
	for (pricerName, pricer) ∈ pricers
		prices = map(strike -> priceAsianFixedStrike(pricer, true, strike, discountFactor, spot, forward, tvar, weights), strikes)
		priceMap[pricerName] = prices
	end
	for (i, strike) ∈ enumerate(strikes)
		@printf("%1.4f ", strike)
		#for pricerName = ["Ju","LB13","LB23","LB33", "Deelstra13","Deelstra23", "Deelstra33","VLE-1","VLE-2","VLE-3"]
		for pricerName ∈ ["Ju", "LB(2,3)", "D(2,3)", "D(3,3)", "VG1", "VG2", "VG3", "VL3"]
			@printf("& %1.2f ", (priceMap[pricerName][i] - pricesMC[i]) * 1e4 / spot)
		end
		@printf("\\\\\n")
	end
	for (i, strike) ∈ enumerate(strikes)
		@test abs((priceMap["D(2,3)"][i] - pricesMC[i]) * 1e4 / spot) < 0.25
		@test abs((priceMap["VG3"][i] - pricesMC[i]) * 1e4 / spot) < 0.3
		@test abs((priceMap["VL3"][i] - pricesMC[i]) * 1e4 / spot) < 0.32
	end
	tte = 30.0
	vol = 0.25
	moneyness = [0.5, 1.0, 1.5]
	strikes = [118.9819, 237.9638, 356.9457]
	nWeights = Int(tte)
	weights = zeros(Float64, nWeights)
	tvar = zeros(Float64, nWeights)
	forward = zeros(Float64, nWeights)
	for i ∈ eachindex(weights)
		weights[i] = 1.0 / (nWeights)
		ti = (i) / (nWeights) * tte
		tvar[i] = vol^2 * ti
		forward[i] = spot * exp((r - q) * ti)
	end
	discountFactor = exp(-r * tte)
	refPrices = [30.5153, 19.1249, 13.1168]
	pricesMC = [30.51545116021278, 19.12509551537445, 13.11697143710736]
	pricesMC = refPrices

	t = [0.1, 1.1]
	tte = 1.1
	nWeights = 2
	weights = zeros(Float64, nWeights)
	tvar = zeros(Float64, nWeights)
	forward = zeros(Float64, nWeights)
	for i ∈ eachindex(weights)
		weights[i] = 1.0 / (nWeights)
		tvar[i] = vol^2 * t[i]
		forward[i] = spot * exp((r - q) * t[i])
	end
	discountFactor = exp(-r * tte)

end



@testset "LordAsianRateLikeTable4" begin
	#Same test case but with asian rate (inspired by paper but not in paper). Table 4 of Lord "Partially exact and bounded approximations for arithmetic Asian options"
	spot = 100.0
	r = 0.05
	q = 0.0
	tte = 5.0
	vol = 0.5
	moneyness = [0.5, 1.0, 1.5]
	strikes = [58.2370, 116.4741, 174.7111] ./ 100
	nWeights = Int(tte)
	weights = zeros(Float64, nWeights)
	tvar = zeros(Float64, nWeights+1)
	forward = zeros(Float64, nWeights+1)
	for i ∈ eachindex(weights)
		weights[i] = 1.0 / (nWeights)
		ti = (i) / (nWeights) * tte
		tvar[i] = vol^2 * ti
		forward[i] = spot * exp((r - q) * ti)
	end
	tvar[end] = tvar[end-1]
	forward[end] = forward[end-1]
	tvar[end-1] = tvar[end]-1e-8
	discountFactor = exp(-r * tte)
	refPrices = [49.9222, 18.0721, 4.8351] # from a PDE accurate to 2e-4.
	pricers = Dict(
		"Ju" => JuBasketPricer(),
		"LB(1,3)" => DeelstraLBBasketPricer(1, 3),
		"LB(2,3)" => DeelstraLBBasketPricer(2, 3), #2,3 matches paper
		"VG2" => VorstGeometricExpansion(2),
		"D(2,3)" => DeelstraBasketPricer(2, 3, AQFED.Math.DoubleExponential()),
		"VG2" => VorstGeometricExpansion(2),
		"VG1" => VorstGeometricExpansion(1),
		"VL1" => VorstLevyExpansion(1),
		"VL2" => VorstLevyExpansion(2),
		"VL3" => VorstLevyExpansion(3),
	)
	priceMap = Dict()
	for (pricerName, pricer) ∈ pricers
		prices = map(strike -> priceAsianFloatingStrike(pricer, true, strike, discountFactor, spot, forward, tvar, weights), strikes)
		priceMap[pricerName] = prices
	end

	for (i, strike) ∈ enumerate(strikes)
		@printf("%1.4f ", strike)
		#for pricerName = ["Ju","LB13","LB23","LB33", "Deelstra13","Deelstra23", "Deelstra33","VLE-1","VLE-2","VLE-3"]
		for pricerName ∈ ["Ju", "LB(2,3)", "D(2,3)", "VL1", "VL2", "VL3", "VG2"]
			@printf("& %1.2f ", (priceMap[pricerName][i] - refPrices[i]) * 1e4 / spot)
		end
		@printf("\\\\\n")
	end

	@test isapprox(refPrices, priceMap["D(2,3)"], atol = 5e-3)
	@test isapprox(refPrices, priceMap["LB(1,3)"], atol = 1.5e-1)
	@test isapprox(refPrices, priceMap["VL3"], atol = 7e-3)
	@test isapprox(refPrices, priceMap["VL2"], atol = 4e-2)
	@test isapprox(refPrices, priceMap["VG2"], atol = 3e-2)

	r = 0.1
	refPrices = [53.584722, 21.255485, 6.236079]
	for i ∈ eachindex(weights)
		weights[i] = 1.0 / (nWeights)
		ti = (i) / (nWeights) * tte
		tvar[i] = vol^2 * ti
		forward[i] = spot * exp((r - q) * ti)
	end
	tvar[end] = tvar[end-1]
	forward[end] = forward[end-1]
	tvar[end-1] = tvar[end]-1e-8

	discountFactor = exp(-r * tte)
	p = DeelstraBasketPricer(1, 3)
	pl = DeelstraLBBasketPricer(1, 3)
	pv2 = pricers["VL2"]
	pv3 = pricers["VL3"]
	for (refPrice, strike) in zip(refPrices, strikes)
		price = priceAsianFloatingStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
		@test isapprox(refPrice, price, atol = 5e-3)
		price = priceAsianFloatingStrike(pl, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f Deelstra-LB %.4f %.2e\n", strike, price, price - refPrice)
		@test isapprox(refPrice, price, atol = 1.5e-1)
		price = priceAsianFloatingStrike(pv2, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VL2 %.4f %.2e\n", strike, price, price - refPrice)
		price = priceAsianFloatingStrike(pv3, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VL3 %.4f %.2e\n", strike, price, price - refPrice)
	end

	r = 0.0
	q = 0.05
	refPrices = [32.604220, 9.611437, 2.114077]
	for i ∈ eachindex(weights)
		weights[i] = 1.0 / (nWeights)
		ti = (i) / (nWeights) * tte
		tvar[i] = vol^2 * ti
		forward[i] = spot * exp((r - q) * ti)
	end
	tvar[end] = tvar[end-1]
	forward[end] = forward[end-1]
	tvar[end-1] = tvar[end]-1e-8
	discountFactor = exp(-r * tte)
	p = DeelstraBasketPricer(1, 3)  #3,3 is worse  1,3 is best
	pl = DeelstraLBBasketPricer(1, 3)
	for (refPrice, strike) in zip(refPrices, strikes)
		price = priceAsianFloatingStrike(p, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f Deelstra    %.4f %.2e\n", strike, price, price - refPrice)
		@test isapprox(refPrice, price, atol = 5e-3)
		price = priceAsianFloatingStrike(pl, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f Deelstra-LB %.4f %.2e\n", strike, price, price - refPrice)
		@test isapprox(refPrice, price, atol = 1.5e-1)
		price = priceAsianFloatingStrike(pv2, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VL2 %.4f %.2e\n", strike, price, price - refPrice)
		price = priceAsianFloatingStrike(pv3, true, strike, discountFactor, spot, forward, tvar, weights)
		@printf("%6.2f VL3 %.4f %.2e\n", strike, price, price - refPrice)
	end
end

using Dates
@testset "japan" begin
	#=
	  startDate = Date(2019,06,14)
	expiryDate = Date(2019,09,16)
settleDate = Date(2019,12,18)
using BusinessDays
 dates = zeros(Date, 65)
 dates[1]=startDate
 for i=2:65
       dates[i]=advancebdays(:USSettlement,startDate,i-1)
       end
t = Dates.value.(dates .- startDate) ./ 365

#otherwise
startDate=Date(2023,3,1)
expiryDate=Date(2023,3,31)
settleDate=Date(2023,11,17)
	=#
	daysToExpiry = 94
	tte = daysToExpiry/365
	ttp = 187/365
	t = [0.0, 0.00821917808219178, 0.010958904109589041, 0.0136986301369863, 0.01643835616438356, 0.019178082191780823, 0.0273972602739726, 0.030136986301369864, 0.03287671232876712, 0.03561643835616438, 0.038356164383561646, 0.04657534246575343, 0.049315068493150684, 0.052054794520547946, 0.057534246575342465, 0.06575342465753424, 0.0684931506849315, 0.07123287671232877, 0.07397260273972603, 0.07671232876712329, 0.08493150684931507, 0.08767123287671233, 0.09041095890410959, 0.09315068493150686, 0.0958904109589041, 0.10410958904109589, 0.10684931506849316, 0.1095890410958904, 0.11232876712328767, 0.11506849315068493, 0.1232876712328767, 0.12602739726027398, 0.12876712328767123, 0.13150684931506848, 0.13424657534246576, 0.14246575342465753, 0.14520547945205478, 0.14794520547945206, 0.1506849315068493, 0.15342465753424658, 0.16164383561643836, 0.1643835616438356, 0.16712328767123288, 0.16986301369863013, 0.1726027397260274, 0.18082191780821918, 0.18356164383561643, 0.1863013698630137, 0.18904109589041096, 0.1917808219178082, 0.2, 0.20273972602739726, 0.2054794520547945, 0.20821917808219179, 0.21095890410958903, 0.2219178082191781, 0.22465753424657534, 0.2273972602739726, 0.23013698630136986, 0.23835616438356164, 0.2410958904109589, 0.24383561643835616, 0.2465753424657534, 0.2493150684931507, 0.25753424657534246]
	spot=108.56
	K=-68.0/100
#vols 1M, 2M, 3M = 21.5, 20.5, 19.85.  Other values= 11.7, 11.9, 12.3 
vol = 0.2
rJPY = -0.08/100
rUSD = 2.32/100
nWeights = length(t)
	weights = zeros(Float64, nWeights)
	tvar = zeros(Float64, nWeights+1)
	forward = zeros(Float64, nWeights+1)
	for (i,ti) ∈ enumerate(t)
		weights[i] = 1.0 / (nWeights)
		ti = (i) / (nWeights) * tte
		tvar[i] = vol^2 * ti
		forward[i] = spot * exp((rJPY - rUSD) * ti)
	end
	tvar[end] = tvar[end-1]
	forward[end] = forward[end-1]
	tvar[end-1] = tvar[end]-1e-8
	discountFactor = exp(-rJPY * ttp)
	Ks = range(-spot*2/100, 0.0,length=21)
	refPrices = AQFED.Asian.priceAsianSpread(AQFED.Basket.MonteCarloEngine(true,1024*1024), false, Ks, discountFactor, spot, forward, tvar, vcat(weights,1),  nWeights)
	basketPricer = AQFED.Basket.VorstGeometricExpansion(3)
	prices = map(K -> AQFED.Asian.priceAsianFloatingStrike(basketPricer, false, 1.0, discountFactor, spot,  vcat(1.0,forward), vcat(0,tvar), vcat(K,weights)),Ks)
end