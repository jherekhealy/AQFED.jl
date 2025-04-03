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
	#=
	#monthly
	mcPrices= [15.299389368297263, 14.57686539496854, 13.856188816900787, 13.13843795070089, 12.425044737027624, 11.717802403042063, 11.018834827058319, 10.330543604033572, 9.655517173703966, 8.996445994327907, 8.355996699402478, 7.736731535780706, 7.1409796180179415, 6.570798661025787, 6.027882223477481, 5.513536051618198, 5.02867842765189, 4.573811533626664, 4.149064430923215, 3.754208711108056, 3.388700004655659, 3.0517309016729675, 2.742269976690799, 2.459096796180759, 2.200857764559134, 1.9661337138886739, 1.7534239467174166, 1.5612016822926376, 1.387953308518999, 1.2321972484574344, 1.092493304346985, 0.9674522072009062, 0.8557592694601095, 0.7561798673899554, 0.6675427909738332, 0.588777779856894, 0.5188896436675161, 0.4569585941402504, 0.4021539919698648, 0.35370850021848865, 0.31092000192847047, 0.2731750494345868, 0.23990870198002806, 0.21060892799999834, 0.1848254386658481, 0.16215079001685728, 0.1422234308998729, 0.12471902347628863, 0.1093466781544504, 0.09585572931635028, 0.08402164384365637]
	strikes=(0.5:0.025:1.75).*spot
	mcPrices = priceAsianFixedStrike(MonteCarloEngine(true,1024*1024*64),true,strikes,discountFactor,spot,forward,tvar,weights)

	p = plot(xlab="Moneyness",ylab="Absolute price error in b.p.")
	for pricerName = ["VG1","LB","Ju","VL2","VG2","Deelstra","VG3"  ]
		prices = priceMap[pricerName]
		plot!(p,strikes./spot,abs.(prices-mcPrices)/spot.*1e4,label=pricerName,linestyle=:auto)
	end
	plot(p, yscale=:log10)
	plot!(yticks=[0.001,0.01,0.1,1.0])
	plot!(legend=:outerright)
	plot!(size=(800,400),margins=3Plots.mm)


	=#

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
	#=
	pricesMC= [60.60793606595354, 57.31247650189226, 54.18267122327183, 51.22169542916771, 48.428689318228436, 45.799892406977825, 43.32961151397672, 41.010861660960984, 38.835923024869295, 36.79673535378197, 34.88514243404879, 33.0931513698638, 31.412976582405495, 29.837175908368867, 28.358712611079447, 26.970903595775752, 25.667502001113547, 24.442724717752604, 23.291130513139986, 22.20760640041994, 21.18754008290338, 20.226566182432585, 19.32064393747319, 18.46605468275837, 17.659361603890037, 16.897395893527765, 16.177157682042477, 15.495953799532687, 14.851175540385004, 14.240563255431692, 13.661933820634692, 13.113203418622547, 12.592604197411841, 12.098355732354102, 11.628815077738501, 11.182494542243257, 10.758053795841896, 10.3541612683104, 9.96958437046821, 9.60322018356278, 9.254050823357348, 8.921077821386628, 8.603376566516136, 8.300101325136666, 8.010444289493691, 7.733695081382217, 7.469144109237979, 7.216094296562982, 6.973973736234091, 6.742237438096218, 6.520300674823362, 6.307674642221388, 6.103880670867341, 5.908473632679069, 5.721009724293115, 5.5411245269133325, 5.368424281306374, 5.202567907208277, 5.0432393259777974, 4.890112978420219, 4.742895199903845, 4.601318688629397, 4.465112302948394]
	strikes=40.0:5.0:350.0
	moneyness = strikes./sum(forward.*weights) .- 1.0
	 #30y    
	pricesMC= [37.365032448359585, 36.48660338642259, 35.63504352533947, 34.80990453125456, 34.01064801857199, 33.23664981514158, 32.487189002611856, 31.76155012362347, 31.058982053848837, 30.378701440907804, 29.719919907882154, 29.08188096873142, 28.463832175512422, 27.865034926482096, 27.28477160714669, 26.722343119154925, 26.177083085911846, 25.648315791333253, 25.135429954329346, 24.63782921911073, 24.154908631345226, 23.686137700484313, 23.230967486253554, 22.788868684629225, 22.359366178867045, 21.942002149312046, 21.536336858000634, 21.141921260158334, 20.758334949681856, 20.385198821450498, 20.022116582162624, 19.668744067551522, 19.32473033503131, 18.989747683043717, 18.663473699118867, 18.345608578060784, 18.03585585752176, 17.73394149223119, 17.439608211322614, 17.152595511776347, 16.87265730365258, 16.59955325460831, 16.33307864817904, 16.07300741174868, 15.819144312989975, 15.571279826401108, 15.329219950454616, 15.092778469243957, 14.86178626469504, 14.636068576230219, 14.415467366301085, 14.199838194878023, 13.989012712690275, 13.782834965146916, 13.581181972271818, 13.383931007232004, 13.190944726331512, 13.002096279427942, 12.817266247431398, 12.636341737254426, 12.459211232571572, 12.285758838104613, 12.115879447780353, 11.949484535029985, 11.786479801969488, 11.626759570476498, 11.47024491585437, 11.316846190164963, 11.16648065052727, 11.019078129069072, 10.87455045174692, 10.73281615962248, 10.593805714881254, 10.457443579079992, 10.323666109992091, 10.192417816791039, 10.06363312276608, 9.937246823652565, 9.813199123614485, 9.691433734921587, 9.571881947415145, 9.45449515576401, 9.33922137221768, 9.226017535594156, 9.114822452556895, 9.005590267659453, 8.898273955588136, 8.792837697641698, 8.689235586156213, 8.587412484273155, 8.487331330236309, 8.388958445475467, 8.29225624153133, 8.197175922257838, 8.10368958596068, 8.01176246449686, 7.9213598259412965, 7.8324353105234215, 7.744962190320569, 7.658917615817006, 7.574263311599525, 7.490970287329001, 7.409006235879907, 7.328338228003912, 7.248951363844231, 7.17082325881341, 7.0939244104296515, 7.018223355774483, 6.943694279458629, 6.870316523059794, 6.798066235362698, 6.726913927303102, 6.656844572019831, 6.587831142702975, 6.519848569819195, 6.452881402893441, 6.386911023928128, 6.321916755849759, 6.257875062020297, 6.194773152032169, 6.132594760620767, 6.071320118725697, 6.0109294224176235, 5.951397795289828, 5.892715985266672, 5.8348652888824795, 5.777834742845455, 5.721612997437061, 5.666180550102959, 5.611520482074068, 5.5576152586233, 5.5044554344170304, 5.452024374513652, 5.4003122558483545, 5.349305850079876, 5.298993210499051, 5.249360662869793, 5.200394858591663, 5.152081133994656, 5.104410943522941, 5.05737298639524, 5.010954869373652, 4.965149590761396, 4.9199418003102355, 4.87532023869603, 4.831285056760686]
	strikes=75.0:5.0:800.0
	#mcPrices = priceAsianFixedStrike(MonteCarloEngine(true,1024*1024*64),true,strikes,discountFactor,spot,forward,tvar,weights)
	moneyness = strikes./sum(forward.*weights) .- 1.0
	p = plot(xlab="Asian moneyness",ylab="Absolute price error in b.p.")
	for pricerName = ["LB(2,3)","VG1","LB(3,3)","Ju","VG2","D(2,3)","D(3,3)","VG3","VL3"  ]
		prices = priceMap[pricerName]
		plot!(p,moneyness,abs.(prices-pricesMC)/spot.*1e4,label=pricerName,linestyle=:auto)
	end
	plot(p, yscale=:log10)
	plot!(yticks=[0.01,0.1,1.0,10.0])
	plot!(legend=:outerright)
	plot!(size=(800,400),margins=3Plots.mm)
	#non log
	p = plot(xlab="Asian moneyness",ylab="Price error in b.p.")
	for pricerName = ["VG1","VG2", "LB(2,3)","LB(3,3)", "Ju" ]
		prices = priceMap[pricerName]
		plot!(p,moneyness,(prices-pricesMC)/spot.*1e4,label=pricerName,linestyle=:auto)
	end
	#    plot!(p,yticks=[0.01,0.1,1.0,10.0])
	#plot!(p,legend=:outerright)
	plot!(p,size=(400,400),margins=3Plots.mm)

	p2 = plot(xlab="Asian moneyness",ylab="Price error in b.p.")
	for pricerName = ["VG3","VG2" ,"D(2,3)","D(3,3)","VL3" ]
		prices = priceMap[pricerName]
		plot!(p2,moneyness,(prices-pricesMC)/spot.*1e4,label=pricerName,linestyle=:auto)
	end
	#    plot!(p2, yticks=[0.01,0.1,1.0,10.0])
	#    plot!(p2, legend=:outerright)
	plot!(p2, size=(400,400),margins=3Plots.mm)


	=#

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