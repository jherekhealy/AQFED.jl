using AQFED, Test, AQFED.Basket, Printf, StatsBase, AQFED.Spread

@testset "basketequiv" begin
	tte = 1.0
	r = 0.05
	df = exp(-r * tte)
	vols = [0.6, 0.4]
	spots = [40.0, 60.0]
	forwards = spots / df
	correlation = [                   1.0 0.0 0.0;
		0.0 1.0 0.28;
		0.0 0.28 1.0
	]
	weights = [1.0, 1.0]
	strikes = range(70, 130, length = 7)
	prices = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(1), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vcat(0.0, vols .^ 2 .* tte), vcat(-1.0, weights), correlation), strikes)

	bprices = map(strike -> AQFED.Basket.priceEuropean(VorstGeometricExpansion(1), true, strike, df, spots, forwards, vols .^ 2 .* tte, weights, correlation[2:3, 2:3]), strikes)
	@test isapprox(bprices, prices, atol = 1e-8)
	prices = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(3), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vcat(0.0, vols .^ 2 .* tte), vcat(-1.0, weights), correlation), strikes)

	bprices = map(strike -> AQFED.Basket.priceEuropean(VorstGeometricExpansion(3), true, strike, df, spots, forwards, vols .^ 2 .* tte, weights, correlation[2:3, 2:3]), strikes)
	@test isapprox(bprices, prices, atol = 1e-8)
end
@testset "deelstra1" begin
	tte = 1.0
	r = 0.05
	df = exp(-r * tte)
	vols = [0.6, 0.6, 0.0]
	spots = [200.0, 100.0]
	forwards = spots / df
	correlation = [                   1.0 0.28 0.0;
		0.28 1.0 0.0;
		0.0 0.0 1.0
	]
	weights = [-1.0, 1.0, 1.0]
	strikes = range(70, 130, length = 7)
	refPrices = [29.0846, 33.6142, 38.5273, 43.8034, 49.4203, 55.3552, 61.5851]
	pricesICUB = [29.0854,
		33.6150,
		38.5281,
		43.8043,
		49.4212,
		55.3561,
		61.5861]
	pricesSLN = [31.0619,
		35.8096,
		40.8772,
		46.2500,
		51.9127,
		57.8497,
		64.0453]
	pricesMC = AQFED.Basket.priceEuropean(AQFED.Basket.MonteCarloEngine(true, 1024 * 1024), true, -collect(strikes), df, spots, forwards, vols[1:end-1] .^ 2 .* tte, weights[1:end-1], correlation[1:end-1, 1:end-1])
	pricesMC = [29.085362398518818, 33.615021957108674, 38.52814591562596, 43.80427654526876, 49.421235464584036, 55.35612506004627, 61.58606556758308]
	pricesVG0 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(0), true, 1.0, df, vcat(spots, strike), vcat(forwards, strike), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG1 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(1), true, 1.0, df, vcat(spots, strike), vcat(forwards, strike), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG2 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(2), true, 1.0, df, vcat(spots, strike), vcat(forwards, strike), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG3 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(3), true, 1.0, df, vcat(spots, strike), vcat(forwards, strike), vols .^ 2 .* tte, weights, correlation), strikes)

	pricesP = map(strike -> AQFED.Spread.priceVanillaSpread(PearsonPricer(128 * 128), true, -strike, df, spots, forwards, vols[1:2] .^ 2 .* tte, correlation[1, 2]), strikes)

	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.0f & %2.4f & %2.4f & %2.4f & %2.4f\n", strike, pricesVG0[i], pricesVG1[i], pricesVG2[i], pricesMC[i])
	end
	priceMap = Dict(
		"VG0" => pricesVG0,
		"VG1" => pricesVG1,
		"VG2" => pricesVG2,
		"VG3" => pricesVG3,
		"SLN" => pricesSLN,
		"ICUB" => pricesICUB,
		"MC" => pricesMC,
	)
	sortedNames = ["VG0", "VG1", "VG2", "VG3", "SLN", "ICUB", "MC"]
	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.0f ", strike)
		for name ∈ sortedNames
			@printf("& %2.4f", priceMap[name][i])
		end
		@printf("\\\\\n")
	end

	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", rmsd(allPrices, pricesMC))
	end
	@printf("\\\\\n")
	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", maximum(abs.(allPrices - pricesMC)))
	end
end

@testset "deelstra2" begin
	tte = 1.0
	r = 0.05
	df = exp(-r * tte)
	vols = [0.0, 0.17, 0.4]
	spots = [40.0, 100.0]
	forwards = spots / df
	correlation = [                   1.0 0.0 0.0
				0.0 1.0 0.12;
		0.0 0.12 1.0]
	weights = [-1.0, -1.0, 1.0]
	strikes = range(45, 75, length = 7)
	pricesVG0 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(0), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG1 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(1), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG2 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(2), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG3 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(3), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesMC = AQFED.Basket.priceEuropean(AQFED.Basket.MonteCarloEngine(true, 1024 * 1024 * 64), true, collect(strikes), df, spots, forwards, vols[2:end] .^ 2 .* tte, weights[2:end], correlation[2:end, 2:end])

	#very accurate for zeroth order compared to Deelstra paper alternatives, much better than SLN...ia

	refPrices = [24.5981,
		21.8247,
		19.3085,
		17.0391,
		15.0029,
		13.1842,
		11.5664]
	pricesICUB = [24.5975,
		21.8240,
		19.3079,
		17.0384,
		15.0022,
		13.1835,
		11.5656]
	pricesSLN = [24.6096,
		21.8441,
		19.3342,
		17.0692,
		15.0355,
		13.2178,
		11.5997]
	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.0f & %2.4f & %2.4f & %2.4f & %2.4f\n", strike, pricesVG0[i], pricesVG1[i], pricesVG2[i], pricesMC[i])
	end
	priceMap = Dict(
		"VG0" => pricesVG0,
		"VG1" => pricesVG1,
		"VG2" => pricesVG2,
		"VG3" => pricesVG3,
		"SLN" => pricesSLN,
		"ICUB" => pricesICUB,
		"MC" => pricesMC,
	)
	sortedNames = ["VG0", "VG1", "VG2", "VG3", "SLN", "ICUB", "MC"]
	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.0f ", strike)
		for name ∈ sortedNames
			@printf("& %2.4f", priceMap[name][i])
		end
		@printf("\\\\\n")
	end

	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", rmsd(allPrices, pricesMC))
	end
	@printf("\\\\\n")
	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", maximum(abs.(allPrices - pricesMC)))
	end
end


@testset "deelstra4" begin
	tte = 1.0
	r = 0.05
	df = exp(-r * tte)
	vols = [0.0, 0.3, 0.22, 0.4]
	spots = [46.0, 24.0, 100.0]
	forwards = spots / df
	correlation =
		[ 1.0 0.0 0.0 0.0;
			 0.0 1.0 0.41 0.91;
			0.0 0.41 1.0 0.17;
			0.0 0.91 0.17 1.0]
	weights = [-1.0, -1.0, -1.0, 1.0]
	strikes = range(15, 45, length = 7)
	pricesVG0 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(0), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG1 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(1), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG2 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(2), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG3 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(3), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesMC = map(strike -> AQFED.Basket.priceEuropean(AQFED.Basket.MonteCarloEngine(), true, strike, df, spots, forwards, vols[2:end] .^ 2 .* tte, weights[2:end], correlation[2:end, 2:end]), strikes)
	#pricesMC =AQFED.Basket.priceEuropean(AQFED.Basket.MonteCarloEngine(true,1024*1024*64),true,collect(strikes),df,spots,forwards,vols[2:end].^2 .* tte,weights[2:end],correlation[2:end,2:end])
	pricesMC = [19.6855714365819, 16.705681889157482, 14.101559154974396, 11.85245731863325, 9.928534342181516, 8.295444779034915, 6.917724246702291]
	refPrices = [19.6849,
		16.7051,
		14.1010,
		11.8519,
		9.9281,
		8.2951,
		6.9174]
	pricesHICUB = [19.5231,
		16.5673,
		13.9944,
		11.7790,
		9.8876,
		8.2837,
		6.9305]
	pricesSLN = [19.6925,
		16.7345,
		14.1460,
		11.9059,
		9.9851,
		8.3506,
		6.9683]
	pricesEBS = [19.6816, 16.7009, 14.0961, 11.8466, 9.9226, 8.2897, 6.9122]
	#also in pellegrino table2

	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.0f & %2.4f & %2.4f & %2.4f & %2.4f\n", strike, pricesVG0[i], pricesVG1[i], pricesVG2[i], refPrices[i])
	end
	priceMap = Dict(
		"VG0" => pricesVG0,
		"VG1" => pricesVG1,
		"VG2" => pricesVG2,
		"VG3" => pricesVG3,
		"SLN" => pricesSLN,
		"HICUB" => pricesHICUB,
		"EBS" => pricesEBS,
		"MC" => pricesMC,
	)
	sortedNames = ["VG0", "VG1", "VG2", "VG3", "SLN", "HICUB", "MC"]
	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.0f ", strike)
		for name ∈ sortedNames
			@printf("& %2.4f", priceMap[name][i])
		end
		@printf("\\\\\n")
	end

	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", rmsd(allPrices, pricesMC))
	end
	@printf("\\\\\n")
	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", maximum(abs.(allPrices - pricesMC)))
	end

end
@testset "deelstra7" begin
	tte = 1.0
	r = 0.05
	df = exp(-r * tte)
	vols = [0.0, 0.63, 0.34, 0.21]
	spots = [12.0, 63.0, 100.0]
	forwards = spots / df
	correlation =
		[ 1.0 0.0 0.0 0.0;
			 0.0 1.0 0.43 0.3;
			0.0 0.43 1.0 0.87;
			0.0 0.3 0.87 1.0]
	weights = [-1.0, -1.0, -1.0, 1.0]
	strikes = range(2.5, 47.5, length = 7)
	pricesVG0 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(0), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG1 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(1), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG2 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(2), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesVG3 = map(strike -> AQFED.Basket.priceEuropeanSpread(VorstGeometricExpansion(3), true, 1.0, df, vcat(strike, spots), vcat(strike, forwards), vols .^ 2 .* tte, weights, correlation), strikes)
	pricesMC = AQFED.Basket.priceEuropean(AQFED.Basket.MonteCarloEngine(true, 1024 * 1024), true, collect(strikes), df, spots, forwards, vols[2:end] .^ 2 .* tte, weights[2:end], correlation[2:end, 2:end])
	pricesMC = [23.593830888360618, 17.20630104828357, 11.411244969717874, 6.602334384575033, 3.187716119759704, 1.2517911317586092, 0.40240907171675183]

	refPrices = [23.5925,
		17.2049,
		11.4099,
		6.6009,
		3.1872,
		1.2518,
		0.4026]
	pricesHICUB = [23.5138,
		17.1373,
		11.3873,
		6.6584,
		3.3147,
		1.3853,
		0.4913]
	pricesSLN = [23.1681,
		16.8591,
		11.3394,
		6.9203,
		3.7629,
		1.7925,
		0.7369]
	#also in pellegrino table3

	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.0f & %2.4f & %2.4f & %2.4f & %2.4f\n", strike, pricesVG0[i], pricesVG1[i], pricesVG2[i], pricesMC[i])
	end
	priceMap = Dict(
		"VG0" => pricesVG0,
		"VG1" => pricesVG1,
		"VG2" => pricesVG2,
		"VG3" => pricesVG3,
		"SLN" => pricesSLN,
		"HICUB" => pricesHICUB,
		"MC" => pricesMC,
	)
	sortedNames = ["VG0", "VG1", "VG2", "VG3", "SLN", "HICUB", "MC"]
	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.1f ", strike)
		for name ∈ sortedNames
			@printf("& %2.4f", priceMap[name][i])
		end
		@printf("\\\\\n")
	end

	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", rmsd(allPrices, pricesMC))
	end
	@printf("\\\\\n")
	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", maximum(abs.(allPrices - pricesMC)))
	end

end

@testset "deelstra8" begin
	tte = 1.0
	r = 0.05
	df = exp(-r * tte)
	vols = [0.25, 0.33]
	spots = [60.0, 100.0]
	forwards = spots / df
	correlation =
		[ 1.0 0.4;
			0.4 1.0]
	t = reverse(tte .- collect(0:29) ./ 365)
	forwardT = @. exp(r * t)
	weightsT = ones(length(t)) / length(t)
	weights = [-1, 1] * weightsT'
	forwards = spots * forwardT'
	tvar = vols .^ 2 * t'
	strikes = range(25, 55, length = 7)
	pricesVG0 = map(strike -> AQFED.Asian.priceAsianBasketSpread(VorstGeometricExpansion(0), true, strike, df, spots, forwards, tvar, weights, correlation, t), strikes)
	pricesVG1 = map(strike -> AQFED.Asian.priceAsianBasketSpread(VorstGeometricExpansion(1), true, strike, df, spots, forwards, tvar, weights, correlation, t), strikes)
	pricesVG2 = map(strike -> AQFED.Asian.priceAsianBasketSpread(VorstGeometricExpansion(2), true, strike, df, spots, forwards, tvar, weights, correlation, t), strikes)
	pricesVG3 = map(strike -> AQFED.Asian.priceAsianBasketSpread(VorstGeometricExpansion(3), true, strike, df, spots, forwards, tvar, weights, correlation, t), strikes)
	#pricesMC =AQFED.Basket.priceEuropean(AQFED.Basket.MonteCarloEngine(true,1024*1024),true,collect(strikes),df,spots,forwards,vols[2:end].^2 .* tte,weights[2:end],correlation[2:end,2:end])
	#=
	engine = AQFED.Basket.MonteCarloRngEngine(AQFED.Basket.MonteCarloEngine(true,1024*1024*16),AQFED.Random.DigitalSobolSeq(1,1,AQFED.Random.Chacha8SIMD(UInt32)),false)
	pricesMC = AQFED.Asian.priceAsianBasketFixedStrike(engine,true, strikes, df, spots, forwards, tvar, weights, correlation,t)
	pricesMC2 = AQFED.Asian.priceAsianBasketFixedStrike(engine,true, strikes, df, spots, forwards, tvar, weights, correlation,t,start=1024*1024*16+1)
	println((pricesMC2+pricesMC) ./ 2)

	=#
	pricesMC = [20.76459253500918, 17.693059486225877, 14.958904244298235, 12.557657740412854, 10.474462580454961, 8.686963398030855, 7.168256041889652]

	refPrices = [20.7645, 17.6931, 14.9591, 12.5579, 10.4747, 8.6873, 7.1685]
	pricesHICUB = [20.7637,
		17.6921,
		14.9580,
		12.5566,
		10.4734,
		8.6859,
		7.1671]
	pricesSLN = [20.8073,
		17.7711,
		15.0630,
		12.6769,
		10.5983,
		8.8065,
		7.2767]

	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.0f & %2.4f & %2.4f & %2.4f & %2.4f\n", strike, pricesVG0[i], pricesVG1[i], pricesVG2[i], refPrices[i])
	end
	priceMap = Dict(
		"VG0" => pricesVG0,
		"VG1" => pricesVG1,
		"VG2" => pricesVG2,
		"VG3" => pricesVG3,
		"SLN" => pricesSLN,
		"HICUB" => pricesHICUB,
		"MC" => pricesMC,
	)
	sortedNames = ["VG0", "VG1", "VG2", "VG3", "SLN", "HICUB", "MC"]
	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.1f ", strike)
		for name ∈ sortedNames
			@printf("& %2.4f", priceMap[name][i])
		end
		@printf("\\\\\n")
	end

	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", rmsd(allPrices, pricesMC))
	end
	@printf("\\\\\n")
	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", maximum(abs.(allPrices - pricesMC)))
	end
end

@testset "deelstra10" begin
	tte = 1.0
	r = 0.05
	df = exp(-r * tte)
	vols = [0.25, 0.3, 0.35]
	spots = [25, 50.0, 100.0]
	forwards = spots / df
	correlation =
		[ 1.0 0.7 0.8;
			0.7 1.0 0.3;
			0.8 0.3 1.0]
	t = reverse(tte .- collect(0:29) ./ 365)
	forwardT = @. exp(r * t)
	weightsT = ones(length(t)) / length(t)
	weights = [-1, -1, 1] * weightsT'
	forwards = spots * forwardT'
	tvar = vols .^ 2 * t'
	strikes = range(10, 40, length = 7)
	pricesVG0 = map(strike -> AQFED.Asian.priceAsianBasketSpread(VorstGeometricExpansion(0), true, strike, df, spots, forwards, tvar, weights, correlation, t), strikes)
	pricesVG1 = map(strike -> AQFED.Asian.priceAsianBasketSpread(VorstGeometricExpansion(1), true, strike, df, spots, forwards, tvar, weights, correlation, t), strikes)
	pricesVG2 = map(strike -> AQFED.Asian.priceAsianBasketSpread(VorstGeometricExpansion(2), true, strike, df, spots, forwards, tvar, weights, correlation, t), strikes)
	pricesVG3 = map(strike -> AQFED.Asian.priceAsianBasketSpread(VorstGeometricExpansion(3), true, strike, df, spots, forwards, tvar, weights, correlation, t), strikes)
	#pricesMC = map( strike -> AQFED.Asian.priceAsianBasketSpread(AQFED.Basket.MonteCarloEngine(true,1024*1024), true, strike, df, spots,forwards, tvar, weights, correlation,t),strikes)
	#=
	engine = AQFED.Basket.MonteCarloRngEngine(AQFED.Basket.MonteCarloEngine(true,1024*1024*16),AQFED.Random.DigitalSobolSeq(1,1,AQFED.Random.Chacha8SIMD(UInt32)),false)
	pricesMC = AQFED.Asian.priceAsianBasketFixedStrike(engine,true, strikes, df, spots, forwards, tvar, weights, correlation,t)

	pricesMCArray = zeros(length(strikes),64)
	for (j,strike) = enumerate(strikes)
		engine.isRngInitialized = false
		println("Strike ",strike)
		for i=1:size(pricesMCArray,2)
			pricesMCArray[j,i] = AQFED.Asian.priceAsianBasketSpread(engine,true, strike, df, spots,forwards, tvar, weights, correlation,t)
			println(i, " done")
		end
	end
	pricesMC = [mean(pricesMCArray[j,:]) for j =1:length(strikes))
	=#
	pricesMC = [20.601024126065475, 17.55455104105075, 14.840911729074312, 12.457439846969944, 10.390615462615724, 8.618557712452333, 7.114494946611764]
	pricesMC = [20.601019916119036, 17.554547034935858, 14.84090800644788, 12.457436407831972, 10.390612306966212, 8.618554840291281, 7.114492357939171]


	refPrices = [20.6014, 17.5549, 14.8411, 12.4577, 10.3907, 8.6186, 7.1146]
	pricesHICUB = [20.5521,
		17.5209,
		14.8236,
		12.4559,
		10.4030,
		8.6425,
		7.1472]
	pricesSLN = [20.7157,
		17.7346,
		15.0677,
		12.7097,
		10.6478,
		8.8635,
		7.3342]
	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.0f & %2.4f & %2.4f & %2.4f & %2.4f\n", strike, pricesVG0[i], pricesVG1[i], pricesVG2[i], refPrices[i])
	end
	priceMap = Dict(
		"VG0" => pricesVG0,
		"VG1" => pricesVG1,
		"VG2" => pricesVG2,
		"VG3" => pricesVG3,
		"SLN" => pricesSLN,
		"HICUB" => pricesHICUB,
		"MC" => pricesMC,
	)
	sortedNames = ["VG0", "VG1", "VG2", "VG3", "SLN", "HICUB", "MC"]
	for (i, strike) ∈ enumerate(strikes)
		@printf("%2.1f ", strike)
		for name ∈ sortedNames
			@printf("& %2.4f", priceMap[name][i])
		end
		@printf("\\\\\n")
	end

	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", rmsd(allPrices, pricesMC))
	end
	@printf("\\\\\n")
	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.4f ", maximum(abs.(allPrices - pricesMC)))
	end
end
@testset "krekel-asianspread-table1" begin
	tk = collect(1:26) ./ 52
	ts = 4.5 .+ tk
	spot = 100.0
	r = 0.05
	q = 0.01
	wk = -ones(26) ./ 26
	ws = ones(26) ./ 26
	strikePct = [0.5, 0.8, 1.0, 1.2, 1.5]
	discountFactor = exp(-r * ts[end])
	t = vcat(tk, ts)
	weights = vcat(wk, ws)
	forwards = spot .* exp.((r - q) .* t)
	vol = 0.1
	totalVariances = vol^2 .* t
	pricesVG0 = map(alpha -> AQFED.Asian.priceAsianSpread(VorstGeometricExpansion(0), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)
	refPrices = [54.87, 31.42, 17.43, 7.71, 1.55]
	@test isapprox(refPrices, prices, atol = 1e-2)
	vol = 0.7
	totalVariances = vol^2 .* t
	pricesVG0 = map(alpha -> AQFED.Asian.priceAsianSpread(VorstGeometricExpansion(0), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)
	pricesVG1 = map(alpha -> AQFED.Asian.priceAsianSpread(VorstGeometricExpansion(1), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)
	pricesVG2 = map(alpha -> AQFED.Asian.priceAsianSpread(VorstGeometricExpansion(2), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)
	pricesVG3 = map(alpha -> AQFED.Asian.priceAsianSpread(VorstGeometricExpansion(3), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)
	pricesP = map(alpha -> AQFED.Spread.priceAsianSpread(PearsonPricer(128), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)

	refPricesKrekel = [67.89, 58.87, 54.18, 50.2, 45.34]
	refPricesMC = [67.8833509197478, 58.87914722291624, 54.19185629905814, 50.2295386581811, 45.287395148927324]
	mcPrices = [67.88261293500662, 58.878399430968976, 54.191104986995065, 50.22873895748191, 45.28669840592715] #16*16randomized
	mcPrices = [67.88347192631099, 58.87926498529496, 54.191997805787715, 50.22965402510399, 45.28752377676765]
    @test isapprox(refPricesKrekel, mcPrices, atol=0.08)
	#[67.88368113180974, 58.879456484973524, 54.19219734454468, 50.229876276493194, 45.2877539562472] #64x16 digital scramble


	# on 64*16M paths
	#=
	engine = AQFED.Basket.MonteCarloRngEngine(AQFED.Basket.MonteCarloEngine(true,1024*1024*16),AQFED.Random.DigitalSobolSeq(1,1,AQFED.Random.Chacha8SIMD(UInt32)),false)
	mcPricesM = zeros(16,length(strikePct))
	for i=1:size(mcPricesM,1)
	mcPricesM[i,:] = AQFED.Asian.priceAsianSpread(engine,true, strikePct, discountFactor,spot,forwards,totalVariances,weights,start=1)
	AQFED.Random.rand!(engine.rng.rng,engine.rng.upoint)
	println(i," done")
	end
	mcPrices = vec(mean(mcPricesM,dims=1))

	#other way:
	 engine = AQFED.Basket.MonteCarloRngEngine(AQFED.Basket.MonteCarloEngine(true,1024*1024*16),AQFED.Random.DigitalSobolSeq(1,1,AQFED.Random.Chacha8SIMD(UInt32)),false)
	mcPricesM = zeros(64,length(strikePct))
	for i=1:size(mcPricesM,1)
	mcPricesM[i,:] = AQFED.Asian.priceAsianSpread(engine,true, strikePct, discountFactor,spot,forwards,totalVariances,weights,start=1024*1024*16*(i-1)+1)
	println(i," done")
	end
	mcPrices = vec(mean(mcPricesM,dims=1))
	#[67.88347192631099, 58.87926498529496, 54.191997805787715, 50.22965402510399, 45.28752377676765]
	=#


	priceMap = Dict(
		"VG0" => pricesVG0,
		"VG1" => pricesVG1,
		"VG2" => pricesVG2,
		"VG3" => pricesVG3,
		"MM" => pricesP,
		"MC" => mcPrices,
	)
	sortedNames = ["VG0", "VG1", "VG2", "VG3", "MM", "MC"]
	for (i, strike) ∈ enumerate(strikePct)
		@printf("%2.1f ", strike)
		for name ∈ sortedNames
			@printf("& %2.5f", priceMap[name][i])
		end
		@printf("\\\\\n")
	end

	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.5f ", rmsd(allPrices, mcPrices))
	end
	@printf("\\\\\n")
	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.5f ", maximum(abs.(allPrices - mcPrices)))
	end

	@test isapprox(refPricesMC, prices, atol = 2e-2) #more accurate than krekel geo adj. Our "strike adj" is better than Vorst additive adj.
	@test isapprox(refPricesMC, pricesVG2, atol = 6e-3) # difficult to estimate higher accuracy due to MC limits.
end

@testset "krekel-asianspread-table2" begin
	t0 = 0.25
	spot = 100.0
	r = 0.05
	q = 0.01
	wk = -ones(26) ./ 26
	ws = ones(26) ./ 26
	strikePct = [0.5, 0.8, 1.0, 1.2, 1.5]
	tk = collect(1:26) ./ 52
	ts = 4.5 .+ tk
	t = vcat(tk, ts) .- 0.25
	weights = vcat(wk, ws)
	forwards = map(t -> ifelse(t <= 0, spot * exp((r - q) * (t + 0.25)), spot * exp((r - q) * (t + 0.25))), t)
	t = max.(t, 0.0)
	discountFactor = exp(-r * t[end])
	vol = 0.1
	totalVariances = vol^2 .* t
	pricesVG0 = map(alpha -> AQFED.Asian.priceAsianSpread(VorstGeometricExpansion(0), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)
	refPrices = [54.87, 31.42, 17.43, 7.71, 1.55]
	@test isapprox(refPrices, prices, atol = 1e-2)
	vol = 0.7
	totalVariances = vol^2 .* t
	pricesVG0 = map(alpha -> AQFED.Asian.priceAsianSpread(VorstGeometricExpansion(0), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)
	pricesVG1 = map(alpha -> AQFED.Asian.priceAsianSpread(VorstGeometricExpansion(1), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)
	pricesVG2 = map(alpha -> AQFED.Asian.priceAsianSpread(VorstGeometricExpansion(2), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)
	pricesVG3 = map(alpha -> AQFED.Asian.priceAsianSpread(VorstGeometricExpansion(3), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)
	pricesP = map(alpha -> AQFED.Spread.priceAsianSpread(PearsonPricer(128 * 128), true, alpha, discountFactor, spot, forwards, totalVariances, weights), strikePct)
	refPricesKrekel = [68.73,
		59.53,
		54.80,
		50.76,
		45.74]
	mcPrices = [68.68810112830643, 59.54899358904961, 54.79239997657148, 50.772226420808174, 45.75917061748162]
	@test isapprox(refPricesKrekel, mcPrices, atol=0.06)


	# on 64*16M paths
	#=
	engine = AQFED.Basket.MonteCarloRngEngine(AQFED.Basket.MonteCarloEngine(true,1024*1024*16),AQFED.Random.DigitalSobolSeq(1,1,AQFED.Random.Chacha8SIMD(UInt32)),false)
	mcPricesM = zeros(16,length(strikePct))
	for i=1:size(mcPricesM,1)
	mcPricesM[i,:] = AQFED.Asian.priceAsianSpread(engine,true, strikePct, discountFactor,spot,forwards,totalVariances,weights,start=1)
	AQFED.Random.rand!(engine.rng.rng,engine.rng.upoint)
	println(i," done")
	end
	mcPrices = vec(mean(mcPricesM,dims=1))

	#other way:
	 engine = AQFED.Basket.MonteCarloRngEngine(AQFED.Basket.MonteCarloEngine(true,1024*1024*16),AQFED.Random.DigitalSobolSeq(1,1,AQFED.Random.Chacha8SIMD(UInt32)),false)
	mcPricesM = zeros(64,length(strikePct))
	for i=1:size(mcPricesM,1)
	mcPricesM[i,:] = AQFED.Asian.priceAsianSpread(engine,true, strikePct, discountFactor,spot,forwards,totalVariances,weights,start=1024*1024*16*(i-1)+1)
	println(i," done")
	end
	mcPrices = vec(mean(mcPricesM,dims=1))
	#[67.88347192631099, 58.87926498529496, 54.191997805787715, 50.22965402510399, 45.28752377676765]
	=#

	pricesMM = [68.70,
		59.56,
		54.81,
		50.79,
		45.78]

	priceMap = Dict(
		"VG0" => pricesVG0,
		"VG1" => pricesVG1,
		"VG2" => pricesVG2,
		"VG3" => pricesVG3,
		"MM" => pricesP,
		"MC" => mcPrices,
	)
	sortedNames = ["VG0", "VG1", "VG2", "VG3", "MM", "MC"]
	for (i, strike) ∈ enumerate(strikePct)
		@printf("%2.1f ", strike)
		for name ∈ sortedNames
			@printf("& %2.5f", priceMap[name][i])
		end
		@printf("\\\\\n")
	end

	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.5f ", rmsd(allPrices, mcPrices))
	end
	@printf("\\\\\n")
	for name ∈ sortedNames
		allPrices = priceMap[name]
		@printf("& %1.5f ", maximum(abs.(allPrices - mcPrices)))
	end

	@test isapprox(refPricesMC, prices, atol = 2e-2) #more accurate than krekel geo adj. Our "strike adj" is better than Vorst additive adj.
	@test isapprox(refPricesMC, pricesVG2, atol = 6e-3) # difficult to estimate higher accuracy due to MC limits.
end
