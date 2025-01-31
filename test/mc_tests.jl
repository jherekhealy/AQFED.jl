using AQFED: AQFED
import AQFED.Random
import AQFED.Random:
	AbstractRNGSeq,
	ZRNGSeq,
	OriginalScramblingRng,
	ScramblingRngAdapter,
	ScrambledSobolSeq,
	Owen, FaureTezuka, NoScrambling

using Random123: Random123
import RandomNumbers: AbstractRNG
import Random: MersenneTwister, rand!
using Statistics
import AQFED.TermStructure:
	SVISection,
	VarianceSurfaceBySection,
	varianceByLogmoneyness,
	HestonModel,
	LocalVolatilityModel,
	ConstantBlackModel,
	TSBlackModel,
	ConstantRateCurve


function estimateError(values::Vector{Float64}, k::Int, rng)
	n = length(values)
	mv = Vector{Float64}(undef, k)
	#stderr =  stdm(values, meanv) / sqrt(length(values))
	for i ∈ 1:k
		indices = (rand(rng, UInt32, 100) .% n) .+ 1
		v = values[indices]
		mv[i] = mean(v)
	end
	stderr = stdm(mv, mean(values)) / sqrt(length(values) / 100)
	return stderr
end
function simulateGBMAnti(rng, nSim, nSteps)
	tte = 1.0
	genTimes = LinRange(0.0, tte, ceil(Int, nSteps * tte) + 1)
	logpayoffValues = Vector{Float64}(undef, nSim * 2)
	logpayoffValues .= 0.0
	t0 = 0.0
	local payoffValues
	for t1 in genTimes[2:end]
		h = t1 - t0
		u = rand(rng, Float64, nSim)
		z = @. AQFED.Math.norminv(u)
		z = vcat(z, -z)
		@. logpayoffValues += -0.5 * h + z * sqrt(h)
		if (t1 == tte)
			payoffValues = @. exp(logpayoffValues)
			payoffValues = @. max(payoffValues, 1 / payoffValues)
		end
		t0 = t1
	end

	payoffMean = mean(payoffValues)
	payoffValuesA = (payoffValues[1:nSim] .+ payoffValues[nSim+1:end]) ./ 2
	return payoffMean, stdm(payoffValuesA, payoffMean) / sqrt(length(payoffValuesA))
end

function simulateGBM(rng, nSim, nSteps)
	tte = 1.0
	genTimes = LinRange(0.0, tte, ceil(Int, nSteps * tte) + 1)
	logpayoffValues = Vector{Float64}(undef, nSim)
	logpayoffValues .= 0.0
	t0 = genTimes[1]
	local payoffValues
	u = Vector{Float64}(undef, nSim)
	@inbounds for j ∈ 2:length(genTimes)
		t1 = genTimes[j]
		h = t1 - t0
		dim = j - 1
		rand!(rng, u)
		z = @. AQFED.Math.norminv(u)
		@. logpayoffValues += -h / 2 + z * sqrt(h)
		if (t1 == tte)
			payoffValues = @. exp(logpayoffValues)
			payoffValues = @. max(payoffValues, 1 / payoffValues)
		end
		t0 = t1
	end
	payoffMean = mean(payoffValues)
	return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end



@testset "Antithetic" begin
	nSteps = 10
	gens = [
		MersenneTwister(201300129),
		AQFED.Random.MersenneTwister64(UInt64(20130129)),
		AQFED.Random.Mixmax17(UInt64(20130129)),
		#        AQFED.Random.Well1024a(UInt32(20130129)),
		Random123.Philox4x(UInt64, (20130129, 20100921), 10),
		AQFED.Random.Chacha8SIMD(),
		AQFED.Random.Blabla8(),
	]
	for rng in gens
		value, mcerr = simulateGBMAnti(rng, 32 * 1024, nSteps)
		println(typeof(rng), " ", value, " ", mcerr)
	end
	gens = [
		MersenneTwister(201300129),
		AQFED.Random.MersenneTwister64(UInt64(20130129)),
		AQFED.Random.Mixmax17(UInt64(20130129)),
		#        AQFED.Random.Well1024a(UInt32(20130129)),
		Random123.Philox4x(UInt64, (20130129, 20100921), 10),
		AQFED.Random.Chacha8SIMD(),
		AQFED.Random.Blabla8(),
	]
	for rng in gens
		value, mcerr = simulateGBM(rng, 32 * 2 * 1024, nSteps)
		println(typeof(rng), " ", value, " ", mcerr)
	end
end

@testset "BlackSim" begin
	model = ConstantBlackModel(1.0, 0.0, 0.0)
	spot = 1.0
	strike = 1.0
	payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(1.0, 1.0, false, 0.0), 0.0)
	refValue = AQFED.Black.blackScholesFormula(true, strike, spot, 1.0, 1.0, 1.0)
	gens = [
		AQFED.Random.MersenneTwister64(UInt64(20130129)),
		AQFED.Random.Mixmax17(UInt64(20130129)),
		AQFED.Random.Well1024a(UInt32(20130129)),
		AQFED.Random.Chacha8SIMD(),
		AQFED.Random.Blabla8(),
		Random123.Philox4x(UInt64, (20130129, 20100921), 10),
	]
	specTimes = AQFED.MonteCarlo.specificTimes(payoff)
	nd = 1
	for rng in gens
		time = @elapsed value = AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng, nd), model, spot, payoff, 64 * 1024)
		println(typeof(rng), " ", value, " ", refValue, " ", value - refValue, " ", time)
		@test isapprox(refValue, value, atol = 1e-2)
	end
end

@testset "FlatLocalVolSim" begin
	spot = 100.0
	strike = 100.0
	payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(10.0, 10.0, false, 0.0), 0.0)
	section = SVISection(0.01, 0.0, 0.0, 0.01, 0.0, 1.0, spot)
	surface = VarianceSurfaceBySection([section], [1.0])
	model = LocalVolatilityModel(surface, 0.0, 0.0)

	refValue = AQFED.Black.blackScholesFormula(true, strike, spot, 0.01 * 10.0, 1.0, 1.0)
	gens = [
		AQFED.Random.MersenneTwister64(UInt64(20130129)),
		AQFED.Random.Mixmax17(UInt64(20130129)),
		AQFED.Random.Well1024a(UInt32(20130129)),
		AQFED.Random.Chacha8SIMD(),
		AQFED.Random.Blabla8(),
		Random123.Philox4x(UInt64, (20130129, 20100921), 10),
	]
	specTimes = AQFED.MonteCarlo.specificTimes(payoff)
	nd = AQFED.MonteCarlo.ndims(model, specTimes, 1.0 / 16)
	for rng in gens
		time = @elapsed value, serr =
			AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng, nd), model, spot, payoff, 0, 1024 * 64, 1.0 / 16)
		println(
			typeof(rng),
			" ",
			value,
			" ",
			refValue,
			" ",
			value - refValue,
			" ",
			serr,
			" ",
			time,
		)
		@test isapprox(refValue, value, atol = serr * 3)
	end
end


@testset "DAXLocalVolSim" begin
	spot = 100.0
	strike = spot
	tte = 1.0
	payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0)
	sections = [
		SVISection(0.030, 0.125, -1.0, 0.050, 0.074, 0.16, spot),
		SVISection(0.032, 0.094, -1.0, 0.041, 0.093, 0.26, spot),
		SVISection(0.028, 0.105, -1.0, 0.072, 0.096, 0.33, spot),
		SVISection(0.026, 0.080, -1.0, 0.098, 0.127, 0.58, spot),
		SVISection(0.026, 0.066, -1.0, 0.113, 0.153, 0.83, spot),
		SVISection(0.031, 0.047, -1.0, 0.065, 0.171, 1.33, spot),
		SVISection(0.037, 0.039, -1.0, 0.030, 0.152, 1.83, spot),
		SVISection(0.036, 0.036, -1.0, 0.083, 0.200, 2.33, spot),
		SVISection(0.038, 0.036, -1.0, 0.139, 0.170, 2.82, spot),
		SVISection(0.034, 0.032, -1.0, 0.199, 0.246, 3.32, spot),
		SVISection(0.044, 0.028, -1.0, 0.069, 0.188, 4.34, spot),
	]

	surface = VarianceSurfaceBySection(
		sections,
		[0.16, 0.26, 0.33, 0.58, 0.83, 1.33, 1.83, 2.33, 2.82, 3.32, 4.34],
	)
	model = LocalVolatilityModel(surface, 0.0, 0.0)

	refValue = AQFED.Black.blackScholesFormula(
		true,
		100.0,
		100.0,
		tte * varianceByLogmoneyness(surface, 0.0, tte),
		1.0,
		1.0,
	)
	gens = [
		AQFED.Random.MersenneTwister64(UInt64(20130129)),
		AQFED.Random.Mixmax17(UInt64(20130129)),
		AQFED.Random.Well1024a(UInt32(20130129)),
		AQFED.Random.Chacha8SIMD(),
		AQFED.Random.Blabla8(),
		Random123.Philox4x(UInt64, (20130129, 20100921), 10),
	]
	specTimes = AQFED.MonteCarlo.specificTimes(payoff)
	nd = AQFED.MonteCarlo.ndims(model, specTimes, 1.0 / 32)
	for rng in gens
		time = @elapsed value, serr =
			AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng, nd), model, spot, payoff, 0, 1024 * 64, 1.0 / 32)
		println(
			typeof(rng),
			" ",
			value,
			" ",
			refValue,
			" ",
			value - refValue,
			" ",
			serr,
			" ",
			time,
		)
		@test isapprox(refValue, value, atol = serr * 5)
	end
end

@testset "HestonSim" begin
	hParams = AQFED.TermStructure.HestonModel(0.04, 0.5, 0.04, -0.9, 1.0, 0.0, 0.0)
	refValue = 13.08467
	gens = [
		MersenneTwister(20130129),
		AQFED.Random.MersenneTwister64(UInt64(20130129)),
		AQFED.Random.Well1024a(UInt32(20130129)),
		AQFED.Random.Mixmax17(UInt64(20130129)),
		Random123.Philox4x(UInt64, (20130129, 20100921), 10),
		AQFED.Random.Chacha8SIMD(),
		AQFED.Random.Blabla8(),
	]
	spot = 100.0
	payoff = AQFED.MonteCarlo.VanillaOption(true, 100.0, AQFED.MonteCarlo.BulletCashFlow(10.0, 10.0, false, 0.0), 0.0)
	timesteps = 8
	specTimes = AQFED.MonteCarlo.specificTimes(payoff)
	ndim = AQFED.MonteCarlo.ndims(hParams, specTimes, 1.0 / timesteps)
	start = 0
	n = 1024 * 64
	for rng in gens
		seq = AbstractRNGSeq(rng, ndim)
		time = @elapsed value, stderror =
			AQFED.MonteCarlo.simulateDVSS2X(seq, hParams, spot, payoff, start, n, 1.0 / timesteps)
		println(
			typeof(rng),
			" ",
			value,
			" ",
			refValue,
			" ",
			value - refValue,
			" ",
			stderror,
			" ",
			time,
		)
		@test isapprox(refValue, value, atol = 3 * stderror)
	end


	seq =
		ScrambledSobolSeq(ndim, n, Owen(30, ScramblingRngAdapter(AQFED.Random.Blabla8())))
	time = @elapsed value, stderrorl = AQFED.MonteCarlo.simulateDVSS2X(
		seq,
		hParams,
		spot,
		payoff,
		start,
		n,
		1.0 / timesteps,
		withBB = false,
	)
	println(
		typeof(seq),
		" ",
		value,
		" ",
		refValue,
		" ",
		value - refValue,
		" ",
		stderrorl,
		" ",
		time,
	)
	@test isapprox(refValue, value, atol = 2 * stderrorl)

	payoff = AQFED.MonteCarlo.VanillaOption(true, 100.0, AQFED.MonteCarlo.BulletCashFlow(1.0, 1.0, false, 0.0), 0.0)
	refValue = 4.4031768153784405
	#for some reasons, FT scheme is really bad on this example => require 1000 steps for a good accuracy
	timesteps = 1000
	specTimes = AQFED.MonteCarlo.specificTimes(payoff)
	ndim = AQFED.MonteCarlo.ndims(hParams, specTimes, 1.0 / timesteps)
	seq = ScrambledSobolSeq(ndim, n, Owen(30, ScramblingRngAdapter(AQFED.Random.Blabla8())))
	#FIXME NoScrambling() leads to NaN 
	time = @elapsed value, stderrorl = AQFED.MonteCarlo.simulateFullTruncation(
		seq,
		hParams,
		spot,
		payoff,
		1,
		n,
		1.0 / timesteps,
		withBB = false,
	)
	println(
		typeof(seq),
		" ",
		value,
		" ",
		refValue,
		" ",
		value - refValue,
		" ",
		stderrorl,
		" ",
		time,
	)
	@test isapprox(refValue, value, atol = stderrorl)

	seq = ScrambledSobolSeq(ndim, n, Owen(30, ScramblingRngAdapter(AQFED.Random.Blabla8())))
	# seq =  ScrambledSobolSeq(ndim, n, NoScrambling()) #FIXME leads to BaB
	time = @elapsed value, stderrorl = AQFED.MonteCarlo.simulateFullTruncation(
		seq,
		hParams,
		spot,
		payoff,
		1,
		n,
		1.0 / timesteps,
		withBB = true,
	)
	println(
		typeof(seq),
		" ",
		value,
		" ",
		refValue,
		" ",
		value - refValue,
		" ",
		stderrorl,
		" ",
		time,
	)
	#@test isapprox(refValue, value, atol = stderrorl/4) #FIXME looks like there is a bug as accuracy is lower

	# ns = [1,2,4,8,16,32,64,128,256,512,1024]
	# vArray = Vector{Float64}(undef, length(ns))
	# for (i,n) in enumerate(ns)
	#     seq = ScrambledSobolSeq(ndims, 1<<29, FaureTezukaScrambling(OriginalScramblingRng()))
	#     value, stderror = AQFED.MonteCarlo.simulateDVSS2X(
	#             seq,
	#             hParams,
	#             payoff,
	#             1,
	#             n*1024,
	#             1.0/timesteps,
	#             withBB = false,
	#         )
	#     vArray[i] = value
	# end
end
# for gen in gens
#        global rng = gen
#        b = @benchmark  value,stderror = AQFED.MonteCarlo.simulateHestonDVSS2X(rng, hParams, payoff,1024*64,8)
#        println(typeof(rng)," ",b)
#        end

using CharFuncPricing

@testset "ZigHeston" begin
	strike = 1.0
	τ = 1.0
	hParams = CharFuncPricing.HestonParams(0.133, 0.35, 0.321, -0.63, 1.388)
	m = 1024
	l = 32
	pricer = makeCosCharFuncPricer(CharFuncPricing.DefaultCharFunc(hParams), τ, m, l)
	#priceEuropean(pricer, false, strike, spot, 1.0)
	n = 1024 * 64
	r = LinRange(0.95, 1.05, 11)
	ve = Vector{Float64}(undef, length(r))
	va = Vector{Float64}(undef, length(r))
	for (i, spot) in enumerate(r)
		#    model = HestonModel(hParams.v0, hParams.κ, hParams.θ, hParams.ρ, hParams.σ, spot, 0.0, 0.0)
		model = AQFED.TermStructure.HestonModel(0.133, 0.35, 0.321, -0.63, 1.388, 0.0, 0.0)
		payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(τ, τ, false, 0.0), 0.0)
		refValue = 0.0 #priceEuropean(pricer, true, strike, spot,τ)
		timesteps = 100
		specTimes = AQFED.MonteCarlo.specificTimes(payoff)
		nd = AQFED.MonteCarlo.ndims(model, specTimes, 1.0 / timesteps)
		rng = MersenneTwister(2020) #AQFED.Random.Blabla8()
		seq = ZRNGSeq(rng, nd)
		# ScrambledSobolSeq(ndims, n, NoScrambling())
		time = @elapsed value, stderror = AQFED.MonteCarlo.simulateFullTruncation(
			seq,
			model,
			spot,
			payoff,
			0,
			n,
			1.0 / timesteps,
			withBB = false, cacheSize = 1000,
		)
		ve[i] = value - refValue
		va[i] = value
		println(spot, " ",
			typeof(rng),
			" ",
			value,
			" ",
			refValue,
			" ",
			value - refValue,
			" ",
			stderror,
			" ",
			time,
		)
	end
end

@testset "DAXSims" begin
	spot = 100.0
	hParams = AQFED.TermStructure.HestonModel(0.04, 0.5, 0.04, -0.9, 1.0, 0.0, 0.0)

	sections = [
		SVISection(0.030, 0.125, -1.0, 0.050, 0.074, 0.16, spot),
		SVISection(0.032, 0.094, -1.0, 0.041, 0.093, 0.26, spot),
		SVISection(0.028, 0.105, -1.0, 0.072, 0.096, 0.33, spot),
		SVISection(0.026, 0.080, -1.0, 0.098, 0.127, 0.58, spot),
		SVISection(0.026, 0.066, -1.0, 0.113, 0.153, 0.83, spot),
		SVISection(0.031, 0.047, -1.0, 0.065, 0.171, 1.33, spot),
		SVISection(0.037, 0.039, -1.0, 0.030, 0.152, 1.83, spot),
		SVISection(0.036, 0.036, -1.0, 0.083, 0.200, 2.33, spot),
		SVISection(0.038, 0.036, -1.0, 0.139, 0.170, 2.82, spot),
		SVISection(0.034, 0.032, -1.0, 0.199, 0.246, 3.32, spot),
		SVISection(0.044, 0.028, -1.0, 0.069, 0.188, 4.34, spot),
	]

	surface = VarianceSurfaceBySection(
		sections,
		[0.16, 0.26, 0.33, 0.58, 0.83, 1.33, 1.83, 2.33, 2.82, 3.32, 4.34],
	)
	gens = [
		AQFED.Random.MersenneTwister64(UInt64(20130129)),
		AQFED.Random.Mixmax17(UInt64(20130129)),
		AQFED.Random.Well1024a(UInt32(20130129)),
		AQFED.Random.Chacha8SIMD(),
		AQFED.Random.Blabla8(),
		Random123.Philox4x(UInt64, (20130129, 20100921), 10),
	]

	ttes = [0.16, 1.0, 4.34]
	strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

	for rng in gens
		time = @elapsed for tte in ttes
			for strike in strikes
				payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0)
				refValue = AQFED.Black.blackScholesFormula(
					true,
					strike,
					100.0,
					tte * varianceByLogmoneyness(surface, 0.0, tte),
					1.0,
					1.0,
				)

				model = TSBlackModel(surface, ConstantRateCurve(0.0), ConstantRateCurve(0.0))
				nd = AQFED.MonteCarlo.ndims(model, AQFED.MonteCarlo.specificTimes(payoff), 100.0)
				value, serr = AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng, nd), model, spot, payoff, 0, 1024 * 64)
				println(
					typeof(rng),
					" ",
					typeof(model),
					" ",
					value,
					" ",
					refValue,
					" ",
					value - refValue,
					" ",
					serr,
				)
			end
		end
		println("elapsed ", time)

		time = @elapsed for tte in ttes
			for strike in strikes
				payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0)
				refValue = AQFED.Black.blackScholesFormula(
					true,
					strike,
					spot,
					tte * varianceByLogmoneyness(surface, log(strike / spot), tte),
					1.0,
					1.0,
				)

				model = LocalVolatilityModel(surface, 0.0, 0.0)
				nd = AQFED.MonteCarlo.ndims(model, AQFED.MonteCarlo.specificTimes(payoff), 100.0)
				value, serr =
					AQFED.MonteCarlo.simulate(AbstractRNGSeq(rng, nd), model, spot, payoff, 0, 1024 * 64, 0.16)
				println(
					typeof(rng),
					" ",
					typeof(model),
					" ",
					value,
					" ",
					refValue,
					" ",
					value - refValue,
					" ",
					serr,
				)
			end
		end
		println(typeof(rng), " elapsed ", time)

		time = @elapsed for tte in ttes
			for strike in strikes
				payoff = AQFED.MonteCarlo.VanillaOption(true, strike, AQFED.MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0)
				refValue = AQFED.Black.blackScholesFormula(
					true,
					strike,
					100.0,
					tte * varianceByLogmoneyness(surface, log(strike / spot), tte),
					1.0,
					1.0,
				)

				nd = AQFED.MonteCarlo.ndims(hParams, AQFED.MonteCarlo.specificTimes(payoff), 100.0)
				value, serr =
					AQFED.MonteCarlo.simulateDVSS2X(AbstractRNGSeq(rng, nd), hParams, spot, payoff, 0, 1024 * 64, 0.16)
				println(
					typeof(rng),
					" ",
					typeof(hParams),
					" ",
					value,
					" ",
					refValue,
					" ",
					value - refValue,
					" ",
					serr,
				)
			end
		end
		println("elapsed ", time)
	end
	#        @test isapprox(refValue, value, atol = serr * 5)
end

using QuadGK,Roots, PPInterpolation

@testset "HestonVVIX" begin
	params = CharFuncPricing.HestonParams(0.04, 0.5, 0.04, -0.9, 1.0)
	hParams = AQFED.TermStructure.HestonModel(0.04, 0.5, 0.04, -0.9, 1.0, 0.0, 0.0)
	gen = AQFED.Random.Chacha8SIMD()
	payoff = AQFED.MonteCarlo.VVIX(hParams)
	timesteps = 365
	specTimes = AQFED.MonteCarlo.specificTimes(payoff)
	nd = AQFED.MonteCarlo.ndims(hParams, specTimes, 1.0 / timesteps)
	start = 0
	n = 1024 * 64
	seq = AbstractRNGSeq(gen, nd)
	g = function (x)
		d = AQFED.VolatilityModels.hestonDensity(params, x)
		-log(sqrt(AQFED.VolatilityModels.integratedVarianceExpectation(params, x))) * 2 * 365 / 30 * d
	end
	gBK = function (x)
		d = AQFED.VolatilityModels.hestonDensityBK(hParams, x)
		-(AQFED.MonteCarlo.logsqrtVix(hParams, x * d[1])) * 2 * 365 / 30 * d[2]
	end
	h = function (x)
		d = AQFED.VolatilityModels.hestonDensity(params, x)
		sqrt(AQFED.VolatilityModels.VIXSquare(params, x)) * d
	end

	# looks like estimate based on f is not reliable ?!? highly dependent on truncation
	# => Not valid with truncation, ok otherwise.
	# MC match call prices/vix future, seems reliable. Could also price options on VIX to imply VVIX.
	#     can we have simple approx? (quadratic variation of VIX)
	#     why is vix future price differnet in PDE? (based on SPX) can we compute vix future from spx analytically (forward density of heston)
	#            looks like atm value is wrong in vvix.
	#     vvix = E[log FutVix] => replication of FutVix, wher  FutVIx = E(sqrt(VIX^2))

	vixIntrinsic = function (strike, signValue, x)
		signValue * (sqrt(AQFED.VolatilityModels.integratedVarianceExpectation(params, x)) - strike)
	end
	vixCall = function (strike, signValue, x)
		d = AQFED.VolatilityModels.hestonDensity(params, x, dt = 30 / 365)
		max(vixIntrinsic(strike, signValue, x), 0) * d
	end
	vixCallBK = function (strike, signValue, x)
		d = AQFED.VolatilityModels.hestonDensityBK(hParams, x, dt = 30 / 365)
		max(vixIntrinsic(strike, signValue, x * d[1]), 0) * d[2]
	end
	payoff = AQFED.MonteCarlo.VVIX(hParams)
	valuemc = AQFED.MonteCarlo.simulateDVSS2X(seq, hParams, 1.0, payoff, start, n, 1.0 / timesteps)
	vixfut = quadgk(x -> vixCall(0.0, 1.0, x), 0.0, 10.0)[1]
	valueRef = quadgk(g, 0.0, 10.0)[1] + 365 / 30 * 2 * log(vixfut)
	println("VVIX Full-Range ", sqrt(valueRef[1]) * 100)
	#vvixApprox1 = 100*sqrt((AQFED.MonteCarlo.VIXSquare(payoff,hParams.v0)-quadgk(h,0.0,10.0)[1]^2)*365/30)/sqrt((AQFED.MonteCarlo.VIXSquare(payoff,hParams.v0)))
	evT = (hParams.v0 - hParams.θ) * exp(-hParams.κ * 30 / 365) + hParams.θ
	evsq = AQFED.MonteCarlo.VIXSquare(hParams, evT)
	evT2 = (hParams.σ^2 / 2hParams.κ + hParams.θ - (hParams.θ - hParams.v0 + hParams.σ^2 / (2hParams.κ)) * exp(-hParams.κ * 30 / 365))^2
	ektm1 = (exp(-hParams.κ * 30 / 365) - 1) / (hParams.κ * 30 / 365)
	ev2 = -2 * ektm1 * evT * (hParams.θ * (ektm1 + 1)) + (hParams.θ * (ektm1 + 1))^2 + ektm1^2 * evT2
	varvix = (ev2 - evsq^2) / 4evsq
	vixapprox = sqrt(log(1 + varvix / AQFED.MonteCarlo.VIXSquare(hParams, evT)) / 30 * 365)
	#we can't truncated this integral, we need to convert to put call first.
	#@test isapprox(valueRef, value, 1.0)
	strikesVIX = range(0.1, 0.5, length = 50)
	refPricesCall = zeros(length(strikesVIX))
	refPricesPut = zeros(length(strikesVIX))
	for (i, strike) ∈ enumerate(strikesVIX)
		lowerBound = if vixIntrinsic(strike, 1.0, eps()) > 0
			eps()
		else
			Roots.find_zero(x -> vixIntrinsic(strike, 1.0, x), (eps(), 10.0), A42())
		end
		refPricesCall[i] = quadgk(x -> vixCall(strike, 1.0, x), lowerBound, 10.0)[1]
		refPricesPut[i] = quadgk(x -> vixCall(strike, -1.0, x), eps(), lowerBound)[1]
	end
	vvix = 0.0
	for (k, strike) ∈ enumerate(strikesVIX)
		dstrike = if k == 1
			strikesVIX[k+1] - strikesVIX[k]
		elseif k == length(strikesVIX)
			strikesVIX[k] - strikesVIX[k-1]
		else
			(strikesVIX[k+1] - strikesVIX[k-1]) / 2
		end
		vvix += ifelse(strikesVIX[k] >= vixfut, refPricesCall[k], refPricesPut[k]) * dstrike / (strike^2)
	end
	vvix = sqrt(vvix * 2 * 365 / 30)
	#sqrt(quadgk(f,0.0,10.5)[1]) is close to vvix especially if strikesVIX[1] = 0.05 instead of 0.1
	#put calll parity not always great with bessel due to mass at 0.
	hParams = HestonModel(0.031, 0.332, 0.076, -0.862, 0.4, 0.0, 0.0) #50%
	hParams = HestonModel(0.035, 0.446, 0.106, -0.735, 0.985, 0.0, 0.0) #200
	hParams = HestonModel(0.039, 3.695, 0.051, -0.751, 2.009, 0.0, 0.0) #180 even though volvol larger.
	hParams = HestonModel(0.026, 0.743, 0.066, -0.724, 0.652, 0.0, 0.0) #174% 
	hParams = HestonModel(0.024, 0.29, 0.082, -0.746, 0.35, 0.0, 0.0) #40%
	hParams = HestonModel(0.051, 0.774, 0.051, -0.911, 0.4, 0.0, 0.0) #15
	hParams = HestonModel(0.053, 0.75, 0.053, -0.782, 0.467, 0.0, 0.0) #38
	hParams = HestonModel(0.053, 1.0, 0.053, -0.779, 0.597, 0.0, 0.0) #100! for relativly small change of kappa and sigma.

	#truncated params
	hParams = AQFED.TermStructure.HestonModel(0.05, 0.49, 0.13, -0.70, 1.56, 0.0, 0.0)
	hParams = AQFED.TermStructure.HestonModel(0.03, 0.75, 0.07, -0.70, 0.65, 0.0, 0.0)
	hParams = AQFED.TermStructure.HestonModel(0.03, 0.29, 0.09, -0.66, 0.40, 0.0, 0.0)

	#vega=100 (forward ts)
	#badcos params = HestonParams{Float64}(0.0252, 0.2895, 0.0861, -0.6619, 0.4019) #rmse = 2.207e-3
	params = HestonParams{Float64}(0.0242, 0.2859, 0.0828, -0.7465, 0.3460)
	#weight=1 (ones)
	#COSBAD  params = HestonParams{Float64}(0.0495, 0.4871, 0.1298, -0.7020, 1.5634) #rmse= 0.0032636853972724165
	params = HestonParams{Float64}(0.0236, 0.2575, 0.0849, -0.7513, 0.315)

	#vega=1e-2
	params = HestonParams{Float64}(0.0371, 3.4490, 0.0497, -0.7558, 1.7522) #rmse= 0.008238985948551988
	#1e-2,kappa=0.75
	params = HestonParams{Float64}(0.0313, 0.7500, 0.0678, -0.7663, 0.7593) # 0.010023447111530193
	#vega=1e-4
	params = HestonParams{Float64}(0.0397, 5.0285, 0.0428, -0.7593, 1.7568) #0.01232436113399774

	hParams = AQFED.TermStructure.HestonModel(params.v0, params.κ, params.θ, params.ρ, params.σ, 0.0, 0.0)

	params = HestonParams{Float64}(0.0371, 3.4490, 0.0497, -0.7558, 1.7522) #rmse= 0.008238985948551988
	params = HestonParams{Float64}(0.0236, 0.2575, 0.0849, -0.7513, 0.315)


	ts = [
		0.0273972602739726,
		0.10410958904109589,
		0.2,
		0.44931506849315067,
		0.6986301369863014,
		0.947945205479452,
		1.1972602739726028,
		1.6931506849315068,
		2.1945205479452055,
		2.6904109589041094,
		3.191780821917808,
		4.189041095890411,
		5.205479452054795,
		6.2027397260273975,
		7.2,
		8.197260273972603,
		9.194520547945206,
	]
	dfs = [0.998656, 0.994938, 0.9905, 0.979911, 0.970367, 0.961542, 0.953184, 0.937349, 0.921779, 0.906637, 0.891596, 0.862471, 0.833291, 0.805002, 0.777273, 0.749989, 0.723404]
	logdfForecastCurve = PPInterpolation.CubicSplineNatural(ts, log.(dfs))

	#sofrcurve
	ts = [
		0.0027397260273972603,
		0.019178082191780823,
		0.038356164383561646,
		0.057534246575342465,
		0.08493150684931507,
		0.16986301369863013,
		0.25205479452054796,
		0.3424657534246575,
		0.4191780821917808,
		0.4986301369863014,
		0.5808219178082191,
		0.6684931506849315,
		0.7479452054794521,
		0.8328767123287671,
		0.9178082191780822,
		1.0,
		1.4986301369863013,
		2.0,
		3.0,
		4.008219178082192,
		5.005479452054795,
		6.002739726027397,
		7.002739726027397,
		8.005479452054795,
		9.013698630136986,
		10.01095890410959,
		12.008219178082191,
		15.016438356164384,
		20.02191780821918,
		25.016438356164382,
		30.019178082191782,
		40.02739726027397,
		50.032876712328765,
	]
	dfs = [
		0.9998655446052758,
		0.9990592658141242,
		0.9981190247355057,
		0.9971796270019692,
		0.995860536384228,
		0.9918793842928559,
		0.9882065552108679,
		0.9843057953051059,
		0.9811316126737677,
		0.977982293434981,
		0.9748020683896175,
		0.971473610055389,
		0.9685778806837968,
		0.9655473398205242,
		0.9625539466969002,
		0.959724969259051,
		0.9433435353852817,
		0.92757331917794,
		0.89692187664775,
		0.8671686254532832,
		0.8383494708781539,
		0.8098040487339867,
		0.7817611882623016,
		0.7541291703258558,
		0.7269915238856094,
		0.7007504513368383,
		0.6498808462271521,
		0.5794733401114949,
		0.48281081406533616,
		0.4118876765643804,
		0.35722809277675743,
		0.28545316076925675,
		0.24132218426777505,
	]
	logdfDiscountCurve = PPInterpolation.CubicSplineNatural(ts, log.(dfs))

	#divcurve
	ts = [
		0.0273972602739726,
		0.10410958904109589,
		0.2,
		0.44931506849315067,
		0.6986301369863014,
		0.947945205479452,
		1.1972602739726028,
		1.6931506849315068,
		2.1945205479452055,
		2.6904109589041094,
		3.191780821917808,
		4.189041095890411,
		5.205479452054795,
		6.2027397260273975,
		7.2,
		8.197260273972603,
		9.194520547945206,
	]
	mids = [0.003913, 0.0039242, 0.0056579, 0.0047029, 0.0051079, 0.0051398, 0.0051334, 0.0048594, 0.0048837, 0.0047432, 0.0046435, 0.004394, 0.0040593, 0.0038011, 0.0036346, 0.0034687, 0.003293]
	divyieldCurve = PPInterpolation.CubicSplineNatural(ts, mids)

	#vol
	ts = [
		0.10410958904109589,
		0.2,
		0.4767123287671233,
		0.6986301369863014,
		0.947945205479452,
		1.1972602739726028,
		1.6931506849315068,
		2.1945205479452055,
		2.6904109589041094,
		3.191780821917808,
		4.189041095890411,
		5.205479452054795,
		6.2027397260273975,
		7.2,
		8.197260273972603,
		9.194520547945206,
	]
	strikes = [
		0.5,
		0.55,
		0.6,
		0.65,
		0.7,
		0.75,
		0.8,
		0.85,
		0.9,
		0.905,
		0.91,
		0.915,
		0.92,
		0.925,
		0.93,
		0.935,
		0.94,
		0.945,
		0.95,
		0.955,
		0.96,
		0.965,
		0.97,
		0.975,
		0.98,
		0.985,
		0.99,
		0.995,
		1.0,
		1.005,
		1.01,
		1.015,
		1.02,
		1.025,
		1.03,
		1.035,
		1.04,
		1.045,
		1.05,
		1.055,
		1.06,
		1.065,
		1.07,
		1.075,
		1.08,
		1.085,
		1.09,
		1.095,
		1.1,
		1.15,
		1.2,
		1.25,
		1.3,
		1.35,
		1.4,
		1.45,
		1.5,
	]
	vols =
		[
			76.6045 70.59 64.2418 57.8388 51.4678 44.9348 38.5177 32.345 26.7839 26.2666 25.7528 25.2422 24.7346 24.2299 23.7281 23.229 22.7323 22.2373 21.7429 21.2468 20.7465 20.2402 19.7273 19.2088 18.6866 18.163 17.6401 17.1204 16.6064 16.1007 15.6066 15.128 14.67 14.238 13.8381 13.4758 13.1556 12.8808 12.6543 12.4797 12.3598 12.2946 12.2807 12.3124 12.3833 12.487 12.618 12.7719 12.9448 15.2002 17.7193 20.1723 22.4814 24.6466 26.6894 28.6293 30.4794;
			64.1937 58.6376 53.2796 48.025 42.8406 37.7493 32.8381 28.3442 24.134 23.7293 23.3264 22.9253 22.5261 22.1293 21.7348 21.3425 20.9517 20.5613 20.1701 19.776 19.3776 18.9741 18.5661 18.1548 17.742 17.3293 16.9185 16.5109 16.1083 15.712 15.3237 14.9453 14.5791 14.2276 13.8938 13.5804 13.2894 13.0217 12.778 12.5596 12.3686 12.2068 12.0741 11.9682 11.8858 11.8238 11.782 11.7631 11.7701 12.8875 14.6584 16.5232 18.3383 20.054 21.6666 23.1914 24.6434;
			50.2098 46.0004 41.9003 38.0821 34.4658 31.0808 27.9027 24.8961 21.9817 21.69 21.398 21.1053 20.8119 20.5177 20.2224 19.9263 19.6294 19.3318 19.0336 18.7353 18.4372 18.1398 17.8434 17.5484 17.255 16.9633 16.6737 16.3866 16.1025 15.8219 15.5452 15.2729 15.0055 14.7436 14.4883 14.2403 14.0006 13.7701 13.5497 13.3407 13.1444 12.9613 12.7906 12.6299 12.4772 12.3315 12.1935 12.0645 11.9459 11.4195 11.7722 12.683 13.7568 14.8547 15.9252 16.9512 17.933;
			43.6288 40.0654 36.7041 33.5602 30.6967 28.0537 25.6223 23.3567 21.16 20.9381 20.7149 20.4902 20.264 20.0366 19.8079 19.578 19.3465 19.1132 18.8777 18.64 18.4007 18.1604 17.92 17.6798 17.44 17.2004 16.961 16.7219 16.4836 16.2466 16.0114 15.7784 15.5479 15.3198 15.0943 14.8719 14.6532 14.4393 14.2311 14.029 13.8331 13.6431 13.4588 13.2803 13.108 12.9424 12.7845 12.6349 12.4939 11.4978 11.2209 11.5075 12.0742 12.7447 13.4517 14.1715 14.8903;
			39.5856 36.5499 33.7399 31.1555 28.8138 26.6755 24.6773 22.7909 20.947 20.7582 20.5687 20.3787 20.1881 19.9966 19.8041 19.6102 19.4148 19.2173 19.0179 18.8165 18.6138 18.4104 18.2068 18.0033 17.7998 17.5961 17.3919 17.1873 16.9825 16.7781 16.5744 16.3716 16.1697 15.9682 15.7667 15.565 15.3635 15.163 14.9648 14.7703 14.5799 14.3939 14.2123 14.0351 13.8625 13.6946 13.5316 13.3736 13.2207 11.999 11.3752 11.244 11.5256 11.9195 12.3931 12.9126 13.4603;
			36.9391 34.281 31.8415 29.6147 27.5898 25.7329 23.9884 22.3451 20.7346 20.5704 20.4053 20.2394 20.0729 19.9058 19.7382 19.5699 19.401 19.2311 19.0602 18.8883 18.7156 18.5425 18.3691 18.1957 18.0222 17.8484 17.6743 17.4999 17.3253 17.1508 16.9767 16.8031 16.6298 16.4565 16.2827 16.108 15.9323 15.7565 15.5817 15.4088 15.2384 15.0707 14.9055 14.7428 14.5825 14.4249 14.2701 14.1184 13.9701 12.6927 11.8415 11.4397 11.4406 11.7352 12.0505 12.4019 12.7906;
			33.6397 31.4781 29.5154 27.7548 26.137 24.5766 23.1837 21.8179 20.4796 20.344 20.2084 20.0729 19.9374 19.8018 19.6659 19.5295 19.3925 19.2547 19.1163 18.9773 18.838 18.6988 18.5598 18.4211 18.2826 18.1441 18.0056 17.867 17.7283 17.5898 17.4516 17.3137 17.1761 17.0387 16.9011 16.7631 16.6245 16.4856 16.3468 16.2086 16.0714 15.9351 15.7998 15.6656 15.5325 15.4006 15.2702 15.1411 15.0136 13.8103 12.7718 12.1131 11.6693 11.505 11.5409 11.6949 11.9168;
			31.2369 29.4357 27.7795 26.2737 24.8539 23.5574 22.5286 21.4173 20.278 20.1811 20.0821 19.9775 19.8648 19.7442 19.6192 19.4943 19.3751 19.2641 19.1618 19.0652 18.9696 18.8717 18.7688 18.6617 18.5506 18.437 18.3219 18.2057 18.0898 17.9735 17.8576 17.7416 17.6253 17.5086 17.391 17.2728 17.1536 17.0341 16.9148 16.7963 16.6793 16.5636 16.4492 16.3357 16.2229 16.1106 15.9985 15.8868 15.7752 14.6849 13.7033 12.7251 12.3162 11.9161 11.6199 11.4919 11.487;
			30.4777 28.9108 27.4503 26.0792 24.7839 23.554 22.3812 21.2588 20.1819 20.0766 19.9717 19.8671 19.763 19.6593 19.556 19.4532 19.3507 19.2486 19.1469 19.0456 18.9447 18.8442 18.7441 18.6444 18.545 18.4461 18.3476 18.2494 18.1517 18.0543 17.9573 17.8607 17.7646 17.6688 17.5734 17.4784 17.3838 17.2896 17.1958 17.1024 17.0095 16.9169 16.8248 16.7331 16.6418 16.5509 16.4605 16.3705 16.281 15.4114 14.5945 13.8415 13.1673 12.5885 12.1198 11.7684 11.5312;
			29.1933 27.6658 26.2467 25.0002 23.9279 22.9794 22.1153 21.2528 20.4 20.3254 20.2466 20.1636 20.0766 19.9857 19.8925 19.8006 19.7098 19.6208 19.5341 19.4485 19.362 19.2739 19.186 19.0968 19.0066 18.9169 18.8282 18.7395 18.6508 18.5634 18.4751 18.3858 18.2963 18.2066 18.1159 18.0245 17.9335 17.8415 17.7489 17.6567 17.565 17.4733 17.3819 17.2914 17.201 17.1108 17.0211 16.9316 16.8422 15.9661 15.1162 14.3682 13.6982 13.0907 12.5727 12.1685 11.8491;
			28.0574 26.9321 25.8858 24.9058 23.9822 23.1073 22.2746 21.4791 20.7163 20.6417 20.5673 20.4933 20.4195 20.346 20.2728 20.1999 20.1272 20.0548 19.9827 19.9108 19.8392 19.7678 19.6967 19.6259 19.5553 19.4849 19.4148 19.3449 19.2753 19.2059 19.1367 19.0678 18.9991 18.9306 18.8624 18.7943 18.7265 18.659 18.5916 18.5245 18.4575 18.3908 18.3243 18.258 18.1919 18.126 18.0603 17.9949 17.9296 17.2875 16.6642 16.0586 15.4703 14.8992 14.3455 13.8103 13.2952;
			27.5021 26.5522 25.6691 24.842 24.0626 23.3241 22.621 21.9489 21.3038 21.2407 21.1778 21.1151 21.0527 20.9905 20.9285 20.8667 20.8052 20.7438 20.6827 20.6218 20.5611 20.5006 20.4403 20.3802 20.3203 20.2606 20.2011 20.1418 20.0827 20.0238 19.965 19.9064 19.848 19.7898 19.7318 19.6739 19.6163 19.5587 19.5014 19.4442 19.3872 19.3303 19.2736 19.2171 19.1607 19.1045 19.0484 18.9925 18.9367 18.3867 17.8498 17.3244 16.8091 16.3025 15.8034 15.3107 14.8231;
			27.3812 26.5356 25.7508 25.0171 24.327 23.6743 23.0543 22.4628 21.8966 21.8413 21.7861 21.7312 21.6765 21.6221 21.5678 21.5137 21.4599 21.4062 21.3528 21.2995 21.2465 21.1936 21.1409 21.0884 21.0362 20.984 20.9321 20.8804 20.8288 20.7774 20.7262 20.6752 20.6244 20.5737 20.5232 20.4728 20.4226 20.3726 20.3227 20.273 20.2235 20.1741 20.1249 20.0758 20.0269 19.9781 19.9295 19.881 19.8326 19.3569 18.8942 18.4432 18.0028 17.5719 17.1496 16.735 16.3273;
			27.4044 26.6413 25.9341 25.2738 24.6538 24.0684 23.5132 22.9845 22.4794 22.43 22.3809 22.332 22.2833 22.2348 22.1864 22.1383 22.0904 22.0426 21.9951 21.9477 21.9005 21.8535 21.8067 21.7601 21.7136 21.6673 21.6212 21.5753 21.5295 21.4839 21.4385 21.3932 21.3481 21.3032 21.2584 21.2137 21.1693 21.1249 21.0808 21.0368 20.9929 20.9492 20.9056 20.8622 20.8189 20.7758 20.7328 20.6899 20.6472 20.2274 19.8202 19.4243 19.0388 18.6628 18.2955 17.9362 17.5842;
			27.4195 26.7073 26.0479 25.4329 24.8559 24.3117 23.7961 23.3057 22.8377 22.792 22.7465 22.7012 22.6561 22.6112 22.5665 22.522 22.4776 22.4335 22.3895 22.3457 22.3021 22.2586 22.2153 22.1722 22.1293 22.0865 22.0439 22.0014 21.9592 21.917 21.8751 21.8333 21.7916 21.7501 21.7088 21.6676 21.6266 21.5857 21.5449 21.5043 21.4639 21.4235 21.3834 21.3433 21.3034 21.2637 21.2241 21.1846 21.1452 20.7588 20.3844 20.021 19.6677 19.3237 18.9883 18.6608 18.3405;
			27.5101 26.8389 26.218 25.6394 25.097 24.5859 24.102 23.6423 23.2038 23.1611 23.1185 23.0761 23.0339 22.9919 22.9501 22.9084 22.8669 22.8256 22.7845 22.7435 22.7027 22.6621 22.6216 22.5813 22.5411 22.5012 22.4613 22.4217 22.3822 22.3428 22.3036 22.2646 22.2257 22.1869 22.1483 22.1098 22.0715 22.0333 21.9953 21.9574 21.9197 21.882 21.8446 21.8072 21.77 21.7329 21.696 21.6591 21.6224 21.2623 20.9138 20.576 20.2479 19.9289 19.6183 19.3153 19.0195
		] ./ 100

	spot = 5751.13
	uForwards = @. exp(-logdfForecastCurve(ts) - divyieldCurve(ts) * ts)
	uPrices, isCall, uWeights = AQFED.VolatilityModels.convertVolsToPricesOTMWeights(ts, uForwards, strikes, vols, vegaFloor = 1e-2)


	sigmaRange = 0.12:0.02:1.4
	result = zeros((length(sigmaRange), 4))
	for (iSigma, sigma) ∈ enumerate(sigmaRange)

		#sParams= HestonParams{Float64}(params.v0,params.κ,params.θ,params.ρ,sigma)
		params, rmse = AQFED.VolatilityModels.calibrateHestonFromPrices(ts, uForwards, strikes, uPrices, isCall, ones(length(ts), length(strikes)), reduction = AQFED.VolatilityModels.SigmaKappaReduction(sigma, 0.0))

		hParams = AQFED.TermStructure.HestonModel(params.v0, params.κ, params.θ, params.ρ, sigma, 0.0, 0.0)

		vixfut = quadgk(x -> vixCall(0.0, 1.0, x), 0.0, 10.0 * sigma)[1]
		valueRef = quadgk(g, 0.0, 10.0 * sigma)[1] + 365 / 30 * 2 * log(vixfut)
		println("VVIX Full-Range ", sqrt(valueRef[1]) * 100)
		#vvixApprox1 = 100*sqrt((AQFED.MonteCarlo.VIXSquare(payoff,hParams.v0)-quadgk(h,0.0,10.0)[1]^2)*365/30)/sqrt((AQFED.MonteCarlo.VIXSquare(payoff,hParams.v0)))
		#we can't truncated this integral, we need to convert to put call first.
		strikesVIX = range(0.1, 0.5, length = 50)
		refPricesCall = zeros(length(strikesVIX))
		refPricesPut = zeros(length(strikesVIX))
		for (i, strike) ∈ enumerate(strikesVIX)
			lowerBound = if vixIntrinsic(strike, 1.0, eps()) > 0
				eps()
			else
				Roots.find_zero(x -> vixIntrinsic(strike, 1.0, x), (eps(), 10.0 * sigma), A42())
			end
			refPricesCall[i] = quadgk(x -> vixCall(strike, 1.0, x), lowerBound, 10.0 * sigma)[1]
			refPricesPut[i] = quadgk(x -> vixCall(strike, -1.0, x), eps(), lowerBound)[1]
		end
		vvix = 0.0
		for (k, strike) ∈ enumerate(strikesVIX)
			dstrike = if k == 1
				strikesVIX[k+1] - strikesVIX[k]
			elseif k == length(strikesVIX)
				strikesVIX[k] - strikesVIX[k-1]
			else
				(strikesVIX[k+1] - strikesVIX[k-1]) / 2
			end
			vvix += ifelse(strikesVIX[k] >= vixfut, refPricesCall[k], refPricesPut[k]) * dstrike / (strike^2)
		end
		vvix = sqrt(vvix * 2 * 365 / 30)
		println(sigma, " ", sqrt(valueRef[1]) * 100, " ", vixapprox * 100, " ", vvix * 100)
		result[iSigma, :] = [sigma, sqrt(valueRef[1]) * 100, vixapprox * 100, vvix * 100]
	end
	#=
	plot(result[:,1] .* 100,result[:,2], label="Log contract",xlab="σ in %", ylab="VVIX estimate")
	plot!(result[:,1] .* 100,result[:,3], label="Approx")
	plot!(result[:,1] .* 100,result[:,4], label="Replication")

	=#
	#FDM
	#SPX at 6000, range = 2500 to 8000
	strikesVIX = range(0.05, 0.5, length = 50)
	strikesSPX = collect(range(0.4, 1.4, length = 50))
	value = Vector{Matrix{Float64}}(undef, length(strikesSPX) * 2)
	for i ∈ eachindex(value)
		value[i] = zeros(400, 100)
	end
	forecastCurve(t) = exp(logdfForecastCurve(t) + divyieldCurve(t) * t) #curve of discount factors
	discountCurve(t) = 1.0

	vPayoff = AQFED.FDM.VIXFDM2DPayoff(strikesSPX, strikesVIX, 60.0 / 365, 30.0 / 365, 1.0 / forecastCurve(60.0 / 365), value, 1.0)
	priceGrid = AQFED.FDM.priceFDM2D(vPayoff, params, 1.0, forecastCurve, discountCurve, 60, method = "RKG2", sDisc = "Exp", exponentialFitting = "Partial Exponential Fitting", Sdev = 6)
	AQFED.FDM.estimateVVIXFull(vPayoff, priceGrid[1], priceGrid[2], 1.0, params.v0)

	priceGrid = AQFED.FDM.priceFDM2D(vPayoff, params, 1.0, forecastCurve, discountCurve, 60, method = "RKG2", sDisc = "Linear", Smax = 3.0, exponentialFitting = "Partial Exponential Fitting")

	value = Vector{Matrix{Float64}}(undef, length(strikesSPX) * 2)
	xSizes = [25, 50, 100, 200, 400]
	tSizes = [16, 30, 60, 120, 240]
	for (xIndex, xSize) ∈ enumerate(xSizes)
		vSize = floor(Int, xSize / 2)
		tSize = tSizes[xIndex]
		for i ∈ eachindex(value)
			value[i] = zeros(xSize, vSize)
		end
		vPayoff = AQFED.FDM.VIXFDM2DPayoff(strikesSPX, strikesVIX, 60.0 / 365, 30.0 / 365, 1.0 / forecastCurve(60.0 / 365), value, 1.0)
		priceGrid = AQFED.FDM.priceFDM2D(vPayoff, params, 1.0, forecastCurve, discountCurve, tSize, method = "RKG2", sDisc = "Exp", exponentialFitting = "Partial Exponential Fitting", Sdev = 6)
		estimate = AQFED.FDM.estimateVVIXFull(vPayoff, priceGrid[1], priceGrid[2], 1.0, params.v0)
		println(xSize, " ", vSize, " ", tSize, " ", estimate)
	end
end
