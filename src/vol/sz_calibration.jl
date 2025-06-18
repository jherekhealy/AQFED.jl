using CharFuncPricing, Random123, LinearAlgebra
using AQFED.Black, AQFED.GlobalOptimization, GaussNewton
import AQFED.Math:
	MQMinTransformation,
	LogisticTransformation,
	inv
export calibrateSchobelZhuFromPrices

function calibrateSchobelZhuFromPrices(
	ts::AbstractVector{T},
	forwards::AbstractVector{T},
	strikes::AbstractMatrix{T},
	prices::AbstractMatrix{T},
	isCall::AbstractMatrix{Bool},
	weights::AbstractMatrix{T};
	isRelative = false,
	method = "Joshi-Yang",
	minimizer = "DE",
	) where {T}
	# uPrices = zeros(T, length(ts),length(strikes))))
	# uWeights = zeros(T, length(ts),length(strikes))
	# for i  = eachindex(forwards)
	# 	@. uPrices[i,:] = prices[i,:] ./ forwards[i]
	# 	@. uWeights[i,:] = weights[i,:] .* forwards[i]
	# end
	uPrices = prices
	uWeights = weights

	function objectiveN(F, x)
		hParams = SchobelZhuParams(x[1], x[2], x[3], x[4], x[5])
		# TX = typeof(x[1])
		# TH = typeof(hParams)
		# cf = DefaultCharFunc{TH,Complex{TX}}(hParams)
		cf = DefaultCharFunc(hParams)

		for (i, t) ∈ enumerate(ts)
			# pricer =JoshiYangCharFuncPricer(cf, t, n=64)
			pricer = if method == "Andersen-Lake"
				CharFuncPricing.ALCharFuncPricer(cf, n = 64)
			elseif method == "Joshi-Yang"
				JoshiYangCharFuncPricer(cf, t, n = 64)
			elseif method == "Cos-128"
				CharFuncPricing.makeCosCharFuncPricer(cf, t, 128, 12)
			elseif method == "Cos"
				CharFuncPricing.makeCosCharFuncPricer(cf, t, 256, 16)
			elseif method == "Cos-Junike"
				CharFuncPricing.makeCosCharFuncPricer(cf, t, tol = 1e-4, maxM = 4096)
			elseif method == "Flinn"
				CharFuncPricing.FlinnCharFuncPricer(cf, t, tTol = 1e-4, qTol = 1e-8)
			elseif method == "Flinn-Transformed"
				#CharFuncPricing.makeCVCharFunc(cf,t,CharFuncPricing.InitialControlVariance())
				CharFuncPricing.AdaptiveFlinnCharFuncPricer(cf, t)
			elseif method == "Swift"
				m, _ = CharFuncPricing.findSwiftScaling(cf, t)
				CharFuncPricing.makeSwiftCharFuncPricer(cf, t, m, 3)
			end
			for (j, strike) ∈ enumerate(strikes[i, :])
				mPrice = CharFuncPricing.priceEuropean(pricer, isCall[i, j], strike / forwards[i], 1.0, t, 1.0) * forwards[i]
				F[i+(j-1)*length(ts)] = if isRelative
					uWeights[i, j] * (mPrice / uPrices[i, j] - 1)
				else
					uWeights[i, j] * (mPrice - uPrices[i, j])
				end
			end
		end
		F
	end
	totalSize = size(prices, 1) * size(prices, 2)
	out = zeros(totalSize)
	function objective1(x)
		objectiveN(out, x)
		norm(out) / norm(uWeights)
	end
	strikeShortIndex = floor(Int, size(strikes, 2) / 2)
	priceShort = uPrices[1, strikeShortIndex]
	volShort = Black.impliedVolatility(isCall[1, strikeShortIndex], priceShort, forwards[1], strikes[1, strikeShortIndex], ts[1], 1.0)
	priceLong = uPrices[end, strikeShortIndex]
	volLong = Black.impliedVolatility(isCall[end, strikeShortIndex], priceLong, forwards[end], strikes[end, strikeShortIndex], ts[end], 1.0)
	x0 = [volShort, 1.0, volLong, -0.5, 0.5] #v0::T, κ::T, θ::T, ρ::T, σ::T)
	lower = [1e-4, 1e-4, 1e-4, -0.99, 1e-2]
	upper = [4.0, 10.0, 4.0, 0.9, 4.0]
	rx0 = x0
	rlower = lower
	rupper =  upper
	rng = Random123.Philox4x(UInt64, (20130129, 20100921), 10)
	problem = GlobalOptimization.Problem(length(rx0), objective1, rlower, rupper)
	result = if minimizer == "DE"
		optim = GlobalOptimization.makeDifferentialEvolutionOptimizer(GlobalOptimization.OptimizerParams(15),
			problem,
			rng, GlobalOptimization.Best1Bin(0.9, 0.5))
		GlobalOptimization.optimize(optim, GlobalOptimization.TerminationCriteria(1000, 100, 1e-6, 1e-5))
	else
		optim = GlobalOptimization.SimulatedAnnealing(problem, rng)
		GlobalOptimization.optimize(optim, rx0)
	end
	println("Result DE ", result, " ", GlobalOptimization.minimizer(optim))
	rx = GlobalOptimization.minimizer(optim) #calibrated params.
	gnRmse = GaussNewton.optimize!(objectiveN, rx, out, autodiff = :single)
	println("Result GN ", gnRmse)
	x =rx
	rmse = norm(uWeights) * GlobalOptimization.minimum(optim) / sqrt(size(prices, 1) * size(prices, 2)) #very close to truncated vol rmse.
	params = SchobelZhuParams(x[1], x[2], x[3], x[4], x[5])
	return params, rmse
end