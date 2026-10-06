import AQFED.Math: norminv
import Random: rand!, randn!, rand, randn
using Statistics
import AQFED.TermStructure:
	LocalVolatilityModel, VarianceSurfaceBySection, localVarianceByLogmoneyness, discountFactor, logForward

function ndims(model::LocalVolatilityModel, specificTimes::Vector{Float64}, timestepSize::Float64)
	genTimes = pathgenTimes(model, specificTimes, timestepSize)
	return (length(genTimes) - 1) #0.0 does not count
end

function simulate(
	rng,
	model::LocalVolatilityModel{S},
	spot::Float64,
	payoff::MCPayoff,
	start::Int,
	nSim::Int,
	timestepSize::Float64;
	withBB = false,
) where {S}
	specTimes = specificTimes(payoff)
	tte = specTimes[end]
	df = discountFactor(model, tte)
	genTimes = pathgenTimes(model, specTimes, timestepSize) #LinRange(0.0, tte, ceil(Int, nSteps * tte) + 1)
	local bb, cache
	if withBB
		cacheSize = ceil(Int, log2(length(genTimes))) * 4
		bb = BrownianBridgeConstruction(genTimes[2:end])
		cache = BBCache{Int, Vector{Float64}}(cacheSize)
	end
    
    currentPathValues = Vector{Float64}(undef, nSim)
	pathValues = Matrix{Float64}(undef, (nSim,length(specTimes)))
	z = Vector{Float64}(undef, nSim)
	t0 = genTimes[1]
	lnspot = log(spot)
	lnf0 = logForward(model, lnspot, t0)
	lnforward = lnf0
	currentPathValues .= lnf0
	local payoffValues
	#println("genTimes ",genTimes)
    iObs=1
	for (dim, t1) in enumerate(genTimes[2:end])
		h = t1 - t0
		sqrth = sqrt(h)
		if withBB
			transformByDim!(bb, rng, start, dim, z, cache)
		else
			skipTo!(rng, dim, start, length(z))
			nextn!(rng, dim, z)
			@. z *= sqrth
		end
		#z = randn(rng, Float64, nSim)
		lnf1 = logForward(model, lnspot, t1)
		lnforward += lnf1 - lnf0
		# indexT0 = searchsortedlast(model.surface.expiries, t0)
		# indexT1 = searchsortedlast(model.surface.expiries, t1)

		# lvh = map(
		#     x -> localVarianceByLogmoneyness(model.surface, x - lnforward, t0, t1, indexT0, indexT1) * h,
		#     logpathValues,
		# )
		logmoneyness = @. currentPathValues - lnforward
		lv = localVarianceByLogmoneyness(model.surface, logmoneyness, t0, t1)
		@. currentPathValues += lnf1 - lnf0 - lv / 2 * h + sqrt(lv) * z
         if abs(genTimes[dim+1] - specTimes[iObs]) < 1e-8
                @. pathValues[:,iObs] = exp(currentPathValues)
                iObs+=1
            end
         if abs(genTimes[dim+1] - tte) < 1e-8      
			payoffValues = map(x -> evaluatePayoffOnPath(payoff, x, df), eachrow(pathValues))
		end
		t0 = t1
		lnf0 = lnf1
	end
	payoffMean = mean(payoffValues)
	return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end

