using COSMO, Convex
using GoldfarbIdnaniSolver
using SparseArrays, LinearAlgebra

export filterConvexPrices, isArbitrageFree

function isArbitrageFree(strikes::Vector{T}, callPrices::Vector{T}, forward::T)::Tuple{Bool, Int} where {T}
	for (i,(xi, yi)) in enumerate(zip(strikes, callPrices))
		if yi < max(forward - xi, 0)
			return (false, i)
		end
	end

	for i ∈ 2:length(callPrices)-1
		s0 = (callPrices[i] - callPrices[i-1]) / (strikes[i] - strikes[i-1])
		s1 = (callPrices[i] - callPrices[i+1]) / (strikes[i] - strikes[i+1])
		if s0 <= -1
			return (false, i)
		end
		if s0 >= s1
			println("s0 > s1")
			return (false, i)
		end
		if s1 >= 0
			return (false, i + 1)
		end
	end

	return (true, -1)
end

function searchsortednearest(a, x)
	idx = searchsortedfirst(a, x)
	if (idx == 1)
		return idx
	end
	if (idx > length(a))
		return length(a)
	end
	if (a[idx] == x)
		return idx
	end
	if (abs(a[idx] - x) < abs(a[idx-1] - x))
		return idx
	else
		return idx - 1
	end
end

using PPInterpolation
function filterConvexPricesFast(
	strikes::Vector{T},
	callPrices::Vector{T}, #undiscounted!
	forward::T;
	lambda = 0.01, weights = abs.((strikes .- forward) / forward),
	adjust = true,
)::Tuple{Vector{T}, Vector{T}, Vector{T}} where {T}
	set = [0.0]
	setReject = []
	pricesReject = []
	setPrices = [forward]
    sizehint!(setPrices, length(strikes)+1)
	indices = sortperm(weights)
	for i ∈ indices
		accepted = false
        k = 1
		if length(set) == 1
			if callPrices[i] > max(forward - strikes[i], 0) && callPrices[i] < forward
				accepted = true
			end
		else #find strike in set immediately before and immediately after strikes[i]
			# max(-1, Ckp-Ck / kp -k) < ckp-ci / kp-i <Ci-Ck / i-k < min(Ck-ckm / k-km, 0)
			k = searchsortedlast(set, strikes[i])
			#can not be zero since there is always strike=0.0 before
			slope = (callPrices[i] - setPrices[k]) / (strikes[i] - set[k])
			# println(slope, " ",strikes[i]," ",set[k])
			if k == 1
				#length = 2 so k+1 is in set.
				#println("check k=1, slopetest=",(setPrices[k+1] - callPrices[i]) / (set[k+1] - strikes[i]))
				if (slope > -1) && (slope < (setPrices[k+1] - callPrices[i]) / (set[k+1] - strikes[i]))
					if k + 2 <= length(set)
						accepted = (setPrices[k+1] - callPrices[i]) / (set[k+1] - strikes[i]) < (setPrices[k+2] - setPrices[k+1]) / (set[k+2] - set[k+1])
					else
						accepted = true
					end
				end
			else
				if (slope > (setPrices[k] - setPrices[k-1]) / (set[k] - set[k-1]))
					#lb ok, check ub
					if k == length(set)
						if slope < 0
							accepted = true
						end
					elseif slope < (setPrices[k+1] - callPrices[i]) / (set[k+1] - strikes[i])
						if k + 2 <= length(set)
							accepted = (setPrices[k+1] - callPrices[i]) / (set[k+1] - strikes[i]) < (setPrices[k+2] - setPrices[k+1]) / (set[k+2] - set[k+1])
						else
							accepted = true
						end
					end
				end
			end
		end
		if accepted
            insert!(set, k+1, strikes[i])
            insert!(setPrices,k+1, callPrices[i])
		else
			append!(setReject, strikes[i])
			append!(pricesReject, callPrices[i])
		end
	end
	if adjust
		for (i, strike) ∈ enumerate(setReject) #already sorted properly (by liquidity)
			k = searchsortedlast(set, strike)
			#slope = (callPrices[i] - setPrices[k]) / (strikes[i] - set[k])
			if k == 1
				#make slope closest to -1 FIXME slope closest to arb broken bound
				slopeSet = (setPrices[k+1] - setPrices[k]) / (set[k+1] - set[k])
				slopeSetp = if k + 2 <= length(set)
					(setPrices[k+2] - setPrices[k+1]) / (set[k+2] - set[k+1])
				else
					0.0
				end
				# slopeSetp > ck+1 - c / sk+1 - s > c-ck / s -sk > -1 and  
				lb = -1 * (strike - set[k]) + setPrices[k]
				ub = slopeSetp * (strike - set[k]) + setPrices[k]
				if (k + 1 <= length(set))
					ub = min(ub, (setPrices[k+1] * (strike - set[k]) + setPrices[k] * (set[k+1] - strike)) / (set[k+1] - set[k]))
					lb = max(lb, -slopeSetp * (set[k+1] - strike) + setPrices[k+1])
				end
				priceA = (1 - lambda) * lb + lambda * ub
				priceB = lambda * lb + (1 - lambda) * ub

				#priceA = (-1*(1-lambda) + (lambda)*slopeSetp) * (strike - set[k+1]) + setPrices[k+1]
				#priceB = (-1*(lambda) + (1-lambda)*slopeSetp) * (strike - set[k+1]) + setPrices[k+1]
				price = if abs(pricesReject[i] - priceA) < abs(pricesReject[i] - priceB)
					priceA
				else
					priceB
				end
				# println("adding ",strike, " price=",price, " slope=", (price - setPrices[k]) / (strike - set[k]))
                insert!(set, k+1, strike)
                insert!(setPrices,k+1, price)
			else
				slopeSetm = (setPrices[k-1] - setPrices[k]) / (set[k-1] - set[k])
				slopeSetp = if k + 2 <= length(set)
					(setPrices[k+2] - setPrices[k+1]) / (set[k+2] - set[k+1])
				else
					0.0
				end
				#must be slopesetp > slopekp > slopek > slopesetm 
				# strike is between set[k] and set[k+1]
				# slopep > c[k+1]-c/sk+1 - s > c - c[k] / s - sk   > slopem  and 
				lb = slopeSetm * (strike - set[k]) + setPrices[k]
				ub = slopeSetp * (strike - set[k]) + setPrices[k]
				if (k + 1 <= length(set))
					ub = min(ub, (setPrices[k+1] * (strike - set[k]) + setPrices[k] * (set[k+1] - strike)) / (set[k+1] - set[k]))
					lb = max(lb, -slopeSetp * (set[k+1] - strike) + setPrices[k+1])
				end
				priceA = (1 - lambda) * lb + lambda * ub
				priceB = lambda * lb + (1 - lambda) * ub
				price = if abs(pricesReject[i] - priceA) < abs(pricesReject[i] - priceB)
					priceA
				else
					priceB
				end

                insert!(set, k+1, strike)
                insert!(setPrices,k+1, price)
			end

		end
        # setIndices = sortperm(set)
        # set = set[setIndices]
        # setPrices = setPrices[setIndices]
		return set[2:end], setPrices[2:end], weights
	else
		#return strikes in set or linear interpolated value from set.
		interp = PPInterpolation.makeLinearPP(set, setPrices)
		return strikes, interp.(strikes), weights
	end
end

#create closest set of arbitrage free prices
function filterConvexPrices(
	strikes::Vector{T},
	callPrices::Vector{T}, #undiscounted!
	weights::Vector{T},
	forward::T;
	tol = 1e-8, forceConvexity = false,
)::Tuple{Vector{T}, Vector{T}, Vector{T}} where {T}
	if !forceConvexity && isArbitrageFree(strikes, callPrices, forward)[1]
		return strikes, callPrices, weights
	end
	n = length(callPrices)
	z = Variable(n)
	G = spzeros(T, 2 * n, n)
	h = zeros(T, 2 * n)
	for i ∈ 2:n-1
		dym = (strikes[i] - strikes[i-1])
		dy = (strikes[i+1] - strikes[i])
		G[i, i-1] = -1 / dym
		G[i, i] = 1 / dym + 1 / dy
		G[i, i+1] = -1 / dy
	end
	G[1, 1] = 1 / (strikes[2] - strikes[1])
	G[1, 2] = -G[1, 1]
	G[n, n] = 1 / (strikes[n] - strikes[n-1])
	G[n, n-1] = -G[n, n]
	for i ∈ 1:n
		h[i] = -tol
		G[n+i, i] = -1
		h[n+i] = -max(forward - strikes[i], 0) - tol
	end
	h[1] = 1 - tol
	W = spdiagm(weights)
	strikesf = strikes
	# problem = minimize(square(norm(W * (z - callPrices))), G * z <= h)
	# #solve!(problem, () -> SCS.Optimizer(verbose = 0))
	# Convex.solve!(problem, () -> COSMO.Optimizer(verbose = false, eps_rel = 1e-8, eps_abs = 1e-8))
	# #println("problem status is ", problem.status, " optimal value is ", problem.optval)
	# pricesf = Convex.evaluate(z)
	amat, aind = convertSparse(copy(-G'))
	dmat = diagm(1.0 ./ weights)
	dvec = @. callPrices * weights^2
	nEqualities = (strikes[1] == 0) ? 1 : 0
	pricesf, lagr, crval, iact, nact, iter = solveQPcompact(dmat, dvec, amat, aind, -h, meq = nEqualities, factorized = true)
	return strikesf, pricesf, weights
end

#transform to X coordinate for collocation
function makeXFromUndiscountedPrices(strikesf::Vector{T}, pricesf::Vector{T}; slopeTolerance = 1e-8) where {T}
	return makeXFromUndiscountedPrices(strikesf, pricesf, s -> -norminv(-s), slopeTolerance)
end

function makeXFromUndiscountedPrices(strikesf::Vector{T}, pricesf::Vector{T}, invphi, slopeTolerance = 1e-8) where {T}
	n = length(strikesf)

	pif = ones(T, n)
	for i ∈ 2:n-1
		dxi = strikesf[i+1] - strikesf[i]
		dxim = strikesf[i] - strikesf[i-1]
		dzi = (pricesf[i+1] - pricesf[i]) / dxi
		dzim = (pricesf[i] - pricesf[i-1]) / dxim
		s = (dxim * dzi + dxi * dzim) / (dxim + dxi)
		previous = pif[i-1] < slopeTolerance ? pif[i-1] : pif[i-1] - slopeTolerance
		pif[i] = min(-s, previous)
	end
	dzi = (pricesf[n] - pricesf[n-1]) / (strikesf[n] - strikesf[n-1])
	pif[n] = -dzi
	dzim = -pif[n-2]
	if dzi * dzim < 0 || dzi < dzim || abs(dzi) < slopeTolerance
		pif = pif[1:n-1]
		strikesf = strikesf[1:n-1]
	end
	dzim = (pricesf[2] - pricesf[1]) / (strikesf[2] - strikesf[1])
	pif[1] = -dzim
	if dzim <= -1 + slopeTolerance
		pif = pif[2:end]
		strikesf = strikesf[2:end]
	end
	xif = invphi.(-pif)
	return strikesf, pif, xif
end
