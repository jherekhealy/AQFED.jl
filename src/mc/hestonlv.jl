import AQFED.Math: norminv
import AQFED.TermStructure: LocalVolatilityModel, HestonModel, discountFactor, logForward, localVarianceByLogmoneyness
import AQFED.Random: next!, nextn!, skipTo!, Chacha8SIMD, AbstractRNGSeq
import AQFED.Math: normcdf
#import Random123: Philox4x
using Statistics
using LinearAlgebra
using PPInterpolation

export HestonLVModel, CompactSupportCalculator, BinSplineCalculator, calibrateHestonLV, simulateQE

struct HestonLVModel{H <: HestonModel, L <: LocalVolatilityModel}
	heston::H
	lv::L
end

abstract type LocalComponentCalculator end

struct CompactSupportCalculator <: LocalComponentCalculator
	gridSize::Int
	maxGridSize::Int
	bandwidthFactor::Float64
	threshold::Float64
	inverse::Bool
end
CompactSupportCalculator(; gridSize = 50, maxGridSize = 100, bandwidthFactor = 2.0, threshold = 1e-4, inverse=false) =
	CompactSupportCalculator(gridSize, maxGridSize, bandwidthFactor, threshold, inverse)

struct BinSplineCalculator <: LocalComponentCalculator
	gridSize::Int
	isMedian::Bool
	inverse::Bool
end
BinSplineCalculator(; gridSize = 50, isMedian = false, inverse=false) = BinSplineCalculator(gridSize, isMedian,inverse)

@inline function quarticKernelValue(x::Float64)
	u = 1.0 - x*x
	return u <= 0.0 ? 0.0 : 0.9375 * u * u  # 15/16 = 0.9375
end

@inline function inverseQuarticKernelValue(x::Float64)
	b = 1.0 - sqrt(1.0666666666666667 * x) # 16/15
	return b <= 0.0 ? 1.0 : sqrt(b)
end

function locateLowerIndex(x::AbstractVector{Float64}, z::Float64, fromIndex::Int = 1)
	n = length(x)
	if z <= x[fromIndex]
		return fromIndex
	end
	low = fromIndex + 1
	high = n
	while low <= high
		mid = low + (high - low) ÷ 2
		midVal = x[mid]
		if midVal < z
			low = mid + 1
		elseif midVal > z
			high = mid - 1
		else
			return mid
		end
	end
	return max(1, low - 1)
end

struct BinCalculator <: LocalComponentCalculator
    gridSize::Int
    inverse::Bool
    truncationFraction::Float64
end

BinCalculator(;
    gridSize = 50,
    inverse = false,
    truncationFraction = 0.0,
) = BinCalculator(
    gridSize,
    inverse,
    truncationFraction,
)

function computeExpectation(
    calc::BinCalculator,
    assetValues::Vector{Float64},
    varianceValues::Vector{Float64},
    sqrtTime::Float64,
    bandwidthBase::Float64,
)
    n = length(assetValues)

    length(varianceValues) == n ||
        throw(DimensionMismatch("assetValues and varianceValues must have equal length"))

    n > 0 || throw(ArgumentError("assetValues must not be empty"))
    calc.gridSize > 0 || throw(ArgumentError("gridSize must be positive"))

    # The Go implementation currently has:
    # truncationIndex := nSim / 1000 * 0
    # hence truncationIndex is always zero.
    truncationIndex = floor(Int, calc.truncationFraction * n)
    truncationIndex = clamp(truncationIndex, 0, (n - 1) ÷ 2)

    firstIndex = truncationIndex + 1
    lastIndex = n - truncationIndex
    nEffective = lastIndex - firstIndex + 1

    gridSize = min(calc.gridSize, nEffective)

    p = sortperm(assetValues)
    xSorted = assetValues[p]
    vSorted = varianceValues[p]

    fGrid = Vector{Float64}(undef, gridSize)
    expectationGrid = Vector{Float64}(undef, gridSize)

    for k in 1:gridSize
        # Equivalent to the Go expressions:
        #
        # binStartIndex := k * n / gridSize
        # nextIndex := (k + 1) * n / gridSize
        #
        # after conversion from zero-based to one-based indexing.
        iStart = firstIndex + ((k - 1) * nEffective) ÷ gridSize
        iEnd = firstIndex + (k * nEffective) ÷ gridSize - 1

        iStart = clamp(iStart, firstIndex, lastIndex)
        iEnd = clamp(iEnd, iStart, lastIndex)

        # Exact Go behaviour: use the first asset value in each bin.
        fGrid[k] = xSorted[iStart]

        varianceSum = 0.0

        @inbounds for i in iStart:iEnd
            varianceSum += max(vSorted[i], 0.0)
        end

        countBin = iEnd - iStart + 1
        conditionalVariance = varianceSum / countBin

        expectationGrid[k] = if !calc.inverse
            inv(max(conditionalVariance, 1e-6))
        else
            max(conditionalVariance,1e-6)
        end
    end

    xMin = fGrid[1]
    xMax = fGrid[end]
    yMin = expectationGrid[1]
    yMax = expectationGrid[end]

    return function (x::Float64)
        if x <= xMin
            return yMin
        elseif x >= xMax
            return yMax
        end

        # searchsortedlast returns the last grid point <= x.
        # It returns zero if x lies below the first point.
        i = clamp(searchsortedlast(fGrid, x), 1, gridSize)

        return expectationGrid[i]
    end
end


struct CompactSupportPCalculator <: LocalComponentCalculator
    gridSize::Int
    maxGridSize::Int
    bandwidthFactor::Float64
    threshold::Float64
    minEffectiveParticles::Int
	inverse::Bool
end


function CompactSupportPCalculator(;
    gridSize = 50,
    maxGridSize = 100,
    bandwidthFactor = 2.0,
    threshold = 1e-4,              # retained for API compatibility
    minEffectiveParticles = 20,
	inverse = true
)
    return CompactSupportPCalculator(
        gridSize,
        maxGridSize,
        bandwidthFactor,
        threshold,
        minEffectiveParticles,
		inverse
    )
end


# Compatibility with the previous four-field positional constructor.
CompactSupportPCalculator(
    gridSize::Int,
    maxGridSize::Int,
    bandwidthFactor::Float64,
    threshold::Float64,
) = CompactSupportCalculator(
    gridSize,
    maxGridSize,
    bandwidthFactor,
    threshold,
    20,
)


@inline function _constantConditionalVariance(
    numerator::Float64,
    denominator::Float64,
)
    return numerator > 0.0 ? denominator / numerator : 0.0
end


@inline function _localLinearConditionalVariance(
    S0::Float64,
    S1::Float64,
    S2::Float64,
    T0::Float64,
    T1::Float64,
)
    determinant = S0 * S2 - S1 * S1

    if determinant <= 1e-14 || S0 <= 0.0
        return NaN
    end

    # Intercept of the local linear regression at x = fj.
    value = (S2 * T0 - S1 * T1) / determinant

    return isfinite(value) ? value : NaN
end


function _linearInterpolation(
    xGrid::Vector{Float64},
    yGrid::Vector{Float64},
)
    n = length(xGrid)

    n == length(yGrid) ||
        throw(DimensionMismatch("xGrid and yGrid must have equal length"))

    n > 0 ||
        throw(ArgumentError("interpolation grid must not be empty"))

    return function (x::Float64)
        if x <= xGrid[1]
            return yGrid[1]
        elseif x >= xGrid[end]
            return yGrid[end]
        end

        i = searchsortedlast(xGrid, x)
        i = clamp(i, 1, n - 1)

        x0 = xGrid[i]
        x1 = xGrid[i + 1]
        y0 = yGrid[i]
        y1 = yGrid[i + 1]

        dx = x1 - x0

        if dx <= 0.0
            return y0
        end

        weight = (x - x0) / dx

        return (1.0 - weight) * y0 + weight * y1
    end
end


function computeExpectation(
    calc::CompactSupportPCalculator,
    logAssetValues::Vector{Float64},
    varianceValues::Vector{Float64},
    sqrtTime::Float64,
    bandwidthBase::Float64,
)
    n = length(logAssetValues)

    n == length(varianceValues) ||
        throw(DimensionMismatch(
            "logAssetValues and varianceValues must have equal length",
        ))

    n > 0 ||
        throw(ArgumentError("particle vectors must not be empty"))

    # Sort particles by log-asset value.
    permutation = sortperm(logAssetValues)

    xSorted = logAssetValues[permutation]
    vSorted = varianceValues[permutation]

    # Grid size adapts mildly with maturity, as in the original code.
    gridSize = round(Int, calc.gridSize * max(sqrtTime, 0.1))
    gridSize = min(gridSize, calc.maxGridSize)
    gridSize = min(gridSize, n)

    if n < 10 || gridSize < 4
        vMean = mean(max.(vSorted, 0.0))
        vMean = max(vMean, 1e-6)
        return x -> vMean
    end

    gridSize = max(gridSize, 4)

    # Bandwidth in log-forward space.
    h = calc.bandwidthFactor * bandwidthBase * n^(-0.2)
    h = max(h, 1e-8)

    fGrid = Vector{Float64}(undef, gridSize)
    varianceGrid = Vector{Float64}(undef, gridSize)

    effectiveIdx = 0

    for k in 1:gridSize
        # Empirical quantile-type grid locations.
        iMid = 1 + ((2 * k - 1) * n) ÷ (2 * gridSize)
        iMid = clamp(iMid, 1, n)

        fj = xSorted[iMid]

        # Full support of the quartic kernel.
        leftIndex = searchsortedfirst(xSorted, fj - h)
        rightIndex = searchsortedlast(xSorted, fj + h)

        if leftIndex > rightIndex
            continue
        end

        # Kernel moments for the local-linear regression.
        S0 = 0.0
        S1 = 0.0
        S2 = 0.0

        T0 = 0.0
        T1 = 0.0

        weightSquareSum = 0.0

        @inbounds for i in leftIndex:rightIndex
            dx = xSorted[i] - fj
            z = dx / h
            weight = quarticKernelValue(z) / h

            if weight <= 0.0
                continue
            end

            variance = max(vSorted[i], 0.0)

            S0 += weight
            S1 += weight * dx
            S2 += weight * dx * dx

            T0 += weight * variance
            T1 += weight * dx * variance

            weightSquareSum += weight * weight
        end

        if S0 <= 0.0 || weightSquareSum <= 0.0
            continue
        end

        # Effective number of particles in the kernel window.
        effectiveParticles = S0^2 / weightSquareSum

        if effectiveParticles < calc.minEffectiveParticles
            continue
        end

        # Local-linear estimate of E[V | X = fj].
        conditionalVariance = _localLinearConditionalVariance(
            S0,
            S1,
            S2,
            T0,
            T1,
        )

        # Fallback to local-constant kernel regression if the local-linear
        # estimate is singular or non-positive.
        if !isfinite(conditionalVariance) ||
           conditionalVariance <= 0.0

            conditionalVariance =
                _constantConditionalVariance(S0, T0)
        end

        if !isfinite(conditionalVariance) ||
           conditionalVariance <= 0.0

            continue
        end

        effectiveIdx += 1
        fGrid[effectiveIdx] = fj
        varianceGrid[effectiveIdx] = max(conditionalVariance, 1e-8)
    end

    # Global fallback.
    if effectiveIdx < 4
        vMean = mean(max.(vSorted, 0.0))
        vMean = max(vMean, 1e-6)
        return x -> vMean
    end

    fGrid = fGrid[1:effectiveIdx]
    varianceGrid = varianceGrid[1:effectiveIdx]

    # Remove duplicate or nearly duplicate interpolation knots.
    keep = trues(length(fGrid))

    @inbounds for k in 2:length(fGrid)
        keep[k] = fGrid[k] > fGrid[k - 1] + 1e-10
    end

    fGrid = fGrid[keep]
    varianceGrid = varianceGrid[keep]

    if length(fGrid) < 2
        vMean = mean(max.(vSorted, 0.0))
        vMean = max(vMean, 1e-6)
        return x -> vMean
    end

    # Directly interpolate E[V | X], not its reciprocal.
    interpolatedVariance = _linearInterpolation(
        fGrid,
        varianceGrid,
    )

    return function (x::Float64)
        value = interpolatedVariance(x)
        return max(value, 1e-8)
    end
end

function computeExpectation(
	calc::CompactSupportCalculator,
	logAssetValues::Vector{Float64},
	varianceValues::Vector{Float64},
	sqrtTime::Float64,
	bandwidthBase::Float64,
)
	n = length(logAssetValues)
	p = sortperm(logAssetValues)
	xSorted = logAssetValues[p]
	vSorted = varianceValues[p]

	gridSize = min(calc.maxGridSize, round(Int, calc.gridSize * max(sqrtTime, 0.1)))
	gridSize = min(gridSize, n ÷ 10)
	gridSize = max(gridSize, 10)

	h = calc.bandwidthFactor * bandwidthBase * (n^(-0.2))
	eta = calc.threshold
	fEta = inverseQuarticKernelValue(h * eta) * h

	fGrid = Vector{Float64}(undef, gridSize)
	expectationGrid = Vector{Float64}(undef, gridSize)

	minIndex = 1
	maxIndex = n
	effectiveIdx = 0
	indexFirstParticle = 1

	for j in 1:gridSize
		midIndex = minIndex + ((j - 1) * (maxIndex - minIndex)) ÷ gridSize + (maxIndex - minIndex) ÷ (2 * gridSize)
		midIndex = clamp(midIndex, 1, n)
		fj = xSorted[midIndex]

		fLow = fj - fEta
		indexFirstParticle = locateLowerIndex(xSorted, fLow, indexFirstParticle)
		fHigh = fj + fEta

		num = 0.0
		denom = 0.0
		for i in indexFirstParticle:n
			fi = xSorted[i]
			z = (fi - fj) / h
			dirac = quarticKernelValue(z) / h
			num += dirac
			v = vSorted[i]
			if v > 0.0
				denom += dirac * v
			end
			if fi >= fHigh && (denom > 0.0 || z >= 1.0)
				break
			end
		end

		if denom > 0.0 && num > 0.0
			effectiveIdx += 1
			fGrid[effectiveIdx] = fj
			expectationGrid[effectiveIdx] = if calc.inverse  
				denom/num
			else 
				num / denom
			end
		end
	end

	if effectiveIdx < 4
		# Fallback to constant ratio 1/mean(v)
		vMean = mean(v -> max(v, 1e-4), varianceValues)
		invV = 1.0 / vMean
			if !calc.inverse
			return x -> invV
		else
			return x-> vMean
		end
	end

	fT = fGrid[1:effectiveIdx]
	eT = expectationGrid[1:effectiveIdx]

	# Ensure strictly increasing knots for spline interpolation
	uniqueIdx = Int[1]
	for k in 2:effectiveIdx
		if fT[k] > fT[uniqueIdx[end]] + 1e-10
			push!(uniqueIdx, k)
		end
	end
	if length(uniqueIdx) < 4
		vMean = mean(v -> max(v, 1e-4), varianceValues)
		invV = 1.0 / vMean
		if !calc.inverse
			return x -> invV
		else
			return x-> vMean
		end
	end

	fSpline = fT[uniqueIdx]
	eSpline = eT[uniqueIdx]

	spl = makeCubicPP(
		fSpline,
		eSpline,
		PPInterpolation.FIRST_DERIVATIVE,
		0.0,
		PPInterpolation.FIRST_DERIVATIVE,
		0.0,
		PPInterpolation.HuynRational(),
	)

	xMin = fSpline[1]
	xMax = fSpline[end]
	yMin = eSpline[1]
	yMax = eSpline[end]

	return function (x::Float64)
		if x <= xMin
			return yMin
		elseif x >= xMax
			return yMax
		else
			val = spl(x)
			return val > 0.0 ? val : yMin
		end
	end
end

function computeExpectation(
	calc::BinSplineCalculator,
	logAssetValues::Vector{Float64},
	varianceValues::Vector{Float64},
	sqrtTime::Float64,
	bandwidthBase::Float64,
)
	n = length(logAssetValues)
	p = sortperm(logAssetValues)
	xSorted = logAssetValues[p]
	vSorted = varianceValues[p]

	gridSize = min(calc.gridSize, n ÷ 10)
	gridSize = max(gridSize, 4)

	fGrid = Vector{Float64}(undef, gridSize)
	expectationGrid = Vector{Float64}(undef, gridSize)

	binSize = n ÷ gridSize

	if !calc.isMedian
		for k in 1:gridSize
			iStart = (k - 1) * binSize + 1
			iEnd = k == gridSize ? n : k * binSize

			xMean = 0.0
			vSum = 0.0

			for i in iStart:iEnd
				xMean += xSorted[i]
				vSum += max(vSorted[i], 0.0)
			end

			countBin = iEnd - iStart + 1
			fGrid[k] = xMean / countBin
			expectationGrid[k] = if calc.inverse 
				max(vSum,1e-6)/countBin
			else 
				countBin / max(vSum, 1e-6)
			end

		end
		keep = trues(length(fGrid))

for k in 2:length(fGrid)
    keep[k] = fGrid[k] > fGrid[k - 1] + 1e-12
end

fGrid = fGrid[keep]
expectationGrid = expectationGrid[keep]
	else
		for k in 1:gridSize
			iStart = (k - 1) * binSize + 1
			iEnd = k == gridSize ? n : k * binSize
			countBin = iEnd - iStart + 1
			iMid1 = iStart + (countBin - 1) ÷ 2
			iMid2 = iStart + countBin ÷ 2
			fGrid[k] = 0.5 * (xSorted[iMid1] + xSorted[iMid2])
			vSum = 0.0
			for i in iStart:iEnd
			    vSum += max(vSorted[i], 0.0)
			end
			vAvg = vSum / (iEnd - iStart + 1)
			expectationGrid[k] = if calc.inverse 
				max(vAvg,1e-6)
				else
					1.0 / max(vAvg, 1e-6)
				end
			##
			#vAvg = mean(max.(view(vSorted, iStart:iEnd), 0.0))
			#expectationGrid[k] = inv(max(vAvg, 1e-6))
		end
	end
	##


	spl = makeCubicPP(
		fGrid,
		expectationGrid,
		PPInterpolation.FIRST_DERIVATIVE,
		0.0,
		PPInterpolation.FIRST_DERIVATIVE,
		0.0,
		PPInterpolation.HuynRational(),
	)

	xMin = fGrid[1]
	xMax = fGrid[end]
	yMin = expectationGrid[1]
	yMax = expectationGrid[end]

	return function (x::Float64)
		if x <= xMin
			return yMin
		elseif x >= xMax
			return yMax
		else
			val = spl(x)
			return val > 0.0 ? val : yMin
		end
	end
end

function ndims(model::HestonLVModel, specificTimes::Vector{Float64}, timestepSize::Float64)
	genTimes = pathgenTimes(model.heston, specificTimes, timestepSize)
	return (length(genTimes) - 1) * 2
end

function ndims(config::GridConfig{<:QE})
	return (length(config.genTimes) - 1) * 2
end

"""
	calibrateHestonLV(rng, model::HestonLVModel, spot::Float64, genTimes::Vector{Float64}, nSim::Int; lcCalc=CompactSupportCalculator())

Calibrate the Heston-LV leverage curves L_k(x) across time steps genTimes using particle propagation.
"""
function calibrateHestonLV(
	rng, #note: there seems to be a bug with rng when used with scrambled sobol. is it the skipping?
	model::HestonLVModel,
	spot::Float64,
	genTimes::Vector{Float64},
	nSim::Int;
	lcCalc::LocalComponentCalculator = CompactSupportCalculator(),
	withMartingale = true,
)
	nSteps = length(genTimes) - 1
	leverageFunctions = Vector{Any}(undef, nSteps)

	lnS = fill(log(spot), nSim)
	v = fill(model.heston.v0, nSim)
	lnspot = log(spot)

	κ = model.heston.κ
	θ = model.heston.θ
	σ = model.heston.σ
	ρ = model.heston.ρ
	ρBar = sqrt(1.0 - ρ^2)

	t0 = genTimes[1]
	lnf0 = logForward(model.heston, lnspot, t0)

	u0 = Vector{Float64}(undef, nSim)
	u1 = Vector{Float64}(undef, nSim)
	w0 = Vector{Float64}(undef, nSim)

	for i in 1:nSteps
		t1 = genTimes[i+1]
		dt = t1 - t0
		sqrtdt = sqrt(dt)
		sqrtTime = sqrt(t1)

		lnf1 = logForward(model.heston, lnspot, t1)
		driftStep = lnf1 - lnf0

		# 1. Compute local variance at log-moneyness for bandwidth estimation
		yStartFwd = lnf0 - lnf0
		lvAtFwd = localVarianceByLogmoneyness(model.lv.surface, [yStartFwd], t0, t1)[1]
		bandwidthBase = sqrt(max(lvAtFwd, 1e-4) * max(t1, 0.25))

		# 2. Compute local component expectation ratio function at t_i
		lcFunc = computeExpectation(lcCalc, lnS, v, sqrtTime, bandwidthBase)
		leverageFunctions[i] = lcFunc

		# 3. Generate random numbers for step i
		skipTo!(rng, 2 * i - 1, 0, length(u0))
		nextn!(rng, 2 * i - 1, u0)
		skipTo!(rng, 2 * i, 0, length(u1))
		next!(rng, 2 * i, u1)

		@. w0 = u0 * sqrtdt

		# 4. Evolve particles using QE scheme with leverage factor
		ekdt = exp(-κ * dt)
		psiC = 1.5

		logm = @. lnS - lnf0
		lvVec = localVarianceByLogmoneyness(model.lv.surface, logm, t0, t1)

		for j in 1:nSim
			vOld = v[j]
			m = θ + (vOld - θ) * ekdt
			s2 = vOld * σ^2 * ekdt / κ * (1.0 - ekdt) + (θ * σ^2) / (2 * κ) * (1.0 - ekdt)^2
			psi = s2 / (m * m)

			vNew = 0.0
			martingaleCorr = 0.0
			uv = u1[j]
	        lvc = lcFunc(lnS[j])
			lvFactor = if lcCalc.inverse
			 max(lvVec[j] /  lvc, 1e-6)
			else
			 max(lvVec[j] * lvc, 1e-6)
			end
			sqrtlv = sqrt(lvFactor)
		
			if psi <= psiC
				psiInv = 1.0 / psi
				b2 = 2.0 * psiInv - 1.0 + sqrt(2.0 * psiInv * (2.0 * psiInv - 1.0))
				a = m / (1.0 + b2)
				b = sqrt(b2)
				zv = norminv(uv)
				vNew = a * (b + zv)^2
				if withMartingale
					A = 0.5 * dt * sqrtlv * (κ * ρ / σ - 0.5*sqrtlv) + ρ / σ  * sqrtlv+ 0.25 * dt * (1.0 - ρ^2) * sqrtlv^2
					B = 0.5 * dt * sqrtlv * (κ * ρ / σ - 0.5*sqrtlv) - ρ / σ * sqrtlv + 0.25 * dt * (1.0 - ρ^2) * sqrtlv^2
					C = ρ / σ * κ * θ * dt * sqrtlv
					martingaleCorr = C - A * b2 * a / (1.0 - 2.0 * A * a) + 0.5 * log(1.0 - 2.0 * A * a) - B * vOld
				end
			else
				p = (psi - 1.0) / (psi + 1.0)
				invBeta = m / (1.0 - p)
				if uv <= p
					vNew = 0.0
				else
					vNew = log((1.0 - p) / (1.0 - uv)) * invBeta
				end
				if withMartingale
					A = 0.5 * dt * sqrtlv * (κ * ρ / σ - 0.5*sqrtlv) + ρ / σ *sqrtlv + 0.25 * dt * (1.0 - ρ^2) * sqrtlv^2
					B = 0.5 * dt * sqrtlv * (κ * ρ / σ - 0.5*sqrtlv) - ρ / σ*sqrtlv + 0.25 * dt * (1.0 - ρ^2) * sqrtlv^2
					C = ρ / σ * κ * θ * dt * sqrtlv
					martingaleCorr = C - log(p + (1.0 - p) / (1.0 - invBeta * A)) - B * vOld
				end
			end

			vInt = 0.5 * (vNew + vOld)

			lnS[j] += driftStep + ρ / σ * sqrtlv * (vNew - vOld - κ * θ * dt) +
					  (κ * ρ / σ - 0.5 * sqrtlv) * sqrtlv * vInt * dt +
					  ρBar * sqrt(max(vInt, 0.0)) * sqrtlv * w0[j] + martingaleCorr
			v[j] = vNew
		end

		t0 = t1
		lnf0 = lnf1
	end

	return leverageFunctions
end

"""
	simulateQE(rng, model::HestonLVModel, spot::Float64, payoff::MCPayoff, start::Int, nSim::Int, timestepSize::Float64; leverageFunctions=nothing, lcCalc=CompactSupportCalculator())

Simulate paths for HestonLVModel using QE scheme and evaluate option payoff.
"""
function simulateScheme(
	rng,
	model::HestonLVModel,
	spot::Float64,
	payoff::MCPayoff,
	start::Int,
	nSim::Int,
	config::GridConfig{QE};
	leverageFunctions = nothing,
	lcCalc::LocalComponentCalculator = CompactSupportCalculator(),
	rngCalib::RNG = AbstractRNGSeq(Chacha8SIMD(), ndims(config)),
	nSimCalib = 8192,
) where {RNG <: AbstractSeq}
	withMartingale = config.method.withMartingale
	specTimes = config.specTimes
	tte = specTimes[end]
	df = discountFactor(model.heston, tte)
	genTimes = config.genTimes
	nSteps = length(genTimes) - 1
	#obsIdx = _obs_indices(genTimes, specTimes)

	if isnothing(leverageFunctions)
		leverageFunctions = calibrateHestonLV(rngCalib, model, spot, genTimes, nSimCalib; lcCalc = lcCalc, withMartingale = withMartingale)
	end

	currentPathValues = fill(log(spot), nSim)
	pathValues = Matrix{Float64}(undef, (nSim, length(specTimes)))
	v = fill(model.heston.v0, nSim)
	lnspot = log(spot)

	κ = model.heston.κ
	θ = model.heston.θ
	σ = model.heston.σ
	ρ = model.heston.ρ
	ρBar = sqrt(1.0 - ρ^2)

	t0 = genTimes[1]
	lnf0 = logForward(model.heston, lnspot, t0)

	u0 = Vector{Float64}(undef, nSim)
	u1 = Vector{Float64}(undef, nSim)
	w0 = Vector{Float64}(undef, nSim)

	iObs = 1
	if !isempty(specTimes) && specTimes[1] <= 1e-8
		@. pathValues[:, 1] = spot
		iObs = 2
	end
	for (dim, t1) in enumerate(genTimes[2:end])
		dt = t1 - t0
		sqrth = sqrt(dt)
		lnf1 = logForward(model.heston, lnspot, t1)
		driftStep = lnf1 - lnf0

		d = 2 * dim - 1
		skipTo!(rng, d, start, length(u0))
		nextn!(rng, d, u0)
		skipTo!(rng, d + 1, start, length(u1))
		next!(rng, d + 1, u1)

		@. w0 = u0 * sqrth

		ekdt = exp(-κ * dt)
		psiC = 1.5
		lcFunc = leverageFunctions[dim]

		logm = @. currentPathValues - lnf0
		lvVec = localVarianceByLogmoneyness(model.lv.surface, logm, t0, t1)

		for j in 1:nSim
			vOld = v[j]
			m = θ + (vOld - θ) * ekdt
			s2 = vOld * σ^2 * ekdt / κ * (1.0 - ekdt) + (θ * σ^2) / (2 * κ) * (1.0 - ekdt)^2
			psi = s2 / (m * m)

			vNew = 0.0
			martingaleCorr = 0.0
			uv = u1[j]

			if psi <= psiC
				psiInv = 1.0 / psi
				b2 = 2.0 * psiInv - 1.0 + sqrt(2.0 * psiInv * (2.0 * psiInv - 1.0))
				a = m / (1.0 + b2)
				b = sqrt(b2)
				zv = norminv(uv)
				vNew = a * (b + zv)^2
				if withMartingale
					A = 0.5 * dt * (κ * ρ / σ - 0.5) + ρ / σ + 0.25 * dt * (1.0 - ρ^2)
					B = 0.5 * dt * (κ * ρ / σ - 0.5) - ρ / σ + 0.25 * dt * (1.0 - ρ^2)
					C = ρ / σ * κ * θ * dt
					martingaleCorr = C - A * b2 * a / (1.0 - 2.0 * A * a) + 0.5 * log(1.0 - 2.0 * A * a) - B * vOld
				end
			else
				p = (psi - 1.0) / (psi + 1.0)
				invBeta = m / (1.0 - p)
				if uv <= p
					vNew = 0.0
				else
					vNew = log((1.0 - p) / (1.0 - uv)) * invBeta
				end
				if withMartingale
					A = 0.5 * dt * (κ * ρ / σ - 0.5) + ρ / σ + 0.25 * dt * (1.0 - ρ^2)
					B = 0.5 * dt * (κ * ρ / σ - 0.5) - ρ / σ + 0.25 * dt * (1.0 - ρ^2)
					C = ρ / σ * κ * θ * dt
					martingaleCorr = C - log(p + (1.0 - p) / (1.0 - invBeta * A)) - B * vOld
				end
			end

			lvc = lcFunc(currentPathValues[j])
			lvFactor = if lcCalc.inverse
			 max(lvVec[j] /  lvc, 1e-6)
			else
			 max(lvVec[j] * lvc, 1e-6)
			end
			sqrtlv = sqrt(lvFactor)
			vInt = 0.5 * (vNew + vOld)

			currentPathValues[j] += driftStep + ρ / σ * sqrtlv * (vNew - vOld - κ * θ * dt) +
									(κ * ρ / σ - 0.5 * sqrtlv) * sqrtlv * vInt * dt +
									ρBar * sqrt(max(vInt, 0.0)) * sqrtlv * w0[j] + martingaleCorr
			v[j] = vNew
		end

		if iObs <= length(specTimes) && abs(genTimes[dim+1] - specTimes[iObs]) < 1e-8
			@. pathValues[:, iObs] = exp(currentPathValues)
			iObs += 1
		end

		t0 = t1
		lnf0 = lnf1
	end

	payoffValues = map(x -> evaluatePayoffOnPath(payoff, x, df), eachrow(pathValues))
	payoffMean = mean(payoffValues)
	return payoffMean, stdm(payoffValues, payoffMean) / sqrt(length(payoffValues))
end



mutable struct NonConformRegression
    useCholesky::Bool
    nNodes::Int
    abscissa::Vector{Float64}
    ordinate::Vector{Float64}
    slope::Vector{Float64}
    isFlatExtrapolation::Bool
end


function makeNonConformRegression(
    useCholesky::Bool,
    nNodes::Int,
    isFlatExtrapolation::Bool,
)
    nNodes > 0 || throw(ArgumentError("nNodes must be positive"))

    return NonConformRegression(
        useCholesky,
        nNodes,
        Float64[],
        Float64[],
        Float64[],
        isFlatExtrapolation,
    )
end


struct NonConformRegressionFactory
    useCholesky::Bool
    numberOfNodes::Int
end


function makeRegression(factory::NonConformRegressionFactory)
    return makeNonConformRegression(
        factory.useCholesky,
        factory.numberOfNodes,
        false,
    )
end

@inline function searchLowerIndexBinary(
    x::Vector{Float64},
    z::Float64,
)
    n = length(x)

    n > 0 || throw(ArgumentError("abscissa must not be empty"))

    if z <= x[1]
        return 1
    elseif z >= x[end]
        return n
    end

    return searchsortedlast(x, z)
end

function _fit_local_line(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    iStart::Int,
    iEnd::Int,
    useCholesky::Bool,
)
    nObs = iEnd - iStart + 1

    nObs >= 2 ||
        return 0.0, mean(@view y[iStart:iEnd])

    xView = @view x[iStart:iEnd]
    yView = @view y[iStart:iEnd]

    sumX = 0.0
    sumX2 = 0.0
    sumY = 0.0
    sumXY = 0.0

    @inbounds for j in eachindex(xView)
        xj = xView[j]
        yj = yView[j]

        sumX += xj
        sumX2 += xj * xj
        sumY += yj
        sumXY += xj * yj
    end

    if useCholesky
        xtx = Symmetric([
            nObs  sumX
            sumX  sumX2
        ])

        rhs = [
            sumY
            sumXY
        ]

        # This is equivalent to the Cholesky solve used in the Go code.
        beta = cholesky(xtx) \ rhs
        return beta[2], beta[1]
    else
        # QR least-squares solve, equivalent to SolveQRVec.
        X = Matrix{Float64}(undef, nObs, 2)
        X[:, 1] .= 1.0
        X[:, 2] .= xView

        beta = X \ collect(yView)
        return beta[2], beta[1]
    end
end

function regressSortedValues!(
    r::NonConformRegression,
    x::Vector{Float64},
    y::Vector{Float64},
)
    length(x) == length(y) ||
        throw(DimensionMismatch("x and y must have equal length"))

    n = length(x)
    n > 0 || throw(ArgumentError("regression data must not be empty"))

    all(diff(x) .>= 0.0) ||
        throw(ArgumentError("x must be sorted in non-decreasing order"))

    nNodes = min(r.nNodes, n)
    nodeSize = max(1, n ÷ nNodes)

    abscissa = Vector{Float64}(undef, nNodes)
    ordinate = Vector{Float64}(undef, nNodes)
    slope = Vector{Float64}(undef, nNodes)

    startIndex = 1

    for k in 1:nNodes
        endIndex = k == nNodes ? n : min(k * nodeSize, n)

        # The Go implementation places the node at the first point of
        # each bin, except for the final implicit endpoint.
        abscissa[k] = x[startIndex]

        a, b = _fit_local_line(
            x,
            y,
            startIndex,
            endIndex,
            r.useCholesky,
        )

        slope[k] = a
        ordinate[k] = b

        startIndex = endIndex + 1

        if startIndex > n
            break
        end
    end

    r.abscissa = abscissa
    r.ordinate = ordinate
    r.slope = slope

    return r
end

function regressSortedValuesPositive!(
    r::NonConformRegression,
    x::Vector{Float64},
    y::Vector{Float64},
)
    length(x) == length(y) ||
        throw(DimensionMismatch("x and y must have equal length"))

    n = length(x)
    n > 0 || throw(ArgumentError("regression data must not be empty"))

    all(diff(x) .>= 0.0) ||
        throw(ArgumentError("x must be sorted in non-decreasing order"))

    nNodes = min(r.nNodes, n)
    nodeSize = max(1, n ÷ nNodes)

    abscissa = Vector{Float64}(undef, nNodes)
    ordinate = Vector{Float64}(undef, nNodes)
    slope = Vector{Float64}(undef, nNodes)

    startIndex = 1

    for k in 1:nNodes
        endIndex = k == nNodes ? n : min(k * nodeSize, n)

        abscissa[k] = x[startIndex]

        a, b = _fit_local_line(
            x,
            y,
            startIndex,
            endIndex,
            r.useCholesky,
        )

        # The Go implementation checks the fitted line at the current
        # bin's left endpoint and at the next bin's left endpoint.
        xLeft = x[startIndex]
        xRight = k == nNodes ? x[endIndex] : x[endIndex + 1]

        valueLeft = b + a * xLeft
        valueRight = b + a * xRight

        if valueLeft <= 0.0 || valueRight <= 0.0
            a = 0.0
            b = mean(@view y[startIndex:endIndex])
        end

        slope[k] = a
        ordinate[k] = b

        startIndex = endIndex + 1

        if startIndex > n
            break
        end
    end

    r.abscissa = abscissa
    r.ordinate = ordinate
    r.slope = slope

    return r
end


@inline function evaluate(
    r::NonConformRegression,
    z::Float64,
)
    n = length(r.abscissa)

    n > 0 || throw(ArgumentError("regression has not been fitted"))

    if r.isFlatExtrapolation
        if z <= r.abscissa[1]
            index = 1
        elseif z >= r.abscissa[end]
            index = n
        else
            index = searchLowerIndexBinary(r.abscissa, z)
        end
    else
        index = searchLowerIndexBinary(r.abscissa, z)
    end

    return r.ordinate[index] + r.slope[index] * z
end

struct NonConformCalculator <: LocalComponentCalculator
    gridSize::Int
    useCholesky::Bool
    inverse::Bool
    truncationFraction::Float64
end


NonConformCalculator(;
    gridSize = 50,
    useCholesky = true,
    inverse = false,
    truncationFraction = 0.0,
) = NonConformCalculator(
    gridSize,
    useCholesky,
    inverse,
    truncationFraction,
)

function computeExpectation(
    calc::NonConformCalculator,
    logAssetValues::Vector{Float64},
    varianceValues::Vector{Float64},
    sqrtTime::Float64,
    bandwidthBase::Float64,
)
    n = length(logAssetValues)

    length(varianceValues) == n ||
        throw(DimensionMismatch("logAssetValues and varianceValues must have equal length"))

    n > 0 || throw(ArgumentError("input vectors must not be empty"))

    truncationIndex = floor(Int, calc.truncationFraction * n)
    truncationIndex = clamp(truncationIndex, 0, (n - 1) ÷ 2)

    # Sort both observations consistently.
    p = sortperm(logAssetValues)

    xSorted = logAssetValues[p]
    vSorted = max.(varianceValues[p], 0.0)

    firstIndex = truncationIndex + 1
    lastIndex = n - truncationIndex

    xUsed = xSorted[firstIndex:lastIndex]
    vUsed = vSorted[firstIndex:lastIndex]

    regression = makeNonConformRegression(
        calc.useCholesky,
        calc.gridSize,
        true,       # flat extrapolation
    )

    regressSortedValuesPositive!(
        regression,
        xUsed,
        vUsed,
    )

    xFirst = xUsed[1]
    xLast = xUsed[end]

    return function (x::Float64)
        value = if x <= xFirst
            evaluate(regression, xFirst)
        elseif x >= xLast
            evaluate(regression, xLast)
        else
            evaluate(regression, x)
        end

        if !calc.inverse
            return inv(max(value, 1e-6))
        else
            return max(value, 1e-6)
        end
    end
end