
# ============================================================================
# slv_precomputed.jl
#
# Precomputed Heston SLV QE simulation.
#
# Interpolation policy:
#   * total implied variance w(y,T): cubic spline in y
#   * w(T) between simulation times: linear in TOTAL variance only
#   * local variance LV(y,T): log-PCHIP in y
#   * E[V|S]: log-PCHIP in log spot
#   * leverage L² = LV/E[V|S]: log-PCHIP in y
#   * production MC: leverage spline lookup only
#
# Assumes the existing project already defines:
#   HestonModel, GridConfig{QE}, MCPayoff, Curve, ConstantRateCurve,
#   BrownianBridgeConstruction, norminv, skipTo!, next!,
#   _obs_indices, _record_observation!, _store_payoff!,
#   _check_nan, _mc_summary, dimensionOut.
export SLVConfig, PreparedSLV, prepareSLV, simulateIterSLV, calibrateSLV!, precomputeSLVLocalVariance, precomputeSLVLeverage

Base.@kwdef struct SLVConfig
	nSimExpectation::Int = 50_000

	# Same role as Go CompactSupportCalculator.
	gridSize::Int = 32
	maxGridSize::Int = 64
	bandwidthFactor::Float64 = 1.5
	threshold::Float64 = 1e-4

	# Same role as Go GetRange().
	nDev::Float64 = 3.0

	# Dupire moneyness domain.
	yMin::Float64 = -4.0
	yMax::Float64 = 4.0
	nY::Int = 257

	# Numerical floors.
	totalVarianceFloor::Float64 = 1e-12
	localVarianceFloor::Float64 = 1e-10
	conditionalVarianceFloor::Float64 = 1e-10
	leverageFloor::Float64 = 1e-10
	dupireDenominatorFloor::Float64 = 1e-10

	# false: varianceByLogmoneyness returns annualized IV variance.
	# true : varianceByLogmoneyness already returns total variance.
	surfaceReturnsTotalVariance::Bool = false

	# At T=0 use w(0,y)=0 and forward-difference total variance in time.
	useZeroInitialTotalVariance::Bool = true

	# If true, invalid Dupire values throw instead of flooring.
	strictDupire::Bool = false
end


# ----------------------------------------------------------------------------
# Cubic spline
# ----------------------------------------------------------------------------

struct CubicSpline1D
	x::Vector{Float64}
	a::Vector{Float64}
	b::Vector{Float64}
	c::Vector{Float64}
	d::Vector{Float64}
end


function CubicSpline1D(
	x::AbstractVector{<:Real},
	y::AbstractVector{<:Real},
)
	n = length(x)
	@assert n == length(y)
	@assert n >= 3

	xx = Float64.(x)
	yy = Float64.(y)

	@inbounds for i in 1:(n-1)
		@assert xx[i+1] > xx[i]
	end

	h = diff(xx)
	a = copy(yy)

	α = zeros(Float64, n)
	@inbounds for i in 2:(n-1)
		α[i] =
			3.0 / h[i] * (a[i+1] - a[i]) -
			3.0 / h[i-1] * (a[i] - a[i-1])
	end

	l = ones(Float64, n)
	μ = zeros(Float64, n)
	z = zeros(Float64, n)

	@inbounds for i in 2:(n-1)
		l[i] =
			2.0 * (xx[i+1] - xx[i-1]) -
			h[i-1] * μ[i-1]

		μ[i] = h[i] / l[i]
		z[i] = (α[i] - h[i-1] * z[i-1]) / l[i]
	end

	cfull = zeros(Float64, n)
	b = zeros(Float64, n-1)
	d = zeros(Float64, n-1)

	@inbounds for j in (n-1):-1:1
		cfull[j] = z[j] - μ[j] * cfull[j+1]
		b[j] =
			(a[j+1] - a[j]) / h[j] -
			h[j] * (cfull[j+1] + 2.0 * cfull[j]) / 3.0
		d[j] = (cfull[j+1] - cfull[j]) / (3.0 * h[j])
	end

	return CubicSpline1D(
		xx,
		a[1:(n-1)],
		b,
		cfull[1:(n-1)],
		d,
	)
end


@inline function _cubicInterval(
	x::Vector{Float64},
	z::Float64,
)
	n = length(x)
	if z <= x[1]
		return 1
	elseif z >= x[n]
		return n - 1
	end
	return searchsortedlast(x, z)
end


@inline function cubicValue(
	s::CubicSpline1D,
	z::Float64,
)
	i = _cubicInterval(s.x, z)
	dz =
		z <= s.x[1] ? 0.0 :
		z >= s.x[end] ? s.x[end] - s.x[i] :
		z - s.x[i]

	return s.a[i] +
		   s.b[i] * dz +
		   s.c[i] * dz^2 +
		   s.d[i] * dz^3
end


@inline function cubicFirstDerivative(
	s::CubicSpline1D,
	z::Float64,
)
	if z <= s.x[1]
		return s.b[1]
	elseif z >= s.x[end]
		i = length(s.a)
		h = s.x[end] - s.x[end-1]
		return s.b[i] + 2.0*s.c[i]*h + 3.0*s.d[i]*h^2
	end

	i = _cubicInterval(s.x, z)
	dz = z - s.x[i]

	return s.b[i] +
		   2.0*s.c[i]*dz +
		   3.0*s.d[i]*dz^2
end


@inline function cubicSecondDerivative(
	s::CubicSpline1D,
	z::Float64,
)
	if z <= s.x[1]
		return 2.0*s.c[1]
	elseif z >= s.x[end]
		i = length(s.a)
		h = s.x[end] - s.x[end-1]
		return 2.0*s.c[i] + 6.0*s.d[i]*h
	end

	i = _cubicInterval(s.x, z)
	dz = z - s.x[i]

	return 2.0*s.c[i] + 6.0*s.d[i]*dz
end


# ----------------------------------------------------------------------------
# PCHIP
# ----------------------------------------------------------------------------

struct PchipSpline1D
	x::Vector{Float64}
	y::Vector{Float64}
	d::Vector{Float64}
end


function _pchipSlopes(
	x::Vector{Float64},
	y::Vector{Float64},
)
	n = length(x)

	if n == 1
		return [0.0]
	elseif n == 2
		d = (y[2]-y[1])/(x[2]-x[1])
		return [d, d]
	end

	h = diff(x)
	δ = diff(y) ./ h
	d = zeros(Float64, n)

	@inbounds for i in 2:(n-1)
		if δ[i-1] * δ[i] <= 0.0
			d[i] = 0.0
		else
			w1 = 2.0*h[i] + h[i-1]
			w2 = h[i] + 2.0*h[i-1]
			d[i] =
				(w1+w2) /
				(w1/δ[i-1] + w2/δ[i])
		end
	end

	# Fritsch-Carlson endpoint limiter.
	d1 =
		((2.0*h[1] + h[2])*δ[1] - h[1]*δ[2]) /
		(h[1] + h[2])

	if d1 * δ[1] <= 0.0
		d1 = 0.0
	elseif δ[1] * δ[2] < 0.0 &&
		   abs(d1) > 3.0*abs(δ[1])
		d1 = 3.0*δ[1]
	end

	dn =
		((2.0*h[end] + h[end-1])*δ[end] - h[end]*δ[end-1]) /
		(h[end] + h[end-1])

	if dn * δ[end] <= 0.0
		dn = 0.0
	elseif δ[end] * δ[end-1] < 0.0 &&
		   abs(dn) > 3.0*abs(δ[end])
		dn = 3.0*δ[end]
	end

	d[1] = d1
	d[end] = dn

	return d
end


function PchipSpline1D(
	x::AbstractVector{<:Real},
	y::AbstractVector{<:Real},
)
	xx = Float64.(x)
	yy = Float64.(y)

	@assert length(xx) == length(yy)
	@assert length(xx) >= 2

	@inbounds for i in 1:(length(xx)-1)
		@assert xx[i+1] > xx[i]
	end

	return PchipSpline1D(
		xx,
		yy,
		_pchipSlopes(xx, yy),
	)
end


@inline function pchipValue(
	s::PchipSpline1D,
	z::Float64,
)
	n = length(s.x)

	if n == 1
		return s.y[1]
	elseif z <= s.x[1]
		return s.y[1]
	elseif z >= s.x[end]
		return s.y[end]
	end

	i = searchsortedlast(s.x, z)
	i = min(i, n-1)

	h = s.x[i+1] - s.x[i]
	τ = (z - s.x[i]) / h

	h00 = 2.0*τ^3 - 3.0*τ^2 + 1.0
	h10 = τ^3 - 2.0*τ^2 + τ
	h01 = -2.0*τ^3 + 3.0*τ^2
	h11 = τ^3 - τ^2

	return h00*s.y[i] +
		   h10*h*s.d[i] +
		   h01*s.y[i+1] +
		   h11*h*s.d[i+1]
end


struct LogPchipSpline1D
	spline::PchipSpline1D
end


function LogPchipSpline1D(
	x::AbstractVector{<:Real},
	values::AbstractVector{<:Real},
	floor::Float64,
)
	xx = Float64.(x)
	vv = max.(Float64.(values), floor)

	if length(xx) == 1
		x0 = xx[1]
		v0 = vv[1]
		return LogPchipSpline1D(
			PchipSpline1D(
				[x0-1.0, x0+1.0],
				[log(v0), log(v0)],
			),
		)
	end

	return LogPchipSpline1D(
		PchipSpline1D(
			xx,
			log.(vv),
		),
	)
end


@inline function logPchipValue(
	s::LogPchipSpline1D,
	z::Float64,
)
	return exp(
		pchipValue(
			s.spline,
			z,
		),
	)
end


# ----------------------------------------------------------------------------
# Total variance / Dupire
# ----------------------------------------------------------------------------

function totalVarianceByLogmoneyness(
	model,
	y::Float64,
	t::Float64;
	surfaceReturnsTotalVariance::Bool = false,
)
	if t <= 0.0
		return 0.0
	end

	v = varianceByLogmoneyness(model, y, t)

	if surfaceReturnsTotalVariance
		return max(v, 0.0)
	end

	return max(t*v, 0.0)
end


@inline function localVarianceDenominator(
	w::Float64,
	wy::Float64,
	wyy::Float64,
	y::Float64,
)
	return 1.0 -
		   y/w*wy +
		   0.25 * wy^2 *
		   (-0.25 - 1.0/w + y^2/w^2) +
		   0.5*wyy
end


# ----------------------------------------------------------------------------
# Precomputed Dupire LV
# ----------------------------------------------------------------------------

struct SLVLocalVarianceGrid
	yGrid::Vector{Float64}
	totalVarianceSplines::Vector{CubicSpline1D}
	localVarianceValues::Matrix{Float64}
	localVarianceSplines::Vector{LogPchipSpline1D}
	localVarianceATM::Vector{Float64}
	times::Vector{Float64}
	logForward::Vector{Float64}
	floor::Float64
end


function _buildTotalVarianceSplines(
	lvModel,
	times::Vector{Float64},
	yGrid::Vector{Float64},
	config::SLVConfig,
)
	splines =
		Vector{CubicSpline1D}(
			undef,
			length(times),
		)

	for i in eachindex(times)
		t = times[i]

		w = zeros(Float64, length(yGrid))

		if t > 0.0
			@inbounds for j in eachindex(yGrid)
				w[j] =
					max(
						totalVarianceByLogmoneyness(
							lvModel,
							yGrid[j],
							t;
							surfaceReturnsTotalVariance =
							config.surfaceReturnsTotalVariance,
						),
						config.totalVarianceFloor,
					)
			end
		end

		splines[i] =
			CubicSpline1D(
				yGrid,
				w,
			)
	end

	return splines
end


function precomputeSLVLocalVariance(
	lvModel,
	spot::Float64,
	genTimes::AbstractVector{<:Real};
	config::SLVConfig = SLVConfig(),
)
	times = Float64.(genTimes)

	@assert length(times) >= 2
	@assert issorted(times)
	@assert config.yMin < config.yMax
	@assert config.nY >= 3

	yGrid =
		collect(
			range(
				config.yMin,
				config.yMax;
				length = config.nY,
			),
		)

	lnspot = log(spot)

	logForwardValues =
		[
			logForward(
				lvModel,
				lnspot,
				t,
			)
			for t in times
		]

	totalSplines =
		_buildTotalVarianceSplines(
			lvModel,
			times,
			yGrid,
			config,
		)

	nSteps = length(times)-1
	lvValues =
		zeros(
			Float64,
			nSteps,
			length(yGrid),
		)

	lvSplines =
		Vector{LogPchipSpline1D}(
			undef,
			nSteps,
		)

	lvATM =
		zeros(
			Float64,
			nSteps,
		)

	for i in 1:nSteps
		t0 = times[i]
		t1 = times[i+1]
		dt = t1-t0

		@assert dt > 0.0

		w0s = totalSplines[i]
		w1s = totalSplines[i+1]
		row = @view lvValues[i, :]

		@inbounds for j in eachindex(yGrid)
			y = yGrid[j]

			w =
				max(
					cubicValue(w1s, y),
					config.totalVarianceFloor,
				)

			wy =
				cubicFirstDerivative(
					w1s,
					y,
				)

			wyy =
				cubicSecondDerivative(
					w1s,
					y,
				)

			# Only calendar interpolation is linear:
			#
			#     w_t ≈ (w(t1)-w(t0))/dt
			#
			# There is no linear interpolation in y.
			w0 =
				if (
					i == 1 &&
					t0 == 0.0 &&
					config.useZeroInitialTotalVariance
				)
					0.0
				else
					cubicValue(
						w0s,
						y,
					)
				end

			wt =
				(w - w0) / dt

			denom =
				localVarianceDenominator(
					w,
					wy,
					wyy,
					y,
				)

			lv =
				wt / denom

			if !(
				isfinite(lv) &&
				isfinite(denom) &&
				denom > config.dupireDenominatorFloor &&
				lv > config.localVarianceFloor
			)
				if config.strictDupire
					error(
						"Invalid Dupire LV: t=$t1 y=$y w=$w wt=$wt denom=$denom lv=$lv",
					)
				end
				lv = config.localVarianceFloor
			end

			row[j] = lv
		end

		# Cubic, positive, shape-preserving interpolation in y.
		lvSplines[i] =
			LogPchipSpline1D(
				yGrid,
				row,
				config.localVarianceFloor,
			)

		lvATM[i] =
			logPchipValue(
				lvSplines[i],
				0.0,
			)
	end

	return SLVLocalVarianceGrid(
		yGrid,
		totalSplines,
		lvValues,
		lvSplines,
		lvATM,
		times,
		logForwardValues,
		config.localVarianceFloor,
	)
end


@inline function localVarianceLookup(
	grid::SLVLocalVarianceGrid,
	timeIndex::Int,
	y::Float64,
)
	return max(
		logPchipValue(
			grid.localVarianceSplines[timeIndex],
			y,
		),
		grid.floor,
	)
end


# ----------------------------------------------------------------------------
# Particle KDE
# ----------------------------------------------------------------------------

struct SLVGridRange
	min::Float64
	max::Float64
	bandwidth::Float64
	sqrtTime::Float64
end


@inline function quarticKernel(
	u::Float64,
)
	z = 1.0-u^2
	return z <= 0.0 ? 0.0 : (15.0/16.0)*z^2
end


@inline function inverseQuarticKernelValue(
	x::Float64,
)
	b =
		1.0 -
		sqrt(
			max(
				0.0,
				16.0/15.0*x,
			),
		)

	return b <= 0.0 ? 1.0 : sqrt(b)
end


@inline function _locateLowerIndex(
	x::Vector{Float64},
	z::Float64,
	fromIndex::Int,
)
	n = length(x)

	if z <= x[fromIndex]
		return fromIndex
	end

	lo = fromIndex+1
	hi = n

	while lo <= hi
		mid = (lo+hi) >>> 1

		if x[mid] < z
			lo = mid+1
		elseif x[mid] > z
			hi = mid-1
		else
			return mid
		end
	end

	return max(fromIndex, lo-1)
end


function _sortParticles!(
	x::Vector{Float64},
	v::Vector{Float64},
)
	p = sortperm(x)
	x[:] = x[p]
	v[:] = v[p]
	return nothing
end


function _particleRange(
	grid::SLVLocalVarianceGrid,
	timeIndex::Int,
	lnspot::Float64,
	config::SLVConfig,
)
	t = grid.times[timeIndex+1]
	lv =
		max(
			grid.localVarianceATM[timeIndex],
			config.localVarianceFloor,
		)

	minx =
		lnspot -
		config.nDev*sqrt(lv*t) -
		0.5*lv*t

	maxx =
		lnspot +
		config.nDev*sqrt(lv*t) -
		0.5*lv*t

	bandwidth =
		sqrt(
			lv*max(t, 0.25),
		)

	return SLVGridRange(
		minx,
		maxx,
		bandwidth,
		sqrt(t),
	)
end


function _conditionalVarianceSpline(
	logAssetValues::Vector{Float64},
	varianceValues::Vector{Float64},
	grange::SLVGridRange,
	config::SLVConfig,
)
	n = length(logAssetValues)

	@assert n == length(varianceValues)

	if n < 8
		value =
			max(
				sum(max.(varianceValues, 0.0))/max(n, 1),
				config.conditionalVarianceFloor,
			)

		return LogPchipSpline1D(
			[-1.0, 1.0],
			[value, value],
			config.conditionalVarianceFloor,
		)
	end

	# Use log spot as the KDE coordinate. This keeps the bandwidth in the
	# same units as the Go path generator's state[0].
	x = logAssetValues
	v = varianceValues

	gridSize =
		max(
			config.maxGridSize,
			floor(
				Int,
				config.gridSize*grange.sqrtTime,
			),
		)

	gridSize = max(gridSize, 2)

	minIndex = n ÷ 1000
	maxIndex = n-1-minIndex

	xmin = x[minIndex+1]
	xmax = x[maxIndex+1]

	if !(isfinite(xmin) && isfinite(xmax) && xmax > xmin)
		value =
			max(
				sum(max.(v, 0.0))/n,
				config.conditionalVarianceFloor,
			)

		return LogPchipSpline1D(
			[x[1]-1.0, x[1]+1.0],
			[value, value],
			config.conditionalVarianceFloor,
		)
	end

	h =
		config.bandwidthFactor *
		grange.bandwidth *
		n^(-0.2)

	if !(isfinite(h) && h > 0.0)
		value =
			max(
				sum(max.(v, 0.0))/n,
				config.conditionalVarianceFloor,
			)

		return LogPchipSpline1D(
			[xmin, xmax],
			[value, value],
			config.conditionalVarianceFloor,
		)
	end

	dx =
		(xmax-xmin) /
		(gridSize-1)

	eta =
		inverseQuarticKernelValue(
			h*config.threshold,
		)

	radius = eta*h

	xgrid =
		Vector{Float64}(
			undef,
			gridSize,
		)

	egrid =
		Vector{Float64}(
			undef,
			gridSize,
		)

	firstParticle = 1
	effective = 0

	for j in 1:gridSize
		xj =
			xmin +
			(j-1)*dx

		xgrid[j] = xj

		firstParticle =
			_locateLowerIndex(
				x,
				xj-radius,
				firstParticle,
			)

		xHigh = xj+radius

		numerator = 0.0
		denominator = 0.0

		@inbounds for k in firstParticle:n
			u =
				(x[k]-xj)/h

			kernel =
				quarticKernel(u)/h

			numerator += kernel

			if (
				isfinite(v[k]) &&
				v[k] > 0.0
			)
				denominator +=
					kernel*v[k]
			end

			if (
				x[k] >= xHigh &&
				(
					denominator > 0.0 ||
					abs(u) >= 1.0
				)
			)
				break
			end
		end

		if (
			numerator > 0.0 &&
			denominator > 0.0
		)
			effective += 1
			egrid[effective] =
				max(
					denominator/numerator,
					config.conditionalVarianceFloor,
				)
		end
	end

	if effective == 0
		value =
			max(
				sum(max.(v, 0.0))/n,
				config.conditionalVarianceFloor,
			)

		return LogPchipSpline1D(
			[xmin, xmax],
			[value, value],
			config.conditionalVarianceFloor,
		)
	end

	resize!(xgrid, effective)
	resize!(egrid, effective)

	# PCHIP on log(E[V|S]) gives a positive cubic spline.
	return LogPchipSpline1D(
		xgrid,
		egrid,
		config.conditionalVarianceFloor,
	)
end


# ----------------------------------------------------------------------------
# QE variance step: same equations as the original Julia simulateIter().
# ----------------------------------------------------------------------------

@inline function _qeVarianceStep(
	v::Float64,
	u2::Float64,
	dt::Float64,
	ekdt::Float64,
	model,
	withMartingale::Bool,
)
	m =
		model.θ +
		(v-model.θ)*ekdt

	m = max(m, 1e-14)

	s2 =
		v*model.σ^2*ekdt/model.κ*(1.0-ekdt) +
		(model.θ*model.σ^2)/(2.0*model.κ) *
		(1.0-ekdt)^2

	psi =
		max(
			s2/(m^2),
			1e-14,
		)

	psiC = 1.5
	vNew = 0.0
	mc = 0.0

	if psi <= psiC
		psiInv = 1.0/psi

		b2 =
			2.0*psiInv -
			1.0 +
			sqrt(
				max(
					0.0,
					2.0*psiInv*(2.0*psiInv-1.0),
				),
			)

		a = m/(1.0+b2)
		b = sqrt(max(b2, 0.0))
		zv = norminv(u2)

		vNew =
			a*(b+zv)^2

		if withMartingale
			A =
				dt/2.0 *
				(
					model.κ*model.ρ/model.σ -
					0.5
				) +
				model.ρ/model.σ +
				dt/4.0 *
				(1.0-model.ρ^2)

			B =
				dt/2.0 *
				(
					model.κ*model.ρ/model.σ -
					0.5
				) -
				model.ρ/model.σ +
				dt/4.0 *
				(1.0-model.ρ^2)

			C =
				model.ρ/model.σ *
				model.κ * model.θ * dt

			den = 1.0-2.0*A*a

			if den > 0.0
				mc =
					C -
					A*b2*a/den +
					0.5*log(den) -
					B*v
			end
		end

	else
		p =
			(psi-1.0)/(psi+1.0)

		invBeta =
			m/(1.0-p)

		if u2 <= p
			vNew = 0.0
		else
			vNew =
				log(
					(1.0-p)/(1.0-u2),
				)*invBeta
		end

		if withMartingale
			A =
				dt/2.0 *
				(
					model.κ*model.ρ/model.σ -
					0.5
				) +
				model.ρ/model.σ +
				dt/4.0 *
				(1.0-model.ρ^2)

			B =
				dt/2.0 *
				(
					model.κ*model.ρ/model.σ -
					0.5
				) -
				model.ρ/model.σ +
				dt/4.0 *
				(1.0-model.ρ^2)

			C =
				model.ρ/model.σ *
				model.κ * model.θ * dt

			den =
				1.0 -
				invBeta*A

			arg =
				p +
				(1.0-p)/den

			if den > 0.0 && arg > 0.0
				mc =
					C -
					log(arg) -
					B*v
			end
		end
	end

	return max(vNew, 0.0), mc
end


# ----------------------------------------------------------------------------
# QE log spot update given a precomputed leverage factor L².
# ----------------------------------------------------------------------------

@inline function _evolveQEWithLeverage(
	x::Float64,
	v::Float64,
	z1::Float64,
	u2::Float64,
	dt::Float64,
	ekdt::Float64,
	model,
	leverageSquared::Float64,
	logForwardPrev::Float64,
	logForwardNext::Float64,
	withMartingale::Bool,
)
	vNew, mc =
		_qeVarianceStep(
			v,
			u2,
			dt,
			ekdt,
			model,
			withMartingale,
		)

	sqrtlv =
		sqrt(
			max(
				leverageSquared,
				0.0,
			),
		)

	vIntegral =
		max(
			0.5*(vNew+v),
			0.0,
		)

	rhoBar =
		sqrt(
			max(
				0.0,
				1.0-model.ρ^2,
			),
		)

	xNew =
		x +
		(logForwardNext-logForwardPrev) +
		model.ρ/model.σ *
		sqrtlv *
		(
			vNew-v-model.κ*model.θ*dt
		) +
		(
			model.κ*model.ρ/model.σ -
			0.5*sqrtlv
		) *
		sqrtlv *
		vIntegral *
		dt +
		rhoBar *
		sqrt(vIntegral) *
		sqrtlv *
		z1 +
		mc

	return xNew, vNew
end


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------

struct SLVCalibration
	localComponents::Vector{LogPchipSpline1D}
end


function calibrateSLV!(
	rngCalibration,
	model,
	spot::Float64,
	lvGrid::SLVLocalVarianceGrid;
	config::SLVConfig = SLVConfig(),
	withMartingale::Bool = true,
	start::Int = 0,
)
	times = lvGrid.times
	nSteps = length(times)-1
	nParticles = config.nSimExpectation
	lnspot = log(spot)

	x = fill(lnspot, nParticles)
	v =
		fill(
			max(
				model.v0,
				config.conditionalVarianceFloor,
			),
			nParticles,
		)

	u =
		zeros(
			Float64,
			2,
			nParticles,
		)

	components =
		Vector{LogPchipSpline1D}(
			undef,
			length(times),
		)

	v0 =
		max(
			model.v0,
			config.conditionalVarianceFloor,
		)

	components[1] =
		LogPchipSpline1D(
			[lnspot-1.0, lnspot+1.0],
			[v0, v0],
			config.conditionalVarianceFloor,
		)

	skipTo!(
		rngCalibration,
		start,
	)

	for i in 1:nSteps
		dt = times[i+1]-times[i]
		ekdt = exp(-model.κ*dt)

		next!(
			rngCalibration,
			u,
		)

		lc = components[i]

		@inbounds for j in 1:nParticles
			y =
				x[j] -
				lvGrid.logForward[i+1]

			lv =
				localVarianceLookup(
					lvGrid,
					i,
					y,
				)

			lvc =
				max(
					logPchipValue(
						lc,
						x[j],
					),
					config.conditionalVarianceFloor,
				)

			leverageSquared =
				max(
					lv/lvc,
					config.leverageFloor,
				)

			z1 =
				norminv(u[1, j]) *
				sqrt(dt)

			x[j], v[j] =
				_evolveQEWithLeverage(
					x[j],
					v[j],
					z1,
					u[2, j],
					dt,
					ekdt,
					model,
					leverageSquared,
					lvGrid.logForward[i],
					lvGrid.logForward[i+1],
					withMartingale,
				)
		end

		_sortParticles!(x, v)

		grange =
			_particleRange(
				lvGrid,
				i,
				lnspot,
				config,
			)

		components[i+1] =
			_conditionalVarianceSpline(
				x,
				v,
				grange,
				config,
			)
	end

	return SLVCalibration(
		components,
	)
end


# ----------------------------------------------------------------------------
# Precompute leverage L² = LV / E[V|S]
# ----------------------------------------------------------------------------

struct SLVLeverageSurface
	yGrid::Vector{Float64}
	leverageValues::Matrix{Float64}
	leverageSplines::Vector{LogPchipSpline1D}
	times::Vector{Float64}
	logForward::Vector{Float64}
	floor::Float64
end


function precomputeSLVLeverage(
	lvGrid::SLVLocalVarianceGrid,
	components::Vector{LogPchipSpline1D};
	config::SLVConfig = SLVConfig(),
)
	nSteps = length(lvGrid.times)-1
	nY = length(lvGrid.yGrid)

	values =
		zeros(
			Float64,
			nSteps,
			nY,
		)

	splines =
		Vector{LogPchipSpline1D}(
			undef,
			nSteps,
		)

	for i in 1:nSteps
		row = @view values[i, :]
		lc = components[i]

		@inbounds for j in 1:nY
			y = lvGrid.yGrid[j]

			x =
				lvGrid.logForward[i+1] +
				y

			lv =
				localVarianceLookup(
					lvGrid,
					i,
					y,
				)

			lvc =
				max(
					logPchipValue(
						lc,
						x,
					),
					config.conditionalVarianceFloor,
				)

			row[j] =
				max(
					lv/lvc,
					config.leverageFloor,
				)
		end

		splines[i] =
			LogPchipSpline1D(
				lvGrid.yGrid,
				row,
				config.leverageFloor,
			)
	end

	return SLVLeverageSurface(
		copy(lvGrid.yGrid),
		values,
		splines,
		copy(lvGrid.times),
		copy(lvGrid.logForward),
		config.leverageFloor,
	)
end


@inline function leverageLookup(
	surface::SLVLeverageSurface,
	timeIndex::Int,
	y::Float64,
)
	return max(
		logPchipValue(
			surface.leverageSplines[timeIndex],
			y,
		),
		surface.floor,
	)
end


# ----------------------------------------------------------------------------
# Prepared cache
# ----------------------------------------------------------------------------

struct PreparedSLV
	localVariance::SLVLocalVarianceGrid
	localComponents::Vector{LogPchipSpline1D}
	leverage::SLVLeverageSurface
	times::Vector{Float64}
	logForward::Vector{Float64}
end


function prepareSLV(
	rngCalibration,
	hestonModel,
	lvModel,
	spot::Float64,
	genTimes::AbstractVector{<:Real};
	config::SLVConfig = SLVConfig(),
	withMartingale::Bool = true,
	calibrationStart::Int = 0,
)
	lvGrid =
		precomputeSLVLocalVariance(
			lvModel,
			spot,
			genTimes;
			config = config,
		)

	calibration =
		calibrateSLV!(
			rngCalibration,
			hestonModel,
			spot,
			lvGrid;
			config = config,
			withMartingale = withMartingale,
			start = calibrationStart,
		)

	leverage =
		precomputeSLVLeverage(
			lvGrid,
			calibration.localComponents;
			config = config,
		)

	return PreparedSLV(
		lvGrid,
		calibration.localComponents,
		leverage,
		copy(lvGrid.times),
		copy(lvGrid.logForward),
	)
end


# ----------------------------------------------------------------------------
# Production MC
# ----------------------------------------------------------------------------

function simulateIterSLV(
	rng,
	model::HestonModel{T},
	lvModel,
	spot::Float64,
	payoff::MCPayoff,
	start::Int,
	nSim::Int,
	config::GridConfig{QE};
	prepared::Union{Nothing, PreparedSLV} = nothing,
	rngCalibration = nothing,
	slvConfig::SLVConfig = SLVConfig(),
	withBB = config.method.withBB,
	withMartingale = config.method.withMartingale,
	calibrationStart::Int = 0,
) where {T}

	specTimes = config.specTimes
	tte = specTimes[end]
	df = discountFactor(model, tte)

	genTimes =
		Float64.(config.genTimes)

	nSteps =
		length(genTimes)-1

	obsIdx =
		_obs_indices(
			genTimes,
			specTimes,
		)

	if prepared === nothing
		if rngCalibration === nothing
			error(
				"rngCalibration is required when prepared=nothing",
			)
		end

		prepared =
			prepareSLV(
				rngCalibration,
				model,
				lvModel,
				spot,
				genTimes;
				config = slvConfig,
				withMartingale = withMartingale,
				calibrationStart = calibrationStart,
			)
	end

	logForwardValues =
		prepared.logForward

	h =
		[
			genTimes[i+1]-genTimes[i]
			for i in 1:nSteps
		]

	ekdt =
		[
			exp(-model.κ*h[i])
			for i in 1:nSteps
		]

	local bb
	if withBB
		bb =
			BrownianBridgeConstruction(
				genTimes[2:end],
			)
	end

	skipTo!(
		rng,
		start,
	)

	u =
		zeros(
			Float64,
			2,
			nSteps,
		)

	u1 = @view u[1, :]
	u2 = @view u[2, :]

	z1 =
		zeros(
			Float64,
			nSteps,
		)

	dimOut =
		dimensionOut(payoff)[1]

	payoffValues =
		zeros(
			T,
			dimOut,
			nSim,
		)

	obsPath =
		zeros(
			T,
			length(specTimes),
		)

	for sim in 1:nSim
		next!(rng, u)

		@. u1 = norminv(u1)

		if withBB
			transform!(bb, u1, z1)
		else
			@inbounds for i in 1:nSteps
				z1[i] =
					u1[i]*sqrt(h[i])
			end
		end

		x =
			logForwardValues[1]

		v =
			max(
				model.v0,
				0.0,
			)

		nextObs = 1

		nextObs =
			_record_observation!(
				obsPath,
				nextObs,
				obsIdx,
				0,
				x,
			)

		@inbounds for i in 1:nSteps
			y =
				x -
				logForwardValues[i+1]

			leverageSquared =
				leverageLookup(
					prepared.leverage,
					i,
					y,
				)

			x, v =
				_evolveQEWithLeverage(
					x,
					v,
					z1[i],
					u2[i],
					h[i],
					ekdt[i],
					model,
					leverageSquared,
					logForwardValues[i],
					logForwardValues[i+1],
					withMartingale,
				)

			nextObs =
				_record_observation!(
					obsPath,
					nextObs,
					obsIdx,
					i,
					x,
				)
		end

		_store_payoff!(
			payoffValues,
			sim,
			payoff,
			obsPath,
			df,
		)

		_check_nan(
			payoffValues,
			sim,
			obsPath,
			u,
		)
	end

	return _mc_summary(
		payoffValues,
	)
end
