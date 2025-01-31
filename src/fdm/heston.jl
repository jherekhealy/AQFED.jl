using CharFuncPricing
using Distributions
using LinearAlgebra
using SparseArrays
using PPInterpolation
using Dierckx #2D Spline

mutable struct VanillaFDM2DPayoff
	maturity::Float64
	strikes::Vector{Float64}
	isCall::Vector{Bool}
	value::Vector{Matrix{Float64}} #strike, value(S,V)
end

currentValue(p::VanillaFDM2DPayoff) = p.value
lastEvaluationTime(p::VanillaFDM2DPayoff) = p.maturity

function evaluateFDM2DPayoff(p::VanillaFDM2DPayoff, t, axis1, axis2)
	if t == p.maturity
		for (iStrike, strike) ∈ enumerate(p.strikes)
			sign = ifelse(p.isCall[iStrike], 1, -1)
			for i ∈ eachindex(axis1)
				p.value[iStrike][i, :] .= max(sign * (axis1[i] - strike), 0.0)
			end
		end
	end
end

mutable struct LogFDM2DPayoff
	maturity::Float64
	value::Vector{Matrix{Float64}} #strike, value(S,V)
end

currentValue(p::LogFDM2DPayoff) = p.value
lastEvaluationTime(p::LogFDM2DPayoff) = p.maturity

function evaluateFDM2DPayoff(p::LogFDM2DPayoff, t, axis1, axis2)
	if t == p.maturity
		for i ∈ eachindex(axis1)
			p.value[1][i, :] .= log(axis1[i])
		end
	end
end

mutable struct VIXFDM2DPayoff
	strikesSPX::Vector{Float64}
	strikesVIX::Vector{Float64} #we will assume same size for strikes
	maturitySPX::Float64
	maturityVIX::Float64
	forwardSPX::Float64
	value::Vector{Matrix{Float64}} #strike, value(S,V)
	strike0SPX::Float64
end

currentValue(p::VIXFDM2DPayoff) = p.value
lastEvaluationTime(p::VIXFDM2DPayoff) = p.maturitySPX
function estimateVVIX(p::VIXFDM2DPayoff, axis1, axis2, forwardVIX, spot, v0; truncationStart = 1)
	strike0Index = searchsortedlast(p.strikesVIX, forwardVIX)
	strike0VIX = p.strikesVIX[strike0Index]
	vvix = 0.0
	strikesVIX = ifelse(p.strikesVIX[1] == 0, p.strikesVIX[truncationStart:end], p.strikesVIX)
	for (k, strike) ∈ enumerate(strikesVIX)
		dstrike = if k == 1
			strikesVIX[k+1] - strikesVIX[k]
		elseif k == length(strikesVIX)
			strikesVIX[k] - strikesVIX[k-1]
		else
			(strikesVIX[k+1] - strikesVIX[k-1]) / 2
		end
		otmPrice = if strike > forwardVIX
			p.value[truncationStart-1+k]
		else
			p.value[truncationStart-1+k+length(p.strikesVIX)]
		end

		spl = Spline2D(axis1, axis2, otmPrice)
		otmValue = Dierckx.evaluate(spl, spot, v0)
		println(strike, " otm value ", otmValue)
		vvix += otmValue * dstrike / (strike^2)
	end
	println("vvix^2 ", vvix)
	vvix = sqrt((2 * vvix - (forwardVIX / strike0VIX - 1)^2) / (p.maturityVIX))

	return vvix #then interpolate on S0,V0.
end

function estimateVVIXFull(p::VIXFDM2DPayoff, axis1, axis2, spot, v0)
	vvix = zeros(size(p.value[1]))
	fEstimate = zeros(size(p.value[1]))
	#find atm strike/forward.
	x = zeros(length(p.strikesVIX), 2)
	x[:, 1] = p.strikesVIX
	x[:, 2] .= 1.0

	for i ∈ eachindex(axis1)
		for j ∈ eachindex(axis2)
			cmp = zeros(length(p.strikesVIX))
			for k ∈ eachindex(cmp)
				cmp[k] = p.value[k][i, j] - p.value[k+length(p.strikesVIX)][i, j]
			end
			fEstimate[i, j] = (x\cmp)[2]
		end
	end

	for i ∈ eachindex(axis1)
		for j ∈ eachindex(axis2)
			for (k, strike) ∈ enumerate(p.strikesVIX)
				dstrike = if k == 1
					p.strikesVIX[k+1] - p.strikesVIX[k]
				elseif k == length(p.strikesVIX)
					p.strikesVIX[k] - p.strikesVIX[k-1]
				else
					(p.strikesVIX[k+1] - p.strikesVIX[k-1]) / 2
				end
				otmPrice = if strike > fEstimate[i, j]
					p.value[k][i, j]
				else
					p.value[k+length(p.strikesVIX)][i, j]
				end
				vvix[i, j] += otmPrice * dstrike / (strike^2)
			end
			#corr = - (fEstimate[i,j] / strike0VIX - 1)^2
			vvix[i, j] = sqrt(max(2 * vvix[i, j], 0.0) / (p.maturityVIX))
		end
	end
	spl = Spline2D(axis1, axis2, vvix)
	return Dierckx.evaluate(spl, spot, v0)
end


function evaluateFDM2DPayoff(p::VIXFDM2DPayoff, t, axis1, axis2; smoothing = "None")
	if abs(t - p.maturitySPX) < 1e-5
		for (iStrike, strike) ∈ enumerate(p.strikesSPX)
			for i ∈ eachindex(axis1)
				if smoothing == "Kreiss"
					if !isIndexBetween(i, axis1, strike)
						p.value[iStrike][i, :] .= @. max(axis1[i] - strike, 0.0)
						p.value[iStrike+length(p.strikesSPX)][i, :] .= @. max(strike - axis1[i], 0.0)
					else
						h = (axis1[i+1] - axis1[i-1]) / 2
						p.value[iStrike][i, :] .= @. applyKreissSmoothing(x -> max(x - strike, 0.0), strike, axis1[i], h) #max(axis1[i] - strike, 0.0)
						p.value[iStrike+length(p.strikesSPX)][i, :] .= @. applyKreissSmoothing(x -> max(strike - x, 0.0), strike, axis1[i], h) #max(strike - axis1[i], 0.0)
					end
				elseif smoothing == "Averaging"
					if i > 1 && i < length(axis1) && (strike < axis1[i] + (axis1[i+1] - axis1[i]) / 2 && strike >= axis1[i] - (axis1[i] - axis1[i-1]) / 2)
						h = (axis1[i+1] - axis1[i-1]) / 2
						obj(x) = max(x - strike, 0.0)
						p.value[iStrike][i, :] .= @. (simpson(obj, axis1[i] - (axis1[i] - axis1[i-1]) / 2, strike - sqrt(eps())) + simpson(obj, strike + sqrt(eps()), axis1[i] + (axis1[i+1] - axis1[i]) / 2)) / h
						objP(x) = max(strike - x, 0.0)
						p.value[iStrike+length(p.strikesSPX)][i, :] .= @. (simpson(objP, axis1[i] - (axis1[i] - axis1[i-1]) / 2, strike - sqrt(eps())) + simpson(objP, strike + sqrt(eps()), axis1[i] + (axis1[i+1] - axis1[i]) / 2)) / h
					else
						p.value[iStrike][i, :] .= @. max(axis1[i] - strike, 0.0)
						p.value[iStrike+length(p.strikesSPX)][i, :] .= @. max(strike - axis1[i], 0.0)
					end
				else
					p.value[iStrike][i, :] .= @. max(axis1[i] - strike, 0.0)
					p.value[iStrike+length(p.strikesSPX)][i, :] .= @. max(strike - axis1[i], 0.0)
				end
			end
		end
	elseif abs(t - p.maturityVIX) < 1e-6
		vix = zeros(size(p.value[1]))
		fEstimate = zeros(size(p.value[1]))
		#find atm strike/forward.
		x = zeros(length(p.strikesSPX), 2)
		x[:, 1] = p.strikesSPX
		x[:, 2] .= 1.0

		for i ∈ eachindex(axis1)
			for j ∈ eachindex(axis2)
				cmp = zeros(length(p.strikesSPX))
				for k ∈ eachindex(cmp)
					cmp[k] = p.value[k][i, j] - p.value[k+length(p.strikesSPX)][i, j]
				end
				fEstimate[i, j] = (x\cmp)[2]
			end
		end


		# for iStrike ∈ eachindex(p.strikesSPX)
		# 	for j ∈ axes(p.value[iStrike], 2)
		# 		spl = PPInterpolation.CubicSplineNatural(axis1, p.value[iStrike][:, j])
		# 		p.value[iStrike][:, j] .= spl(1.0) 
		# 	end
		# end
		for i ∈ eachindex(axis1)
			for j ∈ eachindex(axis2)
				for (k, strike) ∈ enumerate(p.strikesSPX)
					dstrike = if k == 1
						p.strikesSPX[k+1] - p.strikesSPX[k]
					elseif k == length(p.strikesSPX)
						p.strikesSPX[k] - p.strikesSPX[k-1]
					else
						(p.strikesSPX[k+1] - p.strikesSPX[k-1]) / 2
					end
					otmPrice = if strike > fEstimate[i, j]
						p.value[k][i, j]
					else
						p.value[k+length(p.strikesSPX)][i, j]
					end
					vix[i, j] += otmPrice * dstrike / (strike^2)
				end
				#corr = - (fEstimate[i,j] / strike0SPX - 1)^2
				vix[i, j] = sqrt(max(2 * vix[i, j], 0.0) / (p.maturitySPX - p.maturityVIX))
			end
		end
		spl = Spline2D(axis1, axis2, vix)
		v0 = 0.026
		forwardVIX = Dierckx.evaluate(spl, 1.0, v0)
		println("forwardVIX=", forwardVIX)
		for (iStrike, strike) ∈ enumerate(p.strikesVIX)
			if smoothing == "Kreiss"
				# find index such that vix[i,j]=strike
				for i ∈ eachindex(axis1)
					for j ∈ eachindex(axis2)
						if i < length(axis1)  && i > 1 && j < length(axis2) && j > 1 && (((vix[i+1, j] >= strike)  && (vix[i-1, j] < strike)) || ((vix[i, j+1] >= strike) && (vix[i, j-1] < strike)))
							h1 = (axis1[i+1] - axis1[i-1]) / 2
							h2 = (axis2[j+1] - axis2[j-1]) / 2
							p.value[iStrike][i, j] = applyKreissSmoothing( v-> applyKreissSmoothing(x -> max(Dierckx.evaluate(spl, x, v) - strike, 0.0), axis1[i], axis1[i], h1), axis2[j], axis2[j],h2) #max(axis1[i] - strike, 0.0)
							p.value[iStrike+length(p.strikesSPX)][i, j] =  applyKreissSmoothing(v -> applyKreissSmoothing(x -> max(-Dierckx.evaluate(spl, x, v) + strike, 0.0), axis1[i], axis1[i], h1), axis2[j], axis2[j],h2) #max(strike - axis1[i], 0.0)
							else
								p.value[iStrike][i, j] =  max(vix[i, j] - strike, 0.0)
								p.value[iStrike+length(p.strikesSPX)][i, j] =  max(strike - vix[i, j], 0.0)
						end
					end
				end
			else
				p.value[iStrike][:] = @. max(vix - strike, 0.0)
				p.value[iStrike+length(p.strikesVIX)][:] = @. max(-(vix - strike), 0.0)
			end
		end

	end
end


function makeSystem(
	exponentialFitting,
	upwindingThreshold::Real,
	dt::Real,
	S::Vector{T},
	J::Vector{T},
	Jm::Vector{T},
	V::Vector{T},
	JV::Vector{T},
	JVm::Vector{T},
	hm::Real,
	hl::Real,
	kappa::Real,
	theta::Real,
	sigma::Real,
	rho::Real,
	r::Real,
	g::Real,
	useDirichlet::Bool,
	M::Int,
	L::Int,
) where T
	rij = zeros(T, L * M)
	A1ilj = zeros(T, L * M)
	A1ij = zeros(T, L * M)
	A1iuj = zeros(T, L * M)
	A2ij = zeros(T, L * M)
	A2ijl = zeros(T, L * M)
	A2iju = zeros(T, L * M)
	for j ∈ 1:L
		jm = (j - 1) * M
		i = 1
		index = i + jm
		if useDirichlet
			A1ij[index] = 0
			A2ij[index] = 0
		else
			drifti = g * S[i]
			A1iuj[index] = -dt * drifti / (Jm[i+1] * hm)
			A1ij[index] = -dt * (-r * 0.5) - A1iuj[index]
			A2ij[index] = -dt * (-r * 0.5)
		end
		i = M
		index = i + jm
		drifti = g * S[i]
		A1ilj[index] = dt * drifti / (Jm[i] * hm)
		A1ij[index] = -dt * (-r * 0.5) - A1ilj[index]
		A2ij[index] = -dt * (-r * 0.5)

		@simd for i ∈ 2:M-1
			index = i + jm
			svi = S[i]^2 * V[j] / J[i]
			drifti = g * S[i]
			if exponentialFitting == "UpwindingThreshold" || exponentialFitting == "Partial Exponential Fitting"
				if abs(drifti * hm / svi) > upwindingThreshold
					svi = drifti * hm / tanh(drifti * hm / svi)
				end
			elseif exponentialFitting == "Full"
				svi = drifti * hm / tanh(drifti * hm / svi)
			elseif exponentialFitting == "FullUpwind"
				svi = svi + abs(drifti) * hm
			end
			svi /= (hm * hm)
			drifti = drifti / (2 * J[i] * hm)
			#actual peclet cond = 0.5*svi/Jm[i+1] > drifti <=> 0.5*svi/Jm[i+1] < drifti*hm/2J[i] <=> drift/svi > J[i]/Jm[i+1]
			A1iuj[index] = -dt * (0.5 * svi / (Jm[i+1]) + drifti)
			A1ij[index] = -dt * (-svi * 0.5 * (1 / Jm[i+1] + 1 / Jm[i]) - r * 0.5)
			A1ilj[index] = -dt * (0.5 * svi / (Jm[i]) - drifti)
			if exponentialFitting == "OSullivan"
				indicatorAl = ifelse(A1ilj[index] > 0, 1.0, 0.0)
				indicatorAr = ifelse(A1iuj[index] > 0, 1.0, 0.0)
				A1ij[index] += dt * (-indicatorAr / Jm[i] + indicatorAl / Jm[i+1]) * g * S[i] / (hm)
				if indicatorAl > 0 || indicatorAr > 0
					A1ilj[index] += -dt * (drifti) + dt * (-indicatorAr / Jm[i]) * g * S[i] / (hm)
					A1iuj[index] += -dt * (-drifti) + dt * (indicatorAl / Jm[i+1]) * g * S[i] / (hm)
				end
			end
		end
	end

	j = 1
	jm = (j - 1) * M
	driftj = kappa * (theta - V[j])
	for i ∈ 2:M-1
		index = i + jm
		A2iju[index] = -dt * (driftj / (JVm[j+1] * hl))
		A2ij[index] = -dt * (-r * 0.5) - A2iju[index]
	end
	j = L
	jm = (j - 1) * M
	driftj = kappa * (theta - V[j])
	@simd for i ∈ 2:M-1
		index = i + jm
		A2ijl[index] = dt * (driftj / (JVm[j] * hl))
		A2ij[index] = -dt * (-r * 0.5) - A2ijl[index]
	end
	for j ∈ 2:L-1
		driftj = kappa * (theta - V[j])
		svj = sigma^2 * V[j] / JV[j]
		if (exponentialFitting == "UpwindingThreshold") || exponentialFitting == "Partial Exponential Fitting" || (exponentialFitting == "Foulon" && (V[j] == 0 || V[j] > 1))
			if driftj != 0 && abs(driftj * hl / svj) > 1.0
				svj = driftj * hl / tanh(driftj * hl / svj)
			end
		elseif exponentialFitting == "Full"
			svj = driftj * hl / tanh(driftj * hl / svj)
		elseif exponentialFitting == "FullUpwind"
			svj = svj + abs(driftj) * hl
		end

		svj        /= (hl * hl)
		driftj     /= (2 * JV[j] * hl)
		jm         = (j - 1) * M
		rCoeff     = -dt * 0.25 * rho * sigma * V[j] / (JV[j] * hl * hm)
		a2ijuCoeff = -dt * (0.5 * svj / (JVm[j+1]) + driftj)
		a2ijCoeff  = -dt * (-r * 0.5 - svj * 0.5 * (1.0 / JVm[j+1] + 1.0 / JVm[j]))
		a2ijlCoeff = -dt * (svj * 0.5 / (JVm[j]) - driftj)
		if exponentialFitting == "OSullivan"
			indicatorAl = ifelse(a2ijlCoeff > 0, 1.0, 0.0)
			indicatorAr = ifelse(a2ijuCoeff > 0, 1.0, 0.0)
			a2ijCoeff += dt * (-indicatorAr / JVm[j] + indicatorAl / JVm[j+1]) * kappa * (theta - V[j]) / (hl)
			if indicatorAl > 0 || indicatorAr > 0
				a2ijlCoeff += -dt * driftj - dt * (-indicatorAr / JVm[j]) * kappa * (theta - V[j]) / (hl)
				a2ijuCoeff += dt * driftj - dt * (indicatorAl / JVm[j+1]) * kappa * (theta - V[j]) / (hl)
			end
		end
		@simd for i ∈ 2:M-1
			index = i + jm
			A2iju[index] = a2ijuCoeff
			A2ij[index] = a2ijCoeff
			A2ijl[index] = a2ijlCoeff
			rij[index] = rCoeff * S[i] / J[i]
		end
	end
	return rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju
end


function makeSparseSystemLU(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
	indicesI = zeros(Int, L * M)
	indicesJ = zeros(Int, L * M)
	@inbounds for j ∈ 1:L, i ∈ 1:M
		index = i + (j - 1) * M
		indicesI[index] = index
		indicesJ[index] = index
	end
	A1d = sparse(indicesI, indicesJ, A1ij, M * L, M * L)
	A2d = sparse(indicesI, indicesJ, A2ij, M * L, M * L)
	indicesI = zeros(Int, L * M)
	indicesJ = zeros(Int, L * M)
	@inbounds for j ∈ 1:L
		@inbounds @simd for i ∈ 2:M
			index = i + (j - 1) * M
			indicesI[index] = index
			indicesJ[index] = index - 1
		end
		indicesI[1+(j-1)*M] = 1 + (j - 1) * M
		indicesJ[1+(j-1)*M] = 1 + (j - 1) * M
	end
	A1dl = sparse(indicesI, indicesJ, A1ilj, M * L, M * L)
	indicesI = zeros(Int, L * M)
	indicesJ = zeros(Int, L * M)
	@inbounds for j ∈ 1:L
		@inbounds @simd for i ∈ 1:M-1
			index = i + (j - 1) * M
			indicesI[index] = index
			indicesJ[index] = index + 1
		end
		indicesI[M+(j-1)*M] = 1 + (j - 1) * M
		indicesJ[M+(j-1)*M] = 1 + (j - 1) * M
	end
	A1du = sparse(indicesI, indicesJ, A1iuj, M * L, M * L)
	indicesI = zeros(Int, L * M)
	indicesJ = zeros(Int, L * M)
	@inbounds for i ∈ 1:M
		@inbounds @simd for j ∈ 2:L
			index = i + (j - 1) * M
			indicesI[index] = index
			indicesJ[index] = index - M
		end
		indicesI[i] = i
		indicesJ[i] = i
	end
	A2dl = sparse(indicesI, indicesJ, A2ijl, M * L, M * L)
	indicesI = zeros(Int, L * M)
	indicesJ = zeros(Int, L * M)
	@inbounds for i ∈ 1:M
		@inbounds @simd for j ∈ 1:L-1
			index = i + (j - 1) * M
			indicesI[index] = index
			indicesJ[index] = index + M
		end
		indicesI[i+(L-1)*M] = i + (L - 1) * M
		indicesJ[i+(L-1)*M] = i + (L - 1) * M
	end
	A2du = sparse(indicesI, indicesJ, A2iju, M * L, M * L)

	indicesI = zeros(Int, L * M)
	indicesJ = zeros(Int, L * M)
	@inbounds for i ∈ 1:M-1
		@inbounds @simd for j ∈ 1:L-1
			index = i + (j - 1) * M
			indicesI[index] = index
			indicesJ[index] = index + 1 + M
		end
		indicesI[i+(L-1)*M] = i + (L - 1) * M
		indicesJ[i+(L-1)*M] = i + (L - 1) * M
	end
	@inbounds for j ∈ 1:L
		indicesI[M+(j-1)*M] = M + (j - 1) * M
		indicesJ[M+(j-1)*M] = M + (j - 1) * M
	end
	A0uu = sparse(indicesI, indicesJ, rij, M * L, M * L)
	indicesI = zeros(Int, L * M)
	indicesJ = zeros(Int, L * M)
	@inbounds for i ∈ 2:M
		@inbounds @simd for j ∈ 2:L
			index = i + (j - 1) * M
			indicesI[index] = index
			indicesJ[index] = index - 1 - M
		end
		indicesI[i] = i
		indicesJ[i] = i
	end
	for j ∈ 1:L
		indicesI[1+(j-1)*M] = 1 + (j - 1) * M
		indicesJ[1+(j-1)*M] = 1 + (j - 1) * M
	end
	A0ll = sparse(indicesI, indicesJ, rij, M * L, M * L)
	indicesI = zeros(Int, L * M)
	indicesJ = zeros(Int, L * M)
	@inbounds for i ∈ 2:M
		@inbounds @simd for j ∈ 1:L-1
			index = i + (j - 1) * M
			indicesI[index] = index
			indicesJ[index] = index - 1 + M
		end
		indicesI[i+(L-1)*M] = i + (L - 1) * M
		indicesJ[i+(L-1)*M] = i + (L - 1) * M
	end
	for j ∈ 1:L
		indicesI[1+(j-1)*M] = 1 + (j - 1) * M
		indicesJ[1+(j-1)*M] = 1 + (j - 1) * M
	end
	A0lu = sparse(indicesI, indicesJ, -rij, M * L, M * L)
	indicesI = zeros(Int, L * M)
	indicesJ = zeros(Int, L * M)
	@inbounds for i ∈ 1:M-1
		@inbounds @simd for j ∈ 2:L
			index = i + (j - 1) * M
			indicesI[index] = index
			indicesJ[index] = index + 1 - M
		end
		indicesI[i] = i
		indicesJ[i] = i
	end
	@inbounds for j ∈ 1:L
		indicesI[M+(j-1)*M] = M + (j - 1) * M
		indicesJ[M+(j-1)*M] = M + (j - 1) * M
	end
	A0ul = sparse(indicesI, indicesJ, -rij, M * L, M * L)
	return A0ll, A0uu, A0ul, A0lu, A2d, A2dl, A2du, A1d, A1dl, A1du
end
function makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
	A0ll, A0uu, A0ul, A0lu, A2d, A2dl, A2du, A1d, A1dl, A1du = makeSparseSystemLU(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
	A0 = A0ll + A0uu + A0ul + A0lu
	A2 = A2d + A2dl + A2du
	A1 = A1d + A1dl + A1du
	return A0, A1, A2
end


function updatePayoffExplicitTrans(F::Vector{T}, useDirichlet::Bool, lbValue::T, M::Int, L::Int) where T
	if useDirichlet
		@simd for j ∈ 1:L
			F[1+(j-1)*M] = lbValue
		end
	end
end

function explicitStep(rij::Vector{T}, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Y0::Vector{T}, M::Int, L::Int) where T
	explicitStep(1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
end

function explicitStep1(a::Real, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, F::Vector{T}, Y0::Vector{T}, Y1::Vector{T}, M::Int, L::Int) where {T}
	@inbounds for j ∈ 1:L
		jm = (j - 1) * M
		i = 1
		index = i + jm
		Y1[index] = Y0[index] - a * (A1ij[index] * F[index] + A1iuj[index] * F[index+1])
		i = M
		index = i + jm
		Y1[index] = Y0[index] - a * (A1ij[index] * F[index] + A1ilj[index] * F[index-1])
		@inbounds @simd for i ∈ 2:M-1
			index = i + jm
			Y1[index] = Y0[index] - a * (A1ij[index] * F[index] + A1iuj[index] * F[index+1] + A1ilj[index] * F[index-1])
		end
	end
end

function explicitStep2(a::Real, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Y0::Vector{T}, Y1::Vector{T}, M::Int, L::Int) where {T}
	@inbounds for j ∈ 1:L
		jm = (j - 1) * M
		i = 1
		index = i + jm
		Y1[index] = Y0[index]
		i = M
		index = i + jm
		Y1[index] = Y0[index]
	end
	j = 1
	jm = (j - 1) * M
	@inbounds @simd for i ∈ 2:M-1
		index = i + jm
		Y0ij = Y0[index] - a * (A2ij[index] * F[index] + A2iju[index] * F[index+M])
		Y1[index] = Y0ij
	end
	j = L
	jm = (j - 1) * M
	@inbounds @simd for i ∈ 2:M-1
		index = i + jm
		Y0ij = Y0[index] - a * (A2ij[index] * F[index] + A2ijl[index] * F[index-M])
		Y1[index] = Y0ij
	end
	@inbounds for j ∈ 2:L-1
		jm = (j - 1) * M
		@inbounds @simd for i ∈ 2:M-1
			index = i + jm

			Y0ij = Y0[index] - a * (A2ij[index] * F[index] + A2iju[index] * F[index+M] + A2ijl[index] * F[index-M])
			Y1[index] = Y0ij
		end
	end
end


function explicitStep(a::Real, rij::Vector{T}, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Y0::Vector{T}, Y1::Vector{T}, M::Int, L::Int) where T
	explicitStep1(a, A1ilj, A1ij, A1iuj, F, Y0, Y1, M, L)
	explicitStep2(a, A2ijl, A2ij, A2iju, F, Y1, Y1, M, L)
	@inbounds for j ∈ 2:L-1
		jm = (j - 1) * M
		@inbounds @simd for i ∈ 2:M-1
			index = i + jm
			Y0ij = a * rij[index] * (F[index+1+M] - F[index+1-M] + F[index-1-M] - F[index-1+M])
			Y1[index] -= Y0ij
		end
	end
end

function RKGStep(
	s::Int,
	a::Vector{T},
	b::Vector{T},
	w0::Real,
	w1::Real,
	rij::Vector{T},
	A1ilj::Vector{T},
	A1ij::Vector{T},
	A1iuj::Vector{T},
	A2ijl::Vector{T},
	A2ij::Vector{T},
	A2iju::Vector{T},
	F::Vector{T},
	Yjm2::Vector{T},
	Yjm::Vector{T},
	Yj::Vector{T},
	useDirichlet::Bool,
	lbValue::Real,
	M::Int,
	L::Int,
) where T
	mu1b = 3 * b[2] / b[1] * w1
	explicitStep(mu1b, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Yjm, M, L)
	updatePayoffExplicitTrans(Yjm, useDirichlet, lbValue, M, L)
	MY0 = (Yjm - F) / mu1b
	Yjm2 .= F
	for j ∈ 2:s
		muu = (2 * j + 1) * b[j+1] / (j * b[j])
		muj = muu * w0
		mujb = muu * w1
		gammajb = -a[j] * mujb
		nuj = -(j + 1) * b[j+1] / (j * b[j-1])

		@. Yj = muj * Yjm + nuj * Yjm2 + (1 - nuj - muj) * F + gammajb * MY0 # + mujb*MYjm
		explicitStep(mujb, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, Yjm, Yj, Yj, M, L)
		updatePayoffExplicitTrans(Yj, useDirichlet, lbValue, M, L)
		Yjm2, Yjm = Yjm, Yjm2
		Yjm, Yj = Yj, Yjm
	end
	return Yjm
end



#payoff = VanillaEuropean / VanillaAmerican / VVIX /Barrier
function priceFDM2D(
	payoff,
	p::HestonParams{Float64},
	spot,
	forecastCurve,
	discountCurve,
	N;
	damping = "None",
	method = "RKG",
	vDisc = "Sinh",
	vmax = 0.0,
	sDisc = "Linear",
	Smax = 0.0,
	Sdev = 4.0,
	exponentialFitting = "None",
	smoothing = "Kreiss",
	lambdaS = 0.25,
	lambdaV = 2.0,
	rklStages = 0,
	printGamma = false,
	lsSolver = "SOR",
	eigenPlot = nothing,
	sPlot = nothing,
	plotStep = 0,
	rkcShift = 10.0,
	epsilonV = 1e-3,
)
	payoffVector = currentValue(payoff)
	M = size(payoffVector[1], 1)
	L = size(payoffVector[1], 2)
	T = lastEvaluationTime(payoff)
	kappa = p.κ
	theta = p.θ
	sigma = p.σ
	v0 = p.v0
	rho = p.ρ
	#sDisc "Sinh" Linear", "Exp", "Collocation"
	upwindingThreshold = 1.0
	isConstant = false #if rates are constants
	dChi = 4 * kappa * theta / (sigma * sigma)
	chiN = 4 * kappa * exp(-kappa * T) / (sigma * sigma * (1 - exp(-kappa * T)))
	ncx2 = NoncentralChisq(dChi, v0 * chiN)
	if vmax == 0
		vmax = quantile(ncx2, 1 - epsilonV) * exp(-kappa * T) / chiN
	end
	vmin = quantile(ncx2, epsilonV) * exp(-kappa * T) / chiN
	vmin = max(vmin, 1e-3)
	vmin = 0.0
	# println("vmin ",vmin, " ",vmax)
	V = collect(range(0.0, stop = vmax, length = L))
	hl = V[2] - V[1]
	JV = ones(L)
	JVm = ones(L)

	if vDisc == "Sinh"
		vscale = v0 * lambdaV
		u = collect(range(0.0, stop = 1.0, length = L))
		c1 = asinh((vmin - v0) / vscale)
		c2 = asinh((vmax - v0) / vscale)
		V = @. v0 + vscale * sinh((c2 - c1) * u + c1)
		hl = u[2] - u[1]
		JV = @. vscale * (c2 - c1) * cosh((c2 - c1) * u + c1)
		JVm = @. vscale * (c2 - c1) * cosh((c2 - c1) * (u - hl / 2) + c1)
	elseif vDisc == "Foulon"
		vscale = vmax * lambdaV
		u = collect(range(0.0, stop = 1.0, length = L))
		c1 = asinh(0.0)
		c2 = asinh((vmax - vmin) / vscale)
		V = @. vmin + vscale * sinh((c2 - c1) * u + c1)
		hl = u[2] - u[1]
		JV = @. vscale * (c2 - c1) * cosh((c2 - c1) * u + c1)
		JVm = @. vscale * (c2 - c1) * cosh((c2 - c1) * (u - hl / 2) + c1)
	end
	Xspan = Sdev * sqrt(theta * T)
	logK = log(spot) #use spot.
	gDF = forecastCurve(T) #growth rate*T
	rDF = discountCurve(T)

	Xmin = logK - Xspan - log(gDF) - 0.5 * v0 * T
	Xmax = logK + Xspan - log(gDF) - 0.5 * v0 * T
	#TODO eventually truncate for barrier payoffs
	B = 0.0
	hm = 0.0

	if sDisc == "Exp"
		Smin = exp(Xmin)
		Smax = exp(Xmax)
		X = collect(range(Xmin, stop = Xmax, length = M))
		hm = X[2] - X[1]
		S = exp.(X)
		J = exp.(X)
		Jm = @. exp(X - hm / 2)
	else #if sDisc == "Linear"
		if Smax == 0
			Smax = exp(Xmax)
		end
		S = collect(range(B, stop = Smax, length = M))
		X = S
		hm = X[2] - X[1]
		J = ones(M)
		Jm = ones(M)
	end
	Smin = S[1]
	Smax = S[end]
	evaluateFDM2DPayoff(payoff, T, S, V)

	useDirichlet = false #for barrier
	ti = T
	dt = T / N
	lbValue = 0.0
	for iv ∈ eachindex(payoffVector)
		F = reshape(payoffVector[iv], M * L)
		updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
		payoffVector[iv] = reshape(F, (M, L))
	end
	etime = @elapsed begin
		if method == "EU"
			for n ∈ 1:N
				ti -= dt
				gDFi = forecastCurve(ti)
				rDFi = discountCurve(ti)
				g = log(gDFi / gDF) / dt
				r = log(rDFi / rDF) / dt
				println("g=", g, " r=", r, " ti=", ti)
				rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(exponentialFitting, upwindingThreshold, dt, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, g, useDirichlet, M, L)
				A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
				lbValue = 0.0# computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
				for iv ∈ eachindex(payoffVector)
					F = reshape(payoffVector[iv], M * L)
					updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
					F = (I + A0 + A1 + A2) \ F
					payoffVector[iv] = reshape(F, (M, L))
				end
				evaluateFDM2DPayoff(payoff, ti, S, V)
				rDF = rDFi
				gDF = gDFi
			end
		elseif method == "RKG2"
			g = log(1.0 / gDF) / T
			r = log(1.0 / rDF) / T
			rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(exponentialFitting, upwindingThreshold, dt, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, g, useDirichlet, M, L)
			maxnorm = maximum(abs.(A1ij[2:end-1] + A2ij[2:end-1]) + abs.(A1ilj[2:end-1] + abs.(A1iuj[2:end-1]) + abs.(A2ijl[2:end-1]) + abs.(A2iju[2:end-1]) + 4.0 .* abs.(rij[2:end-1])))
			dtexplicit = dt / maxnorm
			s, a, b, w0, w1 = initRKGCoeffsFromExplicitStepSize(dtexplicit, dt, stages = rklStages)

			Y0 = zeros(L * M)
			Y1 = zeros(L * M)
			Y2 = zeros(L * M)
			for n ∈ 1:N
				ti -= dt
				gDFi = forecastCurve(ti)
				rDFi = discountCurve(ti)
				g = log(gDFi / gDF) / dt
				r = log(rDFi / rDF) / dt

				lbValue = 0.0 #computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti + dt * 0.5)
				for iv ∈ eachindex(payoffVector)
					F = reshape(payoffVector[iv], M * L)
					updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
					F .= RKGStep(s, a, b, w0, w1, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, Y1, Y2, useDirichlet, lbValue, M, L)
					payoffVector[iv] = reshape(F, (M, L))
				end

				if !isnothing(sPlot) && n == plotStep
					println("plotting t ", ti)
					Payoff = reshape(F, (M, L))
					# spl = Spline2D(S, V, Payoff; s = 0.0)
					surface!(sPlot, V, S, Payoff)
				end
				evaluateFDM2DPayoff(payoff, ti, S, V)
				rDF = rDFi
				gDF = gDFi
			end
		elseif method == "RKC2"
			Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S)
			rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(exponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
			#dtexplicit = dt / (maximum(abs.(A1ij+A2ij))+maximum(abs.(A1ilj))+maximum(abs.(A1iuj))+maximum(abs.(A2ilj))+maximum(abs.(A2iuj))+4*maximum(abs.(rij)))
			dtexplicit = dt / max(maximum(A1ij + A2ij))
			# if sDisc == "Linear" || (sDisc == "Sinh" && lambdaS >= 1)
			#     dtexplicit /= 2 #lambdaS
			# else

			if !isnothing(eigenPlot)
				isDiagDom = 0
				for i ∈ eachindex(A1ij)
					if (A1ij[i] + A2ij[i]) + 1.0 < abs.(A1ilj[i]) + abs.(A1iuj[i]) + abs.(A2ijl[i]) + abs.(A2iju[i])
						isDiagDom = i
						println(i, " not dom ", (A1ij[i] + A2ij[i]), " >=", abs.(A1ilj[i]) + abs.(A1iuj[i]) + abs.(A2ijl[i]) + abs.(A2iju[i]))

					end
				end
				# isNonDiagPos = 0
				# for i=eachindex(A1ij)
				#     if A1ilj[i] > 1e-6 || A1iuj[i] > 1e-6|| A2ijl[i] > 1e-6 || A2iju[i] > 1e-6 || rij[i] > 1e-6
				#         isNonDiagPos = i
				#         println(i, " ",A1ilj[i], " ", A1iuj[i], " ", A2ijl[i], " ",A2iju[i], " ",rij[i])

				#     end
				# end
				A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
				A = Matrix((A0 + A1 + A2))
				println("built matrix, isDiagDom=", isDiagDom, " will compute eigen dte=", dtexplicit, "dt=", dt)
				eigA = eigvals(A)
				println("built eigenvalues, will plot")
				plot!(eigenPlot, eigA, seriestype = :scatter, label = string(exponentialFitting), ms = 3, msw = 0, markeralpha = 0.33)
				printEigen = false
			end
			# end
			ep = rkcShift
			s = computeRKCStages(dtexplicit, dt, ep)

			#s= Int(floor(sqrt(1+6*dt/dtexplicit)))  #Foulon rule as dtExplicit=lambda/2
			if rklStages > 0
				s = rklStages
			end
			#println(maximum(A1ij + A2ij), " ", 2 * minimum(A1ilj), " ", 2 * minimum(A2ijl), " ", maximum(A1ij + A2ij + A1iuj + A2iju + A1ilj + A2ijl), " DTE ", dtexplicit, " NE ", T / dtexplicit, " s ", s)
			# println("s=",s)
			w0 = 1 + ep / s^2
			_, tw0p, tw0p2 = chebPoly(s, w0)
			w1 = tw0p / tw0p2
			b = zeros(s)
			for jj ∈ 2:s
				_, tw0p, tw0p2 = chebPoly(jj, w0)
				b[jj] = tw0p2 / tw0p^2
			end
			b[1] = b[2]
			a = zeros(s)
			for jj ∈ 2:s
				tw0, _, _ = chebPoly(jj - 1, w0)
				a[jj-1] = (1 - b[jj-1] * tw0)
			end
			Y = Vector{Array{Float64, 1}}(undef, s)
			Y0 = zeros(L * M)
			Y1 = zeros(L * M)
			Y2 = zeros(L * M)
			for n ∈ 1:N
				lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt * 0.5)
				#  RKLStep(s, a, b, w1, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y, useDirichlet, lbValue, M, L)
				F .= RKCStep(s, a, b, w0, w1, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, Y1, Y2, useDirichlet, lbValue, M, L)
				ti -= dt
				if !isnothing(sPlot) && n == plotStep
					println("plotting t ", ti)
					Payoff = reshape(F, (M, L))
					# spl = Spline2D(S, V, Payoff; s = 0.0)
					surface!(sPlot, V, S, Payoff)
				end

				if n < N
					Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S)
					rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(exponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
				end

			end
		end
	end  #elapsed
	return S, V, currentValue(payoff)
end

#to evaluate payoff:
#	spl = Spline2D(S, V, payoff.value[1]; s = 0.0)
#	for i ∈ 1:length(spotArray)
#		spot = spotArray[i]
#		price = Dierckx.evaluate(spl, spot, v0)
#		err = price - priceArray[i]
#		l2 += err^2
#		if abs(err) > maxerr && i > length(spotArray) / 4 && i < length(spotArray) * 3 / 4
#			maxerr = abs(err)
#		end
#   end


#=
for (k,strike) = enumerate(strikesVIX)

=#