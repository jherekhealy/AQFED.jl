using AQFED.Math
import AQFED.Basket: Levy2MMBasketPricer, SLN3MMBasketPricer, forwardAndVariance
export PearsonPricer, priceVanillaSpread
#Numerical integration for spread options under the Black-Scholes model.

struct PearsonPricer
	n::Int
end

function inner(S, S1, S2, sig1, sig2, rho, r, q1, q2, tau, X)
	A1  = r * (1.0 - rho * sig2 / sig1)
	A2  = q2 - q1 * rho * sig2 / sig1
	A3  = 0.5 * rho * sig2 * (sig1 - rho * sig2)
	A   = (A1 - A2 + A3) * tau
	sig = sig2 * sqrt((1.0 - rho * rho) * tau)
	m1  = log(S1) + (r - q1 - 0.5 * sig1 * sig1) * tau
	m2  = log(S2) + (r - q2 - 0.5 * sig2 * sig2) * tau
	M2  = m2 + (rho * sig2 / sig1) * (log(S) - m1)
	x1  = 0
	x2  = 0

	if (S + X > 0)
		if (1 == rho * rho)
			return if M2 > log(S + X)
				exp(A) * S2 * (S / S1)^(rho * sig2 / sig1) - (S + X)
			else
				zero(S2)
			end
		else

			x1 = (M2 + sig * sig - log(S + X)) / sig
			x2 = x1 - sig
			return exp(A) * S2 * (S / S1)^(rho * sig2 / sig1) * normcdf(x1) - (S + X) * normcdf(x2)
		end
	else
		return exp(A) * S2 * (S / S1)^(rho * sig2 / sig1) - (S + X)
	end
end

function oneStepGridSpread(pricer::PearsonPricer, S1, S2, vol1, vol2, rho, r, q1, q2, X, T)

	N    = pricer.n
	N2   = 2 * N
	sig1 = if vol1 > 0
		vol1
	else
		1e-10
	end
	sig2 = if vol2 > 0
		vol2
	else
		1e-10
	end
	sigT = sig1 * sqrt(T)
	h    = 5 * sigT / N
	w    = sigT / h
	m    = log(S1) + T * (r - q1 - 0.5 * sig1 * sig1)
	c    = rho * sig2
	A    = T * (r * (1.0 - c / sig1) - (q2 - q1 * c / sig1) + 0.5 * rho * sig2 * (sig1 - c))


	if (0.0 >= T)
		return if 0.0 > T
			NaN
		else
			max(S2 - S1, 0.0)
		end
	end
	if (vol1 < 0 || vol2 < 0 || S1 < 0 || S2 < 0 || rho < -1 || 1 < rho || N <= 0)
		return NaN
	end
	x2 = exp(m - N * h)
	z2 = inner(x2, S1, S2, sig1, sig2, rho, r, q1, q2, T, X)
	u2 = normcdf(-N / w - sigT)
	v2 = normcdf(-N / w)

	value = zero(S1)
	for i ∈ -N:N-1
		x1 = x2
		z1 = z2
		u1 = u2
		v1 = v2

		x2 = exp(m + h * (i + 1))
		z2 = inner(x2, S1, S2, sig1, sig2, rho, r, q1, q2, T, X)
		u2 = normcdf((i + 1) / w - sigT)
		v2 = normcdf((i + 1) / w)

		a = (z2 - z1) / (x2 - x1)
		b = (x2 * z1 - x1 * z2) / (x2 - x1)

		value += a * (u2 - u1) * S1 * exp((r - q1) * T) + b * (v2 - v1)
	end
	#println("vlaue ",value)
	x1 = exp(m - N * h)

	aN = exp(A) * S2 / (S1)^(c / sig1)
	bN = z2 - aN * (x2)^(c / sig1)
	a0 = exp(A) * S2 / (S1)^(c / sig1)
	b0 = inner(x1, S1, S2, sig1, sig2, rho, r, q1, q2, T, X) -
		 a0 * (x1)^(c / sig1)

	term1 = a0 * normcdf(-N / w - c * sqrt(T)) * exp(c * m / sig1 + 0.5 * c * c * T)
	term2 = b0 * normcdf(-N / w)
	term3 = aN * (1.0 - normcdf(N / w - c * sqrt(T))) * exp(c * m / sig1 + 0.5 * c * c * T)
	term4 = bN * (1.0 - normcdf(N / w))

	value += term1 + term2 + term3 + term4

	return exp(-r * T) * value
end

function margrabe0(isCall::Bool, T, S1, S2, vol1, vol2, rho, r, q1, q2)
	return if isCall

		vol  = sqrt((vol1^2) + (vol2^2) - 2.0 * rho * vol1 * vol2)
		d1   = (log(S2 / S1) + (q1 - q2 + 0.5 * (vol^2)) * T) /
		(vol * sqrt(T))
		d2   = d1 - vol * sqrt(T)
		Nd1  = normcdf(d1)
		Nd2  = normcdf(d2)
		Npd1 = normpdf(d1)
		Npd2 = normpdf(d2)
		div1 = exp(-q1 * T)
		div2 = exp(-q2 * T)

		S2 * sqrt(T) * Npd1 * div2 / vol

		ifelse(T <= 0, max(S2 - S1, 0), div2 * S2 * Nd1 - div1 * S1 * Nd2)
	else
		margrabe0(!isCall, T, S2, S1, vol2, vol1, rho, r, q2, q1)
	end
end

function priceVanillaSpread(pricer::PearsonPricer, isCall::Bool, strike, discountFactor, spot::AbstractArray{<:T},
	forward::AbstractArray{TV}, #forward to option maturity
	totalVariance::AbstractArray{<:T}, #vol^2 * τ
	correlation::T,
) where {T, TV}
	S1 = spot[1]
	S2 = spot[2]
	tau = 1.0
	vol1 = sqrt(totalVariance[1])
	vol2 = sqrt(totalVariance[2])
	r = -log(discountFactor)
	q1 = -log(forward[1] / spot[1]) + r
	q2 = -log(forward[2] / spot[2]) + r
	if iszero(strike) || abs(correlation) > 1 || pricer.n <= 0
		margrabe0(isCall, tau, S1, S2, vol1, vol2, correlation, r, q1, q2)
	else
		if isCall
			oneStepGridSpread(pricer, S1, S2, vol1, vol2, correlation, r, q1, q2, strike, tau)
		else
			#K-(S2-S1) to  S2-S1-K 
			oneStepGridSpread(pricer, S2, S1, vol2, vol1, correlation, r, q2, q1, -strike, tau)
		end
	end
end

function priceAsianSpread(pricer::PearsonPricer, isCall::Bool, strikePct, discountFactor, spot, forwards::AbstractArray{TV}, totalVariances::AbstractArray{<:T}, weight::AbstractArray{<:T}; momentsPricer = Levy2MMBasketPricer()) where {T, TV}

	#forward, v2 = forwardAndVariance(momentsPricer, forwards, totalVariances, weights, correlationp)

	indexPositive = findfirst(x -> x > 0, weight)
	lastZeroVarianceIndex = findlast(v -> v < 1e-16, totalVariances)
	knownPartNegative = zero(TV)
	knownPartPositive = zero(TV)
	if !isnothing(lastZeroVarianceIndex)
		if lastZeroVarianceIndex < indexPositive
			knownPartNegative = forwards[1:lastZeroVarianceIndex]' * weight[1:lastZeroVarianceIndex]
		else
			knownPartNegative = forwards[1:indexPositive-1]' * weight[1:indexPositive-1]
			knownPartPositive = forwards[indexPositive:lastZeroVarianceIndex]' * weight[indexPositive:lastZeroVarianceIndex]
		end
		weight = weight[lastZeroVarianceIndex+1:end]
		forwards = forwards[lastZeroVarianceIndex+1:end]
		totalVariances = totalVariances[lastZeroVarianceIndex+1:end]
		indexPositive = indexPositive - lastZeroVarianceIndex
	end

	nAsset = length(totalVariances)
	correlation = zeros(eltype(totalVariances), (nAsset, nAsset))
	for (i, vi) in enumerate(totalVariances)
		for j ∈ 1:(i-1)
			vj = totalVariances[j]
			if vi != 0 && vj != 0
				correlation[i, j] = min(vi, vj) / sqrt(vi * vj)
				# else is zero
			end
		end
		correlation[i, i] = one(eltype(totalVariances))
	end
	for i ∈ 1:nAsset
		for j ∈ i+1:nAsset
			correlation[i, j] = correlation[j, i]
		end
	end
	strike = -(knownPartPositive + knownPartNegative * strikePct)
	weights = copy(weight)
	weights[1:indexPositive-1] *= strikePct
	forwardsp = forwards[indexPositive:end]
	weightsp = weights[indexPositive:end]
	correlationp = correlation[indexPositive:end, indexPositive:end]
	totalVariancesp = totalVariances[indexPositive:end]
	forwardp, v2p, shiftp = forwardAndVariance(momentsPricer, 0.0, forwardsp, totalVariancesp, weightsp, correlationp)
	forwardsn = forwards[1:indexPositive-1]
	weightsn = abs.(weights[1:indexPositive-1])
	correlationn = correlation[1:indexPositive-1, 1:indexPositive-1]
	totalVariancesn = totalVariances[1:indexPositive-1]
	forwardn, v2n, shiftn = forwardAndVariance(momentsPricer, 0.0, abs.(forwardsn), totalVariancesn, abs.(weightsn), correlationn)
	forwardn = abs(forwardn)
	EApAn = zero(T)
	for i ∈ 1:nAsset
		for j ∈ 1:nAsset
			if (i < indexPositive && j >= indexPositive)
				EApAn += forwards[i] * forwards[j] * abs(weights[i] * weights[j]) * exp(correlation[i, j] * sqrt(totalVariances[i] * totalVariances[j]))
			end
		end
	end
	#println("f-k ",forwardp-forwardn)
	#EApAn = E[Xp+shiftp Xn+shifn] = E[exp(Mp+0.5*v2p+0.5*v2n+Mn)+shiftp*exp(Mp+0.5*v2p)+shiftn*exp(Mn+0.5*v2n)]
	# = (forwardp-shiftp)*(forwardn-shiftn)*exp(rho*sqrt(v2n*v2p))+shiftp*forwardn+shiftn*forwardp
	rho = log((EApAn - shiftp * (forwardn - shiftn) - shiftn * (forwardp - shiftp) - shiftp * shiftn) / ((forwardp - shiftp) * (forwardn - shiftn))) / sqrt(v2p * v2n)
	#rho = log(EApAn / (forwardp * forwardn)) / sqrt(v2p * v2n)
	#NOTE: the shift adj does not seem to work well, would need to rederive the Pearson formulae for the SLN case.
	forwardp -= shiftp
	forwardn -= shiftn
	strike -= shiftp - shiftn
	return priceVanillaSpread(pricer, isCall, strike, discountFactor, [spot - shiftn, spot - shiftp], [forwardn, forwardp], [v2n, v2p], rho)
end
