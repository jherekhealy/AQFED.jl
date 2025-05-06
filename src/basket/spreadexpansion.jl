export priceEuropeanSpread

function priceEuropeanSpread(
	p::VorstGeometricExpansion,
	isCall::Bool,
	strikePct::T, #Strike in percent: 1.0 for 100%
	discountFactor::T, #discount factor to payment
	spot::AbstractArray{<:T},
	forward::AbstractArray{TV}, #forward to option maturity
	totalVariance::AbstractArray{<:T}, #vol^2 * τ
	weight::AbstractArray{<:T}, #for now assume first weights are < 0 and laast wieghts are > 0 for a spread. This is call like spread.
	correlation::Matrix{TV},
)::T where {T, TV}
	signc = if isCall
		1
	else
		-1
	end
	nAsset = length(totalVariance)
	indexPositive = findfirst(x -> x > 0, weight)
	# price = priceEuropean(p, isCall, strike, discountFactor, spots, forward, totalVariance, weight, correlation)
	forwardp = forward[indexPositive:end]
	weightp = weight[indexPositive:end]

	forwardn = forward[1:indexPositive-1]
	weightn = weight[1:indexPositive-1]
	A = weight' * forward
	Ap = weightp' * forwardp
	An = weightn' * forwardn
	ap = @. (weightp * forwardp) / Ap
	an = @. -(weightn * forwardn) / An #An is neg, an is neg
	a = vcat(an, ap)
	vtilde2 = zero(TV)
	vtilde2p = zero(TV)
	vtilde2pn = zero(TV)
	vtilde2n = zero(TV)
	vtilde2np = zero(TV)
	covar = zeros(TV, size(correlation))
	for i ∈ 1:nAsset
		vi = totalVariance[i]
		for j ∈ 1:nAsset
			vj = totalVariance[j]
			covarij = sqrt(vi * vj) * correlation[i, j]
			covar[i, j] = covarij
			product = a[i] * a[j] * covarij
			vtilde2 += product
			if i >= indexPositive
				if j >= indexPositive
					vtilde2p += product
				else
					vtilde2pn += product
				end
			else #i < indexPositive 
				if j < indexPositive
					vtilde2n += product
				else
					vtilde2np += product
				end
			end
		end
	end
	if (Ap <= zero(T))
		return max(signc * (Ap + An), zero(T))
	end
	#println("A ",A," ",vtilde2," ",vtilde2p," ",vtilde2n, " ",vtilde2pn, " ",vtilde2np)

	#mtilde = (-vtilde2p + vtilde2n) / 2 - vtilde2n - vtilde2pn #vtilde2pn is negative due to ani*apj terms, this is equivalent to:
	mtilde = -vtilde2 / 2
	vtilde = sqrt(vtilde2)


	strikeScaled = -strikePct * An / Ap #positive
	astar = copy(a)
	@. astar[1:indexPositive-1] *= strikeScaled
	logStrikeScaled = log(strikeScaled)
	d1 = (mtilde - logStrikeScaled + vtilde2) / vtilde
	d2 = d1 - vtilde
	price = signc * Ap * (normcdf(signc * d1) - strikeScaled * normcdf(signc * d2))
	#above is order 0 price = Geometric approx with adjusted forward to match basket forward (instead of adjusting the strike as in Gentle)
	if p.order > 0
		tvi = zeros(TV, nAsset)
		tvni = zeros(TV, nAsset)
		@inbounds for i ∈ 1:nAsset
			@inbounds for l ∈ 1:nAsset
				tvi[i] += a[l] * covar[i, l]
				if (l < indexPositive)
					tvni[i] += a[l] * covar[i, l]
				end
			end
		end
		#-Gp * dV/DK
		#mtildep =  (-vtilde2p + vtilde2n) / 2 + vtilde2p + vtilde2pn 
		mtildep = vtilde2 / 2
		d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
		d2 = d1 - vtilde
		eTerm = -(normcdf(signc * d2))
		#Gn*dV/DK
		#mtildep =  (-vtilde2p + vtilde2n) / 2 - vtilde2n - vtilde2pn 
		mtildep = (-vtilde2 / 2)
		d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
		d2 = d1 - vtilde
		eTerm += strikeScaled * (normcdf(signc * d2))
		mtilde = (-vtilde2p + vtilde2n) / 2
		for i ∈ 1:nAsset
			tv = tvi[i]
			mtildei = mtilde + tv
			d1 = (mtildei - logStrikeScaled + vtilde2) / vtilde
			d2 = d1 - vtilde
			factor = astar[i]
			eTerm += factor * (normcdf(signc * d2))
		end
		correction = signc * eTerm * Ap
		price += correction
		if p.order > 1
			#Gp^2/Gn * d2V/DK2
			mtildep = mtilde + vtilde2 + vtilde2p + vtilde2pn # =vtilde2/2+vtilde2
			d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
			d2 = d1 - vtilde
			eTerm = 0.5 * normpdf(d2) / (strikeScaled * vtilde) * exp(vtilde2)
			#Gn^2/Gn * d2V/DK2
			mtildep = mtilde - vtilde2n - vtilde2pn
			d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
			d2 = d1 - vtilde
			eTerm += 0.5 * strikeScaled * strikeScaled * normpdf(d2) / (strikeScaled * vtilde)
			#-2GpGn/Gn * d2V/DK2
			mtildep = mtilde + vtilde2p + vtilde2pn
			d1 = (mtildep - logStrikeScaled + vtilde2) / vtilde
			d2 = d1 - vtilde
			eTerm -= strikeScaled * normpdf(d2) / (strikeScaled * vtilde)
			#term in product^2.
			#now term in sum squared
			for i ∈ 1:nAsset
				factori = astar[i] / (strikeScaled * vtilde)
				for j ∈ 1:i-1
					tv = tvi[i] + tvi[j] + vtilde2pn + vtilde2n
					d2 = (mtilde + tv - logStrikeScaled) / vtilde
					factorj = astar[j]
					eTerm += factori * factorj * normpdf(d2) * exp(covar[i, j] + tvni[i] + tvni[j] + vtilde2n)
				end
                tv = 2tvi[i] + vtilde2pn + vtilde2n
                d2 = (mtilde + tv - logStrikeScaled) / vtilde
                eTerm += factori * astar[i] /2 * normpdf(d2) * exp(covar[i, i] + 2tvni[i]  + vtilde2n)
				# for j = 1:i-1
				#     tv = tvi[i] + tvi[j] + vtilde2pn + vtilde2n
				#     d2 = (mtilde + tv - logStrikeScaled) / vtilde
				#     factorj =  a[j] 
				#     if j < indexPositive
				#         factorj *= strikeScaled
				#     end
				#     eTerm += factori * factorj * normpdf(d2) * exp(covar[i, j]+tvni[i]+tvni[j]+vtilde2n)
				# end
				# d2 = (mtilde + 2tvi[i] + vtilde2pn + vtilde2n - logStrikeScaled) / vtilde
				# factorii = a[i]
				# if i < indexPositive
				#     factorii *= strikeScaled
				# end
				# eTerm += factori*factorii / 2 * normpdf(d2) * exp(totalVariance[i]+2tvni[i]+vtilde2n)
			end
			#finally term in product*sum. Gp/Gn*Si
			for i ∈ 1:nAsset
				factori = astar[i] / (strikeScaled * vtilde)
				tv = tvi[i]
				mtildeij = mtilde + tv + vtilde2
				d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
				d2 = d1 - vtilde
				eTerm -= factori * normpdf(d2) * exp(tv + vtilde2n + vtilde2pn)
			end
			#term in -Gn/Gn*Si
			for i ∈ 1:nAsset
				factori = astar[i] / (strikeScaled * vtilde)
				tv = tvi[i]
				mtildeij = mtilde + tv
				d1 = (mtildeij - logStrikeScaled + vtilde2) / vtilde
				d2 = d1 - vtilde
				eTerm += factori * strikeScaled * normpdf(d2)
			end

			price += Ap * eTerm
			if p.order > 2
				#d3CallBS/dK = phi(d2)*(d2/vtilde - 1)/(K^2*vtilde)
				d2 = (3vtilde2 - logStrikeScaled - vtilde2 / 2) / vtilde
				eTerm = normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(3vtilde2) #term in product^3.
				d2 = (2vtilde2 - logStrikeScaled - vtilde2 / 2) / vtilde
				eTerm -= 3 * strikeScaled * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(vtilde2)
				d2 = (vtilde2 - logStrikeScaled - vtilde2 / 2) / vtilde
				eTerm += 3 * strikeScaled^2 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1)
				d2 = (-logStrikeScaled - vtilde2 / 2) / vtilde
				eTerm -= strikeScaled^3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1)


				eTermS3 = zero(TV)

				#sum_ijl
				for i ∈ 1:nAsset
					@inbounds for j ∈ 1:i-1
						factorij = 6 * astar[i] * astar[j] / (strikeScaled^2 * vtilde)
						@inbounds for l ∈ 1:j-1
							tv = tvi[i] + tvi[j] + tvi[l] + 3vtilde2n + 3vtilde2np
							d2v = (tv - logStrikeScaled - vtilde2 / 2) / vtilde2
							eTermS3 += astar[l] * factorij * normpdf(d2v * vtilde) * (d2v - 1) * exp(covar[i, j] + covar[i, l] + covar[l, j] + 2tvni[i] + 2tvni[j] + 2tvni[l] + 3vtilde2n)
						end
						tv = tvi[i] + 2tvi[j] + 3vtilde2n + 3vtilde2np
						d2v = (tv - logStrikeScaled - vtilde2 / 2) / vtilde2
						eTermS3 += astar[j] * factorij / 2 * normpdf(d2v * vtilde) * (d2v - 1) * exp(2covar[i, j] + covar[j, j] + 2tvni[i] + 2tvni[j] + 2tvni[j] + 3vtilde2n)
						tv = 2tvi[i] + tvi[j] + 3vtilde2n + 3vtilde2np
						d2v = (tv - logStrikeScaled - vtilde2 / 2) / vtilde2
						eTermS3 += astar[i] * factorij / 2 * normpdf(d2v * vtilde) * (d2v - 1) * exp(2covar[i, j] + covar[i, i] + 4tvni[i] + 2tvni[j] + 3vtilde2n)
					end
					tv = 3tvi[i] + 3vtilde2n + 3vtilde2np
					d2v = (tv - logStrikeScaled - vtilde2 / 2) / vtilde2
					eTermS3 += astar[i]^3 / (strikeScaled^2 * vtilde) * normpdf(d2v * vtilde) * (d2v - 1) * exp(3covar[i, i] + 6tvni[i] + 3vtilde2n)
				end
				eTerm -= eTermS3
				#now term in sum^2* pi.
				for i ∈ 1:nAsset
					@inbounds for j ∈ 1:nAsset
						tv = tvi[i] + tvi[j]
						mtildeij = tv + vtilde2 + 2vtilde2n + 2vtilde2pn
						d1 = (mtildeij - logStrikeScaled + vtilde2 / 2) / vtilde
						d2 = d1 - vtilde
						eTerm += astar[i] * astar[j] * 3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(tv + covar[i, j] + tvni[i] + tvni[j] + 3vtilde2n + 2vtilde2pn)
					end
				end
				for i ∈ 1:nAsset
					@inbounds for j ∈ 1:nAsset
						tv = tvi[i] + tvi[j]
						mtildeij = tv + 2vtilde2n + 2vtilde2pn
						d1 = (mtildeij - logStrikeScaled + vtilde2 / 2) / vtilde
						d2 = d1 - vtilde
						eTerm -= astar[i] * astar[j] * 3 * strikeScaled * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(covar[i, j] + tvni[i] + tvni[j] + vtilde2n)
					end
				end
				#now term in sum*pi^2
				for i ∈ 1:nAsset
					tv = tvi[i]
					mtildeij = tv + 2vtilde2 + vtilde2n + vtilde2pn
					d1 = (mtildeij - logStrikeScaled + vtilde2 / 2) / vtilde
					d2 = d1 - vtilde
					eTerm -= astar[i] * 3 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(2tv + vtilde2 + 2vtilde2n + 2vtilde2pn)
				end
				for i ∈ 1:nAsset
					tv = tvi[i]
					mtildeij = tv + vtilde2n + vtilde2pn
					d1 = (mtildeij - logStrikeScaled + vtilde2 / 2) / vtilde
					d2 = d1 - vtilde
					eTerm -= astar[i] * 3 * strikeScaled^2 * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1)
				end
				for i ∈ 1:nAsset
					tv = tvi[i]
					mtildeij = tv + vtilde2n + vtilde2pn + vtilde2
					d1 = (mtildeij - logStrikeScaled + vtilde2 / 2) / vtilde
					d2 = d1 - vtilde
					eTerm += astar[i] * 6 * strikeScaled * normpdf(d2) / (strikeScaled^2 * vtilde) * (d2 / vtilde - 1) * exp(tv + vtilde2n + vtilde2pn)
				end
				price += Ap * eTerm / 6

			end
		end
	end
	return price * discountFactor
end
