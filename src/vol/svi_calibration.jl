import AQFED.Math: inv as invTransform, ClosedTransformation
import AQFED.TermStructure:SVISection, varianceByLogmoneyness
using Optim
import StatsBase:rmsd

function calibrateSVISection(tte::T, forward::T, ys::AbstractArray{T}, vols::AbstractArray{T}, weights::AbstractArray{T}; aMin=zero(T), sMin=one(T)/10000, noarbFactor = 2*one(T),nGuess=10) where {T}
    sList = rand(nGuess).+sMin
    mList = rand(nGuess).*(ys[end]-ys[1]) .+ ys[1]
    paramList = SVISection[]
    rmseList= T[]
    for (s,m) = zip(sList,mList)
        (param, rmse) = calibrateSVISectionFromGuess(tte,forward,ys, vols, weights, s=s,m=m,aMin=aMin,sMin=sMin,noarbFactor=noarbFactor)
        push!(rmseList,rmse)
        push!(paramList , param)
    end
    rmse, index = findmin(rmseList)
    return paramList[index],rmse
end
function calibrateSVISectionFromGuess(tte::T, forward::T, ys::AbstractArray{T}, vols::AbstractArray{T}, weights::AbstractArray{T}; s=one(T),m=zero(T), aMin=zero(T), sMin=one(T)/10000, noarbFactor = 2*one(T)) where {T}
    variances = vols.^2
	# 1/sqrteps > y-m / s > sqrt(eps) == y-m *sqrteps <s < (y-m)/sqrteps
	sMinEffective = max(sMin,16*(ys[end]-ys[1])*sMin)
    transformation = ClosedTransformation(sMinEffective, 100.0)
	# println("transformation ",transformation)
	obj = function(x::AbstractArray{TX}) where {TX}
		sTrans = x[1]
		s = transformation(sTrans)
		m = x[2]
		(param, rmse) = solveSVISectionQuasiExplicit(s, m, aMin, noarbFactor, ys, variances, weights, tte, forward)
		return rmse
    end
	sTrans = invTransform(transformation,max(s,sMinEffective))
	start = [sTrans, m]
	res = optimize(obj, start, NelderMead())
    #println(res)
    x = Optim.minimizer(res)
	newS = transformation(x[1])
	newM = x[2]    
	return solveSVISectionQuasiExplicit(newS, newM, aMin, noarbFactor, ys, variances, weights, tte, forward)
end

function computeRMSE(params::SVISection, y::AbstractArray{T}, variances::AbstractArray{T}, weights::AbstractArray{T}) where {T}
    rmsd( weights .* varianceByLogmoneyness.(params, y), weights .* variances)
end

function solveSVISectionQuasiExplicit(s::TS, m::TS, aMin::T, noarbFactor::T, y::AbstractArray{T}, variances::AbstractArray{T}, weights::AbstractArray{T}, tte::T, forward::T) where {T,TS}
	if length(variances) == 1 
		return SVISection(variances[1], zero(T), one(T), one(T), zero(T), tte, forward),zero(T)
    end
	C1 = zero(T)
	C2 = zero(T)
	D1 = zero(T)
	D2 =  zero(T)
	E =  zero(T)
	F1 =  zero(T)
	F2 =  zero(T)
	F3 =  zero(T)
	W =  zero(T)
	for (i, yorig) = enumerate(y) 
		w = weights[i]
		W += w
		yi = (yorig - m) / s
		yisq = yi^2
		sqrtyi = sqrt(yisq + one(T))
		C1 += w * sqrtyi
		C2 += w * (yisq + one(T))
		D1 += w * yi
		D2 += w * yisq
		E += w * yi * sqrtyi
		vitte = variances[i] * tte
		F1 += w * vitte
		F2 += w * yi * vitte
		F3 += w * sqrtyi * vitte
    end
	result = zeros(T,3)
L= [W C1 D1; C1 C2 E; D1 E D2]
	R = [F1,F3,F2]
	try
	#
    result = L \ R
	catch exc
		println("L ",L," R ",R, " s ",s, " m ",m)
	end
	aTilde = result[1]
	c = result[2]
	d = result[3]
	maxVariance = maximum(variances) * tte
	if isInStrictDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
		result = SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward)
		return result, computeRMSE(result,y,variances,weights)
    end
	paramList = SVISection[]
	c = zero(T)
    L = [W D1; D1 D2]
    R = [F1, F2]
	result = L \ R
    aTilde = result[1]
	d = result[2]
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
		push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	c = 2 * noarbFactor * s
    R = [F1 - c*C1, F2-c*E]
	result = L \ R
    aTilde = result[1]
	d = result[2]
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
		push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
    aTilde = aMin*tte
	L = [E D2; C2 E]
    R = [F2 - aTilde*D1, F3 - aTilde*C1]
	result = L \ R
    c = result[1]
	d = result[2]
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
		push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
    aTilde = maxVariance
	L = [E D2; C2 E]
    R = [F2 - aTilde*D1, F3 - aTilde*C1]
	result = L \ R
    c = result[1]
	d = result[2]
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
		push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = zero(T)
	eta = one(T)
	L = [ W  D1 + eta*C1; D1 + eta*C1   D2 + 2*eta*E + C2]
	R= [ F1 - cB*C1, F2 + eta*F3 - cB*(E+eta*C2)]
    result = L \ R
    aTilde = result[1]
	d = result[2]
	c = cB + eta*d
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
	    push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = 2 * noarbFactor * s
    L = [ W  D1 + eta*C1; D1 + eta*C1   D2 + 2*eta*E + C2]
	R= [ F1 - cB*C1, F2 + eta*F3 - cB*(E+eta*C2)]
    result = L \ R
    aTilde = result[1]
	d = result[2]
	c = cB + eta*d
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	
	cB = 2 * noarbFactor * s
	eta = -one(T)
    L = [ W  D1 + eta*C1; D1 + eta*C1   D2 + 2*eta*E + C2]
	R= [ F1 - cB*C1, F2 + eta*F3 - cB*(E+eta*C2)]
    result = L \ R
    aTilde = result[1]
	d = result[2]
	c = cB + eta*d
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = zero(T)
	eta = -one(T)
	 L = [ W  D1 + eta*C1; D1 + eta*C1   D2 + 2*eta*E + C2]
	R= [ F1 - cB*C1, F2 + eta*F3 - cB*(E+eta*C2)]
    result = L \ R
    aTilde = result[1]
	d = result[2]
	c = cB + eta*d
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = zero(T)
	eta = one(T)
	c = cB
	d = eta * cB
	aTilde = F1 - eta*cB*D1 - cB*C1
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	
	cB = zero(T)
	eta = -one(T)
	c = cB
	d = eta * cB
	aTilde = F1 - eta*cB*D1 - cB*C1
    if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
end

	cB = 2 * noarbFactor * s
	eta = one(T)
	c = cB
	d = eta * cB
	aTilde = F1 - eta*cB*D1 - cB*C1
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = 2 * noarbFactor * s
	eta = -one(T)
	c = cB
	d = eta * cB
	aTilde = F1 - eta*cB*D1 - cB*C1
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor)   
         push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = zero(T)
	eta = one(T)
	c = cB
	d = eta * (2*noarbFactor*s - cB)
	aTilde = F1 - eta*(2*noarbFactor*s-cB)*D1 - cB*C1
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor)   
         push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = zero(T)
	eta = -one(T)
	c = cB
	d = eta * (2*noarbFactor*s - cB)
	aTilde = F1 - eta*(2*noarbFactor*s-cB)*D1 - cB*C1
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = 2 * noarbFactor * s
	eta = one(T)
	c = cB
	d = eta * (2*noarbFactor*s - cB)
	aTilde = F1 - eta*(2*noarbFactor*s-cB)*D1 - cB*C1
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = 2 * noarbFactor * s
	eta = -one(T)
	c = cB
	d = eta * (2*noarbFactor*s - cB)
	aTilde = F1 - eta*(2*noarbFactor*s-cB)*D1 - cB*C1
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = zero(T)
	aB = aMin*tte
	c = cB
	aTilde = aMin*tte
	d = (F2 - D1*aB - E*cB) / D2
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor)
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
    
	aB = maxVariance
	c = cB
	aTilde = aMin*tte
	d = (F2 - D1*aB - E*cB) / D2
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = 2 * noarbFactor * s
	aB = aMin*tte
	c = cB
	aTilde = aMin*tte
	d = (F2 - D1*aB - E*cB) / D2
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	cB = 2 * noarbFactor * s
	aB = maxVariance
	c = cB
	aTilde = aMin*tte
	d = (F2 - D1*aB - E*cB) / D2
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	aB = aMin*tte
	eta = one(T)
	aTilde = aB
	c = (eta*F2 + F3 - (eta*D1+C1)*aB) / (D2 + 2*eta*E + C2)
	d = eta * c
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor)
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	aB = maxVariance
	eta = one(T)
	aTilde = aB
	c = (eta*F2 + F3 - (eta*D1+C1)*aB) / (D2 + 2*eta*E + C2)
	d = eta * c
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	aB = aMin*tte
	eta = -one(T)
	aTilde = aB
	c = (eta*F2 + F3 - (eta*D1+C1)*aB) / (D2 + 2*eta*E + C2)
	d = eta * c
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	aB = maxVariance
	eta = -one(T)
	aTilde = aB
	c = (eta*F2 + F3 - (eta*D1+C1)*aB) / (D2 + 2*eta*E + C2)
	d = eta * c
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	aB = aMin*tte
	eta = one(T)
	aTilde = aB
	c = (eta*F2 + F3 + 2*noarbFactor*s*(E+eta*D2) - (eta*D1+C1)*aB) / (D2 + 2*eta*E + C2)
	d = eta * (c - 2*noarbFactor*s)
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	aB = aMin*tte
	eta = -one(T)
	aTilde = aB
	c = (eta*F2 + F3 + 2*noarbFactor*s*(E+eta*D2) - (eta*D1+C1)*aB) / (D2 + 2*eta*E + C2)
	d = eta * (c - 2*noarbFactor*s)
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	aB = maxVariance
	eta = one(T)
	aTilde = aB
	c = (eta*F2 + F3 + 2*noarbFactor*s*(E+eta*D2) - (eta*D1+C1)*aB) / (D2 + 2*eta*E + C2)
	d = eta * (c - 2*noarbFactor*s)
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor) 
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
	aB = maxVariance
	eta = -one(T)
	aTilde = aB
	c = (eta*F2 + F3 + 2*noarbFactor*s*(E+eta*D2) - (eta*D1+C1)*aB) / (D2 + 2*eta*E + C2)
	d = eta * (c - 2*noarbFactor*s)
	if isInDomain(aTilde, c, d, aMin*tte, maxVariance, s, noarbFactor)
        push!(paramList,SVISection(aTilde / tte, c / (tte * s), d / c, s, m, tte, forward))
    end
    rmseList = [computeRMSE(param,y,variances,weights) for param in paramList]
    rmse, index = findmin(rmseList)

	return paramList[index], rmse
end

function isInStrictDomain(a::T, c::T, d::T, aMin, aMax, s, noarbFactor)::Bool where {T}
	if a <= aMin 
		return false
    end
	if a >= aMax 
		return false
    end
	if c <= zero(T)
		return false
    end
	if c >= 2*noarbFactor*s 
		return false
    end
	absD = abs(d)
	if absD >= c 
		return false
    end
	if absD >= 2*noarbFactor*s-c 
		return false
    end
	return true
end

function isInDomain(a::T, c::T, d::T, aMin, aMax, s, noarbFactor)::Bool where {T}
	if a < aMin 
		return false
    end
	if a > aMax 
		return false
    end
	if c < zero(T)
		return false
    end
	if c > 2*noarbFactor*s 
		return false
    end
	absD =abs(d)
	if absD > c 
		return false
    end
	if absD > 2*noarbFactor*s-c 
		return false
    end
	return true
end
