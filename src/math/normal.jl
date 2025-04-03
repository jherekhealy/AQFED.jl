export normcdf, norminv, normcdfCody

const DBL_EPSILON = eps()
const DBL_MIN = floatmin() #different from math.SmallestNon0Float64
const DBL_MAX = floatmax()
const LN_TWO = log(2)
const LN_TWO_PI = log(2pi)
const SQRT_PI = sqrt(pi)


using SpecialFunctions

const sqrt2 = sqrt(2)
const OneOverSqrtTwo = 1 / sqrt(2)

# normcdf(z::Real) =  normcdfCody(z) #
normcdf(z::Float64) = erfc(-z * OneOverSqrtTwo) / 2
norminv(z::Float64) = -sqrt2 * erfcinv(2 * z)

normcdf(z::T) where {T} = erfc(-z / sqrt(T(2))) / 2
norminv(z::T) where {T} = -sqrt(T(2)) * erfcinv(2 * z)

normcdf(z, μ, σ) = normcdf((z - μ) / σ)

#const 1.0 = 1.0
#const 0.5 = 0.5
#const 0 = 0.0
const _normal_sixten = 1.6
const thrsh = 0.66291
const root32 = 5.656854248
const OneOverSqrt2Pi = 1 / sqrt(2 * pi)


#Reference: Cody, W.D. (1993). ALGORITHM 715: SPECFUN - A Portable FORTRAN Package of Special Function Routines and Test Drivers" ACM Transactions on Mathematical Software. 19, 22-32.
# This function evaluates the normal distribution function: The main computation evaluates near-minimax approximations derived from those in "Rational Chebyshev approximations for the error function" by W. J. Cody, Math. Comp., 1969,
# 631-637. This transportable program uses rational functions that theoretically approximate the normal distribution function to at least 18 significant decimal digits. The accuracy achieved depends on the arithmetic system, the compiler,
# the intrinsic functions, and proper selection of the machine-dependent constants.

#
#normcdf(z::Float64) = normcdfCody(z)

const c0 = 0.39894151208813466764
const c1 = 8.8831497943883759412
const c2 = 93.506656132177855979
const c3 = 597.27027639480026226
const c4 = 2494.5375852903726711
const c5 = 6848.1904505362823326
const c6 = 11602.651437647350124
const c7 = 9842.7148383839780218
const c8 = 1.0765576773720192317e-8

const d0 = 22.266688044328115691
const d1 = 235.38790178262499861
const d2 = 1519.377599407554805
const d3 = 6485.558298266760755
const d4 = 18615.571640885098091
const d5 = 34900.952721145977266
const d6 = 38912.003286093271411
const d7 = 19685.429676859990727

const p0 = 0.21589853405795699
const p1 = 0.1274011611602473639
const p2 = 0.022235277870649807
const p3 = 0.001421619193227893466
const p4 = 2.9112874951168792e-5
const p5 = 0.02307344176494017303

const q0 = 1.28426009614491121
const q1 = 0.468238212480865118
const q2 = 0.0659881378689285515
const q3 = 0.00378239633202758244
const q4 = 7.29751555083966205e-5

const a0 = 2.2352520354606839287
const a1 = 161.02823106855587881
const a2 = 1067.6894854603709582
const a3 = 18154.981253343561249
const a4 = 0.065682337918207449113

const b0 = 47.20258190468824187
const b1 = 976.09855173777669322
const b2 = 10260.932208618978205
const b3 = 45507.789335026729956


function normcdfCody(x::Float64)::Float64

	xden = 0.0
	temp = 0.0
	xnum = 0.0
	result = 0.0
	ccum = 0.0
	min = 0.0
	eps = 0.0
	xsq = 0.0
	y = 0.0

	eps = DBL_EPSILON * 0.5
	min = DBL_MIN
	y = abs(x)
	if y <= thrsh
		# Evaluate pnorm for |z| <= 0.66291 */
		xsq = 0
		if y > eps
			xsq = x^2
		end
		xnum = a4 * xsq
		xden = xsq
		xnum = (xnum + a0) * xsq
		xden = (xden + b0) * xsq
		xnum = (xnum + a1) * xsq
		xden = (xden + b1) * xsq
		xnum = (xnum + a2) * xsq
		xden = (xden + b2) * xsq
		result = x * (xnum + a3) / (xden + b3)
		temp = result
		result = 0.5 + temp
		ccum = 0.5 - temp
	elseif y <= root32
		#/* Evaluate pnorm for 0.66291 <= |z| <= sqrt(32) */
		xnum = c8 * y
		xden = y
		xnum = (xnum + c0) * y
		xden = (xden + d0) * y
		xnum = (xnum + c1) * y
		xden = (xden + d1) * y
		xnum = (xnum + c2) * y
		xden = (xden + d2) * y
		xnum = (xnum + c3) * y
		xden = (xden + d3) * y
		xnum = (xnum + c4) * y
		xden = (xden + d4) * y
		xnum = (xnum + c5) * y
		xden = (xden + d5) * y
		xnum = (xnum + c6) * y
		xden = (xden + d6) * y
		result = (xnum + c7) / (xden + d7)
		result *= pdfDenormalized(y)
		#            xsq = fint(y * _normal_sixten) / _normal_sixten;
		#            del = (y - xsq) * (y + xsq);
		#            result = math.Exp(-xsq * xsq * 0.5) * math.Exp(-del * 0.5) * result;
		ccum = 1.0 - result
		if x > 0
			temp = result
			result = ccum
			ccum = temp
		end
	else
		# Evaluate pnorm for |z| > sqrt(32) */
		result = 0
		xsq = 1.0 / (x * x)
		xnum = p5 * xsq
		xden = xsq
		xnum = (xnum + p0) * xsq
		xden = (xden + q0) * xsq
		xnum = (xnum + p1) * xsq
		xden = (xden + q1) * xsq
		xnum = (xnum + p2) * xsq
		xden = (xden + q2) * xsq
		xnum = (xnum + p3) * xsq
		xden = (xden + q3) * xsq
		result = xsq * (xnum + p4) / (xden + q4)
		result = (OneOverSqrt2Pi - result) / y
		result *= pdfDenormalized(y)
		#            xsq = fint(x * _normal_sixten) / _normal_sixten;
		#            del = (x - xsq) * (x + xsq);
		#           result = math.Exp(-xsq * xsq * 0.5) * math.Exp(-del * 0.5) * result;
		ccum = 1.0 - result
		if x > 0
			temp = result
			result = ccum
			ccum = temp
		end
	end
	if result < min
		result = 0.0
	end
	if ccum < min
		ccum = 0.0
	end
	return result
end

function pdfDenormalized(x::Float64)::Float64
	return exp(-0.5 * x * x)
end


function sqrtOnePlusXMinusOne(x)
	if abs(x) < 0.25 # sqrt(1+x) -1 expanded for small x. PJ , 2019.
		g = if x < 0
			# net relative accuracy [of x*(0.5-x*g)] better than 9E-17
			(1.249999999999904E-1 + x * (1.952674587134858E-1 + x * (8.609312134216375E-2 + 9.372014538795577E-3 * x))) / (1 + x * (2.06213966971922 + x * (1.407314806167365 + x * (3.529648863602843E-1 + 2.372729847070329E-2 * x))))
		else # net relative accuracy [of x*(0.5-x*g)] better than 8E-18
			(1.24999999999998E-1 + x * (1.811342940763106E-1 + x * (7.189411491089435E-2 + 6.684805698482948E-3 * x))) / (1 + x * (1.949074352607691 + x * (1.237190095758854 + x * (2.817377533594094E-1 + 1.654456755072703E-2 * x))))
		end
		return x * (0.5 - x * g)
	end
	return sqrt(1.0 + x) - 1.0
end

function lnCdf(z) # ln(Φ(z))
	return if (z > 0)
		lnOfOnePlusX(-normcdf(-z))
	elseif (z <= -9.0)
		log(asymptoticExpansionOfCdfOverPdf(z)) - 0.5 * (z * z + LN_TWO_PI)
	else
		log(normcdf(z))
	end
end

function lnTwoCdf(y) # // ln(2·Φ(y))
	# For |y|<0.4, we use a Nonlinear-Remez optimized approximation given by f(y)=θ(y,g(y)) with θ(y,g) = y·(√(2/π)-y·g) and the (5,4)-rational function
	#   g(y) = (3.183098861837894E-1+y*(-2.1883026451258495E-1+y*(7.2342988529039326E-2+y*(-1.1919277649580325E-2+y*(8.3837904820976024E-4-1.2552162258772677E-6*y)))))/(1+y*(-5.7332388994577462E-1+y*(1.768497738757971E-1+y*(-2.598729916076733E-2+1.7247546759678694E-3*y))))
	# The net accuracy is better than 1.122E-16.
	return if abs(y) > 0.4
		log(2) + lnCdf(y)
	else
		y * (
			0.79788456080286535588 -
			y * (
				(3.183098861837894E-1 + y * (-2.1883026451258495E-1 + y * (7.2342988529039326E-2 + y * (-1.1919277649580325E-2 + y * (8.3837904820976024E-4 - 1.2552162258772677E-6 * y))))) /
				(1 + y * (-5.7332388994577462E-1 + y * (1.768497738757971E-1 + y * (-2.598729916076733E-2 + 1.7247546759678694E-3 * y))))
			)
		)
	end
end

function asymptoticExpansionOfCdfOverPdf(z::T) where {T}
	# The asymptotic expansion  Φ(z) = φ(z)/|z|·[1-1/z^2+...],  Abramowitz & Stegun (26.2.12), suffices for Φ(z) to have

	# Asymptotic expansion for very negative z following (26.2.12) on page 408
	# in M. Abramowitz and A. Stegun, Pocketbook of Mathematical Functions, ISBN 3-87144818-4, is
	#  Φ(z) = φ(z)/(-z)·[1-1/z^2+...].
	# Hence, 
	#   Φ(z) / φ(z) = -(1-1/z^2+...) / z.
	sum = one(T)
	if (z >= -one(T)/sqrt(eps(T)))
		zsqr = z * z
		i = 1
		g = 1
		x = floatmax(T)
		y = floatmax(T)
		a = floatmax(T)
		lasta = a
		while true
			lasta = a
			x = (4 * i - 3) / zsqr
			y = x * ((4 * i - 1) / zsqr)
			a = g * (x - y)
			sum -= a
			g *= y
			i += 1
			a = abs(a)
			(lasta > a && a >= abs(sum * eps(T))) || break
		end
	end
	return -sum / z
end

lnOfOnePlusX(x) = log1p(x)


function inverseLnErfcx(y::T) where {T} # Inverse of y(z) := ln(erfcx(z))

	if (y >= 2.926054502)
		w = 1 / (y - LN_TWO)
		zeta =
			(
				1.008831813880055711E0 +
				w * (5.6930462040180993295E1 + w * (7.8758967157473104837E2 + w * (6.5369557790922766761E3 + w * (3.207323236990434636E4 + w * (4.7081311888818542321E4 + w * ((-1.5356447406703549185E3) - 1.3618612709837885034E4 * w))))))
			) / (1 + w * (4.9386339997921838307E1 + w * (5.6659651701221432338E2 + w * (4.6794735596030363518E3 + w * (1.8438296226659342414E4 + w * (2.3066435716227694152E4 + w * (6.0425284640091166861E3 - 1.8361146033364165539E2 * w)))))))
		z = -(1 + exp(-zeta / w) / (4 * SQRT_PI)) / sqrt(w)
	elseif (y > -1.134492089)
		gamma =
			(
				2.5292753687758446586E-1 +
				y * (
					3.215776976543350707E-1 +
					y * (
						2.1543770801630490759E-1 +
						y * (8.8809464227384136275E-2 + y * (2.473545115937593774E-2 + y * (4.6565738088983181102E-3 + y * (5.9311148077307959971E-4 + y * (4.6804490045539737994E-5 + y * (1.952362913447388075E-6 + 1.3547814811265750855E-9 * y))))))
					)
				)
			) / (
				1 +
				y * (
					1.5915724440392176382E0 +
					y * (
						1.2703842021304194846E0 +
						y * (6.3481645737589602599E-1 + y * (2.1605555010263235345E-1 + y * (5.1491662689748139232E-2 + y * (8.5935378171905465074E-3 + y * (9.7598654688020152114E-4 + y * (6.9714835279954410095E-5 + 2.4561708908515460842E-6 * y))))))
					)
				)
			)
		z = y * (-0.5 * SQRT_PI + y * gamma)
	else
		m = exp(2 * y)
		R =
			(
				1.5707963267948264287E0 +
				m * (1.4881863687985725824E2 + m * (5.0573969043451462837E3 + m * (7.71433208578163961E4 + m * (5.4104053714173542146E5 + m * (1.5996711891638245768E6 + m * (1.5582800959832326392E6 + 2.2979729179982853295E5 * m))))))
			) / (1 + m * (9.6311683061208178375E1 + m * (3.3610552996661054666E3 + m * (5.3544584629105503552E4 + m * (4.039412696733845392E5 + m * (1.3577968131365781872E6 + m * (1.6990431126157576341E6 + 4.9222272232082468876E5 * m)))))))
		z = (1 - m * R) / (SQRT_PI * sqrt(m))
	end
	return z
end
