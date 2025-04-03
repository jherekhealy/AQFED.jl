import AQFED.Math: normcdf, norminv, normpdf, rational_cubic_interpolation, sqrtOnePlusXMinusOne, lnTwoCdf, inverseLnErfcx

houseHolder6Step(nu, h2, h3, h4, h5, h6) = (nu * (1 + nu * (2 * h2 + nu * ((3 * (h2 * h2)) / 4 + h3 / 2 + nu * ((h2 * h3) / 6 + h4 / 12 + (h5 * nu) / 120))))) /
		   (1 + nu * ((5 * h2) / 2 + nu * ((3 * (h2 * h2)) / 2 + (2 * h3) / 3 + nu * ((h2 * h2 * h2) / 8 + (h2 * h3) / 2 + h4 / 8 + nu * ((h3 * h3) / 36 + (h2 * h4) / 24 + h5 / 60 + (h6 * nu) / 720)))))

function is_too_small(x::T) where {T}
	return abs(x) < floatmin(T)
end

function controlParameterToFitInteriorPoint(x_l::T, x_r::T, y_l::T, y_r::T, d_l::T, d_r::T, x_interior::T, y_interior::T; r_min = -(1 - sqrt(eps(T))), r_max = 2 / eps(T)^2) where {T}
	h = (x_r - x_l)
	if (is_too_small(h))
		return 3
	end
	t = (x_interior - x_l) / h
	omt = 1 - t
	t2 = t * t
	omt2 = omt * omt
	y_abs_max = max(abs(y_interior), max(abs(y_l), abs(y_r)))

	cubic_form = y_r * t2 * t + (3 * y_r - h * d_r) * t2 * omt + (3 * y_l + h * d_l) * t * omt2 + y_l * omt2 * omt
	y_interior_minus_cubic_form = y_interior - cubic_form
	if abs(y_interior_minus_cubic_form) <= eps(T) * y_abs_max
		return 3
	end
	linear_form = y_r * t + y_l * omt
	y_interior_minus_linear_form = y_interior - linear_form
	if abs(y_interior_minus_linear_form) <= eps(T) * y_abs_max
		return r_max
	end
	w = t * omt
	if is_too_small(w)
		return 3
	end
	r = 3 - (y_interior_minus_cubic_form / y_interior_minus_linear_form) / w
	r = min(r_max, max(r_min, r))
	return r
end


const LN_TWO = 0.69314718055994530941723212145817656807550013436026
const SQRT_TWO = 1.41421356237309504880168872421
const SQRT_TWO_OVER_PI = 0.7978845608028653558798921198687637
# f(y) = ln(2·Φ(-y)) + α·y  as defined in equation (1.14)
const NY = 11
const Yeps = -8.2535 # ≈ Φ^(-1)(DBL_EPSILON·ln(2)/2)
const Y = [-8.2535, -7.0, -6.3333333333333333333, -5.6666666666666666667, -5, -4.3333333333333333333, -3.6666666666666666667, -3.0, -2.0, -1.0, 0.0]
# c  :  f(y[i],alpha)=c[i]+y[i]*α
const C = [
	6.93147180559945233E-1,
	6.9314718055866549687E-1,
	6.9314718043998508325E-1,
	6.9314717327983520900E-1,
	6.9314689390833234565E-1,
	6.931398371091453459E-1,
	6.9302430662128693903E-1,
	6.9179637059519711562E-1,
	6.7013427123098182E-1,
	5.20393401536495419E-1,
	0.0,
]
# d  :  f'(y[i];α) = d[i]+α
const D = [
	-6.4384757653278754E-16,
	-9.1347204083762840726E-12,
	-7.7784480690027837163E-10,
	-4.2468887773727108186E-8,
	-1.4867199409049057124E-6,
	-3.3371107453824317093E-5,
	-4.8032966799311978792E-4,
	-4.4378390421256637933E-3,
	-5.52478626789899591E-2,
	-2.87599970939178361E-1,
	-7.97884560802865355E-1,
]
#e: f''(y[i];α) = e[i] (not dependent on α)
const E = [
	-5.31399597291336238E-15,
	-6.3943042858717431625E-11,
	-4.926350444306805564E-9,
	-2.4065703252139337512E-7,
	-7.4336019148607112465E-6,
	-1.4460924593071806876E-4,
	-1.7614394992313936031E-3,
	-1.3333211541740806209E-2,
	-1.13548051688576449E-1,
	-3.70313714223394599E-1,
	-6.36619772367581343E-1,
]
#α such that ymax=y : alpha_of_ymax(y)=sqrt(2.0/pi)/erfcx(y/sqrt(2.0))   ---   UNUSED !
const αmax = [
	6.4384757653278754E-16,
	9.1347204083762840726E-12,
	7.7784480690027837163E-10,
	4.2468887773727108186E-8,
	1.4867199409049057124E-6,
	3.3371107453824317093E-5,
	4.8032966799311978792E-4,
	4.4378390421256637933E-3,
	5.5247862678989959102E-2,
	2.8759997093917836123E-1,
	7.9788456080286535588E-1,
]
F(i, α) = C[i+1] + Y[i+1] * α

Fp(i, α) = D[i+1] + α;
# g[i] = y[i]-1/3
const G = [-8.58683333333333333, -7.33333333333333333, -6.66666666666666667, -6.0, -5.33333333333333333, -4.66666666666666667, -4.0, -3.33333333333333333, -2.33333333333333333, -1.33333333333333333, -3.33333333333333333E-1]
# h  :  f(g[i],alpha)=h[i]+g[i]*α
const H = [
	6.93147180559945305E-1,
	6.93147180559833061E-1,
	6.93147180546861385E-1,
	6.93147179573357664E-1,
	6.93147132346910496E-1,
	6.93145649932037368E-1,
	6.9311550881656782E-1,
	6.92718028154026267E-1,
	6.8328336404937177E-1,
	5.97504603819424651E-1,
	2.31998089638832187E-1,
]
# Fi(i,α) = F(y[i]-1/3,α) = F(g[i],α)
Fi(i, α) = H[i+1] + G[i+1] * α

function findInterval(f::T, α::T, i_l::Int, i_r::Int, ymax::T, fmax::T, f_y_lr) where {T}#note: the last two params need update.
	# 'left' and 'right' here refers to the ordering in y.
	i_mid = div((i_l + i_r), 2)
	#Recall that f(y) is (supposed to be) downward sloping.
	f_y_mid = if Y[i_mid+1] < ymax
		fmax
	else
		C[i_mid+1] + Y[i_mid+1] * α
	end
	if (f > f_y_mid) # branch to the left in y to get the upper branch in f
		f_y_lr[2] = f_y_mid
		return if i_l + 1 >= i_mid
			i_l
		else
			findInterval(f, α, i_l, i_mid, ymax, fmax, f_y_lr)
		end
	end
	# branch to the right in y to get the lower branch in f
	f_y_lr[1] = f_y_mid
	return if i_mid + 1 >= i_r
		i_mid
	else
		findInterval(f, α, i_mid, i_r, ymax, fmax, f_y_lr)
	end
end

function findInterval(f::T, α::T, ymax::T, fmax::T, f_y_lr) where {T}
	findInterval(f, α, 0, NY, ymax, fmax, f_y_lr)
end
function findYIndexNotLess(y::T, i_l::Int, i_r::Int) where {T}
	if (i_l >= i_r)
		return i_l
	end
	i_mid = div(i_l + i_r, 2)
	return if y > Y[i_mid+1]
		if i_mid + 1 >= i_r
			i_r
		else
			findYIndexNotLess(y, i_mid, i_r)
		end
	else
		findYIndexNotLess(y, i_l, i_mid)
	end
end

findYIndexNotLess(y) = findYIndexNotLess(y, 0, NY - 1)
function y0ForPositiveFAndNegativeY(f::T, alpha::T, ymax::T, fmax::T) where {T}
	# assert(f >= 0);
	# assert(alpha < 7.97884560802865355E-1);
	# Recall that f(y) is downward sloping: 'left' and 'right' here refers to the ordering in f, i.e., the inverted direction of the ordering in y.
	f_r = if (Y[1] < ymax)
		fmax
	else
		F(0, alpha)
	end
	f_l = 0.0
	y = if (f >= f_r)
		(f - LN_TWO) / alpha
	else
		f_lr = [f_r, f_l]
		k = findInterval(f, alpha, ymax, fmax, f_lr)
		f_r = f_lr[1]
		f_l = f_lr[2]
		y_r = Y[k+1]
		y_l = Y[k+2]
		dydf_r = 1 / Fp(k, alpha)
		dydf_l = 1 / Fp(k + 1, alpha)
		y_i = G[k+2]
		f_i = Fi(k + 1, alpha)

		r = controlParameterToFitInteriorPoint(f_l, f_r, y_l, y_r, dydf_l, dydf_r, f_i, y_i)
		rational_cubic_interpolation(f, f_l, f_r, y_l, y_r, dydf_l, dydf_r, r)
	end
	return y
end

function y0(f::T, alpha::T) where {T}
	if (f < 0 && alpha < 0.1388)

		f2p0 = -6.36619772367581343E-1
		f3p0 = -2.1801361414499016E-1
		f4p0 = 1.14770682054218857E-1
		f5p0 = 4.43768846261782098E-3
		fp0 = -7.97884560802865355E-1 + alpha
		nu = f / fp0
		h2 = f2p0 / fp0
		h3 = f3p0 / fp0
		h4 = f4p0 / fp0
		h5 = f5p0 / fp0
		h22 = h2 * h2
		h23 = h22 * h2
		h24 = h23 * h2
		omh22 = (1 - h2) * (1 - h2)
		h32 = h3 * h3
		h33 = h32 * h3
		h42 = h4 * h4
		a1 = 90 * h22 * omh22 - 60 * (2 - 3 * h2 + h22) * h3 - 40 * h32 + 30 * (h2 - 1) * h4
		a2 = 45 * h22 * (1 + 3 * h2) * omh22 - 30 * (2 + 3 * h2 - 10 * h22 + 5 * h23) * h3 - 20 * (2 + h2) * h32 + (15 - 60 * h2 + 45 * h22 + 10 * h3) * h4 + 6 * (1 - h2) * h5
		a3 = 135 * h23 * (2 + h2) * omh22 - 90 * (4 - 7 * h2 + 3 * h23) * h2 * h3 + 180 * (h22 - 1) * h32 - 80 * h33 + (90 - 90 * h2 + 60 * h2 * h3) * h4 - 15 * h42 + (18 - 18 * h22 + 12 * h3) * h5
		b1 = 90 * h23 * omh22 - 30 * (4 - 7 * h2 + 3 * h22) * h2 * h3 - 10 * (1 + 2 * h2) * h32 + (15 - 45 * h2 + 30 * h22 + 5 * h3) * h4 + 3 * (1 - h2) * h5
		b2 = 135 * h24 * omh22 - 60 * (2 - 5 * h2 + 3 * h22) * h22 * h3 - 40 * (2 - h2 - h22) * h32 - 40 * h33 + 10 * (6 * h2 - 9 * h22 + 3 * h23 - 2 * (1 - 2 * h2) * h3) * h4 - 5 * h42 + 4 * (3 * h2 - 3 * h22 + h3) * h5
		y = sqrtOnePlusXMinusOne(2 * nu * (6 * a1 + nu * (6 * a2 + nu * a3)) / (6 * a1 + nu * (12 * b1 + 3 * nu * b2)))
	else
		if (alpha <= 0) # // Put option branch for f>=0 (the case f<0 is already handled above).
			y = y0ForPositiveFAndNegativeY(f, alpha, -floatmax(T), floatmax(T))
		else
			ymax = SQRT_TWO * inverseLnErfcx(log(SQRT_TWO_OVER_PI / alpha))
			fmax = lnTwoCdf(-ymax) + alpha * ymax
			if (abs(f - fmax) <= eps(T) * abs(fmax))
				return ymax
			end
			if (f > fmax)
				fmax = lnTwoCdf(-ymax) + alpha * ymax
				if (abs(f - fmax) <= eps(T) * abs(fmax))
					return ymax
				end
                #Jherek: a better idea would be to throw an exception. For now, return the closest possible.
				return ymax # Jherek: Original Code was returning NaN: case where we do not reach the delta.
			end
			# The location of the maximum. fmax = f(ymax,alpha) = log(SqrtTwoOverPi/alpha)+alpha*ymax-0.5*ymax*ymax.
			# At the maximum, q = phi(-y)/Phi(-y) = α !
			if (alpha < 0.1388) # // 0 < α < 0.1388  ==>  ymax ≲ -1.5
				# The case f<0 is already handled above
				#assert(f >= 0);
				#assert(ymax < -1);
				#double ymax_r, f_ymax_r, fp_ymax_r, f2p_ymax_r;
				i_ymax_r = findYIndexNotLess(ymax)
				#assert(i_ymax_r + 1 < NY); // Because ymax ≲ -1.5.
				y_shift = 0.5 * (Y[i_ymax_r+2] - Y[i_ymax_r+1])
				i_ymax_r = findYIndexNotLess(ymax + y_shift)
				f_ymax_r = F(i_ymax_r, alpha)
				if (f > f_ymax_r)
					ymax_r = Y[i_ymax_r+1]
					fp_ymax_r = Fp(i_ymax_r, alpha)
					f2p_ymax_r = E[i_ymax_r+1]
					q = alpha
					c2 = q * (ymax - q) / 2
					r2 = sqrt(abs(c2))
					h = sqrt(fmax - f) / r2
					sqrt_df = sqrt(fmax - f_ymax_r)
					hr = sqrt_df / r2
					wr = ymax_r - ymax
					sr = -2 * r2 * sqrt_df / fp_ymax_r
					r = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(0, hr, 0, wr, 1, sr, -2 * r2 * r2 / fp_ymax_r * (1 + 2 * (fmax - f_ymax_r) * (f2p_ymax_r / fp_ymax_r) / fp_ymax_r), true)
					y = ymax + rational_cubic_interpolation(h, 0, hr, 0, wr, 1, sr, r)
				else
					y = y0ForPositiveFAndNegativeY(f, alpha, ymax, fmax)
				end

			else
				q = alpha
				c2 = q * (ymax - q) / 2
				r2 = sqrt(abs(c2))
				h = sqrt(fmax - f) / r2
				ymax2 = ymax * ymax
				r3 = (q * (1 + q * (-2 * q + 3 * ymax) - ymax2)) / 6 / c2
				r4 = (q * (q * (4 + q * (-6 * q + 12 * ymax) - 7 * ymax2) + ymax * (-3 + ymax2))) / 24 / c2  # For reference: r5 = (q * (-3 + ymax2 * (6 - ymax2) + q * (q * (20 + q * (-24 * q + 60 * ymax) - 50 * ymax2) + ymax * (-25 + 15 * ymax2)))) / 120 / c2;
				R = (h * (1 + h * ((3 * r3) / 4 - r4 / r3))) / (1 + h * ((5 * r3) / 4 - r4 / r3))
				if (ymax < 0 && f > -10 * fmax) # Adjust the expansion out of the maximum such that it goes exactly through the origin at (y=0,f=0), but only if the maximum is to the left of.
					# assert(fmax >= 0);
					h0 = sqrt(fmax - 0) / r2
					R0 = (h0 * (1 + h0 * ((3 * r3) / 4 - r4 / r3))) / (1 + h0 * ((5 * r3) / 4 - r4 / r3))
					y = (f * (fmax * (ymax + R0) + f * (ymax + R)) + fmax * fmax * (R - R0)) / (fmax * fmax + f * f)
				else
					y = ymax + R
				end
			end
		end
	end
	# Improvement of initial guess based on asymptotic approximation for y -> +infty where f(y) ~ ln(2/pi)/2+alpha·y-y²/2-ln(y) + O(1/y²)
	# This is applied only when f<0 and y>y_threshold for some moderately large y_threshold. We use y_threshold=5.
	# Then, we solve the quadratic expansion around the current initial guess y. Note that ln(2/pi) = -0.45158270528945486
	# Maxima code:   tmp:ev(log(2/%pi)/2-f+alpha*y-y^2/2-log(y),y=yp*(1+t));taylor(tmp,t,0,2);
	# gives:         -(2*log(yp)-log(2/%pi)+2*f-2*alpha*yp+yp^2)/2+(-1+alpha*yp-yp^2)*t-((-1+yp^2)*t^2)/2+...
	# This "pre-polishing" enables the method to converge for very large values of y. Without this, when, say, alpha=1 and y=300, the above initial guess does not enter the domain of attraction of the subsequent Householder iteration.
	if (f < 0 && y > 5)
		a = (1 - y * y) / 2
		b = y * (alpha - y) - 1
		c = y * (alpha - y / 2) - log(y) - 0.45158270528945486 / 2 - f
		p = b / a
		y *= 1 - 0.5 * p + sqrt(0.25 * p * p - c / a)
	end
	return y
end


function houseHolder6Improvement(y::T, f::T, alpha::T) where {T}
    ln2cdf = lnTwoCdf(-y)
	q = exp(-0.5 * y * y - ln2cdf) * SQRT_TWO_OVER_PI
	fy = ln2cdf + alpha * y
	f1 = alpha - q
    #Jherek: because we do not return NaN above
    if (f1 == zero(T))
        return f1
    end
    f2 = q * (y - q)
	y2 = y * y
	f3 = (q * (1 + q * (-2 * q + 3 * y) - y2))
	f4 = (q * (q * (4 + q * (-6 * q + 12 * y) - 7 * y2) + y * (-3 + y2)))
	f5 = (q * (-3 + y2 * (6 - y2) + q * (q * (20 + q * (-24 * q + 60 * y) - 50 * y2) + y * (-25 + 15 * y2))))
	f6 = q * (y * (15 + y2 * (-10 + y2)) + q * (-28 + y2 * (101 - 31 * y2) + q * (q * (120 + q * (-120 * q + 360 * y) - 390 * y2) + y * (-210 + 180 * y2))))
    return houseHolder6Step(-(fy - f) / f1, f2 / f1, f3 / f1, f4 / f1, f5 / f1, f6 / f1)
end

# f(y) = ln(2·Φ(-y)) + α·y  as defined in equation (1.14)
modifiedDeltaWithPremiumFunction(y, alpha) = lnTwoCdf(-y) + alpha * y
function inverseModifiedDeltaWithPremiumFunction(f, alpha)
	y = y0(f, alpha)
    # ONE HouseHolder(6) improvement suffices for all inputs.
	return y + houseHolder6Improvement(y, f, alpha)
end

function attainableAccuracyFactor(y::T, alpha::T) where {T}
	# max( |f(y)/(y·f'(y))| , 1 )
	return max(abs(modifiedDeltaWithPremiumFunction(y, alpha) / (y * (alpha - normpdf(y) / normcdf(-y)))), 1.0)
end

function logmoneynessForForwardDeltaWithPremium(delta, vol, tte, forward)
	alpha = vol * sqrt(tte) *sign(delta)
	f = log(abs(2 * delta)) + alpha^2 / 2
	y = inverseModifiedDeltaWithPremiumFunction(f, alpha)
	return alpha * y - alpha^2 / 2
end
