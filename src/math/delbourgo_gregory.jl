export rational_cubic_interpolation,convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side,convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side
minimum_rational_cubic_control_parameter_value = -(1 - sqrt(eps()))
maximum_rational_cubic_control_parameter_value = 2 / (eps()^2)


function is_zero(x::Float64)::Bool
	return abs(x) < floatmin()
end

function rational_cubic_interpolation(x::Float64, x_l::Float64, x_r::Float64, y_l::Float64, y_r::Float64, d_l::Float64, d_r::Float64, r::Float64)::Float64
	h = (x_r - x_l)
	if abs(h) <= 0
		return 0.5 * (y_l + y_r)
	end
	#r should be greater than -1. We do not use  assert(r > -1)  here in order to allow values such as NaN to be propagated as they should.
	t = (x - x_l) / h
	if !(r >= maximum_rational_cubic_control_parameter_value)
		t = (x - x_l) / h
		omt = 1 - t
		t2 = t^2
		omt2 = omt^2
		# Formula (2.4) divided by formula (2.5)
		return (y_r*t2*t + (r*y_r-h*d_r)*t2*omt + (r*y_l+h*d_l)*t*omt2 + y_l*omt2*omt) / (1 + (r-3)*t*omt)
	end
	# Linear interpolation without over-or underflow.
	return y_r*t + y_l*(1-t)
end

function rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(x_l::Float64, x_r::Float64, y_l::Float64, y_r::Float64, d_l::Float64, d_r::Float64, second_derivative_l::Float64)::Float64
	h = (x_r - x_l)
	numerator = 0.5*h*second_derivative_l + (d_r - d_l)
	if is_zero(numerator)
		return 0
	end
	denominator = (y_r-y_l)/h - d_l
	if is_zero(denominator)
		if numerator > 0
			return maximum_rational_cubic_control_parameter_value
		 else
			return minimum_rational_cubic_control_parameter_value
		end
	end
	return numerator / denominator
end

function rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(x_l::Float64, x_r::Float64, y_l::Float64, y_r::Float64, d_l::Float64, d_r::Float64, second_derivative_r::Float64)::Float64
	h = (x_r - x_l)
	numerator = 0.5*h*second_derivative_r + (d_r - d_l)
	if is_zero(numerator)
		return 0.0
	end
	denominator = d_r - (y_r-y_l)/h
	if is_zero(denominator)
		if numerator > 0
			return maximum_rational_cubic_control_parameter_value
		 else
			return minimum_rational_cubic_control_parameter_value
		end
	end
	return numerator / denominator
end

function minimum_rational_cubic_control_parameter(d_l::Float64, d_r::Float64, s::Float64, preferShapePreservationOverSmoothness::Bool)::Float64
	monotonic = d_l*s >= 0 && d_r*s >= 0
	convex = d_l <= s && s <= d_r
	concave = d_l >= s && s >= d_r
	if !monotonic && !convex && !concave  # If 3==r_non_shape_preserving_target, this means revert to standard cubic.
		return minimum_rational_cubic_control_parameter_value
	end
	d_r_m_d_l = d_r - d_l
	d_r_m_s = d_r - s
	s_m_d_l = s - d_l
	r1 = -floatmax()
	r2 = r1
	# If monotonicity on this interval is possible, set r1 to satisfy the monotonicity condition (3.8).
	if monotonic
		if !is_zero(s)  # (3.8), avoiding division by zero.
			r1 = (d_r + d_l) / s # (3.8)
		elseif preferShapePreservationOverSmoothness  # If division by zero would occur, and shape preservation is preferred, set value to enforce linear interpolation.
			 r1 = maximum_rational_cubic_control_parameter_value # This value enforces linear interpolation.
		end
	end
	if convex || concave
		if !(is_zero(s_m_d_l) || is_zero(d_r_m_s))  # (3.18), avoiding division by zero.
			r2 = max(abs(d_r_m_d_l/d_r_m_s), abs(d_r_m_d_l/s_m_d_l))
		elseif preferShapePreservationOverSmoothness
			r2 = maximum_rational_cubic_control_parameter_value
		end # This value enforces linear interpolation.
	 elseif monotonic && preferShapePreservationOverSmoothness
		r2 = maximum_rational_cubic_control_parameter_value
	end # This enforces linear interpolation along segments that are inconsistent with the slopes on the boundaries, e.g., a perfectly horizontal segment that has negative slopes on either edge.
	return max(minimum_rational_cubic_control_parameter_value, max(r1, r2))
end

function convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(x_l::Float64, x_r::Float64, y_l::Float64, y_r::Float64, d_l::Float64, d_r::Float64, second_derivative_l::Float64, preferShapePreservationOverSmoothness::Bool)::Float64
	r = rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(x_l, x_r, y_l, y_r, d_l, d_r, second_derivative_l)
	r_min = minimum_rational_cubic_control_parameter(d_l, d_r, (y_r-y_l)/(x_r-x_l), preferShapePreservationOverSmoothness)
	return max(r, r_min)
end

function convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(x_l::Float64, x_r::Float64, y_l::Float64, y_r::Float64, d_l::Float64, d_r::Float64, second_derivative_r::Float64, preferShapePreservationOverSmoothness::Bool)::Float64
	r = rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(x_l, x_r, y_l, y_r, d_l, d_r, second_derivative_r)
	r_min = minimum_rational_cubic_control_parameter(d_l, d_r, (y_r-y_l)/(x_r-x_l), preferShapePreservationOverSmoothness)
	return max(r, r_min)
end
