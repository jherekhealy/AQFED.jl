export discountFactor, logDiscountFactor, forward, logForward, varianceByLogmoneyness, totalVarianceByLogmoneyness
export ConstantBlackModel, TSBlackModel, LocalVolatilityModel
struct ConstantBlackModel
	vol::Float64
	r::Float64
	q::Float64
end

struct TSBlackModel{S, C1 <: Curve, C2 <: Curve}
	surface::S
	discountCurve::C1
	driftCurve::C2
end

struct LocalVolatilityModel{S,C1,C2}
	surface::S
    driftCurve::C1
    discountCurve::C2
end

function LocalVolatilityModel(surface::S, r::Float64, q::Float64) where{S}
	forwardCurve = ConstantRateCurve(r-q)
    discountCurve = ConstantRateCurve(r)
	return LocalVolatilityModel(surface, forwardCurve, discountCurve)
end

function discountFactor(model, t)
	exp(-model.r * t)
end

function discountFactor(model::Union{TSBlackModel{S, C1, C2},LocalVolatilityModel{S,C1,C2}}, t) where {S, C1 <: Curve, C2 <: Curve}
	discountFactor(model.discountCurve, t)
end

function logDiscountFactor(model::Union{TSBlackModel{S, C1, C2},LocalVolatilityModel{S,C1,C2}}, t) where {S, C1 <: Curve, C2 <: Curve}
	log(discountFactor(model.discountCurve, t))
end

function logDiscountFactor(model, t)
	return -model.r * t
end

function logForward(model, lnspot, t)
	return lnspot + (model.r - model.q) * t
end

function forward(model, spot, t)
	return spot * exp((model.r - model.q) * t)
end

function logForward(model::Union{TSBlackModel{S, C1, C2},LocalVolatilityModel{S,C1,C2}}, lnspot, t) where {S, C1 <: Curve, C2 <: Curve}
	return lnspot - log(discountFactor(model.driftCurve, t))
end

function forward(model::Union{TSBlackModel{S, C1, C2},LocalVolatilityModel{S,C1,C2}}, spot, t) where {S, C1 <: Curve, C2 <: Curve}
	return spot / discountFactor(model.driftCurve, t)
end

function varianceByLogmoneyness(model::ConstantBlackModel, y, t)
	return model.vol^2
end


function varianceByLogmoneyness(model::T, y, t) where {T <: Union{TSBlackModel,LocalVolatilityModel}}
	return varianceByLogmoneyness(model.surface, y, t)
end

using FiniteDifferences, ForwardDiff

@inline function localVarianceDenominator(
	w::T,
	wy::T,
	wyy::T,
	y::T,
) where {T}
	return 1.0 -
		   y/w*wy +
		   0.25 * wy^2 *
		   (-0.25 - 1.0/w + y^2/w^2) +
		   0.5*wyy
end

function localVarianceDenominator(w, y; autodiff = :central)
	dwdy, d2wdy2 = if autodiff == :central
		dwdy = FiniteDifferences.central_fdm(5, 1)(w, y)
		d2wdy2 = FiniteDifferences.central_fdm(5, 2)(w, y)
		dwdy, d2wdy2
	else
		dwdy = ForwardDiff.derivative(w, y)
		d2wdy2 = ForwardDiff.derivative(z -> ForwardDiff.derivative(w, z), y)
		dwdy, d2wdy2
	end
	return localVarianceDenominator(w, dwdy, d2wdy2, y)
end

function totalVarianceByLogmoneyness(
	model,
	y::Real,
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
