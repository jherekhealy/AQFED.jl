export discountFactor, logDiscountFactor, forward, logForward, varianceByLogmoneyness
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

struct LocalVolatilityModel{S}
	surface::S
	r::Float64
	q::Float64
end

function discountFactor(model, t)
	exp(-model.r * t)
end

function discountFactor(model::TSBlackModel{S, C1, C2}, t) where {S, C1 <: Curve, C2 <: Curve}
	discountFactor(model.discountCurve, t)
end

function logDiscountFactor(model::TSBlackModel{S, C1, C2}, t) where {S, C1 <: Curve, C2 <: Curve}
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

function logForward(model::TSBlackModel{S, C1, C2}, lnspot, t) where {S, C1 <: Curve, C2 <: Curve}
	return lnspot - log(discountFactor(model.driftCurve, t))
end

function forward(model::TSBlackModel{S, C1, C2}, spot, t) where {S, C1 <: Curve, C2 <: Curve}
	return spot / discountFactor(model.driftCurve, t)
end

function varianceByLogmoneyness(model::ConstantBlackModel, y, t)
	return model.vol^2
end


function varianceByLogmoneyness(model, y, t)
	return varianceByLogmoneyness(model.surface, y, t)
end

using FiniteDifferences, ForwardDiff


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
	return 1 - y / w(y) * dwdy + (dwdy)^2 * (-1 / 4 - 1 / w(y) + y^2 / w(y)^2) / 4 + d2wdy2 / 2
end
