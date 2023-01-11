export NoConditioner, OneSidedConditioner, PartialExponentialFittingConditioner, ExponentialFittingConditioner, ApproximateFittingConditioner
abstract type PecletConditioner end

struct NoConditioner  <: PecletConditioner
end
struct OneSidedConditioner <: PecletConditioner
end

struct PartialExponentialFittingConditioner <: PecletConditioner
end

struct ApproximateFittingConditioner <: PecletConditioner
end

struct PartialApproximateFittingConditioner <: PecletConditioner
end

struct ExponentialFittingConditioner <: PecletConditioner
end

function conditionedVariance(conditioner::NoConditioner, variance, drift, S, hm, h)
        return variance
end

function conditionedVariance(conditioner::OneSidedConditioner, variance, drift, S, hm, h)
    pecletRatio = drift * hm / variance
    if pecletRatio > 1
        if pecletRatio > 0
            return drift * h
        else
            return -drift * hm
        end
    else
        return variance
    end
end

function conditionedVariance(conditioner::ExponentialFittingConditioner, variance, drift, S, hm, h)
    pecletRatio = drift * hm / variance
    eterm = exp(-2 * pecletRatio)
    return -drift * (hm^2 * eterm - h^2 / eterm - (hm^2 - h^2)) / (hm * eterm + h / eterm - (hm + h))
end

function conditionedVariance(conditioner::ApproximateFittingConditioner, variance, drift, S, hm, h)
    pecletRatio = drift * hm / variance
    return drift*hm*coth(pecletRatio)
end

function conditionedVariance(conditioner::PartialApproximateFittingConditioner, variance, drift, S, hm, h)
    pecletRatio = drift * hm / variance
    if abs(pecletRatio) > 1
        return drift*hm*coth(pecletRatio)
    else
        return variance
    end
end

function conditionedVariance(conditioner::PartialExponentialFittingConditioner, variance, drift, S, hm, h)
    pecletRatio = drift * hm / variance

    if abs(pecletRatio) > 1
        eterm = exp(-2 * pecletRatio)
        return -drift * (hm^2 * eterm - h^2 / eterm - (hm^2 - h^2)) / (hm * eterm + h / eterm - (hm + h))

    else
        return variance
    end
end
