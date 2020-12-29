import AQFED.Math: norminv
using Statistics
import AQFED.TermStructure:
    ConstantBlackModel


function simulate(rng, model::ConstantBlackModel, payoff::VanillaOption, nSim::Int)
    tte = payoff.maturity
    sqrtte = sqrt(tte)
    df = exp(-model.r * tte)
    u = rand(rng, Float64, nSim)
    z = @. norminv(u)
    pathValues =
        @. exp(model.vol * z * sqrtte + (model.r - model.q - 0.5 * model.vol^2) * tte)
    mean(x -> evaluatePayoff(payoff, x, df), pathValues)
end
