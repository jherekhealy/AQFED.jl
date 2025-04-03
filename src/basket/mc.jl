using AQFED.Random
using AQFED.MonteCarlo
import AQFED.TermStructure:ConstantBlackModel
export MonteCarloEngine
struct MonteCarloEngine <: BasketPricer
    withBB::Bool
    nSim::Int
end
MonteCarloEngine() = MonteCarloEngine(true, 1024 * 64)

function priceEuropean(
    p::MonteCarloEngine,
    isCall::Bool,
    strike::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}
)::T where {T,TV}
    #rng = ScrambledSobolSeq(length(spot), p.nSim*2, FaureTezuka(ScramblingRngAdapter(Chacha8SIMD(UInt32))))
    rng = DigitalSobolSeq(length(spot), p.nSim,Chacha8SIMD(UInt32))
    models = Vector{ConstantBlackModel}(undef,length(spot))
    tte = 1.0 #total variance is the important bit
    for (i, tvar) = enumerate(totalVariance)
        models[i] = ConstantBlackModel(sqrt(tvar / tte), 0.0, 0.0)
    end
    payoff = MonteCarlo.VanillaBasketOption(isCall, strike, weight, MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0)
    price = MonteCarlo.simulate(rng, models, forward, correlation, payoff,p.nSim)
    return price * discountFactor
end