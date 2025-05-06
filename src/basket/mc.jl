using AQFED.Random
using AQFED.MonteCarlo
import AQFED.TermStructure:ConstantBlackModel
export MonteCarloEngine
struct MonteCarloEngine <: BasketPricer
    withBB::Bool
    nSim::Int
end
MonteCarloEngine() = MonteCarloEngine(true, 1024 * 64)

mutable struct MonteCarloRngEngine <: BasketPricer
    engine::MonteCarloEngine
    rng::DigitalSobolSeq
    isRngInitialized::Bool
end
function numberOfSimulations(p::MonteCarloEngine)
    return p.nSim
end

function numberOfSimulations(p::MonteCarloRngEngine)
    return p.engine.nSim
end
function makeRng(p::MonteCarloEngine, dim::Int)
    return DigitalSobolSeq(dim, p.nSim,Chacha8SIMD(UInt32))
end

function isBrownianBridge(p::MonteCarloEngine)
    return p.withBB
end


function isBrownianBridge(p::MonteCarloRngEngine)
    return p.engine.withBB
end

function makeRng(p::MonteCarloRngEngine, dim::Int)
    if !p.isRngInitialized 
        p.rng = DigitalSobolSeq(dim, p.engine.nSim*128,Chacha8SIMD(UInt32))
        p.isRngInitialized = true
    end
    return p.rng
end
   
function priceEuropean(
    p::Union{MonteCarloEngine,MonteCarloRngEngine},
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
    rng = makeRng(p,length(spot))
    models = Vector{ConstantBlackModel}(undef,length(spot))
    tte = 1.0 #total variance is the important bit
    for (i, tvar) = enumerate(totalVariance)
        models[i] = ConstantBlackModel(sqrt(tvar / tte), 0.0, 0.0)
    end
    payoff = MonteCarlo.VanillaBasketOption(isCall, strike, weight, MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0)
    price = MonteCarlo.simulate(rng, models, forward, correlation, payoff,numberOfSimulations(p))
    return price * discountFactor
end

function priceEuropean(
    p::MonteCarloEngine,
    isCall::Bool,
    strikes::AbstractArray{T},
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}
)where {T,TV}
    #rng = ScrambledSobolSeq(length(spot), p.nSim*2, FaureTezuka(ScramblingRngAdapter(Chacha8SIMD(UInt32))))
    rng = DigitalSobolSeq(length(spot), p.nSim,Chacha8SIMD(UInt32))
    models = Vector{ConstantBlackModel}(undef,length(spot))
    tte = 1.0 #total variance is the important bit
    for (i, tvar) = enumerate(totalVariance)
        models[i] = ConstantBlackModel(sqrt(tvar / tte), 0.0, 0.0)
    end
    payoffs = [MonteCarlo.VanillaBasketOption(isCall, strike, weight, MonteCarlo.BulletCashFlow(tte, tte, false, 0.0), 0.0) for strike = strikes]
    payoff = MonteCarlo.ListPayoff(payoffs)
    price = MonteCarlo.simulate(rng, models, forward, correlation, payoff,p.nSim)
    return price .* discountFactor
end

function priceEuropeanSpread(
    p::Union{MonteCarloEngine,MonteCarloRngEngine},
    isCall::Bool,
    strikePct::T,
    discountFactor::T, #discount factor to payment
    spot::AbstractArray{<:T},
    forward::AbstractArray{TV}, #forward to option maturity
    totalVariance::AbstractArray{<:T}, #vol^2 * τ
    weight::AbstractArray{<:T},
    correlation::Matrix{TV}
)where {T,TV}
#todo to indexPositive-1 *strikePct
   return priceEuropean(p,
   isCall,
   0.0,
   discountFactor, #discount factor to payment
   spot,
   forward, #forward to option maturity
   totalVariance, #vol^2 * τ
   weight,
   correlation)
end