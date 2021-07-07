module AQFED

__precompile__(true)
module Math
include("math/erfc.jl")
include("math/normal.jl")
include("math/as241.jl")
include("math/lambertw.jl")
include("math/delbourgo_gregory.jl")
end

__precompile__(true)
module Black
include("black/black.jl")
include("black/iv_solver_common.jl")
include("black/iv_solver_halley.jl")
include("black/iv_solver_householder.jl")
include("black/iv_solver_lisor.jl")
include("black/iv_solver_jaeckel.jl")

impliedVolatility(isCall::Bool, price::T, f::T, strike::T, tte::T,  df::T)  where {T} =  impliedVolatilitySRHalley(isCall, price,f,strike,tte, df, T(0.0), 64, Householder())
export impliedVolatility
end

__precompile__(true)
module Random
include("random/mixmax.jl")
include("random/well1024a.jl")
include("random/mt64.jl")
include("random/blabla.jl")
include("random/philox.jl")
include("random/chacha.jl")
include("random/chacha_simd.jl")
include("random/mrg63k3a.jl")
include("random/ssobol.jl")
include("random/dsobol.jl")
include("random/rngseq.jl")
end

module TermStructure
include("ts/heston.jl")
include("ts/lv.jl")
include("ts/surface.jl")
include("ts/dividend.jl")
end

module American
include("american/andersenlake.jl")
include("american/andersenlaken.jl")
end

module Basket
include("basket/deelstra.jl")
include("basket/asian.jl")
include("basket/pln.jl")
end

module PLN
include("pln/etore_gobet.jl")
end

module MonteCarlo
include("mc/bb.jl")
include("mc/bbvector.jl")
include("mc/payoffs.jl")
include("mc/black.jl")
include("mc/heston.jl")
include("mc/lv.jl")
end
end
