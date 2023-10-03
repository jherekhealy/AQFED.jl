using AQFED, Test, StatsBase
using AQFED.Collocation, PPInterpolation, AQFED.Bachelier

@testset "flat-basket" begin
strikes = [0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4]
vols = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
tte = 1.0
forward = 1.0
prices, weights = Collocation.weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor=1e-7)
bvols = Bachelier.bachelierImpliedVolatility.(prices,true,strikes,tte,forward,1.0)
bvolSpline = CubicSplineNatural(log.(strikes./forward),bvols)
guess = AQFED.VolatilityModels.initialGuessNormalATM(forward,tte,0.0,bvolSpline(0.0),PPInterpolation.evaluateDerivative(bvolSpline,0.0),PPInterpolation.evaluateSecondDerivative(bvolSpline,0.0))
section = AQFED.VolatilityModels.calibrateNormalSABRSectionFromGuess(tte,forward,strikes,bvols,ones(length(bvols)),guess)
fitPrices = [Bachelier.bachelierFormula(true,strikei,forward,bvols[i],tte,1.0) for (i,strikei)=enumerate(strikes)]
fitPrices-prices
end