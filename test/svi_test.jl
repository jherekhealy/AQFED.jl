using AQFED, Test
using StatsBase
using AQFED.Black
using AQFED.Collocation
import Polynomials: coeffs
using AQFED.Math, AQFED.PDDE, AQFED.VolatilityModels
using PPInterpolation
import AQFED.TermStructure: varianceByLogmoneyness, XSSVISection, InterpolatedLogDiscountFactorCurve, VarianceSurfaceBySection
import AQFED.VolatilityModels: calibrateXSSVISection, VarianceSurfaceBySection

include("market_data.jl")

@testset "SVI-20260512" begin
  spot, ts, dfs, uForwards, strikes, vols = makeMarketData20260512()
  logdfInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, log.(dfs)))
  logForwardInterp = PPInterpolation.CubicSplineNatural(vcat(0.0, ts), vcat(0.0, -log.(uForwards)))
  discountCurve = InterpolatedLogDiscountFactorCurve(logdfInterp)
  driftCurve = InterpolatedLogDiscountFactorCurve(logForwardInterp)
  surface = AQFED.VolatilityModels.SplineSurface(ts, uForwards, log.(strikes ./ uForwards), vols)

  sections = Vector{AQFED.TermStructure.XSSVISection}(undef, length(ts))

  for (i, ti) in enumerate(ts)
    sections[i] = AQFED.VolatilityModels.calibrateXSSVISection(
      ti,
      uForwards[i],
      log.(strikes[i, :] ./ uForwards[i]),
      vols[i, :],
      ones(length(vols[i, :])),
      prevSection=(i-1) == 0 ? nothing : sections[i-1]
    )
  end
  MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0

  T_5m = 5.0 / MINUTES_PER_YEAR
  T_1h = 60.0 / MINUTES_PER_YEAR
  T_1d = 24.0 * 60.0 / MINUTES_PER_YEAR
  surface = AQFED.TermStructure.VarianceSurfaceBySection{XSSVISection}(sections, ts)
  vol = sqrt(AQFED.TermStructure.varianceByLogmoneyness(surface, log(strikes[2]/uForwards[2]), ts[2]))
  volAtm = sqrt(varianceByLogmoneyness(surface, 0.0, ts[1]))

  using Plots
  p = plot(title="Standard XSSVI Extrapolation", ylabel="Implied Volatility in %", xlabel="Scaled log-moneyness y/sqrt(volATM * T)")

  for T_i = (T_1d, T_1h, T_5m)
    ydev = volAtm*sqrt(T_i)
    yrange = range(-ydev, +ydev, 1000)
    plot!(p, yrange ./ ydev, map(y->100*sqrt(varianceByLogmoneyness(surface, y, T_i)), yrange), label=string("T=", round(T_i*365*24*60), " minutes"))
  end
  plot(p)

  shortEndStrategy=AQFED.VolatilityModels.PowerLawPsiExtrapolation(0.55)
  p = plot(title=string("Power Law XSSVI Extrapolation γ=",shortEndStrategy.γ), ylabel="Implied Volatility in %", xlabel="Scaled log-moneyness y/sqrt(volATM * T)")

  for T_i = (T_1d, T_1h, T_5m)
           ydev = volAtm*sqrt(T_i)
           yrange = range(-ydev, +ydev, 1000)
           plot!(p, yrange ./ ydev, map(y->100*sqrt(varianceByLogmoneyness(surface, y, T_i,shortEndStrategy=shortEndStrategy)), yrange), label=string("T=", round(T_i*365*24*60), " minutes"))
         end
  plot(p)
  T_i = T_5m
  ydev = volAtm*sqrt(T_i)
  yrange = range(-ydev, +ydev, 100)
  map(y -> AQFED.Black.blackScholesFormula(true, exp(y), 1.0, varianceByLogmoneyness(surface, y, T_i)*T_i, 1.0, 1.0), yrange)

  using ForwardDiff

  deltaK50 = ForwardDiff.derivative.(Ref(y -> AQFED.Black.blackScholesFormula(true, exp(y), 1.0, varianceByLogmoneyness(surface, y, T_i,shortEndStrategy=AQFED.VolatilityModels.PowerLawPsiExtrapolation(0.50))*T_i, 1.0, 1.0)), yrange)
  deltaK55 = ForwardDiff.derivative.(Ref(y -> AQFED.Black.blackScholesFormula(true, exp(y), 1.0, varianceByLogmoneyness(surface, y, T_i,shortEndStrategy=AQFED.VolatilityModels.PowerLawPsiExtrapolation(0.55))*T_i, 1.0, 1.0)), yrange)
 deltaK = ForwardDiff.derivative.(Ref(y -> AQFED.Black.blackScholesFormula(true, exp(y), 1.0, varianceByLogmoneyness(surface, y, T_i)*T_i, 1.0, 1.0)), yrange)

  plot(-yrange ./ ydev,-deltaK, title="Cumulative Probability Density")

  p = plot(-yrange[2:end] ./ ydev,deltaK[2:end]-deltaK[1:end-1], title="Probability Density", label="Linear ψ")
  plot!(p, -yrange[2:end] ./ ydev,deltaK50[2:end]-deltaK50[1:end-1], title="Probability Density", label="Power Law 0.50")
  plot!(p, -yrange[2:end] ./ ydev,deltaK55[2:end]-deltaK55[1:end-1], title="Probability Density", label="Power Law 0.55")

end
