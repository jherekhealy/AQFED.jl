using AQFED, Test, StatsBase
import AQFED.Black


@testset "JaeckelAccuracyFixedStrike" begin
  strike = 200.0
  forward = 100.0
  tte = 1.0
  isCall = true
  intr = forward - strike
  if !isCall
    intr = -intr
  end
  vols =0.02:0.01:4.0
  mvols = zeros(length(vols))
  mprices = zeros(length(vols))
  prices = zeros(length(vols))

  for (j,vol) in enumerate(vols)
    price = Black.blackScholesFormula(isCall, strike, forward, vol*vol * tte, 1.0, 1.0)
    prices[j] = price
    mvols[j]= Black.impliedVolatilityJaeckel(isCall, price, forward, strike, tte, 1.0)
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # println(mprice-price, " ",price)
  end
  println("JAECKEL RMSD ",rmsd(mvols,vols)," MAXAD ",maxad(mvols,vols)," PriceMAXAD ",maxad(mprices,prices), " ",maxad(mprices ./ prices,prices ./ prices))
  @test isless(maxad(mvols,vols)/vols[end], 3e-15)
  for (j,vol) in enumerate(vols)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
  end
  println("HALLEY RMSD ",rmsd(mvols,vols)," MAXAD ",maxad(mvols,vols)," PriceMAXAD ",maxad(mprices,prices), " ",maxad(mprices ./ prices,prices ./ prices))

  for (j,vol) in enumerate(vols)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
  end
  println("CMETHOD RMSD ",rmsd(mvols,vols)," MAXAD ",maxad(mvols,vols)," PriceMAXAD ",maxad(mprices,prices), " ",maxad(mprices ./prices,prices ./prices))

  for (j,vol) in enumerate(vols)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64,Black.Householder())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # mprice = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # println(mprice-price, " ",price)
  end
  println("HOUSEHOLDER RMSD ",rmsd(mvols,vols)," MAXAD ",maxad(mvols,vols)," PriceMAXAD ",maxad(mprices,prices)," ",maxad(mprices ./ prices,prices ./ prices))

end

@testset "JaeckelAccuracyFixedVol" begin
  forward = 100.0
  tte = 1.0
  isCall = true
  vol =0.1
  strikes = 100.0:1.0:500.0
  mvols = zeros(length(strikes))
  mprices = zeros(length(strikes))
  prices = zeros(length(strikes))
  vols = ones(length(strikes))*vol
  for (j,strike) in enumerate(strikes)
    price = Black.blackScholesFormula(isCall, strike, forward, vol*vol * tte, 1.0, 1.0)
    prices[j] = price
    mvols[j]= Black.impliedVolatilityJaeckel(isCall, price, forward, strike, tte, 1.0)
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # println(mprice-price, " ",price)
  end
  println("JAECKEL RMSD ",rmsd(mvols,vols)," MAXAD ",maxad(mvols,vols)," PriceMAXAD ",maxad(mprices,prices), " ",maxad(mprices ./ prices,prices ./ prices))
  @test isless(maxad(mvols,vols), 4e-15)
  for (j,strike) in enumerate(strikes)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
  end
  println("HALLEY RMSD ",rmsd(mvols,vols)," MAXAD ",maxad(mvols,vols)," PriceMAXAD ",maxad(mprices,prices), " ",maxad(mprices ./ prices,prices ./ prices))
  @test isless(maxad(mvols,vols), 4e-15)

  for (j,strike) in enumerate(strikes)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
  end
  println("CMETHOD RMSD ",rmsd(mvols,vols)," MAXAD ",maxad(mvols,vols)," PriceMAXAD ",maxad(mprices,prices), " ",maxad(mprices ./prices,prices ./prices))
  @test isless(maxad(mvols,vols), 4e-15)

  for (j,strike) in enumerate(strikes)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64,Black.Householder())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # mprice = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # println(mprice-price, " ",price)
  end
  println("HOUSEHOLDER RMSD ",rmsd(mvols,vols)," MAXAD ",maxad(mvols,vols)," PriceMAXAD ",maxad(mprices,prices)," ",maxad(mprices ./ prices,prices ./ prices))
  @test isless(maxad(mvols,vols), 4e-15)

end

@testset "JaeckelSet6" begin
  price = 271.43234885190117
  strike = 275.0
  forward = 356.73063159822254
  tte = 1.5917808219178082
  isCall = false
  intr = forward - strike
  if !isCall
    intr = -intr
  end
  ivj = Black.impliedVolatilityJaeckel(isCall, price, forward, strike, tte, 1.0)
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println("Jaeckel ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-25)
  @test isapprox(e2, 0, atol = 1e-25)

  guess = 0.0
  ivj = Black.impliedVolatilityLiSOR(isCall, price, forward, strike, tte, 1.0, guess, 0e-14, 64, Black.SORTS()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SOR-Li ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley())
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-Halley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.SuperHalley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-SHalley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-CMethod ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)
end

@testset "JaeckelSet5" begin
  price = 355.12907714987386
  forward = 2838.9998434067384
  strike = 2485.0
  tte = 0.08493150684931507

  isCall = true
  intr = forward - strike
  if !isCall
    intr = -intr
  end
  ivj = Black.impliedVolatilityJaeckel(isCall, price, forward, strike, tte, 1.0)
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-25)
  @test isapprox(e2, 0, atol = 1e-25)

  guess = 0.0
  ivj = Black.impliedVolatilityLiSOR(isCall, price, forward, strike, tte, 1.0, guess, 0e-14, 64, Black.SORTS())
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SOR-Li ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-Halley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.SuperHalley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-SHalley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-CMethod ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

end

@testset "JaeckelSet1" begin
  price = 0.35000000053458924
  forward = 1.0
  strike = 0.65
  tte = 0.0191780822

  isCall = true
  intr = forward - strike
  if !isCall
    intr = -intr
  end
  ivj = Black.impliedVolatilityJaeckel(isCall, price, forward, strike, tte, 1.0)
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-25)
  @test isapprox(e2, 0, atol = 1e-25)

  guess = 0.0
  ivj = Black.impliedVolatilityLiSOR(isCall, price, forward, strike, tte, 1.0, guess, 0e-14, 64, Black.SORTS())
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SOR-Li ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-Halley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.SuperHalley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-SHalley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-CMethod ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)
end

@testset "JaeckelSetATM" begin
  forward = 100.0
  strike = 100.0
  vol = 0.08
  tte = 1.0
  df = 1.0
  isCall = true
  price = Black.blackScholesFormula(isCall, strike, forward, vol * vol * tte, 1.0, df)
  intr = forward - strike
  if !isCall
    intr = -intr
  end
  ivj = Black.impliedVolatilityJaeckel(isCall, price, forward, strike, tte, 1.0)
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-25)
  @test isapprox(e2, 0, atol = 1e-25)
  strike = 101.0
  price = Black.blackScholesFormula(isCall, strike, forward, vol * vol * tte, 1.0, df)
  intr = forward - strike
  if !isCall
    intr = -intr
  end
  ivj = Black.impliedVolatilityJaeckel(isCall, price, forward, strike, tte, 1.0)
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-25)
  @test isapprox(e2, 0, atol = 1e-25)

  guess = 0.0
  ivj = Black.impliedVolatilityLiSOR(isCall, price, forward, strike, tte, 1.0, guess, 0e-14, 64, Black.SORTS())
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SOR-Li ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-Halley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.SuperHalley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-SHalley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-CMethod ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol = 1e-12)
  @test isapprox(e2, 0, atol = 1e-12)

end
