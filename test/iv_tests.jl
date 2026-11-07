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
  vols = 0.02:0.01:4.0
  mvols = zeros(length(vols))
  mprices = zeros(length(vols))
  prices = zeros(length(vols))

  for (j, vol) in enumerate(vols)
    price = Black.blackScholesFormula(isCall, strike, forward, vol * vol * tte, 1.0, 1.0)
    prices[j] = price
    mvols[j] = Black.impliedVolatilityJaeckel(isCall, price, forward, strike, tte, 1.0)
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # println(mprice-price, " ",price)
  end
  println("JAECKEL RMSD ", rmsd(mvols, vols), " MAXAD ", maxad(mvols, vols), " PriceMAXAD ", maxad(mprices, prices), " ", maxad(mprices ./ prices, prices ./ prices))
  @test isless(maxad(mvols, vols) / vols[end], 3e-15)
  for (j, vol) in enumerate(vols)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
  end
  println("HALLEY RMSD ", rmsd(mvols, vols), " MAXAD ", maxad(mvols, vols), " PriceMAXAD ", maxad(mprices, prices), " ", maxad(mprices ./ prices, prices ./ prices))

  for (j, vol) in enumerate(vols)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
  end
  println("CMETHOD RMSD ", rmsd(mvols, vols), " MAXAD ", maxad(mvols, vols), " PriceMAXAD ", maxad(mprices, prices), " ", maxad(mprices ./ prices, prices ./ prices))

  for (j, vol) in enumerate(vols)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Householder())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # mprice = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # println(mprice-price, " ",price)
  end
  println("HOUSEHOLDER RMSD ", rmsd(mvols, vols), " MAXAD ", maxad(mvols, vols), " PriceMAXAD ", maxad(mprices, prices), " ", maxad(mprices ./ prices, prices ./ prices))
  for (j, vol) in enumerate(vols)
    price = prices[j]
    mvols[j] = Black.impliedVolatilityIG(isCall, price, forward, strike, tte, 1.0)
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # mprice = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # println(mprice-price, " ",price)
  end
  println("INVERSEGAUSSIAN RMSD ", rmsd(mvols, vols), " MAXAD ", maxad(mvols, vols), " PriceMAXAD ", maxad(mprices, prices), " ", maxad(mprices ./ prices, prices ./ prices))
end

@testset "JaeckelAccuracyFixedVol" begin
  forward = 100.0
  tte = 1.0
  isCall = true
  vol = 0.1
  strikes = 100.0:1.0:500.0
  mvols = zeros(length(strikes))
  mprices = zeros(length(strikes))
  prices = zeros(length(strikes))
  vols = ones(length(strikes)) * vol
  for (j, strike) in enumerate(strikes)
    price = Black.blackScholesFormula(isCall, strike, forward, vol * vol * tte, 1.0, 1.0)
    prices[j] = price
    mvols[j] = Black.impliedVolatilityJaeckel(isCall, price, forward, strike, tte, 1.0)
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # println(mprice-price, " ",price)
  end
  println("JAECKEL RMSD ", rmsd(mvols, vols), " MAXAD ", maxad(mvols, vols), " PriceMAXAD ", maxad(mprices, prices), " ", maxad(mprices ./ prices, prices ./ prices))
  @test isless(maxad(mvols, vols), 4e-15)
  for (j, strike) in enumerate(strikes)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
  end
  println("HALLEY RMSD ", rmsd(mvols, vols), " MAXAD ", maxad(mvols, vols), " PriceMAXAD ", maxad(mprices, prices), " ", maxad(mprices ./ prices, prices ./ prices))
  @test isless(maxad(mvols, vols), 4e-15)

  for (j, strike) in enumerate(strikes)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
  end
  println("CMETHOD RMSD ", rmsd(mvols, vols), " MAXAD ", maxad(mvols, vols), " PriceMAXAD ", maxad(mprices, prices), " ", maxad(mprices ./ prices, prices ./ prices))
  @test isless(maxad(mvols, vols), 4e-15)

  for (j, strike) in enumerate(strikes)
    price = prices[j]
    mvols[j] = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Householder())
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # mprice = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # println(mprice-price, " ",price)
  end
  println("HOUSEHOLDER RMSD ", rmsd(mvols, vols), " MAXAD ", maxad(mvols, vols), " PriceMAXAD ", maxad(mprices, prices), " ", maxad(mprices ./ prices, prices ./ prices))
  @test isless(maxad(mvols, vols), 4e-15)
  for (j, strike) in enumerate(strikes)
    price = prices[j]
    mvols[j] = Black.impliedVolatilityIG(isCall, price, forward, strike, tte, 1.0)
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # mprice = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # println(mprice-price, " ",price)
  end
  println("IG RMSD ", rmsd(mvols, vols), " MAXAD ", maxad(mvols, vols), " PriceMAXAD ", maxad(mprices, prices), " ", maxad(mprices ./ prices, prices ./ prices))
  @test isless(maxad(mvols, vols), 1e-14) #IG is less accurate on deep OTM: 7.8e-15

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
  @test isapprox(e1, 0, atol=1e-25)
  @test isapprox(e2, 0, atol=1e-25)

  guess = 0.0
  ivj = Black.impliedVolatilityLiSOR(isCall, price, forward, strike, tte, 1.0, guess, 0e-14, 64, Black.SORTS()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SOR-Li ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley())
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-Halley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.SuperHalley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-SHalley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-CMethod ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)
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
  @test isapprox(e1, 0, atol=1e-25)
  @test isapprox(e2, 0, atol=1e-25)

  guess = 0.0
  ivj = Black.impliedVolatilityLiSOR(isCall, price, forward, strike, tte, 1.0, guess, 0e-14, 64, Black.SORTS())
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SOR-Li ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=2e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-Halley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.SuperHalley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-SHalley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-CMethod ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

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
  @test isapprox(e1, 0, atol=1e-25)
  @test isapprox(e2, 0, atol=1e-25)

  guess = 0.0
  ivj = Black.impliedVolatilityLiSOR(isCall, price, forward, strike, tte, 1.0, guess, 0e-14, 64, Black.SORTS())
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SOR-Li ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-Halley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.SuperHalley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-SHalley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-CMethod ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)
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
  @test isapprox(e1, 0, atol=1e-25)
  @test isapprox(e2, 0, atol=1e-25)
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
  @test isapprox(e1, 0, atol=1e-14)
  @test isapprox(e2, 0, atol=1e-25)

  guess = 0.0
  ivj = Black.impliedVolatilityLiSOR(isCall, price, forward, strike, tte, 1.0, guess, 0e-14, 64, Black.SORTS())
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SOR-Li ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.Halley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-Halley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.SuperHalley()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-SHalley ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilitySRHalley(isCall, price, forward, strike, tte, 1.0, 0e-14, 64, Black.CMethod()) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" SR-CMethod ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

  ivj = Black.impliedVolatilityIG(isCall, price, forward, strike, tte, 1.0) #1e-14=> 0 but 0=> 1e-13 error; likely because blackfor erfc
  e1 = Black.blackScholesFormula(isCall, strike, forward, ivj * ivj * tte, 1.0, 1.0) - price
  e2 = Black.blackScholesFormula(isCall, forward, strike, ivj * ivj * tte, 1.0, 1.0) - price + intr
  println(" IG ", ivj, " ", e1, " ", e2)
  @test isapprox(e1, 0, atol=1e-12)
  @test isapprox(e2, 0, atol=1e-12)

end


@testset "Cui" begin
  S0 = 100
  r = 0.03
  q = 0.0
  num_K = 39
  num_tau = 39
  num_sigma = 39
  K0 = 105:(800-105)/num_K:800
  tau0 = 0.01:(2-0.01)/num_tau:2
  sigma0 = 0.01:(0.99-0.01)/num_sigma:0.99
  K = [K for K in K0 for tau in tau0 for sigma in sigma0]
  tau = [tau for K in K0 for tau in tau0 for sigma in sigma0]
  sigma = [sigma for K in K0 for tau in tau0 for sigma in sigma0]
  C_real = [Black.blackScholesFormula(true, K, S0, sigma^2 * tau, exp(-(r - q) * tau), exp(-r * tau)) for K in K0 for tau in tau0 for sigma in sigma0]
  indices = findall(x -> x > 1e-20, C_real)
  C_real = C_real[indices]
  K = K[indices]
  tau = tau[indices]
  sigma = sigma[indices]
  IV_jaeckel = [Black.impliedVolatilityJaeckel(true, C_reali, S0 * exp((r - q) * taui), Ki, taui, exp(-r * taui)) for (C_reali, Ki, taui, sigmai) in zip(C_real, K, tau, sigma)]
  t6_a = abs.(IV_jaeckel - sigma)
  stats = [mean(t6_a), std(t6_a), maximum(t6_a), minimum(t6_a)]
  @test isapprox(stats[1], 0, atol=1e-15)
  @test isapprox(stats[3], 0, atol=2e-14)
  IV_srhou = [Black.impliedVolatilitySRHalley(true, C_reali, S0 * exp((r - q) * taui), Ki, taui, exp(-r * taui), eps(), 32, Black.Householder()) for (C_reali, Ki, taui, sigmai) in zip(C_real, K, tau, sigma)]
  t7_a = abs.(IV_srhou - sigma)
  stats = [mean(t7_a), std(t7_a), maximum(t7_a), minimum(t7_a)]
  @test isapprox(stats[1], 0, atol=1e-15)
  @test isapprox(stats[3], 0, atol=2e-14)
  IV_ig = [Black.impliedVolatilityIG(true, C_reali, S0 * exp((r - q) * taui), Ki, taui, exp(-r * taui)) for (C_reali, Ki, taui, sigmai) in zip(C_real, K, tau, sigma)]
  t7_a = abs.(IV_ig - sigma)
  stats = [mean(t7_a), std(t7_a), maximum(t7_a), minimum(t7_a)]
  @test isapprox(stats[1], 0, atol=1e-15)
  @test isapprox(stats[3], 0, atol=3e-14)
  #jaeck 35.4 ms hou 23.4
end
@testset "ThiopheneIV" begin
  forward = 100.0; tte = 1.0; df = 0.97
  #reference values from the Java implementation (mypapers/iv3/thiophene-iv)
  @test isapprox(Black.impliedVolatilityThiophene(true, 1.9825882293463228, forward, 99.9, tte, df, plus=true), 0.04999999999999997, rtol=1e-15)
  @test isapprox(Black.impliedVolatilityThiophene(false, 0.03869744810591092, forward, 100.0000001, tte, df, plus=true), 9.999999999996438e-4, rtol=1e-15)
  @test isapprox(Black.impliedVolatilityThiophene(true, 7.632928120005742e-27, forward, 100.1, tte, df, plus=true), 1e-4, rtol=1e-15)
  maxErr = 0.0; maxErrPlus = 0.0
  for strike in (50.0, 80.0, 99.9, 100.0, 100.1, 120.0, 200.0), vol in (0.01, 0.05, 0.2, 0.5, 1.0)
    isCall = strike >= forward #out-of-the-money prices are well conditioned
    price = Black.blackScholesFormula(isCall, strike, forward, vol^2 * tte, 1.0, df)
    price < 1e-12 * forward && continue
    maxErr = max(maxErr, abs(Black.impliedVolatilityThiophene(isCall, price, forward, strike, tte, df) / vol - 1))
    maxErrPlus = max(maxErrPlus, abs(Black.impliedVolatilityThiophene(isCall, price, forward, strike, tte, df, plus=true) / vol - 1))
  end
  println("ThiopheneIV max rel error ", maxErr, " ThiopheneIV+ ", maxErrPlus)
  @test maxErr < 1e-11
  @test maxErrPlus < 1e-13
  @test_throws DomainError Black.impliedVolatilityThiophene(true, forward * df, forward, 100.0, tte, df)
end
