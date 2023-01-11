using AQFED, Test, AQFED.FDM
using AQFED.TermStructure

@testset "SmallGridSolvers" begin
r = 0.1
σ = 0.2
q = 0.0
tte = 0.25
strike = 110.0; spot=111.0
M=301; N=20;Smax=300.0
dividends = Vector{CapitalizedDividend{Float64}}()
payoff = AQFED.FDM.VanillaAmerican(false,strike,tte)
refPrice = 2.964544
price = @time AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp((r-q) * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=10001,Smax=Smax,solverName="TDMAPolicyIteration")(spot)
@test isapprox(refPrice, price, atol=1e-6)
price = @time AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp((r-q) * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=N,Smax=Smax,solverName="SOR")(spot)
@test isapprox(refPrice, price, atol=1e-6)
refPrice = price     
solverNames = ["TDMA","TDMAPolicyIteration","DoubleSweep","LUUL","BrennanSchwartz"]
    for solverName in solverNames
    price = @time AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp((r-q) * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=N,Smax=Smax,solverName=solverName)(spot)
    println(solverName," ",price," ",price-refPrice)
    end

payoffB = AQFED.FDM.ButterflyAmerican(false,100.0,strike,tte)
price = @time AQFED.FDM.priceTRBDF2(payoffB, spot, spot * exp((r-q) * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=N,Smax=Smax,solverName="SOR")(spot)
@test isapprox(refPrice, price, atol=1e-6)
refPrice = price     
solverNames = ["TDMA","TDMAPolicyIteration","DoubleSweep","LUUL","BrennanSchwartz"]
    for solverName in solverNames
    price = @time AQFED.FDM.priceTRBDF2(payoffB, spot, spot * exp((r-q) * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=N,Smax=Smax,solverName=solverName)(spot)
    println(solverName," ",price," ",price-refPrice)
    end
    σ=1.0;#spot=110.0;payoffB = AQFED.FDM.ButterflyAmerican(false,90.0,110.0,tte)
    Ns=[2,5,10,20,100]
    for N in Ns
        refPrice = AQFED.FDM.priceTRBDF2(payoffB, spot, spot * exp((r-q) * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=N,Smax=Smax,solverName="SOR")(spot)
        for solverName in solverNames
            price = AQFED.FDM.priceTRBDF2(payoffB, spot, spot * exp((r-q) * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=N,Smax=Smax,solverName=solverName)(spot)
            println(N," ",solverName," ",price," ",price-refPrice)
        end
    end    
#Convergence
strike = 100.0; spot=100.0;r=0.05;Smax=500.0;M=501;tte=1.0
payoff = AQFED.FDM.VanillaAmerican(false,strike,tte)
N = 640*16
refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp(r * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=N,solverName="TDMAPolicyIteration",Smax=Smax)(spot)

Ns=[20,40,80,160,320,640]
previousError = 0.0
for N in Ns
    price = AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp(r * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=N,solverName="TDMAPolicyIteration",Smax=Smax)(spot)
    err = price - refPrice
    println(N, " Policy Iteration ",price," ",err," ", previousError/err)
    previousError = err
end

end

       
@testset "AmericanPutFixedGrid" begin
    #policy iteration as fast as double sweep when r positive?
    r = 0.05
    σ = 0.2
    q = 0.0
    tte = 1.0
    strike = 100.0; spot=100.0
    M=501; N=100;Smax=500.0
    dividends = Vector{CapitalizedDividend{Float64}}()
    payoff = AQFED.FDM.VanillaAmerican(false,strike,tte)
     
    #Convergence
    N = 640*16
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp(r * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=N,solverName="TDMAPolicyIteration",Smax=Smax)(spot)
    N = 100
    payoff = AQFED.FDM.VanillaAmerican(false,strike,tte)
    solverNames = ["TDMA","TDMAPolicyIteration","DoubleSweep"]
    for solverName in solverNames
    price = @time AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp(r * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=N,Smax=Smax,solverName=solverName)(spot)
    println(solverName," ",price," ",price-refPrice)
    end
    Ns=[20,40,80,160,320,640]
    for solverName in solverNames
    previousError = 0.0
    for N in Ns
        price = AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp(r * tte), σ^2 * tte, exp(-r * tte), dividends,M=M,N=N,solverName=solverName,Smax=Smax)(spot)
        err = price - refPrice
        println(N, " ",solverName," ",price," ",err," ", previousError/err)
        previousError = err
    end
end
 
    end
    
    @testset "ExactForward" begin
        #policy iteration as fast as double sweep when r positive?
        r = -0.12
        σ = 0.1
        q = -0.04+r
        tte = 360.0/365
        spot=100.0
        M=101;
        varianceSurface = FlatSurface(σ)
        discountCurve = ConstantRateCurve(r)
        driftCurve = ConstantRateCurve(r-q)
      
        dividends = Vector{CapitalizedDividend{Float64}}()
        payoff = AQFED.FDM.VanillaEuropean(true,0.0,tte)
         Smax = spot*exp( 2*σ*sqrt(tte))
         price = AQFED.FDM.priceTRBDF2(AQFED.FDM.DiscountBond(tte), spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=10,Smax=Smax,calibration=ForwardCalibration())(spot)
         bondPrice = df(discountCurve,tte)
         @test isapprox(bondPrice, price, atol=1e-11)
         price = AQFED.FDM.priceLogTRBDF2(AQFED.FDM.DiscountBond(tte), spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=10,Smax=Smax,calibration=ForwardCalibration())(spot)
         @test isapprox(bondPrice, price, atol=1e-11)
         
         price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=10,Smax=Smax,calibration=ForwardCalibration())(spot)
        forwardPrice = spot*df(discountCurve,tte)/df(driftCurve,tte)
        @test isapprox(forwardPrice, price, atol=1e-10)
        price = AQFED.FDM.priceLogTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=10,Smax=Smax,calibration=ForwardCalibration())(spot)
        @test isapprox(forwardPrice, price, atol=1e-10)   
    end

    @testset "GridShift" begin
        
        r=0.0488
        tte = 0.5833
        spot = 40.0
        σ = 0.4
        q= 0.0
        varianceSurface = FlatSurface(σ)
        discountCurve = ConstantRateCurve(r)
        driftCurve = ConstantRateCurve(r-q)
        dividends = Vector{CapitalizedDividend{Float64}}()
        strikes = [25.0,45.0]
        for strike in strikes
        payoff = AQFED.FDM.VanillaEuropean(false,strike,tte)
        blackPrice = AQFED.Black.blackScholesFormula(false, strike, spot, σ^2*tte, df(driftCurve,tte), df(discountCurve,tte))

        Smax = spot*exp( 3*σ*sqrt(tte))
        M=201; N=51;
        price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,
        calibration=ForwardCalibration(),grid="Shift")(spot)
        println(strike, " UniformZero Calibrated ",price, " ",price-blackPrice)
        price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,calibration=NoCalibration(),grid="Shift")(spot)
        println(strike, " UniformZero Uncalibrated ",price, " ",price-blackPrice)
        price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,Smin=-1, calibration=ForwardCalibration(),grid="Shift")(spot)
        println(strike, " Uniform Calibrated ",price, " ",price-blackPrice)
        price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,Smin=-1,calibration=NoCalibration(),grid="Shift")(spot)
        println(strike, " Uniform Uncalibrated ",price, " ",price-blackPrice)
        price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,Smin=-1,calibration=NoCalibration(),grid="Cubic")(spot)
        println(strike, " Cubic Uncalibrated ",price, " ",price-blackPrice)
        price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,Smin=-1,calibration=NoCalibration(),grid="LogShift")(spot)
        println(strike, " LogUniform Uncalibrated ",price, " ",price-blackPrice)
        price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,Smin=-1,calibration=ForwardCalibration(),grid="LogShift")(spot)
        println(strike, " LogUniform Calibrated ",price, " ",price-blackPrice)

        price = AQFED.FDM.priceLogTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,calibration=NoCalibration(),grid="Shift")(spot)
        println(strike, " LogSchemeUniform Uncalibrated ",price, " ",price-blackPrice)
        price = AQFED.FDM.priceLogTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,calibration=ForwardCalibration(),grid="Shift")(spot)
        println(strike, " LogSchemeUniform Calibrated ",price, " ",price-blackPrice)
        price = AQFED.FDM.priceLogTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,Smin=-1,calibration=NoCalibration(),grid="Cubic")(spot)
        println(strike, " LogSchemeCubic Uncalibrated ",price, " ",price-blackPrice)
        end
    end
        
    end
    @testset "AmericanPutNegativeRate" begin
        #policy iteration as fast as double sweep when r positive?
        r = -0.012
        σ = 0.1
        q = -0.004+r
        tte = 360.0/365
        strike = 100.0; spot=100.0
        M=1001;
        varianceSurface = FlatSurface(σ)
        discountCurve = ConstantRateCurve(r)
        driftCurve = ConstantRateCurve(r-q)
      
        dividends = Vector{CapitalizedDividend{Float64}}()
        payoff = AQFED.FDM.VanillaAmerican(false,strike,tte)
        # Smax = spot*exp(abs((r-q)*tte+σ^2*tte/2) + 4*σ*sqrt(tte))
       Smax = spot*exp( 3*σ*sqrt(tte))
        #Convergence
        refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=4001,N=4001, 
        solverName="TDMAPolicyIteration",Smax=Smax)(spot)
        N = 640*16
        refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N, 
        solverName="TDMAPolicyIteration",Smax=Smax)(spot)
        @test isapprox(3.830520425, refPrice, atol=1e-4)
        N = 100
        price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,solverName="LUUL",Smax=Smax)(spot)
        @test isapprox(refPrice, price, atol=1e-4)

        N = 100
        payoff = AQFED.FDM.VanillaAmerican(false,strike,tte)
        solverNames = ["TDMA","BrennanSchwartz","TDMAPolicyIteration","DoubleSweep","LUUL","PSOR"]
        for solverName in solverNames
        price = @time AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,solverName=solverName,Smax=Smax)(spot)
        println(solverName," ",price," ",price-refPrice)
        end
        
        Ns=[20,40,80,160,320]
        for solverName in solverNames
        previousError = 0.0
        for N in Ns
            price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,solverName=solverName)(spot)
            err = price - refPrice
            println(N, " ",solverName," ",price," ",err," ", previousError/err)
            previousError = err
        end
    end
     
        end
        