using AQFED, Test, AQFED.FDM, AQFED.TermStructure
using DataFrames

@testset "AmericanPutGamma" begin
    strike = 110.0
	spot = 100.0
	r = 0.01
	q = 0.0
	vol = 0.1
	tte = 0.5
	xSteps = 1000 
	tSteps = 100  
    min = 80.890436
	max = 123.593012
    varianceSurface = FlatSurface(vol)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    spot = 100.0
    dividends = Vector{CapitalizedDividend{Float64}}()
    payoffV = VanillaAmerican(false, strike, tte)
    payoffKV = KreissSmoothDefinition(payoffV)
    grid = UniformGrid(false)
    model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
    priceTRBDF2 = AQFED.FDM.priceTRBDF2(payoffKV, spot,model, dividends, M=xSteps, N=tSteps, Smin=min, Smax=max, grid=grid,varianceConditioner=PartialExponentialFittingConditioner())
    priceCN = AQFED.FDM.priceTRBDF2(payoffKV, spot,model, dividends, M=xSteps, N=tSteps, Smin=min, Smax=max, grid=grid,timeSteppingName="CN",varianceConditioner=PartialExponentialFittingConditioner())
    priceRAN = AQFED.FDM.priceTRBDF2(payoffKV, spot,model, dividends, M=xSteps, N=tSteps, Smin=min, Smax=max, grid=grid,timeSteppingName="Rannacher4",varianceConditioner=PartialExponentialFittingConditioner())
    priceRKL = AQFED.FDM.priceRKL2(payoffKV, spot,model, dividends, M=xSteps, N=tSteps, Smin=min, Smax=max, grid=grid,varianceConditioner=PartialExponentialFittingConditioner())
    priceRKG = AQFED.FDM.priceRKG2(payoffKV, spot,model, dividends, M=xSteps, N=tSteps, Smin=min, Smax=max, grid=grid,varianceConditioner=PartialExponentialFittingConditioner())
    #=
    p1 = plot(priceCN.x,PPInterpolation.evaluateSecondDerivative.(priceCN,priceCN.x),ylims=(-0.005,0.075),yticks=[0,0.025,0.05,0.075],label="Crank-Nicolson")
    p2 = plot(priceRAN.x,PPInterpolation.evaluateSecondDerivative.(priceRAN,priceRAN.x),ylims=(-0.005,0.075),yticks=([0,0.025,0.05,0.075],["","","",""]),label="Rannacher")
    p3 = plot(priceTRBDF2.x,PPInterpolation.evaluateSecondDerivative.(priceTRBDF2,priceTRBDF2.x),ylims=(-0.005,0.075),yticks=([0,0.025,0.05,0.075],["","","",""]),label="TR-BDF2")
    p4 = plot(priceRKL.x,PPInterpolation.evaluateSecondDerivative.(priceRKL,priceRKL.x),ylims=(-0.005,0.075),yticks=([0,0.025,0.05,0.075],["","","",""]),label="RKL")
    p5 = plot(priceRKL.x,PPInterpolation.evaluateSecondDerivative.(priceRKG,priceRKG.x),ylims=(-0.005,0.075),yticks=([0,0.025,0.05,0.075],["","","",""]),label="RKG")
     plot(p1,p2,p3,p4,p5,layout=(1,5),size=(1024,200),margins=1Plots.mm)

    =#

end
@testset "SmallGridSolvers" begin
    r = 0.1
    σ = 0.2
    q = 0.0
    tte = 0.25
    strike = 110.0
    spot = 111.0
    M = 301
    N = 20
    Smax = 300.0
    dividends = Vector{CapitalizedDividend{Float64}}()
    payoff = AQFED.FDM.VanillaAmerican(false, strike, tte)
    refPrice = 2.964544
    price = @time AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp((r - q) * tte), σ^2 * tte, exp(-r * tte), dividends, M=M, N=10001, Smax=Smax, solverName="TDMAPolicyIteration")(spot)
    @test isapprox(refPrice, price, atol=1e-6)
    price = @time AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp((r - q) * tte), σ^2 * tte, exp(-r * tte), dividends, M=M, N=N, Smax=Smax, solverName="SOR")(spot)
    @test isapprox(refPrice, price, atol=1e-6)
    refPrice = price
    solverNames = ["TDMA", "TDMAPolicyIteration", "DoubleSweep", "LUUL", "BrennanSchwartz"]
    for solverName in solverNames
        price = @time AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp((r - q) * tte), σ^2 * tte, exp(-r * tte), dividends, M=M, N=N, Smax=Smax, solverName=solverName)(spot)
        println(solverName, " ", price, " ", price - refPrice)
    end

    payoffB = AQFED.FDM.ButterflyAmerican(false, 100.0, strike, tte)
    price = @time AQFED.FDM.priceTRBDF2(payoffB, spot, spot * exp((r - q) * tte), σ^2 * tte, exp(-r * tte), dividends, M=M, N=N, Smax=Smax, solverName="SOR")(spot)
    @test isapprox(refPrice, price, atol=1e-6)
    refPrice = price
    solverNames = ["TDMA", "TDMAPolicyIteration", "DoubleSweep", "LUUL", "BrennanSchwartz"]
    for solverName in solverNames
        price = @time AQFED.FDM.priceTRBDF2(payoffB, spot, spot * exp((r - q) * tte), σ^2 * tte, exp(-r * tte), dividends, M=M, N=N, Smax=Smax, solverName=solverName)(spot)
        println(solverName, " ", price, " ", price - refPrice)
    end
    σ = 1.0#spot=110.0;payoffB = AQFED.FDM.ButterflyAmerican(false,90.0,110.0,tte)
    Ns = [2, 5, 10, 20, 100]
    for N in Ns
        refPrice = AQFED.FDM.priceTRBDF2(payoffB, spot, spot * exp((r - q) * tte), σ^2 * tte, exp(-r * tte), dividends, M=M, N=N, Smax=Smax, solverName="SOR")(spot)
        for solverName in solverNames
            price = AQFED.FDM.priceTRBDF2(payoffB, spot, spot * exp((r - q) * tte), σ^2 * tte, exp(-r * tte), dividends, M=M, N=N, Smax=Smax, solverName=solverName)(spot)
            println(N, " ", solverName, " ", price, " ", price - refPrice)
        end
    end
    #Convergence
    strike = 100.0
    spot = 100.0
    r = 0.05
    Smax = 500.0
    M = 501
    tte = 1.0
    payoff = AQFED.FDM.VanillaAmerican(false, strike, tte)
    N = 640 * 16
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp(r * tte), σ^2 * tte, exp(-r * tte), dividends, M=M, N=N, solverName="TDMAPolicyIteration", Smax=Smax)(spot)

    Ns = [20, 40, 80, 160, 320, 640]
    previousError = 0.0
    for N in Ns
        price = AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp(r * tte), σ^2 * tte, exp(-r * tte), dividends, M=M, N=N, solverName="TDMAPolicyIteration", Smax=Smax)(spot)
        err = price - refPrice
        println(N, " Policy Iteration ", price, " ", err, " ", previousError / err)
        previousError = err
    end

end

@testset "DiscretKOTavellaRandall" begin
    #paper is reproduced
    #TODO, check thta ppinterpolation is fast enough (seems slow)
    #TODO, add Kreiss smoothing
    #NOTE: using small Smin but non zero (for example 1.0, 2.0) breaks stability! is this a problem in log? likely not! log may be more stable in general.
    tte = 1.0
    σ = 0.2
    t1 = 1.0 / 250
    r = 0.07
    q = 0.02
    isCall = true
    strike = 100.0
    level = 125.0
    isDown = false
    min = 0.0
    max = 150.0 #3 dev = 200.0 !
    obsTimes = collect(range(t1, stop=tte, length=250))
    N = 1501
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    spot = 100.0
    dividends = Vector{CapitalizedDividend{Float64}}()
    payoffV = VanillaEuropean(true, strike, tte)
    payoffKV = KreissSmoothDefinition(payoffV)
    payoff = AQFED.FDM.DiscreteKO(payoffV, level, isDown, 0.0, obsTimes)
    payoffK = KreissSmoothDefinition(payoff)
    refPriceV = AQFED.FDM.priceTRBDF2(payoffV, spot, driftCurve, varianceSurface, discountCurve, dividends, M=16001, N=N, Smin=min, Smax=max, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=16001, N=N, Smin=min, Smax=max, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    println("reference KO ", refPrice(spot), " vanilla ", refPriceV(spot))
    α = 2.0 / (max - min)
    Ms = [251, 501, 1001, 2001]
    grids = [UniformGrid(false), ShiftedGrid(UniformGrid(false)), SmoothlyDeformedGrid(UniformGrid(false)), CubicGrid(α), SmoothlyDeformedGrid(CubicGrid(α)), SinhGrid(α), SmoothlyDeformedGrid(SinhGrid(α))]
    for M in Ms
        for grid in grids
            priceV = AQFED.FDM.priceTRBDF2(payoffV, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=min, Smax=max, grid=grid)(spot)
            priceKV = AQFED.FDM.priceTRBDF2(payoffKV, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=min, Smax=max, grid=grid)(spot)

            price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=min, Smax=max, grid=grid)(spot)
            priceK = AQFED.FDM.priceTRBDF2(payoffK, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=min, Smax=max, grid=grid)(spot)
            println(M, " ", grid, " Vanilla ", priceV, " ", priceV - refPriceV(spot))
            println(M, " ", grid, " Vanilla ", priceKV, " ", priceKV - refPriceV(spot))
            println(M, " ", grid, " KO ", price, " ", price - refPrice(spot))
            println(M, " ", grid, " KO ", priceK, " ", priceK - refPrice(spot))
        end
    end
    grids = [UniformGrid(false), SmoothlyDeformedGrid(UniformGrid(false)), LogGrid(false), SmoothlyDeformedGrid(LogGrid(false)), CubicGrid(α), SmoothlyDeformedGrid(CubicGrid(α)), SinhGrid(α), SmoothlyDeformedGrid(SinhGrid(α))]
    names = ["Uniform", "Uniform-Smooth", "Cubic", "Cubic-"]
    data = DataFrame(m=Int[], grid=String[], price=Float64[], error=Float64[])
    #Error as a function of m
    for M = 250:10:501
        for grid in grids
            localMin = min
            if typeof(grid) <: LogGrid || typeof(grid) <: SmoothlyDeformedGrid{LogGrid}
                localMin = 50.0
                #localMax = 180.0
            end
            priceV = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=localMin, Smax=max, grid=grid)(spot)
            keyname = if typeof(grid) <: UniformGrid
                "Uniform"
            elseif typeof(grid) <: SmoothlyDeformedGrid{UniformGrid}
                "Uniform-Deformed"
            elseif typeof(grid) <: SmoothlyDeformedGrid{CubicGrid}
                "Cubic-Deformed"
            elseif typeof(grid) <: SmoothlyDeformedGrid{SinhGrid}
                "Sinh-Deformed"
            elseif typeof(grid) <: CubicGrid
                "Cubic"
            elseif typeof(grid) <: SinhGrid
                "Sinh"
            elseif typeof(grid) <: LogGrid
                "Log"
            elseif typeof(grid) <: SmoothlyDeformedGrid{LogGrid}
                "Log-Deformed"
            end
            push!(data, [M, keyname, priceV, priceV - refPrice(spot)])
            if !(typeof(grid) <: SmoothlyDeformedGrid)
                priceKV = AQFED.FDM.priceTRBDF2(payoffK, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=localMin, Smax=max, grid=grid)(spot)
                push!(data, [M, string(keyname, "-", "Kreiss"), priceKV, priceKV - refPrice(spot)])
            end
        end
    end
    # udata = data[startswith.(data.grid,"Uniform"),:]
    # plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:log10,ylabel="Absolute error in price", xlabel="number of space-steps",size=(400,300))
    # savefig("/home/fabien/mypapers/eqd_book/uniform_kreiss_time_vanilla.pdf")
    # udata = data[startswith.(data.grid,"Cubic"),:]
    # plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:log10,ylabel="Absolute error in price", xlabel="number of space-steps",size=(400,300))
    # savefig("/home/fabien/mypapers/eqd_book/cubic_kreiss_time_vanilla.pdf")
    #plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:ln,yticks=([0.0001,0.0002,0.0004],[0.0001,0.0002,0.0004]),ylabel="Absolute error in price", xlabel="number of time-steps",mode="markers")


    #American
    #max = exp((r-q)*tte+3*σ*sqrt(tte))*spot
    max = 150.0
    α = 2.0 / (max - min)
    grids = [UniformGrid(false), SmoothlyDeformedGrid(UniformGrid(false)), CubicGrid(α), SmoothlyDeformedGrid(CubicGrid(α)), SinhGrid(α), SmoothlyDeformedGrid(SinhGrid(α))]
    payoffVA = VanillaAmerican(false, strike, tte)
    refPriceV = AQFED.FDM.priceTRBDF2(payoffVA, spot, driftCurve, varianceSurface, discountCurve, dividends, M=16001, N=N, Smin=min, Smax=max, grid=SmoothlyDeformedGrid(UniformGrid(false)), solverName="LUUL")
    payoffKVA = KreissSmoothDefinition(payoffVA)
    data = DataFrame(m=Int[], grid=String[], price=Float64[], error=Float64[])
    #Error as a function of m
    for M = 250:501
        for grid in grids
            priceV = AQFED.FDM.priceTRBDF2(payoffVA, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=min, Smax=max, grid=grid, solverName="LUUL")(spot)
            keyname = if typeof(grid) <: UniformGrid
                "Uniform"
            elseif typeof(grid) <: SmoothlyDeformedGrid{UniformGrid}
                "Uniform-Deformed"
            elseif typeof(grid) <: SmoothlyDeformedGrid{CubicGrid}
                "Cubic-Deformed"
            elseif typeof(grid) <: SmoothlyDeformedGrid{SinhGrid}
                "Sinh-Deformed"
            elseif typeof(grid) <: CubicGrid
                "Cubic"
            elseif typeof(grid) <: SinhGrid
                "Sinh"
            elseif typeof(grid) <: LogGrid
                "Log"
            elseif typeof(grid) <: SmoothlyDeformedGrid{LogGrid}
                "Log-Deformed"
            end
            push!(data, [M, keyname, priceV, priceV - refPriceV(spot)])
            if !(typeof(grid) <: SmoothlyDeformedGrid)
                priceKV = AQFED.FDM.priceTRBDF2(payoffKVA, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=min, Smax=max, grid=grid, solverName="LUUL")(spot)
                push!(data, [M, string(keyname, "-", "Kreiss"), priceKV, priceKV - refPriceV(spot)])
            end
        end
    end
    #Error as a function of spot - nothing to see, similar conclusions as single spot.
    M = 251
    M=1001; N=101; σ=0.1;r=0.01;q=0.0;tte=0.5; strike=110.0;max = exp((r-q)*tte+3*σ*sqrt(tte))*spot;payoffVA = VanillaAmerican(false, strike, tte)
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    
    data = DataFrame(S=Float64[], grid=String[], price=Float64[], error=Float64[], delta=Float64[], deltaError=Float64[], gamma=Float64[], gammaError=Float64[])
    #Error as a function of m 
    for grid in grids
        localMin = min
        if typeof(grid) <: LogGrid || typeof(grid) <: SmoothlyDeformedGrid{LogGrid}
            localMin = 50.0
        end

        priceV = AQFED.FDM.priceTRBDF2(payoffVA, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=localMin, Smax=max, grid=grid, solverName="LUUL")
        keyname = if typeof(grid) <: UniformGrid
            "Uniform"
        elseif typeof(grid) <: SmoothlyDeformedGrid{UniformGrid}
            "Uniform-Deformed"
        elseif typeof(grid) <: SmoothlyDeformedGrid{CubicGrid}
            "Cubic-Deformed"
        elseif typeof(grid) <: SmoothlyDeformedGrid{SinhGrid}
            "Sinh-Deformed"
        elseif typeof(grid) <: CubicGrid
            "Cubic"
        elseif typeof(grid) <: SinhGrid
            "Sinh"
        elseif typeof(grid) <: LogGrid
            "Log"
        elseif typeof(grid) <: SmoothlyDeformedGrid{LogGrid}
            "Log-Deformed"
        end
        priceKV = AQFED.FDM.priceTRBDF2(payoffKVA, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=localMin, Smax=max, grid=grid, solverName="LUUL")
        for s = range(80.0, stop=120.0, length=100)
            push!(data, [s, keyname, priceV(s), priceV(s) - refPriceV(s), evaluateDerivative(priceV, s), evaluateDerivative(priceV, s) - evaluateDerivative(refPriceV, s), evaluateSecondDerivative(priceV, s), evaluateSecondDerivative(priceV, s) - evaluateSecondDerivative(refPriceV, s)])
            if !(typeof(grid) <: SmoothlyDeformedGrid)

                push!(data, [s, string(keyname, "-", "Kreiss"), priceKV(s), priceKV(s) - refPriceV(s), evaluateDerivative(priceKV, s), evaluateDerivative(priceKV, s) - evaluateDerivative(refPriceV, s), evaluateSecondDerivative(priceKV, s), evaluateSecondDerivative(priceKV, s) - evaluateSecondDerivative(refPriceV, s)])
            end
        end
    end

    #Error in delta
end
@testset "AmericanPutFixedGrid" begin
    #policy iteration as fast as double sweep when r positive?
    r = 0.05
    σ = 0.2
    q = 0.0
    tte = 1.0
    strike = 100.0
    spot = 100.0
    M = 501
    N = 100
    Smax = 500.0
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    dividends = Vector{CapitalizedDividend{Float64}}()
    payoff = AQFED.FDM.VanillaAmerican(false, strike, tte)

    #Convergence
    N = 640 * 16
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, solverName="TDMAPolicyIteration", Smax=Smax)(spot)
    N = 100
    payoff = AQFED.FDM.VanillaAmerican(false, strike, tte)
    solverNames = ["TDMA", "TDMAPolicyIteration", "DoubleSweep"]
    for solverName in solverNames
        price = @time AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp(r * tte), σ^2 * tte, exp(-r * tte), dividends, M=M, N=N, Smax=Smax, solverName=solverName)(spot)
        println(solverName, " ", price, " ", price - refPrice)
    end
    Ns = [20, 40, 80, 160, 320, 640]
    for solverName in solverNames
        previousError = 0.0
        for N in Ns
            price = AQFED.FDM.priceTRBDF2(payoff, spot, spot * exp(r * tte), σ^2 * tte, exp(-r * tte), dividends, M=M, N=N, solverName=solverName, Smax=Smax)(spot)
            err = price - refPrice
            println(N, " ", solverName, " ", price, " ", err, " ", previousError / err)
            previousError = err
        end
    end

end

@testset "ExerciseBoundary" begin
    tte = 1.0
    strike = 100.0
    spot = 100.0
    eti = [0, 1, 2, 3, 35, 64, 93, 184, 275, 367, 735, 1099] ./ 365.0
    edfi = [1.0, 1.00000232, 0.99998979, 0.9999798, 0.99937204, 0.99951607, 0.99782219, 0.99406162, 0.98825549, 0.98231822, 0.96108459, 0.94354772]
    eurpp = CubicSplineNatural(eti, log.(edfi))
    #TRY discount
    ti = [0, 1, 8, 15, 22, 31, 63, 92, 184, 276, 367, 552, 734, 1098] ./ 365.0
    dfi = [1.0, 0.9997808, 0.99743484, 0.99403111, 0.98925055, 0.98313541, 0.96031598, 0.92064074, 0.83544403, 0.75603321, 0.68572107, 0.56887972, 0.48014397, 0.35315213]
    trypp = CubicSplineNatural(ti, log.(dfi))
    tryeurdf = trypp.(ti) .- eurpp.(ti)
    tryeurpp = CubicSplineNatural(ti, tryeurdf)
    eurtrydf = .- tryeurdf
    eurtrypp = CubicSplineNatural(ti, eurtrydf)
    
    discountCurve = AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(eurpp)
    driftCurve = AQFED.TermStructure.ConstantRateCurve(0.01)
    # discountCurve = AQFED.TermStructure.ConstantRateCurve(0.04)
    nodividends = Vector{CapitalizedDividend{Float64}}()
    payoff = AQFED.FDM.VanillaAmerican(false, strike, tte)
    payoffB = AQFED.FDM.VanillaAmericanWithExerciseBoundary(payoff)
    #first test - discount=EUR, drift=1% = r-q
    σ=0.3
    ms = [3,5,7,9,11,13,15,17,19,21,31]
    alPrices = Vector{Float64}()
    alppPrices = Vector{Float64}()
    σs = [0.1,0.2,0.3]
    for σ in σs
    varianceSurface = FlatSurface(σ)
    model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, nodividends, M=16001, N=2001, grid=SmoothlyDeformedGrid(CubicGrid(0.2)), solverName="LUUL", ndev=4)(spot) 
    for m in ms
        al = AQFED.American.AndersenLakeRepresentation(model, tte, 1e-8, m, 32, 61, 121, isCall=false)
        alPrice = AQFED.American.priceAmerican(al, strike, spot)
        append!(alPrices,alPrice)
        println(σ," ", m," AL ",alPrice," ",alPrice-refPrice)
        al = AQFED.American.AndersenLakePPRepresentation(model, tte,false, atol=1e-8, nC=5, nIter=32, nTS1=21, nTS2=41,nPP=m)
        alPrice = AQFED.American.priceAmerican(al, strike, spot)
        append!(alppPrices,alPrice)
        println(σ," ", m," AL-PP3 ",alPrice," ",alPrice-refPrice)
        
    end
    #= plot(ms,abs.(alPrices .- refPrice), label="Andersen-Lake nₚₚ=1", xlabel="m", ylabel="Absolute error in price", yscale= :log10)
     plot!(ms,abs.(alppPrices .- refPrice), label="Andersen-Lake nₚₚ=m", xlabel="m", ylabel="Absolute error in price", yscale= :log10)
plot!(size=(300,200))

     savefig("/home/fabien/mypapers/eqd_book/alpp_eur_error.pdf")
     savefig("/home/fabien/mypapers/eqd_book/alpp_try_error_vol20.pdf")

     =#

end
# => good convergence

#0th
discountCurve = AQFED.TermStructure.ConstantRateCurve(0.20)
driftCurve = AQFED.TermStructure.ConstantRateCurve(0.01)
 
#Second test - discount=EUR, drift=-1%
discountCurve = AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(eurpp)
driftCurve = AQFED.TermStructure.ConstantRateCurve(-0.01)

# = erratic, esp for sigma<=0.2 , PP3 to PP5?

#test2.1 - discount=EUR, drift=-1%+EUR #= erratic, even for sigma=0.3. 
eurspreaddf = eurpp.(ti) + 0.01 .* ti
    eurspreadpp = CubicSplineNatural(ti, eurspreaddf)
    discountCurve = AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(eurpp)
    driftCurve = AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(eurspreadpp)
m=9; σ=0.3; varianceSurface = FlatSurface(σ)
model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, nodividends, M=16001, N=2001, grid=SmoothlyDeformedGrid(CubicGrid(0.2)), solverName="LUUL", ndev=4)(spot) 
AQFED.FDM.priceTRBDF2(payoffB, spot,model, nodividends, M=8001, N=4001, grid=SmoothlyDeformedGrid(CubicGrid(0.3)), solverName="LUUL")(spot)
al = AQFED.American.AndersenLakeRepresentation(model, tte, 1e-8, m, 32, 61, 121, isCall=false)
alPrice = AQFED.American.priceAmerican(al, strike, spot)
alpp = AQFED.American.AndersenLakePPRepresentation(model, tte,false, atol=1e-8, nC=5, nIter=32, nTS1=21, nTS2=41,nPP=m)
alppPrice = AQFED.American.priceAmerican(alpp, strike, spot)
alg1 = AQFED.American.AndersenLakeGRepresentation(model, tte, false,nIter=32,collocation=AQFED.American.ChebyshevCollocation(9,tte),quadrature=AQFED.American.TanhSinhQuadrature(51))
alg5 = AQFED.American.AndersenLakeGRepresentation(model, tte, false,nIter=32,collocation=AQFED.American.ChebyshevPPCollocation([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,1.0],5,isDiscontinuous=false),quadrature=AQFED.American.TanhSinhPPQuadrature([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,1.0],51))
#=   
using PlotThemes
theme(:vibrant)
plot(payoffB.exerciseTimes, payoffB.exerciseBoundary, label="TR-BDF2", xlab="Time to maturity", ylab="Asset price")
plot!( t,AQFED.American.exerciseBoundary(al, spot, t), label="Andersen-Lake m=9 nₚₚ=1")
plot!( t,AQFED.American.exerciseBoundary(alpp, spot, t), label="Andersen-Lake m=5 nₚₚ=9")
plot!(size=(400,300))
plot( t,AQFED.American.exerciseBoundary(alg1, spot, t), label="Andersen-Lake m=9 nₚₚ=1")
plot!(payoffB.exerciseTimes, payoffB.exerciseBoundary, label="TR-BDF2", xlab="Time", ylab="Asset price", size=(450,300))
plot!( t,AQFED.American.exerciseBoundary(alg5, spot, t), label="Andersen-Lake m=5 nₚₚ=9",linestyle=:dot)
savefig("/home/fabien/mypapers/eqd_book/alpp_eur_spread.pdf")

savefig("/home/fabien/mypapers/eqd_book/alpp_eur_spread.pdf")
plot(t, -ForwardDiff.derivative.(eurpp,t) .* 100,xlab="time in ACT/365", ylab="Rate in %",label="")
plot!(size=(400,200))
savefig("/home/fabien/mypapers/eqd_book/eur_forward_rate.pdf")
plot(t, -eurpp.(t) ./t .* 100,xlab="time in ACT/365", ylab="Rate in %",label="")
plot!(size=(400,200))
savefig("/home/fabien/mypapers/eqd_book/eur_rate.pdf")
plot(t, -ForwardDiff.derivative.(trypp,t) .* 100,xlab="time in ACT/365", ylab="Rate in %",label="")
plot!(size=(400,200))
savefig("/home/fabien/mypapers/eqd_book/try_forward_rate.pdf")
plot(t, -trypp.(t) ./t .* 100,xlab="time in ACT/365", ylab="Rate in %",label="")
plot!(size=(400,200))
savefig("/home/fabien/mypapers/eqd_book/try_rate.pdf")

=#   


#test 2.2 - discount=EUR, drift=1%+EUR OKish not as good for sigma=0.1
eurspreadpdf = eurpp.(ti) - 0.01 .* ti
    eurspreadppp = CubicSplineNatural(ti, eurspreadpdf)
    discountCurve = AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(eurpp)
    driftCurve = AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(eurspreadppp)


#Third test - discount=TRY, drift=TRY-EUR  erratic even sigma=0.3
discountCurve = AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(trypp)
driftCurve = AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(tryeurpp)
m=7; σ=0.3; varianceSurface = FlatSurface(σ)
model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, nodividends, M=16001, N=2001, grid=SmoothlyDeformedGrid(CubicGrid(0.2)), solverName="LUUL", ndev=4)(spot) 
AQFED.FDM.priceTRBDF2(payoffB, spot, model, nodividends, M=8001, N=4001, grid=SmoothlyDeformedGrid(CubicGrid(0.3)), solverName="LUUL")(spot)
al = AQFED.American.AndersenLakeRepresentation(model, tte, 1e-8, m, 32, 61, 121, isCall=false)
#al = AQFED.American.AndersenLakePPRepresentation(model, tte,false, atol=1e-8, nC=m, nIter=32, nTS1=21, nTS2=41,nPP=1)
alPrice = AQFED.American.priceAmerican(al, strike, spot)
alpp = AQFED.American.AndersenLakePPRepresentation(model, tte,false, atol=1e-8, nC=5, nIter=32, nTS1=21, nTS2=41,nPP=m)
alppPrice = AQFED.American.priceAmerican(alpp, strike, spot)
alg5 = AQFED.American.AndersenLakeGRepresentation(model, tte, false,nIter=32,collocation=AQFED.American.ChebyshevPPCollocation(cutDates,5,isDiscontinuous=false),quadrature=AQFED.American.TanhSinhPPQuadrature(cutDates,51,isDiscontinuous=true))

#=   
using PlotThemes
theme(:vibrant)
plot(payoffB.exerciseTimes, payoffB.exerciseBoundary, label="TR-BDF2", xlab="Time to maturity", ylab="Asset price")
plot!( t,AQFED.American.exerciseBoundary(al, spot, t), label="Andersen-Lake m=7 nₚₚ=1")
plot!( t,AQFED.American.exerciseBoundary(alpp, spot, t), label="Andersen-Lake m=5 nₚₚ=7")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/alpp_try_eur.pdf")
=#   

#Fourth test - discount=EUR, drift=EUR-TRY #no issue
discountCurve = AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(eurpp)
driftCurve = AQFED.TermStructure.InterpolatedLogDiscountFactorCurve(eurtrypp)

    # plot(t,AQFED.American.exerciseBoundary(al, strike, t))
#fifth test, discrete prop
tte=1.0; spot=100.0; strike=100.0;
payoff = VanillaAmerican(false, strike, tte)
varianceSurface = FlatSurface(0.3)
discountCurve = AQFED.TermStructure.ConstantRateCurve(0.03)
driftCurve = AQFED.TermStructure.ConstantRateCurve(0.01) 
driftCurve = AQFED.TermStructure.ConstantRateCurve(0.03+0.07) 
dividends = [Dividend(0.01,0.2,0.2,true,false),Dividend(0.01,0.45,0.45,true,false),Dividend(0.01,0.70,0.70,true,false),Dividend(0.01,0.95,0.95,true,false)]
divDates = [x.exDate for x in dividends]
cutDates = vcat(0.0,divDates,tte)
#dividends = [Dividend(0.01,0.2,0.2,true,false)]
cdividends = [ CapitalizedDividend(dividend,1.0) for dividend in dividends]
driftCurveDiv = AQFED.TermStructure.DividendDriftCurve(driftCurve, dividends,spot)
model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurveDiv)
modeln = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
refPrice = AQFED.FDM.priceTRBDF2(payoff, spot,modeln, cdividends, M=16001, N=8001, grid=SmoothlyDeformedGrid(CubicGrid(0.2)), solverName="LUUL", ndev=5)(spot) 
refPriceYield = AQFED.FDM.priceTRBDF2(payoff, spot,model, nodividends, M=16001, N=2001, grid=SmoothlyDeformedGrid(CubicGrid(0.2)), solverName="LUUL", ndev=5)(spot) 
alpp = AQFED.American.AndersenLakePPRepresentation(modeln, tte,false, atol=1e-8, nC=5, nIter=16, nTS1=21, nTS2=41,dividends=dividends)
alg = AQFED.American.AndersenLakeGRepresentation(modeln, tte, false,nIter=32,collocation=AQFED.American.ChebyshevPPCollocation(cutDates,15),quadrature=AQFED.American.TanhSinhPPQuadrature(cutDates,51), dividends=dividends)
algPrice = AQFED.American.priceAmerican(alg, strike, spot)
ms = [3,5,7,9,11,13,15,17,19,21,31]
for m in ms
    alpp = AQFED.American.AndersenLakePPRepresentation(modeln, tte,false, atol=1e-8, nC=m, nIter=32, nTS1=51, nTS2=51,dividends=dividends)
    alppPrice = AQFED.American.priceAmerican(alpp, strike, spot)
    alg = AQFED.American.AndersenLakeGRepresentation(modeln, tte, false,nIter=32,collocation=AQFED.American.ChebyshevPPCollocation(cutDates,m),quadrature=AQFED.American.TanhSinhPPQuadrature(cutDates,51), dividends=dividends)
    algPrice = AQFED.American.priceAmerican(alg, strike, spot)
    println(m," ",alppPrice," ",alppPrice-refPrice," ",algPrice," ",algPrice-refPrice)    
end
# plot( t,AQFED.American.exerciseBoundary(alg, spot, t), label="Andersen-Lake m=15")
# plot!(payoffB.exerciseTimes, payoffB.exerciseBoundary, label="TR-BDF2", xlab="Time", ylab="Asset price", size=(450,300))
# savefig("/home/fabien/mypapers/eqd_book/al_prop_nolog.pdf")


# test put call symmetry with dividends
payoffC = AQFED.FDM.VanillaAmerican(true, spot, tte)
#r' = q, r'-q' = q - r
discountCurve = AQFED.TermStructure.ConstantRateCurve(0.02)
dividends = [Dividend(0.01,0.2,0.2,true,false),Dividend(0.01,0.45,0.45,true,false),Dividend(0.01,0.70,0.70,true,false),Dividend(0.01,0.95,0.95,true,false)]
cdividends = [ CapitalizedDividend(dividend,1.0) for dividend in dividends]
driftCurve = AQFED.TermStructure.OppositeCurve(AQFED.TermStructure.DividendDriftCurve(discountCurve, dividends,spot))
discountCurve = AQFED.TermStructure.OppositeCurve(AQFED.TermStructure.DividendDriftCurve(AQFED.TermStructure.ConstantRateCurve(0.0), dividends,spot))
symPrice = AQFED.FDM.priceTRBDF2(payoffC, strike, driftCurve, varianceSurface, discountCurve, nodividends, M=16001, N=2001, grid=SmoothlyDeformedGrid(CubicGrid(0.2)), solverName="LUUL", ndev=4)(spot) 


##test discrete div with different drift
discountCurve = ConstantRateCurve{Float64}(0.03)
driftCurve = ConstantRateCurve{Float64}(0.04)

end

@testset "SingleProp" begin
σ = 0.3
varianceSurface = FlatSurface(σ)
discountCurve = AQFED.TermStructure.ConstantRateCurve(0.06)
tte = 91.0/365
strike = 100.0;spot=100.0
driftCurve = discountCurve
dividends = [Dividend(0.07,40.0/365,40.0/365,true,false)]
divDates = [x.exDate for x in dividends]
cutDates = vcat(0.0,divDates,tte)
cdividends = [ CapitalizedDividend(dividend,1.0) for dividend in dividends]
payoff = AQFED.FDM.VanillaAmerican(false, strike, tte)
payoffB = AQFED.FDM.VanillaAmericanWithExerciseBoundary(payoff)

modeln = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, modeln, cdividends, M=16001, N=8001, grid=SmoothlyDeformedGrid(CubicGrid(0.2)), solverName="LUUL", ndev=5)(spot) 
alg = AQFED.American.AndersenLakeGRepresentation(modeln, tte, false,nIter=32,collocation=AQFED.American.ChebyshevPPCollocation(cutDates,15),quadrature=AQFED.American.TanhSinhPPQuadrature(cutDates,51), dividends=dividends)
algPrice = AQFED.American.priceAmerican(alg, strike, spot)
for nC in [3,5,7,9,15,21,31]
    alpp = AQFED.American.AndersenLakePPRepresentation(modeln, tte,false, atol=1e-8, nC=nC, nIter=16, nTS1=101, nTS2=101,dividends=dividends)
    alppPrice = AQFED.American.priceAmerican(alpp, strike, spot)
    alg = AQFED.American.AndersenLakeGRepresentation(modeln, tte, false,nIter=32,collocation=AQFED.American.ChebyshevPPCollocation(cutDates,nC),quadrature=AQFED.American.TanhSinhPPQuadrature(cutDates,51), dividends=dividends)
    algPrice = AQFED.American.priceAmerican(alg, strike, spot)
    println(nC," ",alppPrice," ",alppPrice-refPrice," ",algPrice," ",algPrice-refPrice)    
end
end

@testset "MultipleVellekoopAmerican" begin
    σ = 0.25
    discountRate = 0.06
    tte = 7.0
    ttd = 0.9
    spot = 100.0;    strike = 100.0;
    dividends = [Dividend(6.0/spot, ttd, ttd, true, false),
        Dividend(6.5/spot, ttd + 1, ttd + 1, true, false),
        Dividend(7.0/spot, ttd + 2, ttd + 2, true, false),
        Dividend(7.5/spot, ttd + 3, ttd + 3, true, false), 
        Dividend(8.0/spot, ttd + 4, ttd + 4, true, false), 
        Dividend(8.0/spot, ttd + 5, ttd + 5, true, false), 
        Dividend(8.0/spot, ttd + 6, ttd + 6, true, false)]
    #dividends = [Dividend(8.0/spot, ttd+6, ttd+6, true, false)]
    divDates = [x.exDate for x in dividends]
    cutDates = vcat(0.0,divDates,tte)
        varianceSurface = FlatSurface(σ)
    discountCurve = AQFED.TermStructure.ConstantRateCurve(discountRate)
    driftCurve = discountCurve
    cdividends = [ CapitalizedDividend(dividend,1.0) for dividend in dividends]
    payoff = AQFED.FDM.VanillaAmerican(false, strike, tte)
    payoffB = AQFED.FDM.VanillaAmericanWithExerciseBoundary(payoff)
    modeln = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot,modeln, cdividends, M=16001, N=4001, grid=SmoothlyDeformedGrid(CubicGrid(0.2)), solverName="LUUL", ndev=5)(spot) 
    alg = AQFED.American.AndersenLakeGRepresentation(modeln, tte, false,nIter=32,collocation=AQFED.American.ChebyshevPPCollocation(cutDates,15),quadrature=AQFED.American.TanhSinhPPQuadrature(cutDates,51), dividends=dividends)
    algPrice = AQFED.American.priceAmerican(alg, strike, spot)
        for nC in [3,5,7,9,15,21,31]
        alpp = AQFED.American.AndersenLakePPRepresentation(modeln, tte,false, atol=1e-8, nC=nC, nIter=16, nTS1=101, nTS2=101,dividends=dividends)
        alppPrice = AQFED.American.priceAmerican(alpp, strike, spot)
        alg = AQFED.American.AndersenLakeGRepresentation(modeln, tte, false,nIter=32,collocation=AQFED.American.ChebyshevPPCollocation(cutDates,nC),quadrature=AQFED.American.TanhSinhPPQuadrature(cutDates,101), dividends=dividends)
        algPrice = AQFED.American.priceAmerican(alg, strike, spot)
        println(nC," ",alppPrice," ",alppPrice-refPrice," ",algPrice," ",algPrice-refPrice)    
    
    end
    end
    
@testset "ExactForward" begin
    #policy iteration as fast as double sweep when r positive?
    r = -0.12
    σ = 0.3
    q = -0.04 + r
    tte = 360.0 / 365
    spot = 100.0
    M = 101
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)

    dividends = Vector{CapitalizedDividend{Float64}}()
    payoff = AQFED.FDM.VanillaEuropean(true, 0.0, tte)
    Smax = spot * exp(2 * σ * sqrt(tte))
    grids = [UniformGrid(false), CubicGrid(0.01), SmoothlyDeformedGrid(CubicGrid(0.01))]
    bondPrice = discountFactor(discountCurve, tte)
    for grid in grids
        price = AQFED.FDM.priceTRBDF2(AQFED.FDM.DiscountBond(tte), spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=10, Smax=Smax, grid=grid, calibration=ForwardCalibration())(spot)
        @test isapprox(bondPrice, price, atol=1e-11)
    end
    price = AQFED.FDM.priceLogTRBDF2(AQFED.FDM.DiscountBond(tte), spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=10, Smax=Smax, calibration=ForwardCalibration())(spot)
    @test isapprox(bondPrice, price, atol=1e-11)
    strike = spot - 1.0
    payoffC = AQFED.FDM.VanillaEuropean(true, strike, tte)
    payoffP = AQFED.FDM.VanillaEuropean(false, strike, tte)
    payoffCK = AQFED.FDM.KreissSmoothDefinition(payoffC)
    payoffPK = AQFED.FDM.KreissSmoothDefinition(payoffP)
    forwardPrice = spot * discountFactor(discountCurve, tte) / discountFactor(driftCurve, tte)
    for grid in grids
        price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=10, Smax=Smax, calibration=ForwardCalibration(), grid=grid)(spot)
        @test isapprox(forwardPrice, price, atol=1e-10)
    end
    price = AQFED.FDM.priceLogTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=10, Smax=Smax, calibration=ForwardCalibration())(spot)
    @test isapprox(forwardPrice, price, atol=1e-10)
    Smax = spot * exp(4 * σ * sqrt(tte))
    for grid in grids
        priceC = AQFED.FDM.priceTRBDF2(payoffC, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=10, Smax=Smax, calibration=ForwardCalibration(), grid=grid)(spot)
        priceP = AQFED.FDM.priceTRBDF2(payoffP, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=10, Smax=Smax, calibration=ForwardCalibration(), grid=grid)(spot)
        println(grid, " parity ", priceC - priceP, " == ", forwardPrice - strike * bondPrice, " err ", priceC - priceP - (forwardPrice - strike * bondPrice))
        @test isapprox(forwardPrice - strike * bondPrice, priceC - priceP, atol=1e-10)
        priceCK = AQFED.FDM.priceTRBDF2(payoffCK, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=10, Smax=Smax, calibration=ForwardCalibration(), grid=grid)(spot)
        pricePK = AQFED.FDM.priceTRBDF2(payoffPK, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=10, Smax=Smax, calibration=ForwardCalibration(), grid=grid)(spot)
        println(grid, " parity ", priceCK - pricePK, " == ", forwardPrice - strike * bondPrice, " err ", priceCK - pricePK - (forwardPrice - strike * bondPrice))
        @test isapprox(forwardPrice - strike * bondPrice, priceCK - pricePK, atol=1e-10)
    end

end

@testset "GridShift" begin

    r = 0.0488
    tte = 0.5833
    spot = 40.0
    σ = 0.4
    q = 0.0
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    dividends = Vector{CapitalizedDividend{Float64}}()
    strikes = range(25.0, stop=55.0, length=31)
    #[25.0,40.0,45.0,55.0]
    M = 201
    N = 51
    Smax = spot / discountFactor(driftCurve, tte) * exp(3 * σ * sqrt(tte))
    Smax = 100.0
    Smin = 15.0 # try 0 as well
    α = 0.1
    calibration = ForwardCalibration()
    calibration = NoCalibration()
    grids = [UniformGrid(false), LogGrid(false), CubicGrid(α), SinhGrid(α), SmoothlyDeformedGrid(UniformGrid(false)), SmoothlyDeformedGrid(LogGrid(false)), SmoothlyDeformedGrid(CubicGrid(α)), SmoothlyDeformedGrid(SinhGrid(α))]
    #European, calibrated, Smin=NaN
    data = DataFrame(Strike=Float64[], grid=String[], price=Float64[], error=Float64[])
    for (i, strike) in enumerate(strikes)
        # payoff = AQFED.FDM.VanillaEuropean(false, strike, tte)
        payoff = AQFED.FDM.VanillaAmerican(true, strike, tte, exerciseStartTime=0.0)
        payoffK = AQFED.FDM.KreissSmoothDefinition(payoff)
        blackPrice = AQFED.Black.blackScholesFormula(false, strike, spot, σ^2 * tte, discountFactor(driftCurve, tte), discountFactor(discountCurve, tte))
        blackPrice = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=8001, N=1001, Smax=Smax * 2, calibration=ForwardCalibration(), grid=ShiftedGrid(UniformGrid(true)), solverName="LUUL")(spot)

        for (j, grid) in enumerate(grids)
            keyname = if typeof(grid) <: UniformGrid
                "Uniform"
            elseif typeof(grid) <: SmoothlyDeformedGrid{UniformGrid}
                "Uniform-Deformed"
            elseif typeof(grid) <: SmoothlyDeformedGrid{CubicGrid}
                "Cubic-Deformed"
            elseif typeof(grid) <: SmoothlyDeformedGrid{SinhGrid}
                "Sinh-Deformed"
            elseif typeof(grid) <: SmoothlyDeformedGrid{LogGrid}
                "LogGrid-Deformed"
            elseif typeof(grid) <: CubicGrid
                string("Cubic-", α)
            elseif typeof(grid) <: SinhGrid
                string("Sinh-", α)
            elseif typeof(grid) <: LogGrid
                "LogGrid"
            end
            # if (typeof(grid) <: LogGrid) || (typeof(grid) <: SmoothlyDeformedGrid{LogGrid})
            #     Smin = 15.0
            # else 
            #     Smin = 0.0
            #  end
            price = if typeof(grid) <: SmoothlyDeformedGrid
                AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smax=Smax, Smin=Smin, solverName="LUUL", calibration=calibration, grid=grid)(spot)
            else
                AQFED.FDM.priceTRBDF2(payoffK, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smax=Smax, Smin=Smin, solverName="LUUL", calibration=calibration, grid=grid)(spot)
            end
            push!(data, [strike, keyname, price, price - blackPrice])
            println(strike, " ", keyname, " ", price, " ", price - blackPrice)
        end
        price = AQFED.FDM.priceLogTRBDF2(payoffK, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smax=Smax, calibration=calibration, grid=UniformGrid(false))(spot)
        keyname = "LogScheme"
        push!(data, [strike, keyname, price, price - blackPrice])
        println(strike, " ", keyname, " ", price, " ", price - blackPrice)
        price = AQFED.FDM.priceLogTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smax=Smax, calibration=calibration, grid=UniformGrid(false))(spot)
        keyname = "LogScheme-Deformed"
        push!(data, [strike, keyname, price, price - blackPrice])
        println(strike, " ", keyname, " ", price, " ", price - blackPrice)
        price = AQFED.FDM.priceLogTRBDF2(payoffK, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smax=Smax, calibration=calibration, grid=CubicGrid(α))(spot)
        keyname = "LogScheme-Cubic"
        push!(data, [strike, keyname, price, price - blackPrice])
        println(strike, " ", keyname, " ", price, " ", price - blackPrice)
    end
    #udata = data[@.(startswith(data.grid,"Log")),:]
    #julia> udata = data[@.(startswith(data.grid,"Cubic-0") || startswith(data.grid,"Uniform") || startswith(data.grid,"LogGrid")),:]
    #julia> plot(udata.Strike,abs.(udata.error),group=udata.grid,size=(800,300),xlabel="Strike",ylabel="Absolute error in price",margin=3Plots.mm)



    #        
    #         labels[1,8] = "LogSchemeUniform Uncalibrated"
    #         data[i,8]=price-blackPrice
    #         println(strike, " LogSchemeUniform Uncalibrated ",price, " ",price-blackPrice)
    #         price = AQFED.FDM.priceLogTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,calibration=ForwardCalibration(),grid="Shift")(spot)
    #         labels[1,9] = "LogSchemeUniform Calibrated"
    #         data[i,9]=price-blackPrice
    # println(strike, " LogSchemeUniform Calibrated ",price, " ",price-blackPrice)
    #         price = AQFED.FDM.priceLogTRBDF2(payoff, spot, driftCurve,varianceSurface, discountCurve, dividends,M=M,N=N,Smax=Smax,Smin=-1,calibration=NoCalibration(),grid="Cubic")(spot)
    #         labels[1,10] = "LogSchemeCubic Uncalibrated"
    #         data[i,10]=price-blackPrice
    #         println(strike, " LogSchemeCubic Uncalibrated ",price, " ",price-blackPrice)
    #             end
    #         end
    #julia> plot(strikes,abs.(data),label=labels,legend=:outertopleft,markershapes=[:circle],yscale=:log10)

    # end

end
@testset "AmericanPutNegativeRate" begin
    #policy iteration as fast as double sweep when r positive?
    r = -0.012
    σ = 0.1
    q = -0.004 + r
    tte = 360.0 / 365
    strike = 100.0
    spot = 100.0
    M = 1001
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)

    dividends = Vector{CapitalizedDividend{Float64}}()
    payoff = AQFED.FDM.VanillaAmerican(false, strike, tte)
    # Smax = spot*exp(abs((r-q)*tte+σ^2*tte/2) + 4*σ*sqrt(tte))
    Smax = spot * exp(3 * σ * sqrt(tte))
    #Convergence
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=4001, N=4001,
        solverName="TDMAPolicyIteration", Smax=Smax)(spot)
    N = 640 * 16
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N,
        solverName="TDMAPolicyIteration", Smax=Smax)(spot)
    @test isapprox(3.830520425, refPrice, atol=1e-4)
    N = 100
    price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, solverName="LUUL", Smax=Smax)(spot)
    @test isapprox(refPrice, price, atol=1e-4)

    N = 100
    payoff = AQFED.FDM.VanillaAmerican(false, strike, tte)
    solverNames = ["TDMA", "BrennanSchwartz", "TDMAPolicyIteration", "DoubleSweep", "LUUL", "PSOR"]
    for solverName in solverNames
        price = @time AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, solverName=solverName, Smax=Smax)(spot)
        println(solverName, " ", price, " ", price - refPrice)
    end

    Ns = [20, 40, 80, 160, 320]
    for solverName in solverNames
        previousError = 0.0
        for N in Ns
            price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, solverName=solverName)(spot)
            err = price - refPrice
            println(N, " ", solverName, " ", price, " ", err, " ", previousError / err)
            previousError = err
        end
    end

end

function batchALRKL(threshold, rInfq, isBatch, m; smoothing="None",sDisc = "Sinh",useSqrt=false,factor=10,lambdaS=0.25)
    rs = [0.02, 0.04, 0.06, 0.08, 0.10]
    qs = [0.0, 0.04, 0.08, 0.12]
    spots = [25.0, 50.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0, 175.0, 200.0]
    ttes = [1.0 / 12, 0.25, 0.5, 0.75, 1.0]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    strike = 100.0
    nodividends = Vector{CapitalizedDividend{Float64}}()

    refPrices = zeros((length(rs), length(qs), length(ttes), length(sigmas), length(spots)))
    @elapsed for (ir, r) in enumerate(rs), (iq, q) in enumerate(qs)
        if !skipRInfQ(rInfq, r, q)
            for (it, tte) in enumerate(ttes), (iv, vol) in enumerate(sigmas)
                model = ConstantBlackModel(vol, r, q)
                pricer = AndersenLakeRepresentation(model, tte, 1e-8, 16, 16, 31, 63)
                for (is, spot) in enumerate(spots)
                    refPrices[ir, iq, it, iv, is] = priceAmerican(pricer, strike, spot)
                end

            end
        end

    end
    prices = zeros(Float64,(length(rs), length(qs), length(ttes), length(sigmas), length(spots)))
    elap = @elapsed for (ir, r) in enumerate(rs), (iq, q) in enumerate(qs)
        if !skipRInfQ(rInfq, r, q)
            for (it, tte) in enumerate(ttes), (iv, vol) in enumerate(sigmas)
                model = ConstantBlackModel(vol, r, q)                
                local pricer
                if isBatch
                    pricer = AQFED.American.makeFDMPriceInterpolation(false, false, model, tte,strike, m+1,(factor*m+1),smoothing=smoothing,useSqrt=useSqrt,lambdaS=lambdaS,Xdev=4,sDisc=sDisc)
                end
                for (is, spot) in enumerate(spots)
                    if refPrices[ir, iq, it, iv, is] > threshold
                        if !isBatch
                            pricer = AQFED.American.makeFDMPriceInterpolation(false, false, model, tte,strike, m+1,(factor*m+1),smoothing=smoothing,useSqrt=useSqrt,lambdaS=lambdaS,Xdev=4,sDisc=sDisc)
                        end
                        prices[ir, iq, it, iv, is] = pricer(spot)
                        if abs(prices[ir, iq, it, iv, is] - refPrices[ir, iq, it, iv, is]) > 0.1
                            println(
                                ir,
                                " ",
                                iq,
                                " ",
                                it,
                                " ",
                                iv,
                                " ",
                                is,
                                " ",
                                refPrices[ir, iq, it, iv, is],
                                " ",
                                prices[ir, iq, it, iv, is],
                                " ",
                                spot,
                                " ",
                                vol,
                                " ",
                                tte,
                                " ",
                                q,
                                " ",
                                r,
                            )
                        end
                    end
                end

            end
        end

    end
    return prices, refPrices, elap
end

function batchAL(threshold, rInfq, isBatch, m; grid=SmoothlyDeformedGrid(CubicGrid(0.1)),useSqrt=false,factor=10)
    rs = [0.02, 0.04, 0.06, 0.08, 0.10]
    qs = [0.0, 0.04, 0.08, 0.12]
    spots = [25.0, 50.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0, 175.0, 200.0]
    ttes = [1.0 / 12, 0.25, 0.5, 0.75, 1.0]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    strike = 100.0
    nodividends = Vector{CapitalizedDividend{Float64}}()

    refPrices = zeros((length(rs), length(qs), length(ttes), length(sigmas), length(spots)))
    @elapsed for (ir, r) in enumerate(rs), (iq, q) in enumerate(qs)
        if !skipRInfQ(rInfq, r, q)
            for (it, tte) in enumerate(ttes), (iv, vol) in enumerate(sigmas)
                model = ConstantBlackModel(vol, r, q)
                pricer = AndersenLakeRepresentation(model, tte, 1e-8, 16, 16, 31, 63)
                for (is, spot) in enumerate(spots)
                    refPrices[ir, iq, it, iv, is] = priceAmerican(pricer, strike, spot)
                end

            end
        end

    end
    prices = zeros(Float64,(length(rs), length(qs), length(ttes), length(sigmas), length(spots)))
    elap = @elapsed for (ir, r) in enumerate(rs), (iq, q) in enumerate(qs)
        if !skipRInfQ(rInfq, r, q)
            for (it, tte) in enumerate(ttes), (iv, vol) in enumerate(sigmas)
                model = ConstantBlackModel(vol, r, q)                
                payoff = VanillaAmerican(false, strike, tte)
                local pricer
                if isBatch
                    pricer = AQFED.FDM.priceTRBDF2(payoff, spot, model, nodividends, M=(factor*m+1), N=(m+1), grid=grid, solverName="BrennanSchwartz",useSqrt=useSqrt)                
                end
                for (is, spot) in enumerate(spots)
                    if refPrices[ir, iq, it, iv, is] > threshold
                        if !isBatch
                            pricer =
                            AQFED.FDM.priceTRBDF2(payoff, spot, model, nodividends, M=(factor*m+1), N=(m+1), grid=grid, solverName="BrennanSchwartz",useSqrt=useSqrt)                
                        end
                        prices[ir, iq, it, iv, is] = pricer(spot)
                        if abs(prices[ir, iq, it, iv, is] - refPrices[ir, iq, it, iv, is]) > 0.1
                            println(
                                ir,
                                " ",
                                iq,
                                " ",
                                it,
                                " ",
                                iv,
                                " ",
                                is,
                                " ",
                                refPrices[ir, iq, it, iv, is],
                                " ",
                                prices[ir, iq, it, iv, is],
                                " ",
                                spot,
                                " ",
                                vol,
                                " ",
                                tte,
                                " ",
                                q,
                                " ",
                                r,
                            )
                        end
                    end
                end

            end
        end

    end
    return prices, refPrices, elap
end

@testset "AndersenLakeBenchmark" begin
    threshold = 0.5
    rInfq = 0 #1, -1 or 0
    isBatch = true
    prices, refPrices, elap = batchAL(threshold, rInfq, isBatch,20,useSqrt=true,grid=SmoothlyDeformedGrid(CubicGrid(0.035)));elap
    # prices, refPrices, elap = batchAL(threshold, rInfq, isBatch,20,useSqrt=true,grid=SmoothlyDeformedGrid(SinhGrid(0.05)));elap
    thrIndices = findall(z -> z > threshold, refPrices);
    mae = maxad(prices[thrIndices], refPrices[thrIndices])
    mre = maxad(prices[thrIndices] ./ refPrices[thrIndices], ones(length(thrIndices)))
    rmse = rmsd(prices[thrIndices], refPrices[thrIndices])
    println(
        "AL ",
        20,
        " ",
        rmse,
        " ",
        mae,
        " ",
        mre,
        " ",
        elap,
        " ",
        length(thrIndices) / elap,
    )    
    prices, refPrices, elap = batchAL(threshold, rInfq, isBatch,40,useSqrt=true,grid=SmoothlyDeformedGrid(CubicGrid(0.035)));elap
    # prices, refPrices, elap = batchAL(threshold, rInfq, isBatch,20,useSqrt=true,grid=SmoothlyDeformedGrid(SinhGrid(0.05)));elap
    thrIndices = findall(z -> z > threshold, refPrices);
    mae = maxad(prices[thrIndices], refPrices[thrIndices])
    mre = maxad(prices[thrIndices] ./ refPrices[thrIndices], ones(length(thrIndices)))
    rmse = rmsd(prices[thrIndices], refPrices[thrIndices])
    println(
        "AL ",
        40,
        " ",
        rmse,
        " ",
        mae,
        " ",
        mre,
        " ",
        elap,
        " ",
        length(thrIndices) / elap,
    )    
end
