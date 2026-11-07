using AQFED, Test, AQFED.FDM, AQFED.TermStructure, AQFED.Black
using DataFrames

@testset "BlackUpAndOut" begin
    spot = 6317.80
	barrierLevel = spot * 1.2
	rebate = 0.0 * spot
	r = 0.01
    q = -0.02	
	tte = 1.0
    σ = 0.2
    strike = 0.8 * spot
    isCall = true
	xSteps = 401
	tSteps = 1001
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    dividends = Vector{CapitalizedDividend{Float64}}()
    payoffV = VanillaEuropean(isCall, strike, tte)
    payoffKV = KreissSmoothDefinition(payoffV)
    payoff = AQFED.FDM.ContinuousKO(payoffV, barrierLevel, false, 0.0, 0.0,tte)
    payoffK = AQFED.FDM.ContinuousKO(payoffKV, barrierLevel, false, 0.0, 0.0,tte)
    Smin = 0.0
    Smax = 2*barrierLevel
    model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
    refPriceV = AQFED.FDM.priceTRBDF2(payoffV, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPriceKV = AQFED.FDM.priceTRBDF2(payoffKV, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
    refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
    refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(CubicGrid(0.1)))
    refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(CubicGrid(0.1)))
    refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(CubicGrid(0.1)))

    barrierEngine = AQFED.Black.AnalyticBarrierEngine(spot, σ^2 *tte,  exp(-(r-q)*tte), exp(-r*tte),exp(-r*tte))
    anPrice = AQFED.Black.calculate(barrierEngine, AQFED.Black.BarrierPayoff(isCall,strike,barrierLevel,0.0,AQFED.Black.UpOut))
    #error vs spot
    spots = range(0.5*spot,stop=1.2*spot,length=101)
    anPriceA = @. AQFED.Black.calculate(AQFED.Black.AnalyticBarrierEngine(spots, σ^2 *tte,  exp(-(r-q)*tte), exp(-r*tte),exp(-r*tte)),AQFED.Black.BarrierPayoff(isCall,strike,barrierLevel,0.0,AQFED.Black.UpOut))
    spot= strike
    barrierEngine = AQFED.Black.AnalyticBarrierEngine(spot, σ^2 *tte,  exp(-(r-q)*tte), exp(-r*tte),exp(-r*tte))
    anPrice = AQFED.Black.calculate(barrierEngine, AQFED.Black.BarrierPayoff(isCall,strike,barrierLevel,0.0,AQFED.Black.UpOut))
     println("reference KO ", refPrice(spot), " vanilla ", refPriceV(spot))
     xStepsA = collect(201:301)
     errK = zeros(length(xStepsA))
     err = zeros(length(xStepsA))
     errS = zeros(length(xStepsA))
     refGrid = UniformGrid(false)
     refGrid = CubicGrid(0.1)
     for (i,xSteps) = enumerate(xStepsA)
        refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(refGrid))
       errK[i] = refPriceK(spot)-anPrice
        refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(refGrid))
          err[i] = refPrice(spot)-anPrice
          refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(refGrid))
          errS[i] = refPriceS(spot)-anPrice
          end
end
@testset "Peclet" begin
    spot = 100.0
    r = 0.1
    q = 0.0
    σ = 0.02
    tte = 1.0
    strike = spot
    isCall = false
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    dividends = Vector{CapitalizedDividend{Float64}}()
    model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
    payoffV = Binary(isCall, strike, tte)
    #payoffKV = KreissSmoothDefinition(payoffV)
    payoffKO = AQFED.FDM.DiscreteKO(payoffV, spot*0.1, true, 0.0,[tte])
    payoffK = KreissSmoothDefinition(payoffKO)
    payoff = AQFED.FDM.DiscreteKO(payoffV, spot*0.1, true, 0.0,[tte])
    refPrice = AQFED.FDM.priceRKL2(payoffK, spot, model, dividends, M=401, N=101, Smin=0.0, Smax=150.0, grid=(CubicGrid(1)),varianceConditioner=PartialExponentialFittingConditioner())
#oscillations in RKL scheme, even with exponential fitting or two points upwinding. no oscillations in RKG/TRBDF2
# eigsvals with imag eigenvalues even with exp fitting (removes some but not all)
    refPrice = AQFED.FDM.priceRKL2(payoffK, spot, model, dividends, M=401, N=21, Smin=0.0, Smax=150.0, grid=CubicGrid(0.01),varianceConditioner=AQFED.FDM.PartialExponentialFittingConditioner())

    refPrice = AQFED.FDM.priceRKL2(payoffK, spot, model, dividends, M=101, N=101, Smin=0.0, Smax=150.0, grid=UniformGrid(false))
end

@testset "BlackOneTouch" begin
    spot = 6317.80
	barrierLevel = spot * 1.2
	rebate = 0.0 * spot
	r = 0.0
    q = -0.0
   #  q = -1.5;barrierLevel=4spot
    # q = 1.5;barrierLevel=0.5*spot
tte = 1.0
    σ = 0.2
    isCall = true
	xSteps = 101
	tSteps = 401
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    dividends = Vector{CapitalizedDividend{Float64}}()
    payoffV = Binary(isCall, barrierLevel, tte)
    payoffKV = KreissSmoothDefinition(payoffV)
    payoff = AQFED.FDM.ContinuousKO(payoffV, barrierLevel, false, 1.0, 0.0,tte)
    payoffK = AQFED.FDM.ContinuousKO(payoffKV, barrierLevel, false, 1.0, 0.0,tte)
    Smin = 0.0
    Smax = 0.0
    α = 0.01
    model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
    refPriceV = AQFED.FDM.priceTRBDF2(payoffV, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPriceKV = AQFED.FDM.priceTRBDF2(payoffKV, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
    refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
    refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(CubicGrid(α)))
    refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(CubicGrid(α)))
    refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(CubicGrid(α)))

    barrierEngine = AQFED.Black.AnalyticBarrierEngine(spot, σ^2 *tte,  exp(-(r-q)*tte), exp(-r*tte),exp(-r*tte))
    anPrice = AQFED.Black.calculateOneTouch(barrierEngine, false,barrierLevel,1.0)
   
    dS = refPrice.x[2]-refPrice.x[1]
    ius = searchsorted(refPrice.x,barrierLevel)
    ium = ius.stop; iu=ius.start
    
    dtThreshold = 4*dS^2/(σ^2*refPrice.x[ium]^2*(3+(refPrice.x[iu]-barrierLevel)/(barrierLevel-refPrice.x[ium])))
    n = ceil(Int,tte/dtThreshold)
    refPriceRBad = AQFED.FDM.priceExplicitEuler(payoff, spot, model, dividends, M=101, N=401, Smin=Smin, Smax=Smax, grid=UniformGrid(false))
    refPriceR = AQFED.FDM.priceExplicitEuler(payoff, spot, model, dividends, M=101, N=401, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
#worse for Smax=13784.105478964535, dtThreashold seems to be *2 of what it should be.
refPriceR = AQFED.FDM.priceRKG2(payoff, spot, model, dividends, M=101, N=101, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))

end

@testset "BlackUpAndIn" begin
    spot = 6317.80
	barrierLevel = spot * 1.2
	rebate = 0.0 * spot
	r = 0.0
    q = -0.0
	tte = 1.0
    σ = 0.2
    strike = 0.8 * spot
    isCall = true
	xSteps = 401
	tSteps = 1001
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    dividends = Vector{CapitalizedDividend{Float64}}()
    payoffV = VanillaEuropean(isCall, strike, tte)
    payoffKV = KreissSmoothDefinition(payoffV)
    payoff = AQFED.FDM.ContinuousKI(payoffV, barrierLevel, false, 0.0, 0.0,tte)
    payoffK = AQFED.FDM.ContinuousKI(payoffKV, barrierLevel, false, 0.0, 0.0,tte)
   Smin = 0.0
    Smax = 0.0
    α = 0.01
    model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
    refPriceV = AQFED.FDM.priceTRBDF2(payoffV, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPriceKV = AQFED.FDM.priceTRBDF2(payoffKV, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
    refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
    refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(CubicGrid(α)))
    refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(CubicGrid(α)))
    refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(CubicGrid(α)))

    barrierEngine = AQFED.Black.AnalyticBarrierEngine(spot, σ^2 *tte,  exp(-(r-q)*tte), exp(-r*tte),exp(-r*tte))
    anPrice = AQFED.Black.calculate(barrierEngine, AQFED.Black.BarrierPayoff(isCall,strike,barrierLevel,0.0,AQFED.Black.UpIn))
    #error vs spot
    spots = range(0.5*spot,stop=1.2*spot,length=101)
    anPriceA = @. AQFED.Black.calculate(AQFED.Black.AnalyticBarrierEngine(spots, σ^2 *tte,  exp(-(r-q)*tte), exp(-r*tte),exp(-r*tte)),AQFED.Black.BarrierPayoff(isCall,strike,barrierLevel,0.0,AQFED.Black.UpIn))
    spot= strike
    barrierEngine = AQFED.Black.AnalyticBarrierEngine(spot, σ^2 *tte,  exp(-(r-q)*tte), exp(-r*tte),exp(-r*tte))
    anPrice = AQFED.Black.calculate(barrierEngine, AQFED.Black.BarrierPayoff(isCall,strike,barrierLevel,0.0,AQFED.Black.UpIn))
     println("reference KO ", refPrice(spot), " vanilla ", refPriceV(spot))
     xStepsA = collect(201:301)
     errK = zeros(length(xStepsA))
     err = zeros(length(xStepsA))
     errS = zeros(length(xStepsA))
     refGrid = UniformGrid(false)
     refGrid = CubicGrid(0.1)
     for (i,xSteps) = enumerate(xStepsA)
        refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(refGrid))
       errK[i] = refPriceK(spot)-anPrice
        refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(refGrid))
          err[i] = refPrice(spot)-anPrice
          refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(refGrid))
          errS[i] = refPriceS(spot)-anPrice
          end
end


@testset "BlackUpAndInThenDownAndOut" begin
    spot = 6317.80
	barrierLevelUp = spot * 1.2
    barrierLevelDown = spot * 0.9
	rebate = 0.0 * spot
	r = 0.0
    q = -0.0
	tte = 1.0
    σ = 0.2
    strike = 0.8 * spot
    isCall = true
	xSteps = 401
	tSteps = 1001
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    dividends = Vector{CapitalizedDividend{Float64}}()
    payoffV = VanillaEuropean(isCall, strike, tte)
    payoffKV = KreissSmoothDefinition(payoffV)
    payoffKO = AQFED.FDM.ContinuousKO(payoffV, barrierLevelDown, true, 0.0, 0.0,tte)
    payoffKOK = AQFED.FDM.ContinuousKO(payoffKV, barrierLevelDown, true, 0.0, 0.0,tte)
    
    payoffK = AQFED.FDM.ContinuousKI(payoffKOK, barrierLevelUp, false, 0.0, 0.0,tte)
    payoff = AQFED.FDM.ContinuousKI(payoffKO, barrierLevelUp, false, 0.0, 0.0,tte)
   Smin = 0.0
    Smax = 0.0
    α = 0.01
    model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
    refPriceV = AQFED.FDM.priceTRBDF2(payoffV, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPriceKV = AQFED.FDM.priceTRBDF2(payoffKV, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
    refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
    refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(CubicGrid(α)))
    refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(CubicGrid(α)))
    refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(CubicGrid(α)))

    barrierEngine = AQFED.Black.AnalyticBarrierEngine(spot, σ^2 *tte,  exp(-(r-q)*tte), exp(-r*tte),exp(-r*tte))
    anPrice = AQFED.Black.calculateUpInDownOutEnd(barrierEngine, isCall, strike, barrierLevelDown, barrierLevelUp)
    #error vs spot
    spots = range(0.5*spot,stop=1.19*spot,length=101)
    anPriceA = @. AQFED.Black.calculateUpInDownOutEnd(AQFED.Black.AnalyticBarrierEngine(spots, σ^2 *tte,  exp(-(r-q)*tte), exp(-r*tte),exp(-r*tte)),isCall, strike, barrierLevelDown, barrierLevelUp)
    spot= strike
    barrierEngine = AQFED.Black.AnalyticBarrierEngine(spot, σ^2 *tte,  exp(-(r-q)*tte), exp(-r*tte),exp(-r*tte))
    anPrice = AQFED.Black.calculateUpInDownOutEnd(barrierEngine, isCall, strike, barrierLevelDown, barrierLevelUp)
    println("reference KO ", refPrice(spot), " vanilla ", refPriceV(spot))
     xStepsA = collect(201:301)
     errK = zeros(length(xStepsA))
     err = zeros(length(xStepsA))
     errS = zeros(length(xStepsA))
     refGrid = UniformGrid(false)
     refGrid = CubicGrid(0.1)
     for (i,xSteps) = enumerate(xStepsA)
        refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(refGrid))
       errK[i] = refPriceK(spot)-anPrice
        refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(refGrid))
          err[i] = refPrice(spot)-anPrice
          refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(refGrid))
          errS[i] = refPriceS(spot)-anPrice
          end
end

@testset "BlackDKO" begin
    spot = 100.0
    barrierLevelUp = 101.0
    barrierLevelDown = 99.9
	rebate = 0.0 * spot
	r = 0.0
    q = -0.0
	tte = 1.0/52
    σ = 0.2
  	xSteps = 1501
	tSteps = 201
    varianceSurface = FlatSurface(σ)
    discountCurve = ConstantRateCurve(r)
    driftCurve = ConstantRateCurve(r - q)
    dividends = Vector{CapitalizedDividend{Float64}}()
    payoffV = Binary(true, barrierLevelUp, tte)
    payoffKV = KreissSmoothDefinition(payoffV)
    payoff = AQFED.FDM.ContinuousDKO(payoffV, barrierLevelUp, barrierLevelDown, 1.0, 0.0,tte)
    payoffK = AQFED.FDM.ContinuousDKO(payoffKV,barrierLevelUp, barrierLevelDown, 1.0, 0.0,tte)
    Smin = 0.0
    Smax = 0.0
    α = 0.01
    model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
    refPriceV = AQFED.FDM.priceTRBDF2(payoffV, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPriceKV = AQFED.FDM.priceTRBDF2(payoffKV, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
    refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
    refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(CubicGrid(α)))
    refPriceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(CubicGrid(α)))
    refPriceS = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(CubicGrid(α)))

    priceCN = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(CubicGrid(α)),timeSteppingName="CN")
    priceRAN = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(CubicGrid(α)),timeSteppingName="Rannacher")
    priceRAN4 = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(CubicGrid(α)),timeSteppingName="Rannacher4")
    priceRKG = AQFED.FDM.priceRKG2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(CubicGrid(α)))
end
@testset "ContinousKO" begin
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
   Smin = 0.0
    Smax = 150.0 #3 dev = 200.0 !
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
    model = AQFED.TermStructure.TSBlackModel(varianceSurface, discountCurve, driftCurve)
	
    refPriceV = AQFED.FDM.priceTRBDF2(payoffV, spot, model, dividends, M=16001, N=N, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=16001, N=N, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    println("reference KO ", refPrice(spot), " vanilla ", refPriceV(spot))
    α = 2.0 / (Smax -Smin)
    Ms = [251, 501, 1001, 2001]
    grids = [UniformGrid(false), ShiftedGrid(UniformGrid(false)), SmoothlyDeformedGrid(UniformGrid(false)), CubicGrid(α), SmoothlyDeformedGrid(CubicGrid(α)), SinhGrid(α), SmoothlyDeformedGrid(SinhGrid(α))]
    for M in Ms
        for grid in grids
            priceV = AQFED.FDM.priceTRBDF2(payoffV, spot, model, dividends, M=M, N=N, Smin=Smin, Smax=Smax, grid=grid)(spot)
            priceKV = AQFED.FDM.priceTRBDF2(payoffKV, spot, model, dividends, M=M, N=N, Smin=Smin, Smax=Smax, grid=grid)(spot)

            price = AQFED.FDM.priceTRBDF2(payoff, spot, model,dividends, M=M, N=N, Smin=Smin, Smax=Smax, grid=grid)(spot)
            priceK = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=M, N=N, Smin=Smin, Smax=Smax, grid=grid)(spot)
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
            localMin =Smin
            if typeof(grid) <: LogGrid || typeof(grid) <: SmoothlyDeformedGrid{LogGrid}
                localMin = 50.0
                #localMax = 180.0
            end
            priceV = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=M, N=N, Smin=localMin, Smax=Smax, grid=grid)(spot)
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
                priceKV = AQFED.FDM.priceTRBDF2(payoffK, spot, model, dividends, M=M, N=N, Smin=localMin, Smax=Smax, grid=grid)(spot)
                push!(data, [M, string(keyname, "-", "Kreiss"), priceKV, priceKV - refPrice(spot)])
            end
        end
    end
end

#ghost point against grid deformation with the barrier off the grid (Smax at 4 deviations): with a consistent quadratic
#ghost point on both sides of the time step, both are equally accurate; a linear ghost point on the explicit side made the
#up barrier errors 3 to 5 times larger.
@testset "GhostPointOffGrid" begin
    spot, tte, σ = 6317.80, 1.0, 0.2
    strike, up, down = 0.8spot, 1.2spot, 0.9spot
    model = TSBlackModel(FlatSurface(σ), ConstantRateCurve(0.0), ConstantRateCurve(0.0))
    engine(s) = AQFED.Black.AnalyticBarrierEngine(s, σ^2 * tte, 1.0, 1.0, 1.0)
    call, put = AQFED.FDM.VanillaEuropean(true, strike, tte), AQFED.FDM.VanillaEuropean(false, 1.2spot, tte)
    #name, payoff, analytic price, spot range (1% away from the barriers, where the interpolated price has a kink), max error uniform, cubic
    cases = [
        ("UpOut", AQFED.FDM.ContinuousKO(call, up, false, 0.0, 0.0, tte), s -> AQFED.Black.calculate(engine(s), AQFED.Black.BarrierPayoff(true, strike, up, 0.0, AQFED.Black.UpOut)), (0.5spot, 0.99up), (0.048, 0.035)),
        ("UpIn", AQFED.FDM.ContinuousKI(call, up, false, 0.0, 0.0, tte), s -> AQFED.Black.calculate(engine(s), AQFED.Black.BarrierPayoff(true, strike, up, 0.0, AQFED.Black.UpIn)), (0.5spot, 0.99up), (0.036, 0.018)),
        ("KIKO", AQFED.FDM.ContinuousKI(AQFED.FDM.ContinuousKO(call, down, true, 0.0, 0.0, tte), up, false, 0.0, 0.0, tte), s -> AQFED.Black.calculateUpInDownOutEnd(engine(s), true, strike, down, up), (1.01down, 0.99up), (0.049, 0.015)),
        ("DownIn", AQFED.FDM.ContinuousKI(put, 0.8spot, true, 0.0, 0.0, tte), s -> AQFED.Black.calculate(engine(s), AQFED.Black.BarrierPayoff(false, 1.2spot, 0.8spot, 0.0, AQFED.Black.DownIn)), (1.01 * 0.8spot, 1.5spot), (0.12, 0.053))]
    for (name, payoff, analytic, (lo, hi), maxErrors) in cases, (grid, maxError) in zip((UniformGrid(false), CubicGrid(0.1)), maxErrors)
        spots = collect(range(lo, hi, length=101))
        price(g) = AQFED.FDM.priceTRBDF2(payoff, spot, model, CapitalizedDividend{Float64}[], M=401, N=1001, Smin=0.0, Smax=0.0, grid=g)
        maxErr(p) = maximum(abs.(p.(spots) .- analytic.(spots)))
        errGhost, errDeformed = maxErr(price(grid)), maxErr(price(SmoothlyDeformedGrid(grid)))
        println(name, " ", typeof(grid), " ghost ", errGhost, " deformed ", errDeformed)
        @test errGhost < maxError
        @test errDeformed < maxError
        @test errGhost < 1.6 * errDeformed
    end
end

#explicit schemes place a continuous barrier on the grid: a ghost point modifies the barrier row, and on a grid stretched at
#the barrier, the number of stages estimated from the interior operator was too small (NaN for this cubic grid with M=201)
@testset "ExplicitBarrierOnGrid" begin
    spot, tte, σ = 6317.80, 1.0, 0.2
    strike, up, down = 0.8spot, 1.2spot, 0.9spot
    model = TSBlackModel(FlatSurface(σ), ConstantRateCurve(0.0), ConstantRateCurve(0.0))
    engine(s) = AQFED.Black.AnalyticBarrierEngine(s, σ^2 * tte, 1.0, 1.0, 1.0)
    call = AQFED.FDM.VanillaEuropean(true, strike, tte)
    cases = [
        ("UpOut", AQFED.FDM.ContinuousKO(call, up, false, 0.0, 0.0, tte), s -> AQFED.Black.calculate(engine(s), AQFED.Black.BarrierPayoff(true, strike, up, 0.0, AQFED.Black.UpOut)), (0.5spot, 0.99up), 0.19),
        ("UpIn", AQFED.FDM.ContinuousKI(call, up, false, 0.0, 0.0, tte), s -> AQFED.Black.calculate(engine(s), AQFED.Black.BarrierPayoff(true, strike, up, 0.0, AQFED.Black.UpIn)), (0.5spot, 0.99up), 0.07),
        ("KIKO", AQFED.FDM.ContinuousKI(AQFED.FDM.ContinuousKO(call, down, true, 0.0, 0.0, tte), up, false, 0.0, 0.0, tte), s -> AQFED.Black.calculateUpInDownOutEnd(engine(s), true, strike, down, up), (1.01down, 0.99up), 0.065)]
    for (name, payoff, analytic, (lo, hi), maxError) in cases, pricer in (AQFED.FDM.priceRKL2, AQFED.FDM.priceRKG2)
        spots = collect(range(lo, hi, length=101))
        p = pricer(payoff, spot, model, CapitalizedDividend{Float64}[], M=201, N=101, Smin=0.0, Smax=0.0, grid=CubicGrid(0.1))
        err = maximum(abs.(p.(spots) .- analytic.(spots)))
        println(name, " ", pricer, " ", err)
        @test err < maxError
    end
end

#a barrier level just above a grid point (rounding) is on the grid, the price is continuous in the level
@testset "BarrierJustAboveNode" begin
    spot, tte, σ = 6317.80, 1.0, 0.2
    strike, up = 0.8spot, 1.2spot
    model = TSBlackModel(FlatSurface(σ), ConstantRateCurve(0.0), ConstantRateCurve(0.0))
    call = AQFED.FDM.VanillaEuropean(true, strike, tte)
    for makePayoff in (b -> AQFED.FDM.ContinuousKO(call, b, false, 0.0, 0.0, tte), b -> AQFED.FDM.ContinuousKI(call, b, false, 0.0, 0.0, tte))
        price(b) = priceTRBDF2(makePayoff(b), spot, model, CapitalizedDividend{Float64}[], M=201, N=201, Smin=0.0, Smax=2up, grid=UniformGrid(false))(spot)
        @test isapprox(price(up + 1e-9), price(up), atol=1e-8) #up is on the uniform grid
    end
end
