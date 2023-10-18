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
    #=
    plot(spots./strike, abs.(refPrice.(spots) .- anPriceA), xlab="Spot/Strike", ylab="Absolute error in price", label="Ghost");    plot!(spots./strike, abs.(refPriceK.(spots) .- anPriceA), label="Ghost Kreiss");    plot!(spots./strike, abs.(refPriceS.(spots) .- anPriceA), label="Deformed")
    plot!(size=(400,300),margins=1Plots.mm)
    savefig("~//mypapers/eqd_book/barrier_ko_spot_401.pdf")
    =#
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
	#=
    plot(xStepsA,abs.(err),label="Ghost");plot!(xStepsA, abs.(errK),xlab="Number of space steps", ylab="Absolute error in price",label="Ghost Kreiss");plot!(xStepsA,abs.(errS),label="Deformed")
  plot!(size=(400,300),margins=1Plots.mm)
 savefig("~//mypapers/eqd_book/barrier_ko_steps_100.pdf")  
 =#
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
    #=
    etraceE = AQFED.FDM.ExtendedTrace()
    etrace = AQFED.FDM.ExtendedTrace()
    refPrice = AQFED.FDM.priceRKG2(payoffK, spot, model, dividends, M=101, N=101, Smin=0.0, Smax=150.0, grid=UniformGrid(false),eTrace=etraceE,varianceConditioner=PartialExponentialFittingConditioner())
    refPrice = AQFED.FDM.priceRKG2(payoffK, spot, model, dividends, M=101, N=101, Smin=0.0, Smax=150.0, grid=UniformGrid(false),eTrace=etrace)
    plot(refPrice.x,PPInterpolation.evaluateDerivative.(refPrice,refPrice.x),xlab="Underlying spot price", ylab="Option Δ",label="RKL",size=(400,300))
     plot(etrace.eigenvalues, seriestype=:scatter,label="Partial Exponential Fitting",size=(400,300),ms=3,markerstrokewidth=0,markeralpha=0.5)
plot(refPrice.x[2:end-1],etrace.peclets[2:end-1],xlab="Underlying asset price", ylab="Peclet ratio",yscale=:log10,ylims=(0.5,250),yticks=([1,10,100],[1,10,100]),label="No upwinding")
 plot!(refPrice.x[2:end-1],etraceE.peclets[2:end-1],label="Partial exponential fitting")

     =#
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

   #= 
   is = collect(n-50:n); prices = zeros(length(is))
   for (i,ni)=enumerate(is)
   refPriceR = AQFED.FDM.priceExplicitEuler(payoff, spot, model, dividends, M=101, N=ni, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
   prices[i] = refPriceR(spot)
   end
    plot(is,abs.(prices.-anPrice),yscale=:log10,xlab="Number of time-steps",ylab="Absolute error in price",legend=:none)
    #plot(is[50:end],abs.(prices[50:end].-anPrice),yscale=:log10,xlab="Number of time-steps",ylab="Absolute error in price",legend=:none)
Srange = 13590.0:1.0:13990.0
nA = zeros(length(Srange));
neffA = zeros(length(Srange));
nETruncA = zeros(length(Srange));
    for (i,Smax)=enumerate(Srange)
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, model, dividends, M=xSteps, N=tSteps, Smin=Smin, Smax=Smax, grid=(UniformGrid(false)))
      dS = refPrice.x[2]-refPrice.x[1]
    ius = searchsorted(refPrice.x,barrierLevel)
    ium = ius.stop; iu=ius.start
   
     dtThresholdETrunc = dS^2/(σ^2*barrierLevel^2)
    nETrunc = ceil(Int,tte/dtThresholdETrunc)
 dtThresholdEFull = dS^2/(σ^2*refPrice.x[end]^2)
    nEFull = ceil(Int,tte/dtThresholdE)
    nETruncA[i] = nETrunc
    dtThreshold = 4*dS^2/(σ^2*refPrice.x[ium]^2*(3+(refPrice.x[iu]-barrierLevel)/(barrierLevel-refPrice.x[ium])))
    n = ceil(Int,tte/dtThreshold)
    errorR = 1.0
    ni = n
    while errorR > 1e-3 && ni < 30000
        ni+=1
        refPriceRBad = AQFED.FDM.priceExplicitEuler(payoff, spot, model, dividends, M=101, N=ni, Smin=Smin, Smax=Smax, grid=UniformGrid(false))
        errorR = abs(refPriceRBad(spot)-anPrice)
    end
  while errorR < 1e-3 && ni > 1
        refPriceRBad = AQFED.FDM.priceExplicitEuler(payoff, spot, model, dividends, M=101, N=ni, Smin=Smin, Smax=Smax, grid=UniformGrid(false))
        errorR = abs(refPriceRBad(spot)-anPrice)
        ni-=1
    end
    nA[i] = n
    neffA[i] = ni
    println(Smax," ",barrierLevel-refPrice.x[ium], " ",(barrierLevel-refPrice.x[ium])/dS," ",n, " ",ni," ",errorR)
end
nA2 = max.(nA, nETruncA)
plot(Srange, nA2, label="Theoretical",xlab="Grid upper boundary Sₘₐₓ",ylab="Number of time-steps for stability")
plot!(Srange,neffA,label="Actual",legendtitle="Threshold")
plot!(yscale=:log10)
plot!(size)
   =#
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
    #=
    plot(spots./strike, abs.(refPrice.(spots) .- anPriceA), xlab="Spot/Strike", ylab="Absolute error in price", label="Ghost");    plot!(spots./strike, abs.(refPriceK.(spots) .- anPriceA), label="Ghost Kreiss");    plot!(spots./strike, abs.(refPriceS.(spots) .- anPriceA), label="Deformed")
    plot!(size=(400,300),margins=1Plots.mm)
    savefig("~//mypapers/eqd_book/barrier_ki_spot_401.pdf")
    =#
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
	#=
    plot(xStepsA,abs.(err),label="Ghost");plot!(xStepsA, abs.(errK),xlab="Number of space steps", ylab="Absolute error in price",label="Ghost Kreiss");plot!(xStepsA,abs.(errS),label="Deformed")
  plot!(size=(400,300),margins=1Plots.mm)
 savefig("~//mypapers/eqd_book/barrier_ko_steps_100.pdf")  
 =#
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
    #=
    plot(spots./strike, abs.(refPrice.(spots) .- anPriceA), xlab="Spot/Strike", ylab="Absolute error in price", label="Ghost");    plot!(spots./strike, abs.(refPriceK.(spots) .- anPriceA), label="Ghost Kreiss");    plot!(spots./strike, abs.(refPriceS.(spots) .- anPriceA), label="Deformed")
    plot!(size=(400,300),margins=1Plots.mm)
    savefig("~//mypapers/eqd_book/barrier_kiko_spot_401_cubic01.pdf")
    =#
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
	#=
    plot(xStepsA,abs.(err),label="Ghost");plot!(xStepsA, abs.(errK),xlab="Number of space steps", ylab="Absolute error in price",label="Ghost Kreiss");plot!(xStepsA,abs.(errS),label="Deformed")
  plot!(size=(400,300),margins=1Plots.mm)
 savefig("~//mypapers/eqd_book/barrier_ko_steps_100.pdf")  
 =#
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
#= 
   plot(refPriceS.x,PPInterpolation.evaluateDerivative.(refPriceS,refPriceS.x),xlab="Underlying spot price", ylab="Option Δ",label="TR-BDF2")
   plot!(priceRKG.x,PPInterpolation.evaluateDerivative.(priceRKG,priceRKG.x),xlab="Underlying spot price", ylab="Option Δ",label="RKG")
   plot!(xlim=(barrierLevelDown,barrierLevelUp),size=(400,200),margin=3Plots.mm)
 savefig("~//mypapers/eqd_book/doubleonetouch_delta_trbdf2.pdf")
  plot(priceCN.x,PPInterpolation.evaluateDerivative.(priceCN,priceCN.x),xlab="Underlying spot price", ylab="Option Δ",label="")
   plot!(xlim=(barrierLevelDown,barrierLevelUp),size=(400,200),margin=3Plots.mm)
 savefig("~//mypapers/eqd_book/doubleonetouch_delta_cn.pdf")
  plot(priceRAN.x,PPInterpolation.evaluateDerivative.(priceRAN,priceRAN.x),xlab="Underlying spot price", ylab="Option Δ",label="")
   plot!(xlim=(barrierLevelDown,barrierLevelUp),size=(400,200),margin=3Plots.mm)
 savefig("~//mypapers/eqd_book/doubleonetouch_delta_ran.pdf")
plot(priceRAN4.x,PPInterpolation.evaluateDerivative.(priceRAN4,priceRAN4.x),xlab="Underlying spot price", ylab="Option Δ",label="")
   plot!(xlim=(barrierLevelDown,barrierLevelUp),size=(400,200),margin=3Plots.mm)
 savefig("~//mypapers/eqd_book/doubleonetouch_delta_ran4.pdf")

   =#
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
    refPriceV = AQFED.FDM.priceTRBDF2(payoffV, spot, driftCurve, varianceSurface, discountCurve, dividends, M=16001, N=N, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    refPrice = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=16001, N=N, Smin=Smin, Smax=Smax, grid=SmoothlyDeformedGrid(UniformGrid(false)))
    println("reference KO ", refPrice(spot), " vanilla ", refPriceV(spot))
    α = 2.0 / (max -Smin)
    Ms = [251, 501, 1001, 2001]
    grids = [UniformGrid(false), ShiftedGrid(UniformGrid(false)), SmoothlyDeformedGrid(UniformGrid(false)), CubicGrid(α), SmoothlyDeformedGrid(CubicGrid(α)), SinhGrid(α), SmoothlyDeformedGrid(SinhGrid(α))]
    for M in Ms
        for grid in grids
            priceV = AQFED.FDM.priceTRBDF2(payoffV, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=Smin, Smax=Smax, grid=grid)(spot)
            priceKV = AQFED.FDM.priceTRBDF2(payoffKV, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=Smin, Smax=Smax, grid=grid)(spot)

            price = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=Smin, Smax=Smax, grid=grid)(spot)
            priceK = AQFED.FDM.priceTRBDF2(payoffK, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=Smin, Smax=Smax, grid=grid)(spot)
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
            priceV = AQFED.FDM.priceTRBDF2(payoff, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=localMin, Smax=Smax, grid=grid)(spot)
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
                priceKV = AQFED.FDM.priceTRBDF2(payoffK, spot, driftCurve, varianceSurface, discountCurve, dividends, M=M, N=N, Smin=localMin, Smax=Smax, grid=grid)(spot)
                push!(data, [M, string(keyname, "-", "Kreiss"), priceKV, priceKV - refPrice(spot)])
            end
        end
    end
    # udata = data[startswith.(data.grid,"Uniform"),:]
    # plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:log10,ylabel="Absolute error in price", xlabel="number of space-steps",size=(400,300))
    # savefig("~//mypapers/eqd_book/uniform_kreiss_time_vanilla.pdf")
    # udata = data[startswith.(data.grid,"Cubic"),:]
    # plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:log10,ylabel="Absolute error in price", xlabel="number of space-steps",size=(400,300))
    # savefig("~//mypapers/eqd_book/cubic_kreiss_time_vanilla.pdf")
    #plot(udata.m,abs.(udata.error), group=udata.grid,yscale=:ln,yticks=([0.0001,0.0002,0.0004],[0.0001,0.0002,0.0004]),ylabel="Absolute error in price", xlabel="number of time-steps",mode="markers")
end
