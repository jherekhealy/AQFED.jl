using AQFED, Test, StatsBase
using AQFED.Collocation, PPInterpolation, AQFED.Bachelier,AQFED.Basket
using LinearAlgebra

@testset "flat-basket" begin
vols = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
tte = 1.0
r=0.05
q=0.0
spot = 1.0
f = spot*exp((r-q)*tte)
strikes = [0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4].*f
discountFactor = exp(-r*tte)
prices, weights = Collocation.weightedPrices(true, strikes, vols, ones(length(vols)), f, discountFactor, tte, vegaFloor=1e-7)
bvols = Bachelier.bachelierImpliedVolatility.(prices,true,strikes,tte,f,discountFactor)
fitPrices = [Bachelier.bachelierFormula(true,strikei,f,bvols[i],tte,discountFactor) for (i,strikei)=enumerate(strikes)]
isapprox(prices,fitPrices,atol=1e-15) #check bvol algo is ok.

bvolSpline = CubicSplineNatural(log.(strikes./f),bvols)
guess = AQFED.VolatilityModels.initialGuessNormalATM(f,tte,0.0,bvolSpline(0.0),PPInterpolation.evaluateDerivative(bvolSpline,0.0),PPInterpolation.evaluateSecondDerivative(bvolSpline,0.0))
section = AQFED.VolatilityModels.calibrateNormalSABRSectionFromGuess(tte,f,strikes,bvols,ones(length(bvols)),guess)
fitBvols = sqrt.(AQFED.TermStructure.normalVarianceByMoneyness.(section, strikes .- f))
fitPrices = [Bachelier.bachelierFormula(true,strikei,f,fitBvols[i],tte,discountFactor) for (i,strikei)=enumerate(strikes)]
fitPrices-prices
#fit is amazingly good. SABR normal fits nearly exactly to Black.
#=
x=range(0.5,stop=2.0,length=201)
 isoc,m = Collocation.makeIsotonicCollocation(strikes,prices,weights,tte,f,discountFactor,deg=3,degGuess=1)
 sol = Collocation.Polynomial(isoc)
 cpricesx = Collocation.priceEuropean.(sol, true, x, f, discountFactor)
plot(x, @.(AQFED.Black.impliedVolatility(true,Bachelier.bachelierFormula(true, x, f, sqrt(AQFED.TermStructure.normalVarianceByMoneyness(section,x-f)),tte,discountFactor),f,x,tte,discountFactor)).*100 .- 20,label="Normal SABR")
plot!(xlab="Strike",ylab="Error in Black volatility %")
plot!(x,AQFED.Black.impliedVolatility.(true,cpricesx,f,x,tte,discountFactor).*100 .-20,label="Cubic collocation")
 plot!(ylims=(-0.02,0.04))
  plot!(size=(400,300))
#savefig("black_normal_sabr_collo.pdf")
=#
#Hagspihl kind of correlation test
nWeights=2
w = zeros(Float64, nWeights)
tvar = zeros(Float64, nWeights)
forward = zeros(Float64, nWeights)
spot=zeros(nWeights)
for i = eachindex(w)
    w[i] = 1.0 / (nWeights)
    tvar[i] = 0.2^2 * tte
    spot[i] = 1.0
    forward[i] = spot[i] * exp((r - q) * tte)
end
	rhoMin = -1.0
	rhoMax = 1.0
	rhoSteps = 50
sabrPrices = []
sabrIPrices = []
refPrices = []
rhos = range(rhoMin,stop=rhoMax,length=rhoSteps)
for rho=rhos
correlation = [
    1.0 rho 
    rho 1.0 
]
psiMatrix = [section.params.ρ section.params.ρ; section.params.ρ section.params.ρ]
correlationFull = [correlation psiMatrix; psiMatrix' ones((2,2))]
psiMatrixIndep = diagm([section.params.ρ, section.params.ρ])
correlationIndep = [correlation psiMatrixIndep; psiMatrixIndep' Matrix(1.0I,2,2)]
strike=1.1
refPrice = Basket.priceEuropean(Basket.QuadBasketPricer(AQFED.Math.GaussLegendre(64*2)), true, strike, discountFactor, spot, forward, tvar, w, correlation,isSplit=true)
price = Basket.priceEuropean(Basket.SABRBasketPricer{Basket.SingleStochasticVol}(),true,strike,discountFactor,spot,forward,[section,section],w,correlation)
priceF = Basket.priceEuropean(Basket.SABRBasketPricer{Basket.FullCorrelation}(),true,strike,discountFactor,spot,forward,[section,section],w,correlationFull)
priceI = Basket.priceEuropean(Basket.SABRBasketPricer{Basket.FullCorrelation}(),true,strike,discountFactor,spot,forward,[section,section],w,correlationIndep)
println(rho, " SABR ",price, " ",price-refPrice," ",price/refPrice-1," FULL ",priceF-refPrice," ",priceI-refPrice)
push!(refPrices,refPrice)
push!(sabrPrices,price)
push!(sabrIPrices, priceI)
end

#=
plot(rhos, abs.(sabrPrices./refPrices.-1), yscale=:log10, xlab="Correlation", ylab="Relative error")
=#

weight = [0.25, 0.25, 0.25, 0.25]
spot = [1.0, 1.0, 1.0, 1.0]
strike = 1.0
r = 0.0
q = 0.0
discountFactor = exp(-r*tte)
sigmas = range(0.02,stop=1.0,length=50)
rho = 0.5
tte = 2.0
f = exp((r-q)*tte)
p = DeelstraBasketPricer(3, 3)
sigmaFixed = 1.0
strikes = collect(range(exp(-sigmaFixed*0.5*sqrt(tte)),stop=exp(sigmaFixed*0.5*sqrt(tte)),length=10))
prices, weights = Collocation.weightedPrices(true, strikes, sigmaFixed .* ones(length(strikes)), ones(length(vols)), f, discountFactor, tte, vegaFloor=1e-7)
bvols = Bachelier.bachelierImpliedVolatility.(prices,true,strikes,tte,f,discountFactor)
bvolSpline = CubicSplineNatural(log.(strikes./f),bvols)
guess = AQFED.VolatilityModels.initialGuessNormalATM(f,tte,0.0,bvolSpline(0.0),PPInterpolation.evaluateDerivative(bvolSpline,0.0),PPInterpolation.evaluateSecondDerivative(bvolSpline,0.0))
sectionFixed = AQFED.VolatilityModels.calibrateNormalSABRSectionFromGuess(tte,f,strikes,bvols,ones(length(bvols)),guess)

for (i, sigma) in enumerate(sigmas)
    correlation = [
        1.0 rho rho rho
        rho 1.0 rho rho
        rho rho 1.0 rho
        rho rho rho 1.0
    ]
    tvar = [sigmaFixed^2, sigma^2, sigma^2, sigma^2] .* tte
    forward = spot .* exp((r - q) * tte)
    discountFactor = exp(-r * tte)
    strikes = collect(range(exp(-sigma*0.5*sqrt(tte)),stop=exp(sigma*0.5*sqrt(tte)),length=10))
    priceD = Basket.priceEuropean(p, true, strike, discountFactor, spot, forward, tvar, weight, correlation)
    priceJ = Basket.priceEuropean(JuBasketPricer(), true, strike, discountFactor, spot, forward, tvar, weight, correlation)
   
    prices, weights = Collocation.weightedPrices(true, strikes, sigma .* ones(length(strikes)), ones(length(vols)), f, 1.0, tte, vegaFloor=1e-7)
bvols = Bachelier.bachelierImpliedVolatility.(prices,true,strikes,tte,f,discountFactor)
bvolSpline = CubicSplineNatural(log.(strikes./f),bvols)
guess = AQFED.VolatilityModels.initialGuessNormalATM(f,tte,0.0,bvolSpline(0.0),PPInterpolation.evaluateDerivative(bvolSpline,0.0),PPInterpolation.evaluateSecondDerivative(bvolSpline,0.0))
section = AQFED.VolatilityModels.calibrateNormalSABRSectionFromGuess(tte,f,strikes,bvols,ones(length(bvols)),guess)
priceS = Basket.priceEuropean(Basket.SABRBasketPricer{Basket.SingleStochasticVol}(),true,strike,discountFactor,spot,forward,[sectionFixed, section, section,section],weight,correlation)
psiMatrix = [sectionFixed.params.ρ .* ones(1,4); section.params.ρ .* ones(1,4); section.params.ρ .* ones(1,4); section.params.ρ .* ones(1,4)]
correlationFull = [correlation psiMatrix; psiMatrix' ones((4,4))]
priceF = Basket.priceEuropean(Basket.SABRBasketPricer{Basket.FullCorrelation}(),true,strike,discountFactor,spot,forward,[sectionFixed, section, section,section],weight,correlationFull)
psiMatrixIndep = diagm([sectionFixed.params.ρ, section.params.ρ, section.params.ρ, section.params.ρ])
correlationIndep = [correlation psiMatrixIndep; psiMatrixIndep' Matrix(1.0I,4,4)]
priceI = Basket.priceEuropean(Basket.SABRBasketPricer{Basket.FullCorrelation}(),true,strike,discountFactor,spot,forward,[sectionFixed, section, section,section],weight,correlationIndep)
   
    @printf("%.2f %.4f %.4f %.4f %.4f %.2e %.2e %.2e\n", sigma, priceD,priceJ, priceS,  priceI, priceS-priceD, priceF-priceS, priceI-priceD)
end
end

@testset "basket-smile" begin
    #black-quad vs sabr-smile vs black-sabr. 
   #show impact of smile modeling on basket price ATM, ATM+25%. 

   #SPX1m
   strikes = Float64.([1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2325, 2350, 2375, 2400, 2425, 2450, 2470, 2475, 2480, 2490, 2510, 2520, 2530, 2540, 2550, 2560, 2570, 2575, 2580, 2590, 2600, 2610, 2615, 2620, 2625, 2630, 2635, 2640, 2645, 2650, 2655, 2660, 2665, 2670, 2675, 2680, 2685, 2690, 2695, 2700, 2705, 2710, 2715, 2720, 2725, 2730, 2735, 2740, 2745, 2750, 2755, 2760, 2765, 2770, 2775, 2780, 2785, 2790, 2795, 2800, 2805, 2810, 2815, 2835, 2860, 2900])
    vols = [0.684882717072609, 0.6548002174209514, 0.6279717042323061, 0.6040669049212617, 0.5769233835086068, 0.5512534351594732, 0.5260245499632258, 0.5004353919117, 0.4741366518169333, 0.46171589561249216, 0.4457089283432941, 0.4336614266663264, 0.420159764469498, 0.4074628373496824, 0.3931682390848574, 0.3814047904881801, 0.37929970817058073, 0.3771088224218263, 0.3724714977308359, 0.36029419336555424, 0.35467069448268806, 0.3505327949033959, 0.3441904382413214, 0.3392727917494692, 0.33306859556194446, 0.32820593458977093, 0.3243137942797042, 0.32204084870033645, 0.3168000315981532, 0.3109143207658998, 0.3050420836154825, 0.30241566311445206, 0.29948796266862154, 0.29609035936524486, 0.2923777072285143, 0.28951623883712746, 0.28584033838767425, 0.283342147794602, 0.2808533651372528, 0.27703523377755246, 0.27371493615870945, 0.2708906740100327, 0.2678887418986713, 0.2645328136650213, 0.26234402136468965, 0.2585977172018311, 0.25550003988953746, 0.2521896614376435, 0.2495339851370865, 0.24665927818229774, 0.24355339309186683, 0.24020198229067014, 0.23658800157061083, 0.23457380906338043, 0.23040670495884208, 0.2278656924642955, 0.22304945749920857, 0.21988751701341647, 0.2184983910827269, 0.21470247194448602, 0.21050624458263925, 0.20817463333507674, 0.20550780781621286, 0.19996741584940433, 0.19900703596491134, 0.19506177682405323, 0.19054732989021844, 0.18842657099566548, 0.18589260856179804, 0.18287835748424114, 0.17929170978590483, 0.17500063441150882, 0.18575101811296996, 0.20717302702012957, 0.22524785579801052]
    weightsV = [1.1789826122551597, 1.224744871391589, 1.284523257866513, 1.3601470508735443, 1.4317821063276355, 1.5165750888103102, 1.6124515496597098, 1.7175564037317668, 1.8395212376698413, 1.8973665961010275, 1.949358868961793, 2.024845673131659, 2.0976176963403033, 2.179449471770337, 2.258317958127243, 2.32379000772445, 2.345207879911715, 2.3664319132398464, 2.4083189157584592, 2.479919353527449, 2.5199206336708304, 2.569046515733026, 2.6076809620810595, 2.6551836094703507, 2.7625312572654126, 2.7477263328068173, 2.765863337187866, 2.7928480087537886, 2.871393034605969, 2.964704653791087, 3.0174928596261394, 3.0017001984929568, 3, 3.024896692450835, 3.120391338480345, 2.9916550603303182, 2.947349434130382, 2.8809720581775866, 2.8284271247461903, 2.8083087326973732, 2.711088342345192, 2.6685599339741506, 2.62445329583912, 2.565469285152568, 2.4899799195977463, 2.4289915602982237, 2.4279079146675357, 2.313006701244076, 2.258317958127243, 2.202271554554524, 2.1447610589527217, 2.085665361461421, 2.024845673131659, 1.97484176581315, 1.91049731745428, 1.857417562100671, 1.7888543819998317, 1.7320508075688772, 1.6881943016134133, 1.6278820596099706, 1.5652475842498528, 1.5165750888103102, 1.466287829861518, 1.3964240043768943, 1.3601470508735443, 1.3038404810405297, 1.2449899597988732, 1.2041594578792296, 1.161895003862225, 1.118033988749895, 1.0723805294763609, 1.02469507659596, 1.0099504938362078, 1.0910894511799618, 1.005037815259212]
 #  weightsV=ones(length(strikes))
    f = 2629.8026715608194
    tte = 0.0821917808219178
    logmoneynessA = log.(strikes ./ f)
    sumw2 = sum(weightsV .^ 2)
    w = weightsV ./ sqrt(sumw2)
    prices, weights = Collocation.weightedPrices(true, strikes, vols, w, f, 1.0, tte, vegaFloor=1e-7)
    bvols = Bachelier.bachelierImpliedVolatility.(prices,true,strikes,tte,f,1.0)  
    bvolSpline = CubicSplineNatural(log.(strikes./f),bvols)
    guess = AQFED.VolatilityModels.initialGuessNormalATM(f,tte,0.0,bvolSpline(0.0),PPInterpolation.evaluateDerivative(bvolSpline,0.0),PPInterpolation.evaluateSecondDerivative(bvolSpline,0.0))
    sectionSPX = AQFED.VolatilityModels.calibrateNormalSABRSectionFromGuess(tte,f,strikes,bvols,w,guess)
     
    tte2 = 0.095890
    f2 = 357.755926
    strikes2 = Float64.([150, 155, 160, 165, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 500, 520, 540, 560, 580])
    vols2 = [1.027152094560499, 0.9905195749900226, 0.9657262376591365, 0.9405597986379826, 0.9181603362313814, 0.9019382314978117, 0.8846745842549402, 0.865754243981787, 0.8456155492434201, 0.8245634579529838, 0.8028174604214972, 0.78053958851195, 0.7636802684802435, 0.7454192306685303, 0.7260651215584285, 0.7058414693439228, 0.6849143304434797, 0.663409356115238, 0.6462309799739909, 0.6301291739261891, 0.6130540004186168, 0.5946923076348443, 0.5811921286363728, 0.5687314890047378, 0.5539815904720001, 0.5422671292669776, 0.5338887990387771, 0.5234154661207794, 0.5168510552270313, 0.5072806473672073, 0.4997973159961656, 0.4896563997378466, 0.4823975850368014, 0.47936818364069134, 0.48000585384055006, 0.4757525564073338, 0.4711478482467228, 0.46788352167691083, 0.46562175169660713, 0.46299652559206567, 0.45939930288424485, 0.458565105643866, 0.45790487479637937, 0.45521398441321903, 0.453447302139774, 0.4504013827012644, 0.448004721643358, 0.4491995553643971, 0.4478840707248649, 0.45006593113797866, 0.4517530880150887, 0.4499007489879635, 0.448814967685824, 0.45160477568536983, 0.4563938928347205, 0.4600222064217672, 0.46102443173801966, 0.46406468170261544, 0.4709795491400157, 0.4762595045128011, 0.4810009989573377, 0.4855906965577297, 0.4906446878461756, 0.4960612773473766, 0.5011170526132832, 0.5059204240563133, 0.5159102206249263, 0.5505625146941026, 0.5783881966646062, 0.599260903580561, 0.6259792014943735]
    weightsV2 = [1.7320508075688772, 1, 1.224744871391589, 1, 2.738612787525831, 1.558387444947959, 1.9999999999999998, 1.2602520756252087, 1.3301243435223526, 2.273030282830976, 1.3944333775567928, 1.2089410496539776, 1.9999999999999998, 2.0976176963403033, 3.500000000000001, 3.286335345030995, 2.6692695630078282, 2.7838821814150116, 3.1622776601683804, 3.605551275463988, 3.3541019662496834, 3, 2.9742484506432634, 3.6469165057620923, 3.8729833462074152, 4.183300132670376, 3.7505555144093887, 4.1918287860346295, 3.7670248460125917, 4.795831523312714, 4.527692569068711, 3.482097069296032, 3.2333489534143167, 3.687817782917155, 6.3245553203367555, 6.837397165588683, 7.365459931328131, 7.0992957397195395, 7.628892449104261, 7.461009761866454, 8.706319543871567, 8.78635305459552, 7.000000000000021, 7.745966692414834, 8.093207028119338, 6.16441400296897, 4.974937185533098, 4.650268809434567, 4.315669125408015, 4.636809247747854, 4.732863826479693, 3.1144823004794873, 2.8809720581775857, 2.8284271247461894, 2.7718093060793882, 4.092676385936223, 2.7041634565979926, 2.652259934210953, 3.710691413905333, 3.777926319123662, 3.929942040850535, 3.921096785339529, 3.70809924354783, 3.517811819867573, 3.3354160160315844, 3.1622776601683777, 1.3483997249264843, 1.8929694486000912, 1.914854215512676, 1.699673171197595, 1.8708286933869707]
    #note the weights are not good they are in rel price
    vols2 = vols2 .* sqrt.(tte2./tte)
    sumw2 = sum(weightsV2 .^ 2)
    w = weightsV2 ./ sqrt(sumw2)
    prices2, weights2 = Collocation.weightedPrices(true, strikes2, vols2, w, f2, 1.0, tte, vegaFloor=1e-7)
    bvols = Bachelier.bachelierImpliedVolatility.(prices2,true,strikes2,tte,f2,1.0)  
    bvolSpline = CubicSplineNatural(log.(strikes2./f2),bvols)
    guess = AQFED.VolatilityModels.initialGuessNormalATM(f2,tte,0.0,bvolSpline(0.0),PPInterpolation.evaluateDerivative(bvolSpline,0.0),PPInterpolation.evaluateSecondDerivative(bvolSpline,0.0))
    sectionTSLA = AQFED.VolatilityModels.calibrateNormalSABRSectionFromGuess(tte,f2,strikes2,bvols,w,guess)
   
    rho = 0.5
    weights = [0.5/f, 0.5/f2]
    spot = [f, f2]
    forward = [f, f2]
    discountFactor = 1.0
    correlation = [
    1.0 rho 
    rho 1.0 
]
volATM1 = AQFED.Black.impliedVolatility(true,Bachelier.bachelierFormula(true, f, f, sqrt(AQFED.TermStructure.normalVarianceByMoneyness(sectionSPX,0.0)),tte,discountFactor),f,f,tte,discountFactor)
volATM2 = AQFED.Black.impliedVolatility(true,Bachelier.bachelierFormula(true, f2, f2, sqrt(AQFED.TermStructure.normalVarianceByMoneyness(sectionTSLA,0.0)),tte,discountFactor),f2,f2,tte,discountFactor)
tvarATM = [volATM1^2 * tte, volATM2^2 * tte]
prices10, weights10 = Collocation.weightedPrices(true, strikes, volATM1 .* ones(length(strikes)), ones(length(vols)), f, 1.0, tte, vegaFloor=1e-7)
bvols10 = Bachelier.bachelierImpliedVolatility.(prices10,true,strikes,tte,f,1.0)
bvolSpline10 = CubicSplineNatural(log.(strikes./f),bvols10)
guess = AQFED.VolatilityModels.initialGuessNormalATM(f,tte,0.0,bvolSpline10(0.0),PPInterpolation.evaluateDerivative(bvolSpline10,0.0),PPInterpolation.evaluateSecondDerivative(bvolSpline10,0.0))
section10 = AQFED.VolatilityModels.calibrateNormalSABRSectionFromGuess(tte,f,strikes,bvols10,ones(length(bvols10)),guess)
prices20, weights20 = Collocation.weightedPrices(true, strikes2, volATM2 .* ones(length(strikes2)), ones(length(vols2)), f2, 1.0, tte, vegaFloor=1e-7)
bvols20 = Bachelier.bachelierImpliedVolatility.(prices20,true,strikes2,tte,f2,1.0)
bvolSpline20 = CubicSplineNatural(log.(strikes2./f2),bvols20)
guess = AQFED.VolatilityModels.initialGuessNormalATM(f2,tte,0.0,bvolSpline20(0.0),PPInterpolation.evaluateDerivative(bvolSpline20,0.0),PPInterpolation.evaluateSecondDerivative(bvolSpline20,0.0))
section20 = AQFED.VolatilityModels.calibrateNormalSABRSectionFromGuess(tte,f2,strikes2,bvols20,ones(length(bvols20)),guess)

priceBlack = []
priceBlackATM = []
priceSABR = []
priceSABRBlack = []
bstrikes = range(0.75,stop=1.25,length=51)
for strike = bstrikes
    isCall = strike > 1.0
refPrice = Basket.priceEuropean(Basket.QuadBasketPricer(AQFED.Math.GaussLegendre(64*2)), isCall, strike, discountFactor, spot, forward, tvarATM, weights, correlation,isSplit=true)
refPriceB = Basket.priceEuropean(Basket.QuadBasketPricer(AQFED.Math.GaussLegendre(64*2)), isCall, strike, discountFactor, spot, forward, tvar, weights, correlation,isSplit=true)
vol1 = AQFED.Black.impliedVolatility(true,Bachelier.bachelierFormula(true, f*strike, f, sqrt(AQFED.TermStructure.normalVarianceByMoneyness(sectionSPX,f-f*strike)),tte,discountFactor),f,f*strike,tte,discountFactor)
vol2 = AQFED.Black.impliedVolatility(true,Bachelier.bachelierFormula(true, f2*strike, f2, sqrt(AQFED.TermStructure.normalVarianceByMoneyness(sectionTSLA,f2-f2*strike)),tte,discountFactor),f2,f2*strike,tte,discountFactor)
tvar = [vol1^2 * tte, vol2^2 * tte]
price = Basket.priceEuropean(Basket.SABRBasketPricer{Basket.SingleStochasticVol}(),isCall,strike,discountFactor,spot,forward,[sectionSPX,sectionTSLA],weights,correlation)


priceS = Basket.priceEuropean(Basket.SABRBasketPricer{Basket.SingleStochasticVol}(),isCall,strike,discountFactor,spot,forward,[section10, section20],weights,correlation)
 

println(strike," ",refPrice, " ",price," ",price-refPrice," ",price/refPrice-1)
push!(priceBlackATM,refPrice)
push!(priceBlack,refPriceB)
push!(priceSABR,price)
push!(priceSABRBlack,priceS)
end

    #=
    ## fit of sabr
    x = range(strikes[1],stop=strikes[end],length=201)
     fitPricesx = [Bachelier.bachelierFormula(true,strikei,f,sqrt(AQFED.TermStructure.normalVarianceByMoneyness(sectionSPX, strikei .- f)),tte,1.0) for (i,strikei)=enumerate(x)]
 plot(strikes./f,vols.*100,seriestype=:scatter,xlab="Strike", ylab="Volatility in %",ms=4,markerstrokewidth=0,markeralpha=0.7,label="Market")
 plot!(x./f,abs.(AQFED.Black.impliedVolatility.(true, fitPricesx, f, x, tte, 1.0)).*100,xlab="Strike/Forward",label="Normal SABR")
 plot!(size=(400,300),margins=1Plots.mm)
    x = range(strikes2[1],stop=strikes2[end],length=201)
     fitPricesx = [Bachelier.bachelierFormula(true,strikei,f2,sqrt(AQFED.TermStructure.normalVarianceByMoneyness(sectionTSLA, strikei .- f2)),tte,1.0) for (i,strikei)=enumerate(x)]
 plot(strikes2./f2,vols2.*100,seriestype=:scatter,xlab="Strike", ylab="Volatility in %",ms=4,markerstrokewidth=0,markeralpha=0.7,label="Market")
 plot!(x./f2,abs.(AQFED.Black.impliedVolatility.(true, fitPricesx, f2, x, tte, 1.0)).*100,xlab="Strike/Forward",label="Normal SABR")
 plot!(size=(400,300),margins=1Plots.mm)
 savefig("normal_sabr_tsla_1m.pdf")

## relative error in OTM prices
plot(bstrikes,abs.(priceSABR./priceBlack .- 1).*100,yscale=:log10,yticks=([0.1,1,10,100],["0.1%","1%","10%","100%"]),label="SABR / Black")
plot!(bstrikes,abs.(priceSABR./priceBlackATM .- 1).*100,label="SABR / Black ATM",xlab="Basket strike", ylab="Relative difference")
plot!(bstrikes,abs.(priceSABRBlack./priceBlackATM .- 1).*100,label="SABR Flat ATM / Black ATM")
 plot!(size=(800,300),margins=3Plots.mm)
savefig("normal_sabr_basket_1m.pdf")

=#
end