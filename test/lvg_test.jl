using AQFED, Test, StatsBase
using AQFED.PDDE, AQFED.Black, AQFED.Collocation

@testset "lvg-flatblack-c3" begin
strikes = [0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4]
vols = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
sigmas = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
l = 0.25
u = 4.0
tte = 0.25
forward = 1.025
#weights = ones(length(vols))
prices, weights = Collocation.weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor=1e-7)
lvg = PDDE.calibrateLinearBlackLVG(tte, forward, strikes, prices, weights, useVol=false, L=l, U=u)
ivkLVG = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvg, true, strikes), forward, strikes, tte, 1.0)
rmseLVG = StatsBase.rmsd(vols, ivkLVG)
@test isapprox(0.0,rmseLVG,atol=1e-8)
lvgq = PDDE.calibrateEQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, L=l, U=u,model="Quadratic")
ivkLVGq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
rmseLVGq = StatsBase.rmsd(vols, ivkLVGq)
@test isapprox(0.0,rmseLVGq,atol=1e-8)
lvgqc3 = PDDE.calibrateEQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, L=l, U=u,model="Quadratic",location="Mid-XX")
ivkLVGq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgqc3, true, strikes), forward, strikes, tte, 1.0)
rmseLVGq = StatsBase.rmsd(vols, ivkLVGq)
@test isapprox(0.0,rmseLVGq,atol=1e-8)
lvgqnotc3 = PDDE.calibrateEQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, L=l, U=u,model="Quadratic",isC3=false,location="Mid-XX")
ivkLVGq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgqnotc3, true, strikes), forward, strikes, tte, 1.0)
rmseLVGq = StatsBase.rmsd(vols, ivkLVGq)
@test isapprox(0.0,rmseLVGq,atol=1e-8)
#=
kFine = range(0.8,stop=strikes[end],length=501)
plot(kFine, @.(AQFED.Math.normpdf((log(forward/kFine)-0.5*vols[1]^2*tte)/(vols[1]*sqrt(tte)) )/(kFine*vols[1]*sqrt(tte)) ),xlab="Strike",ylab="Probabiliy density",label="Lognormal",xticks=(vcat(kFine[1]:0.1:kFine[end], forward),vcat(string.(kFine[1]:0.1:kFine[end]),"F")),linestyle=:dot)
 plot!(kFine,(PDDE.derivativePrice.(lvgqnotc3,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgqnotc3,true,kFine)).*10000, label="C1 quadratic B-spline")
  plot!(kFine,(PDDE.derivativePrice.(lvgqc3,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgqc3,true,kFine)).*10000, label="Quadratic B-Spline with C3 condition")

  plot(kFine, @.(normpdf((log(forward/kFine)-0.5*vols[1]^2*tte)/(vols[1]*sqrt(tte)) )/(kFine*vols[1]*sqrt(tte)) ),xlab="Strike",ylab="Probabiliy density",label="Lognormal",xticks=(vcat(kFine[1]:0.1:kFine[end], forward),vcat(string.(kFine[1]:0.1:kFine[end]),"F")))
 plot!(kFine,(PDDE.derivativePrice.(lvgqnotc3,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgqnotc3,true,kFine)).*10000, label="C1 quadratic B-spline")
  plot!(kFine,(PDDE.derivativePrice.(lvgqc3,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgqc3,true,kFine)).*10000, label="Quadratic B-Spline with C3 condition")


=#

strikes=round.([95.25466637597646,
  99.74155408701951,
 108.83682736543783,
 116.84297587845171,
 121.51717213620066,
 122.68847999646283,
 126.01033716036022,
 133.4592851856828,
 133.82546157127993,
 134.5683605635301],digits=2)
 forward = 101.0
 #weights = ones(length(vols))
 prices, weights = Collocation.weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor=1e-7)
 locations = ["Strikes","Mid-X","Mid-XX","Mid-Strikes","Uniform"]
 lvgq = PDDE.calibrateEQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, L=forward/3, U=forward*3,model="Quadratic",location="Strikes")
 ivkLVGq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
 rmseLVGq = StatsBase.rmsd(vols, ivkLVGq)
 @test isapprox(0.0,rmseLVGq,atol=1e-8)
#= kFine = range(strikes[1]/1.25,stop=strikes[end]*1.25,length=401)
 p1=plot(kFine, @.(AQFED.Math.normpdf((log(forward/kFine)-0.5*vols[1]^2*tte)/(vols[1]*sqrt(tte)) )/(kFine*vols[1]*sqrt(tte)) ),xlab="Strike",ylab="Probabiliy density",label="Lognormal",linestyle=:dot)
for location in locations
 lvgq = PDDE.calibrateEQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, L=forward/3, U=forward*3,model="Quadratic",location=location)
 ivkLVGq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
 rmseLVGq = StatsBase.rmsd(vols, ivkLVGq)
 println(location," ",rmseLVGq)
 plot!(p1, kFine,(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label=location)
end
plot(p1)
 plot!(p1,size=(480,320))
savefig("/home/fabien/mypapers/eqd_book/lvgq_lognormal_density_set_D.pdf")


 =#  
 #some very close by
 strikes = sort(rand(length(vols)))*(140-85) .+ 85
 strikes=round.([87.0653770852329, 92.55057909409112, 92.55584927988729, 93.10805223968916, 98.65030015379912, 104.5260453270286, 113.7778097399039, 124.52668961177353, 124.90596814653736, 128.21267359110428],digits=2)
#similar to first - best is midXX. mid should be excluded too much noise - set A
strikesA= round.([88.765135169592, 92.85494511181187, 93.38043933991867, 99.36630953103418, 107.9930908363475, 120.28927001531655, 122.03374332554367, 123.89923835687514, 134.7053265024023, 135.43198727132489],digits=2)
#right wing info only
strikesB=round.([85.02320730206546, 101.92119620448395, 103.54923033678908, 114.44768735553629, 121.8544889307987, 123.69290176127272, 125.07479849838776, 125.57703562269575, 131.63046140652378, 133.85728148034934],digits=2)
#forward close to strike === use "Strikes" is bad. As we remove the closest two points, not an issue for "MidX" and "MidXX".
strikesC=round.([98.06881859879773, 100.92597117414611, 101.05812725277457, 106.87903884044188, 109.1197408201026, 110.92917871923213, 119.7631067321654, 119.82704376084504, 132.1932799832046, 138.26522048510896],digits=2)

strikes=round.([ 89.41370489828515,
90.99359147310548,
95.34096598232097,
98.4081807678219,
114.78124668457718,
115.83300408171641,
124.09477093886409,
135.8728166397164,
137.82553133115383,
139.10235889326142],digits=2)
#strikes and midxx lead to perfect calib if forward not in strikes
#now add forward and check again
strikes = sort(vcat(forward,strikes[1:end-1]))

strikesD = [0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4].*100
strikesD = sort(vcat(forward,strikesD[1:end-1]))

strikesD = [0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4].*100
strikesD = sort(vcat(forward,strikesD[1:end-1]))

for (name,strikes) in zip(["A","B,","C","D"],[strikesA,strikesB,strikesC,strikesD])
    prices, weights = Collocation.weightedPrices(true, strikes, vols, ones(length(vols)), forward, 1.0, tte, vegaFloor=1e-7)
    p1=plot(kFine, @.(AQFED.Math.normpdf((log(forward/kFine)-0.5*vols[1]^2*tte)/(vols[1]*sqrt(tte)) )/(kFine*vols[1]*sqrt(tte)) ),xlab="Strike",ylab="Probabiliy density",label="Lognormal",linestyle=:dot)
    for location in locations
     lvgq = PDDE.calibrateEQuadraticLVG(tte, forward, strikes, prices, weights, useVol=false, L=forward/3, U=forward*3,model="Quadratic",location=location)
     ivkLVGq = @. Black.impliedVolatility(true, PDDE.priceEuropean(lvgq, true, strikes), forward, strikes, tte, 1.0)
     rmseLVGq = StatsBase.rmsd(vols, ivkLVGq)
     println(name," ",location," ",rmseLVGq)
      plot!(p1, kFine,(PDDE.derivativePrice.(lvgq,true,kFine.+0.0001) .- PDDE.derivativePrice.(lvgq,true,kFine)).*10000, label=location)
    end
     plot!(p1,xlims=(80,140))
      plot!(p1,size=(480,320))
      savefig(p1,string("/home/fabien/mypapers/eqd_book/lvgq_lognormal_density_set_",name,".pdf"))
 #best fit is MidXX
end
