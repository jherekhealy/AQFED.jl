rmsd(pricesA1 ./ pricesA3, ones(length(pricesA1)))
2.6340919922262273e-6

maxad(pricesA1 ./ pricesA3, ones(length(pricesA1)))
0.0002718939332517767

maxad(pricesA1 , pricesA3)
1.8059589060648307e-5

# a=kappa
# b=theta
# c=sigma
# -ia *( ia -i)   =  a*(a+1).    If a = -0.5 => negative

# 175
# 156
# RandomNumberGeneratorDoubleNextPerformance.nextMixmax17      1000000  thrpt  400  0.190 ± 0.003  ops/ms
# RandomNumberGeneratorDoubleNextPerformance.nextMixmax240     1000000  thrpt  400  0.160 ± 0.003  ops/ms

# Go vs Julia on implied volatilities

# Julia functions:
#  memory estimate:  80 bytes
#   allocs estimate:  5
#   --------------
#   minimum time:     794.615 ns (0.00% GC)
#   median time:      827.813 ns (0.00% GC)
#   mean time:        858.378 ns (0.07% GC)
#   maximum time:     7.238 μs (87.60% GC)
#   --------------
#   samples:          10000
#   evals/sample:     96


# Cody + AS
# BenchmarkTools.Trial: 
#   memory estimate:  80 bytes
#   allocs estimate:  5
#   --------------
#   minimum time:     837.521 ns (0.00% GC)
#   median time:      874.890 ns (0.00% GC)
#   mean time:        908.164 ns (0.12% GC)
#   maximum time:     11.647 μs (92.30% GC)
#   --------------
#   samples:          10000
#   evals/sample:     73

# golang: 1300860	       919 ns/op


# => Julia SpecialFunctions is faster *and* more accurate/consistent.
# erfcx more accurate. Perhaps we test only 1 branch here.
# actually depends on the branch taken, sometimes, cody is as fast or a few percent faster.

# for vol=0.08 and strike*=2, jaeckel becomes much faster?

# Why does this high accuracy matter? Market option quotes will never be so small, and it may appear silly to spend much effort on this problem. This misses however one main use: to fit a model in terms of implied volatilities. In order to calibrate a volatility model to market quotes, the model may produce very small prices, which need to be inverted. Very small prices are relatively common in the wings, when the model does not yet fit well the market. Furthermore, a poor implied volatility accuracy may lead the optimization towards a suboptimal solution, by creating artificially some local minima, for example because of noisy numerical gradients. A high accuracy implied volatility solver alleviates all those issues.

# key to accelerate is decomposition with erfcx. This makes a sizeable difference (40%)
# householder vs halley not big difference, more sizable very otm(10%). CMethod preferable as more stable.
# lisordr simple, effective, much better far strikes than sorts

# alanPrice(150.0,0.02)
# 1.3316709837414707e-92
# probably should not depend too much on such small values. More important to be accurate up to eps accuracy for not too small prices. 
# Corner cases.

hTimes = Float64[]
shTimes = Float64[]
cTimes = Float64[]
iTimes = Float64[]
jTimes = Float64[]
lTimes = Float64[]
lrTimes = Float64[]
hhTimes = Float64[]
#vols = collect(0.02:0.01:0.64)
vols = collect(0.02:0.02:1.0)
for voli in vols
global strike = 200.0
         global vol=voli
  t = @benchmark halleyIVAlan(strike, x,Black.Halley()) setup=(x=alanPrice(strike,vol))
  append!(hTimes, median(t.times))
  println(vol, " Halley ", median(t.times))
  t = @benchmark halleyIVAlan(strike, x,Black.CMethod()) setup=(x=alanPrice(strike,vol))
   append!(cTimes, median(t.times))
    println(vol, " CMethod ", median(t.times))
  t = @benchmark halleyIVAlan(strike, x,Black.InverseQuadratic()) setup=(x=alanPrice(strike,vol))
    append!(iTimes, median(t.times))
println(vol, " InverseQuadratic ", median(t.times))
  t = @benchmark jaeckelIVAlan(strike, x) setup=(x=alanPrice(strike,vol))
    append!(jTimes, median(t.times))
  println(vol, " Jaeckel ", median(t.times))
  t = @benchmark householderIVAlan(strike, x) setup=(x=alanPrice(strike,vol))
  append!(hhTimes, median(t.times))
    println(vol, " Householder ", median(t.times))
end


  t = @benchmark liIVAlan(strike, x,Black.SORTS()) setup=(x=alanPrice(strike,vol))
    append!(lTimes, median(t.times))
  println(vol, " Li-SORTS ", median(t.times))
    t = @benchmark liIVAlan(strike, x,Black.SORDR()) setup=(x=alanPrice(strike,vol))
    append!(lrTimes, median(t.times))
  println(vol, " Li-SORDR ", median(t.times))

plotd = Plots.plot(vols, jTimes, foreground_color_border=:white, label="Jäckel")
Plots.plot!(vols, hTimes, foreground_color_border=:white, label="Halley",ylab="Time (ns)", xlab="Implied volatility")
#Plots.plot!(vols, lTimes, foreground_color_border=:white, label="Li-SORTS")
#Plots.plot!(vols, cTimes, foreground_color_border=:white, label="CMethod")
#Plots.plot!(vols, iTimes, foreground_color_border=:white, label="InverseQuadratic")
Plots.plot!(vols, hhTimes, foreground_color_border=:white, label="Householder")
#Plots.plot!(vols, lrTimes, foreground_color_border=:white, label="Li-SORDR")
Plots.plot!([0.085], seriestype="vline", color=:black, xticks=(vcat(0.0:0.25:1.0, 0.085),vcat(0.0:0.25:1.0,"ε")),label=:none)





@benchmark Black.impliedVolatilitySRGuess(true, x, 100.0, strike, 1.0, 1.0) setup=(x=Black.blackScholesFormula(true, strike, 100.0, vol * vol * 1.0, 1.0, 1.0))
@benchmark Black.impliedVolatilitySRHouseholder(true, x, 100.0, strike, 1.0, 1.0,0.0,64) setup=(x=Black.blackScholesFormula(true, strike, 100.0, vol * vol * 1.0, 1.0, 1.0))
@benchmark Black.impliedVolatilityJaeckel(true, x, 100.0, strike, 1.0, 1.0) setup=(x=Black.blackScholesFormula(true, strike, 100.0, vol * vol * 1.0, 1.0, 1.0))

hTimes = Float64[]
shTimes = Float64[]
cTimes = Float64[]
iTimes = Float64[]
jTimes = Float64[]
lTimes = Float64[]
lrTimes = Float64[]
hhTimes = Float64[]
 
  strikes = collect(100.0:20.0:500.0)

  for strikei in strikes
  global strike =strikei
           global vol=0.1
    t = @benchmark halleyIVAlan(strike, x,Black.Halley()) setup=(x=Black.blackScholesFormula(true, strike, 100.0, vol * vol * 1.0, 1.0, 1.0))
    append!(hTimes, median(t.times))
    println(strike, " Halley ", median(t.times))
    t = @benchmark halleyIVAlan(strike, x,Black.InverseQuadratic()) setup=(x=alanPrice(strike,vol))
      append!(iTimes, median(t.times))
  println(strike, " InverseQuadratic ", median(t.times))
     t = @benchmark householderIVAlan(strike, x) setup=(x=alanPrice(strike,vol))
      append!(hhTimes, median(t.times))
  println(strike, " Householder ", median(t.times))
    t = @benchmark jaeckelIVAlan(strike, x) setup=(x=alanPrice(strike,vol))
      append!(jTimes, median(t.times))
    println(strike, " Jaeckel ", median(t.times))
  end

    t = @benchmark halleyIVAlan(strike, x,Black.CMethod()) setup=(x=alanPrice(strike,vol))
     append!(cTimes, median(t.times))
    println(strike, " CMethod ", median(t.times))
    t = @benchmark liIVAlan(strike, x,Black.SORTS()) setup=(x=alanPrice(strike,vol))
      append!(lTimes, median(t.times))
    println(strike, " Li-SORTS ", median(t.times))
      t = @benchmark liIVAlan(strike, x,Black.SORDR()) setup=(x=alanPrice(strike,vol))
      append!(lrTimes, median(t.times))
    println(strike, " Li-SORDR ", median(t.times))

plotd = Plots.plot(strikes, jTimes, foreground_color_border=:white, label="Jäckel")
Plots.plot!(strikes, hTimes, foreground_color_border=:white, label="Halley",ylab="Time (ns)", xlab="Strikes")
Plots.plot!(strikes, lTimes, foreground_color_border=:white, label="Li-SORTS")

Plots.plot!(strikes, cTimes, foreground_color_border=:white, label="CMethod")
Plots.plot!(strikes, iTimes, foreground_color_border=:white, label="InverseQuadratic")
Plots.plot!(strikes, lrTimes, foreground_color_border=:white, label="Li-SORDR")
Plots.plot!(strikes, hhTimes, foreground_color_border=:white, label="Householder")
Plots.plot!([227.0], seriestype="vline", color=:black, xticks=(vcat(100:80:500, 227),vcat(100:80:500,"ε")),label=:none)

Plots.plot!(ylims=(0,1000))
lrrTimes = Float64[]
for strikei in strikes
 global vol = 0.1
         global strike=strikei
    t = @benchmark liIVAlan(strike, x,Black.SORDR()) setup=(x=alanPrice(strike,vol))
    append!(lrrTimes, median(t.times))
  println(strike, " Li-SORDR ", median(t.times))
end


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
    mvols[j]= Black.impliedVolatility(isCall, price, forward, strike, tte, 1.0)
    mprices[j] = Black.blackScholesFormula(isCall, strike, forward, mvols[j]^2 * tte, 1.0, 1.0)
    # println(mprice-price, " ",price)
  end
  println("JAECKEL RMSD ",rmsd(mvols,vols)," MAXAD ",maxad(mvols,vols)," PriceMAXAD ",maxad(mprices,prices), " ",maxad(mprices ./ prices,prices ./ prices))
jvols=copy(mvols)
jprices=copy(mprices)
plotd=Plots.plot(vols, jvols-vols, foreground_color_border=:white, label="Jäckel",ylab="Error", xlab="Implied volatility")
Plots.plot!(vols, mvols-vols, foreground_color_border=:white, label="Householder")
ticks = [-2e-15,0,2e-15, 8e-15]
Plots.plot!(yticks=ticks)
savefig(plotd,"/home/fabien/mypapers/eqd_book/iv_strike200_vols_error.pdf")


##### plots on data in txt
using DelimitedFiles, DataFrames, Plots 
theme(:vibrant)

m = readdlm("/home/fabien/mypapers/eqd_book/aapl170120_160203_euro_vol.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
plot(df.Strike,(df.Call).*100,margin=3Plots.mm,xlabel="Strike",ylabel="Implied Volatility in %",label="Call",ylim=(25.5,50))
plot!(df.Strike,df.Put .* 100,label="Put",size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/aapl170120_160203_euro_vol.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/aapl170120_160203_am_yield_vol_tr.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
plot(df.Strike,(df.CallVol).*100,margin=3Plots.mm,xlabel="Strike",ylabel="Implied Volatility in %",label="Call",ylim=(25.5,50))
plot!(df.Strike,df.PutVol .* 100,label="Put",size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/aapl170120_160203_am_yield.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/aapl170120_160203_am_cash_vol.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
plot(df.Strike,(df.Call).*100,margin=3Plots.mm,xlabel="Strike",ylabel="Implied Volatility in %",label="Call",ylim=(25.5,50))
plot!(df.Strike,df.Put .* 100,label="Put",size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/aapl170120_160203_am_cash_vol.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/aapl170120_160203_am_prop_vol.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
plot(df.Strike,(df.Call).*100,margin=3Plots.mm,xlabel="Strike",ylabel="Implied Volatility in %",label="Call",ylim=(25.5,50))
plot!(df.Strike,df.Put .* 100,label="Put",size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/aapl170120_160203_am_prop_vol.pdf")

df[!,:Vol] = df[!,:Call]
df[df.Strike .< 95.0, :Vol] = df[df.Strike .< 95.0, :Put] 
plot(df.Strike,(df.CallAsk).*100,margin=3Plots.mm,xlabel="Strike",ylabel="Implied Volatility in %",label="CallAsk",ylim=(16,50))
plot!(df.Strike,df.CallBid .* 100,label="CallBid",size=(400,400))
plot!(df.Strike,df.Vol .* 100, label="OTM")
savefig("/home/fabien/mypapers/eqd_book/aapl170120_160203_am_prop_vol_call_otm.pdf")
plot(df.Strike,(df.PutAsk).*100,margin=3Plots.mm,xlabel="Strike",ylabel="Implied Volatility in %",label="PutAsk",ylim=(16,50))
plot!(df.Strike,df.PutBid .* 100,label="PutBid",size=(400,400))
plot!(df.Strike,df.Vol .* 100, label="OTM")
savefig("/home/fabien/mypapers/eqd_book/aapl170120_160203_am_prop_vol_put_otm.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/aapl_160304_160212_am_prop_vol.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
m = readdlm("/home/fabien/mypapers/eqd_book/aapl_160304_160212_am_prop_vol_parabola.txt", ' ',header=true)
df2 = DataFrame(m[1], vec(m[2]))
n = df2[df2.Method .== "Strike" .|| df2.Method .== "Logmoneyness",:]
plot(df.Strike,df.Vol .* 100,label="Mid",size=(400,400),seriestype = :scatter,markersize=3,ylim=(29,50.0),markerstrokewidth=-1)
plot!(n.Strike,(n.Vol).*100,margin=3Plots.mm,xlabel="Strike",ylabel="Implied Volatility in %",group=n.Method)
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_parabola.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/aapl_160304_160212_am_prop_vol_cubic.txt", ' ',header=true)
df2 = DataFrame(m[1], vec(m[2]))
n = df2[df2.Method .== "Parabola" .|| df2.Method .== "Spline3",:]
plot(df.Strike,df.Vol .* 100,label="Mid",size=(400,400),seriestype = :scatter,markersize=3,ylim=(29,50.0),markerstrokewidth=-1)
plot!(n.Strike,(n.Vol).*100,margin=3Plots.mm,xlabel="Strike",ylabel="Implied Volatility in %",group=n.Method)
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_spline.pdf")

weights = [3.0, 5.0, 6.928203230275509, 5.291502622129181, 3.3166247903554, 4.898979485566356, 5.0, 3.7416573867739413, 12.68857754044952, 3.4641016151377544, 5.830951894845301, 7.483314773547883, 12.409673645990857, 5, 11.445523142259598, 6.244997998398398, 10.392304845413264, 5.291502622129181, 5.385164807134504, 2, 7.745966692414834, 6.48074069840786, 19.313207915827967, 1.4142135623730951, 11.445523142259598, 6.164414002968976, 10.04987562112089, 4.358898943540674, 5, 5.916079783099616, 10.908712114635714, 5.196152422706632, 31.796226191169293, 14.071247279470288, 6.6332495807108, 4.898979485566356, 3.1622776601683795, 5.477225575051661, 2.8284271247461903, 6.324555320336759, 1.7320508075688772]
nweights = weights / sum(weights) * length(weights)
m = readdlm("/home/fabien/mypapers/eqd_book/aapl_160304_160212_am_prop_vol_quartic.txt", ' ',header=true)
df2 = DataFrame(m[1], vec(m[2]))
n = df2[df2.Method .== "Quartic" .|| df2.Method .== "Spline3" .|| df2.Method .== "RBF",:]
n[n.Method .== "Quartic",:Weight] = nweights
n[n.Method .== "Spline3",:Weight] = nweights
n[n.Method .== "RBF",:Weight] = nweights
plot(n.Strike,abs.(n.Error*nweights*100),margin=3Plots.mm,xlabel="Strike",ylabel="Weighted Volatility Error in %",group=n.Method,size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_quartic_error.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/aapl_1m_svi_a0.txt", ' ',header=true)
df2 = DataFrame(m[1], vec(m[2]))
m = df2[df2.Method .== "Reference",:]
n = df2[df2.Method .== "SVI" .|| df2.Method .== "Parabolic" .|| df2.Method .== "SVI-a-positive",:]
plot(m.K,m.Vol .* 100,label="Market",seriestype = :scatter,markersize=3,size=(400,400),ylim=(29,50.0),markerstrokewidth=-1)
plot!(n.K,(n.Vol).*100,margin=3Plots.mm,xlabel="Strike",ylabel="Implied Volatility in %",group=n.Method)
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_svi.pdf")
m = readdlm("/home/fabien/mypapers/eqd_book/aapl_1m_svi_unbonuded.txt", ' ',header=true)
df2 = DataFrame(m[1], vec(m[2]))
m = df2[df2.Method .== "Reference",:]
n = df2[df2.Method .== "SVI" .|| df2.Method .== "Parabolic",:]
plot(m.K,m.Vol .* 100,label="Market",size=(400,400),seriestype = :scatter,markersize=3,ylim=(29,50.0),markerstrokewidth=-1)
plot!(n.K,(n.Vol).*100,margin=3Plots.mm,xlabel="Strike",ylabel="Implied Volatility in %",group=n.Method)
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_svi_unbounded.pdf")


using AQFED
forward = 93.6769575815521
tte = 0.057534246575342465
a=0.0000000000;b=0.6637133353;rho=-0.4073556325;sigma=0.1493655118;m=0.0084858398;
svip = AQFED.TermStructure.SVISection(a,b,rho,sigma,m,tte,forward)
a=-4.0799259866;b=18.5213470421;rho= 0.8767046157;sigma= 0.4680205198;m= 0.9311057129;
svi2=AQFED.TermStructure.SVISection(a,b,rho,sigma,m,tte,forward)
slope(s) = s.b*(1+abs(s.rho))
price = (svi,x) -> AQFED.Black.blackScholesFormula(true, x, forward, AQFED.TermStructure.varianceByLogmoneyness(svi,log(x/forward))*tte,1.0,1.0)
lstrikes = range(forward,stop=forward*10,length=1001)
plot(lstrikes, price.(svi2,lstrikes), label="unbounded",xlab="Strike",ylab="Call price",xscale=:log10, yscale=:log10,legendtitle="a",legend=:bottomright,xticks=([forward,2forward,4forward,8forward],["f","2f","4f","8f"]))
plot!(lstrikes, price.(svip,lstrikes), label="positive")
plot!(size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_svi_prices.pdf")
function gatheralDenom(w, y)
  dwdy = ForwardDiff.derivative(w, y)
  d2wdy2 = ForwardDiff.derivative(x -> ForwardDiff.derivative(w,x),y)
  return 1-y/w(y)*dwdy+(dwdy)^2 * (-1/4 - 1/w(y) + y^2/w(y)^2) / 4 + d2wdy2 / 2
end
strikes=range(100.0,stop=1000.0,length=1001)
w2 = x -> AQFED.TermStructure.varianceByLogmoneyness(svi2,x)
wp = x -> AQFED.TermStructure.varianceByLogmoneyness(svip,x)
plot(strikes, @.(gatheralDenom(wp,log(strikes/spot))),label="positive",legendtitle="a",legend=:topright, xlab="Strike",ylab="Local volatility denominator")
plot!(strikes, @.(gatheralDenom(w2,log(strikes/spot))),label="unbounded")
 


m = readdlm("/home/fabien/mypapers/eqd_book/aapl_1m_bdk.txt", ' ',header=true)
df2 = DataFrame(m[1], vec(m[2]))
n = df2[df2.Method .!= "Reference" ,:]
plot(df.Strike,df.Vol .* 100,label="Market",seriestype = :scatter,markersize=3,ylim=(29,50.0),markerstrokewidth=-1)
plot!(n.Strike, n.Vol .* 100, group=n.Method, xlabel="Strike", ylabel="Volatility in %",legendtitle="Extrapolation")
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_parabola_bdk.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/aapl_1m_extrapolation.txt", ' ',header=true)
df2 = DataFrame(m[1], vec(m[2]))
n = df2[df2.Method .!= "Reference" ,:]
n1 = df2[df2.Method .== "Parabolic" .|| df2.Method .== "Spline" .|| df2.Method .== "Parabolic-BDK" .|| df2.Method .=="Spline-BDK",:]
plot(n1.Strike, sqrt.(n1.w) .* 100, group=n1.Method, xlabel="Strike", ylabel="Volatility in %",legendtitle="Method",ylim=(20,400.0),size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_extrapolation_bdk.pdf")
n2 = df2[df2.Method .== "Parabolic" .|| df2.Method .== "Spline" .|| df2.Method .== "Parabolic-RK" .|| df2.Method .=="Spline-RK",:]
plot(n2.Strike, sqrt.(n2.w) .* 100, group=n2.Method, xlabel="Strike", ylabel="Volatility in %",legendtitle="Method",ylim=(20,400.0),size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_extrapolation_rk.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/aapl_180119_160212_am_prop_vol.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
m = readdlm("/home/fabien/mypapers/eqd_book/aapl_180119_160212_am_prop_vol_parabola.txt", ' ',header=true)
df2 = DataFrame(m[1], vec(m[2]))
n = df2[df2.Method .== "Strike" .|| df2.Method .== "Logmoneyness",:]
plot(df.Strike,df.Vol .* 100,label="Mid",size=(400,400),seriestype = :scatter,markersize=3,markerstrokewidth=-1)
plot!(n.Strike,(n.Vol).*100,margin=3Plots.mm,xlabel="Strike",ylabel="Implied Volatility in %",group=n.Method)
savefig("/home/fabien/mypapers/eqd_book/aapl_2y_parabola.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/aapl_3y.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
m = readdlm("/home/fabien/mypapers/eqd_book/aapl_1m.txt", ' ',header=true)
df2 = DataFrame(m[1], vec(m[2]))
plot(df2.Strike,df2.Call .* 100,label="1m",legendtitle="Expiry")
plot!(df.Strike,df.Call .* 100,size=(400,400),label="2y",xlabel="Strike",ylabel="Implied Volatility in %")
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_2y_strike.pdf")

df2[!,:Delta] = @. AQFED.Math.normcdf( -(-log(df2[!,:Strike]/93.6)+0.5*df2[!,:Call]^2*30.0/365)/(df2[!,:Call]*sqrt(30.0/365)))
df[!,:Delta] = @. AQFED.Math.normcdf( -(-log(df[!,:Strike]/92.6)+0.5*df[!,:Call]^2*30.0/365)/(df[!,:Call]*sqrt(30.0/365)))
plot(df2.Delta,df2.Call .* 100,label="1m",legendtitle="Expiry")
plot!(df.Delta,df.Call .* 100,size=(400,400),label="2y",xlabel="- Put Δ",ylabel="Implied Volatility in %")
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_2y_delta.pdf")


m = readdlm("/home/fabien/mypapers/svi_guess/bancorp_10y_svi_asymptotes.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
n=df[df.method .== "Reference",:]
o=df[df.method .=="Left-Raw" .|| df.method .== "Left-Truncated" .|| df.method .== "Right-Raw" ,:]
plot(n.strike, n.vol .* 100, label="Reference", seriestype=:scatter, ylabel="Volatility in %", xlabel="Strike")
plot!(o.strike,o.vol .*100,group=o.method,legendtitle="Method")
savefig("/home/fabien/mypapers/eqd_book/bancorp_10y_svi_asymptotes.pdf")


m = readdlm("/home/fabien/mypapers/svi_guess/airliq_10y_svi.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
data=df[df.method .== "Reference",:]
a=0.03282573662747978
b=0.02252578300083763
sigma=0.01931401931608065
rho=-0.9999901697624964
m=0.36908963892694535
a_q=2.666984783436798E-10
b_q=0.022407412867936443
sigma_q=5.000000054946551E-4
rho_q=-1.0
m_q=1.104208741949215
a_c=0.0
b_c=0.022682627008952412
sigma_c=0.25000000000001094
rho_c=-1.0
m_c=1.0761732271448656
S = 95.05
k = range(-1.5,stop=1.5,step=0.01)
svi = AQFED.TermStructure.SVISection(a,b,rho,sigma,m,1.0,S)
svi_q = AQFED.TermStructure.SVISection(a_q,b_q,rho_q,sigma_q,m_q,1.0,S)
svi_c = AQFED.TermStructure.SVISection(a_c,b_c,rho_c,sigma_c,m_c,1.0,S)
plot(log.(data.strike ./ S),data.vol .*100, label="Reference",seriestype=:scatter, ylabel="Volatility in %", xlabel="Log-moneyness",markerstrokewidth=-1)
plot!(k, @.(sqrt(AQFED.TermStructure.varianceByLogmoneyness(svi,k))*100), label="Parabolic-LM")
plot!(k, @.(sqrt(AQFED.TermStructure.varianceByLogmoneyness(svi_q,k))*100),label="Quasi-Explicit")
plot!(k,  @.(sqrt(AQFED.TermStructure.varianceByLogmoneyness(svi_c,k))*100),label="Quasi-Explicit σ ⪭ 0.25")
savefig("/home/fabien/mypapers/eqd_book/airliq_10y_svi_sigma.pdf")

α₀
α₁
m = readdlm("/home/fabien/mypapers/eqd_book/explicit_fit_sabr_0479_beta1.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
data=df[df.Method .== "Reference",:]
plot(data.Strike, data.Volatility .* 100, label="Reference", ylabel="Volatility in %", seriestype=:scatter, xlabel="Strike",markerstrokewidth=-1)
data=df[df.Method .!= "Reference",:]
#plot!(data.Strike, data.Volatility .* 100, group=data.Method)
plot!(data.Strike, data.Volatility .* 100, group=data.Method,labels=["Parabola" "α₀" "α₁"])
#+geom_line(data=m[m$Method!="Reference",])+scale_shape_manual("Stage", values=c(NA,NA,NA,1), breaks=c("alpha_0","alpha_1","Parabola","Reference"), labels=c(expression(paste(alpha[0]," guess")), expression(paste(alpha[1]," guess")), "Parabola", "Reference"))+scale_linetype_manual("Stage", values=c(2,1,3,0), breaks=c("alpha_0","alpha_1","Parabola","Reference"), labels=c(expression(paste(alpha[0]," guess")), expression(paste(alpha[1]," guess")), "Parabola", "Reference"))+theme(legend.position="bottom")
savefig("/home/fabien/mypapers/eqd_book/explicit_fit_sabr_0479_beta1.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/aapl_1m_sabr.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
n=df[df.Method .!= "Reference" .&& df.Method .!= "SVI",:]
m=df[df.Method .== "Reference",:]
plot(m.Strike, m.Vol .* 100, label="Market", ylabel="Volatility in %", seriestype=:scatter, xlabel="Strike",markerstrokewidth=-1)
plot!(n.Strike,n.Vol .* 100, group=n.Method, legendtitle="Interpolation")
plot!(size=(400,380))
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_sabr.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/two_sabr_1_smile.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
n=df[df.alpha .!= "Reference",:]
m=df[df.alpha .== "Reference",:]
plot(m.Strike, m.Volatility .* 100, label="Market vols", ylabel="Volatility in %", seriestype=:scatter, xlabel="Strike",markerstrokewidth=-1)
plot!(n.Strike,n.Volatility .* 100, group=n.alpha, legendtitle="Parameters", labels=["α=0.237" "α=0.850"])
plot!(size=(400,380))
savefig("/home/fabien/mypapers/eqd_book/two_sabr_1_smile.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/spxw170316_1w.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
n=df[df.Model .!= "Reference" .&& df.Model .!= "Parabola",:]
ind = sortperm(n.Strike)
m=df[df.Model .== "Reference",:]
plot(m.Strike, m.Volatility .* 100, label="Market", ylabel="Volatility in %", seriestype=:scatter, xlabel="Strike", markersize=3, markerstrokewidth=-1)
plot!(n.Strike[ind],n.Volatility[ind] .* 100, group=n.Model[ind], legendtitle="Interpolation",labels=["Least-squares Spline" "SABR" "SVI"])
plot!(size=(400,380))
savefig("/home/fabien/mypapers/eqd_book/spxw170316_1w.pdf")
n2 = n[1:273,:]
plot(n2.Strike, n2.Error.*100, group=n2.Model, xlab="Strike", ylab="Absolute error in volatility %")

m = readdlm("/home/fabien/mypapers/eqd_book/spxw170316_1w_fulls.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
n=df[df.Model .!= "Reference" .&& df.Model .!= "Parabola",:]
ind = sortperm(n.Strike)
m=df[df.Model .== "Reference",:]
plot(m.Strike, m.Volatility .* 100, label="Market", ylabel="Volatility in %", seriestype=:scatter, xlabel="Strike", markersize=3, markerstrokewidth=-1)
plot!(n.Strike[ind],n.Volatility[ind] .* 100, group=n.Model[ind], legendtitle="Interpolation",labels=["Least-squares Spline" "SABR" "SVI"])
plot!(size=(400,380))
savefig("/home/fabien/mypapers/eqd_book/spxw170316_1w_fulls.pdf")

#1M aapl
using AQFED.SSVI
tte=0.057534246575342465 
forward=93.6769575815521 
sqrtw = [3, 5, 6.928203230275509, 5.291502622129181, 3.3166247903554, 4.898979485566356, 5, 3.7416573867739413, 12.68857754044952, 3.4641016151377544, 5.830951894845301, 7.483314773547883, 12.409673645990857, 5, 11.445523142259598, 6.244997998398398, 10.392304845413264, 5.291502622129181, 5.385164807134504, 2, 7.745966692414834, 6.48074069840786, 19.313207915827967, 1.4142135623730951, 11.445523142259598, 6.164414002968976, 10.04987562112089, 4.358898943540674, 5, 5.916079783099616, 10.908712114635714, 5.196152422706632, 31.796226191169293, 14.071247279470288, 6.6332495807108, 4.898979485566356, 3.1622776601683795, 5.477225575051661, 2.8284271247461903, 6.324555320336759, 1.7320508075688772]
y = [-0.291357, -0.222364, -0.157825, -0.097201, -0.091336, -0.085505, -0.073944, -0.068213, -0.062515, -0.056849, -0.051216, -0.045613, -0.040042, -0.034502, -0.028992, -0.023513, -0.018063, -0.012643, -0.007252, -0.00189, 0.003443, 0.008748, 0.014025, 0.019274, 0.024496, 0.029691, 0.034859, 0.04, 0.045116, 0.050205, 0.055268, 0.060306, 0.065318, 0.075269, 0.085121, 0.094877, 0.104539, 0.114108, 0.123587, 0.160628, 0.20508]
v =[0.324092242681, 0.24657977862400002, 0.186966300816, 0.144333887569, 0.14130081, 0.138231522025, 0.132426121216, 0.129557523481, 0.127000289641, 0.12465701262399999, 0.12244960518400001, 0.11985236280899998, 0.117753981409, 0.115628561764, 0.11343895524900001, 0.11194377639999999, 0.10989755406400002, 0.108062755441, 0.10603098937600001, 0.1045099584, 0.10286837436099999, 0.101191427236, 0.099962204224, 0.09846353652099998, 0.09741702168899999, 0.09611178036099999, 0.09492622620100001, 0.093888926569, 0.09263866195600001, 0.0920212225, 0.09084738528099999, 0.09000180000900002, 0.089596054276, 0.08850030009999998, 0.08827197523599999, 0.087838140625, 0.088819496676, 0.08968946832400002, 0.09121487232400001, 0.10995856000000001, 0.14437416115600002]
vols = sqrt.(v);
weights = sqrtw.^2;
svi = AQFED.TermStructure.SVISection(0.0000000000, 0.6637133353, -0.4073556325, 0.1493655118, 0.0084858398, tte, forward)

xssvi = SSVI.calibrateXSSVISection(tte, forward, y, vols, weights)
strikes = exp.(y) .* forward;
plot(strikes, vols.*100, label="Market", ylabel="Volatility in %", seriestype=:scatter, xlabel="Strike", markersize=3, markerstrokewidth=-1)
lstrikes = range(strikes[1],stop=strikes[end],length=101)
plot!(lstrikes, @.( sqrt(AQFED.TermStructure.varianceByLogmoneyness(svi, log(lstrikes/forward)))*100), label="SVI")
plot!(lstrikes, @.( sqrt(AQFED.TermStructure.varianceByLogmoneyness(xssvi, log(lstrikes/forward)))*100), label="XSSVI")
plot!(size=(400,380))
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_xssvi.pdf")
rmse = sqrt(sum(w * (sqrt(AQFED.TermStructure.varianceByLogmoneyness(xssvi, k)) - vol)^2 for (k, vol, w) in zip(y, vols, weights))/sum(weights))
rmse = sqrt(sum(w * (sqrt(AQFED.TermStructure.varianceByLogmoneyness(svi, k)) - vol)^2 for (k, vol, w) in zip(y, vols, weights))/sum(weights))
#using Polynomials
#parabola = Polynomial([0.1025161325407153, -0.2884248981692053, 1.6506756695333973])
#plot!(lstrikes, @.(parabola(log(lstrikes/forward))), label="Parabola")

#2y aapl
tte=1.936986301369863 
forward=95.79205142667695 
y=[-0.70145, -0.650157, -0.554847, -0.467835, -0.387793, -0.313685, -0.244692, -0.180153, -0.119529, -0.090541, -0.06237, -0.034971, -0.008303, 0.017673, 0.04299, 0.091781, 0.138301, 0.182752, 0.225312, 0.266134, 0.305355, 0.343095, 0.379463, 0.414554, 0.448455, 0.573619, 0.630777]
v=[0.165566796201, 0.160235286436, 0.149636675241, 0.14268146382400002, 0.132878475625, 0.12624732796899998, 0.12292176240400002, 0.11877741888099998, 0.114789505636, 0.113083565841, 0.110983592164, 0.10859595252100002, 0.1055470144, 0.10508294722499999, 0.103609309456, 0.10128433550399998, 0.10002481528900001, 0.09683423712400001, 0.09441608198399999, 0.09437736968100001, 0.09293535160899999, 0.09121849657600001, 0.09058956236099999, 0.08931311560899999, 0.08799706944899999, 0.08668372524099999, 0.086451876729]
sqrtw=[12.449899597988733, 3.1622776601683795, 3.1622776601683795, 7.416198487095663, 3.7416573867739413, 11.224972160321824, 22.80350850198276, 8.602325267042627, 5.0990195135927845, 3.3166247903554, 10.099504938362077, 8, 16.792855623746664, 3.872983346207417, 19.974984355438178, 11.090536506409418, 12.409673645990857, 5.744562646538029, 7.0710678118654755, 9.486832980505138, 15.264337522473747, 7.0710678118654755, 8.54400374531753, 7.0710678118654755, 5.830951894845301, 7.681145747868608, 1.7320508075688772]
vols = sqrt.(v);
weights = sqrtw.^2;
svi = AQFED.TermStructure.SVISection(0.0000010186, 0.0962772082, -0.6277030428, 1.1289857530, -0.0435839844,tte,forward)
xssvi = SSVI.calibrateXSSVISection(tte, forward, y, vols, weights)
strikes = exp.(y) .* forward;
plot(strikes, vols.*100, label="Market", ylabel="Volatility in %", seriestype=:scatter, xlabel="Strike", markersize=3, markerstrokewidth=-1)
lstrikes = range(strikes[1],stop=strikes[end],length=101)
plot!(lstrikes, @.( sqrt(AQFED.TermStructure.varianceByLogmoneyness(svi, log(lstrikes/forward)))*100), label="SVI")
plot!(lstrikes, @.( sqrt(AQFED.TermStructure.varianceByLogmoneyness(xssvi, log(lstrikes/forward)))*100), label="XSSVI")
plot!(size=(400,380))
savefig("/home/fabien/mypapers/eqd_book/aapl_2y_xssvi.pdf")
rmse = sqrt(sum(w * (sqrt(AQFED.TermStructure.varianceByLogmoneyness(xssvi, k)) - vol)^2 for (k, vol, w) in zip(y, vols, weights))/sum(weights))
rmse = sqrt(sum(w * (sqrt(AQFED.TermStructure.varianceByLogmoneyness(svi, k)) - vol)^2 for (k, vol, w) in zip(y, vols, weights))/sum(weights))



m = readdlm(IOBuffer(replace(read("/home/fabien/mypapers/sabr_extra/jackel_dec_ratspline.txt"), UInt8('\t') => UInt8(' '))), ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
m = readdlm(IOBuffer(replace(read("/home/fabien/mypapers/collocation-springer/jaeckel_dec5_bach_opt_x.txt"), UInt8('\t') => UInt8(' '))), ' ',header=true)
mref = DataFrame(m[1], vec(m[2]))
m = readdlm(IOBuffer(replace(read("/home/fabien/mypapers/sabr_extra/jackel_dec_ratspline_0.txt"), UInt8('\t') => UInt8(' '))), ' ', header=true)
m0 = DataFrame(m[1], vec(m[2]))
ind = sortperm(df.Strike)
plot(mref.Strike, mref.VolRef .*100, label="Reference",  ylabel="Black implied volatility in %",  xlabel="Strike", seriestype=:scatter,  markersize=3, markerstrokewidth=-1)
plot!(df.Strike[ind], df.Vol[ind] .* 100, label="Lognormal")
ind0 = sortperm(m0.Strike)
plot!(m0.Strike[ind0], m0.Vol[ind0] .* 100, label="Zero strike")
plot!(xlim=(0.0068,1),xscale=:log10, legendtitle="Tail")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/jackel_dec_vol_leftd1.pdf")

plot(df.Strike[ind],df.Density[ind],xlab="Strike",ylab="Implied probability density",label="Lognormal",color=2)
plot!(m0.Strike[ind0], m0.Density[ind0] , label="Zero strike",color=3)
plot!(xlim=(0.0068,1),xscale=:log10, legendtitle="Tail")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/jackel_dec_dens_leftd1.pdf")


#

n = readdlm("/home/fabien/mypapers/collocation-springer/black_xy_ref.txt", ' ',header=true)
mref = DataFrame(n[1], vec(n[2]))
n = readdlm("/home/fabien/mypapers/collocation-springer/black_xy.txt", ' ',header=true)
m = DataFrame(n[1], vec(n[2]))
n = readdlm("/home/fabien/mypapers/collocation-springer/black_xy_cubic.txt", ' ',header=true)
mc = DataFrame(n[1], vec(n[2]))
p3=plot(mref.x,mref.y, label="",seriestype = :scatter,markershape=:cross,xlab="x",ylab="y",yticks=mref.y)
for i = eachindex(mref.x)
  plot!(p3,[-5.0;mref.x[i]],[mref.y[i]; mref.y[i]],linestyle=:dot,lc=:black,label="")
  plot!(p3,[mref.x[i];mref.x[i]],[0.0; mref.y[i]],linestyle=:dot,lc=:black,label="")
end
plot!(p3,m.x, m.y, label="Black-Scholes")
#plot!(m.x,m.yb, label="B-spline collocation")
plot!(p3, mc.x,mc.yb, label="Cubic polynomial collocation")
plot!(p3, size=(400,400), legend=:topleft)
savefig(p3, "/home/fabien/mypapers/eqd_book/blackxy.pdf")

p3=plot(mref.y,mref.p, label="",seriestype = :scatter,markershape=:cross,xlab="y",ylab="Cumulative probability density",xticks=mref.y)
for i = eachindex(mref.x)
  plot!(p3,[0.0;mref.y[i]],[mref.p[i]; mref.p[i]],linestyle=:dot,lc=:black,label="")
  plot!(p3,[mref.y[i];mref.y[i]],[0.0; mref.p[i]],linestyle=:dot,lc=:black,label="")
end
plot!(p3,mc.y, mc.p, label="Black-Scholes")
#plot!(m.x,m.yb, label="B-spline collocation")
plot!(p3, mc.y,mc.pb, label="Cubic polynomial collocation")
plot!(p3, size=(400,400), legend=:bottomright, xlims=(20,230))
savefig(p3, "/home/fabien/mypapers/eqd_book/black_yp.pdf")

x=range(-5.0,stop=5,step=0.01)
a0=[2786.596071756937, 147.34209111751903 ,0, 0.6016951058879941, 0, 0]
q = function(a, x) 
	@. a[1]+a[2]*x+a[3]*x^2+a[4]*x^3+a[5]*x^4+a[6]*x^5
end
n = readdlm("/home/fabien/mypapers/eqd_book/density_spw_1m_230118_collo5_cum.txt", ' ',header=true)
mcum = DataFrame(n[1], vec(n[2]))
a1=[2785.3521606355916, 146.73682894029116, 0, -0.5950579887459523, 0, 0]
a2=[2876.848315810806 ,56.80448843832256, -49.76759382860678, 8.487744676264446, 3.9730404748463086, 0.2896176339589605]
a3=[2849.4246271474985, 59.39270440282361, -6.271793622794486, 3.6687255123195968, -1.3843300393219482, 0.7362716519921044]

plot(mcum.x,mcum.Strike,label="forward estimate",seriestype=:scatter, markershape=:,xlab="x")
plot(mcum.x,mcum.Strike,label="raw estimate",seriestype=:scatter, markersize=3,xlab="x",markerstrokewidth=-1,markeralpha=0.5)
plot!(x,q(a1,x),label="cubic guess",ylab="strike")
plot!(x,q(a2,x),label="quintic guess")
plot!(x,q(a3,x),label="optimal quintic collocation")
plot!(ylims=(-2000,6000))#,breaks=c(-2000,0,2000,4000,6000))
plot!(size=(800,400))
savefig( "/home/fabien/mypapers/eqd_book/density_spw_1m_230118_collo5_cum_bw.pdf")

a0=[2786.596071756937, 147.34209111751903 ,0, 0.6016951058879941, 0, 0]
n = readdlm("/home/fabien/mypapers/eqd_book/density_spw_1m_230118_collo5_cumqp.txt", ' ',header=true)
mcum = DataFrame(n[1], vec(n[2]))
a1=[2805.1169123299496, -11.117360654194995, 0 ,23.059650244031207,0,0]
a2=[2848.9760316890156, 49.63700482922661 ,-5.980960073681884 ,7.008424288762293, -1.3317427361985148, 0.5102028868911852]
a3=[2849.434477060442, 59.397024909112744, -6.288430108083293 ,3.6641083110376385 ,-1.382067848540136, 0.7368652527688424]
plot(mcum.x,mcum.Strike,label="QP estimate",seriestype=:scatter, markersize=3,xlab="x",markerstrokewidth=-1,markeralpha=0.5)
plot!(x,q(a1,x),label="cubic guess",ylab="strike")
plot!(x,q(a2,x),label="quintic guess")
plot!(x,q(a3,x),label="optimal quintic collocation")
plot!(ylims=(-2000,6000))#,breaks=c(-2000,0,2000,4000,6000))
plot!(size=(400,400))
savefig( "/home/fabien/mypapers/eqd_book/density_spw_1m_230118_collo5_cumqp_bw.pdf")

n = readdlm("/home/fabien/mypapers/eqd_book/density_spw_1m_230118_collo5_cums.txt", ' ',header=true)
mcum = DataFrame(n[1], vec(n[2]))
a1=[2859.069739671715, 5.857576611515046, -16.22072552838554, 19.177240740417457,0,0]
a2=[2851.155970143886, 65.10356540971928 ,-9.663751977115254, -0.047207132214036975, -0.8307915866774332, 1.2708791758944042]
plot(mcum.x,mcum.Strike,label="forward sweep estimate",seriestype=:scatter, markersize=3,xlab="x",markerstrokewidth=-1,markeralpha=0.5)
plot!(x,q(a1,x),label="cubic guess",ylab="strike")
plot!(x,q(a2,x),label="quintic guess")
plot!(x,q(a3,x),label="optimal quintic collocation")
plot!(ylims=(-2000,6000))#,breaks=c(-2000,0,2000,4000,6000))
plot!(size=(400,400))
savefig( "/home/fabien/mypapers/eqd_book/density_spw_1m_230118_collo5_cums_bw.pdf")

n = readdlm("/home/fabien/mypapers/eqd_book/density_spw_1m_230118_collo3.txt", ' ',header=true)
m2c3= DataFrame(n[1], vec(n[2]))
n = readdlm("/home/fabien/mypapers/eqd_book/density_spw_1m_230118_collo5.txt", ' ',header=true)
m2c5= DataFrame(n[1], vec(n[2]))
n = readdlm("/home/fabien/mypapers/eqd_book/density_spw_1m_230118_svi.txt", ' ',header=true)
msvi= DataFrame(n[1], vec(n[2]))
#msvi$Method = "SVI"
#msvi$Density =msvi$DensityNum
#m2c3$Method = "Cubic collocation"
#m2c5$Method = "Quintic collocation"
#m= rbind(msvi[,c("Strike","Vol","VolError","Method","DensityNum")],m2c3[,c("Strike","Vol","VolError","Method","DensityNum")],m2c5[,c("Strike","Vol","VolError","Method","DensityNum")])
#mref<-msvi[,c("Strike","Vol","VolError","Method","DensityNum")]
#mref$Method = "Reference"
#mref$Vol = msvi$Vol-msvi$VolError
#m = rbind(m,mref)
plot(msvi.Strike, (msvi.Vol .- msvi.VolError).*100, seriestype=:scatter, markersize=3,xlab="Strike",markerstrokewidth=-1,markeralpha=0.5,ylab="Volatility in %",label="Market")
plot!(msvi.Strike,msvi.Vol .* 100,label="SVI")
plot!(m2c3.Strike,m2c3.Vol .* 100,label="Cubic collocation")
plot!(m2c5.Strike,m2c5.Vol .* 100,label="Quintic collocation")
plot!(size=(800,400),margin=3Plots.mm)
savefig( "/home/fabien/mypapers/eqd_book/vol_spw_1m_230118_collo_bw.pdf")

n = readdlm("/home/fabien/mypapers/eqd_book/density_spw_1m_050218_collo5.txt", ' ',header=true)
m2c3= DataFrame(n[1], vec(n[2]))
n = readdlm("/home/fabien/mypapers/eqd_book/density_spw_1m_050218_collo9.txt", ' ',header=true)
m2c5= DataFrame(n[1], vec(n[2]))
n = readdlm("/home/fabien/mypapers/eqd_book/density_spw_1m_050218_svi.txt", ' ',header=true)
msvi= DataFrame(n[1], vec(n[2]))
plot(msvi.Strike, (msvi.Vol .- msvi.VolError).*100, seriestype=:scatter, markersize=3,xlab="Strike",markerstrokewidth=-1,markeralpha=0.5,ylab="Volatility in %",label="Market")
plot!(msvi.Strike,msvi.Vol .* 100,label="SVI")
plot!(m2c3.Strike,m2c3.Vol .* 100,label="Quintic collocation")
plot!(m2c5.Strike,m2c5.Vol .* 100,label="Nonic collocation")
plot!(size=(800,400),margin=3Plots.mm)
savefig( "/home/fabien/mypapers/eqd_book/vol_spw_1m_050218_collo_bw.pdf")

plot(msvi.Strike,msvi.DensityNum ,label="SVI",xlab="Strike",ylab="Probability density")
plot!(m2c3.Strike,m2c3.DensityNum,label="Quintic collocation")
plot!(m2c5.Strike,m2c5.DensityNum,label="Nonic collocation")
plot!(size=(800,300),margin=3Plots.mm)
savefig( "/home/fabien/mypapers/eqd_book/density_spw_1m_050218_collo_bw.pdf")


n = readdlm("/home/fabien/mypapers/eqd_book/bimodal_collo2.txt", ' ',header=true)
df= DataFrame(n[1], vec(n[2]))
m=df[df.Method .!= "Septic" .&& df.Method .!= "Spline",:]
plot(m.Strike,m.Density,group=m.Method,xlab="Strike",ylab="Probability density")
plot!(ylim=(0,5),size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/density_bimodal_collo.pdf")
m=df[df.Method .== "Kernel" .|| df.Method .== "Spline",:]
plot(m.Strike,m.Density,group=m.Method,xlab="Strike",ylab="Probability density")
plot!(ylim=(0,5),size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/density_bimodal_collo_s.pdf")

m=df[df.Method .== "Kernel" .|| df.Method .== "Quintic" .|| df.Method .== "Nonic",:]
plot(m.Strike,m.Vol .* 100,group=m.Method, xlab="Strike" , ylab="Implied volatility in %")
plot!(size=(800,400),margin=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/vol_bimodal_collo.pdf")

#Andreasen huge on jackel dec example. Nice smooth density. This is very different for fukasawa
m = readdlm("/home/fabien/mypapers/collocation-springer/jaeckel_dec5_ah_h_vols.txt", ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
plot(log.(df.Strike), df.Density)
#could write a small blog entry on this. Fukasaza is not arbitrage free. And density is not smooth
#still useful for robust var swap pricing, even on a preexisting other interp/extrap. range is nice as function of pdf..
#IS EXP COLLO similar to Fuka?


m = readdlm("/home/fabien/mypapers/eqd_book/dupire_labordere_liquidator_modelvol.txt",' ', header=true)
df = DataFrame(m[1], vec(m[2]))
#model vol
plot(df.Strike, sqrt.(df.Before).*100, label="3.499", xlab="Strike",ylab="Model volatility in %",linestyle=:auto)
plot!(df.Strike, sqrt.(df.BeforeShifted).*100, label="3.499 shifted",linestyle=:auto)
plot!(df.Strike, sqrt.(df.At).*100, label="3.5",linestyle=:auto)
plot!(df.Strike,  sqrt.(df.Maturity).*100, label="5.0",linestyle=:auto)
plot!(legendtitle="Expiry", xscale=:log, xlims=(2,500))
plot!(xticks=([5,10,20,40,80,160],[5,10,20,40,80,160]))
plot!(size=(480,360))
savefig("/home/fabien/mypapers/eqd_book/dupire_labordere_liquidator_modelvol_log.pdf")

plot!(legendtitle="Expiry", xscale=:continuous, ylims=(40,70),xlims=(0.5,20))
plot!(size=(480,360))
savefig("/home/fabien/mypapers/eqd_book/dupire_labordere_liquidator_modelvol.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/dupire_labordere_liquidator_localvol100.txt",' ', header=true)
df = DataFrame(m[1], vec(m[2]))
#model vol
plot(df.Strike, sqrt.(df[:,7]).*100, label="3.499", xlab="Strike",ylab="Model volatility in %",linestyle=:auto)
plot!(df.Strike, sqrt.(df[:,6]).*100, label="3.5",linestyle=:auto)
plot!(df.Strike,  sqrt.(df[:,5]).*100, label="5.0",linestyle=:auto)
plot!(legendtitle="Expiry", xscale=:log, xlims=(4,500))
plot!(xticks=([5,10,20,40,80,160],[5,10,20,40,80,160]))
plot!(size=(480,360))
savefig("/home/fabien/mypapers/eqd_book/dupire_labordere_liquidator_localvol_log.pdf")
plot!(legendtitle="Expiry", xscale=:continuous, xlims=(5,100))
plot!(size=(480,360))
savefig("/home/fabien/mypapers/eqd_book/dupire_labordere_liquidator_localvol.pdf")

m = readdlm("/home/fabien/mypapers/svi_guess/svi_localvol_sp500.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
n = df[df.time .> 6.5 .&& df.time .< 7.5,:]
plot(n.strike, n.localvol .* 100, group=string.("Local vol. T=",round.(n.time,digits=3)), ylab="Volatility in %",xlab="Strike")
plot!(n.strike, n.vol .* 100, group=string.("Implied vol. T=",round.(n.time,digits=3)), linestyle=:auto)
plot!(size=(480,360))
savefig("/home/fabien/mypapers/eqd_book/svi_localvol_sp500_positive.pdf")

m = readdlm("/home/fabien/mypapers/svi_guess/svi_localvol_sp500_increasing.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
n = df[df.time .> 6.5 .&& df.time .< 7.5,:]
plot(n.strike, n.localvol .* 100, group=string.("Local vol. T=",round.(n.time,digits=3)), ylab="Volatility in %",xlab="Strike")
plot!(n.strike, n.vol .* 100, group=string.("Implied vol. T=",round.(n.time,digits=3)), linestyle=:auto)
plot!(size=(480,360))
savefig("/home/fabien/mypapers/eqd_book/svi_localvol_sp500_increasing.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/svi_zeliade_arb.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
plot(df.Logmoneyness,df.Variance,group=df.Expiry, linestyle=:auto,xlab="Forward log-moneyness", ylab="Total variance",legendtitle="Expiry")
plot!(size=(480,400))
savefig("/home/fabien/mypapers/eqd_book/svi_zeliade_arb.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/svi_zeliade_arb_zero.txt", ' ', header=true)
df0 = DataFrame(m[1], vec(m[2]))
df = df0[df0.Model .== "TRBDF2-LV",:]
plot(df.Logmoneyness,df.Volatility.^2 .* df.Expiry,group=df.Expiry, linestyle=:auto,xlab="Forward log-moneyness", ylab="Total variance",legendtitle="Expiry")
plot!(size=(480,400))
savefig("/home/fabien/mypapers/eqd_book/svi_zeliade_arb_zero.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/svi_vogt_arb_zero.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfv = dfm[dfm.Method .== "Vogt",:]
dfm = dfm[dfm.Method .== "TRBDF2-LV",:]
dfm.Method .= "Floor-0"
#plot(dfm.Logmoneyness, dfm.Volatility .* 100, ylab="Volatility in %", xlab="Forward log-moneyness")
n=readdlm("/home/fabien/mypapers/eqd_book/svi_vogt_arb_abs.txt", ' ', header=true)
dfn = DataFrame(n[1], vec(n[2]))
dfn = dfn[dfn.Method .== "TRBDF2-LV",:]
dfn.Method .= "Abs"
dfm = vcat(dfm,dfn)
n=readdlm("/home/fabien/mypapers/eqd_book/svi_vogt_arb_cap10.txt", ' ', header=true)
dfn = DataFrame(n[1], vec(n[2]))
dfn = dfn[dfn.Method .== "TRBDF2-LV",:]
dfn.Method .= "Cap-10"
dfm = vcat(dfm,dfn)
n=readdlm("/home/fabien/mypapers/eqd_book/svi_vogt_arb_cap1000.txt", ' ', header=true)
dfn = DataFrame(n[1], vec(n[2]))
dfn = dfn[dfn.Method .== "TRBDF2-LV",:]
dfn.Method .= "Cap-1000"
dfm = vcat(dfm,dfn)
n=readdlm("/home/fabien/mypapers/eqd_book/svi_vogt_arb_search.txt", ' ', header=true)
dfn = DataFrame(n[1], vec(n[2]))
dfn = dfn[dfn.Method .== "TRBDF2-LV",:]
dfn.Method .= "Search"
dfm = vcat(dfm,dfn)
#n <- m[m$Method!="RE-LV" & m$Method!="TRBDF2-Vogt" & m$Method!="Gatheral",]
#o <- m[m$Method=="Vogt",]
plot(dfv.Logmoneyness, dfv.Volatility .* 100, seriestype=:scatter, label="Vogt", msw=2,msc=:auto, markershape=:cross)
plot!(dfm.Logmoneyness, dfm.Volatility .* 100, group=dfm.Method, linestyle=:auto, legendtitle="Method", ylab="Volatility in %", xlab="Forward log-moneyness")
plot!(size=(800,500),margin=3Plots.mm) 
savefig("/home/fabien/mypapers/eqd_book/svi_vogt_arb_zero.pdf")

m=readdlm("/home/fabien/mypapers/eqd_book/svi_vogt_arb_rk.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
dfm = df[df.Parameters .== "Cap-10" .|| df.Parameters .== "Cap-100",:]
dfv = df[df.Parameters .=="Vogt",:]
plot(dfv.y, sqrt.(dfv.w) .* 100, label="Vogt", seriestype=:scatter, msw=2,msc=:auto, markershape=:cross,ylab="Volatility in %",xlab="Forward log-moneyness")
plot!(dfm.y, sqrt.(dfm.w) .* 100, group=dfm.Parameters)
plot!(size=(800,500),margin=3Plots.mm) 
savefig("/home/fabien/mypapers/eqd_book/svi_vogt_arb_vol_cap.pdf")

dfm = df[df.Parameters .== "Cap-100",:] 
plot(dfm.y, dfm.w./dfm.g, label="",ylab="Local variance",xlab="Log-moneyness")
plot!(size=(400,250),margin=1Plots.mm) 
savefig("/home/fabien/mypapers/eqd_book/svi_vogt_arb_lv_cap100.pdf")
dfm = df[df.Parameters .== "Cap-10",:] 
plot(dfm.y, dfm.w./dfm.g, label="",ylab="Local variance",xlab="Log-moneyness")
plot!(size=(400,250),margin=1Plots.mm) 
savefig("/home/fabien/mypapers/eqd_book/svi_vogt_arb_lv_cap10.pdf")


m=readdlm("/home/fabien/mypapers/eqd_book/svi_vogt_arb.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
dfm = df[df.Parameters .== "Gatheral" .|| df.Parameters .== "QE-Penalty",:]
dfo  = df[df.Parameters .== "Vogt",:]
plot(dfo.y,sqrt.(dfo.w) .* 100, label="Vogt",seriestype=:scatter, msw=2,msc=:auto, markershape=:cross,ylab="Volatility in %",xlab="Forward log-moneyness")
plot!(dfm.y,sqrt.(dfm.w) .* 100, group=dfm.Parameters)
plot!(size=(400,300))
plot!(legend=:top)
savefig("/home/fabien/mypapers/eqd_book/svi_vogt_arb_vol.pdf")
plot(dfo.y,dfo.g, label="Vogt",seriestype=:scatter, msw=2,msc=:auto, markershape=:cross,ylab="Denominator",xlab="Forward log-moneyness")
plot!(dfm.y,dfm.g, group=dfm.Parameters)
hline!([0.0],color=:black,label="")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/svi_vogt_arb_g.pdf")


m=readdlm("/home/fabien/mypapers/eqd_book/svi_vogt_arb_spline10.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
dfm = df[df.Parameters .!= "Vogt",:]
dfo  = df[df.Parameters .== "Vogt",:]
plot(dfo.y,sqrt.(dfo.w) .* 100, label="Vogt",seriestype=:scatter, msw=2,msc=:auto, markershape=:cross,ylab="Volatility in %",xlab="Forward log-moneyness")
plot!(dfm.y,sqrt.(dfm.w) .* 100, group=dfm.Parameters)
plot!(size=(400,300))
plot!(legend=:top)
savefig("/home/fabien/mypapers/eqd_book/svi_vogt_arb_spline10_vol.pdf")
plot(dfo.y,dfo.g, label="Vogt",seriestype=:scatter, msw=2,msc=:auto, markershape=:cross,ylab="Denominator",xlab="Forward log-moneyness")
plot!(dfm.y,dfm.g, group=dfm.Parameters)
hline!([0.0],color=:black,label="",z_order=:back)
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/svi_vogt_arb_spline10_g.pdf")

r=readdlm("/home/fabien/mypapers/eqd_book/kahale_raw_lv.txt", ' ', header=true)
dfr = DataFrame(r[1], vec(r[2]))
m=readdlm("/home/fabien/mypapers/eqd_book/kahale_raw_lv_spline.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
n=readdlm("/home/fabien/mypapers/eqd_book/kahale_fixed_lv2.txt", ' ', header=true)
dfn = DataFrame(n[1], vec(n[2]))
dfm1 = dfm[dfm.t .== 0.425,:]
dfn1 = dfn[dfn.t .== 0.425,:]
dfr1 = dfr[dfr.t .== 0.425,:]
plot(dfr1.y,sqrt.(dfr1.w ./ dfr1.t) .* 100, label="Reference",seriestype=:scatter,msw=2,msc=:auto, markershape=:cross,ylab="Volatility in %",xlab="Forward log-moneyness")
plot!(dfn1.y,sqrt.(dfn1.w ./ dfn1.t) .* 100, label="De-arbitraged Spline",xlab="Log-moneyness", ylab="Implied volatility in %")
plot!(dfm1.y,sqrt.(dfm1.w ./ dfm1.t) .* 100, label="Raw Spline")
plot!(size=(400,300))
plot!(legend=:top)
savefig("/home/fabien/mypapers/eqd_book/kahale_raw_lv_spline_iv.pdf")

plot(dfn1.y,(dfn1.g ), label="De-arbitraged Spline",xlab="Log-moneyness", ylab="Denominator",color=2)
plot!(dfm1.y,(dfm1.g ), label="Raw Spline",color=3)
plot!(size=(400,300))
hline!([0.0],color=:black,label="",z_order=:back)
plot!(legend=:top)
savefig("/home/fabien/mypapers/eqd_book/kahale_raw_lv_spline_lv.pdf")
ggsave(file="/home/fabien/mypapers/eqd_book/kahale_raw_lv_spline_lv_bw.eps",width=4,height=4)

using StatsPlots
m=readdlm("/home/fabien/mypapers/eqd_book/lv_discretisation_zeliade.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
#dfm =dfm[dfm.t.==4.34,:]
dfn = dfm[dfm.Method .=="RE-LV" .|| dfm.Method .=="RE-LVD" .||  dfm.Method .== "RE-LVFD" .|| dfm.Method .== "TRBDF2-LV" .|| dfm.Method .=="TRBDF2-LVD" .||  dfm.Method .=="TRBDF2-LVFD" .||  dfm.Method .=="TRBDF2-Black" ,:]
dfn.Method = @. ifelse(dfn.Method == "RE-LV", "RE-(D)",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "RE-LVD", "RE-(B)",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "RE-LVFD", "RE-(A)",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "TRBDF2-LV", "TRBDF2-(D)",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "TRBDF2-LVD", "TRBDF2-(B)",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "TRBDF2-LVFD", "TRBDF2-(A)",dfn.Method)
dfn1 = dfn[dfn.t .> 4.3,:]
@df dfn1 boxplot(:Method, abs.(:Error),yscale=:log10,label="",color=7,ylab="Absolute error in price")
plot!(size=(900,400),margin=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/lv_discretisation_zeliade_box.pdf")

m=readdlm("/home/fabien/mypapers/eqd_book/lv_discretisation_zeliade_100_500.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn=dfm[dfm.Method .=="TRBDF2-LV" .||  dfm.Method .=="TRBDF2-LVG" .||  dfm.Method .=="TRBDF2-Black",:]
dfn.Method = @. ifelse(dfn.Method == "TRBDF2-LV", "Analytic",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "TRBDF2-LVG", "Numeric",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "TRBDF2-Black", "Black",dfn.Method)
plot(dfn.Strike,abs.(dfn.Error), linestyle=:auto, group=dfn.Method, ylab="Absolute error in price",xlab="Strike")
plot!(yscale=:log10,legend=:bottomright)
plot!(size=(800,300),margin=5Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/lv_discretisation_zeliade_100_500.pdf")


m=readdlm("/home/fabien/mypapers/eqd_book/lv_milstein_zeliade_ssobolbb_016.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn = dfm[dfm.N .== 65535,:]
dfn.Method = @. ifelse(dfn.Method == "LV-Euler-Discrete", "Euler-A",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Euler-Backward", "Euler-B",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Euler-Mid", "Euler-D",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Milstein-Discrete", "Milstein-A",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Milstein-Backward", "Milstein-B",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Milstein-Mid", "Milstein-D",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Milstein-Clark", "Milstein-Clark",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Euler-Clark", "Euler-Clark",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Euler-F", "Euler",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Milstein-F", "Milstein",dfn.Method)

mh=readdlm("/home/fabien/mypapers/eqd_book/lv_milstein_zeliade_ssobolbb_008.txt", ' ', header=true)
dfmh = DataFrame(mh[1], vec(mh[2]))
dfnh = dfmh[dfmh.N .== 65535 .&& dfmh.Method .== "LV-Euler-F",:]
dfnh.Method = @. ifelse(dfnh.Method == "LV-Euler-F", "Euler-Half",dfnh.Method)
dfo = dfn[dfn.Expiry .==0.16 .&& (dfn.Method .=="Euler" .|| dfn.Method .=="Milstein"),:]
dfoh = dfnh[dfnh.Expiry .==0.16,:]
dfn2 = vcat(dfo,dfoh)
#Euler vs. Milstein 0.16
plot(dfn2.Strike,abs.(dfn2.Error),group=dfn2.Method,ylab="Absolute error", xlab="Strike")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/lv_milstein_zeliade_ssobolbb_016_milstein_short.pdf")

#Euler vs. Milstein 4.34
dfo = dfn[dfn.Expiry .==4.34 .&& (dfn.Method .=="Euler-D" .|| dfn.Method .=="Milstein-D" .|| dfn.Method .=="Milstein-Clark"),:]
dfoh = dfnh[dfnh.Expiry .==4.34,:]
dfn2 = vcat(dfo,dfoh)
#plot(dfo.Strike,dfo.Error,group=dfo.Method,ylab="Error in price",xlab="Strike")
plot(dfn2.Strike,abs.(dfn2.Error),group=dfn2.Method,ylab="Absolute error", xlab="Strike")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/lv_milstein_zeliade_ssobolbb_016_milstein.pdf")

dfo=dfn[dfn.Expiry .==4.34 .&& (dfn.Method .=="Euler-D" .|| dfn.Method .=="Euler-A" .|| dfn.Method .=="Euler-B" .|| dfn.Method .=="Euler-Clark"),:]
plot(dfo.Strike,dfo.Error,group=dfo.Method,ylab="Error in price",xlab="Strike")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/lv_milstein_zeliade_ssobolbb_016_euler.pdf")

m=readdlm("/home/fabien/mypapers/eqd_book/lv_milstein_zeliade_ssobolbb_001.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn = dfm[dfm.N .== 65535,:]
dfn.Method = @. ifelse(dfn.Method == "LV-Euler-Discrete", "Euler-A",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Euler-Backward", "Euler-B",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Euler-Mid", "Euler-D",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Milstein-Discrete", "Milstein-A",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Milstein-Backward", "Milstein-B",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Milstein-Mid", "Milstein-D",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Milstein-Clark", "Milstein-Clark",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Euler-Clark", "Euler-Clark",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Euler-F", "Euler",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "LV-Milstein-F", "Milstein",dfn.Method)

mh=readdlm("/home/fabien/mypapers/eqd_book/lv_milstein_zeliade_ssobolbb_0005.txt", ' ', header=true)
dfmh = DataFrame(mh[1], vec(mh[2]))
dfnh = dfmh[dfmh.N .== 65535 .&& dfmh.Method .== "LV-Euler-F",:]
dfnh.Method = @. ifelse(dfnh.Method == "LV-Euler-F", "Euler-Half",dfnh.Method)
dfo = dfn[dfn.Expiry .==0.16 .&& (dfn.Method .=="Euler" .|| dfn.Method .=="Milstein"),:]
dfoh = dfnh[dfnh.Expiry .==0.16,:]
dfn2 = vcat(dfo,dfoh)
plot(dfn2.Strike,abs.(dfn2.Error),group=dfn2.Method,ylab="Absolute error", xlab="Strike")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/lv_milstein_zeliade_ssobolbb_001_milstein_short.pdf")

m=readdlm("/home/fabien/mypapers/eqd_book/mcah_zeliade_016.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
df = dfm[dfm.Method .!= "Analytic",:]
plot(df.Strike,df.Error, group = df.Method, xlab="Strike",ylab="Error")
plot!(size=(400,300))
plot!(legend=:topleft)
savefig("/home/fabien/mypapers/eqd_book/mcah_zeliade_016.pdf")
m=readdlm("/home/fabien/mypapers/eqd_book/mcah_zeliade_0016.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
df = dfm[dfm.Method .!= "Analytic",:]
plot(df.Strike,df.Error, group = df.Method, xlab="Strike",ylab="Error")
plot!(size=(400,300))
plot!(legend=:topleft)
savefig("/home/fabien/mypapers/eqd_book/mcah_zeliade_0016.pdf")

m=readdlm("/home/fabien/mypapers/eqd_book/localvol_flatlinear.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfmbl = dfm[dfm.Method .== "TRBDF2-Black",:]
dfmlv = dfm[dfm.Method .== "TRBDF2-LocalVol",:]
plot(dfmbl.y,dfmbl.Volatility .* 100, seriestype=:scatter, label="TRBDF2 Black", ylab="Implied Black volatility in %", xlab="Log-moneyness")
plot(dfmbl.y,dfmbl.Volatility .* 100, seriestype=:scatter, label="TRBDF2 Black", ylab="Implied Black volatility in %", xlab="Log-moneyness",msw=1,mc=:white)
plot!(dfmlv.y,dfmlv.Volatility .* 100, label = "TRBDF2 Local vol.")
#plot!(ylims=(50,65))
plot!(size=(400,300))
plot!(legend=:topleft)
savefig("/home/fabien/mypapers/eqd_book/localvol_flatlinear.pdf")
m=readdlm("/home/fabien/mypapers/eqd_book/localvol_flatlinear_dirac.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfmbl = dfm[dfm.Method .== "TRBDF2-Black",:]
dfmlv = dfm[dfm.Method .== "TRBDF2-LocalVol",:]
plot(dfmbl.y,dfmbl.Volatility .* 100, seriestype=:scatter, label="TRBDF2 Black", ylab="Implied Black volatility in %", xlab="Log-moneyness")
plot(dfmbl.y,dfmbl.Volatility .* 100, seriestype=:scatter, label="TRBDF2 Black", ylab="Implied Black volatility in %", xlab="Log-moneyness",msw=1,mc=:white)
plot!(dfmlv.y,dfmlv.Volatility .* 100, label = "TRBDF2 Local vol.")
#plot!(ylims=(50,65))
plot!(size=(400,300))
plot!(legend=:topleft)
savefig("/home/fabien/mypapers/eqd_book/localvol_flatlinear_dirac.pdf")

m=readdlm("/home/fabien/mypapers/eqd_book/kahale_linear_price400.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
dfm = df[df.Method .== "TRBDF2-LV",:]
plot(dfm.Strike, abs.(dfm.Error./dfm.Price), group=(dfm.t),xlab="Strike",ylab="Relative error in price",ylims=(1e-7,1e-1),yaxis=:log10, seriestype=:scatter,palette=:acton10,legend=:outertopright,legendtitle="Expiry")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/kahale_linear_price400.pdf")

m=readdlm("/home/fabien/mypapers/eqd_book/kahale_flat_price400.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
dfm = df[df.Method .== "TRBDF2-LV",:]
plot(dfm.Strike, abs.(dfm.Error./dfm.Price), group=(dfm.t),xlab="Strike",ylab="Relative error in price",ylims=(1e-7,1e-1),yaxis=:log10, seriestype=:scatter,palette=:acton10,legend=:outertopright,legendtitle="Expiry")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/kahale_flat_price400.pdf")

m=readdlm("/home/fabien/mypapers/eqd_book/kahale_linear_price_positive400.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
dfm = df[df.Method .== "TRBDF2-LV" .&& df.Strike .== 826,:]
plot(dfm.t, abs.(dfm.Error./dfm.Price), label="Raw",xlab="Strike",ylab="Relative error in price",yaxis=:log10)
m=readdlm("/home/fabien/mypapers/eqd_book/kahale_linear_price400.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
dfn = df[df.Method .== "TRBDF2-LV" .&& df.Strike .== 826,:]
plot!(dfn.t, abs.(dfn.Error./dfn.Price), label="Capped",xlab="Expiry in ACT/365",ylab="Relative error in price",yaxis=:log10)
plot!(size=(800,200),margins=5Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/kahale_linear_price_positive400.pdf")


m = readdlm(IOBuffer(replace(read("/home/fabien/mypapers/eqd_book/aapl_1m_lv_trv.txt",String), "  " => " ")), ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
dfn = df[df.Method .== "TRBDF2-LV" .|| df.Method .== "TRBDF2-LVG" .|| df.Method .== "Black",:]
dfn.Method = @. ifelse(dfn.Method == "TRBDF2-LV", "Linear-IV",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "TRBDF2-LVG", "Flat-LV",dfn.Method)
dfo = dfn[dfn.Method .=="Black",:]
dfoo=dfn[dfn.Method .!= "Black",:]
plot(dfo.Strike, dfo.Volatility.*100,seriestype=:scatter, label="Reference", xlab="Strike",ylab="Volatility in %")
plot!(dfoo.Strike, dfoo.Volatility.*100, group=dfoo.Method)
plot!(xlims=(68,117),ylims=(28,60))
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_lv_flat_black.pdf")


m = readdlm(IOBuffer(replace(read("/home/fabien/mypapers/eqd_book/aapl_1m_lv_flat.txt",String), "  " => " ")), ' ',header=true)
df = DataFrame(m[1], vec(m[2]))
dfm=df[df.Method .!= "Black",:]
mycolormap = Dict("Linear-IV"=>3, "Flat-LV"=>2)

plot(dfm.y,dfm.w ./ dfm.g, group=dfm.Method,ylab="Local variance", xlab="Log-moneyness",color=[mycolormap[me] for me in dfm.Method])
plot!(ylims=(0,2))
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_lv_flat.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/qlv_1m_expansion.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
#m<-m[m$Method != "LPP3",]
dfo=df[df.Method .== "Analytic",:]
dfoo=df[df.Method .!= "Analytic",:]
plot(dfo.Strike, dfo.Volatility .*100, group=dfo.Method,xlab="Strike", ylab="Black volatility in %",seriestype=:scatter,markeralpha=0.5,markersize=4)
plot!(dfoo.Strike,dfoo.Volatility .*100,group=dfoo.Method)
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/qlv_1m_expansion.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/qlv_2y_expansion.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
#m<-m[m$Method != "LPP3",]
dfo=df[df.Method .== "Analytic",:]
dfoo=df[df.Method .!= "Analytic",:]
plot(dfo.Strike, dfo.Volatility .*100, group=dfo.Method,xlab="Strike", ylab="Black volatility in %",seriestype=:scatter,markeralpha=0.5,markersize=4)
plot!(dfoo.Strike,dfoo.Volatility .*100,group=dfoo.Method)
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/qlv_2y_expansion.pdf")
plot(dfoo.Strike, dfoo.Error .*100,group=dfoo.Method, xlab="Strike", ylab="Error in volatility %")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/qlv_2y_expansion_error.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/aapl_1m_qlv.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
dfo=df[df.Method .!= "Reference",:]
dfoo=df[df.Method .== "Reference",:]
plot(dfoo.K, dfoo.Vol .*100, group=dfoo.Method,seriestype=:scatter, markersize=4,xlab="Strike", ylab="Black volatility in %")
plot!(dfo.K, dfo.Vol .*100, group=dfo.Method)
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/aapl_1m_qlv.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/aapl_2y_qlv.txt", ' ', header=true)
df = DataFrame(m[1], vec(m[2]))
dfo=df[df.Method .!= "Reference",:]
dfoo=df[df.Method .== "Reference",:]
plot(dfoo.K, dfoo.Vol .*100, group=dfoo.Method,seriestype=:scatter, markersize=4,xlab="Strike", ylab="Black volatility in %")
plot!(dfo.K, dfo.Vol .*100, group=dfo.Method)
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/aapl_2y_qlv.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/kahale_quadraticlv.txt", ' ', header=true)
dfn = DataFrame(m[1], vec(m[2]))
dfn.Method = @. ifelse(dfn.Method == "QuadraticTS-Average", "Parameter averaging",dfn.Method)
dfn.Method = @. ifelse(dfn.Method == "QuadraticTS-FDM", "Fokker-Planck",dfn.Method)
dfm = dfn[dfn.Expiry .== 0.94 .&& dfn.Method .!= "Quadratic" .&& dfn.Method .!= "SVI",:]
plot(dfm.Strike,dfm.Vol .* 100, group=dfm.Method, xlab="Strike", ylab="Black volatility in %")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/qlv_kahale_average.pdf")

dfm = dfn[dfn.Expiry .== 0.94 .&& dfn.Method .!= "Parameter averaging",:]
plot(dfm.Strike,dfm.VolError .*100, group=dfm.Method, xlab="Strike",ylab="Error in Black volatility %")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/qlv_kahale_fit.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/spx500_1y_2017_ah.txt", '\t', header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.logspot, dfm.Linear .^2, label="Linear")
plot!(dfm.logspot, dfm.Huynh .^2, label="Huynh",xlab="Logarithm of asset price",ylab="Local variance")
plot!(xlims=(7,8),legendtitle="Representation")
plot!(size=(800,400),margins=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/spx500_1y_2017_ah.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/integrand_heston_angle0.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
ind = sortperm(dfm.u)
m = readdlm("/home/fabien/mypapers/eqd_book/integrand_heston_angle1.txt", ' ', header=true)
dfm1 = DataFrame(m[1], vec(m[2]))
ind1 = sortperm(dfm1.u)
plot(dfm.u[ind],dfm.f[ind],ylab="Integrand",xlab="u",label="0")
plot!(dfm1.u[ind1],dfm1.f[ind1],label="π/12")
plot!(xlims=(0.01,1000000),ylims=(-1,2), xaxis=:log10, legendtitle="Angle shift")
plot!(size=(800,400),margins=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/integrand_heston_angle0.pdf")

m = readdlm("/home/fabien/mypapers/filon/kahale_closed_8.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
#ind = sortperm(dfm.u)
dfm.SqrtTheta = sqrt.(dfm.Theta)
dfm.SqrtV0=sqrt.(dfm.V0)
#dfm.LogRMSE=log.(dfm.RMSE)
#dfn = dfm[:,[1,3,5,8,9]]
dfn = dfm
p1 = plot(dfn.Iteration, dfn.Kappa, ylab="κ", label="",color=1, xshowaxis=false)
p2 = plot(dfn.Iteration, dfn.Sigma, ylab="σ", label="",color=2, xshowaxis=false)
p3 = plot(dfn.Iteration,dfn.SqrtTheta,ylab="√θ", label="",color=3, xshowaxis=false)
p4 = plot(dfn.Iteration,dfn.SqrtV0,ylab="√V0", label="",color=4, xshowaxis=false)
p5 = plot(dfn.Iteration,dfn.RMSE,ylab="RMSE", xlab="Iterations",label="",color=5,yaxis=:log10)
plot(p1, p2, p3, p4, p5,layout=(5,1),size=(1000,800),margins=2Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/kahale_closed_8.pdf")

#pyplot()
m = readdlm("/home/fabien/mypapers/cos/double_heston_cos3d.txt", ' ', header=false)
y =  [0.057534247,0.153424658,0.230136986,0.479452055,0.728767123,1.22739726,1.726027397,2.243835616,2.742465753,3.24109589,4.238356164]
x= [1008,1209.6,1411.2,1612.8,1814.4,1915.2,2016,2116.8,2217.6,2419.2,2620.8,3024]
p=plot(y, x, abs.(m'),st=:surface,camera=(50,40),ylab="Strike",xlab="Expiry",zlab="Error", legend=:none, size=(600,600), zguidefontrotation=90,margin=0Plots.mm,zlims=(0,0.02),framestyle=:box)
savefig(p,"/home/fabien/mypapers/eqd_book/double_heston_cos3d.pdf")
m = readdlm("/home/fabien/mypapers/cos/bates_cos3d.txt", ' ', header=false)
p= plot(y, x, abs.(m'),st=:surface,camera=(50,40),ylab="Strike",xlab="Expiry",zlab="Error", legend=:none, size=(600,600), zguidefontrotation=90,margin=0Plots.mm,zlims=(0,0.02),framestyle=:box)
savefig(p, "/home/fabien/mypapers/eqd_book/bates_cos3d.pdf")
m = readdlm("/home/fabien/mypapers/cos/schobelzhu_cos3d.txt", ' ', header=false)
p=plot(y, x, abs.(m'),st=:surface,camera=(50,40),ylab="Strike",xlab="Expiry",zlab="Error", legend=:none, size=(600,600), zguidefontrotation=90,margin=0Plots.mm,zlims=(0,0.02),framestyle=:box)
savefig(p,"/home/fabien/mypapers/eqd_book/schobelzhu_cos3d.pdf")
m = readdlm("/home/fabien/mypapers/cos/heston_cos3d.txt", ' ', header=false)
#there is an extra space in the data
m = m[:,1:12]
p=plot(y, x, abs.(m'),st=:surface,camera=(50,40),ylab="Strike",xlab="Expiry",zlab="Error", legend=:none, size=(600,600), zguidefontrotation=90,margin=0Plots.mm,zlims=(0,0.02),framestyle=:box)
savefig(p,"/home/fabien/mypapers/eqd_book/heston_cos3d.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/heston_qe_case1.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfm.Up = dfm.Error + 1.96 .* dfm.StdErr
dfm.Down = dfm.Error - 1.96 .* dfm.StdErr
dfm.Up = dfm.Price .+ 1.96 .*dfm.StdErr
dfm.Down = dfm.Price .- 1.96 .* dfm.StdErr
dfn=dfm[dfm.dt .<= 0.5,:]
dfo = dfn[(dfn.Method .== "Exact" .|| dfn.Method .== "QE" .|| dfn.Method .== "Euler-FT".|| dfn.Method .== "DVSS2") .&& dfn.Strike .==100,:]
dfo = dfn[(dfn.Method .== "Exact" .|| dfn.Method .== "QE" .|| dfn.Method .== "Euler-FT") .&& dfn.Strike .==100,:]
plot(dfo.dt,dfo.Price,group=dfo.Method,xlab="dt",ylab="Price",xscale=:log10,ribbon=(dfo.Price-dfo.Down,dfo.Up-dfo.Price))
plot!(xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]))
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/heston_ft_price_andersen1_100.pdf")
dfo = dfn[(dfn.Method .== "Exact" .|| dfn.Method .== "QE" .|| dfn.Method .== "Euler-FT") .&& dfn.Strike .==140,:]
plot(dfo.dt,dfo.Price,group=dfo.Method,xlab="dt",ylab="Price",xscale=:log10,ribbon=(dfo.Price-dfo.Down,dfo.Up-dfo.Price))
#plot(dfo.dt,dfo.Price,yerr=1.96.* dfo.StdErr,group=dfo.Method,xlab="dt",ylab="Price",xscale=:log10,msw=2,msc=:auto)
plot!(xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]))
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/heston_ft_price_andersen1_140.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/heston_qe_qmc.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.N, dfm.Error, group=dfm.Method, xlab="N", ylab="Error in Price", xaxis=:log10)
dfn = dfm[dfm.Method .=="Memt" .|| dfm.Method .== "Sobol" .|| dfm.Method .== "Sobol-BB",:]
mycolormap = Dict("Memt"=>1, "Sobol"=>2,"Sobol-BB"=>3)
plot(dfn.N, dfn.Error, group=dfn.Method, xlab="N", ylab="Error in Price", xaxis=:log10,ribbon=dfn.StdErr,color=[mycolormap[me] for me in dfn.Method], fillalpha=0.15,ylims=(-0.02,0.02))
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/heston_qe_qmc_sobol.pdf")
dfn = dfm[dfm.Method .=="Memt" .|| dfm.Method .== "Digital-Equi" .|| dfm.Method .== "Digital-Equi-BB",:]
mycolormap = Dict("Memt"=>1, "Digital-Equi"=>2,"Digital-Equi-BB"=>3)
plot(dfn.N, dfn.Error, group=dfn.Method, xlab="N", ylab="Error in Price", xaxis=:log10,ribbon=dfn.StdErr,color=[mycolormap[me] for me in dfn.Method], fillalpha=0.15,ylims=(-0.02,0.02))
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/heston_qe_qmc_dequi.pdf")
dfn = dfm[dfm.Method .=="Memt" .|| dfm.Method .== "Digital-Memt" .|| dfm.Method .== "Digital-Memt-BB",:]
mycolormap = Dict("Memt"=>1, "Digital-Memt"=>2,"Digital-Memt-BB"=>3)
plot(dfn.N, dfn.Error, group=dfn.Method, xlab="N", ylab="Error in Price", xaxis=:log10,ribbon=dfn.StdErr,color=[mycolormap[me] for me in dfn.Method], fillalpha=0.15,ylims=(-0.02,0.02))
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/heston_qe_qmc_dmemt.pdf")
dfn = dfm[dfm.Method .=="Memt" .|| dfm.Method .== "FaureTezuka" .|| dfm.Method .== "FaureTezuka-BB",:]
mycolormap = Dict("Memt"=>1, "FaureTezuka"=>2,"FaureTezuka-BB"=>3)
plot(dfn.N, dfn.Error, group=dfn.Method, xlab="N", ylab="Error in Price", xaxis=:log10,ribbon=dfn.StdErr,color=[mycolormap[me] for me in dfn.Method], fillalpha=0.15,ylims=(-0.02,0.02))
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/heston_qe_qmc_ft.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/heston_qe_case1.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
#ind = sortperm(dfm.dt)
dfm=dfm[dfm.dt .<= 0.5,:]
dfo=dfm[dfm.Method .== "Exact" .|| dfm.Method .== "QE" .|| dfm.Method .== "QELong",:]
dfn = dfo[dfo.Strike .== 70,:]
p1 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=70"),legend=:none)
dfn = dfo[dfo.Strike .== 100,:]
p2 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=100"),legend=:none)
dfn = dfo[dfo.Strike .== 140,:]
p3 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=true,annotate=((0.5,0.5),"K=140"),xlab="time-step size")
plot(p1,p2,p3,layout=(3,1),size=(300,600),leftmargin=5Plots.mm,bottom_margin=0Plots.mm,top_margin=0Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/heston_qel_price_andersen1.pdf")



m = readdlm("/home/fabien/mypapers/eqd_book/heston_qe_case2.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
#ind = sortperm(dfm.dt)
dfm=dfm[dfm.dt .<= 0.25,:]
dfo=dfm[dfm.Method .== "Exact" .|| dfm.Method .== "QE" .|| dfm.Method .== "QELong",:]
dfn = dfo[dfo.Strike .== 70,:]
p1 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=70"),legend=:none)
dfn = dfo[dfo.Strike .== 100,:]
p2 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=100"),legend=:none)
dfn = dfo[dfo.Strike .== 140,:]
p3 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=true,annotate=((0.5,0.5),"K=140"),xlab="time-step size",legend=:none)
plot(p1,p2,p3,layout=(3,1),size=(300,600),leftmargin=5Plots.mm,bottom_margin=0Plots.mm,top_margin=0Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/heston_qel_price_andersen2.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/heston_qe_case3.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
#ind = sortperm(dfm.dt)
dfm=dfm[dfm.dt .<= 0.25,:]
dfo=dfm[dfm.Method .== "Exact" .|| dfm.Method .== "QE" .|| dfm.Method .== "QELong",:]
dfn = dfo[dfo.Strike .== 70,:]
p1 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=70"),legend=:none)
dfn = dfo[dfo.Strike .== 100,:]
p2 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=100"),legend=:none)
dfn = dfo[dfo.Strike .== 140,:]
p3 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=true,annotate=((0.5,0.5),"K=140"),xlab="time-step size",legend=:none)
plot(p1,p2,p3,layout=(3,1),size=(300,600),leftmargin=5Plots.mm,bottom_margin=0Plots.mm,top_margin=0Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/heston_qel_price_andersen3.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/heston_qe_correlation_andersen1.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
m = readdlm("/home/fabien/mypapers/eqd_book/heston_dvss2_correlation_andersen1.txt", ' ', header=true)
dfn = DataFrame(m[1], vec(m[2]))
df = vcat(dfm,dfn)
ind = sortperm(df.T)
plot(df.T[ind],df.Correlation[ind],group=df.Method[ind],xlab="T",ylab="Correlation")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/heston_qe_correlation_andersen1.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/heston_qe_correlation_andersen2.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
m = readdlm("/home/fabien/mypapers/eqd_book/heston_dvss2_correlation_andersen2.txt", ' ', header=true)
dfn = DataFrame(m[1], vec(m[2]))
df = vcat(dfm,dfn)
ind = sortperm(df.T)
plot(df.T[ind],df.Correlation[ind],group=df.Method[ind],xlab="T",ylab="Correlation")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/heston_qe_correlation_andersen2.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/heston_qe_case1.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
#ind = sortperm(dfm.dt)
dfm=dfm[dfm.dt .<= 0.25,:]
dfo=dfm[dfm.Method .== "Exact" .|| dfm.Method .== "QE" .|| dfm.Method .== "DVSS2" .|| dfm.Method .== "Alfonsi2",:]
mycolormap = Dict("Exact"=>:black, "QE"=>1,"DVSS2"=>2, "Alfonsi2"=>3)
dfn = dfo[dfo.Strike .== 70,:]
p1 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=70"),legend=:none,color=[mycolormap[me] for me in dfn.Method])
dfn = dfo[dfo.Strike .== 100,:]
p2 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=100"),legend=:none,color=[mycolormap[me] for me in dfn.Method])
dfn = dfo[dfo.Strike .== 140,:]
p3 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=true,annotate=((0.5,0.5),"K=140"),xlab="time-step size",color=[mycolormap[me] for me in dfn.Method])
plot(p1,p2,p3,layout=(3,1),size=(300,600),leftmargin=5Plots.mm,bottom_margin=0Plots.mm,top_margin=0Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/heston_qe_price_andersen1.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/heston_qe_case2.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
#ind = sortperm(dfm.dt)
dfm=dfm[dfm.dt .<= 0.25,:]
dfo=dfm[dfm.Method .== "Exact" .|| dfm.Method .== "QE" .|| dfm.Method .== "DVSS2",:]
dfn = dfo[dfo.Strike .== 70,:]
p1 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=70"),legend=:none,color=[mycolormap[me] for me in dfn.Method])
dfn = dfo[dfo.Strike .== 100,:]
p2 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=100"),legend=:none,color=[mycolormap[me] for me in dfn.Method])
dfn = dfo[dfo.Strike .== 140,:]
p3 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=true,annotate=((0.5,0.5),"K=140"),xlab="time-step size",legend=:none,color=[mycolormap[me] for me in dfn.Method])
plot(p1,p2,p3,layout=(3,1),size=(300,600),leftmargin=5Plots.mm,bottom_margin=0Plots.mm,top_margin=0Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/heston_qe_price_andersen2.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/heston_qe_case3.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
#ind = sortperm(dfm.dt)
dfm=dfm[dfm.dt .<= 0.25,:]
dfo=dfm[dfm.Method .== "Exact" .|| dfm.Method .== "QE" .|| dfm.Method .== "DVSS2",:]
dfn = dfo[dfo.Strike .== 70,:]
p1 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=70"),legend=:none,color=[mycolormap[me] for me in dfn.Method])
dfn = dfo[dfo.Strike .== 100,:]
p2 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=100"),legend=:none,color=[mycolormap[me] for me in dfn.Method])
dfn = dfo[dfo.Strike .== 140,:]
p3 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=true,annotate=((0.5,0.5),"K=140"),xlab="time-step size",legend=:none,color=[mycolormap[me] for me in dfn.Method])
plot(p1,p2,p3,layout=(3,1),size=(300,600),leftmargin=5Plots.mm,bottom_margin=0Plots.mm,top_margin=0Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/heston_qe_price_andersen3.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/heston_qe_case_forward_start.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
#ind = sortperm(dfm.dt)
dfm=dfm[dfm.dt .<= 1.0,:]
dfo=dfm[dfm.Method .== "Exact" .|| dfm.Method .== "QE" .|| dfm.Method .== "DVSS2" .|| dfm.Method .== "Alfonsi2",:]
dfn = dfo[dfo.Strike .== 70,:]
p1 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=70"),legend=:none,color=[mycolormap[me] for me in dfn.Method])
dfn = dfo[dfo.Strike .== 100,:]
p2 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=100"),legend=:none,color=[mycolormap[me] for me in dfn.Method])
dfn = dfo[dfo.Strike .== 140,:]
p3 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=true,annotate=((0.5,0.5),"K=140"),xlab="time-step size",color=[mycolormap[me] for me in dfn.Method],legend=:bottomleft)
plot(p1,p2,p3,layout=(3,1),size=(400,800),leftmargin=5Plots.mm,bottom_margin=0Plots.mm,top_margin=0Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/heston_forward_start_dvss2.pdf")
mycolormap = Dict("Exact"=>:black, "QE"=>1,"DVSS2X"=>2,"DVSS2"=>2, "Alfonsi2"=>3)
dfo=dfm[dfm.Method .== "Exact" .|| dfm.Method .== "QE" .|| dfm.Method .== "DVSS2X" .|| dfm.Method .== "Alfonsi2",:]
dfn = dfo[dfo.Strike .== 70,:]
p1 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=70"),legend=:none,color=[mycolormap[me] for me in dfn.Method])
dfn = dfo[dfo.Strike .== 100,:]
p2 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=false,annotate=((0.5,0.5),"K=100"),legend=:none,color=[mycolormap[me] for me in dfn.Method])
dfn = dfo[dfo.Strike .== 140,:]
p3 = plot(dfn.dt,dfn.Price,group=dfn.Method,ribbon=1.96.*dfn.StdErr, fillalpha=0.15,xticks=([1/32,1/16,1/8,1/4,1/2,1],["1/32","1/16","1/8","1/4","1/2","1"]),xaxis=:log10,xshowaxis=true,annotate=((0.5,0.5),"K=140"),xlab="time-step size",color=[mycolormap[me] for me in dfn.Method],legend=:bottomleft)
plot(p1,p2,p3,layout=(3,1),size=(400,800),leftmargin=5Plots.mm,bottom_margin=0Plots.mm,top_margin=0Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/heston_forward_start_dvss2x.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/heston_dvss2_lowvolvol.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.Spot,dfm.Price,group=dfm.Scheme,xlab="Spot",ylab="Digital option price")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/heston_dvss2_lowvolvol.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/black_discrete_sampling.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.Spot,dfm.Price,group=dfm.Scheme,xlab="Spot",ylab="Digital option price")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/black_discrete_sampling.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/schobelzhu_eae_qmc", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn=dfm[dfm.Method .== "Memt" .|| dfm.Method .== "FaureTezuka2" .|| dfm.Method .== "FaureTezuka",:]
plot(dfn.N, dfn.Error, group=dfn.Method,xlab="N", ylab="Error in Price",ylims=(-0.02,0.02),xscale=:log10,ribbon=dfn.StdErr, fillalpha=0.15,)
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/schobelzhu_eae_qmc_ft.pdf")
dfn=dfm[dfm.Method .== "Memt" .|| dfm.Method .== "FaureTezuka2-BB" .|| dfm.Method .== "FaureTezuka-BB",:]
plot(dfn.N, dfn.Error, group=dfn.Method,xlab="N", ylab="Error in Price",ylims=(-0.02,0.02),xscale=:log10,ribbon=dfn.StdErr, fillalpha=0.15,)
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/schobelzhu_eae_qmc_ft_bb.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/slv_comp8.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfo=dfm[dfm.Method .== "Bin" .|| dfm.Method .== "BinSpline" ,:]
dfn=dfm[dfm.Method .== "Reference" ,:]
plot(exp.(dfn.X),dfn.V,label="Reference",seriestype=:scatter,xlab="F",ylab="V",ms=4,markeralpha=0.15)
plot!(exp.(dfo.X),dfo.V,group=dfo.Method)
plot!(xlims=(0,1.8),ylims=(0,0.25))
plot!(size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/slv_comp8_bin.pdf")

dfo=dfm[dfm.Method .== "Bin" .|| dfm.Method .== "NonConform" ,:]
plot(exp.(dfn.X),dfn.V,label="Reference",seriestype=:scatter,xlab="F",ylab="V",ms=4,markeralpha=0.15)
plot!(exp.(dfo.X),dfo.V,group=dfo.Method)
plot!(xlims=(0,1.8),ylims=(0,0.25))
plot!(size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/slv_comp8_nc.pdf")

dfo=dfm[dfm.Method .== "Bin" .|| dfm.Method .== "Particle" ,:]
plot(exp.(dfn.X),dfn.V,label="Reference",seriestype=:scatter,xlab="F",ylab="V",ms=4,markeralpha=0.15)
plot!(exp.(dfo.X),dfo.V,group=dfo.Method)
plot!(xlims=(0,1.8),ylims=(0,0.25))
plot!(size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/slv_comp8_particle15.pdf")

dfo=dfm[dfm.Method .== "Bin" .|| dfm.Method .== "Particle-6" ,:]
plot(exp.(dfn.X),dfn.V,label="Reference",seriestype=:scatter,xlab="F",ylab="V",ms=4,markeralpha=0.15)
plot!(exp.(dfo.X),dfo.V,group=dfo.Method)
plot!(xlims=(0,1.8),ylims=(0,0.25))
plot!(size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/slv_comp8_particle06.pdf")

dfo=dfm[dfm.Method .== "Bin" .|| dfm.Method .== "Particle-6" .|| dfm.Method .== "Particle" ,:]
plot(exp.(dfn.X),dfn.V,label="Reference",seriestype=:scatter,xlab="F",ylab="V",ms=4,markeralpha=0.15)
plot!(exp.(dfo.X),dfo.V,group=dfo.Method)
plot!(xlims=(0,1.8),ylims=(0,0.25))
plot!(size=(400,400))
savefig("/home/fabien/mypapers/eqd_book/slv_comp8_particle.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/schobelzhulv_sobol_convergence.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn = dfm
dfn=dfm[dfm.Method .== "Sobol" .|| dfm.Method .== "Sobol-BB" .|| dfm.Method .== "Memt" ,:]
plot(dfn.N,dfn.Error,group=dfn.Method,xscale=:log10, xlab="N",ylab="Error")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/schobelzhulv_sobol_convergence.pdf")

dfn=dfm[dfm.Method .== "Sobol" .|| dfm.Method .== "Sobol-FaureTezuka-BB" .|| dfm.Method .== "Sobol-FaureTezuka" ,:]
plot(dfn.N,dfn.Error,group=dfn.Method,xscale=:log10, xlab="N",ylab="Error")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/schobelzhulv_sobolft_convergence.pdf")

dfn=dfm[dfm.Method .== "Sobol" .|| dfm.Method .== "Sobol-Digital-Memt-BB" .|| dfm.Method .== "Sobol-Digital-Memt" ,:]
plot(dfn.N,dfn.Error,group=dfn.Method,xscale=:log10, xlab="N",ylab="Error")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/schobelzhulv_sobold_convergence.pdf")

dfn=dfm[dfm.Method .== "Sobol" .|| dfm.Method .== "Memt" .|| dfm.Method .== "Memt-Anti" ,:]
plot(dfn.N,dfn.Error,group=dfn.Method,xscale=:log10, xlab="N",ylab="Error")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/schobelzhulv_sobold_convergence.pdf")

n<-m[m$Method == "Sobol" | m$Method == "Memt" | m$Method=="Memt-Anti",]
qplot(x=N,y=Error,data=n,linetype=Method,geom="line")+scale_x_log10(breaks=c(1024,4096,16384,65536,262144))+theme_bw()+theme(panel.border = element_rect(colour = "black"),panel.grid.minor=element_blank(),legend.justification = c(1, 1), legend.position = c(1, 1))+scale_linetype_manual(values=c(1,2,3))
ggsave(file="/home/fabien/mypapers/eqd_book/schobelzhulv_memt_convergence.eps",width=4,height=4)


m = readdlm("/home/fabien/mypapers/eqd_book/forward_start_hestonlv.txt", ' ', header=true)
dfm = DataFrame(m[1], vec(m[2]))

#dfn=dfm[dfm.Method .== "Sobol" .|| dfm.Method .== "Sobol-BB" .|| dfm.Method .== "Memt" ,:]
df1 = dfm[dfm.Model .== "Black",:]
df2 =  dfm[dfm.Model .== "Black-Static",:]
df3 =  dfm[dfm.Model .== "LocalVol",:]
df4 = dfm[dfm.Model .== "LocalVol-Frozen",:]
df5 = dfm[dfm.Model .== "Heston",:]
df1.Difference = df1.Price - df5.Price
df1.RelativeDifference = df1.Price ./ df5.Price .- 1
df2.Difference = df2.Price - df5.Price
df2.RelativeDifference = df2.Price ./ df5.Price .- 1
df3.Difference = df3.Price - df5.Price
df3.RelativeDifference = df3.Price ./ df5.Price .- 1
df4.Difference = df4.Price - df5.Price
df4.RelativeDifference = df4.Price ./ df5.Price .- 1
df5.Difference .= 0.0
df5.RelativeDifference .= 0.0
dfn = vcat(df1,df2,df3,df4)
# + facet_wrap(~ T1,scales="free")+theme_minimal(base_family="Libertinus Sans",base_size=14)+theme(axis.text=element_text(color="black"),strip.text=element_text(color="black"),legend.position="bottom")+scale_color_grey()
ps = []
for t = unique(dfn.T1) 
  dfo = dfn[dfn.T1 .== t,:]
  p = plot(dfo.Strike, dfo.RelativeDifference, group=dfo.Model,xlab="Strike",ylab="Relative difference",annotate=((0.5,0.5),string("T1=",t)))
  if t != 0.5
    plot!(p,legend=:none)
  end
  push!(ps,p)
end
plot(ps[1],ps[2],ps[3], layout=(1,3),margin=3Plots.mm)
plot!(size=(800,300))
savefig("/home/fabien/mypapers/eqd_book/forward_start_heston_rel.pdf")


strike=6.0
forward=5.996
vol=range(0.001,stop=0.3,step=0.001)
t=1.0
d = @. (log(forward/strike)-vol^2*t/2)/(vol*sqrt(t))
put = @. AQFED.Math.norminv(-d)
plot(vol.*100,put,xlab="Black volatility in %", ylab="Digital Put Price",legend=:none)
plot!(size=(800,300),margin=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/digital_impliedvol.pdf")

r = -0.02/100
q = -0.22/100
s = range(0.0,stop=0.12, step=0.001)
plot(s,@.(0.25*s^4+(r+q)*s^2+(r-q)^2),ylab="λ²",xlab="σ",legend=:none)
plot!(xticks=([0,2.5,5.0,7.5,10.0,12.5]./100,["0.0%","2.5%","5.0%", "7.5%", "10.0%", "12.5%"]))
plot!(size=(800,300),margin=5Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/lambda.pdf")

m = readdlm(IOBuffer(replace(read("/home/fabien/mypapers/eqd_book/barrier_termstructure_5y.txt",String),r"[^\S\r\n]+" => ";")), ';',header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfm.Error .= 0.0
dfn=dfm[dfm.Method .== "Black",:]
dfn.Error = dfn.Price .- dfm[dfm.Method .== "TR-BDF2",:].Price
dfn.RelError = dfn.Price ./ dfm[dfm.Method .== "TR-BDF2",:].Price .- 1.0
p1=plot(dfn.Level,dfn.Price,label="Price",ylab="Price", xlab="Barrier Level",size=(800,300),legend=:bottomleft,margin=5Plots.mm)
plot!(twinx(),dfn.Level,dfn.RelError,label="Relative error",color=2,xticks=:none,legend=:bottomright,ylab="Error",yticks=([0,-0.05,-0.10,-0.15,-0.2],["0%","-5%","-10%","-15%","-20%"]))
savefig("/home/fabien/mypapers/eqd_book/barrier_termstructure_5y.pdf")

m = readdlm(IOBuffer(replace(read("/home/fabien/mypapers/eqd_book/barrier_cashdividend_1y.txt",String),r"[^\S\r\n]+" => ";")), ';',header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.ExDate,dfm.Price,group=dfm.Method,ylab="Price",xlab="Dividend ex-date")
plot!(size=(800,300),margins=5Plots.mm,legend=:bottomleft)
savefig("/home/fabien/mypapers/eqd_book/barrier_cashdividend_1y.pdf")

m = readdlm(IOBuffer(replace(read("/home/fabien/mypapers/eqd_book/barrier_smile.txt",String),r"[^\S\r\n]+" => ";")), ';',header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn=dfm[dfm.Method .!= "LMG2",:]
dfn.Method = @. ifelse(dfn.Method == "TR-BDF2", "Local vol",dfn.Method)
plot(dfn.Level,dfn.Price,group=dfn.Method, ylab="Price", xlab="Barrier Level")
plot!(size=(400,300),margins=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/barrier_smile.pdf")
m = readdlm(IOBuffer(replace(read("/home/fabien/mypapers/eqd_book/barrier_zeliade4.txt",String),r"[^\S\r\n]+" => ";")), ';',header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn=dfm[dfm.Method .!= "LMG2",:]
dfn.Method = @. ifelse(dfn.Method == "TR-BDF2", "Local vol",dfn.Method)
plot(dfn.Level,dfn.Price,group=dfn.Method, ylab="Price", xlab="Barrier Level")
plot!(size=(400,300),margins=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/barrier_zeliade4.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/deelstra_hagspihl_correlation.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.rho, abs.(dfm.Ju)./dfm.price, label="Ju",xlab="Correlation",ylab="Relative error")
plot!(dfm.rho,abs.(dfm.Deelstra)./dfm.price,label="Deelstra FA3 s=3")
plot!(dfm.rho, abs.(dfm."Deelstra-12")./dfm.price,label="Deelstra FA2 s=1")
plot!(dfm.rho, abs.(dfm."Deelstra-LB")./dfm.price,label="Curran lower bound")
plot!(yscale=:log10)
plot!(size=(400,400),margins=1Plots.mm,legend=:bottomleft,ylims=(1e-10,1))
plot!(yticks=[1e-10,1e-5,1e-4,1e-3,1e-2,1e-1,1.0])
plot!(rhos, abs.(sabrPrices./refPrices.-1), label="SABR single") #see sabr_test.jl
plot!(rhos, abs.(sabrIPrices./refPrices.-1), label="SABR independent")
savefig("/home/fabien/mypapers/eqd_book/deelstra_correlation.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/deelstra_krekel_vol100.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.sigma.*100, dfm.Ju .+ dfm.price, label="Ju",xlab="Volatility in %",ylab="Price")
plot!(dfm.sigma.*100, dfm.MM3 .+ dfm.price, label="MM3")
plot!(dfm.sigma.*100, dfm.Deelstra .+ dfm.price, label="Deelstra FA2 s=3")
plot!(dfm.sigma.*100, dfm.VL2 .+ dfm.price, label="VL2")
plot!(dfm.sigma.*100, dfm.VL3 .+ dfm.price, label="VL3")
plot!(dfm.sigma.*100, dfm.price, label="QMC")
plot!(size=(400,400),margins=1Plots.mm,legend=:bottomright)
savefig("/home/fabien/mypapers/eqd_book/deelstra_vol100.pdf")

Plots.plot(dfm.sigma .* 100, abs.(dfm.Ju)./dfm.price, label="Ju",xlab="Volatility in %",ylab="Relative error")
Plots.plot!(dfm.sigma .* 100, abs.(dfm.Deelstra23)./dfm.price, label="Deelstra FA2 s=3")
Plots.plot!(dfm.sigma .* 100, abs.(dfm.Deelstra13)./dfm.price, label="Deelstra FA1 s=3")
Plots.plot!(dfm.sigma .* 100, abs.(dfm.Deelstra21)./dfm.price, label="Deelstra FA2 s=1")
Plots.plot!(dfm.sigma .* 100, abs.(dfm.DeelstraLB)./dfm.price, label="Curran lower bound")
Plots.plot!(yscale=:log10,yticks=[1e-3,1e-2,1e-1,1])
Plots.plot!(size=(800,400),margins=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/deelstra_vol100_err.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/deelstra_krekel_sabr_vol100_2y.txt",header=true) 
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.Rho, dfm.Ju.*100, label="Ju")
plot!(dfm.Rho, dfm.Deelstra.*100, label="Deelstra")
plot!(dfm.Rho, dfm.SABRs.*100, label="SABR single")
plot!(dfm.Rho, dfm.SABRi.*100, label="SABR independent")
plot!(xlab="Correlation",ylab="Price")
plot!(size=(400,300),margins=1Plots.mm,legend=:bottomright)
savefig("/home/fabien/mypapers/eqd_book/deelstra_sabr_vol100_2y.pdf")

## quanto
spotEquity = 1.0
tte = 2.0
betaEquity = 1.0/16
volEquity = 28.98/100
betaFX = 1
volFX = 10.0/100
correlation = 50/100
spotFX=1.0
strike = 0.05:0.05:1.6
c=correlation*volEquity*volFX*betaEquity*betaFX*tte
cApprox=correlation*volEquity*volFX*betaEquity*tte
quantoExact = @. 1.0/betaFX* AQFED.Black.blackScholesFormula(true, strike+(1.0/betaEquity-1)*spotEquity,spotEquity/betaEquity*exp(-c), (volEquity*betaEquity)^2*tte,1.0,1.0) - (1.0/betaFX-1)* AQFED.Black.blackScholesFormula(true, strike+(1.0/betaEquity-1)*spotEquity,spotEquity/betaEquity, (volEquity*betaEquity)^2*tte,1.0,1.0)
quantoApprox= @. AQFED.Black.blackScholesFormula(true, strike+(1.0/betaEquity-1)*spotEquity,spotEquity/betaEquity*exp(-cApprox), (volEquity*betaEquity)^2*tte,1.0,1.0) 
##black approx
blackPriceEquity =  @.  AQFED.Black.blackScholesFormula(true, strike+(1-betaEquity)/betaEquity*spotEquity,spotEquity/betaEquity, (volEquity*betaEquity)^2*tte,1.0,1.0) 
blackVolEquity= @. AQFED.Black.impliedVolatility(true,blackPriceEquity,spotEquity,strike,tte,1.0)
blackPriceEquityATM = AQFED.Black.blackScholesFormula(true, spotEquity+(1-betaEquity)/betaEquity*spotEquity,spotEquity/betaEquity, (volEquity*betaEquity)^2*tte,1.0,1.0) 
blackVolEquityATM=@. AQFED.Black.impliedVolatility(true,blackPriceEquityATM,spotEquity,spotEquity,tte,1.0)

blackPriceFX = spotFX*AQFED.Black.blackScholesFormula(true, 1/spotFX+(1-betaFX)/betaFX*spotFX,spotFX/betaFX, (volFX*betaFX)^2*tte,1.0,1.0) 
blackVolFX = AQFED.Black.impliedVolatility(true,blackPriceFX,spotFX,spotFX,tte,1.0)

cBlack = correlation*blackVolEquity*blackVolFX*tte
cBlackATM = correlation*blackVolEquityATM*blackVolFX*tte
quantoBlack=@.AQFED.Black.blackScholesFormula(true, strike,spotEquity*exp(-cBlackATM), (blackVolEquity)^2*tte,1.0,1.0) 
quantoBlackATM=@. AQFED.Black.blackScholesFormula(true, strike,spotEquity*exp(-cBlackATM), (blackVolEquityATM)^2*tte,1.0,1.0) 

#+geom_line(aes(strike,quantoApprox,colour="Approx",linetype="Approx"))
plot(strike, quantoExact,linestyle=:auto, label="Exact", ylab="Price", xlab="Strike")
plot!(strike,quantoBlack,linestyle=:auto,label="Black")
plot!(strike,quantoBlackATM,linestyle=:auto,label="BlackATM")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/quanto_skew_price.pdf")
plot(strike, (quantoBlack.-quantoExact)./quantoExact, label="Black",linestyle=:auto,ylab="Relative error in price", xlab="Strike")
plot!(strike, (quantoBlackATM .-quantoExact)./quantoExact,label="BlackATM")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/quanto_skew_error.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/schobelzhu_quanto.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn = dfm[dfm.Model .!= "Schobel-Zhu",:]
plot(dfn.Strike,dfn.Error, group=dfn.Model, ylab="Absolute error in price")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/schobelzhu_quanto_abs.pdf")
 plot(dfn.Strike,dfn.RelError.*100, group=dfn.Model,ylab="Relative error in price in %")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/schobelzhu_quanto_rel.pdf")


K=60.0:10.0:140.0
F=100.0
x=60.0:0.1:140.0
plot(K,K ./F .- 1 .- log.(K./F), label="Piecewise-linear", xlab="Strike",ylab="Value")
plot!(x,x./F .-1 .- log.(x./F),label="f",linestyle=:auto)
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/varswap_derman.pdf")

m = readdlm("/home/fabien/mypapers/volswap/varswap_adapt_8.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
m = readdlm("/home/fabien/mypapers/volswap/varswap_trap_8.txt",header=true)
dfn = DataFrame(m[1], vec(m[2]))
ind = sortperm(dfn.K);
plot(dfn.K[ind],dfn.value[ind],label="Trapezoidal")
ind = sortperm(dfm.K);
plot!(dfm.K[ind],dfm.value[ind], label="integrand",xlab="Strike",ylab="Value",linestyle=:auto)
plot!(xlim=(60,140),size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/varswap_trap.pdf")


m = readdlm("/home/fabien/mypapers/volswap-mdpi/bates_schoutens.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfm = dfm[dfm.Method .!= "LocalVol",:]
plot(dfm.k,dfm.VarSqrt,group=dfm.Method, xlab="jump mean k", ylab="Price in volatility")
plot!(ylims=(30.8,33),size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/bates_schoutens_var.pdf")
plot(dfm.k,dfm.Vol,group=dfm.Method,xlab="jump mean k", ylab="Price in volatility")
plot!(ylims=(30.8,33),size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/bates_schoutens_vol.pdf")

m = readdlm("/home/fabien/mypapers/volswap-mdpi/bates_broadie.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfm = dfm[dfm.Method .!= "LocalVol",:]
plot(-dfm.a,dfm.VarSqrt,group=dfm.Method, xlab="jump mean alpha", ylab="Price in volatility")
plot!(ylims=(11.5,15),size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/bates_broadie_var.pdf")
plot(-dfm.a,dfm.Vol,group=dfm.Method,xlab="jump mean alpha", ylab="Price in volatility")
plot!(ylims=(11.5,15),size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/bates_broadie_vol.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/schobelzhu_quanto_raw.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfm[dfm.Method .!= "LocalVol",:].Price = dfm[dfm.Method .!= "LocalVol",:].Price .* (365 .* 8) ./ (365 .* 8 .-1)
plot(dfm.Correlation,dfm.Price,group=dfm.Method, xlab="Correlation", ylab="Price")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/quanto_schobelzhu_10.pdf")
m = readdlm("/home/fabien/mypapers/eqd_book/schobelzhu_quanto_calib.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfm[dfm.Method .!= "LocalVol",:].Price = dfm[dfm.Method .!= "LocalVol",:].Price .* (365 .* 8) ./ (365 .* 8 .-1)
plot(dfm.Correlation,dfm.Price,group=dfm.Method, xlab="Correlation", ylab="Price")
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/quanto_schobelzhu_10c.pdf")

m = readdlm("/home/fabien/mypapers/volswap/Bos.txt",header=true)
m = readdlm("/home/fabien/mypapers/volswap/FDM.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn = dfm[dfm.Logmoneyness .> -0.4 .&& dfm.Logmoneyness .< 0.4,:]
Expiries =unique(dfn.Expiry)
Logmoneyness = unique(dfn.Logmoneyness)
dfn.BlackVolatility .*= 100
m=Matrix(unstack(dfn, :Logmoneyness,:Expiry,:BlackVolatility))[1:end,2:end]
p1=surface(Logmoneyness, Expiries, (m-m1)', xlab="Log-moneyness",ylab="Expiry",zlab="Black Volatility in %",legend=:none,framestyle=:box,camera=(45,30), zguidefontrotation=90,margin=0Plots.mm,colorbar=:none,xguidefontrotation=-30,size=(400,400))
Plots.savefig("/home/fabien/mypapers/eqd_book/bosfdm_flat_interpolation_2.pdf")
"/home/fabien/mypapers/eqd_book/bosfdm_flat_interpolation_2.pdf"
p1=surface(Logmoneyness, Expiries, (m)', xlab="Log-moneyness",ylab="Expiry",zlab="Black Volatility in %",legend=:none,framestyle=:box,camera=(45,30), zguidefontrotation=90,margin=0Plots.mm,colorbar=:none,xguidefontrotation=-30,size=(400,400))
Plots.savefig("/home/fabien/mypapers/eqd_book/bos_flat_interpolation_2.pdf")

k = 0.1
v = 0.001^2
h = function(z) exp(-(4*z*z+1)/8*v)*(z*cos(z*k)-0.5*sin(z*k))*2/sqrt(8*z*z+2) end
x = 0.0:0.1:5000.0
plot(x,h.(x),xlab="x",ylab="h(x)",legend=:none,size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/integrand_fourier_ksmall.pdf")
v = 0.1*0.1
x = 0.0:0.1:100.0
plot(x,h.(x),xlab="x",ylab="h(x)",legend=:none,size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/integrand_fourier_ksmall_vlarge.pdf")

m = readdlm("/home/fabien/mypapers/volswap/volswap_return_s_filon.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.variance.*10000, dfm.price, legend=:none, xlab="accumulated variance ×10000", ylab="volatility swap price",size=(320,320))
savefig("/home/fabien/mypapers/eqd_book/volswap_return_s_filon.pdf")
m = readdlm("/home/fabien/mypapers/volswap/volswap_return_l_filon.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.variance, dfm.price, legend=:none, xlab="accumulated variance ×1", ylab="volatility swap price",size=(320,320))
savefig("/home/fabien/mypapers/eqd_book/volswap_return_l_filon.pdf")
m = readdlm("/home/fabien/mypapers/volswap/volswap_return_s_adaptsim.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
p = plot(dfm.variance.*10000, dfm.price, legend=:none, xlab="accumulated variance ×10000", ylab="volatility swap price",size=(320,320))
savefig("/home/fabien/mypapers/eqd_book/volswap_return_s_adaptsim.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/vix_future_heston.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.Maturity, dfm.Price, group=dfm.Method, xlab="Maturity", ylab="Price",size=(400,300),legend=:bottomright)
savefig("/home/fabien/mypapers/eqd_book/vix_future_heston.pdf")
m = readdlm("/home/fabien/mypapers/eqd_book/vix_future_heston_vv.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.VolOfVar, dfm.Price, group=dfm.Method, xlab="Vol of var", ylab="Price",size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/vix_future_heston_vv.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/vix170419_1m.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn = dfm[dfm.Model .!= "Reference" .&& dfm.Model .!="Parabola" .&& dfm.Model .!= "SABR",:]
dfn = dfm[dfm.Model .!= "Reference" .&& dfm.Model .!="Parabola",:]
dfo = dfm[dfm.Model .== "Reference",:]
ind = sortperm(dfn.Strike)
plot(dfo.Strike, dfo.Volatility .* 100, group=dfo.Model,seriestype=:scatter,  xlab="Strike", ylab="Volatility in %",ms=4,markerstrokewidth=0,markeralpha=0.8)
plot!(dfn.Strike[ind],dfn.Volatility[ind] .* 100, group=dfn.Model[ind])
plot!(size=(800,400),margins=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/vix170419_1m.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/spxw170421_1m_EQ.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn = dfm[dfm.Model .!= "Reference" .&& dfm.Model .!="Parabola" .&& dfm.Model .!= "SABR",:]
dfn = dfm[dfm.Model .!= "Reference" .&& dfm.Model .!="Parabola",:]
dfo = dfm[dfm.Model .== "Reference",:]
ind = sortperm(dfn.Strike)
plot(dfo.Strike, dfo.Volatility .* 100, group=dfo.Model,seriestype=:scatter,  xlab="Strike", ylab="Volatility in %",ms=4,markerstrokewidth=0,markeralpha=0.8)
plot!(dfn.Strike[ind],dfn.Volatility[ind] .* 100, group=dfn.Model[ind])
plot!(size=(800,400),margins=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/spxw170421_1m_eq.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/holee_autocall.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
ind = sortperm(dfm.Correlation)
plot(dfm.Correlation[ind],dfm.Price[ind] .* 100, group=dfm.Model[ind], xlab="Correlation", ylab="Price in %",size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/holee_autocall.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/heston_autocall.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
ind = sortperm(dfm.Strike)
plot(dfm.Strike[ind],dfm.Price[ind] .* 100, group=dfm.Model[ind], xlab="Strike", ylab="Price in %",size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/heston_autocall.pdf")

eigen=[0.11106295619505602, 0.18972542661377423, 0.2464735118199204, 0.3489278865668937, 0.43301516887262687, 0.4488071836403987, 0.628673634428382, 0.677395457830409, 2.3174466153150517, 4.59847215871749]
eigenb=[0.03270750048573754,0.07136055122094749,0.08816331196074648, 0.16747855944850265, 0.17878925736728732, 0.2990924188266631, 1.0694235527066924, 1.9252298518436854, 2.428959853333082, 3.7387951428066564]
x= 1:10
plot(x,eigen,label="C1",ylab="Eigenvalue")
plot!(x,eigenb,label="C2")
plot!(size=(800,400),margins=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/altiplano_eigenvalues.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/altiplano_heston.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.Correlation,dfm.Price, group=dfm.Model, xlab="Mixing parameter (lambda)", ylab="Price in %")
plot!(ylims=(2.25,3.25),size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/altiplano_heston.pdf")
m = readdlm("/home/fabien/mypapers/eqd_book/altiplano_heston_b.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.Correlation,dfm.Price, group=dfm.Model, xlab="Mixing parameter (lambda)", ylab="Price in %")
plot!(ylims=(2.25,3.25),size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/altiplano_heston_b.pdf")


m = readdlm("/home/fabien/mypapers/eqd_book/kahale_ah_constant_200_lm.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
expiries = [0.175,0.425,0.695,0.94,1.0,1.5,2.0,3.0,4.0,5.0] 
p1 = plot(xlab="Strike",ylab="Discrete local volatility",xlims=(450,850),legendtitle="Maturity",palette=:Spectral_10)
for (i,expiry) = enumerate(expiries)
  plot!(p1, exp.(parse.(Float64,names(dfm))), Matrix(dfm)[i,:] .* 100, label=string(expiry))
end
plot!(p1, size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/kahale_ah_constant_200_lm.pdf")
m = readdlm("/home/fabien/mypapers/eqd_book/kahale_ah_constant_400_lm.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
expiries = [0.175,0.425,0.695,0.94,1.0,1.5,2.0,3.0,4.0,5.0] 
p1 = plot(xlab="Strike",ylab="Discrete local volatility",xlims=(450,850),legendtitle="Maturity",palette=:Spectral_10)
for (i,expiry) = enumerate(expiries)
  plot!(p1, exp.(parse.(Float64,names(dfm))), Matrix(dfm)[i,:] .* 100, label=string(expiry))
end
plot!(p1, size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/kahale_ah_constant_400_lm.pdf")
m = readdlm("/home/fabien/mypapers/eqd_book/kahale_ah_linear_400_lm.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
expiries = [0.175,0.425,0.695,0.94,1.0,1.5,2.0,3.0,4.0,5.0] 
p1 = plot(xlab="Strike",ylab="Discrete local volatility",xlims=(450,850),legendtitle="Maturity",palette=:Spectral_10)
for (i,expiry) = enumerate(expiries)
  plot!(p1, exp.(parse.(Float64,names(dfm))), Matrix(dfm)[i,:] .* 100, label=string(expiry))
end
plot!(p1, size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/kahale_ah_linear_400_lm.pdf")
m = readdlm("/home/fabien/mypapers/eqd_book/kahale_ah_linear_200_lm.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
expiries = [0.175,0.425,0.695,0.94,1.0,1.5,2.0,3.0,4.0,5.0] 
p1 = plot(xlab="Strike",ylab="Discrete local volatility",xlims=(450,850),legendtitle="Maturity",palette=:Spectral_10)
for (i,expiry) = enumerate(expiries)
  plot!(p1, exp.(parse.(Float64,names(dfm))), Matrix(dfm)[i,:] .* 100, label=string(expiry))
end
plot!(p1, size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/kahale_ah_linear_200_lm.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/flat_ah_linear_400_lm.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
expiries = [0.175,0.425,0.695,0.94,1.0,1.5,2.0,3.0,4.0,5.0] 
p1 = plot(xlab="Strike",ylab="Discrete local volatility",xlims=(450,850),legendtitle="Maturity",palette=:Spectral_10)
for (i,expiry) = enumerate(expiries)
  plot!(p1, exp.(parse.(Float64,names(dfm))), Matrix(dfm)[i,:] .* 100, label=string(expiry))
end
plot!(p1, size=(800,400))
savefig("/home/fabien/mypapers/eqd_book/flat_ah_linear_400_lm.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/ah_flat_calibrated_longstep.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
df25 = dfm[dfm."Time-step" .== 0.25,:]
df50c = dfm[dfm."Time-step" .== 0.5 .&& dfm.Method .== "Long-step-calibrated",:]
plot(df25.Strike, abs.(df25.RelError) .*100, group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Relative error in %",yscale=:log10)
plot!(df50c.Strike, abs.(df50c.RelError) .*100, label="Long-step-calibrated Δt=0.5")
plot(df25.Strike, (df25.Error) , group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Error in % of spot")
plot!(df50c.Strike, (df50c.Error), label="Long-step-calibrated Δt=0.5")
plot!(legend=:outerbottom, size=(400,400))
plot(df25.Strike, abs.(df25.Error) , group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Error in % of spot",yscale=:log10)
plot!(df50c.Strike, abs.(df50c.Error), label="Long-step-calibrated Δt=0.5")
plot!(legend=:outerbottom, size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/ah_flat_calibrated_longstep.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/ah_flat_calibrated_longstep_vanilla.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
df25 = dfm[dfm."Time-step" .== 0.25,:]
df50c = dfm[dfm."Time-step" .== 0.5 .&& dfm.Method .== "Long-step-calibrated",:]
plot(df25.Strike, abs.(df25.RelError) .*100, group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Relative error in %",yscale=:log10)
plot!(df50c.Strike, abs.(df50c.RelError) .*100, label="Long-step-calibrated Δt=0.5")
plot(df25.Strike, (df25.Error) , group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Error in % of spot")
plot!(df50c.Strike, (df50c.Error), label="Long-step-calibrated Δt=0.5")
plot(df25.Strike, abs.(df25.Error) , group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Error in % of spot",yscale=:log10)
plot!(df50c.Strike, abs.(df50c.Error), label="Long-step-calibrated Δt=0.5")
plot!(legend=:outerbottom, size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/ah_flat_calibrated_longstep_vanilla.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/ah_zeliade_calibrated_longstep.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
df25 = dfm[dfm."Time-step" .== 0.25,:]
plot(df25.Strike, abs.(df25.RelError) .*100, group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Relative error in %",yscale=:log10)
plot(df25.Strike, (df25.Error) , group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Error in % of spot")
plot(df25.Strike, abs.(df25.Error) , group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Error in % of spot",yscale=:log10)
plot(df25.Strike, (df25.Error) , group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Error in % of spot")
plot!(legend=:outerbottom, size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/ah_zeliade_calibrated_longstep.pdf")

m = readdlm("/home/fabien/mypapers/eqd_book/ah_zeliade_calibrated_longstep_vanilla.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
df25 = dfm[dfm."Time-step" .== 0.25,:]
plot(df25.Strike, abs.(df25.RelError) .*100, group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Relative error in %",yscale=:log10)
plot(df25.Strike, (df25.Error) , group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Error in % of spot")
plot(df25.Strike, abs.(df25.Error) , group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Error in % of spot",yscale=:log10)
plot(df25.Strike, (df25.Error) , group=string.(df25.Method," Δt=0.25"), xlab="Strike",ylab="Error in % of spot")
plot!(legend=:outerbottom, size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/ah_zeliade_calibrated_longstep_vanilla.pdf")


#### dividend yield
S = 100.0
C = function(t) 1.0 end
cashYield = function (t) 
	sum = 0.0
	if (t >= 0.5) sum = sum+0.5  end
	if (t >= 1) sum=sum+ 0.5 end
	if (t >= 1.5) sum =sum+ 0.5 end
	if (t >= 2.0) sum = sum +0.5 end
	-1/(t)*log(1-sum/(S*C(t)))
end

propYield = function (t) 
	sum = 1.0
	if (t >= 0.5) sum = sum * (1-0.5/100) end
	if (t >= 1) sum= sum * (1-0.5/100) end
	if (t >= 1.5) sum =sum * (1-0.5/100) end
	if (t >= 2.0) sum = sum * (1-0.5/100) end
	-1/(t)*log(sum)
end

t = range(0,stop=2.0,step=0.01)
cashYield100 = 100*cashYield.(t)
propYield100 = 100*propYield.(t)
S=200
cashYield200 = 100*cashYield.(t)
propYield200 = 100*propYield.(t)

plot(t,cashYield100, label="100", ylab="Dividend yield in %", xlab="Time")
plot!(t,cashYield200, label="200",legendtitle="S(0)")
plot!(size=(400,300))
savefig("~/mypapers/eqd_book/yield_cash.pdf")
plot(t,propYield100, label="100", ylab="Dividend yield in %", xlab="Time")
plot!(t,propYield200, label="200",legendtitle="S(0)")
plot!(size=(400,300))
savefig("~/mypapers/eqd_book/yield_prop.pdf")

S=100
plot(t, cashYield.(t) .* (- t), ylab="- q * t", xlab="Time", legend=:none)
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/logyield_cash_100.pdf")

using AlgebraOfGraphics, CairoMakie
f = Figure(size=(400,300))
ax = Axis(f[1, 1],xlabel="Time",ylabel="Dividend yield in %", xticks=LinearTicks(5), yticks=LinearTicks(5), yminorgridvisible = true)
lines!(ax, t, cashYield100, label="100")
lines!(ax, t, cashYield200, label="200")
axislegend("S(0)",position=:rb)
save("yield_cash.pdf", f)
f = Figure(size=(400,300))
ax = Axis(f[1, 1],xlabel="Time",ylabel="Dividend yield in %", xticks=LinearTicks(5), yticks=LinearTicks(5), yminorgridvisible = true)
lines!(ax, t, propYield100, label="100")
lines!(ax, t, propYield200, label="200")
axislegend("S(0)",position=:rb)
save("yield_prop.pdf", f)
S=100

f = Figure(size=(400,300))
ax = Axis(f[1, 1],xlabel="t",ylabel="-q * t", xticks=LinearTicks(5), yticks=LinearTicks(5), yminorgridvisible = true)
lines!(ax, t,  cashYield.(t) .* (- t))
save("logyield_cash_100.pdf", f)

plot(t, cashYield.(t) .* (- t), ylab="- q * t", xlab="Time", legend=:none)
plot!(size=(400,300))

#set_theme!(palette = (color = my_colors, marker = my_markers))

m = readdlm("/home/cian/mypapers/eqd_book/price_continuity_01.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfm[!,:Color] .= "Red"
using AlgebraOfGraphics, CairoMakie

Price = :Price => (t -> t) => "Option Price"
T = :T => (t -> t) => "Option maturity"

plt = data(dfm) * mapping(T, Price, color = :Color => "") *visual(Lines) 
f = Figure(size=(400,300))
draw!(f, plt,axis=(limits=(0.45,0.55,8,9),yminorgridvisible=true,xticks=LinearTicks(6),yticks=LinearTicks(5),))
save("price_continuity_01.pdf",f)

save("/home/niamh/mypapers/eqd_book/single_r6_strike100_td.pdf", fig)

plot(dfm.T, dfm.Price, ylab="Option price",xlab="Option maturity",legend=:none, xlim=(0.45,0.55),ylim=(8.0,9.0))
plot!(size=(400,300))
savefig("/home/fabien/mypapers/eqd_book/price_continuity_01.pdf")


m = readdlm("hhl_greeks_calibrated.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
plotd = plot(dfm.S, dfm.Delta, ylab="Δ", linestyle=:auto, group=dfm.Model, xlab="")
plotg = plot(dfm.S, dfm.Gamma, ylab="Γ", linestyle=:auto,group=dfm.Model, xlab="",legend=:none)
plotv =  plot(dfm.S, dfm.Vega, ylab="Vega", linestyle=:auto,group=dfm.Model, xlab="",legend=:none)
plot(plotd,plotg,plotv, layout=(3,1),size=(800,800))
savefig("/home/fabien/mypapers/eqd_book/hhl_greeks_calibrated_dgv.pdf")

using AlgebraOfGraphics, CairoMakie
f=Figure(size=(800,800))
ax1 = Axis(f[1,1],xlabel="S", ylabel="Delta",xticks=LinearTicks(5), yticks=LinearTicks(5), yminorgridvisible = true)
ax2 = Axis(f[2,1],xlabel="S", ylabel="Gamma",xticks=LinearTicks(5), yticks=LinearTicks(5), yminorgridvisible = true)
ax3 = Axis(f[3,1],xlabel="S", ylabel="Vega",xticks=LinearTicks(5), yticks=LinearTicks(5), yminorgridvisible = true)

S = :S => (t -> t) => ""
Delta = :Delta => (t -> t) => "Delta"
Gamma = :Gamma => (t -> t) => "Gamma"
Vega = :Vega => (t -> t) => "Vega"
plt1 = data(dfm) * mapping(:S, :Delta, color = :Model, linestyle = :Model) *visual(Lines) 
plt2 = data(dfm) * mapping(:S, :Gamma, color = :Model, linestyle = :Model) *visual(Lines) 
plt3 = data(dfm) * mapping(:S, :Vega, color = :Model, linestyle = :Model) *visual(Lines) 
grid1 = draw!(ax1, plt1)
legend!(f[1,1],grid1;tellheight=false, tellwidth=false, halign=:right, valign=:bottom)
draw!(ax2, plt2)
draw!(ax3, plt3)
save("hhl_greeks_calibrated_dgv.pdf",f)
#(yminorgridvisible=true,xticks=LinearTicks(7),yticks=LinearTicks(6),
#save("price_continuity_01.pdf",f)


m = readdlm("lehman_hhl.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfm[!, :VolError] .= Float64.(dfm.VolError)
dfm[!, :S] .= Float64.(dfm.S)
plot(dfm.S, dfm.VolError .* 100, data=m, linestyle=:auto, group=dfm.Model, xlab="S(0)",ylab="Error in Black volatility %")
plot!(size=(800,350),margins=3Plots.mm)
savefig("/home/fabien/mypapers/eqd_book/hhl_lehman_volerror.pdf")
f=Figure(size=(800,350))
VolError = :VolError => (t -> t * 100) => "Error in Black volatility %"
S = :S => (t -> t) => "S(0)"
plt1 = data(dfm) * mapping(S, VolError, color = :Model, linestyle = :Model) * visual(Lines) 
f = draw( plt1, axis=(width=800,height=300,yminorgridvisible=true,xticks=LinearTicks(6),yticks=LinearTicks(5),))
save("hhl_lehman_volerror.pdf",f)

m = readdlm("/home/niamh/mypapers/eqd_book/theta_vanilla_mc_1step.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfm.Theta .*= 10000
dfm.StdErr .*= 10000
dfm[!,:Up] = dfm[!,:Theta] .+ 1.96.* dfm[!,:StdErr]
dfm[!,:Down] = dfm[!,:Theta] .- 1.96.* dfm[!,:StdErr]
vol=0.2
T=1.0
t1=1.0/365
dt=range(0, stop=t1,length=101)
bb= @. exp(vol^2*(T-dt))*(exp(vol^2*(T-dt))-1)+exp(vol^2*T)*(exp(vol^2*T)-1)-2*exp(vol^2*(T+T-dt)/2)*( exp(vol^2*sqrt(T*(T-dt)))-1)
sequential= @. exp(vol^2*(T-dt))*(exp(vol^2*(T-dt))-1)+exp(vol^2*T)*(exp(vol^2*T)-1)-2*exp(0.5*vol^2*(4*T-4*t1+t1+t1-dt+2*sqrt(t1*(t1-dt))))+2*exp(0.5*vol^2*(T-dt+T))
plot(dt .* 365, bb,label="BB",ylab="variance",xlab="dt in days")
plot!(dt .* 365, sequential,legend="Incremental",yscale=:log10)
bb= @. exp(dt*vol^2)*exp(vol^2*(T-dt))*(exp(vol^2*(T-dt))-1)+exp(vol^2*T)*(exp(vol^2*T)-1)-2*exp(0.5*dt*vol^2)*exp(vol^2*(T+T-dt)/2)*( exp(vol^2*sqrt(T*(T-dt)))-1)
sequential= @. exp(dt*vol^2)*exp(vol^2*(T-dt))*(exp(vol^2*(T-dt))-1)+exp(vol^2*T)*(exp(vol^2*T)-1)-2*exp(0.5*dt*vol^2)*exp(0.5*vol^2*(4*T-4*t1+t1+t1-dt+2*sqrt(t1*(t1-dt))))+2*exp(0.5*dt*vol^2)*exp(0.5*vol^2*(T-dt+T))

#vol=1.0
mc095=0.004877/0.000220
mc050=0.003499/0.000220
vol=0.2
mc050=0.000432/0.000020
mc095=0.000603/0.000020
incremental=@. exp(vol^2*T)*(1+exp(-vol^2*dt)-2*exp(vol^2*(sqrt(t1-dt)*sqrt(t1)-t1)))
bb= @. exp(vol^2*T)*(1+exp(-vol^2*dt)-2*exp(vol^2*(sqrt(T-dt)*sqrt(T)-T)))
plot(dt .*365, sqrt.(incremental./bb),label="Theoretical",ylab="Ratio of standard errors Incremental/BB",xlab="dt in days")
plot!([0.5,0.95],[mc050,mc095],seriestype=:scatter, label="Monte-Carlo",legend=:outerright,size=(800,400))
savefig("/home/niamh/mypapers/eqd_book/theta_vanilla_variance.pdf")
m = readdlm("/home/niamh/mypapers/eqd_book/onetouch_theta_mrg_bb_smooth.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
n = readdlm("/home/niamh/mypapers/eqd_book/onetouch_theta_mrg_naive.txt",header=true)
dfn = DataFrame(n[1], vec(n[2]))
o = readdlm("/home/niamh/mypapers/eqd_book/onetouch_theta_mrg_lr.txt",header=true)
dfo = DataFrame(o[1], vec(o[2]))
dfo = dfo[dfo.Estimate .== "Theta",:]
ind = sortperm(dfo.N)
p = readdlm("/home/niamh/mypapers/eqd_book/onetouch_theta_mrg_inc_smooth.txt",header=true)
dfp = DataFrame(p[1], vec(p[2]))
plot(dfo.N[ind], dfo.Value[ind],label="Likelihood ratio",xscale=:log10,xlims=(1024,1000000),ylims=(-1500,500),size=(400,300))
plot!(dfn.N,dfn.Value,label="Naive", xscale=:log10,xlims=(1024,1000000),ylims=(-1500,500),size=(400,300))
savefig("/home/niamh/mypapers/eqd_book/theta_lr_onetouch.pdf")

plot(dfm.N,dfm.Value,label="Brownian-bridge")
plot!(dfp.N, dfp.Value,label="Incremental",xscale=:log10,xlims=(1024,1000000),ylims=(-1500,500),size=(400,300))
savefig("/home/niamh/mypapers/eqd_book/theta_smooth_onetouch.pdf")

m = readdlm("/home/niamh/mypapers/eqd_book/qd_halley_100.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
plot(dfm.Iteration,dfm.S, shape=:auto, group=dfm.Method, ylab="S*",xlab="Iterations",xlim=(0,20),size=(800,400),legend=:bottomright,margins=3Plots.mm)
hline!([48.488698],linestyle=:dash,color=:black,label="")
savefig("/home/niamh/mypapers/eqd_book/qd_halley_100.pdf")


##Multiple plots for dividends not easy to show as nicely as ggplot


m = readdlm("/home/niamh/mypapers/cash_dividend_basket/single_r6_strike150_td.txt",header=true)
dfm1 = DataFrame(m[1], vec(m[2]))
dfm1[!,:Strike] .= 150.0
m = readdlm("/home/niamh/mypapers/cash_dividend_basket/single_r6_strike50_td.txt",header=true)
dfm2 = DataFrame(m[1], vec(m[2]))
dfm2[!,:Strike] .= 50.0
m = readdlm("/home/niamh/mypapers/cash_dividend_basket/single_r6_strike100_td.txt",header=true)
dfm3 = DataFrame(m[1], vec(m[2]))
dfm3[!,:Strike] .= 100.0
dfm = vcat(dfm1,dfm2,dfm3)
dfn = dfm[dfm.Model .== "Zhang-2" .|| dfm.Model .== "BB-LB" .|| dfm.Model .== "LL-3" .|| dfm.Model .== "GS" .|| dfm.Model .== "BB" .|| dfm.Model .== "BB-Ju",:]
dfn.StrikeName = string.("Strike = ",dfn.Strike)
using AlgebraOfGraphics, CairoMakie
VolErrorPct = :VolError => (t -> abs(t * 100)+1e-6) => "Absolute Error in Volatility (%)"
Td = :Td => (t -> t) => "Dividend ex-date in years"
plt = data(dfn) * mapping(Td, VolErrorPct)  * mapping(col=:StrikeName) * mapping(color=:Model) *  visual(Lines)
fig = draw(plt, legend=(position=:bottom,titleposition=:left), axis=(yscale=log10,width=250,height=250,))
save("/home/niamh/mypapers/eqd_book/single_r6_strike100_td.pdf", fig)

m = readdlm("/home/niamh/mypapers/cash_dividend_basket/gocsei_10y.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn = dfm[dfm.Model .== "Zhang-2" .|| dfm.Model .== "BB-LB" .|| dfm.Model .== "LL-3" .|| dfm.Model .== "GS" .|| dfm.Model .== "BB" .|| dfm.Model .== "BB-Ju",:]
plt = data(dfn) * mapping(:Strike, VolErrorPct)  * mapping(color=:Model) *  visual(Lines)
fig = draw(plt, axis=(yscale=log10,width=700,height=350,))
save("/home/niamh/mypapers/eqd_book/gocsei_10y.pdf", fig)

#plot(dfn.Strike, abs.(dfn.VolError .* 100), group=dfn.Model, ylab="Absolute Error in Volatility (%)", linestyle=:auto, yscale=:log10)
#plot!(size=(1000,400),legend=:outerright,margins=3Plots.mm,legendtitle="Model",xlab="Strike",yticks=[1e-1,1e-2,1e-3,1e-4,1e-5])
#savefig("~/mypapers/eqd_book/gocsei_10y.pdf")

m = readdlm("/home/niamh/mypapers/cash_dividend_basket/zhang_extreme.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfn = dfm[dfm.Method .== "Zhang-2" .|| dfm.Method .== "BB-LB" .|| dfm.Method .== "LL-3" .|| dfm.Method .== "GS" .|| dfm.Method .== "BB" .|| dfm.Method .== "BB-Ju",:]
PriceError = :Error => (t -> abs(t)) => "Absolute Error in Price"

plt = data(dfn) * mapping(:Strike, PriceError)  * mapping(color=:Method) *  visual(Lines)
fig = draw(plt, axis=(yscale=log10,width=700,height=350,))
save("/home/niamh/mypapers/eqd_book/zhang_extreme.pdf", fig)


m = readdlm("/home/niamh/mypapers/eqd_book/deelstra_pln_100.txt",header=true)
dfm = DataFrame(m[1], vec(m[2]))
dfm=dfm[dfm.N .!= 1000,:]
dfm.RelError = dfm.Error./dfm.Price
dfm.NDividend = string.(dfm.N," dividend(s)")
RelError = :RelError => (t -> abs(t)) => "Relative Error in Price"
plt = data(dfm) * mapping(:Strike, RelError)  * mapping(col=:NDividend) * mapping(color=:Approximation) *  visual(Lines)
fig = draw(plt, legend=(position=:bottom,titleposition=:left), axis=(yscale=log10,width=250,height=250,limits=(nothing,(1e-8,1e-5)),))
save("/home/niamh/mypapers/eqd_book/deelstra_pln_100.pdf", fig)
