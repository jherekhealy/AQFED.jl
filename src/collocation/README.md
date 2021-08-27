## Stochastic Collocation
In the Collocation module, you will find an implementation of the polynomial stochastic collocation to fit a given set of implied volatilities, or of call option prices.

The prices need not to be arbitrage free:
- A utility function `isArbitrageFree` verifies if given undiscounted call option prices are arbitrage free.
- The function `filterConvexPrices` returns the closest set of arbitrage free prices.

There is a small error in the original paper and in the third edition of the book, the polynomials used in the sum of square may be of equal degree, while the paper specifies one of the polynomials with a lower degree.

## Example
### TSLA input
```julia
using AQFED; using AQFED.Collocation
#TSLA example
strikes = Float64.([20, 25, 50, 55, 75, 100, 120, 125, 140, 150, 160, 175, 180, 195, 200, 210, 230, 240, 250, 255, 260, 270, 275, 280, 285, 290, 300, 310, 315, 320, 325, 330, 335, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 550, 580, 590, 600, 650, 670, 680, 690, 700]);
vols =[1.21744983334323, 1.1529735541872308, 1.0013512993166844, 1.0087013871410198, 0.9055919576135943, 0.8196499269009432, 0.779704840770866, 0.753927847741657, 0.7255349986293694, 0.7036962946028743, 0.6870907597202961, 0.6631489459500445, 0.6542809839143336, 0.6310048431894977, 0.6231979513191437, 0.6154526014300009, 0.5866214834144697, 0.5783751483731193, 0.5625036590124762, 0.5625539176150428, 0.5572331684618123, 0.5485212417739607, 0.5456131657256524, 0.540060895711996,0.5384776792271245, 0.5325298112839504, 0.5222410647552144, 0.5202396738775005, 0.5168414254536685, 0.5127405490209541, 0.5100440087558921, 0.50711984442627, 0.5042896073005682, 0.5013959697030379, 0.4961897259221575, 0.4914478237113829, 0.48571052433313705, 0.4820982302575811, 0.4776551485043659, 0.4682253137830999, 0.46912624306506934, 0.4652049749994563, 0.4621036693145566, 0.45969798571592985, 0.4561356005182957, 0.45418189139835186, 0.4515451651258398,0.44541885580442636, 0.4452833907060621, 0.44303755690672525, 0.43939212779385645, 0.4413175310749832, 0.4336322023390991, 0.4297053821023934, 0.4284357423754355, 0.4241077476619805, 0.4222672729031064, 0.4203436892852212, 0.4193419518701644, 0.41934732346075626, 0.41758929420417745];
forward = 356.73063159822254
tte = 1.5917808219178082
w1 = ones(length(strikes))
prices, weights = Collocation.weightedPrices(true, strikes, vols, w1, forward, 1.0, tte)
```
### Plot X coordinate vs. strikes

(mostly useful to understand the technique, not to use the technique)
```julia
using Plots
strikesf, pricesf = Collocation.filterConvexPrices(strikes, prices, weights, forward,tol=1e-6)
strikesf, pif, xif = Collocation.makeXFromUndiscountedPrices(strikesf, pricesf)
p1 = plot(xif, strikesf, seriestype= :scatter, label="reference")
```
### Plot the actual degree-5 and degree-11 collocations on top.

The `degGuess` parameter allows to choose the kind of initial guess we want to use.

- `degGuess=1` corresponds to a Bachelier model initial guess and works well in general while being very simple.
- `degGuess=3` is for a cubic, which also works well, and is closer to the solution in general. Input prices are filtered for convexity.
- Larger degrees use a tweak (an additional point at strike=0 and x=-3) to avoid the cases where the polynomial fits very well the X array, but is ill-suited to price as it crosses 0 too "early" on, at a too large strike; as such they are not recommended to use.

High degree collocation also tends to require much more steps in the minimizer.
```julia
isoc,m = Collocation.makeIsotonicCollocation(strikesf, pricesf, weights, tte, forward, 1.0,deg=5)
sol5 = Collocation.Polynomial(isoc)
println("Solution ", sol5, " ",Collocation.stats(sol5), " measure ",m)
isoc,m = Collocation.makeIsotonicCollocation(strikes, pricesf, weights, tte, forward, 1.0,deg=11,degGuess=1)
sol = Collocation.Polynomial(isoc)
println("Solution ", sol, " ",Collocation.stats(sol), " measure ",m)
x = collect(-3.0:0.01:3.0);
plot!(x, sol5.(x), label="degree-5")
plot!(x, sol.(x), label="degree-11")
```
![Implied volatilities](/resources/images/collocation_x_y.png)

### Plot of the density
A high degree has a tendency to create a spike. Prefer a degree <= 7.

This may be seen as a disadvantage: it looks awkward, since it is located in the extrapolation part. It is artificial since it is due to the interpolation part, but impact the extrapolation.
It could also be interpreted more positively, as it allows to fit well implied volatilities with a steep curvature. In the latter case, the
spike will be rightly located in the interpolation part.

```julia
k = collect(10:1.0:2000);
p2 = plot(k, Collocation.density.(sol,k), label="degree-11")
plot!(k, Collocation.density.(sol5,k), label="degree-5")
```
It is possible to mitigate the spike via an appropriate choice of the `minSlope` parameter (e.g. `minSlope=0.1` on this example), at the cost of a worse fit. A more appropriate solution is to rely on regularization (as in the B-spline collocation case), a term in `penalty*hermiteIntegral(derivative(p,2)^2)` seems appropriate on this example, with `penalty=1e-7`.

```julia
isoc,m = Collocation.makeIsotonicCollocation(strikes, pricesf, weights, tte, forward, 1.0,deg=11,degGuess=1,penalty=1e-7)
p3 = plot(k, Collocation.density.(Collocation.Polynomial(isoc),k), label="degree-11-P")
plot!(k, Collocation.density.(sol,k), label="degree-11")
plot!(k, Collocation.density.(sol5,k), label="degree-5")
```
![Probability Density](/resources/images/collocation_density.png)


### Plot of the implied vols
```julia
ivk = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, k,forward,1.0), forward, k, tte, 1.0);
ivk5 = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol5,true, k,forward,1.0), forward, k, tte, 1.0);
p3 = plot(strikes, vols, seriestype= :scatter)
plot!(k, ivk)
plot!(k, ivk5)
  ```
![Implied volatilities](/resources/images/collocation_vols.png)

We use by default the inverse quadratic method to find x for at a given strike. The `useHalley=false` flag may be used to rely on the Polynomials.roots function instead, which is more robust, but slower.

### Shorter maturity
```julia
strikes = Float64.([150, 155, 160, 165, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 500, 520, 540, 560, 580])
vols = [1.0354207293271083, 0.9594199540767598, 0.9691592997116513, 0.921628193615727, 0.922211330770905, 0.8847766875013793, 0.8783544778003536, 0.8513146324654995, 0.8827044619549536, 0.8306675656174141, 0.7908245509298024, 0.7558509594446964, 0.7597364594588467, 0.7403822080716124, 0.7298680110662008, 0.7091863887722245, 0.6818559320159979, 0.661436341750887, 0.6462309799739886, 0.629398669713696, 0.6145630586631312, 0.5938324665922039, 0.5805662721081856, 0.5694652563270426, 0.5536895819277889, 0.5422671292669782, 0.533888799038778, 0.5234154661207774, 0.5168510552270291, 0.5072806473672078, 0.4997973159961659, 0.4896563997378466, 0.48239758503680114, 0.47936812363581066, 0.48000589145706996, 0.47575254134423506, 0.47114784824672207, 0.46788352167691066, 0.4656217516966071, 0.462996525592066, 0.4593993028842441, 0.4585651056438656, 0.45790487479638, 0.4552139844132186, 0.45344730213977413, 0.45040138270126456, 0.44800472164335725, 0.44919955536439704, 0.44788407072486525, 0.4498560128677439, 0.45218341600249046, 0.44936231891738604, 0.44881496768582435, 0.4516047756853702, 0.45554688648955244, 0.4608658483565173, 0.4599545390661706, 0.4634099812656424, 0.4716998585738154, 0.4758803092917464, 0.48100099895733817, 0.48559069655772896, 0.49064468784617526, 0.49606127734737687, 0.5011170526132833, 0.5059204240563129, 0.5149247954706585, 0.5517620518081904, 0.5776531692627265, 0.5992609035805616, 0.6259792014943727]
forward = 357.75592553175875
tte = 0.0958904109589041
w1 = ones(length(strikes))
prices, weights = Collocation.weightedPrices(true, strikes, vols, w1, forward, 1.0, tte)
isoc,m = Collocation.makeIsotonicCollocation(strikes, prices, weights, tte, forward, 1.0,deg=7,degGuess=3)
sol = Collocation.Polynomial(isoc)
println("Solution ", sol, " ",coeffs(sol), " ",Collocation.stats(sol), " measure ",m)
ivk = @. Black.impliedVolatility(true, Collocation.priceEuropean(sol, true, k,forward,1.0), forward, k, tte, 1.0);
p3 = plot(strikes, vols, seriestype= :scatter)
plot!(k, ivk, label="degree-7")
```
![Implied volatilities for a short maturity](/resources/images/collocation_vols_short.png)
## References
Le Floc'h, F. and Oosterlee, C. W. (2019) [Model-free stochastic collocation for an arbitrage-free implied volatility: Part I](https://link.springer.com/article/10.1007/s10203-019-00238-x)
