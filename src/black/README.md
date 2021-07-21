## Black-Scholes Formula

Julia allows to compute greeks via forward or reverse automatic differentiation.
In order to do this, we create a vector with the variables towards which we want to obtain sensitivities.
```julia
import AQFED.Black
using ForwardDiff
strike = 100.0; spot = 100.0; tte = 1.0; vol=0.4; r=0.05
x = [strike, spot, tte, vol, r, r]
f = function(x::AbstractArray)  Black.blackScholesFormula(true,x[1],x[2],x[3]*x[4]^2,exp(-x[5]*x[3]),exp(-x[6]*x[3])) end
ForwardDiff.gradient(f, x)

using ReverseDiff
ReverseDiff.gradient(f, x)
```
The reverse differentiation can be made faster by pre-compiling the tape as follows
```julia
f_tape = ReverseDiff.GradientTape(f,  rand(length(x))) #0 not ok
compiled_f_tape = ReverseDiff.compile(f_tape)
ReverseDiff.gradient!(y, compiled_f_tape, x)
```

## Implied Volatility Solver
### Standard Example
```julia
import AQFED.Black

f = 100.0; T=1.0; dfPayment=1.0;
vol = 0.01; strike = 150.0;
price = Black.blackScholesFormula(true, strike, f, vol * vol * T, 1.0, dfPayment)
Black.impliedVolatility(true, price, f, strike, T, dfPayment)
```

The output should read
```
julia> price = Black.blackScholesFormula(true, strike, f, vol * vol * T, 1.0, dfPayment)
9.01002030921698e-25

julia> Black.impliedVolatility(true, price, f, strike, T, dfPayment)
0.03999999999999892
```

### Multiple precision
```julia
import AQFED.Black

f = BigFloat("100.0"); T=BigFloat("1.0"); dfPayment=BigFloat("1.0");
vol = BigFloat("0.01"); strike = BigFloat("150.0");
price = Black.blackScholesFormula(true, strike, f, vol * vol * T, BigFloat(1.0), dfPayment)
Black.impliedVolatility(true, price, f, strike, T, dfPayment)
```
and the output is: 0.009999999999999999999999999999999999999999999999999999999999999999999999999787059, for a price of 3.005699303008737538090695390409788090781673843536520016899477489819371529654813e-361

### Benchmarking
```julia
import AQFED.Black
using BenchmarkTools

f = 100.0; T=1.0; dfPayment=1.0;
vol = 0.04; strike = 150.0;
price = Black.blackScholesFormula(true, strike, f, vol * vol * T, 1.0, dfPayment)

@benchmark Black.impliedVolatilitySRGuess(true, price, f, strike, T, dfPayment)
@benchmark Black.impliedVolatilitySRHalley(true, price, f, strike, T, dfPayment, 0.0, 64, Black.Householder())
@benchmark Black.impliedVolatilityLiSOR(true, price, f, strike, T, dfPayment, 0.0, 0.0, 64, Black.SORTS())
@benchmark Black.impliedVolatilityJaeckel(true, price, f, strike, T, dfPayment)
```
