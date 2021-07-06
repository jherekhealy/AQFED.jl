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
