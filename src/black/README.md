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
# Single iv lookup
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

# Test from Zhenyu Cui,Yanchu Liu and Yuhang Yao 
Reproduction of  the Table 2 from "Tighter bounds for implied volatility based on the Dirac delta family method"

```julia
import AQFED.Black
using BenchmarkTools, StatsBase

S0 = 100;r = 0.03;q = 0.0;
num_K = 39; num_tau = 39; num_sigma = 39;
K0 = 105:(800-105)/num_K:800;
tau0 = 0.01:(2-0.01)/num_tau:2;
sigma0 = 0.01:(0.99-0.01)/num_sigma:0.99;
K = [K for K in K0 for tau in tau0 for sigma in sigma0]
tau = [tau for K in K0 for tau in tau0 for sigma in sigma0]
sigma = [sigma for K in K0 for tau in tau0 for sigma in sigma0]
C_real = [Black.blackScholesFormula(true, K, S0, sigma^2 * tau, exp(-(r-q)*tau), exp(-r*tau)) for K in K0 for tau in tau0 for sigma in sigma0]
indices = findall( x-> x > 1e-20, C_real)
C_real = C_real[indices];  K = K[indices];  tau = tau[indices]; sigma = sigma[indices];
IV_jaeckel = [Black.impliedVolatilityJaeckel(true, C_reali, S0*exp((r-q)*taui),Ki,taui,exp(-r*taui)) for (C_reali, Ki, taui, sigmai) in zip(C_real,K,tau,sigma)]
t6_a = abs.(IV_jaeckel - sigma)
stats = [mean(t6_a),std(t6_a),maximum(t6_a),minimum(t6_a)]
IV_srhou = [Black.impliedVolatilitySRHalley(true, C_reali, S0*exp((r-q)*taui),Ki,taui,exp(-r*taui),eps(),32,Black.Householder()) for (C_reali, Ki, taui, sigmai) in zip(C_real,K,tau,sigma)]
  t7_a = abs.(IV_srhou - sigma)
stats = [mean(t7_a),std(t7_a),maximum(t7_a),minimum(t7_a)]
```
Output for Jackel is 
```4-element Vector{Float64}:
 3.560309266995074e-16
 4.644469145487699e-16
 1.4876988529977098e-14
 0.0
 ```

 Output for SR-Householder is
 ```
 -element Vector{Float64}:
 3.680984110213631e-16
 4.74453849417662e-16
 1.4765966227514582e-14
 0.0
 ```
