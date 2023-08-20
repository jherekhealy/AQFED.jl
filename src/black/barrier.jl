using QuadGK
using AQFED.Math
export BarrierPayoff, BarrierKind, AnalyticBarrierEngine, calculateOneTouch, calculate, calculateUpInDownOutEnd
@enum BarrierKind DownIn UpIn DownOut UpOut
struct BarrierPayoff{T}
    isCall::Bool
    strike::T
    level::T
    rebate::T
    kind::BarrierKind
end

struct AnalyticBarrierEngine{T}
	spot              ::T
	variance          ::T
	driftDf           ::T
	discountDf        ::T
	discountDfPayment ::T
	stdDev            ::T
	mu                ::T
	muSigma           ::T
end

function AnalyticBarrierEngine(spot::T, variance::T, driftDf::T, discountDf::T, discountDfPayment::T) where {T}
	stdDev = sqrt(variance)
	mu = -log(driftDf)/variance - 0.5
	muSigma = (1 + mu) * stdDev
	return AnalyticBarrierEngine(spot, variance, driftDf, discountDf, discountDfPayment, stdDev, mu, muSigma)
end

function calculateOneTouch(an::AnalyticBarrierEngine{T}, isDown::Bool, level, payout) where {T}
	payoff = BarrierPayoff(isDown, an.spot, level, payout, DownOut)
	eta = if isDown 1 else -1 end
	value = calculateF(an, payoff, eta)
	value *= an.discountDfPayment / an.discountDf
	return value
end

function calculateUpInDownOutEnd(an::AnalyticBarrierEngine, isCall::Bool, strike::T, down::T, up::T) where {T}
	r = -log(an.discountDf) / an.variance
	mu = -log(an.driftDf) / an.variance
	obj = function(w::TW) where {TW} 
		df = exp(-r * w)
		mudf = exp(-mu * w)
		anw = AnalyticBarrierEngine(up, an.variance-w, an.driftDf/mudf, an.discountDf/df, an.discountDfPayment/df)
		downOutPrice = calculate(anw, BarrierPayoff(isCall, strike, down, zero(typeof(strike)), DownOut))
		if abs(w-an.variance) < sqrt(eps(TW))
			downOutPrice = max(up-strike, 0) * an.discountDfPayment / df
        end
		if abs(w) < sqrt(eps(TW))
			return zero(TW)
        end
		ln = log(up / an.spot)
		d = ln - (mu-0.5)*w
		q = abs(ln) / sqrt(2* π *w) * exp(-d^2 /(2*w)) / w
		return df * downOutPrice * q
	end
	kikoPrice,_ = quadgk(obj, zero(T), an.variance)
	return kikoPrice
end

#Calculate computes the barrier option price
function  calculate(an::AnalyticBarrierEngine, payoff::BarrierPayoff{T}) where {T}
	local value = zero(T)
	if payoff.isCall 
		if payoff.kind  == DownIn
			if payoff.strike >= payoff.level 
				value = calculateC(an, payoff, 1, 1) + calculateE(an, payoff, 1)
			 else 
				value = calculateA(an, payoff, 1) - calculateB(an,payoff, 1) + calculateD(an,payoff, 1, 1) + calculateE(an,payoff, 1)
             end
            elseif payoff.kind == UpIn
			if payoff.strike >= payoff.level 
				value = calculateA(an,payoff, 1) + calculateE(an, payoff, -1)
			 else 
				value = calculateB(an,payoff, 1) - calculateC(an, payoff, -1, 1) + calculateD(an,payoff, -1, 1) + calculateE(an,payoff, -1)
             end
            elseif payoff.kind == DownOut
			if payoff.strike >= payoff.level 
				value = calculateA(an,payoff, 1) - calculateC(an,payoff, 1, 1) + calculateF(an,payoff, 1)
			 else 
				value = calculateB(an,payoff, 1) - calculateD(an,payoff, 1, 1) + calculateF(an,payoff, 1)
             end
            elseif payoff.kind == UpOut
			if payoff.strike >= payoff.level 
				value = calculateF(an,payoff, -1)
			 else 
				value = calculateA(an,payoff, 1) - calculateB(an,payoff, 1) + calculateC(an,payoff, -1, 1) - calculateD(an,payoff, -1, 1) + calculateF(an,payoff, -1)
             end
            end
	 else 
     if payoff.kind ==  DownIn
			if payoff.strike >= payoff.level 
				value = calculateB(an,payoff, -1) - calculateC(an,payoff, 1, -1) + calculateD(an,payoff, 1, -1) + calculateE(an,payoff, 1)
			 else 
				value = calculateA(an,payoff, -1) + calculateE(an,payoff, 1)
            end
       elseif payoff.kind == UpIn
			if payoff.strike >= payoff.level 
				value = calculateA(an,payoff, -1) - calculateB(an,payoff, -1) + calculateD(an,payoff, -1, -1) + calculateE(an,payoff, -1)
			 else 
				value = calculateC(an,payoff, -1, -1) + calculateE(an,payoff, -1)
             end

            elseif payoff.kind ==  DownOut
			if payoff.strike >= payoff.level 
				value = calculateA(an,payoff, -1) - calculateB(an,payoff, -1) + calculateC(an,payoff, 1, -1) - calculateD(an,payoff, 1, -1) + calculateF(an,payoff, 1)
			 else 
				value = calculateF(an,payoff, 1)
             end
            elseif payoff.kind ==  UpOut
			if payoff.strike >= payoff.level 
				value = calculateB(an,payoff, -1) - calculateD(an,payoff, -1, -1) + calculateF(an,payoff, -1)
			 else 
				value = calculateA(an,payoff, -1) - calculateC(an,payoff, -1, -1) + calculateF(an,payoff, -1)
             end

            end
        end
	value *= an.discountDfPayment / an.discountDf #payment lag
	return value
    end

function calculateA(an::AnalyticBarrierEngine,payoff::BarrierPayoff, phi::Int) 
	x1 = log(an.spot/payoff.strike)/an.stdDev + an.muSigma
	N1 = normcdf(phi * x1)
	N2 = normcdf(phi * (x1 - an.stdDev))
	return phi * (an.spot/an.driftDf*N1 - payoff.strike*N2) * an.discountDf
end

function calculateB(an::AnalyticBarrierEngine,payoff::BarrierPayoff, phi::Int)
	x2 = log(an.spot/payoff.level)/an.stdDev + an.muSigma
	N1 = normcdf(phi * x2)
	N2 = normcdf(phi * (x2 - an.stdDev))
	return phi * (an.spot/an.driftDf*N1 - payoff.strike*N2) * an.discountDf
end

function calculateC(an::AnalyticBarrierEngine,payoff::BarrierPayoff, eta::Int, phi::Int)
	HS = payoff.level / an.spot
	powHS0 = HS^(2*an.mu)
	powHS1 = powHS0 * HS^2
	y1 = log(payoff.level*HS/payoff.strike)/an.stdDev + an.muSigma
	N1 = normcdf(eta * y1)
	N2 = normcdf(eta * (y1 - an.stdDev))
	return phi * (an.spot/an.driftDf*powHS1*N1 - payoff.strike*powHS0*N2) * an.discountDf
end

function calculateD(an::AnalyticBarrierEngine,payoff::BarrierPayoff, eta::Int, phi::Int)
	HS = payoff.level / an.spot
	powHS0 = HS^(2*an.mu)
	powHS1 = powHS0 * HS^2
	y2 = log(payoff.level/an.spot)/an.stdDev + an.muSigma
	N1 = normcdf(eta * y2)
	N2 = normcdf(eta * (y2 - an.stdDev))
	return phi * (an.spot/an.driftDf*powHS1*N1 - payoff.strike*powHS0*N2) * an.discountDf
end

function calculateE(an::AnalyticBarrierEngine,payoff::BarrierPayoff, eta::Int)
	if payoff.rebate > 0 
		HS = payoff.level / an.spot
		powHS0 = HS^(2*an.mu)
		x2 = log(an.spot/payoff.level)/an.stdDev + an.muSigma
		y2 = log(payoff.level/an.spot)/an.stdDev + an.muSigma
		N1 = normcdf(eta * (x2 - an.stdDev))
		N2 = normcdf(eta * (y2 - an.stdDev))
		return payoff.rebate * (N1 - powHS0*N2) * an.discountDf
	 else 
		return zero(typeof(an.spot))
     end
    end

function calculateF(an::AnalyticBarrierEngine,payoff::BarrierPayoff, eta::Int)
	if payoff.rebate != zero(typeof(payoff.rebate)) 
		r = -log(an.discountDf) / an.variance
		HS = payoff.level / an.spot
		disc = an.mu*an.mu + 2.0*r
		if disc < 0             
			lambdai = sqrt(-disc)
			powHS = HS^an.mu
			logHS = log(HS)
			powHSplus = complex(powHS, 0) * exp(complex(0, lambdai*logHS))
			powHSminus = complex(powHS, 0) * exp(complex(0, -lambdai*logHS))
			z = complex(log(payoff.level/an.spot)/an.stdDev, lambdai*an.stdDev)
			etac = complex(eta, 0)
			N1 = normcdf(etac*z)
			N2 = normcdf(etac*(z-complex(0, 2*lambdai*an.stdDev)))
			return payoff.rebate * real(powHSplus*N1+powHSminus*N2)
        end
		lambda = sqrt(disc)
		powHSplus = HS^(an.mu+lambda)
		powHSminus = HS^(an.mu-lambda)
		z = log(payoff.level/an.spot)/an.stdDev + lambda*an.stdDev
		N1 = normcdf(eta * z)
		N2 = normcdf(eta * (z - 2.0*lambda*an.stdDev))
		return payoff.rebate * (powHSplus*N1 + powHSminus*N2)
    end
	return zero(typeof(an.spot))
end


# type AnalyticRainbowBarrierEngine struct {
# 	spot1             float64
# 	variance1         float64
# 	driftDf1          float64
# 	spot2             float64
# 	variance2         float64
# 	driftDf2          float64
# 	rho               float64
# 	discountDfPayment float64
# }

# //NewAnalyticBarrierEngine instantiates a new Black-Scholes AnalyticBarrierEngine
# func NewAnalyticRainbowBarrierEngine(spot1 float64, variance1 float64, driftDf1 float64, spot2 float64, variance2 float64, driftDf2 float64, rho float64, discountDfPayment float64) *AnalyticRainbowBarrierEngine {
# 	return &AnalyticRainbowBarrierEngine{spot1, variance1, driftDf1, spot2, variance2, driftDf2, rho, discountDfPayment}
# }

# func (an *AnalyticRainbowBarrierEngine) Calculate(payoff BarrierPayoff) (float64, error) {
# 	omega1 := 1.0
# 	omega2 := 1.0
# 	if payoff.IsCall {
# 		switch payoff.Type {
# 		case DownOut, DownIn:
# 			omega1 = 1.0
# 			omega2 = -1.0
# 		case UpOut, UpIn:
# 			omega1 = 1.0
# 			omega2 = 1.0
# 		}
# 	} else {
# 		switch payoff.Type {
# 		case DownOut, DownIn:
# 			omega1 = -1.0
# 			omega2 = -1.0
# 		case UpOut, UpIn:
# 			omega1 = -1.0
# 			omega2 = 1.0
# 		}
# 	}
# 	sqrtDev1 := math.Sqrt(an.variance1)
# 	d1 := (math.Log(an.spot1/an.driftDf1/payoff.strike) + 0.5*an.variance1) / sqrtDev1
# 	d2 := d1 - sqrtDev1
# 	bs2 := math.Log(payoff.level / an.spot2)
# 	sqrtDev2 := math.Sqrt(an.variance2)
# 	d3 := d1 + 2*an.rho*bs2/sqrtDev2
# 	d4 := d2 + 2*an.rho*bs2/sqrtDev2
# 	e1 := (bs2 + math.Log(an.driftDf2) + 0.5*an.variance2 - an.rho*sqrtDev1*sqrtDev2) / sqrtDev2
# 	e2 := e1 + an.rho*sqrtDev1
# 	e3 := e1 - 2*bs2/sqrtDev2
# 	e4 := e2 - 2*bs2/sqrtDev2
# 	ex1 := math.Exp(2 * (an.rho*sqrtDev1*sqrtDev2 - 0.5*an.variance2 - math.Log(an.driftDf2)) * bs2 / an.variance2)
# 	ex2 := math.Exp(2 * (-0.5*an.variance2 - math.Log(an.driftDf2)) * bs2 / an.variance2)
# 	value := omega1*an.spot1/an.driftDf1*(normal.BivariateCdf(omega1*d1, omega2*e1, -omega1*omega2*an.rho)-ex1*normal.BivariateCdf(omega1*d3, omega2*e3, -an.rho*omega1*omega2)) - payoff.strike*omega1*(normal.BivariateCdf(omega1*d2, omega2*e2, -omega1*omega2*an.rho)-ex2*normal.BivariateCdf(omega1*d4, omega2*e4, -omega1*omega2*an.rho))
# 	value *= an.discountDfPayment

# 	switch payoff.Type {
# 	case DownIn, UpIn:
# 		vanilla := black.BlackScholesFormula(payoff.IsCall, payoff.strike, an.spot1, an.variance1, an.driftDf1, an.discountDfPayment)
# 		value = vanilla - value
# 	}

# 	return value, nil
# }

# func ComputeDiscreteBarrierLevel(level float64, spot float64, vol float64, tte float64, nObs int) float64 {
# 	u := math.Log(spot/level) / (vol * math.Sqrt(tte/float64(nObs)))
# 	y := 0.5826 + 0.1245*math.Exp(-2.7*math.Pow(math.Abs(u), 1.2))
# 	discreteLevel := level * math.Exp(y*vol*math.Sqrt(tte/float64(nObs)))
# 	//fmt.Println(u, y, discreteLevel)
# 	return discreteLevel
# }

