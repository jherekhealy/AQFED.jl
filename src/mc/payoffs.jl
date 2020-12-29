
struct VanillaOption
    isCall::Bool
    strike::Float64
    maturity::Float64
end

#advancePath(gen, pathValues, t0, t1)
#advancePayoff(time, pathValues)
function evaluatePayoff(payoff::VanillaOption, x, df)
    if payoff.isCall
        return df * max(x - payoff.strike, 0)
    else
        return df * max(payoff.strike - x, 0)
    end
end
