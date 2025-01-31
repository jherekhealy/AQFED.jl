function bachelierVolatilityFunctionC3(v; forward=1.0, moneyness=exp(-1.0), targetPrice=0.12693673750664397)
    fmk = forward - forward / moneyness
    h = fmk / v
    cEstimate = fmk * normcdf(h) + v * normpdf(h)
    vega = normpdf(h)
    volga = vega * h^2 / v
    c3 = vega * (-3 * h^2 / v^2 + h^4 / v^2)
    return cEstimate - targetPrice, vega, volga, c3
end
