using AQFED.Math

function blackVolatilityFunctionC3(v; forward=1.0, moneyness=exp(-1.0), targetPrice=0.12693673750664397, isLog=false)
    h = log(moneyness) ./ v
    t = v / 2
    cEstimate = normcdf(h + t) - 1 / moneyness * normcdf(h - t)
    vega = exp(-(h + t)^2 / 2) / sqrt(2 * pi)
    volga = vega * (h + t) * (h - t) / v
    c3 = vega * (-3 * h^2 - t^2 + (h^2 - t^2)^2) / (v^2)
    if isLog
        log(cEstimate) - log(targetPrice), vega / cEstimate, volga / cEstimate - (vega / cEstimate)^2, c3 / cEstimate - volga * vega / cEstimate^2 + 2 * vega / cEstimate * (volga / cEstimate - (vega / cEstimate)^2)
    else
        return cEstimate - targetPrice, vega, volga, c3
    end
end

function makeBlackFractal(w::Int, h::Int, xMin, xMax, yMin, yMax; maxIter=32, isLog::Bool=false, solver=Householder(), accuracy=1e-8, coloring=PaletteColoring())
    return makeFractal(w, h, xMin, xMax, yMin, yMax, maxIter=maxIter, f=if isLog
        x -> blackVolatilityFunctionC3(x, isLog=true)
    else
        blackVolatilityFunctionC3
    end, solver=solver, accuracy=accuracy, coloring=coloring)
end