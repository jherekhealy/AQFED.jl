export discountFactor

struct ConstantBlackModel
    vol::Float64
    r::Float64
    q::Float64
end

struct TSBlackModel{S}
    surface::S
    r::Float64
    q::Float64
end

struct LocalVolatilityModel{S}
    surface::S
    r::Float64
    q::Float64
end

function discountFactor(model, t) 
    exp(-model.r * t)
end

function logForward(model, lnspot, t)
    return lnspot + (model.r-model.q)*t
end

function forward(model, spot, t) 
    return spot*exp((model.r-model.q)*t)
end
    
function varianceByLogmoneyness(model::ConstantBlackModel, y, t)
    return model.vol^2
end

    
function varianceByLogmoneyness(model, y, t)
    return varianceByLogmoneyness(model.surface, y, t)
end