struct ConstantBlackModel
    spot::Float64
    vol::Float64
    r::Float64
    q::Float64
end

struct TSBlackModel{S}
    spot::Float64
    surface::S
    r::Float64
    q::Float64
end

struct LocalVolatilityModel{S}
    spot::Float64
    surface::S
    r::Float64
    q::Float64
end
