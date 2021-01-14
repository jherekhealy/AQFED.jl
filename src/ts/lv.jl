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
