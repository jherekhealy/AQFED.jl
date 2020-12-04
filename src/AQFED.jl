module AQFED

module Math
include("math/erfc.jl")
include("math/normal.jl")
include("math/as241.jl")
include("math/delbourgo_gregory.jl")
end

module Black
include("black/black.jl")
include("black/jaeckel.jl")
include("black/sr.jl")
include("black/lisor.jl")

const impliedVolatility = impliedVolatilitySRHouseholder
export impliedVolatility
end

end
