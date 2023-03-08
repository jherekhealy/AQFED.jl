export SVISection,XSSVISection

struct XSSVISection <: VarianceSection
    ystar::Float64
    θstar::Float64
    ρ::Float64
    ψ::Float64
    tte::Float64
    f::Float64
end 

Base.broadcastable(p::XSSVISection) = Ref(p)
function varianceByLogmoneyness(s::XSSVISection, y)
    θ = theta(s)
    w = θ/2*(1+s.ρ*s.ψ/θ*y+sqrt((s.ψ/θ*y+s.ρ)^2+(1-s.ρ^2)))
    return w / s.tte
end

theta(s::XSSVISection) = s.θstar - s.ρ*s.ψ*s.ystar


struct SVISection <: VarianceSection
    a::Float64
    b::Float64
    rho::Float64
    s::Float64
    m::Float64
    tte::Float64
    f::Float64
end
Base.broadcastable(p::SVISection) = Ref(p)
function varianceByLogmoneyness(s::SVISection, y)
    sqrsy = s.s^2 + (y - s.m)^2
    return s.a + s.b * (s.rho * (y - s.m) + sqrt(sqrsy))
end

function varianceSlopeCurvature(s::SVISection, y::Float64)
    ym = (y - s.m)
    sqrsy = s.s * s.s + ym * ym
    sqrtsy = sqrt(sqrsy)
    variance = s.a + s.b * (s.rho * ym + sqrtsy)
    slope = s.b * (ym / sqrtsy + s.rho)
    curvature = s.b * (1 - ym^2 / sqrsy) / sqrtsy
    return variance, slope, curvature
end

function varianceByLogmoneyness(s::SVISection, y::Array{Float64})
    sqrsy = @. s.s^2 + (y - s.m)^2
    return @. s.a + s.b * (s.rho * (y - s.m) + sqrt(sqrsy))
end

function varianceSlopeCurvature(s::SVISection, y::Array{Float64})
    ym = @. (y - s.m)
    sqrsy = @. (s.s^2 + ym^2)
    sqrtsy = sqrt.(sqrsy)
    variance = @. s.a + s.b * (s.rho * ym + sqrtsy)
    slope = @. s.b * (ym / sqrtsy + s.rho)
    curvature = @. s.b * (1 - ym^2 / sqrsy) / sqrtsy
    return variance, slope, curvature
end
