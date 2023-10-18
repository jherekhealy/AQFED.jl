abstract type Quadrature end

using FFTW #only for fft cehbyshev
using FastGaussQuadrature
using QuadGK
import DoubleExponentialFormulas: quadde, quaddeo
export Quadrature, integrate, GaussLegendre, DoubleExponential, TanhSinh, GaussKronrod, Chebyshev, Simpson

struct FourierBoyd{T} <: Quadrature
    x::Vector{T}
    w::Vector{T}
end
function FourierBoyd(n::Int, l::T) where {T}
    t = pi * (1:n) / (n + 1)
    x = @. l * cot(t / 2)^2
    w = zeros(T, n)
    for (i, ti) = enumerate(t)
        sumi = zero(T)
        for j = 1:n
            sumi += sin(j * ti) * (1 - cos(j * pi)) / j
        end
        w[i] = 2l * sin(ti) / (1 - cos(ti))^2 * sumi * 2 / (n + 1)
    end
    return FourierBoyd(x, w)
end

function integrate(q::FourierBoyd{T}, f, a::T, b::T) where {T}
    # println("a=",a)
    #integrate from -infty to b. a is ignored.    
    #first map to 0 infty : z = -x +b or x = -z+b
    l = -2a / cot(pi / 8)^2
    # println("L=",l)
    i2 = dot(q.w, f.((b .- l .* q.x))) * l
    return i2
end

struct Chebyshev{T,kind} <: Quadrature
    x::Vector{T}
    w::Vector{T}
    isSplit::Bool
end
function Chebyshev{T,1}(N::Int, isSplit=false) where {T}
    x = chebnodes(T, N)
    n = length(x) - 1
    m = vcat(sqrt(2), 0.0, [(1 + (-1)^k) / (1 - k^2) for k = 2:n])
    ws = sqrt(2 / (n + 1)) * idct(m)
    return Chebyshev{T,1}(x, ws, isSplit)
end
function Chebyshev{T,2}(N::Int, isSplit=false) where {T}
    x = cheb2nodes(T, N)
    n = length(x)
    c = vcat(2, [2 / (1 - i^2) for i = 2:2:(n-1)])          # Standard Chebyshev moments
    c = vcat(c, c[Int(floor(n / 2)):-1:2])           #Mirror for DCT via FFT 
    ws = real(ifft(c))                          # Interior weightt
    ws[1] /= 2
    ws = vcat(ws, ws[1]) # Boundary weights
    return Chebyshev{T,2}(x, ws, isSplit)
    # x =  cheb2nodes(T, N+2)
    # return Chebyshev{T,2}([T(pi) / (N + 1) * sin(i / (N + 1) * T(pi))^2 / sqrt(1-x[i+1]^2) for i = 1:N],x[2:end-1], isSplit)
end
isSplit(q::Chebyshev{T,K}) where {T,K} = q.isSplit
kind(::Chebyshev{T,K}) where {T,K} = K


function integrate(q::Chebyshev{T,1}, f, a::T, b::T) where {T}
    bma2 = (b - a) / 2
    bpa2 = (b + a) / 2
    i2 = bma2 * dot(q.w, f.(bma2 .* q.x .+ bpa2))
    return i2
end


function integrate(q::Chebyshev{T,2}, f, a::T, b::T) where {T}
    bma2 = (b - a) / 2
    bpa2 = (b + a) / 2
    i2 = bma2 * dot(q.w, f.(bma2 .* q.x .+ bpa2))
    return i2
end

struct Simpson <: Quadrature
    N::Int
end

Base.broadcastable(p::Quadrature) = Ref(p)
function integrate(q::Simpson, f, a::T, b::T) where {T}
    n = if q.N % 2 == 0
        q.N
    else
        q.N + 1
    end
    h = (b - a) / n
    s = f(a) + f(b)
    s += 4sum(f.(a .+ collect(1:2:n) * h))
    s += 2sum(f.(a .+ collect(2:2:n-1) * h))
    return h / 3 * s
end

struct GaussLegendre <: Quadrature
    N::Int
    x::Vector{Float64} #quadrature abscissae in -1, 1
    w::Vector{Float64} #quadrature weights
    function GaussLegendre(N::Int=33)
        if N == 33
            xG = [0.9974246942464552, 0.9864557262306425, 0.9668229096899927, 0.9386943726111684, 0.9023167677434336, 0.8580096526765041, 0.8061623562741665, 0.7472304964495622, 0.6817319599697428, 0.610242345836379, 0.5333899047863476, 0.4518500172724507, 0.36633925774807335, 0.27760909715249704, 0.18643929882799157, 0.0936310658547334, 0, -0.0936310658547334, -0.18643929882799157, -0.27760909715249704, -0.36633925774807335, -0.4518500172724507, -0.5333899047863476, -0.610242345836379, -0.6817319599697428, -0.7472304964495622, -0.8061623562741665, -0.8580096526765041, -0.9023167677434336, -0.9386943726111684, -0.9668229096899927, -0.9864557262306425, -0.9974246942464552]
            wG = [0.0066062278475874535, 0.015321701512934681, 0.023915548101749465, 0.03230035863232906, 0.04040154133166957, 0.04814774281871171, 0.05547084663166358, 0.0623064825303174, 0.06859457281865682, 0.07427985484395423, 0.07931236479488682, 0.08364787606703872, 0.08724828761884422, 0.09008195866063856, 0.09212398664331695, 0.09335642606559616, 0.09376844616020999, 0.09335642606559616, 0.09212398664331695, 0.09008195866063856, 0.08724828761884422, 0.08364787606703872, 0.07931236479488682, 0.07427985484395423, 0.06859457281865682, 0.0623064825303174, 0.05547084663166358, 0.04814774281871171, 0.04040154133166957, 0.03230035863232906, 0.023915548101749465, 0.015321701512934681, 0.0066062278475874535]
        else
            xG, wG = gausslegendre(N)
        end
        new(N, xG, wG)
    end
end

function integrate(q::GaussLegendre, integrand, a::T, b::T) where {T}
    bma2 = (b - a) / 2
    bpa2 = (b + a) / 2
    i2 = bma2 * dot(q.w, integrand.(bma2 .* q.x .+ bpa2))
    # i2 = zero(T)
    # @sync Threads.@threads for i=1:length(p.w)
    #     i2 += p.w[i] * integrand(bma2 * p.x[i] + bpa2)
    # end
    # i2*=bma2
    return i2
end

struct GaussLegendreParallel <: Quadrature
    N::Int
    x::Vector{Float64} #quadrature abscissae in -1, 1
    w::Vector{Float64} #quadrature weights
    function GaussLegendreParallel(N::Int=33)
        xG, wG = gausslegendre(N)
        new(N, xG, wG)
    end
end

function integrate(q::GaussLegendreParallel, integrand, a::Float64, b::Float64)::Float64
    bma2 = (b - a) / 2
    bpa2 = (b + a) / 2
    i2 = zero(T)
    @sync Threads.@threads for i = 1:length(p.w)
        i2 += p.w[i] * integrand(bma2 * p.x[i] + bpa2)
    end
    i2 *= bma2
    return i2
end

struct GaussKronrod <: Quadrature
    rtol::Float64
    GaussKronrod(rtol::Float64=1e-8) = new(rtol)
end

function integrate(q::GaussKronrod, integrand, a::T, b::T)::T where {T}
    i2, err = quadgk(integrand, a, b, rtol=q.rtol)
    return i2
end
struct DoubleExponential <: Quadrature
    rtol::Float64
    DoubleExponential(rtol::Float64=1e-8) = new(rtol)
end

function integrate(q::DoubleExponential, integrand, a::T, b::T)::T where {T}
    i2, err = quadde(integrand, a, b, rtol=q.rtol)
    return i2
end

struct TanhSinh{T} <: Quadrature
    h::T
    y::Array{T,1}
    w::Array{T,1}
    tol::T
    isParallel::Bool
    function TanhSinh(n::Int, tol::T, h::T=zero(T), isParallel::Bool=false) where {T}
        y = Vector{T}(undef, n)
        w = Vector{T}(undef, n)
        if n <= 0
            throw(DomainError(n, "the number of points must be > 0"))
        end
        if is_zero(h)
            h = convert(T, lambertW(Float64(pi * n)) * 2 / (n+2))
        end
        for i = 1:n
            t = (2i - n - 1) * h / 2
            ct = cosh(t)
            st = sinh(t)
            ct2 = cosh(0.5 * pi * st)
            y[i] = tanh(0.5 * pi * st)
            w[i] = pi * h * ct / (2 * ct2^2)
        end
        w_sum = sum(w)
        @. w = 2 * w / w_sum #adjust the weights such that they sum exactly to 2.0
        # println("w=",w)
        return new{T}(h, y, w, tol, isParallel)
    end
end

function integrate(q::TanhSinh{T}, integrand, a::T, b::T)::T where {T}
    if b <= a
        return zero(T)
    end
    bma2 = (b - a) / 2
    bpa2 = (b + a) / 2
    I = Base.zero(T)
    if q.isParallel
        @sync Threads.@threads for (wi, yi) = collect(zip(q.w, q.y))
            zi = bpa2 + bma2 * yi
            fyi = integrand(zi)
            I += wi * fyi
        end
    else
        I = dot(q.w, integrand.(bma2 .* q.y .+ bpa2))
    end
    return I * bma2
end

