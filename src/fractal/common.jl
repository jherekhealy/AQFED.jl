using AQFED.Math, AQFED.Black
using Images

abstract type FractalSolver end
struct Householder <: FractalSolver end
struct FractalSRSolver <: FractalSolver
    srSolver::SRSolver
end
struct Newton <: FractalSolver end

abstract type FractalColoring end
struct PaletteColoring <: FractalColoring
    palette::Vector{RGB}
    power::Float64
    isInverted::Bool
end
function PaletteColoring()
    return PaletteColoring([RGB(5.0 / 255, 5.0 / 255, 30.0 / 255), RGB(80.0 / 255, 20.0 / 255, 20.0 / 255), RGB((255 - 20.0) / 255, (245 - 20.0) / 255, (215 - 20.0) / 255)], 1.0, false)
end

struct SpectralColoring <: FractalColoring
    power::Float64
    isInverted::Bool
end
function SpectralColoring()
    return SpectralColoring(0.5, true)
end

struct SinColoring <: FractalColoring
    rMultiplier::Float64
    gMultiplier::Float64
    bMultiplier::Float64
    factor::Float64
    power::Float64
    isInverted::Bool
end
SinColoring() = SinColoring(0.015, 0.013, 0.01, 300.0, 0.5, false)




function makeFractal(w::Int, h::Int, xMin, xMax, yMin, yMax; maxIter=32, f=bachelierVolatilityFunctionC3, solver=Householder(), accuracy=1e-8, coloring=PaletteColoring())
    x = collect(range(xMin, xMax, w))
    y = collect(range(yMin, yMax, h))
    xy = zeros(Float64, w, h)
    maxIdx = 1.0
    minIdx = 0.0
    @sync @inbounds for i = eachindex(x)
        Threads.@spawn @inbounds for j = eachindex(y)
            _, idx =  iterationSize(solver, f, complex(x[i], y[j]), n=maxIter, r=accuracy)
            maxIdx = max(idx, maxIdx)
            minIdx = min(idx, minIdx)
            xy[i, j] = idx
        end
    end
    img = zeros(RGB, h, w)
    @inbounds for i = eachindex(x)
        @inbounds @simd for j = eachindex(y)
            idx = xy[i, j]
            rgb = complexHeatMap(coloring, idx, minIdx, maxIdx)
            img[w-j+1, i] = rgb
        end
    end
    return img
end

iterationSize(solver::Householder, f, b; n::Integer, r::Float64) = householderIterationSize(f, b, n=n, r=r)
iterationSize(solver::Newton, f, b; n::Integer, r::Float64) = newtonIterationSize(f, b, n=n, r=r)
iterationSize(solver::FractalSRSolver, f, b; n::Integer, r::Float64) = iterationSize(solver.srSolver, f, b, n=n, r=r)



function newtonIterationSize(f, b; n::Integer, r::Float64)
    fb, fpb = f(b)
    fa = fb
    ftol = r
    xtol = r
    iterations = 0
    if abs(fb) > ftol
        while iterations <= n
            x0 = b
            x1 = x0 - fb / fpb
            a = x0
            fa = fb
            fb, fpb = f(x1)
            if isnan(fb) || isnan(fpb)
                # println("NaN value for", a, " ", fa, " ", b, " ", fb, " ", fpb)
                break
            end
            b = x1
            xtol_ = xtol
            ftol_ = ftol
            if abs(b - a) <= xtol_
                break
            end

            if abs(fb) <= ftol_
                break
            end
            iterations += 1
        end
    end
    if isnan(fb) || isnan(fa)
        return fb, n + 1
    end
    afb = abs(fb)
    if afb > ftol
        return fb, n + 1
    end
    afa = max(1e-16, abs(fa))
    afb = max(1e-16, afb)
    zmag = (log(ftol) - log(afa)) / (log(afb) - log(afa))
    if afa == 0 || afb == 0
        zmag = 0.0
    end
    if afa == afb
        if afb < ftol
            zmag = 0.0
        end
    end
    if zmag > 1
        println("zmag>1", ftol, fa, fb)
    end
    mu = iterations + zmag
    return fb, mu
end

function iterationSize(solver::SRSolver, f, b; n::Integer, r::Float64)
    fb, fpb, fp2b = f(b)
    fa = fb
    ftol = r
    xtol = r
    iterations = 0
    if abs(fb) > ftol
        while iterations <= n
            x0 = b
            fbOverfpb = fb / fpb
            fp2bOverfpb = fp2b / fpb
            lf = fbOverfpb * fp2bOverfpb
            # nfpbsq = nfpb * nfpb
            x1 = x0 + Black.srSolve(solver, fb, fbOverfpb, lf)
            a = x0
            fa = fb
            fb, fpb, fp2b = f(x1)
            if isnan(fb) || isnan(fpb)
                # println("NaN value for", a, " ", fa, " ", b, " ", fb, " ", fpb)
                break
            end
            b = x1
            xtol_ = xtol
            ftol_ = ftol
            if abs(b - a) <= xtol_
                break
            end

            if abs(fb) <= ftol_
                break
            end
            iterations += 1
        end
    end
    if isnan(fb) || isnan(fa)
        return fb, n + 1
    end
    afb = abs(fb)
    if afb > ftol
        return fb, n + 1
    end
    afa = max(1e-16, abs(fa))
    afb = max(1e-16, afb)
    zmag = (log(ftol) - log(afa)) / (log(afb) - log(afa))
    if afa == 0 || afb == 0
        zmag = 0.0
    end
    if afa == afb
        if afb < ftol
            zmag = 0.0
        end
    end
    if zmag > 1
        println("zmag>1", ftol, fa, fb)
    end
    mu = iterations + zmag
    # println("b=",b)
    return fb, mu
end

function householderIterationSize(f, b; n::Integer, r::Float64)
    fb, fpb, fp2b, fp3b = f(b)
    fa = fb
    ftol = r
    xtol = r
    iterations = 0
    if abs(fb) > ftol
        while iterations <= n
            x0 = b
            nfpb = fpb / fb
            nfpbsq = nfpb * nfpb
            x1 = x0 - (fpb * nfpb - fp2b / 2) / (nfpbsq * fpb - nfpb * fp2b + fp3b / 6)

            a = x0
            fa = fb
            fb, fpb, fp2b, fp3b = f(x1)
            if isnan(fb) || isnan(fpb)
                # println("NaN value for", a, " ", fa, " ", b, " ", fb, " ", fpb)
                break
            end
            b = x1
            xtol_ = xtol
            ftol_ = ftol
            if abs(b - a) <= xtol_
                break
            end

            if abs(fb) <= ftol_
                break
            end
            iterations += 1
        end
    end
    if isnan(fb) || isnan(fa)
        return fb, n + 1
    end
    afb = abs(fb)
    if afb > ftol
        return fb, n + 1
    end
    afa = max(1e-16, abs(fa))
    afb = max(1e-16, afb)
    zmag = (log(ftol) - log(afa)) / (log(afb) - log(afa))
    if afa == 0 || afb == 0
        zmag = 0.0
    end
    if afa == afb
        if afb < ftol
            zmag = 0.0
        end
    end
    if zmag > 1
        println("zmag>1", ftol, fa, fb)
    end
    mu = iterations + zmag
    return fb, mu
end

function spectralColor(l)
    r = 0.0
    g = 0.0
    b = 0.0
    if ((l >= 400.0) && (l < 410.0))
        t = (l - 400.0) / (410.0 - 400.0)
        r = +(0.33 * t) - (0.20 * t * t)
    elseif ((l >= 410.0) && (l < 475.0))
        t = (l - 410.0) / (475.0 - 410.0)
        r = 0.14 - (0.13 * t * t)
    elseif ((l >= 545.0) && (l < 595.0))
        t = (l - 545.0) / (595.0 - 545.0)
        r = +(1.98 * t) - (t * t)
    elseif ((l >= 595.0) && (l < 650.0))
        t = (l - 595.0) / (650.0 - 595.0)
        r = 0.98 + (0.06 * t) - (0.40 * t * t)
    elseif ((l >= 650.0) && (l < 700.0))
        t = (l - 650.0) / (700.0 - 650.0)
        r = 0.65 - (0.84 * t) + (0.20 * t * t)
    end
    if ((l >= 415.0) && (l < 475.0))
        t = (l - 415.0) / (475.0 - 415.0)
        g = +(0.80 * t * t)
    elseif ((l >= 475.0) && (l < 590.0))
        t = (l - 475.0) / (590.0 - 475.0)
        g = 0.8 + (0.76 * t) - (0.80 * t * t)
    elseif ((l >= 585.0) && (l < 639.0))
        t = (l - 585.0) / (639.0 - 585.0)
        g = 0.84 - (0.84 * t)
    end
    if ((l >= 400.0) && (l < 475.0))
        t = (l - 400.0) / (475.0 - 400.0)
        b = +(2.20 * t) - (1.50 * t * t)
    elseif ((l >= 475.0) && (l < 560.0))
        t = (l - 475.0) / (560.0 - 475.0)
        b = 0.7 - (t) + (0.30 * t * t)
    end
    return RGB(r, g, b)
end

function transformZeroOne(coloring, t)
    ##t = (1 - cos(frac * π)) / 2    
    return if coloring.isInverted
        (1 - t)^coloring.power
    else
        t^coloring.power
    end
end

function complexHeatMap(coloring::SpectralColoring, value, min, max)
    frac = Float64((value - min) / (max - min))
    t = transformZeroOne(coloring, frac)
    return spectralColor(400.0 + (300.0 * t))
end

function complexHeatMap(coloring::PaletteColoring, value, min, max)
    frac = Float64((value - min) / (max - min))
    t = transformZeroOne(coloring, frac)
    pmax = length(coloring.palette) - 1
    t = (1 - frac)^0.5
    tp = floor(t * pmax)
    t = t * pmax - tp
    pIndex = Int(tp) + 1
    pmax += 1
    if pIndex == pmax
        return coloring.palette[pmax]
    else
        return coloring.palette[pIndex] * (1 - t) + t * coloring.palette[pIndex+1]
    end
end


function complexHeatMap(coloring::SinColoring, value, min, max)
    frac = Float64((value - min) / (max - min))
    t = transformZeroOne(coloring, frac)
    t *= coloring.factor
    return RGB((sin(coloring.rMultiplier * frac + 4) * 230 + 25) / 255,
        (sin(coloring.gMultiplier * frac) * 230 + 25) / 255,
        (sin(coloring.bMultiplier * frac + 1) * 230 + 25) / 255)
end

# *(x::HSV{T}, f::T) where {T} = HSV(x.h * f, x.s * f, x.v * f)
# *(f::T, x::HSV{T}) where {T} = x * f
# +(x::HSV{T}, y::HSV{T}) where {T} = HSV(x.h + y.h, x.s + y.s, x.v + y.v)
#usage:
# factor = 2.5; w = 1024; h = 1024;
# img = AQFED.Fractal.makeFractal(w,h,-factor,factor,-factor,factor);
#using FileIO
# save(File{format"PNG"}("test.png"),img)
