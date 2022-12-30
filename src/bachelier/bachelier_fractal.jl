using AQFED.Math
using Images

function bachelierVolatilityFunctionC3(v; forward=1.0, moneyness=exp(-1.0), targetPrice=0.12693673750664397)
    fmk = forward - forward / moneyness
    h = fmk / v
    cEstimate = fmk * normcdf(h) + v * normpdf(h)
    vega = normpdf(h)
    volga = vega * h^2 / v
    c3 = vega * (-3 * h^2 / v^2 + h^4 / v^2)
    return cEstimate - targetPrice, vega, volga, c3
end

function impliedVolatilityFractal(w::Int, h::Int, xMin, xMax, yMin, yMax; maxIter=32, f=bachelierVolatilityFunctionC3, accuracy=1e-8, palette=[RGB(5.0/255, 5.0/255, 30.0/255), RGB(80.0/255, 20.0/255, 20.0/255), RGB((255 - 20.0)/255, (245 - 20.0)/255, (215 - 20.0)/255)]  )

    x = collect(range(xMin, xMax, w))
    y = collect(range(yMin, yMax, h))
    xy = zeros(Float64, w, h)
    maxIdx = 1.0
    minIdx = 0.0
    @sync @inbounds for i = 1:length(x)
        Threads.@spawn @inbounds for j = 1:length(y)
            valueij,idx = householderIterationSize(f, complex(x[i] , y[j]), n=maxIter, r=accuracy)
            maxIdx = max(idx, maxIdx)
            minIdx = min(idx, minIdx)
            xy[i, j] = idx
        end
    end
    img = zeros(RGB, h, w)
    @inbounds for i = 1:length(x)
        @inbounds @simd for j = 1:length(y)
            idx = xy[i, j]
            rgb = complexHeatMap(idx, minIdx, maxIdx, palette)
            img[j, i] = rgb
        end
    end
    return img
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
            iterations+=1
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


function complexHeatMap(value, min, max, palette)
    frac = (value - min) / (max - min)
    pmax = length(palette) - 1
    #t = (1 - cos(frac * π)) / 2
    t = frac
    tp = floor(t * pmax)
    t = t * float64(pmax) - tp
    pIndex = Int(tp) + 1
    pmax +=1
    if pIndex == pmax
        return palette[pmax]
    else
        return palette[pIndex] * (1 - t) + t * palette[pIndex+1]
    end
end
#usage:
# factor = 2.5; w = 1024; h = 1024;
# img,idx = Bachelier.impliedVolatilityFractal(w,h,-factor,factor,-factor,factor);
#using FileIO
# save(File{format"PNG"}("test.png"),img)
