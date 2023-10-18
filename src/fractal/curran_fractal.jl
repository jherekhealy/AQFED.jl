using AQFED.Math,AQFED.Random
using Images
using LinearAlgebra
function curranFractal(w::Int, h::Int, xMin::T, xMax, yMin::TV, yMax; maxIter=32, accuracy=1e-8, coloring=SpectralColoring(),solver=FractalSRSolver(SuperHalley()),isLog=false) where {T,TV}
    nAsset = 10
    spot = ones(nAsset)
    forward = spot
    # totalVariance = collect(range(1/(nAsset-1),stop=1.0,length=nAsset))
    totalVariance=  rand(MRG32k3a(),nAsset)
    weight = ones(nAsset) ./ nAsset
    correlation = ones((nAsset, nAsset)) .* 0.5
    correlation[diagind(correlation)] .= 1.0
    strike = 1.0

    n = length(spot)
    δ = zeros(TV, n) #we store δ * S here.
    f = zero(T)
    u1 = zero(T)
    for (i, wi) in enumerate(weight)
            δ[i] = forward[i]
        f += wi * δ[i]
        u1 += wi * forward[i]
    end
    sumwbd = zero(TV)
    wsdv = zeros(TV, n)

    for (i, wi) in enumerate(weight)
        wTildei = wi * δ[i] / f
        βi = log(forward[i]) - totalVariance[i] / 2
        sumwbd += wTildei * (βi - log(δ[i]))
        if totalVariance[i] > 0
            wsdv[i] = wi * δ[i] * sqrt(totalVariance[i])
        end
    end
    dGamma = f * (log(strike / f) - sumwbd)
    println("dGamma=", dGamma)
    varGamma = zero(TV)
    r = zeros(TV, n)
    for (i, wsdi) in enumerate(wsdv)
        for (j, wsdj) in enumerate(wsdv)
            covar = wsdj * correlation[i, j]
            varGamma += covar * wsdi
            r[i] += covar
        end
    end
    varGamma = max(varGamma, sqrt(eps(T)))
    volGamma = sqrt(varGamma)
    r ./= volGamma
    function objective(λ)
        eS = zero(typeof(λ))
        DeS = zero(typeof(λ))
        D2eS = zero(typeof(λ))
        for (i, fi) in enumerate(forward)
            viT = totalVariance[i]
            eSi = weight[i] * fi
            if viT > 0
                sqrti = sqrt(viT)
                eSi *= exp(-r[i]^2 * viT / 2 + r[i] * sqrti * (λ) / (volGamma))
                DeS += eSi * (r[i] * sqrti) / volGamma
                D2eS += eSi * ((r[i] * sqrti) / volGamma)^2
            end
            eS += eSi
        end
        #  println(λ," ", eS-strike," ", DeS, " ",D2eS)
        return (eS - strike,DeS, D2eS)
    end
    function objectiveLog(λ)
        eSstrike,DeS, D2eS = objective(λ)
        eS = eSstrike + strike
        return (log(eS)-log(strike), DeS/eS, -DeS^2/eS^2+D2eS/eS)
    end
    #init guess dGamma
   println(objective.(range(dGamma/10,stop=dGamma,length=10)))
    println("Halley ",iterationSize(Black.Halley(),objective,dGamma*8;n=32,r=1e-8))
    println("SuperHalley ",iterationSize(Black.SuperHalley(),objective,dGamma*8;n=32,r=1e-8))
    println("CMethod ",iterationSize(Black.CMethod(),objective,dGamma*8;n=32,r=1e-8))
    println("Halley ",iterationSize(Black.Halley(),objectiveLog,dGamma*8;n=32,r=1e-8))
    println("SuperHalley ",iterationSize(Black.SuperHalley(),objectiveLog,dGamma*8;n=32,r=1e-8))
    println("CMethod ",iterationSize(Black.CMethod(),objectiveLog,dGamma*8;n=32,r=1e-8))
    #maybe try C2/SupperHalley
    return makeFractal(w, h, xMin, xMax, yMin, yMax, maxIter=maxIter, f=if isLog objectiveLog else objective end, solver=solver, accuracy=accuracy, coloring=coloring)
end

# palette = [RGB(0/255, 0.0/255, 255.0/255), RGB(255/255, 0/255, 0/255)]
#palette=[RGB(0,0.0,0.0),RGB(0,0.3,0.1),RGB(0,0.5,0.2),RGB(0.7,0.9,0.4)]
