using AQFED.TermStructure
import AQFED.Math: normcdf, normpdf, lambertW, Bijection, IdentityTransformation
import AQFED.Math: inv as invtransform
import AQFED.Black: blackScholesFormula
import AQFED.Math: norminv
import Roots: find_zero, Newton, A42
using ForwardDiff

export AndersenLakeGRepresentation, priceAmerican
#@inline normcdf(z::Float64) = normcdfCody(z) #Faster apparently
abstract type ALQuadrature end
abstract type ALCollocation end

struct AndersenLakeGRepresentation{TM,TB <:Bijection}
    isCall::Bool
    model::TM
    dividends::Vector{Dividend{Float64}}
    tauMax::Float64
    tauHat::Float64
    t::Vector{Float64} #split times
    capX::Float64
    collocation::ALCollocation #Cheb, by piece includes transf to -1,1?
    quadrature::ALQuadrature #Tanh sinh, by piece
    bTransform::TB
end

function makeSplitTimes(tauMax::Float64, nPP::Int, dividends::AbstractArray{Dividend{Float64}}=Dividend{Float64}[])
    if isempty(dividends)
        taus = Vector{Float64}()
        for i = 1:nPP
            append!(taus, tauMax * (i - 1) / (nPP))
        end
    else
        taus = [dividend.exDate for dividend in dividends]
        if nPP < length(dividends)
            prepend!(taus, 0.0)
            nPP = length(taus)
        else
            for i = 1:nPP
                append!(taus, tauMax * (i - 1) / (nPP))
            end
            nPP = length(taus)
        end
        sort!(taus)
    end
    append!(taus, tauMax)
    return taus
end

struct TanhSinhQuadrature <: ALQuadrature
    nTS1::Int
    wvec::Vector{Float64}
    yvec::Vector{Float64}
    abscissae::Vector{Float64}
    scaledWeights::Vector{Float64}
end

function TanhSinhQuadrature(nTS1::Int)
    wvec = zeros(nTS1)
    yvec = zeros(nTS1)
    ndiv2 = trunc(Int, (nTS1 + 1) / 2)
    hn = tanhsinhStep(nTS1)
    hvec = hn .* (ndiv2-1:-1:1)
    svec = @. sinh(hvec) * pi / 2
    @. @view(yvec[1:ndiv2-1]) = tanh(svec) #may be more precise to store y+1 directly instead
    @. @view(wvec[1:ndiv2-1]) = hn * pi * cosh(hvec) / (2 * (cosh(svec))^2)
    for i = 1:ndiv2-1
        yvec[nTS1+1-i] = -yvec[i]
        wvec[nTS1+1-i] = wvec[i]
    end
    yvec[ndiv2] = 0
    wvec[ndiv2] = pi * hn / 2
    reverse!(yvec)
    reverse!(wvec)
    abscissae = collect(range(-1.0, stop=1.0, length=nTS1))
    scaledWeights = copy(wvec)
    return TanhSinhQuadrature(nTS1, wvec, yvec, abscissae, scaledWeights)
end

function updateInterval(quadrature::TanhSinhQuadrature, ti, tauMax)
    #z = 2*sqrt(t-ti)/sqrt(tauMax-ti)-1, dz= 1/sqrt(t-ti) / sqrt(tau-ti) dt = 2/(z+1)/(tauMax-ti)dt
    # t = (z+1)^2 /4 * (tau-ti) + ti
    # @. quadrature.abscissae = (tauMax-ti)/4 * (1+quadrature.yvec)^2 + ti
    # @. quadrature.scaledWeights = (tauMax-ti) * collocation.wvec * (yvec+1)/2
    # z = 2 * (t-ti / tauMax-ti)  -1   dz = 2 * dt/(tauMax-ti)    ,  t = (z+1)/2
    @. quadrature.abscissae = (tauMax - ti) * (1 + quadrature.yvec) / 2 + ti
    @. quadrature.scaledWeights = (tauMax - ti) / 2 * quadrature.wvec
end

function Base.length(quadrature::TanhSinhQuadrature)
    return quadrature.nTS1
end

function abscissaeAndWeights(quadrature::TanhSinhQuadrature)
    return quadrature.abscissae, quadrature.scaledWeights
end



struct PPQuadrature{T} <: ALQuadrature
    nPP::Int
    taus::Vector{Float64}
    delegate::Vector{T}
    isActive::Vector{Bool}
    isDiscontinuous::Bool
end
function TanhSinhPPQuadrature(taus::Vector{Float64}, nTS::Int;isDiscontinuous::Bool=true)
    nPP = length(taus)-1
    v = Array{TanhSinhQuadrature,1}(undef, nPP)
    q = TanhSinhQuadrature(nTS)
    for i=1:nPP
        v[i] = TanhSinhQuadrature(nTS,q.wvec,q.yvec,copy(q.abscissae),copy(q.scaledWeights))
        if isDiscontinuous && i < nPP
            updateInterval(v[i], taus[i],taus[i+1]-eps(taus[i+1]))
        else
            updateInterval(v[i], taus[i],taus[i+1])
        end
    end
    isActive = ones(Bool, nPP)
    return PPQuadrature{TanhSinhQuadrature}(nPP, taus, v, isActive,isDiscontinuous)
end
function abscissaeAndWeights(quadrature::PPQuadrature{T}) where {T}    
    len = 0
    for c in quadrature.delegate
        len += length(c)
    end
    t = zeros(Float64, len) #TODO cache in PPQuadrature
    w = zeros(Float64, len)
    currentIndex = 1
    for (cIndex,c) in enumerate(quadrature.delegate)
        if quadrature.isActive[cIndex]
            ti,wi = abscissaeAndWeights(c)    
            t[currentIndex:currentIndex+length(ti)-1] = ti[1:end]
            w[currentIndex:currentIndex+length(ti)-1] = wi[1:end]
            currentIndex+=length(ti)
        end
    end
    return (@view(t[1:currentIndex-1]),@view(w[1:currentIndex-1]))
end
function updateInterval(quadrature::PPQuadrature, ti, tauMax)
    #find pp corresponding to ti, update to ti,ppTauMAx, 
    i=1
    while i<quadrature.nPP && quadrature.taus[i+1] < ti + (eps(ti))
        quadrature.isActive[i] = false
        i+=1
    end
    quadrature.isActive[i] = true
    
    if quadrature.isDiscontinuous && i < quadrature.nPP
        updateInterval(quadrature.delegate[i],ti,quadrature.taus[i+1]-eps(quadrature.taus[i+1]))
    else
        updateInterval(quadrature.delegate[i],ti,quadrature.taus[i+1])       
    end
    for j=i+1:quadrature.nPP
        quadrature.isActive[j] = true
        if quadrature.isDiscontinuous && j < quadrature.nPP
            updateInterval(quadrature.delegate[j],quadrature.taus[j],quadrature.taus[j+1]-eps(quadrature.taus[j+1]))
        else
            updateInterval(quadrature.delegate[j],quadrature.taus[j],quadrature.taus[j+1])
        end
    end   
end

function Base.length(quadrature::PPQuadrature)
    len = 0
    for q in quadrature.delegate
        len += length(q)
    end
    return len
end

struct ChebyshevCollocation <: ALCollocation
    nC::Int
    avec::Vector{Float64}
    tauMin::Float64
    tauMax::Float64
end

function ChebyshevCollocation(nC::Int, tauMax::Float64; start::Float64 = 0.0)
    avec = zeros(Float64, nC + 1)
    return ChebyshevCollocation(nC, avec, start, tauMax)
end

function Base.length(collocation::ChebyshevCollocation)
    return collocation.nC+1
end

function abscissae(collocation::ChebyshevCollocation)
    nC = collocation.nC
    t = zeros(Float64, nC + 1)
    for i = 1:nC+1
        zi = cos((i-1) * pi / nC)
        t[i] = (collocation.tauMax-collocation.tauMin)  *(1 -  (1 + zi)^2 / 4) + collocation.tauMin
        # println("z ",zi," t ",t[i])
    end
    return t
end
# function evaluate(collocation::ChebyshevCollocation, u::Vector{Float64})
#     chebQck(collocation.avec, u)
# end

function evaluate!(qValues::AbstractArray{Float64}, collocation::ChebyshevCollocation, qAbscissae::AbstractArray{Float64})
    for i = eachindex(qAbscissae)
        u = (collocation.tauMax-qAbscissae[i])/(collocation.tauMax-collocation.tauMin)
        # println(u)
        z = 2*sqrt(max(u,0.0)) - 1 
        # println("t ",qAbscissae[i], " z ",z)
        qValues[i] = chebQck(collocation.avec, z)
    end
end

function computeCollocation(collocation::ChebyshevCollocation, qvec::AbstractArray{Float64})
    updateAvec!(collocation.avec, collocation.nC, qvec)
end

struct PPCollocation{T} <: ALCollocation
    nPP::Int
    taus::Vector{Float64}
    delegate::Vector{T} #vector of cheb colo
    isDiscontinuous::Bool
end

function ChebyshevPPCollocation(taus::Vector{Float64}, nC::Int; isDiscontinuous=true) #from 0,tau1,tau2,...,tauMax
    #TODO handle discont : create a vector with tau1, tau1+eps, tau2, tau2+eps,....
    nPP = length(taus)-1
    v = Array{ChebyshevCollocation,1}(undef, nPP)
    for i=1:nPP
        if isDiscontinuous && i < nPP
          v[i] = ChebyshevCollocation(nC, taus[i+1]-eps(taus[i+1]),start=taus[i])
        else
            v[i] = ChebyshevCollocation(nC, taus[i+1],start=taus[i])      
        end
    end
    return PPCollocation{ChebyshevCollocation}(nPP, taus, v,isDiscontinuous)
end

#at div date: not jump yet, jump is eps before.
function abscissae(collocation::PPCollocation{T}) where {T}
    nP = collocation.nPP
    len = 1
    for c in collocation.delegate
        len += length(c) - 1
        if collocation.isDiscontinuous
            len += 1
        end
    end
    t = zeros(Float64, collocation.isDiscontinuous ? len-1 : len)
    currentIndex = 1
    for c in collocation.delegate
        ti = abscissae(c)    
        t[currentIndex:currentIndex+length(ti)-1] = ti[1:end]
        if collocation.isDiscontinuous
            currentIndex+=length(ti)
        else
            currentIndex+=length(ti)-1
        end
    end
    return t
end

function evaluate!(qValues::AbstractArray{Float64}, collocation::PPCollocation{T}, qAbscissae::AbstractArray{Float64}) where {T}
    #FIXME what if qqAbsiccsiaee is in reverse order?
    iAbscissae = 1
    for i = 1:collocation.nPP
        start = iAbscissae
        while iAbscissae <= length(qAbscissae) && collocation.taus[i+1] >= qAbscissae[iAbscissae]
            iAbscissae += 1
        end
        if iAbscissae != start
            #println(iAbscissae," ",start," ",qAbscissae[start]," ",qAbscissae[iAbscissae-1]," ",abscissae(collocation.delegate[i])," ",qAbscissae)
            evaluate!(@view(qValues[start:iAbscissae-1]), collocation.delegate[i], @view(qAbscissae[start:iAbscissae-1]))
        end
        if iAbscissae > length(qAbscissae)
            break
        end
    end
end

function computeCollocation(collocation::PPCollocation{T}, qvec::AbstractArray{Float64}) where {T}
    #qvec vector of length absciasses
    iAbscissae = 1
    for i=1:collocation.nPP
        len = length(collocation.delegate[i])
        if collocation.isDiscontinuous && i != collocation.nPP
            len+=1
        end
         computeCollocation(collocation.delegate[i],@view(qvec[iAbscissae:iAbscissae+len-1]))
        iAbscissae += len-1
    end
end


function AndersenLakeGRepresentation(
    mmodel::TSBlackModel,
    tauMax::Float64,
    isCall::Bool=false;
    atol::Float64=1e-8,
    nIter::Int=8,
    isLower::Bool=false,
    collocation::ALCollocation=ChebyshevCollocation(7, tauMax), #Cheb, by piece includes transf to -1,1?
    quadrature::ALQuadrature=TanhSinhQuadrature(31), #Tanh sinh, by piece
    dividends::AbstractArray{Dividend{Float64}}=Dividend{Float64}[],
    transform::TB=IdentityTransformation{Float64}()
) where {TB <: Bijection}

    K = 1.0
    local model::TSBlackModel = mmodel
    if isCall  #use McDonald and Schroder symmetry
        model = TSBlackModel(model.surface, SpreadCurve(model.discountCurve, model.driftCurve), model.discountCurve)
    end

    rShort = -(logDiscountFactor(model, tauMax + 1e-7) - logDiscountFactor(model, tauMax)) / 1e-7
    qShort = rShort - (logForward(model, 0.0, tauMax + 1e-7) - logForward(model, 0.0, tauMax)) / 1e-7
    vol = sqrt(varianceByLogmoneyness(model, 0.0, tauMax))
    capX = isLower ? K * rShort / qShort : K

    if qShort > rShort
        capX = K * rShort / qShort
    end
    logCapX = log(capX)
    tauHat = tauMax
    # if r < 0 && q < r && vol >= sqrt(-2 * q) - sqrt(-2 * r)
    #     #double boundary which intersect before infinite time
    #     objHat = function (τ)
    #         t = τ
    #         value = abs(norminv(-expm1(q * t)) - norminv(-expm1(r * t))) / sqrt(t) - vol
    #         # println(τ, " v ", value)
    #         return value
    #     end
    #     if objHat(tauMax) < 0 #
    #         # derHat =  x -> ForwardDiff.derivative(objHat,float(x))
    #         #  tauHat = (find_zero((objHat,derHat), sqrt(tauMax), Newton()))^2
    #         tauHat = find_zero(objHat, (1e-7, tauMax), A42())
    #         #    println("tauHat ", tauHat)
    #         tauHat = min(tauHat, tauMax)
    #     end
    # end

    t = abscissae(collocation)  #  ...,td- ,td ...   OR qvec of length nC and iterate on each piece? 
    #tanhsinh will query at td- or other quad. Value should be diff from td. Ok to use td-eps.
    qvec = zeros(Float64, length(t))
    local fprev = capX
    r = -logDiscountFactor(model, tauHat) / tauHat
    q = r - logForward(model, 0.0, tauHat) / tauHat
    modelB = ConstantBlackModel(vol, r, q)
    #println("t ",t," ",modelB," ",capX)
    qvec[end] = transform(fprev)
    for i = length(t)-1:-1:1
        fi = americanBoundaryPutQDP(isLower, modelB, fprev, K, tauMax - t[i], atol)
        #println(tauMax - t[i]," ",fi)
        fprev = fi
        qvec[i] = transform(fi / capX)
    end
    # println("init-qvec ", qvec')
    if nIter == 0
        computeCollocation(collocation, qvec)
    end
    dftauMax = discountFactor(model, tauMax)
    ftauMax = forward(model, 1.0, tauMax)
    qValues = zeros(Float64, length(quadrature))
    Bmin = 1e-16
    for j = 1:nIter
        qvec[end] = transform(1.0)
        for i = length(t)-1:-1:1
            isOnExDate = false
            for dividend in dividends
                if dividend.exDate == t[i+1]
                    isOnExDate = true
                    break
                end
            end
            if isOnExDate
                qvec[i] = transform(Bmin) #just before exDate boundary = 0.
            end
        end
        computeCollocation(collocation, qvec)
        for i = length(t)-1:-1:1
            if abs(t[i+1]-t[i])< sqrt(eps(t[i+1]))
                continue
            end
            #compute integral from ti to T. using collocation
            ti = t[i]
            lnBti = logCapX + log(invtransform(transform, qvec[i]))
            updateInterval(quadrature, ti, tauMax)
            qAbscissae, qWeights = abscissaeAndWeights(quadrature)
            evaluate!(qValues, collocation, qAbscissae)
            fi = forward(model, 1.0, ti)
            dfi = discountFactor(model, ti)
            #FIXME: we are integrating the same piece over and over. We could detect similar qAbscissae values and cache.
            sumk1 = 0.0
            sumk2 = 0.0
            for (iu, u) = enumerate(qAbscissae)
                lnBu = logCapX + log(max(invtransform(transform, qValues[iu]),Bmin))
                sqrtv = sqrt(max(varianceByLogmoneyness(model, 0.0, u) * u - varianceByLogmoneyness(model, 0.0, ti) * ti, 0.0))
                if sqrtv == 0.0
                    #    return (0.0,0.0)
                else
                    fu = forward(model, 1.0, u)
                    frac = fu / fi
                    dfu = discountFactor(model, u)
                    rFrac = dfu / dfi
                    qFrac = frac * rFrac
                    rRate = -(log(discountFactor(model, u + 1e-7)) - log(dfu)) / (1e-7) #log(discountFactor(model, (tauHat - taui) + tauk)) / ((tauHat - taui) + tauk)
                    qRate = rRate - (log(forward(model, 1.0, u + 1e-7)) - log(fu)) / 1e-7
                    # objrRate = function(x) 
                    #     -log(discountFactor(model, x))
                    # end
                    # rRate = ForwardDiff.derivative( objrRate,  (tauHat - taui) + tauk)
                    gk = dividendGrowthFactor(dividends, ti, u)

                    d1 = ((lnBti - lnBu) + log(frac * gk)) / sqrtv + sqrtv / 2
                    d2 = d1 - sqrtv
                    k1 = qFrac * qRate * gk* normcdf(-d1)
                    k2 = rFrac * rRate * normcdf(-d2)
                    sumk1 += qWeights[iu] * k1
                    sumk2 += qWeights[iu] * k2
                end
            end
            rFrac = dftauMax / dfi
            qFrac = rFrac * ftauMax / fi
            sqrtv = sqrt(max(-varianceByLogmoneyness(model, 0.0, ti) * ti + varianceByLogmoneyness(model, 0.0, tauMax) * tauMax, 0.0))
            gi = dividendGrowthFactor(dividends, ti, tauMax)
             sumgi = 0.0
          
            d1i = ((lnBti - log(K *rFrac / (qFrac*gi)))) / sqrtv + sqrtv / 2
            d2i = d1i - sqrtv

            Ni, Di = if isLower
                dfi / dftauMax - 1 - sumk2, ftauMax * dftauMax / (fi * dfi) - 1 - sumk1
            else
                1 - rFrac * normcdf(-d2i) - sumk2, 1 - qFrac * normcdf(-d1i) * gi - sumk1 - sumgi
            end
            # println("AL ", Ni, " ", Di)
            # if Di == 0.0 && Ni == 0.0
            #     #use asymptotic expansion cdf = erfc(-x/sqrt2)/2 and erfc(x) = e^{-x^2}/(x*sqrtpi)*(1-1/(2*x^2))
            #     NiOverDi = exp(-(d2i^2 - d1i^2) / 2) * (d1i / d2i)
            # end
            fi = K * Ni / Di
            if fi <= 0
                # B = Kstar * N/D   to B = B + Kstar*N - B*D
                #lnBtaui = isLower ? logCapX + sqrt(qvec[i]) : logCapX - sqrt(qvec[i])
                Bti = exp(lnBti)
                fi = Bti + K * Ni - Bti * Di
            end
            lfc = transform(fi / capX)
            if isnan(lfc)
                throw(DomainError(
                    fi,
                    string("Nan qvec ", capX, " ", lnBti, " ", qvec[i],  " ", i),
                ))
            end
            qvec[i] = lfc

        end
        #println("iter ",qvec)
    end

    return AndersenLakeGRepresentation(
        isCall,
        model, dividends,
        tauMax,
        tauHat,
        t,  capX,
        collocation, quadrature, transform
    )
end

function exerciseBoundary(p::AndersenLakeGRepresentation{TSBlackModel{TS,TC1,TC2}}, K::Float64, t::AbstractArray{Float64}) where {TS,TC1,TC2}
    Bzk = zeros(Float64, length(t))
    evaluate!(Bzk, p.collocation, t)
     @. Bzk = invtransform(p.bTransform,Bzk)*K *p.capX
    return Bzk
end

function priceAmerican(p::AndersenLakeGRepresentation{TSBlackModel{TS,TC1,TC2}}, K::Float64, S::Float64;   quadraturePrice::ALQuadrature=p.quadrature)::Float64 where {TS,TC1,TC2}
    if p.isCall #use McDonald and Schroder symmetry
        K, S = S, K
    end
    capX = p.capX * K
    f0 = invtransform(p.bTransform, 0.0) * capX
# f0 = capX
    if S < f0
        return max(K - S, 0.0)
    end

    tauMax = p.tauMax
    model = p.model
    transform = p.bTransform
    dividends = p.dividends
    sum4k = 0.0

    updateInterval(quadraturePrice, 0.0, tauMax)
    qAbscissae, qWeights = abscissaeAndWeights(quadraturePrice)
    qValues = similar(qAbscissae)
    evaluate!(qValues, p.collocation, qAbscissae)
    
    for (iu, u) = enumerate(qAbscissae)        
        Bu = max(K* invtransform(transform, qValues[iu]) * p.capX,1e-16)
        # Bu = max(K*qValues[iu]* p.capX,1e-16)
        sqrtv = sqrt((varianceByLogmoneyness(model, 0.0, u) * u))
        if sqrtv == 0.0
            #    return (0.0,0.0)
        else
            fu = forward(model, 1.0, u)
            fwd = fu*S
            dfu = discountFactor(model, u)
            qFrac = fwd * dfu
            rRate = -(log(discountFactor(model, u + 1e-7)) - log(dfu)) / (1e-7) #log(discountFactor(model, (tauHat - taui) + tauk)) / ((tauHat - taui) + tauk)
            qRate = rRate - (log(forward(model, 1.0, u + 1e-7)) - log(fu)) / 1e-7
            gk = dividendGrowthFactor(dividends, 0.0, u)
            d1 = (log(fwd * gk/Bu)) / sqrtv + sqrtv / 2
            d2 = d1 - sqrtv
            k1 = qFrac * qRate *gk* normcdf(-d1)
            k2 = dfu * K * rRate * normcdf(-d2)
            sum4k += qWeights[iu] * (k2-k1)
        end
    end
    # sumgi = 0.0
    # previousDate = 0.0
    # previousG = dividendGrowthFactor(dividends, previousDate, tauMax)
    # for dividend in dividends
    #     if dividend.exDate > previousDate
    #         currentG = dividendGrowthFactor(dividends, previousDate, dividend.exDate)
    #         sumgi += (currentG - previousG) #* forward(model, 1.0, dividend.exDate) / forward(model, 1.0, previousDate) / discountFactor(model, previousDate) * discountFactor(model, dividend.exDate)
    #         previousG = currentG
    #     end
    # end
    # sum4k -= sumgi
    euro = blackScholesFormula(
        false,
        K,
        forward(p.model, S, tauMax) * dividendGrowthFactor(dividends, 0.0, tauMax),
        varianceByLogmoneyness(p.model, 0.0, tauMax) * tauMax,
        1.0,
        discountFactor(p.model, tauMax)
    )
    #  println("euro ", euro, " ", sum4k)
    price = euro + sum4k
    price = max(K - S, price)
    return price
end
