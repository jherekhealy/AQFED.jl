using AQFED.TermStructure
import AQFED.Math: normcdf, normpdf, lambertW
import AQFED.Black: blackScholesFormula
import AQFED.Math: norminv
import Roots: find_zero, Newton, A42
using ForwardDiff

export AndersenLakeRepresentation, priceAmerican, americanBoundaryPutQDP

#@inline normcdf(z::Float64) = normcdfCody(z) #Faster apparently

struct AndersenLakeRepresentation{TM}
    isCall::Bool
    model::TM
    tauMax::Float64
    tauHat::Float64
    nC::Int
    nTS1::Int
    nTS2::Int
    capX::Float64
    avec::Vector{Float64}
    qvec::Vector{Float64}
    wvec::Vector{Float64}
    yvec::Vector{Float64}
end


function AndersenLakeRepresentation(
    model::ConstantBlackModel,
    tauMax::Float64,
    atol::Float64,
    nC::Int,
    nIter::Int,
    nTS1::Int,
    nTS2::Int;
    isCall::Bool=false,
    isLower::Bool=false
)

    if iseven(nTS1)
        throw(DomainError(string("nTS1 must be odd but was ", nTS1)))
    end
    if iseven(nTS2)
        throw(DomainError(string("nTS2 must be odd but was ", nTS2)))
    end
    K = 1.0

    avec = zeros(nC + 1)
    qvec = zeros(nC + 1)
    wvec = zeros(nTS1)
    yvec = zeros(nTS1)
    ndiv2 = trunc(Int, (nTS1 + 1) / 2)
    hn = tanhsinhStep(nTS1)
    hvec = hn .* (ndiv2-1:-1:1)
    svec = @. pi * sinh(hvec) / 2
    @. @view(yvec[1:ndiv2-1]) = tanh(svec) #may be more precise to store y+1 directly instead
    @. @view(wvec[1:ndiv2-1]) = hn * pi * cosh(hvec) / (2 * (cosh(svec))^2)
    for i = 1:ndiv2-1
        yvec[nTS1+1-i] = -yvec[i]
        wvec[nTS1+1-i] = wvec[i]
    end
    yvec[ndiv2] = 0
    wvec[ndiv2] = pi * hn / 2
    local r::Float64 = model.r
    local q::Float64 = model.q
    vol = model.vol
    capX = K
    if isLower 
        capX = K * r / q
    end
    modelB = model
    if isCall  #use McDonald and Schroder symmetry        
        modelB = ConstantBlackModel(vol, q, r)
        r, q = q, r
    end
    if q > r
        capX = K * r / q #FIXME why do we redefine here, is this correct for LB?
    end
    logCapX = log(capX)
    tauHat = tauMax
    if r < 0 && q < r && vol >= sqrt(-2 * q) - sqrt(-2 * r)
        #double boundary which intersect before infinite time
        objHat = function (τ)
            t = τ
            value = abs(norminv(-expm1(q * t)) - norminv(-expm1(r * t))) / sqrt(t) - vol
            # println(τ, " v ", value)
            return value
        end
        if objHat(tauMax) < 0 #
            # derHat =  x -> ForwardDiff.derivative(objHat,float(x))
            #  tauHat = (find_zero((objHat,derHat), sqrt(tauMax), Newton()))^2
            tauHat = find_zero(objHat, (1e-7, tauMax), A42())
            #    println("tauHat ", tauHat)
            tauHat = min(tauHat, tauMax)
        end
    end
    local fprev = capX
    qvec[nC+1] = 0
    for i = nC:-1:1
        zi = cos((i - 1) * pi / nC)
        taui = tauHat / 4 * (1 + zi)^2
        fi = americanBoundaryPutQDP(isLower, modelB, fprev, K, taui, atol)
        fprev = fi
        qvec[i] = max(qvec[i+1], (log(fi / capX))^2)
    end
    #  println("init-qvec ", sqrt.(qvec))
    d2Vector = zeros(nTS1)
    d1Vector = zeros(nTS1)
    k1 = zeros(nTS1)
    k2 = zeros(nTS1)
    tauVector = zeros(nTS1)
    for j = 1:nIter
        updateAvec!(avec, nC, qvec)
        # println("avec", avec, " q ",sqrt.(qvec))
        qvec[nC+1] = 0

        for i = nC:-1:1
            zi = cos((i - 1) * pi / nC)
            taui = tauHat / 4 * (1 + zi)^2
            Kstari = K * exp(-(r - q) * taui)
            lnBtaui = isLower ? logCapX + sqrt(qvec[i]) : logCapX - sqrt(qvec[i])
            sum1k = 0.0
            sum2k = 0.0
            @. tauVector = taui / 4 * (1 + yvec)^2
            @inbounds for sk1 = 1:nTS1
                if yvec[sk1] != -1
                    tauk = tauVector[sk1]
                    zck = 2 * sqrt((taui - tauk) / tauHat) - 1
                    qck = max(chebQck(avec, zck),0.0)
                    lnBtauk = isLower ? logCapX + sqrt(qck) : logCapX - sqrt(qck)
                    sqrtv = sqrt(tauk) * vol

                    # println("sqrtv", sqrtv, " ", lnBtaui, " ", lnBtauk, " ", (r - q) * tauk)

                    d1Vector[sk1] =
                        ((lnBtaui - lnBtauk) + (r - q) * tauk) / sqrtv + sqrtv / 2
                    d2Vector[sk1] = d1Vector[sk1] - sqrtv
                end
            end
            @. k1 = wvec * exp(-q * tauVector) * (yvec + 1) * normcdf(d1Vector)
            @. k2 = wvec * exp(-r * tauVector) * (yvec + 1) * normcdf(d2Vector)
            sum1k = exp(q * taui) / 2 * taui * sum(k1)
            sum2k = exp(r * taui) / 2 * taui * sum(k2)
            sqrtv = sqrt(taui) * vol
            d1i = ((lnBtaui - log(K)) + (r - q) * taui) / sqrtv + sqrtv / 2
            d2i = d1i - sqrtv

            Ni = r * sum2k
            Di = q * sum1k
            # println("d1vector ", d1Vector, " ", Ni, " ", Di)
            if isLower
                Ni = exp(r * taui) - 1 - Ni
                Di = exp(q * taui) - 1 - Di
            else
                Ni += normcdf(d2i)
                Di += normcdf(d1i)
            end
            NiOverDi = Ni / Di
            if Di == 0.0 && Ni == 0.0
                #use asymptotic expansion cdf = erfc(-x/sqrt2)/2 and erfc(x) = e^{-x^2}/(x*sqrtpi)*(1-1/(2*x^2))
                NiOverDi = exp(-(d2i^2 - d1i^2) / 2) * (d1i / d2i)
            end
            fi = Kstari * NiOverDi
            if fi <= 0
                # B = Kstar * N/D   to B = B + Kstar*N - B*D
                #lnBtaui = isLower ? logCapX + sqrt(qvec[i]) : logCapX - sqrt(qvec[i])
                Btaui = exp(lnBtaui)
                fi = Btaui + Kstari * Ni - Btaui * Di
            end
            lfc = log(fi / capX)
            if isnan(lfc)
                throw(DomainError(
                    fi,
                    string("Nan qvec ", capX, " ", lnBtaui, " ", qvec[i]),
                ))
            end
            qvec[i] = lfc^2
            qvec[i] = max(qvec[i+1], qvec[i])
            if !isLower && r < 0 && q < r
                qvec[i] = min(qvec[i], (log(K * (r / q) / capX))^2)
            end
        end
        # println("iteration ", j, " ", sqrt.(qvec))

    end
    if nTS2 != nTS1
        wvec = zeros(nTS2)
        yvec = zeros(nTS2)
        ndiv2 = trunc(Int, (nTS2 + 1) / 2)
        hn = tanhsinhStep(nTS2)
        hi = (ndiv2 - 1) * hn
        for i = 1:ndiv2-1
            p2s = pi * sinh(hi) / 2
            yvec[i] = tanh(p2s)
            p2c = pi * cosh(hi) / 2
            cp2s = cosh(p2s)
            wvec[i] = hn * p2c / (cp2s^2)
            hi = hi - hn
        end
        yvec[ndiv2] = 0
        wvec[ndiv2] = pi * hn / 2
        for i = 1:ndiv2-1
            yvec[nTS2+1-i] = -yvec[i]
            wvec[nTS2+1-i] = wvec[i]
        end
    end
    return AndersenLakeRepresentation(
        isCall,
        model,
        tauMax,
        tauHat,
        nC,
        nTS1,
        nTS2,
        capX,
        avec,
        qvec,
        wvec,
        yvec,
    )
end

#American put
function priceAmerican(p::AndersenLakeRepresentation{ConstantBlackModel}, K::Float64, S::Float64)::Float64
    vol = p.model.vol
    local r::Float64 = p.model.r
    local q::Float64 = p.model.q
    if p.isCall #use McDonald and Schroder symmetry
        K, S = S, K
        r, q = q, r
    end
    capX = p.capX * K
    lfc0 = -sqrt(p.qvec[1])
    f0 = exp(lfc0) * capX
    if S < f0
        return max(K - S, 0.0)
    end

    tauMax, nTS2 = p.tauMax, p.nTS2
    wvec, yvec, avec = p.wvec, p.yvec, p.avec
    nC, rK, qS = p.nC, r * K, q * S

    uMax = tauMax
    uMin = 0.0
    uScale = (uMax - uMin) / 2
    uShift = (uMax + uMin) / 2
    sum4k = 0.0
    for sk2 = 1:nTS2
        wk = wvec[sk2]
        yk = yvec[sk2]
        uk = uScale * yk + uShift
        if abs(yk) != 1
            zck = 2 * sqrt(uk / tauMax) - 1
            qck = max(0.0,chebQck(avec, zck))
            Bzk = capX * exp(-sqrt(qck))
            tauk = uMax - uk
            d1k, d2k = vaGBMd1d2(S, Bzk, r, q, tauk, vol)
            sum4k += wk * rK * exp(-r * tauk) * normcdf(-d2k)
            sum4k += -wk * qS * exp(-q * tauk) * normcdf(-d1k)
        end
    end

    euro = blackScholesFormula(
        false,
        K,
        S,
        vol * vol * tauMax,
        exp(-(r - q) * tauMax),
        exp(-r * tauMax),
    )
    price = euro + uScale * sum4k
    price = max(K - S, price)
    return price
end





function AndersenLakeRepresentation(
    model::TSBlackModel,
    tauMax::Float64,
    atol::Float64,
    nC::Int,
    nIter::Int,
    nTS1::Int,
    nTS2::Int;
    isCall::Bool=false,
    isLower::Bool=false
)

    if iseven(nTS1)
        throw(DomainError(string("nTS1 must be odd but was ", nTS1)))
    end
    if iseven(nTS2)
        throw(DomainError(string("nTS2 must be odd but was ", nTS2)))
    end
    K = 1.0

    avec = zeros(nC + 1)
    qvec = zeros(nC + 1)
    wvec = zeros(nTS1)
    yvec = zeros(nTS1)
    ndiv2 = trunc(Int, (nTS1 + 1) / 2)
    hn = tanhsinhStep(nTS1)
    hvec = hn .* (ndiv2-1:-1:1)
    svec = @. pi * sinh(hvec) / 2
    @. @view(yvec[1:ndiv2-1]) = tanh(svec) #may be more precise to store y+1 directly instead
    @. @view(wvec[1:ndiv2-1]) = hn * pi * cosh(hvec) / (2 * (cosh(svec))^2)
    for i = 1:ndiv2-1
        yvec[nTS1+1-i] = -yvec[i]
        wvec[nTS1+1-i] = wvec[i]
    end
    yvec[ndiv2] = 0
    wvec[ndiv2] = pi * hn / 2
    local r::Float64 = -logDiscountFactor(model, tauMax) / tauMax
    local q::Float64 = r - logForward(model, 0.0, tauMax) / tauMax
    vol = sqrt(varianceByLogmoneyness(model, 0.0, tauMax))
    capX = isLower ? K * r / q : K
    if isCall  #use McDonald and Schroder symmetry FIXME: change TS
        r, q = q, r
    end
    modelB = ConstantBlackModel(vol, r, q)

    if q > r
        capX = K * r / q #FIXME why do we redefine here, is this correct for LB?
    end
    logCapX = log(capX)
    tauHat = tauMax
    if r < 0 && q < r && vol >= sqrt(-2 * q) - sqrt(-2 * r)
        #double boundary which intersect before infinite time
        objHat = function (τ)
            t = τ
            value = abs(norminv(-expm1(q * t)) - norminv(-expm1(r * t))) / sqrt(t) - vol
            # println(τ, " v ", value)
            return value
        end
        if objHat(tauMax) < 0 #
            # derHat =  x -> ForwardDiff.derivative(objHat,float(x))
            #  tauHat = (find_zero((objHat,derHat), sqrt(tauMax), Newton()))^2
            tauHat = find_zero(objHat, (1e-7, tauMax), A42())
            #    println("tauHat ", tauHat)
            tauHat = min(tauHat, tauMax)
        end
    end
    local fprev = capX
    qvec[nC+1] = 0
    for i = nC:-1:1
        zi = cos((i - 1) * pi / nC)
        taui = tauHat / 4 * (1 + zi)^2
        fi = americanBoundaryPutQDP(isLower, modelB, fprev, K, taui, atol)
        fprev = fi
        logfX = log(fi / capX)
        qvec[i] = logfX # max(abs(qvec[i+1]), abs(logfX))*sign(logfX) # don't square as it might be neg and pos
    end
    # println("init-qvec ", qvec)
    d2Vector = zeros(nTS1)
    d1Vector = zeros(nTS1)
    k1 = zeros(nTS1)
    k2 = zeros(nTS1)
    tauVector = zeros(nTS1)
    qVector = zeros(nTS1)
    rVector = zeros(nTS1)
    #TODO cache variance, logforward,forward,df per time.
    #beyong nC=5 ot 7, may worth to use many "avec" PP. nPP = 1,...,10. to avoid oscillations. 10 times slower, much faster than nC=70 (which actually does not make much sense)
    for j = 1:nIter
        updateAvec!(avec, nC, qvec)
        # println("avec", avec, " q ",qvec)

        qvec[nC+1] = 0

        for i = nC:-1:1
            zi = cos((i - 1) * pi / nC)
            taui = tauHat / 4 * (1 + zi)^2 #taui = tau
            Kstari = K / forward(model, 1.0, tauHat) * forward(model, 1.0, tauHat - taui) #P(T-tau,T)/Q(T-tau,T)
            lnBtaui = logCapX + qvec[i]
            sum1k = 0.0
            sum2k = 0.0
            @. tauVector = taui / 4 * (1 + yvec)^2
            @inbounds for sk1 = 1:nTS1
                if yvec[sk1] != -1
                    tauk = tauVector[sk1] #tau-u         
                    zck = 2 * sqrt((taui - tauk) / tauHat) - 1
                    qck = chebQck(avec, zck)
                    lnBtauk = logCapX + qck
                    #from tauHat-taui to (tauHat-taui)+tauk
                    sqrtv = sqrt(max(varianceByLogmoneyness(model, 0.0, (tauHat - taui) + tauk) * ((tauHat - taui) + tauk) - varianceByLogmoneyness(model, 0.0, tauHat - taui) * (tauHat - taui), 0.0))
                    if sqrtv == 0.0
                        qVector[sk1] = 0.0
                        rVector[sk1] = 0.0
                        d1Vector[sk1] = 0.0
                        d2Vector[sk1] = 0.0
                    else
                        frac = forward(model, 1.0, (tauHat - taui) + tauk) / forward(model, 1.0, tauHat - taui)
                        rFrac = discountFactor(model, (tauHat - taui) + tauk) / discountFactor(model, tauHat)
                        rRate = -(log(discountFactor(model, (tauHat - taui) + tauk+1e-7)) - log(discountFactor(model, (tauHat - taui) + tauk)))/(1e-7) #log(discountFactor(model, (tauHat - taui) + tauk)) / ((tauHat - taui) + tauk)
                        qRate = -(log(discountFactor(model, (tauHat - taui) + tauk+1e-7) * forward(model, 1.0, (tauHat - taui) + tauk+1e-7))-log(discountFactor(model, (tauHat - taui) + tauk) * forward(model, 1.0, (tauHat - taui) + tauk))) / 1e-7
                        # objrRate = function(x) 
                        #     -log(discountFactor(model, x))
                        # end
                        # rRate = ForwardDiff.derivative( objrRate,  (tauHat - taui) + tauk)
                        
                        qVector[sk1] = forward(model, 1.0, (tauHat - taui) + tauk) / forward(model, 1.0, tauHat) * rFrac * qRate
                        rVector[sk1] = rFrac * rRate
                        # if (isnan(rVector[sk1]))
                        #     println("ALNaN ",rFrac, " ",tauHat, " ",((tauHat-taui)+tauk)," ",tauHat-taui, " ",tauk)
                        # end
                        # println("sqrtv", sqrtv, " ", lnBtaui, " ", qck, " ", log(frac))
                        d1Vector[sk1] =
                            ((lnBtaui - lnBtauk) + log(frac)) / sqrtv + sqrtv / 2
                        d2Vector[sk1] = d1Vector[sk1] - sqrtv
                    end
                end
            end
            @. k1 = wvec * qVector * (yvec + 1) * normcdf(d1Vector)
            @. k2 = wvec * rVector * (yvec + 1) * normcdf(d2Vector)
            sum1k = taui * sum(k1) / 2
            sum2k = taui * sum(k2) / 2
            #from tauhait-taui to tauhat
            sqrtv = sqrt(max(-varianceByLogmoneyness(model, 0.0, tauHat - taui) * (tauHat - taui) + varianceByLogmoneyness(model, 0.0, tauHat) * tauHat, 0.0))
            d1i = ((lnBtaui - log(K)) + logForward(model, 0.0, tauHat) - logForward(model, 0.0, tauHat - taui)) / sqrtv + sqrtv / 2
            d2i = d1i - sqrtv

            Ni = sum2k
            Di = sum1k
            # println("d1vector ", d1Vector, " ", Ni, " ", Di)
            if isLower
                Ni = discountFactor(model, tauHat - taui) / discountFactor(model, tauHat) - 1 - Ni
                Di = forward(model, 1.0, tauHat) * discountFactor(model, tauHat) / (forward(model, 1.0, tauHat - taui) * discountFactor(model, tauHat - taui)) - 1 - Di
            else
                Ni += normcdf(d2i)
                Di += normcdf(d1i)
            end
            NiOverDi = Ni / Di
            # println("AL ",Kstari," ",Ni, " ",Di, " ",rVector)
            if Di == 0.0 && Ni == 0.0
                #use asymptotic expansion cdf = erfc(-x/sqrt2)/2 and erfc(x) = e^{-x^2}/(x*sqrtpi)*(1-1/(2*x^2))
                NiOverDi = exp(-(d2i^2 - d1i^2) / 2) * (d1i / d2i)
            end
            fi = Kstari * NiOverDi
            if fi <= 0
                # B = Kstar * N/D   to B = B + Kstar*N - B*D
                #lnBtaui = isLower ? logCapX + sqrt(qvec[i]) : logCapX - sqrt(qvec[i])
                Btaui = exp(lnBtaui)
                fi = Btaui + Kstari * Ni - Btaui * Di
            end
            lfc = log(fi / capX)
            if isnan(lfc)
                throw(DomainError(
                    fi,
                    string("Nan qvec ", capX, " ", lnBtaui, " ", qvec[i]),
                ))
            end
            qvec[i] = lfc
            #qvec[i] = max(qvec[i+1], qvec[i])
            #if !isLower && r < 0 && q < r
            # qvec[i] = min(qvec[i], (log(K * (r / q) / capX))) # commented out with TS as we don't know for sure
            #end
        end

    end
    if nTS2 != nTS1
        wvec = zeros(nTS2)
        yvec = zeros(nTS2)
        ndiv2 = trunc(Int, (nTS2 + 1) / 2)
        hn = tanhsinhStep(nTS2)
        hi = (ndiv2 - 1) * hn
        for i = 1:ndiv2-1
            p2s = pi * sinh(hi) / 2
            yvec[i] = tanh(p2s)
            p2c = pi * cosh(hi) / 2
            cp2s = cosh(p2s)
            wvec[i] = hn * p2c / (cp2s^2)
            hi = hi - hn
        end
        yvec[ndiv2] = 0
        wvec[ndiv2] = pi * hn / 2
        for i = 1:ndiv2-1
            yvec[nTS2+1-i] = -yvec[i]
            wvec[nTS2+1-i] = wvec[i]
        end
    end
    return AndersenLakeRepresentation(
        isCall,
        model,
        tauMax,
        tauHat,
        nC,
        nTS1,
        nTS2,
        capX,
        avec,
        qvec,
        wvec,
        yvec,
    )
end


function exerciseBoundary(p::AndersenLakeRepresentation{TSBlackModel{TS,TC1,TC2}}, K::Float64, t::AbstractArray{Float64}) where {TS,TC1,TC2}
    capX = p.capX * K
    tauMax, nTS2 = p.tauMax, p.nTS2
    wvec, yvec, avec = p.wvec, p.yvec, p.avec
    nC = p.nC
    zck = @. 2 * sqrt(1 - t / tauMax) - 1
    Bzk = zeros(Float64,Base.length(t))
    for i = eachindex(zck)
        qck = chebQck(avec, zck[i])
        Bzk[i] = capX * exp(qck)
    end
    return Bzk
end

function priceAmerican(p::AndersenLakeRepresentation{TSBlackModel{TS,TC1,TC2}}, K::Float64, S::Float64)::Float64 where {TS,TC1,TC2}
    if p.isCall #use McDonald and Schroder symmetry #FIXME TS
        K, S = S, K
    end
    capX = p.capX * K
    lfc0 = p.qvec[1]
    f0 = exp(lfc0) * capX

    if S < f0
        return max(K - S, 0.0)
    end

    tauMax, nTS2 = p.tauMax, p.nTS2
    wvec, yvec, avec = p.wvec, p.yvec, p.avec
    nC = p.nC

    uMax = tauMax
    uMin = 0.0
    uScale = (uMax - uMin) / 2
    uShift = (uMax + uMin) / 2
    sum4k = 0.0
    for sk2 = 1:nTS2
        wk = wvec[sk2]
        yk = yvec[sk2]
        uk = uScale * yk + uShift
        if abs(yk) != 1
            zck = 2 * sqrt(uk / tauMax) - 1
            qck = chebQck(avec, zck)
            Bzk = capX * exp(qck)
            tauk = uMax - uk #T-u
            vol = sqrt(varianceByLogmoneyness(p.model, 0.0, tauk))
            d1k, d2k = vaGBMd1d2(forward(p.model, S, tauk), Bzk, 0.0, 0.0, tauk, vol)
            rRate = -(log(discountFactor(p.model, tauk+1e-6))-log(discountFactor(p.model, tauk))) / 1e-6
            qRate = -(log(forward(p.model, 1.0, tauk+1e-6) * discountFactor(p.model, tauk+1e-6)) - log(forward(p.model, 1.0, tauk) * discountFactor(p.model, tauk))) / 1e-6
            sum4k += wk * K * rRate * discountFactor(p.model, tauk) * normcdf(-d2k)
            sum4k += -wk * qRate * forward(p.model, S, tauk) * discountFactor(p.model, tauk) * normcdf(-d1k)
        end
    end

    euro = blackScholesFormula(
        false,
        K,
        forward(p.model, S, tauMax),
        varianceByLogmoneyness(p.model, 0.0, tauMax) * tauMax,
        1.0,
        discountFactor(p.model, tauMax)
    )
    # println("euro ", euro, " ", sum4k, " ", uScale * sum4k)
    price = euro + uScale * sum4k
    price = max(K - S, price)
    return price
end


@inline function chebQck(avec::AbstractArray{Float64}, zck::Float64)
    b2 = 0.0
    nC = Base.length(avec) - 1
    b1 = avec[nC+1] / 2
    @inbounds @fastmath for sk22 = nC:-1:2
        bd = avec[sk22] - b2
        b2 = b1
        b1 = 2 * zck * b1 + bd
    end
    b0 = avec[1] + 2 * zck * b1 - b2
    qck = (b0 - b2) / 2
    qck
end

function americanBoundaryPutQDP(
    isLower::Bool,
    model::ConstantBlackModel,
    Szero::Float64,
    K::Float64,
    tauMax::Float64,
    atol::Float64,
)::Float64
    r = model.r
    q = model.q
    vol = model.vol
    if r == 0
        r = 1e-5
    end
    lowerSign = 1.0
    if isLower
        lowerSign = -1.0
    end
    eqT = exp(-q * tauMax)
    erT = exp(-r * tauMax)
    hT = 1 - erT
    vol2 = vol * vol
    capM = 2 * r / vol2
    capN = 2 * (r - q) / vol2 - 1
    SqrV = vol * sqrt(tauMax)
    q1 = sqrt(capN^2 + 4 * capM / hT)
    qQD = -0.5 * (capN + lowerSign * q1)
    q2 = 2 * qQD + capN
    qQD2 = lowerSign * capM / (hT * hT * q1)
    t1 = r * K * erT
    t2 = q * eqT
    t3 = eqT * vol / sqrt(tauMax)
    Sstar = Szero
    #println("Sstar init ",Sstar)
    fS = atol
    iter = 0
    obj = function (Sstar::Float64)
        d1, d2 = vaGBMd1d2(Sstar, K, r, q, tauMax, vol)
        Nd1 = normcdf(-d1)
        snd1 = normpdf(d1)
        Nd2 = normcdf(-d2)
        snd2 = normpdf(d2)
        d1dS = 1.0 / (Sstar * SqrV)
        Nd1dS = -snd1 * d1dS
        snd1dS = -d1 * snd1 * d1dS
        d1d2S = -d1dS / Sstar
        Nd1d2S = -snd1 * d1d2S - snd1dS * d1dS
        snd1d2S = -d1 * snd1 * d1d2S - (d1 * snd1dS + snd1 * d1dS) * d1dS
        d2dS = 1.0 / (Sstar * SqrV)
        Nd2dS = -snd2 * d2dS
        snd2dS = -d2 * snd2 * d2dS
        d2d2S = -d2dS / Sstar
        Nd2d2S = -snd2 * d2d2S - snd2dS * d2dS
        theta = t1 * Nd2 - t2 * Sstar * Nd1 - 0.5 * t3 * Sstar * snd1
        thetadS =
            t1 * Nd2dS - t2 * (Sstar * Nd1dS + Nd1) - 0.5 * t3 * (Sstar * snd1dS + snd1)
        thetad2S =
            t1 * Nd2d2S - t2 * (Sstar * Nd1d2S + 2 * Nd1dS) -
            0.5 * t3 * (Sstar * snd1d2S + 2 * snd1dS)
        vp = K - Sstar - erT * K * Nd2 + eqT * Sstar * Nd1
        vpdS = -1 - erT * K * Nd2dS + eqT * (Sstar * Nd1dS + Nd1)
        vpd2S = -erT * K * Nd2d2S + eqT * (Sstar * Nd1d2S + 2 * Nd1dS)
        tdvp = theta / vp
        tdvpdS = (thetadS * vp - theta * vpdS) / (vp * vp)
        tdvpd2S = (thetad2S - 2 * tdvpdS * vpdS - tdvp * vpd2S) / vp
        c0 = 0.0
        c0dS = 0.0
        c0d2S = 0.0
        if vp != 0
            c0m = -(1 - hT) * capM / q2
            c0 = c0m * (1 / hT + qQD2 / q2 - tdvp / (erT * r))
            c0dS = -c0m * tdvpdS / (erT * r)
            c0d2S = -c0m * tdvpd2S / (erT * r)
        end
        fS = Sstar - eqT * Sstar * Nd1 + qQD * vp + c0 * vp
        fSdS = 1 - eqT * (Sstar * Nd1dS + Nd1) + qQD * vpdS + c0 * vpdS + c0dS * vp
        fSd2S = -eqT * (Sstar * Nd1d2S + 2 * Nd1dS) + qQD * vpd2S
        fSd2S = fSd2S + c0 * vpd2S + 2 * c0dS * vpdS + c0d2S * vp
        # if isnan(fS)
        #     println(Sstar," ",d1," ",d2, " ",qQD, " ",Nd1, " ",vp)
        # end
        return fS, fSdS, fSd2S
    end
    local fSdS
    local fSd2S
    while abs(fS) >= atol && iter < 128
        fS, fSdS, fSd2S = obj(Sstar)
        hn = -fS / fSdS
        halleyTerm = -fSd2S / fSdS * hn
        local Sstarn
        # switch solverType {
        # case Halley:
        # Sstarn = Sstar + hn/(1-0.5*halleyTerm)
        # case InverseQuadratic:
        # Sstarn = Sstar + hn * (1 + 0.5 * halleyTerm)
        # case CMethod:
        # Sstarn = Sstar + hn*(1+0.5*halleyTerm+0.5*halleyTerm^2)
        # case SuperHalley:
        Sstarn = Sstar + hn * (1 + 0.5 * halleyTerm / (1 - halleyTerm))
        # }
        if Sstarn < 0.0
            Sstarn = -Sstarn
        end
        iter += 1
        Sstar = Sstarn
        if isnan(Sstar) || Sstar == 0 || isinf(Sstar)
            println(iter, "Sstar=", Sstar, fS, fSdS, fSd2S)
            return 0.0
        end
    end
    #printf("%d %v %f %f %f\n", iter, solverType, Szero, Sstar, fS)

    if r < 0.0 && q < r
        sigmaStar = sqrt(-2q) - sqrt(-2r)
        if vol < sigmaStar
            mu = q - r - vol2 / 2
            if isLower
                lambda = (-mu + sqrt(mu^2 + 2 * q * vol2)) / vol2
                Sstar = min(K / lambda * (lambda - 1), Sstar)
            else
                lambda = (-mu - sqrt(mu^2 + 2 * q * vol2)) / vol2
                Sstar = max(K / lambda * (lambda - 1), Sstar)
            end
        else
            # Sstar = min(K,max(K * sqrt(r / q), Sstar))
            Sstar = min(K, max(K * (r / q), Sstar))
        end
    end
    #println("Sstar", Sstar, iter)
    return Sstar
end


function tanhsinhStep(nTS2::Int)::Float64
    n = trunc(Int, (nTS2 + 1) / 2)
    a = pi / 2
    return lambertW(4 * a * n) / n
end



@inline function vaGBMd1d2(Btaui::T, Btauk::T, r::T, q::T, tauk::T, vol::T) where {T}
    sqrtv = sqrt(tauk) * vol
    d1 = (log(Btaui / Btauk) + (r - q) * tauk) / sqrtv + sqrtv / 2
    d2 = d1 - sqrtv
    return d1, d2
end

function updateAvec!(avec::AbstractArray{Float64}, nC::Int, qvec::AbstractArray{Float64})
    for sk = 0:nC
        @inbounds sumi = qvec[1] / 2
        @inbounds @simd for i = 2:nC
            sumi += qvec[i] * cos((i - 1) * sk * pi / nC)
        end
        @inbounds sumi += qvec[nC+1] / 2 * cos(sk * pi)
        avec[sk+1] = 2 * sumi / nC
    end
end
