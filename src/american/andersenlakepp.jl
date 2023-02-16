using AQFED.TermStructure
import AQFED.Math: normcdf, normpdf, lambertW
import AQFED.Black: blackScholesFormula
import AQFED.Math: norminv
import Roots: find_zero, Newton, A42
using ForwardDiff

export AndersenLakePPRepresentation, priceAmerican
#@inline normcdf(z::Float64) = normcdfCody(z) #Faster apparently

struct AndersenLakePPRepresentation{TM}
    isCall::Bool
    model::TM
    dividends::Vector{Dividend{Float64}}
    tauMax::Float64
    tauHat::Float64
    nPP::Int
    taus::Vector{Float64}
    nC::Int
    nTS1::Int
    nTS2::Int
    capX::Float64
    avec::Matrix{Float64}
    qvec::Matrix{Float64}
    wvec::Vector{Float64}
    yvec::Vector{Float64}
end

function AndersenLakePPRepresentation(
    model::TSBlackModel,
    tauMax::Float64,
    isCall::Bool=false;
    atol::Float64=1e-8,
    nPP::Int=1,
    nC::Int=5,
    nIter::Int=8,
    nTS1::Int=21,
    nTS2::Int=121,
    isLower::Bool=false,
    dividends::AbstractArray{Dividend{Float64}}=Dividend{Float64}[]
)

    if iseven(nTS1)
        throw(DomainError(string("nTS1 must be odd but was ", nTS1)))
    end
    if iseven(nTS2)
        throw(DomainError(string("nTS2 must be odd but was ", nTS2)))
    end
    K = 1.0
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
    avec = zeros(Float64, (nC + 1, nPP))
    qvec = zeros(Float64, (nC + 1, nPP))
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
    if isCall  #use McDonald and Schroder symmetry
        model = AQFED.TermStructure.TSBlackModel(model.surface, SpreadCurve(model.discountCurve, model.driftCurve), model.discountCurve)
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
    local fprev = capX
    append!(taus, tauHat)
    r = -logDiscountFactor(model, tauHat) / tauHat
    q = r - logForward(model, 0.0, tauHat) / tauHat
    modelB = ConstantBlackModel(vol, r, q)
    #println("taus ",taus)
    isDiscontinuous = !isempty(dividends)
    for iPP = nPP:-1:1
        if iPP == nPP
            qvec[nC+1, iPP] = isDiscontinuous ? 1.0 : 0.0
        else
            qvec[nC+1, iPP] = qvec[1, iPP+1]
        end
        for i = nC:-1:1
            zi = cos((i - 1) * pi / nC)
            taui = tauHat - taus[iPP+1] + (taus[iPP+1] - taus[iPP]) / 4 * (1 + zi)^2

            fi = americanBoundaryPutQDP(isLower, modelB, fprev, K, taui, atol)
            fprev = fi
            logfX = isDiscontinuous ? fi / capX : log(fi / capX)
            qvec[i, iPP] = logfX # max(abs(qvec[i+1]), abs(logfX))*sign(logfX) # don't square as it might be neg and pos
        end
    end
    # println("init-qvec ", qvec')
    d2Vector = zeros(nTS1)
    d1Vector = zeros(nTS1)
    k1 = zeros(nTS1)
    k2 = zeros(nTS1)
    tauVector = zeros(nTS1)
    qVector = zeros(nTS1)
    rVector = zeros(nTS1)
    #TODO cache variance, logforward,forward,df per time.
    #beyong nC=5 ot 7, may worth to use many "avec" PP. nPP = 1,...,10. to avoid oscillations. 10 times slower, much faster than nC=70 (which actually does not make much sense)
    if nIter == 0
        for iPP = nPP:-1:1
            updateAvec!(@view(avec[:, iPP]), nC, @view(qvec[:, iPP]))
        end
    end
    ϵ = 0.0
    if isDiscontinuous
        ϵ = 1e-7
    end
    Bmin = 1e-16
    for j = 1:nIter
        for iPP = nPP:-1:1
            isOnExDate = false
            for dividend in dividends
                # println("dividend ",dividend.exDate," ",taus[iPP+1])
                if dividend.exDate == taus[iPP+1]
                    isOnExDate = true
                    break
                end
            end
            if iPP < nPP
                if isOnExDate
                    # println(j, "adjusting ",iPP)
                    qvec[nC+1, iPP] = Bmin
                else
                    qvec[nC+1, iPP] = qvec[1, iPP+1]
                end
            end
        end
        # if isDiscontinuous
        #      for iPP = nPP-1:-1:1
        #          qvec[nC+1, iPP] = Bmin
        #      end
        # else
        #     for iPP = nPP-1:-1:1
        #         qvec[nC+1, iPP] = qvec[1, iPP+1]
        #     end
        # end
        for iPP = nPP:-1:1
            updateAvec!(@view(avec[:, iPP]), nC, @view(qvec[:, iPP]))
        end
        # println("avec", avec, " q ",qvec)

        qvec[nC+1, nPP] = isDiscontinuous ? 1.0 : 0.0

        for iPP = nPP:-1:1

            iMax = nC
            if iPP < nPP
                iMax = nC
            end

            for i = iMax:-1:1
                zi = cos((i - 1) * pi / nC)
                taui = tauHat - taus[iPP+1] + ϵ + (taus[iPP+1] - taus[iPP] - 2ϵ) / 4 * (1 + zi)^2 #taui = tau
                Kstari = K / forward(model, 1.0, tauHat) * forward(model, 1.0, tauHat - taui) #P(T-tau,T)/Q(T-tau,T)
                lnBtaui = logCapX + (isDiscontinuous ? log(qvec[i, iPP]) : qvec[i, iPP])
                sum1k = 0.0
                sum2k = 0.0
                for iPPk = nPP:-1:iPP
                    taukEnd = taui
                    if iPPk == iPP
                        @. tauVector = (taui - (tauHat - taus[iPP+1] - 2ϵ)) / 4 * (1 + yvec)^2
                    else
                        taukEnd = tauHat - taus[iPPk] - ϵ
                        @. tauVector = (taus[iPPk+1] - taus[iPPk] - 2ϵ) / 4 * (1 + yvec)^2
                    end
                    @inbounds for sk1 = 1:nTS1
                        tauk = tauHat - taus[iPPk+1] + ϵ + tauVector[sk1] #tau-u    
                        taukCompl = taus[iPPk+1] + 2ϵ - taukEnd + tauk
                        zck = 2 * sqrt(max((taukEnd - tauk) / (taus[iPPk+1] - taus[iPPk]), 0.0)) - 1.0
                        #iPPk = index of tauk in taus
                        qck = chebQck(@view(avec[:, iPPk]), zck)
                        lnBtauk = logCapX + (isDiscontinuous ? log(max(qck, Bmin)) : qck)
                        #  println("nIter ",j," iPPk ",iPPk," iPP ",iPP," ",sk1," tauk ",tauk, " taui ",taui, " ",taukCompl," zk^2+1=",(taukEnd - tauk) / (taus[iPPk+1] - taus[iPPk])," lnB ",lnBtauk, " ",lnBtaui)    
                        #from tauHat-taui to (tauHat-taui)+tauk
                        sqrtv = sqrt(max(varianceByLogmoneyness(model, 0.0, taukCompl) * (taukCompl) - varianceByLogmoneyness(model, 0.0, tauHat - taui) * (tauHat - taui), 0.0))
                        if sqrtv == 0.0
                            qVector[sk1] = 0.0
                            rVector[sk1] = 0.0
                            d1Vector[sk1] = 0.0
                            d2Vector[sk1] = 0.0
                        else
                            frac = forward(model, 1.0, taukCompl) / forward(model, 1.0, tauHat - taui)
                            rFrac = discountFactor(model, taukCompl) / discountFactor(model, tauHat)
                            rRate = -(log(discountFactor(model, taukCompl + 1e-7)) - log(discountFactor(model, taukCompl))) / (1e-7) #log(discountFactor(model, (tauHat - taui) + tauk)) / ((tauHat - taui) + tauk)
                            qRate = rRate-(log(forward(model, 1.0, taukCompl + 1e-7))-log(forward(model, 1.0, taukCompl))) / 1e-7
                            # objrRate = function(x) 
                            #     -log(discountFactor(model, x))
                            # end
                            # rRate = ForwardDiff.derivative( objrRate,  (tauHat - taui) + tauk)
                            gk = dividendGrowthFactor(dividends, tauHat - taui, taukCompl)
                            qVector[sk1] = forward(model, 1.0, taukCompl) / forward(model, 1.0, tauHat) * rFrac * qRate * gk
                            rVector[sk1] = rFrac * rRate ###FIXME strange that rvector does not follow frac, likely bc of Kstar factors but still
                                                        # if (isnan(rVector[sk1]))
                            #  println("AL ",rFrac, " ",tauHat, " ",taukCompl, " ",tauk, " ",rRate," ",qRate)
                            # end
                            #  println("sqrtv", sqrtv, " ", lnBtaui, " ", qck, " ", log(frac))

                            d1Vector[sk1] =
                                ((lnBtaui - lnBtauk) + log(frac * gk)) / sqrtv + sqrtv / 2
                            d2Vector[sk1] = d1Vector[sk1] - sqrtv
                        end
                    end
                    # println(j, " ",iPP, " ", iPPk," ",d2Vector)
                    @. k1 = wvec * qVector * (yvec + 1) * normcdf(d1Vector)
                    @. k2 = wvec * rVector * (yvec + 1) * normcdf(d2Vector)
                    sum1k += (taukEnd - (tauHat - taus[iPPk+1])) * sum(k1) / 2
                    sum2k += (taukEnd - (tauHat - taus[iPPk+1])) * sum(k2) / 2
                end
                #from tauhait-taui to tauhat
                sqrtv = sqrt(max(-varianceByLogmoneyness(model, 0.0, tauHat - taui) * (tauHat - taui) + varianceByLogmoneyness(model, 0.0, tauHat) * tauHat, 0.0))
                d1i = ((lnBtaui - log(K)) + logForward(model, 0.0, tauHat) - logForward(model, 0.0, tauHat - taui)) / sqrtv + sqrtv / 2
                gi = dividendGrowthFactor(dividends, tauHat - taui, tauHat)
                sumgi = 0.0
                previousG = 1.0
                previousDate = tauHat - taui
                for dividend in dividends
                    if dividend.exDate > previousDate
                        currentG = dividendGrowthFactor(dividends, previousDate, dividend.exDate)
                        sumgi += (currentG - previousG) * forward(model, 1.0, dividend.exDate) / forward(model, 1.0, previousDate) / discountFactor(model, previousDate) * discountFactor(model, dividend.exDate)
                        previousG = currentG
                    end
                end
                d1i += log(gi)
                d2i = d1i - sqrtv
                # println("iPP ",iPP, " taui ", taui, " B ", lnBtaui, " drift ", logForward(model, 0.0, tauHat) - logForward(model, 0.0, tauHat - taui))
                # println("iPP ", iPP, " taui ", taui, " ", logSum)
                Ni = sum2k
                Di = sum1k
                if isLower
                    Ni = discountFactor(model, tauHat - taui) / discountFactor(model, tauHat) - 1 - Ni
                    Di = forward(model, 1.0, tauHat) * discountFactor(model, tauHat) / (forward(model, 1.0, tauHat - taui) * discountFactor(model, tauHat - taui)) - 1 - Di
                else
                    Ni += normcdf(d2i)
                    Di += normcdf(d1i) * gi - sumgi
                end
                NiOverDi = Ni / Di
                # println("AL ", Kstari, " ", Ni, " ", Di, " ", rVector)
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
                lfc = isDiscontinuous ? fi / capX : log(fi / capX)
                if isnan(lfc)
                    throw(DomainError(
                        fi,
                        string("Nan qvec ", capX, " ", lnBtaui, " ", qvec[i, iPP], " at iPP=", iPP, " ", i),
                    ))
                end
                qvec[i, iPP] = lfc
                #qvec[i] = max(qvec[i+1], qvec[i])
                #if !isLower && r < 0 && q < r
                # qvec[i] = min(qvec[i], (log(K * (r / q) / capX))) # commented out with TS as we don't know for sure
                #end
            end
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
    return AndersenLakePPRepresentation(
        isCall,
        model, dividends,
        tauMax,
        tauHat,
        nPP, taus,
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

function dividendGrowthFactor(dividends, t, u) #t excluded, u included
    factor = 1.0
    for dividend in dividends
        if dividend.exDate > t + eps(Float64) && dividend.exDate <= u + eps(Float64)
            factor *= (1 - dividend.amount)
        end
    end
    return factor
end

function exerciseBoundary(p::AndersenLakePPRepresentation{TSBlackModel{TS,TC1,TC2}}, K::Float64, t::AbstractArray{Float64}) where {TS,TC1,TC2}
    tauMax, nTS2 = p.tauMax, p.nTS2
    wvec, yvec, avec = p.wvec, p.yvec, p.avec
    nC = p.nC
    Bzk = zeros(Float64, length(t))
    iPP = 1
    for i = eachindex(t)
        if p.taus[iPP+1] < t[i]
            iPP += 1
        end
        zck = 2 * sqrt(-(t[i] - p.taus[iPP+1]) / (p.taus[iPP+1] - p.taus[iPP])) - 1
        qck = chebQck(@view(avec[:, iPP]), zck)
        Bzk[i] = isempty(p.dividends) ? p.capX * K * exp(qck) : p.capX * K * qck
    end
    return Bzk
end

function boundaryTimes(p::AndersenLakePPRepresentation{TSBlackModel{TS,TC1,TC2}}) where {TS,TC1,TC2}
    t = Vector{Float64}()
    for iPP=1:p.nPP
    for i=1:p.nC
         zi = cos((i - 1) * pi / p.nC)
        taui = p.tauHat - p.taus[iPP+1] +(p.taus[iPP+1] - p.taus[iPP] ) / 4 * (1 + zi)^2 #taui = tau
        append!(t,p.tauHat-taui)
    end
end
   return t
end
function priceAmerican(p::AndersenLakePPRepresentation{TSBlackModel{TS,TC1,TC2}}, K::Float64, S::Float64)::Float64 where {TS,TC1,TC2}
    if p.isCall #use McDonald and Schroder symmetry
        K, S = S, K
    end
    capX = p.capX * K
    f0 = isempty(p.dividends) ? exp(p.qvec[1]) * capX : p.capX * K * max(p.qvec[1], 1e-16)

    if S < f0
        return max(K - S, 0.0)
    end

    tauMax, nTS2 = p.tauMax, p.nTS2
    wvec, yvec, avec = p.wvec, p.yvec, p.avec
    nC = p.nC
    dividends = p.dividends
    sum4k = 0.0
    for iPP = 1:p.nPP
        uMax = p.taus[iPP+1]
        uMin = p.taus[iPP]
        uScale = (uMax - uMin) / 2
        uShift = (uMax + uMin) / 2
        for sk2 = 1:nTS2
            wk = wvec[sk2]
            yk = yvec[sk2]
            uk = uScale * yk + uShift
            zck = 2 * sqrt(max((uk - p.taus[iPP]) / (p.taus[iPP+1] - p.taus[iPP]), 0)) - 1
            qck = chebQck(@view(avec[:, iPP]), zck)
            Bzk = isempty(p.dividends) ? p.capX * K * exp(qck) : p.capX * K * max(qck, 1e-16)
            tauk = p.taus[iPP+1] - uk + p.taus[iPP]  #T-u
            # println("u ", uk, " ", zck, " t ", tauk, " ", Bzk)
            if tauk != 0
                vol = sqrt(varianceByLogmoneyness(p.model, 0.0, tauk))

                fg = forward(p.model, S, tauk) * dividendGrowthFactor(dividends, 0.0, tauk)
                d1k, d2k = vaGBMd1d2(fg, Bzk, 0.0, 0.0, tauk, vol)
                rRate = -(log(discountFactor(p.model, tauk + 1e-6)) - log(discountFactor(p.model, tauk))) / 1e-6
                qRate = rRate - (log(forward(p.model, 1.0, tauk + 1e-6)) - log(forward(p.model, 1.0, tauk))) / 1e-6
                sum4k += uScale * wk * K * rRate * discountFactor(p.model, tauk) * normcdf(-d2k)
                sum4k += -uScale * wk * qRate * fg * discountFactor(p.model, tauk) * normcdf(-d1k)
            end
        end
        # for sk2 = 1:nTS2
        #     wk = wvec[sk2]
        #     yk = yvec[sk2]
        #     uk = uScale * yk + uShift
        #     zck = 2 * sqrt(max((uk - p.taus[iPP]) / (p.taus[iPP+1] - p.taus[iPP]), 0)) - 1
        #     qck = chebQck(@view(avec[:, iPP]), zck)
        #     Bzk = isempty(p.dividends) ? p.capX * K * exp(qck) : p.capX * K * max(qck, 1e-16)
        #     tauk = p.taus[iPP+1] - uk + p.taus[iPP]  #T-u
        #     # println("u ", uk, " ", zck, " t ", tauk, " ", Bzk)
        #     if tauk != 0
        #         vol = sqrt(varianceByLogmoneyness(p.model, 0.0, tauk))

        #         fg = forward(p.model, S, tauk) * dividendGrowthFactor(dividends, 0.0, tauk)
        #         d1k, d2k = vaGBMd1d2( Bzk,fg, 0.0, 0.0, tauk, vol)
        #         rRate = -(log(discountFactor(p.model, tauk + 1e-6)) - log(discountFactor(p.model, tauk))) / 1e-6
        #         qRate = rRate - (log(forward(p.model, 1.0, tauk + 1e-6)) - log(forward(p.model, 1.0, tauk))) / 1e-6
        #         sum4k += uScale * wk * K * rRate * discountFactor(p.model, tauk) * normcdf(d1k)
        #         sum4k += -uScale * wk * qRate * fg * discountFactor(p.model, tauk) * normcdf(d2k)
        #     end
        # end
    end
    # sumgi = 0.0
    # previousG = 1.0
    # previousDate = 0.0
    # for dividend in dividends
    #     if dividend.exDate < tauMax
    #         currentG = dividendGrowthFactor(dividends, 0.0, dividend.exDate)
    #         sumgi += (currentG - previousG)*forward(p.model,1.0,dividend.exDate)/forward(p.model,1.0,previousDate) /discountFactor(p.model,previousDate)*discountFactor(p.model, dividend.exDate)
    #         previousG = currentG
    #     end
    # end
    # println("sumgi ",sumgi)
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
