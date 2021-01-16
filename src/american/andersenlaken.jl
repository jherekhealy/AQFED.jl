import AQFED.TermStructure: ConstantBlackModel
import AQFED.Black: blackScholesFormula

export AndersenLakeRepresentation, priceAmerican, americanBoundaryPutQDP

#Andersen-Lake American option pricing under negative rates
struct AndersenLakeNRepresentation
    isCall::Bool
    model::ConstantBlackModel
    tauMax::Float64
    tauMaxOrig::Float64
    nC::Int
    nTS1::Int
    nTS2::Int
    capX::Float64
    capXD::Float64
    avec::Vector{Float64}
    avecD::Vector{Float64}
    qvec::Vector{Float64}
    qvecD::Vector{Float64}
    wvec::Vector{Float64}
    yvec::Vector{Float64}
end

function AndersenLakeNRepresentation(
    model::ConstantBlackModel,
    tauMax::Float64,
    atol::Float64,
    nC::Int,
    nIter::Int,
    nTS1::Int,
    nTS2::Int;
    isCall::Bool = false,
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
    avecD = zeros(nC + 1)
    qvecD = zeros(nC + 1)
    wvec = zeros(nTS1)
    yvec = zeros(nTS1)
    ndiv2 = trunc(Int, (nTS1 + 1) / 2)
    hn = tanhsinhStep(nTS1)
    hvec = hn .* (ndiv2-1:-1:1)
    svec = @. pi * sinh(hvec) / 2
    @. @view(yvec[1:ndiv2-1]) = tanh(svec) #may be more precise to store y+1 directly instead
    @. @view(wvec[1:ndiv2-1]) = hn * pi * cosh(hvec) / (2 * cosh(svec)^2)
    for i = 1:ndiv2-1
        yvec[nTS1+1-i] = -yvec[i]
        wvec[nTS1+1-i] = wvec[i]
    end
    yvec[ndiv2] = 0
    wvec[ndiv2] = pi * hn / 2
    r = model.r
    q = model.q
    modelB = model
    if isCall  #use McDonald and Schroder symmetry
        r, q = q, r
        modelB = ConstantBlackModel(vol, r, q)
    end
    vol = model.vol

    capX = K
    capXD = K * r / q
    logCapX = log(capX)
    logCapXD = log(capXD)
    local fprev = capX
    local fprevD = capXD
    qvec[nC+1] = 0
    qvecD[nC+1] = 0
    for i = nC:-1:1
        zi = cos((i - 1) * pi / nC)
        taui = tauMax / 4 * (1 + zi)^2
        fi = americanBoundaryPutQDP(false, modelB, fprev, K, taui, atol)
        fprev = fi
        qvec[i] = (log(fi / capX))^2
        fi = americanBoundaryPutQDP(true, modelB, fprevD, K, taui, atol)
        fprevD = fi
        qvecD[i] = (log(fi / capXD))^2
    end
    B = computeBoundaryFromQVec(-1, qvec, capX)
    Bdown = computeBoundaryFromQVec(1, qvecD, capXD)
    crossingIndex = updateCrossingBoundariesALUp!(nC, B, capX, Bdown, capXD)
    tauMaxOrig = tauMax
    if crossingIndex >= 1
        #refine crossingIndex
        i = crossingIndex
        zi = cos(i * pi / nC)
        taui = tauMax / 4 * (1 + zi)^2
        tauStart = taui
        fStart = B[i]
        fStartD = Bdown[i]
        i += 1
        zi = cos(i * pi / nC)
        taui = tauMax / 4 * (1 + zi)^2
        tauEnd = taui
        fEnd = B[i]
        fEndD = Bdown[i]

        tauCrossing = searchCrossingTimeQDPN(
            tauEnd,
            fEnd,
            fEndD,
            tauStart,
            fStart,
            fStartD,
            K,
            model,
            atol,
        )

        tauMax = tauCrossing
        fprev = capX
        fprevD = capXD
        for i = nC:-1:1
            zi = cos((i - 1) * pi / nC)
            taui = tauMax / 4 * (1 + zi)^2
            fi = americanBoundaryPutQDP(false, modelB, fprev, K, taui, atol)
            fprev = fi
            qvec[i] = (log(fi / capX))^2
            fi = americanBoundaryPutQDP(true, modelB, fprevD, K, taui, atol)
            fprevD = fi
            qvecD[i] = (log(fi / capXD))^2
        end
        B = computeBoundaryFromQVec(-1, qvec, capX)
        Bdown = computeBoundaryFromQVec(1, qvecD, capXD)
    end ##of crossing refinement
    @. qvec = log(B / capX)^2
    @. qvecD = log(Bdown / capXD)^2
    tauVector = zeros(nTS1)
    k1 = zeros(nTS1)
    k2 = zeros(nTS1)
    d1Vector = zeros(nTS1)
    d2Vector = zeros(nTS1)
    #println("AL init ", "Up ", computeBoundaryFromQVec(-1, qvec, capX), " Down Boundary ", computeBoundaryFromQVec(1, qvecD, capXD))
    for j = 1:nIter
        updateAvec!(avec, nC, qvec)
        ##up boundary
        updateAvec!(avecD, nC, qvecD)
        for i = 1:nC
            zi = cos((i - 1) * pi / nC)
            taui = tauMax / 4 * (1 + zi)^2
            Kstari = K * exp(-(r - q) * taui)
            lnBtaui = logCapX - sqrt(qvec[i])
            sum1k = 0.0
            sum2k = 0.0
            @. tauVector = taui / 4 * (1 + yvec)^2
            crossIndex = 0
            @inbounds for sk1 = 1:nTS1
                if yvec[sk1] != -1
                    tauk = tauVector[sk1]
                    zck = 2 * sqrt((taui - tauk) / tauMax) - 1
                    qckD = chebQck(avecD, zck)
                    qck = chebQck(avec, zck)
                    lnBtaukD = logCapXD + sqrt(qckD)
                    lnBtauk = logCapX - sqrt(qck)
                    if (lnBtaukD > lnBtauk) && crossIndex == 0
                        crossIndex = sk1
                    end
                    if crossIndex == 0
                        sqrtv = sqrt(tauk) * vol
                        d1k = ((lnBtaui - lnBtaukD) + (r - q) * tauk) / sqrtv + sqrtv / 2
                        d2k = d1k - sqrtv
                        sum1k -= wvec[sk1] * exp(-q * tauk) * (yvec[sk1] + 1) * normcdf(d1k)
                        sum2k -= wvec[sk1] * exp(-r * tauk) * (yvec[sk1] + 1) * normcdf(d2k)

                        d1k = ((lnBtaui - lnBtauk) + (r - q) * tauk) / sqrtv + sqrtv / 2
                        d2k = d1k - sqrtv
                        sum1k += wvec[sk1] * exp(-q * tauk) * (yvec[sk1] + 1) * normcdf(d1k)
                        sum2k += wvec[sk1] * exp(-r * tauk) * (yvec[sk1] + 1) * normcdf(d2k)
                    end
                end
            end

            sum1k = exp(q * taui) / 2 * taui * sum1k
            sum2k = exp(r * taui) / 2 * taui * sum2k
            sqrtv = sqrt(taui) * vol
            d1i = ((lnBtaui - log(K)) + (r - q) * taui) / sqrtv + sqrtv / 2
            d2i = d1i - sqrtv

            Ni = -1 + exp(r * taui) + normcdf(d2i) + r * sum2k
            Di = -1 + exp(q * taui) + normcdf(d1i) + q * sum1k
            NiOverDi = Ni / Di
            if Di == 0.0 && Ni == 0.0
                #use asymptotic expansion cdf = erfc(-x/sqrt2)/2 and erfc(x) = e^{-x^2}/(x*sqrtpi)*(1-1/(2*x^2))
                NiOverDi = exp(-(d2i^2 - d1i^2) / 2) * (d1i / d2i)
            end
            fi = Kstari * NiOverDi
            if fi < 0
                fi = 1e-4 * Kstari
            end
            lfc = log(fi / capX)
            if isnan(lfc)
                throw(DomainError(
                    fi,
                    string("Nan qvec ", capX, " ", lnBtaui, " ", qvec[i]),
                ))
            end
            qvec[i] = lfc^2
        end
        qvec[nC+1] = 0

        ##Down boundary
        updateAvec!(avec, nC, qvec)
        for i = 1:nC
            zi = cos((i - 1) * pi / nC)
            taui = tauMax / 4 * (1 + zi)^2
            Kstari = K * exp(-(r - q) * taui)
            lnBtauiD = logCapXD + sqrt(qvecD[i])
            sum1kD = 0.0
            sum2kD = 0.0
            @. tauVector = taui / 4 * (1 + yvec)^2
            crossIndex = 0
            @inbounds for sk1 = 1:nTS1
                if yvec[sk1] != -1
                    tauk = tauVector[sk1]
                    zck = 2 * sqrt((taui - tauk) / tauMax) - 1
                    qckD = chebQck(avecD, zck)
                    qck = chebQck(avec, zck)
                    lnBtaukD = logCapXD + sqrt(qckD)
                    lnBtauk = logCapX - sqrt(qck)
                    if (lnBtaukD > lnBtauk) && crossIndex == 0
                        crossIndex = sk1
                    end
                    if crossIndex == 0
                        sqrtv = sqrt(tauk) * vol
                        d1k = ((lnBtauiD - lnBtaukD) + (r - q) * tauk) / sqrtv + sqrtv / 2
                        d2k = d1k - sqrtv
                        sum1kD -=
                            wvec[sk1] * exp(-q * tauk) * (yvec[sk1] + 1) * normcdf(d1k)
                        sum2kD -=
                            wvec[sk1] * exp(-r * tauk) * (yvec[sk1] + 1) * normcdf(d2k)

                        d1k = ((lnBtauiD - lnBtauk) + (r - q) * tauk) / sqrtv + sqrtv / 2
                        d2k = d1k - sqrtv
                        sum1kD +=
                            wvec[sk1] * exp(-q * tauk) * (yvec[sk1] + 1) * normcdf(d1k)
                        sum2kD +=
                            wvec[sk1] * exp(-r * tauk) * (yvec[sk1] + 1) * normcdf(d2k)
                    end
                end
            end

            sum1kD = exp(q * taui) / 2 * taui * sum1kD
            sum2kD = exp(r * taui) / 2 * taui * sum2kD
            sqrtv = sqrt(taui) * vol
            d1i = ((lnBtauiD - log(K)) + (r - q) * taui) / sqrtv + sqrtv / 2
            d2i = d1i - sqrtv
            fiold = capXD * exp(sqrt(qvecD[i]))
            Ni =
                -1 + exp(r * taui) + normcdf(d2i) + r * sum2kD -
                fiold / Kstari * q * (sum1kD)
            Di = -1 + exp(q * taui) + normcdf(d1i)
            NiOverDi = Ni / Di
            if Di == 0.0 && Ni == 0.0
                #use asymptotic expansion cdf = erfc(-x/sqrt2)/2 and erfc(x) = e^{-x^2}/(x*sqrtpi)*(1-1/(2*x^2))
                NiOverDi = exp(-(d2i^2 - d1i^2) / 2) * (d1i / d2i)
            end
            fi = Kstari * NiOverDi
            if fi < 0
                fi = 1e-4 * Kstari
            end
            lfc = log(fi / capXD)
            if isnan(lfc)
                throw(DomainError(
                    fi,
                    string("Nan qvec ", capX, " ", lnBtaui, " ", qvec[i]),
                ))
            end
            qvecD[i] = lfc^2
        end
        qvecD[nC+1] = 0
        ##update boundaries
        B = computeBoundaryFromQVec(-1, qvec, capX)
        Bdown = computeBoundaryFromQVec(1, qvecD, capXD)
        updateCrossingBoundariesALUp!(nC, B, capX, Bdown, capXD)
        @. qvec = (log(B / capX))^2
        @. qvecD = (log(Bdown / capXD))^2
        # updateAvec!(avec, nC, qvec)
        # updateAvec!(avecD, nC, qvecD)
        # println("AL ",j, " Up ", computeBoundaryFromQVec(-1, qvec, capX), " Down Boundary ", computeBoundaryFromQVec(1, qvecD, capXD))
    end

    updateAvec!(avecD, nC, qvecD)
    updateAvec!(avec, nC, qvec)
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
    return AndersenLakeNRepresentation(
        isCall,
        model,
        tauMax,
        tauMaxOrig,
        nC,
        nTS1,
        nTS2,
        capX,
        capXD,
        avec,
        avecD,
        qvec,
        qvecD,
        wvec,
        yvec,
    )

    ##up

end



function priceAmerican(p::AndersenLakeNRepresentation, K::Float64, S::Float64)::Float64
    vol, r, q = p.model.vol, p.model.r, p.model.q
    if p.isCall #use McDonald and Schroder symmetry
        K, S = S, K
        r, q = q, r
    end
    capX, capXD = p.capX * K, p.capXD * K
    f0 = exp(-sqrt(p.qvec[1])) * capX
    f0D = exp(sqrt(p.qvecD[1])) * capXD
    if S <= f0 && S >= f0D && p.tauMax == p.tauMaxOrig
        return max(K - S, 0.0)
    end

    tauMax, tauMaxOrig, nTS2 = p.tauMax, p.tauMaxOrig, p.nTS2
    wvec, yvec, avec, avecD = p.wvec, p.yvec, p.avec, p.avecD
    nC, rK, qS = p.nC, r * K, q * S

    uMax = tauMax
    uMin = 0.0
    uScale = (uMax - uMin) / 2
    uShift = (uMax + uMin) / 2
    sum4k = 0.0
    isCrossed = false
    for sk2 = nTS2:-1:1
        wk = wvec[sk2]
        yk = yvec[sk2]
        uk = uScale * yk + uShift
        if abs(yk) != 1
            zck = 2 * sqrt(uk / tauMax) - 1
            qck = chebQck(avec, zck)
            qckD = chebQck(avecD, zck)
            Bzk = capX * exp(-sqrt(qck))
            BzkD = capXD * exp(sqrt(qckD))
            if Bzk < BzkD
                isCrossed = true
            end
            if !isCrossed
                tauk = uMax - uk + tauMaxOrig - tauMax
                d1k, d2k = vaGBMd1d2(S, Bzk, r, q, tauk, vol)
                sum4k += wk * rK * exp(-r * tauk) * normcdf(-d2k)
                sum4k += -wk * qS * exp(-q * tauk) * normcdf(-d1k)
                d1k, d2k = vaGBMd1d2(S, BzkD, r, q, tauk, vol)
                sum4k -= wk * rK * exp(-r * tauk) * normcdf(-d2k)
                sum4k += wk * qS * exp(-q * tauk) * normcdf(-d1k)
            end
        end
    end

    euro = blackScholesFormula(
        false,
        K,
        S,
        vol * vol * tauMaxOrig,
        exp(-(r - q) * tauMaxOrig),
        exp(-r * tauMaxOrig),
    )
    price = euro + uScale * sum4k
    price = max(K - S, price)
    return price
end

computeBoundaryFromQVec(sign::Int, qvec::Vector{Float64}, capX::Float64) =
    @. exp(sign * sqrt(qvec)) * capX

function updateCrossingBoundariesALUp!(
    m::Int,
    B::Vector{Float64},
    capX::Float64,
    Bdown::Vector{Float64},
    capXD::Float64,
)::Int
    for i = m-1:-1:1
        if B[i+1] < B[i]
            B[i] = B[i+1]
        end
        if Bdown[i] < Bdown[i+1]
            Bdown[i] = Bdown[i+1]
        end
        if B[i] > capX
            B[i] = capX
        end
        if Bdown[i] < capXD
            Bdown[i] = capXD
        end
    end
    icross = m
    while icross > 0 && Bdown[icross] < B[icross]
        icross -= 1
    end
    if icross >= 1
        if Bdown[icross] > capX
            Bdown[icross] = capX
        end
        if B[icross] < capXD
            B[icross] = capXD
        end
        level = B[icross]
        level = max(Bdown[icross+1], level)
        level = min(B[icross+1], level)
        for i = icross:-1:1
            B[i] = level
            Bdown[i] = B[i]
        end
    end
    return icross
end
