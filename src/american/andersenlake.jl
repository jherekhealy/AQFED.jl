import AQFED.TermStructure: ConstantBlackModel
import AQFED.Math: normcdfCody, normpdf
import AQFED.Black: blackScholesFormula

export AndersenLakeRepresentation, priceAmerican, americanBoundaryPutQDP

@inline normcdf(z::Float64) =  normcdfCody(z) #Faster apparently

struct AndersenLakeRepresentation
    model::ConstantBlackModel
    K::Float64
    tauMax::Float64
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
    K::Float64,
    tauMax::Float64,
    atol::Float64,
    nC::Int,
    nIter::Int,
    nTS1::Int,
    nTS2::Int,
)
    avec = zeros(nC + 1)
    qvec = zeros(nC + 1)
    wvec = zeros(nTS1)
    yvec = zeros(nTS1)
    ndiv2 = trunc(Int, (nTS1 + 1) / 2)
    hn = tanhsinhStep(nTS1)
    hvec = hn .* (ndiv2-1:-1:1)
    svec = @. pi * sinh(hvec) / 2
    @. @view(yvec[1:ndiv2-1]) = tanh(svec)
    @. @view(wvec[1:ndiv2-1]) = hn * pi * cosh(hvec) / (2 * (cosh(svec))^2)
    for i = 1:ndiv2-1
        yvec[nTS1+1-i] = -yvec[i]
        wvec[nTS1+1-i] = wvec[i]
    end
    yvec[ndiv2] = 0
    wvec[ndiv2] = pi * hn / 2
    capX = K
    r = model.r
    q = model.q
    vol = model.vol
    if q > r
        capX = K * r / q
    end
    local fprev = capX
    qvec[nC+1] = 0
    for i = nC:-1:1
        zi = cos((i - 1) * pi / nC)
        taui = tauMax / 4 * (1 + zi)^2
        fi = americanBoundaryPutQDP(false, model, fprev, K, taui, atol)
        fprev = fi
        qvec[i] = (log(fi / capX))^2
    end
    # tauVector = zeros(nTS1)
    # k1 = zeros(nTS1)
    # k2 = zeros(nTS1)
    # d1Vector = zeros(nTS1)
    # d2Vector = zeros(nTS1)
    for j = 1:nIter
        #println(j, "qvec", qvec, "avec", avec)
        for sk = 0:nC
            sumi = 0.5 * qvec[1]
            for i = 2:nC
                sumi += qvec[i] * cos((i - 1) * sk * pi / nC)
            end
            sumi = sumi + qvec[nC+1] / 2 * cos(sk * pi)
            avec[sk+1] = 2 * sumi / nC
        end
        for i = 1:nC
            zi = cos((i - 1) * pi / nC)
            taui = tauMax / 4 * (1 + zi)^2
            Kstari = K * exp(-(r - q) * taui)
            lnBtaui = log(capX) - sqrt(qvec[i])
            sum1k = 0.0
            sum2k = 0.0
            # @. tauVector = taui / 4 * (1 + yvec)^2
            @inbounds for sk1 = 1:nTS1
                wk = wvec[sk1]
                yk = yvec[sk1]
                # tauk = tauVector[sk1]
                tauk =taui / 4 * (1 + yk)^2
                if yk != -1
                    zck = 2 * sqrt((taui - tauk) / tauMax) - 1
                    qck = chebQck(avec, zck)
                    lnBtauk = log(capX) - sqrt(qck)
                    sqrtv = sqrt(tauk) * vol
                    # d1Vector[sk1] =
                    #     ((lnBtaui - lnBtauk) + (r - q) * tauk) / sqrtv + sqrtv / 2
                    # d2Vector[sk1] = d1Vector[sk1] - sqrtv
                    d1k = ((lnBtaui - lnBtauk) + (r - q) * tauk) / sqrtv + sqrtv / 2
                    d2k = d1k - sqrtv
                    sum1k += wk * exp(-q * tauk) * (yk + 1) * normcdf(d1k)
                    sum2k += wk * exp(-r * tauk) * (yk + 1) * normcdf(d2k)
                end
            end
            # @. k1 = wvec * exp(-q * tauVector) * (yvec + 1) * normcdf(d1Vector)
            # @. k2 = wvec * exp(-r * tauVector) * (yvec + 1) * normcdf(d2Vector)
            # sum1k = exp(q * taui) / 2 * taui * sum(k1)
            # sum2k = exp(r * taui) / 2 * taui * sum(k2)
            sum1k = exp(q * taui) / 2 * taui * sum1k
           sum2k = exp(r * taui) / 2 * taui * sum2k
           sqrtv = sqrt(taui) * vol
            d1i = ((lnBtaui - log(K)) + (r - q) * taui) / sqrtv + sqrtv / 2
            d2i = d1i - sqrtv

            Ni = normcdf(d2i) + r * sum2k
            Di = normcdf(d1i) + q * sum1k
            NiOverDi = Ni / Di
            if Di == 0.0 && Ni == 0.0
                #use asymptotic expansion cdf = erfc(-x/sqrt2)/2 and erfc(x) = e^{-x^2}/(x*sqrtpi)*(1-1/(2*x^2))
                NiOverDi = exp(-(d2i^2 - d1i^2) / 2) * (d1i / d2i)
            end
            fi = Kstari * NiOverDi
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
        #println("new iteration", j, computeBoundaryFromQVec(qvec, capX))
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
        model,
        K,
        tauMax,
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


function priceAmerican(p::AndersenLakeRepresentation, S::Float64)::Float64
    K, capX = p.K, p.capX
    lfc0 = -sqrt(p.qvec[1])
    f0 = exp(lfc0) * capX
    if S < f0
        return max(K - S, 0.0)
    end

    tauMax, nTS2 = p.tauMax, p.nTS2
    wvec, yvec, avec = p.wvec, p.yvec, p.avec
    vol,r , q = p.model.vol, p.model.r, p.model.q
    nC, rK, qS  = p.nC, r * K, q * S

    uMax = tauMax
    uMin = 0.0
    uScale = (uMax - uMin) / 2
    uShift = (uMax + uMin) / 2
    sum4k = 0.0
    for sk2 = 1:nTS2
        wk = wvec[sk2]
        yk = yvec[sk2]
        uk = uScale * yk + uShift
        if yk != 1
            zck = 2 * sqrt(uk / tauMax) - 1
            qck = chebQck(avec, zck)
            Bzk = capX * exp(-sqrt(qck))
            tauk = uMax - uk
            d1k, d2k = vaGBMd1d2(S, Bzk, r, q, tauk, vol)
            sum4k += wk * rK * exp(-r * tauk) *normcdf(-d2k)
            sum4k += - wk * qS * exp(-q * tauk) * normcdf(-d1k)
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


@inline function chebQck(avec::Vector{Float64}, zck::Float64)
    b2 = 0.0
    nC = length(avec)-1
    b1 = avec[nC+1] / 2
     @inbounds @fastmath for sk22 = nC:-1:2
        bd = avec[sk22] - b2
        b2 = b1
        b1 = 2 * zck * b1 + bd
    end
    b0 = avec[1] + 2 * zck * b1 - b2
    qck = max(0.0, (b0 - b2) / 2)
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
    obj = @inline function (Sstar::Float64)
        d1, d2 = vaGBMd1d2(Sstar, K, r, q, tauMax, vol)
        Nd1 = normcdf(-d1)
        d1dS = 1.0 / (Sstar * SqrV)
        snd1 = normpdf(d1)
        Nd1dS = -snd1 * d1dS
        snd1dS = -d1 * snd1 * d1dS
        d1d2S = -d1dS / Sstar
        Nd1d2S = -snd1 * d1d2S - snd1dS * d1dS
        snd1d2S = -d1 * snd1 * d1d2S - (d1 * snd1dS + snd1 * d1dS) * d1dS
        Nd2 = normcdf(-d2)
        snd2 = normpdf(d2)
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
        	Sstarn = Sstar + hn*(1+0.5*halleyTerm/(1-halleyTerm))
        # }
        if Sstarn < 0
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

    #println("Sstar", Sstar, iter)
    return Sstar
end


function tanhsinhStep(nTS2::Int)::Float64
    n = trunc(Int, (nTS2 + 1) / 2)
    a = pi / 2
    return lambertW(4 * a * n) / n
end

function lambertW(x::Float64)::Float64
    a = log(x)
    b = log(a)
    A0x = a - b + b / a
    A0x += b * (-2 + b) / (2 * a^2)
    A0x += b * (6 - 9 * b + 2 * b^2) / (6 * a^3)
    A0x += b * (-12 + 36 * b - 22 * b^2 + 3 * b^3) / (12 * a^4)
    A0x += b * (60 - 300 * b + 350 * b^2 - 125 * b^3 + 12 * b^4) / (60 * a^5)
    wn = A0x
    zn = log(x / wn) - wn
    qn = 2 * (1 + wn) * (1 + wn + 2 * zn / 3)
    en = (zn / (1 + wn)) * ((qn - zn) / (qn - 2 * zn))
    return wn * (1 + en)
end

@inline function vaGBMd1d2(Btaui::T, Btauk::T, r::T, q::T, tauk::T, vol::T) where {T}
    sqrtv = sqrt(tauk) * vol
    d1 = (log(Btaui / Btauk) + (r - q) * tauk) / sqrtv + sqrtv / 2
    d2 = d1 - sqrtv
    return d1, d2
end
