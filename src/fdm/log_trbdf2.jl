export priceLogTRBDF2
using AQFED.TermStructure
using LinearAlgebra
using PPInterpolation

function priceLogTRBDF2(definition::StructureDefinition,
    spot::T,
    driftCurve::Curve, #The raw forward to τ (without cash dividends)
    varianceSurface::VarianceSurface, #variance to maturity
    discountCurve::Curve, #discount factor to payment date
    dividends::AbstractArray{CapitalizedDividend{T}};
    solverName="TDMA", M=400, N=100, ndev=4, Smax=zero(T), Smin=zero(T), dividendPolicy::DividendPolicy=Liquidator, grid="", alpha=0.01, useSpline=true, varianceConditioner::PecletConditioner=NoConditioner(), calibration=NoCalibration()) where {T}
    specialPoints = nonSmoothPoints(definition)
    obsTimes = observationTimes(definition)
    τ = last(obsTimes)
    t = collect(range(τ, stop=zero(T), length=N))
    dividends = filter(x -> x.dividend.exDate <= τ, dividends)
    sort!(dividends, by=x -> x.dividend.exDate)
    divDates = [x.dividend.exDate for x in dividends]
    t = vcat(t, divDates, obsTimes)
    sort!(t, order=Base.Order.Reverse)

    xi = (range(zero(T), stop=one(T), length=M))
    rawForward = log(spot / df(driftCurve, τ))
    Ui = if Smax == zero(T) || isnan(Smax)
        rawForward + ndev * sqrt(varianceByLogmoneyness(varianceSurface, 0.0, τ))
    else
        log(Smax)
    end
    Li =   if Smin <= zero(T) || isnan(Smin)
        Li = 2rawForward - Ui
    else
        log(Smin)
    end
    if grid == "Cubic"
        lnSi = makeCubicGrid(xi, Li, Ui, log.(specialPoints), 0.5, shift=0.0)
        if !isempty(specialPoints)
            strikeIndex = searchsortedlast(lnSi, log(specialPoints[1])) #FIXME handle strikeIndex=end
            diff = log(specialPoints[1])- (lnSi[strikeIndex]+lnSi[strikeIndex+1])/2
            if diff^2 > eps(T)
                @. lnSi += diff
                if diff < 0
                    append!(lnSi, Ui)
                else
                    prepend!(lnSi, Li)
                end
            end
        end
    elseif grid == "Shift" #Shift up, max is changed, not min.
        lnSi = @. Li + xi * (Ui - Li)
        if !isempty(specialPoints)
            strikeIndex = searchsortedlast(lnSi, log(specialPoints[1])) #FIXME handle strikeIndex=end
            diff = log(specialPoints[1])- (lnSi[strikeIndex]+lnSi[strikeIndex+1])/2
            if diff^2 > eps(T)
                @. lnSi += diff
                if diff < 0
                    append!(lnSi, Ui)
                else
                    prepend!(lnSi, Li)
                end
            end
        end
        if Smin < zero(T) || isnan(Smin)
            prepend!(Si, zero(T))
        end
    else #Uniform
        lnSi = @. Li + xi * (Ui - Li)
   
    end
    #    println("S ",Si)
    #    println("t ",t)
    Si = @. exp(lnSi)
    tip = t[1]
    payoff = makeFDMStructure(definition, Si)
    advance(payoff, tip)
    evaluate(payoff, Si)
    vLowerBound = zeros(T, length(Si))
    isLBActive = isLowerBoundActive(payoff)
    if isLBActive
        lowerBound!(payoff, vLowerBound)
    else
        ##FIXME how does the solver knows it is active or not?
    end
    vMatrix = currentValue(payoff)
    Jhi = @. (lnSi[2:end] - lnSi[1:end-1])
    rhsd = Array{T}(undef, length(Si))
    lhsd = ones(T, length(Si))
    rhsdl = Array{T}(undef, length(Si) - 1)
    lhsdl = Array{T}(undef, length(Si) - 1)
    rhsdu = Array{T}(undef, length(Si) - 1)
    lhsdu = Array{T}(undef, length(Si) - 1)
    lhs = Tridiagonal(lhsdl, lhsd, lhsdu)
    solverLB = if solverName == "TDMA" || solverName == "Thomas"
        TDMAMax{T}()
    elseif solverName == "DoubleSweep"
        DoubleSweep{T}(length(vLowerBound)) #not so great we need to length-can not use it as param to method
    elseif solverName == "LUUL"
        LUUL{T}(length(vLowerBound))
    elseif solverName == "SOR" || solverName == "PSOR"
        PSOR{T}(length(vLowerBound))
    elseif solverName == "BrennanSchwartz"
        BrennanSchwartz{T}(length(vLowerBound))
    else #solverName==PolicyIteration
        TDMAPolicyIteration{T}(length(vLowerBound))
    end
    solver = LowerBoundSolver(solverLB, isLBActive, lhs, vLowerBound)
    rhs = Tridiagonal(rhsdl, rhsd, rhsdu)
    v0Matrix = similar(vMatrix)
    v1 = Array{T}(undef, length(Si))
    muS = Array{T}(undef, length(Si))
    s2S = Array{T}(undef, length(Si))
    #pp = PPInterpolation.PP(3, T, T, length(Si))
    currentDivIndex = length(dividends)
    if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
        #jump and interpolate        
        for v in eachcol(vMatrix)
            # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
            pp = QuadraticLagrangePP(Si, copy(v))

            if dividendPolicy == Shift
                @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
            elseif dividendPolicy == Survivor
                @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
            else #liquidator
                @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                evaluateSorted!(pp, v, v1)
                # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
            end
        end
        currentDivIndex -= 1
    end
    beta = 2 * one(T) - sqrt(2 * one(T))
    for i = 2:length(t)
        ti = t[i]
        dt = tip - ti
        if dt < 1e-8
            continue
        end
        dfi = df(discountCurve, ti)
        dfip = df(discountCurve, tip)
        ri = calibrateRate(calibration, beta, dt, dfi, dfip)
        driftDfi = df(driftCurve, ti)
        driftDfip = df(driftCurve, tip)
        μi = calibrateDrift(calibration, beta, dt, dfi, dfip, driftDfi, driftDfip, ri)
        σi2 = (varianceByLogmoneyness(varianceSurface, 0.0, tip) * tip - varianceByLogmoneyness(varianceSurface, 0.0, ti) * ti) / (tip - ti)
        adjustDriftAndVol!(calibration, muS, s2S, μi, σi2, Jhi)

        @inbounds for j = 2:M-1
            s2S[j] = conditionedVariance(varianceConditioner, s2S[j], muS, lnSi[j], Jhi[j-1], Jhi[j])
            rhsd[j] = one(T) - dt * beta / 2 * ((muS[j] * (Jhi[j-1] - Jhi[j]) + s2S[j]) / (Jhi[j] * Jhi[j-1]) + ri)
            rhsdu[j] = dt * beta / 2 * (s2S[j] + muS[j] * Jhi[j-1]) / (Jhi[j] * (Jhi[j] + Jhi[j-1]))
            rhsdl[j-1] = dt * beta / 2 * (s2S[j] - muS[j] * Jhi[j]) / (Jhi[j-1] * (Jhi[j] + Jhi[j-1]))
        end
        #linear or Ke-rt same thing

        rhsd[1] = one(T) - dt * beta / 2 * (ri + muS[1] / Jhi[1])
        rhsdu[1] = dt * beta / 2 * muS[1] / Jhi[1]

        rhsd[M] = one(T) - dt * beta / 2 * (ri - muS[end] / Jhi[end])
        rhsdl[M-1] = -dt * beta / 2 * muS[end] / Jhi[end]

        v0Matrix[1:end, 1:end] = vMatrix
        advance(payoff, tip - dt * beta)
        for (iv, v) in enumerate(eachcol(vMatrix))
            mul!(v, rhs, @view v0Matrix[:, iv])
            #  evaluate(payoff, Si, iv)  #necessary to update knockin values from vanilla.

        end

        @. lhsd = one(T) - (rhsd - one(T))
        @. lhsdu = -rhsdu
        @. lhsdl = -rhsdl
        # lhsf = lu!(lhs)
        # lhsf = factorize(lhs)
        # ldiv!(v, lhsf , v1)
        decompose(solver, lhs)
        advance(payoff, tip - dt * beta)
        for (iv, v) in enumerate(eachcol(vMatrix))
            isLBActive = isLowerBoundActive(payoff)
            setLowerBoundActive(solver, isLBActive)
            solve!(solver, v1, v)
            v[1:end] = v1
            # evaluate(payoff, Si, iv)  #necessary to update knockin values from vanilla.
            if isLBActive
                lowerBound!(payoff, vLowerBound)
            end
        end

        #BDF2 step
        advance(payoff, ti)
        for (iv, v) in enumerate(eachcol(vMatrix))
            @. v1 = (v - (1 - beta)^2 * @view v0Matrix[:, iv]) / (beta * (2 - beta))
            # ldiv!(v , lhsf ,v1)
            isLBActive = isLowerBoundActive(payoff)
            setLowerBoundActive(solver, isLBActive)
            solve!(solver, v, v1)
            evaluate(payoff, Si, iv)  #necessary to update knockin values from vanilla.
            if isLBActive
                lowerBound!(payoff, vLowerBound)
            end
        end

        tip = ti
        if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
            #jump and interpolate        
            for v in eachcol(vMatrix)
                # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
                pp = QuadraticLagrangePP(Si, copy(v))

                if dividendPolicy == Shift
                    @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
                elseif dividendPolicy == Survivor
                    @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
                else #liquidator
                    @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                    evaluateSorted!(pp, v, v1)
                    # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
                end
            end
            currentDivIndex -= 1
        end
    end
    #PPInterpolation.computePP(pp,Si, @view(vMatrix[:,end]), PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())
    #return pp
    return QuadraticLagrangePP(Si, vMatrix[:, end])
end


function adjustDriftAndVol!(calibration::ForwardCalibration, muS, s2S, μi, σi2, Jhi)
    h = Jhi[1]
    eh = exp(h)
    muS[1] = μi * h / (eh - 1)
    for i = 2:length(Jhi)
        ehm = eh
        hm = h
        h = Jhi[i]
        eh = exp(h)
        s2S[i] = σi2 / 2 * (hm^2 * eh - h^2 / ehm - (hm^2 - h^2)) / (hm * eh + h / ehm - (hm + h))
        μid = μi * (hm * h * (hm + h)) / (hm^2 * eh - h^2 / ehm - (hm^2 - h^2))
        muS[i] = μid - σi2 / 2
    end
    muS[end] = μi * h / (1 - 1 / eh)
end


function adjustDriftAndVol!(calibration::NoCalibration, muS, s2S, μi, σi2, Jhi)
    muS[1] = μi
    s2S[1] = 0.0
    for i = 2:length(muS)-1
        muS[i] = μi - σi2 / 2
        s2S[i] = σi2
    end
    muS[end] = μi
    s2S[end]= 0.0
end

