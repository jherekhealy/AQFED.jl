export DividendPolicy, priceTRBDF2, NoCalibration, ForwardCalibration
using AQFED.TermStructure
using LinearAlgebra
using PPInterpolation
@enum DividendPolicy begin
    Liquidator
    Survivor
    Shift
end



#TODO use term structure of rates and vols from a model.
#TODO use upstream/downstream deriv/expo fitting if convect dominates.
function priceTRBDF2(definition::StructureDefinition,
    spot::T,
    rawForward::T, #The raw forward to τ (without cash dividends)
    variance::T, #variance to maturity
    discountDf::T, #discount factor to payment date
    dividends::AbstractArray{CapitalizedDividend{T}};
    solverName="TDMA", M=400, N=100, ndev=4, Smax=zero(T), Smin=zero(T), dividendPolicy::DividendPolicy=Liquidator, calibration=NoCalibration(), varianceConditioner::PecletConditioner=NoConditioner()) where {T}
    obsTimes = observationTimes(definition)
    τ = last(obsTimes)
    varianceSurface = FlatSurface(sqrt(variance / τ))
    discountCurve = ConstantRateCurve(-log(discountDf) / τ)
    driftCurve = ConstantRateCurve(log(rawForward / spot) / τ)
    return priceTRBDF2(definition, spot, driftCurve, varianceSurface, discountCurve, dividends, solverName=solverName, M=M, N=N, ndev=ndev, Smax=Smax, Smin=Smin, dividendPolicy=dividendPolicy, calibration=calibration, varianceConditioner=varianceConditioner)
end

function priceTRBDF2(definition::StructureDefinition,
    spot::T,
    driftCurve::Curve, #The raw forward to τ (without cash dividends)
    varianceSurface::VarianceSurface, #variance to maturity
    discountCurve::Curve, #discount factor to payment date
    dividends::AbstractArray{CapitalizedDividend{T}};
    solverName="LUUL", M=400, N=100, ndev=4, Smax=zero(T), Smin=zero(T), dividendPolicy::DividendPolicy=Liquidator, grid::Grid=UniformGrid(false), varianceConditioner::PecletConditioner=NoConditioner(), calibration=NoCalibration()) where {T}
    obsTimes = observationTimes(definition)
    τ = last(obsTimes)
    t = collect(range(τ, stop=zero(T), length=N))
    dividends = filter(x -> x.dividend.exDate <= τ, dividends)
    sort!(dividends, by=x -> x.dividend.exDate)
    divDates = [x.dividend.exDate for x in dividends]
    t = vcat(t, divDates, obsTimes)
    sort!(t, order=Base.Order.Reverse)
    specialPoints = sort(nonSmoothPoints(definition))
    xi = (range(zero(T), stop=one(T), length=M))
    rawForward = spot / df(driftCurve, τ)
    Ui = if Smax == zero(T) || isnan(Smax)
        rawForward * exp(ndev * sqrt(varianceByLogmoneyness(varianceSurface, 0.0, τ)))
    else
        Smax
    end
    Li = if Smin < zero(T) || isnan(Smin)
        rawForward^2 / Smax
    else
        Smin
    end
    if !isempty(specialPoints)
    Ui = max(Ui, maximum(specialPoints))
    Li = min(Li, minimum(specialPoints))
    end
    isMiddle = ones(Bool,length(specialPoints))
    # isMiddle = zeros(Bool,length(specialPoints))
    Si = makeArray(grid, xi, Li, Ui, specialPoints,isMiddle)
        # println("S ",Si)
    #    println("t ",t)
    tip = t[1]
    payoff = makeFDMStructure(definition, Si)
    advance(definition, payoff, tip)
    evaluate(definition, payoff, Si)
    vLowerBound = zeros(T, length(Si))
    isLBActive = isLowerBoundActive(definition, payoff)
    if isLBActive
        lowerBound!(payoff, vLowerBound)
    else
        ##FIXME how does the solver knows it is active or not?
    end
    vMatrix = currentValue(payoff)
    Jhi = @. (Si[2:end] - Si[1:end-1])
    rhsd = zeros(T, length(Si))
    lhsd = ones(T, length(Si))
    rhsdl = zeros(T, length(Si) - 1)
    lhsdl = zeros(T, length(Si) - 1)
    rhsdu = zeros(T, length(Si) - 1)
    lhsdu = zeros(T, length(Si) - 1)
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
    v1 = zeros(T, length(Si))
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

        @inbounds for j = 2:M-1
            s2S = σi2 * Si[j]^2
            muS = μi * Si[j]
            s2S = conditionedVariance(varianceConditioner, s2S, muS, Si[j], Jhi[j-1], Jhi[j])
            rhsd[j] = one(T) - dt * beta / 2 * ((muS * (Jhi[j-1] - Jhi[j]) + s2S) / (Jhi[j] * Jhi[j-1]) + ri)
            rhsdu[j] = dt * beta / 2 * (s2S + muS * Jhi[j-1]) / (Jhi[j] * (Jhi[j] + Jhi[j-1]))
            rhsdl[j-1] = dt * beta / 2 * (s2S - muS * Jhi[j]) / (Jhi[j-1] * (Jhi[j] + Jhi[j-1]))
        end
        #linear or Ke-rt same thing
        rhsd[1] = one(T) - dt * beta / 2 * (ri + μi * Si[1] / Jhi[1])
        rhsdu[1] = dt * beta / 2 * μi * Si[1] / Jhi[1]

        rhsd[M] = one(T) - dt * beta / 2 * (ri - μi * Si[end] / Jhi[end])
        rhsdl[M-1] = -dt * beta / 2 * μi * Si[end] / Jhi[end]

        v0Matrix[1:end, 1:end] = vMatrix
        advance(definition,payoff, tip - dt * beta)
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
        advance(definition, payoff, tip - dt * beta)
        for (iv, v) in enumerate(eachcol(vMatrix))
            isLBActive = isLowerBoundActive(definition, payoff)
            setLowerBoundActive(solver, isLBActive)
            solve!(solver, v1, v)
            v[1:end] = v1
            # evaluate(payoff, Si, iv)  #necessary to update knockin values from vanilla.
            if isLBActive
                lowerBound!(payoff, vLowerBound)
            end
        end

        #BDF2 step
        advance(definition, payoff, ti)
        for (iv, v) in enumerate(eachcol(vMatrix))
            @. v1 = (v - (1 - beta)^2 * @view v0Matrix[:, iv]) / (beta * (2 - beta))
            # ldiv!(v , lhsf ,v1)
            isLBActive = isLowerBoundActive(definition, payoff)
            setLowerBoundActive(solver, isLBActive)
            solve!(solver, v, v1)
            evaluate(definition, payoff, Si, iv)  #necessary to update knockin values from vanilla.
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


abstract type DiscreteCalibration end
struct NoCalibration <: DiscreteCalibration
end

function calibrateRate(c::NoCalibration, beta, dt, dfi, dfip)
    log(dfi / dfip) / dt
end

function calibrateDrift(c::NoCalibration, beta, dt, dfi, dfip, driftDfi, driftDfip, r)
    log(driftDfi / driftDfip) / dt
end

struct ForwardCalibration <: DiscreteCalibration
end

calibrateRate(c::ForwardCalibration, beta, dt, dfi, dfip) = calibrateRate(c, beta, dt, dfip / dfi)

function calibrateRate(c::ForwardCalibration, beta, dt, factor)
    a = beta * (1 - beta) * factor / 2
    b = ((2 - beta^2) * factor + 1 + (1 - beta)^2) / 2
    c = (2 - beta) * (factor - 1)
    r = (-b + sqrt(max(0, b^2 - 4 * a * c))) / (2 * a * dt)
    r
end

calibrateDrift(c::ForwardCalibration, beta, dt, dfi, dfip, driftDfi, driftDfip, r) = r - calibrateRate(c, beta, dt, dfip / dfi * driftDfi / driftDfip)

