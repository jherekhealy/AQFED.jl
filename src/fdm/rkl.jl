export priceRKL2
using AQFED.TermStructure
using AQFED.Math
using LinearAlgebra
using PPInterpolation

function priceRKG2B(definition::StructureDefinition,
    spot::T,
    model,
    dividends::AbstractArray{CapitalizedDividend{T}};
    M=400, N=100, ndev=4, Smax=zero(T), Smin=zero(T), dividendPolicy::DividendPolicy=Liquidator, grid::Grid=UniformGrid(false), varianceConditioner::PecletConditioner=NoConditioner(), calibration=NoCalibration(), useSqrt=false) where {T}
    obsTimes = observationTimes(definition)
    τ = last(obsTimes)
    t = collect(range(τ, stop=zero(T), length=N))
    if useSqrt
        du = sqrt(τ) / (N - 1)
        for i = 1:N-1
            ui = du * (i - 1)
            t[i] = τ - ui^2
        end
    end

    dividends = filter(x -> x.dividend.exDate <= τ, dividends)
    sort!(dividends, by=x -> x.dividend.exDate)
    divDates = [x.dividend.exDate for x in dividends]
    t = vcat(t, divDates, obsTimes)
    sort!(t, order=Base.Order.Reverse)
    specialPoints = sort(nonSmoothPoints(definition))
    if isempty(specialPoints)
        append!(specialPoints, spot)
    end
    xi = (range(zero(T), stop=one(T), length=M))
    rawForward = forward(model, spot, τ)
    Ui = if Smax == zero(T) || isnan(Smax)
        v0 = varianceByLogmoneyness(model, 0.0, τ) * τ
        rawForward * exp(ndev * sqrt(v0) - 0.5 * v0)
    else
        Smax
    end
    Li = if Smin < zero(T) || isnan(Smin)
        rawForward^2 / Ui
    else
        Smin
    end
    if !isempty(specialPoints)
        Ui = max(Ui, maximum(specialPoints) * 1.01)
        Li = min(Li, minimum(specialPoints) * 0.99)
    end
    isMiddle = zeros(Bool, length(specialPoints))
    Si = makeArray(grid, xi, Li, Ui, specialPoints, isMiddle)
    #  println("S ",Si)   
    #     println("t ",t)
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
    # println("initial payoff ",vMatrix[:,1], " ",vMatrix[:,2])
    Jhi = @. (Si[2:end] - Si[1:end-1])
    rhsd = zeros(T, length(Si))
    lhsd = ones(T, length(Si))
    rhsdl = zeros(T, length(Si) - 1)
    lhsdl = zeros(T, length(Si) - 1)
    rhsdu = zeros(T, length(Si) - 1)
    lhsdu = zeros(T, length(Si) - 1)
    lhs = Tridiagonal(lhsdl, lhsd, lhsdu)
    rhs = Tridiagonal(rhsdl, rhsd, rhsdu)
    v0Matrix = similar(vMatrix)
    v1Matrix = similar(vMatrix)
    v2Matrix = similar(vMatrix)
    #pp = PPInterpolation.PP(3, T, T, length(Si))
    currentDivIndex = length(dividends)
    if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
        #jump and interpolate        
        for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
            # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
            pp = QuadraticLagrangePP(Si, copy(v))
            if dividends[currentDivIndex].dividend.isProportional
                @. v = pp(Si * (1 - dividends[currentDivIndex].dividend.amount))
            else
                if dividendPolicy == Shift
                    @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
                elseif dividendPolicy == Survivor
                    @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
                else #liquidator
                    v1 = @view v1Matrix[:, iv]
                    @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                    evaluateSorted!(pp, v, v1)
                    # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
                end
            end
        end
        currentDivIndex -= 1
    end

    s = 0.0
    local a, b, w0, w1
    α = 1.5
    for i = 2:length(t)
        ti = t[i]
        dt = tip - ti
        if dt > 1e-8
            dfi = discountFactor(model, ti)
            dfip = discountFactor(model, tip)
            ri = calibrateRate(calibration, 1.0, dt, dfi, dfip)
            driftDfi = 1 / forward(model, 1.0, ti)
            driftDfip = 1 / forward(model, 1.0, tip)
            μi = calibrateDrift(calibration, 1.0, dt, dfi, dfip, driftDfi, driftDfip, ri)
            σi2 = (varianceByLogmoneyness(model, 0.0, tip) * tip - varianceByLogmoneyness(model, 0.0, ti) * ti) / (tip - ti)

            @inbounds for j = 2:M-1
                s2S = σi2 * Si[j]^2
                muS = μi * Si[j]
                s2S = conditionedVariance(varianceConditioner, s2S, muS, Si[j], Jhi[j-1], Jhi[j])
                rhsd[j] = ((muS * (Jhi[j-1] - Jhi[j]) + s2S) / (Jhi[j] * Jhi[j-1]) + ri)
                rhsdu[j] = -(s2S + muS * Jhi[j-1]) / (Jhi[j] * (Jhi[j] + Jhi[j-1]))
                rhsdl[j-1] = -(s2S - muS * Jhi[j]) / (Jhi[j-1] * (Jhi[j] + Jhi[j-1]))
            end
            #linear or Ke-rt same thing
            if s == 0.0

                s, a, b, w0, w1 = initRKGCoeffs(dt, rhs, stages=0)
            end
            #  v0Matrix[1:end, 1:end] = vMatrix
            advectionCoeffDown = μi * Si[1] / Jhi[1]
            sinkCoeffDown = ri
            advectionCoeffUp = -μi * Si[end] / Jhi[end]
            sinkCoeffUp = ri
            advance(definition, payoff, tip - dt)
            #for each stage, iterate of payoff variables.
            mu1b = 3 * b[2] / b[1] * w1 * dt
            print(ti, " ", 1, " ", tip - mu1b)
            for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix))) #we don't go reverse here as we want to use initial vanilla price in KO for explicit step.
                bcDown, bcUp = boundaryConditions(definition, tip - mu1b, iv)
                # applyPostBoundaryCondition(Up(), iv, bcUp, Si, vMatrix, vMatrix,  mu1b, advectionCoeffUp, sinkCoeffUp)
                rhsi = I - mu1b * rhs #TODO keep a fixed list of lhs,rhs.                
                applyPreBoundaryCondition(Down(), iv, bcDown, Si, vMatrix, rhsi, mu1b, advectionCoeffDown, sinkCoeffDown) #if Dirichlet on grid, just do it, otherwise use Ghost point. rhs vs lhs?
                applyPreBoundaryCondition(Up(), iv, bcUp, Si, vMatrix, rhsi, mu1b, advectionCoeffUp, sinkCoeffUp)
                Y1 = @view v1Matrix[:, iv]
                mul!(Y1, rhsi, v)
                applyPostBoundaryCondition(Up(), iv, bcUp, Si, v1Matrix, v1Matrix, mu1b, advectionCoeffUp, sinkCoeffUp)
                #alternative: update Y1 bc after solving, based on v for vanilla.
                #    println(iv, " explicitStep", mu1b, " ", v, "\n ", Y1)
            end
            MY0Matrix = (v1Matrix - vMatrix) / mu1b
            #TODO  enforceLowerBound(Yjm, lowerBound, M)
            v0Matrix .= vMatrix
            for j = 2:s
                #b and a indices are shifted by one compared to paper.
                tj = tip - dt * (j + 2α + 1) * (j - 1) / ((s + 2α + 1) * (s - 1))
                muu = (2 * j + 1) * b[j+1] / (j * b[j])
                muj = muu * w0
                mujb = muu * w1 * dt
                gammajb = -a[j] * mujb
                nuj = -(j + 1) * b[j+1] / (j * b[j-1])

                print(ti, " ", j, " ", tj)
                for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix))) #we don't go reverse here as we want to use initial 
                    MY0 = @view MY0Matrix[:, iv]
                    Y0 = @view v0Matrix[:, iv]
                    Y1 = @view v1Matrix[:, iv]
                    Y2 = @view v2Matrix[:, iv]

                    bcDown, bcUp = boundaryConditions(definition, tj, iv)
                    rhsi = I - mujb * rhs
                    #   applyPostBoundaryCondition(Up(), iv, bcUp, Si, MY0Matrix, MY0Matrix,  mujb, advectionCoeffUp, sinkCoeffUp)
                    applyPreBoundaryCondition(Down(), iv, bcDown, Si, v1Matrix, rhsi, mujb, advectionCoeffDown, sinkCoeffDown) #if Dirichlet on grid, just do it, otherwise use Ghost point. rhs vs lhs?
                    applyPreBoundaryCondition(Up(), iv, bcUp, Si, v1Matrix, rhsi, mujb, advectionCoeffUp, sinkCoeffUp)
                    #Y2 += rhsi*Y1 - Y1
                    mul!(Y2, rhsi, Y1)
                    #  applyPostBoundaryCondition(Up(), iv, bcUp, Si, v2Matrix, v2Matrix,  mujb, advectionCoeffUp, sinkCoeffUp)
                    @. Y2 += (muj - 1) * Y1 + nuj * Y0 + (1 - nuj - muj) * v + gammajb * MY0 # + mujb*MYjm
                    applyPostBoundaryCondition(Up(), iv, bcUp, Si, v2Matrix, v2Matrix, mujb, advectionCoeffUp, sinkCoeffUp)
                    #   explicitStep(mujb, rhs, Y1, Y2, Y2)
                    #TODO enforceLowerBound(Yj, lowerBound, M)
                end
                v0Matrix, v1Matrix = v1Matrix, v0Matrix
                v1Matrix, v2Matrix = v2Matrix, v1Matrix
            end
            vMatrix .= v1Matrix
            for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
                evaluate(definition, payoff, Si, iv)
                if isLBActive
                    lowerBound!(payoff, vLowerBound)
                end
            end
        end
        tip = ti
        if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
            #jump and interpolate        
            for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
                # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
                pp = QuadraticLagrangePP(Si, copy(v))
                if dividends[currentDivIndex].dividend.isProportional
                    @. v = pp(Si * (1 - dividends[currentDivIndex].dividend.amount))
                else
                    if dividendPolicy == Shift
                        @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
                    elseif dividendPolicy == Survivor
                        @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
                    else #liquidator
                        v1 = @view v1Matrix[:, iv]
                        @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                        evaluateSorted!(pp, v, v1)
                        # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
                    end
                end
            end
            currentDivIndex -= 1

        end
    end
    #PPInterpolation.computePP(pp,Si, @view(vMatrix[:,end]), PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())
    #return pp
    return QuadraticLagrangePP(Si, vMatrix[:, 1])
end

function applyPreBoundaryCondition(way::Union{Up,Down}, colIndex::Int, bcDown::LinearBoundaryCondition, S::AbstractArray{TS}, v0Matrix, rhs, dt, advectionCoeff, sinkCoeff) where {TS}
    applyBoundaryCondition(way, colIndex, bcDown, S, v0Matrix, rhs, dt, advectionCoeff, sinkCoeff)
end

function applyPreBoundaryCondition(way::Union{Up,Down}, colIndex::Int, bcDown::Union{DirichletBoundaryCondition,DirichletPayoffBoundaryCondition}, S::AbstractArray{TS}, v0Matrix, rhs, dt, advectionCoeff, sinkCoeff) where {TS}
    #   applyBoundaryCondition(way,colIndex,bcDown,S,v0Matrix, rhs,dt, advectionCoeff,sinkCoeff)
end

function applyPostBoundaryCondition(way::Union{Up,Down}, colIndex::Int, bcDown::LinearBoundaryCondition, Si::AbstractArray{TS}, v0Matrix, v1Matrix, dt, advectionCoeff, sinkCoeff; interpolation="Quadratic") where {TS}
end
function applyPostBoundaryCondition(way::Union{Up,Down}, colIndex::Int, bcDown::Union{DirichletBoundaryCondition,DirichletPayoffBoundaryCondition}, Si::AbstractArray{TS}, v1Matrix, v0Matrix, dt, advectionCoeff, sinkCoeff; interpolation="Linear") where {TS}
    rowIndex = searchsortedfirst(Si, bcDown.level)
    for i = rowIndex+1:length(Si)-1
        v1Matrix[i, colIndex] = getBoundaryValue(bcDown, v0Matrix, i)
    end
    if abs(bcDown.level - Si[rowIndex]) <= sqrt(eps(TS))
        applyBoundaryConditionOnGrid(way, rowIndex, colIndex, bcDown, v0Matrix, rhs)
        # println(" OnGrid ",v0Matrix[rowIndex, colIndex], " ", v0Matrix[rowIndex, colIndex]," ",v0Matrix[rowIndex-1, colIndex], " ", v0Matrix[rowIndex-2, colIndex])
    else
        #interpolate 
        Ui = bcDown.level
        Sip = Si
        iU = rowIndex # S[iU-1] < level < S[iU]
        value = interpolateBoundaryValue(v0Matrix, Sip, iU, Ui, bcDown; interpolation=interpolation)
        if interpolation == "Quadratic"
            divisor = ((Ui - Sip[iU-1]) * (Ui - Sip[iU-2]) / ((Sip[iU] - Sip[iU-1]) * (Sip[iU] - Sip[iU-2])))
            v1Matrix[iU, colIndex] = (value - (Ui - Sip[iU-1]) * (Ui - Sip[iU]) / ((Sip[iU-2] - Sip[iU-1]) * (Sip[iU-2] - Sip[iU])) * v1Matrix[iU-2, colIndex] - (Ui - Sip[iU]) * (Ui - Sip[iU-2]) / ((Sip[iU-1] - Sip[iU]) * (Sip[iU-1] - Sip[iU-2])) * v1Matrix[iU-1, colIndex]) / divisor
        else
            v1Matrix[iU, colIndex] = ((Sip[iU] - Sip[iU-1]) * value - (Sip[iU] - Ui) * v1Matrix[iU-1, colIndex]) / (Ui - Sip[iU-1])
        end
        println(" Quadratic ", value, " ", v1Matrix[iU, colIndex], " ", v0Matrix[iU-1, colIndex], " ", v0Matrix[iU-2, colIndex])
        # println("interpolated value ", value, " ", v0Matrix[iU, colIndex], " ", Ui, " ", v0Matrix[iU-1, colIndex], " ", v0Matrix[iU-2, colIndex])
    end
end

function priceRKL2S(definition::StructureDefinition,
    spot::T,
    model,
    dividends::AbstractArray{CapitalizedDividend{T}};
    M=400, N=100, ndev=4, Smax=zero(T), Smin=zero(T), dividendPolicy::DividendPolicy=Liquidator, grid::Grid=UniformGrid(false), varianceConditioner::PecletConditioner=NoConditioner(), calibration=NoCalibration(), useSqrt=false) where {T}
    obsTimes = observationTimes(definition)
    τ = last(obsTimes)
    t = collect(range(τ, stop=zero(T), length=N))
    if useSqrt
        du = sqrt(τ) / (N - 1)
        for i = 1:N-1
            ui = du * (i - 1)
            t[i] = τ - ui^2
        end
    end

    dividends = filter(x -> x.dividend.exDate <= τ, dividends)
    sort!(dividends, by=x -> x.dividend.exDate)
    divDates = [x.dividend.exDate for x in dividends]
    t = vcat(t, divDates, obsTimes)
    sort!(t, order=Base.Order.Reverse)
    specialPoints = sort(nonSmoothPoints(definition))
    if isempty(specialPoints)
        append!(specialPoints, spot)
    end
    xi = (range(zero(T), stop=one(T), length=M))
    rawForward = forward(model, spot, τ)
    Ui = if Smax == zero(T) || isnan(Smax)
        v0 = varianceByLogmoneyness(model, 0.0, τ) * τ
        rawForward * exp(ndev * sqrt(v0) - 0.5 * v0)
    else
        Smax
    end
    Li = if Smin < zero(T) || isnan(Smin)
        rawForward^2 / Ui
    else
        Smin
    end
    if !isempty(specialPoints)
        Ui = max(Ui, maximum(specialPoints) * 1.01)
        Li = min(Li, minimum(specialPoints) * 0.99)
    end
    isMiddle = zeros(Bool, length(specialPoints))
    Si = makeArray(grid, xi, Li, Ui, specialPoints, isMiddle)
    #  println("S ",Si)   
    #     println("t ",t)
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
    # println("initial payoff ",vMatrix[:,1], " ",vMatrix[:,2])
    Jhi = @. (Si[2:end] - Si[1:end-1])
    rhsd = zeros(T, length(Si))
    lhsd = ones(T, length(Si))
    rhsdl = zeros(T, length(Si) - 1)
    lhsdl = zeros(T, length(Si) - 1)
    rhsdu = zeros(T, length(Si) - 1)
    lhsdu = zeros(T, length(Si) - 1)
    lhs = Tridiagonal(lhsdl, lhsd, lhsdu)
    rhs = Tridiagonal(rhsdl, rhsd, rhsdu)
    v0Matrix = similar(vMatrix)
    v1Matrix = similar(vMatrix)
    v2Matrix = similar(vMatrix)
    #pp = PPInterpolation.PP(3, T, T, length(Si))
    currentDivIndex = length(dividends)
    if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
        #jump and interpolate        
        for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
            # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
            pp = QuadraticLagrangePP(Si, copy(v))
            if dividends[currentDivIndex].dividend.isProportional
                @. v = pp(Si * (1 - dividends[currentDivIndex].dividend.amount))
            else
                if dividendPolicy == Shift
                    @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
                elseif dividendPolicy == Survivor
                    @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
                else #liquidator
                    v1 = @view v1Matrix[:, iv]
                    @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                    evaluateSorted!(pp, v, v1)
                    # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
                end
            end
        end
        currentDivIndex -= 1
    end

    s = 0.0
    local a, b, w0, w1

    for i = 2:length(t)
        ti = t[i]
        dt = tip - ti
        if dt > 1e-8
            dfi = discountFactor(model, ti)
            dfip = discountFactor(model, tip)
            ri = calibrateRate(calibration, 1.0, dt, dfi, dfip)
            driftDfi = 1 / forward(model, 1.0, ti)
            driftDfip = 1 / forward(model, 1.0, tip)
            μi = calibrateDrift(calibration, 1.0, dt, dfi, dfip, driftDfi, driftDfip, ri)
            σi2 = (varianceByLogmoneyness(model, 0.0, tip) * tip - varianceByLogmoneyness(model, 0.0, ti) * ti) / (tip - ti)

            @inbounds for j = 2:M-1
                s2S = σi2 * Si[j]^2
                muS = μi * Si[j]
                s2S = conditionedVariance(varianceConditioner, s2S, muS, Si[j], Jhi[j-1], Jhi[j])
                rhsd[j] = ((muS * (Jhi[j-1] - Jhi[j]) + s2S) / (Jhi[j] * Jhi[j-1]) + ri)
                rhsdu[j] = -(s2S + muS * Jhi[j-1]) / (Jhi[j] * (Jhi[j] + Jhi[j-1]))
                rhsdl[j-1] = -(s2S - muS * Jhi[j]) / (Jhi[j-1] * (Jhi[j] + Jhi[j-1]))
            end
            #linear or Ke-rt same thing
            if s == 0.0
                s, a, b, w0, w1 = initRKLCoeffs(dt, rhsd, epsilonRKL=0.0, rklStages=0)
            end
            v0Matrix[1:end, 1:end] = vMatrix
            v1Matrix[1:end, 1:end] = vMatrix
            v2Matrix[1:end, 1:end] = vMatrix

            advectionCoeffDown = μi * Si[1] / Jhi[1]
            sinkCoeffDown = ri
            advectionCoeffUp = -μi * Si[end] / Jhi[end]
            sinkCoeffUp = ri
            advance(definition, payoff, tip - dt)
            #for each stage, iterate of payoff variables.
            mu1b = b[1] * w1 * dt
            for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix))) #we don't go reverse here as we want to use initial vanilla price in KO for explicit step.
                bcDown, bcUp = boundaryConditions(definition, tip - mu1b, iv)
                rhsi = I - mu1b * rhs #TODO keep a fixed list of lhs,rhs.
                applyBoundaryCondition(Down(), iv, bcDown, Si, vMatrix, rhsi, mu1b, advectionCoeffDown, sinkCoeffDown) #if Dirichlet on grid, just do it, otherwise use Ghost point. rhs vs lhs?
                applyBoundaryCondition(Up(), iv, bcUp, Si, vMatrix, rhsi, mu1b, advectionCoeffUp, sinkCoeffUp)
                Y1 = @view v1Matrix[:, iv]
                mul!(Y1, rhsi, v)
                # println(iv, " explicitStep", mu1b, " ", v, "\n ", Y1)
                MY0 = (Y1 - v) / mu1b
                #TODO  enforceLowerBound(Yjm, lowerBound, M)
                Y0 = @view v0Matrix[:, iv]
                Y1 = @view v1Matrix[:, iv]
                Y2 = @view v2Matrix[:, iv]
                Y0 .= v
                for j = 2:s
                    tj = tip - dt * (j^2 + j - 2) / (s^2 + s - 2)
                    muu = (2 * j - 1) * b[j] / (j * b[j-1])
                    muj = muu * w0
                    mujb = muu * w1 * dt
                    gammajb = -a[j-1] * mujb
                    nuj = -1.0 * b[2] / (2.0 * b[1]) #b0 = b[1]
                    if j > 2
                        nuj = -(j - 1) * b[j] / (j * b[j-2])
                    end
                    bcDown, bcUp = boundaryConditions(definition, tj, iv)
                    rhsi = I - mujb * rhs
                    applyBoundaryCondition(Down(), iv, bcDown, Si, v1Matrix, rhsi, mujb, advectionCoeffDown, sinkCoeffDown) #if Dirichlet on grid, just do it, otherwise use Ghost point. rhs vs lhs?
                    applyBoundaryCondition(Up(), iv, bcUp, Si, v1Matrix, rhsi, mujb, advectionCoeffUp, sinkCoeffUp)
                    #Y2 += rhsi*Y1 - Y1
                    mul!(Y2, rhsi, Y1)
                    @. Y2 += (muj - 1) * Y1 + nuj * Y0 + (1 - nuj - muj) * v + gammajb * MY0 # + mujb*MYjm
                    #   explicitStep(mujb, rhs, Y1, Y2, Y2)
                    #TODO enforceLowerBound(Yj, lowerBound, M)
                    Y0, Y1 = Y1, Y0
                    Y1, Y2 = Y2, Y1
                end
            end
            vMatrix .= v1Matrix
            for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
                evaluate(definition, payoff, Si, iv)
                if isLBActive
                    lowerBound!(payoff, vLowerBound)
                end
            end
        end
        tip = ti
        if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
            #jump and interpolate        
            for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
                # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
                pp = QuadraticLagrangePP(Si, copy(v))
                if dividends[currentDivIndex].dividend.isProportional
                    @. v = pp(Si * (1 - dividends[currentDivIndex].dividend.amount))
                else
                    if dividendPolicy == Shift
                        @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
                    elseif dividendPolicy == Survivor
                        @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
                    else #liquidator
                        v1 = @view v1Matrix[:, iv]
                        @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                        evaluateSorted!(pp, v, v1)
                        # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
                    end
                end
            end
            currentDivIndex -= 1

        end
    end
    #PPInterpolation.computePP(pp,Si, @view(vMatrix[:,end]), PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())
    #return pp
    return QuadraticLagrangePP(Si, vMatrix[:, 1])
end


function explicitStep(a, rhs::Tridiagonal, F::AbstractArray{T}, Y0::AbstractArray{T}, Y1::AbstractArray{T}) where {T}
    mul!(Y1, rhs, F)
    @. Y1 = Y0 - a * Y1
end



function initRKLCoeffs(dt, A1ij; epsilonRKL=0.0, rklStages=0)
    dtexplicit = 1.0 / max(maximum(A1ij))
    dtexplicit = 2.442287776486202e-5 
    dtexplicit /= 1.1 #lambdaS        
    s = 0.0
    delta = 1 + 4 * (2 + 4 * dt / dtexplicit)
    s = ceil(Int, (-1 + sqrt(delta)) / 2)
    # s *= 2 #seems necessary for KI Ghost
    if s % 2 == 0
        s += 1
    end
    if epsilonRKL > 0
        s = computeRKLStages(dtexplicit, dt, epsilonRKL)
    end
    if rklStages > 0
        s = rklStages
    end
    println("s=", s, " dtexplicit=", dtexplicit)
    a = zeros(s)
    b = zeros(s)
    w0 = 1.0
    w1 = 0.0
    if epsilonRKL == 0
        w1 = 4 / (s^2 + s - 2)
        b[1] = 1.0 / 3
        b[2] = 1.0 / 3
        a[1] = 1.0 - b[1]
        a[2] = 1.0 - b[2]
        for i = 3:s
            b[i] = (i^2 + i - 2.0) / (2 * i * (i + 1.0))
            a[i] = 1.0 - b[i]
        end
    else
        w0 = 1 + epsilonRKL / s^2
        _, tw0p, tw0p2 = legPoly(s, w0)
        w1 = tw0p / tw0p2
        b = zeros(s)
        for jj = 2:s
            _, tw0p, tw0p2 = legPoly(jj, w0)
            b[jj] = tw0p2 / tw0p^2
        end
        b[1] = b[2]
        a = zeros(s)
        for jj = 2:s
            tw0, _, _ = legPoly(jj - 1, w0)
            a[jj-1] = (1 - b[jj-1] * tw0)
        end
    end
    return s, a, b, w0, w1
end

function initRKGCoeffs(dt, rhs; stages=0)
    maxnorm = maximum(abs.(rhs.d[2:end-1]) + abs.(rhs.dl[1:end-1] + abs.(rhs.du[2:end])))
    maxnorm = maximum((maxnorm, abs(rhs.d[1]) + abs(rhs.du[1]), abs(rhs.d[end]) + abs(rhs.dl[end-1])))
    # maxnorm =  61865/2
    dtexplicit = 1.0 / maxnorm
    # dtexplicit = 2.442287776486202e-5 
    dtexplicit /= 1.1 #lambdaS        
    α = 1.5
    #taumax = dtexpl/w1
    allS = quadRootsReal(1.0, 2α, -2α - 1 - dt / dtexplicit / (3 + 2α))
    # println("allS ", allS)
    s = if allS[1] < 0
        ceil(Int, allS[2])
    else
        ceil(Int, allS[1])
    end
    if s > 1e5
        println("d ", rhs.d)
        println("l ", rhs.dl)
        throw(DomainError(s))
    end
    w1 = (3 + 2α) / ((s + 2α + 1) * (s - 1))
    while w1 > dtexplicit / dt
        s += 1
        w1 = (3 + 2α) / ((s + 2α + 1) * (s - 1))
    end
    s *= 1   #seems necessary for KI Ghost
    # if s % 2 == 0
    #     s += 1
    # end
    # s = computeRKGStages(dtexplicit, dt)
    if stages > 0
        s = stages
    end
    a = zeros(s + 1)
    b = zeros(s + 1)
    w0 = 1.0
    w1 = (3 + 2α) / ((s + 2α + 1) * (s - 1))
    # println("s=", s, " dtexplicit=", dtexplicit, " w1 ", 1 / w1, " ", dt / dtexplicit)
    b[1] = 1.0
    b[2] = 1.0 / 3
    a[1] = 1.0 - b[1]
    a[2] = 1.0 - 3 * b[2]
    for i = 2:s
        b[i+1] = 4 * (i - 1) * (i + 4) / (3 * i * (i + 3) * (i + 1) * (i + 2))
        a[i+1] = 1.0 - b[i+1] * (i + 1) * (i + 2) / 2
    end

    return s, a, b, w0, w1
end


function priceRKG2(definition::StructureDefinition,
    spot::T,
    model,
    dividends::AbstractArray{CapitalizedDividend{T}};
    M=400, N=100, ndev=4, Smax=zero(T), Smin=zero(T), dividendPolicy::DividendPolicy=Liquidator, grid::Grid=UniformGrid(false), varianceConditioner::PecletConditioner=NoConditioner(), calibration=NoCalibration(), useSqrt=false,
    useTrace=false,trace=Trace(Vector{T}(),Vector{Vector{T}}())) where {T}
    obsTimes = observationTimes(definition)
    τ = last(obsTimes)
    t = collect(range(τ, stop=zero(T), length=N))
    if useSqrt
        du = sqrt(τ) / (N - 1)
        for i = 1:N-1
            ui = du * (i - 1)
            t[i] = τ - ui^2
        end
    end

    dividends = filter(x -> x.dividend.exDate <= τ, dividends)
    sort!(dividends, by=x -> x.dividend.exDate)
    divDates = [x.dividend.exDate for x in dividends]
    t = vcat(t, divDates, obsTimes)
    sort!(t, order=Base.Order.Reverse)
    specialPoints = sort(nonSmoothPoints(definition))
    if isempty(specialPoints)
        append!(specialPoints, spot)
    end
    xi = (range(zero(T), stop=one(T), length=M))
    rawForward = forward(model, spot, τ)
    Ui = if Smax == zero(T) || isnan(Smax)
        v0 = varianceByLogmoneyness(model, 0.0, τ) * τ
        rawForward * exp(ndev * sqrt(v0) - 0.5 * v0)
    else
        Smax
    end
    Li = if Smin < zero(T) || isnan(Smin)
        rawForward^2 / Ui
    else
        Smin
    end
    if !isempty(specialPoints)
        Ui = max(Ui, maximum(specialPoints))
        Li = min(Li, minimum(specialPoints))
    end
    isMiddle = zeros(Bool, length(specialPoints))
    Si = makeArray(grid, xi, Li, Ui, specialPoints, isMiddle)
    # println("S ", Si)
    #     println("t ",t)
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
    # println("initial payoff ",vMatrix[:,1], " ",vMatrix[:,2])
    Jhi = @. (Si[2:end] - Si[1:end-1])
    rhsd = zeros(T, length(Si))
    rhsdl = zeros(T, length(Si) - 1)
    rhsdu = zeros(T, length(Si) - 1)
    rhs = Tridiagonal(rhsdl, rhsd, rhsdu)
    tri = Tridiagonal(zeros(T, length(rhsdl)), zeros(T, length(rhsd)), zeros(length(rhsdu)))
   
    v0Matrix = similar(vMatrix)
    v1Matrix = similar(vMatrix)
    v2Matrix = similar(vMatrix)
    #pp = PPInterpolation.PP(3, T, T, length(Si))
    currentDivIndex = length(dividends)
    if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
        #jump and interpolate        
        for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
            # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
            pp = QuadraticLagrangePP(Si, copy(v))
            if dividends[currentDivIndex].dividend.isProportional
                @. v = pp(Si * (1 - dividends[currentDivIndex].dividend.amount))
            else
                if dividendPolicy == Shift
                    @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
                elseif dividendPolicy == Survivor
                    @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
                else #liquidator
                    v1 = @view v1Matrix[:, iv]
                    @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                    evaluateSorted!(pp, v, v1)
                    # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
                end
            end
        end
        currentDivIndex -= 1
    end

    s = 0.0
    local a, b, w0, w1
    α = 1.5
    for i = 2:length(t)
        ti = t[i]
        dt = tip - ti
        if dt > 1e-8
            dfi = discountFactor(model, ti)
            dfip = discountFactor(model, tip)
            ri = calibrateRate(calibration, 1.0, dt, dfi, dfip)
            driftDfi = 1 / forward(model, 1.0, ti)
            driftDfip = 1 / forward(model, 1.0, tip)
            μi = calibrateDrift(calibration, 1.0, dt, dfi, dfip, driftDfi, driftDfip, ri)
            σi2 = (varianceByLogmoneyness(model, 0.0, tip) * tip - varianceByLogmoneyness(model, 0.0, ti) * ti) / (tip - ti)

            @inbounds for j = 2:M-1
                s2S = σi2 * Si[j]^2
                muS = μi * Si[j]
                s2S = conditionedVariance(varianceConditioner, s2S, muS, Si[j], Jhi[j-1], Jhi[j])
                rhsd[j] = ((muS * (Jhi[j-1] - Jhi[j]) + s2S) / (Jhi[j] * Jhi[j-1]) + ri)
                rhsdu[j] = -(s2S + muS * Jhi[j-1]) / (Jhi[j] * (Jhi[j] + Jhi[j-1]))
                rhsdl[j-1] = -(s2S - muS * Jhi[j]) / (Jhi[j-1] * (Jhi[j] + Jhi[j-1]))
            end
            #linear or Ke-rt same thing
            advectionCoeffDown = μi * Si[1] / Jhi[1]
            sinkCoeffDown = ri
            advectionCoeffUp = -μi * Si[end] / Jhi[end]
            sinkCoeffUp = ri
            if s == 0.0
                s, a, b, w0, w1 = initRKGCoeffs(dt, rhs, stages=0)
            end
            #  v0Matrix[1:end, 1:end] = vMatrix
            advance(definition, payoff, tip - dt)
            #for each stage, iterate of payoff variables.
            mu1b = 3 * b[2] / b[1] * w1 * dt
            # println(" EIGENV0 ", minimum(eigen(rhs).values), " ", maximum(eigen(rhs).values))
            # print(ti, " ", 1, " ", tip - mu1b)
            # tempMatrix = copy(vMatrix)
            for (iv, v) in (enumerate(eachcol(vMatrix))) #we don't go reverse here as we want to use initial vanilla price in KO for explicit step.
                bcDown, bcUp = boundaryConditions(definition, tip - mu1b, iv)
                @. tri.d = one(T) - mu1b*rhsd
                    @. tri.dl = -mu1b*rhsdl
                    @. tri.du = -mu1b*rhsdu
                 
                applyBoundaryCondition(Down(), iv, bcDown, Si, vMatrix, tri, mu1b, advectionCoeffDown, sinkCoeffDown) #if Dirichlet on grid, just do it, otherwise use Ghost point. rhs vs lhs?
                applyBoundaryCondition(Up(), iv, bcUp, Si, vMatrix, tri, mu1b, advectionCoeffUp, sinkCoeffUp)
                 if ti >= τ - dt
                # println("eig=",eigen(rhs).values)
                                    # println(iv, " ", mu1b, " EIGENV ", (eigen(-I + rhsi).values), " ", maximum(eigen(-I + rhsi).values))
                end
                Y1 = @view v1Matrix[:, iv]
                mul!(Y1, tri, v)
                if isLBActive
                    @. Y1 = max(vLowerBound, Y1)
                end
                #alternative: update Y1 bc after solving, based on v for vanilla.
                #    println(iv, " explicitStep", mu1b, " ", v, "\n ", Y1)
            end
            MY0Matrix = (v1Matrix - vMatrix) / mu1b
            #TODO  enforceLowerBound(Yjm, lowerBound, M)
            v0Matrix .= vMatrix
            for j = 2:s
                #b and a indices are shifted by one compared to paper.
                tj = tip - dt * (j + 2α + 1) * (j - 1) / ((s + 2α + 1) * (s - 1))
                muu = (2 * j + 1) * b[j+1] / (j * b[j])
                muj = muu * w0
                mujb = muu * w1 * dt
                gammajb = -a[j] * mujb
                nuj = -(j + 1) * b[j+1] / (j * b[j-1])

                # print(ti, " ", j, " ", tj)
                # tempMatrix = copy(v1Matrix)
                for (iv, v) in (enumerate(eachcol(vMatrix))) #we don't go reverse here as we want to use initial 
                    MY0 = @view MY0Matrix[:, iv]
                    Y0 = @view v0Matrix[:, iv]
                    Y1 = @view v1Matrix[:, iv]
                    Y2 = @view v2Matrix[:, iv]
                    bcDown, bcUp = boundaryConditions(definition, tj, iv)
                    @. tri.d = one(T) - mujb*rhsd
                    @. tri.dl = -mujb*rhsdl
                    @. tri.du = -mujb*rhsdu
                    # rhsi = I - mujb * rhs
                    applyBoundaryCondition(Down(), iv, bcDown, Si, v1Matrix, tri, mujb, advectionCoeffDown, sinkCoeffDown) #if Dirichlet on grid, just do it, otherwise use Ghost point. rhs vs lhs?
                    applyBoundaryCondition(Up(), iv, bcUp, Si, v1Matrix, tri, mujb, advectionCoeffUp, sinkCoeffUp)
                    #Y2 += rhsi*Y1 - Y1
                    mul!(Y2, tri, Y1)#tempMatrix[:,iv])
                    @. Y2 += (muj - 1) * Y1 + nuj * Y0 + (1 - nuj - muj) * v + gammajb * MY0 # + mujb*MYjm
                    #   explicitStep(mujb, rhs, Y1, Y2, Y2)
                    if isLBActive
                        @. Y2 = max(vLowerBound, Y2)
                    end
                end
                v0Matrix, v1Matrix = v1Matrix, v0Matrix
                v1Matrix, v2Matrix = v2Matrix, v1Matrix
            end
            vMatrix .= v1Matrix
            for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
                evaluate(definition, payoff, Si, iv)
                if isLBActive
                    lowerBound!(payoff, vLowerBound)
                end
            end
        end
        tip = ti
        if ti > 0.9
            if useTrace
                push!(trace.times,ti)
                push!(trace.values,vMatrix[:,1])
            end
        end
        if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
            #jump and interpolate        
            for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
                # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
                pp = QuadraticLagrangePP(Si, copy(v))
                if dividends[currentDivIndex].dividend.isProportional
                    @. v = pp(Si * (1 - dividends[currentDivIndex].dividend.amount))
                else
                    if dividendPolicy == Shift
                        @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
                    elseif dividendPolicy == Survivor
                        @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
                    else #liquidator
                        v1 = @view v1Matrix[:, iv]
                        @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                        evaluateSorted!(pp, v, v1)
                        # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
                    end
                end
            end
            currentDivIndex -= 1

        end
    end
    #PPInterpolation.computePP(pp,Si, @view(vMatrix[:,end]), PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())
    #return pp
    return QuadraticLagrangePP(Si, vMatrix[:, 1])
end

function priceRKL2(definition::StructureDefinition,
    spot::T,
    model,
    dividends::AbstractArray{CapitalizedDividend{T}};
    M=400, N=100, ndev=4, Smax=zero(T), Smin=zero(T), dividendPolicy::DividendPolicy=Liquidator, grid::Grid=UniformGrid(false), varianceConditioner::PecletConditioner=NoConditioner(), calibration=NoCalibration(), useSqrt=false,
    useTrace=false,trace=Trace(Vector{T}(),Vector{Vector{T}}())) where {T}
    obsTimes = observationTimes(definition)
    τ = last(obsTimes)
    t = collect(range(τ, stop=zero(T), length=N))
    if useSqrt
        du = sqrt(τ) / (N - 1)
        for i = 1:N-1
            ui = du * (i - 1)
            t[i] = τ - ui^2
        end
    end

    dividends = filter(x -> x.dividend.exDate <= τ, dividends)
    sort!(dividends, by=x -> x.dividend.exDate)
    divDates = [x.dividend.exDate for x in dividends]
    t = vcat(t, divDates, obsTimes)
    sort!(t, order=Base.Order.Reverse)
    specialPoints = sort(nonSmoothPoints(definition))
    if isempty(specialPoints)
        append!(specialPoints, spot)
    end
    xi = (range(zero(T), stop=one(T), length=M))
    rawForward = forward(model, spot, τ)
    Ui = if Smax == zero(T) || isnan(Smax)
        v0 = varianceByLogmoneyness(model, 0.0, τ) * τ
        rawForward * exp(ndev * sqrt(v0) - 0.5 * v0)
    else
        Smax
    end
    Li = if Smin < zero(T) || isnan(Smin)
        rawForward^2 / Ui
    else
        Smin
    end
    if !isempty(specialPoints)
        Ui = max(Ui, maximum(specialPoints) * 1.01)
        Li = min(Li, minimum(specialPoints) * 0.99)
    end
    isMiddle = zeros(Bool, length(specialPoints))
    Si = makeArray(grid, xi, Li, Ui, specialPoints, isMiddle)
    #  println("S ",Si)   
    #     println("t ",t)
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
    # println("initial payoff ",vMatrix[:,1], " ",vMatrix[:,2])
    Jhi = @. (Si[2:end] - Si[1:end-1])
    rhsd = zeros(T, length(Si))
    lhsd = ones(T, length(Si))
    rhsdl = zeros(T, length(Si) - 1)
    lhsdl = zeros(T, length(Si) - 1)
    rhsdu = zeros(T, length(Si) - 1)
    lhsdu = zeros(T, length(Si) - 1)
    lhs = Tridiagonal(lhsdl, lhsd, lhsdu)
    rhs = Tridiagonal(rhsdl, rhsd, rhsdu)
    v0Matrix = similar(vMatrix)
    v1Matrix = similar(vMatrix)
    v2Matrix = similar(vMatrix)
    #pp = PPInterpolation.PP(3, T, T, length(Si))
    currentDivIndex = length(dividends)
    if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
        #jump and interpolate        
        for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
            # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
            pp = QuadraticLagrangePP(Si, copy(v))
            if dividends[currentDivIndex].dividend.isProportional
                @. v = pp(Si * (1 - dividends[currentDivIndex].dividend.amount))
            else
                if dividendPolicy == Shift
                    @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
                elseif dividendPolicy == Survivor
                    @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
                else #liquidator
                    v1 = @view v1Matrix[:, iv]
                    @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                    evaluateSorted!(pp, v, v1)
                    # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
                end
            end
        end
        currentDivIndex -= 1
    end

    s = 0.0
    local a, b, w0, w1

    for i = 2:length(t)
        ti = t[i]
        dt = tip - ti
        if dt > 1e-8
            dfi = discountFactor(model, ti)
            dfip = discountFactor(model, tip)
            ri = calibrateRate(calibration, 1.0, dt, dfi, dfip)
            driftDfi = 1 / forward(model, 1.0, ti)
            driftDfip = 1 / forward(model, 1.0, tip)
            μi = calibrateDrift(calibration, 1.0, dt, dfi, dfip, driftDfi, driftDfip, ri)
            σi2 = (varianceByLogmoneyness(model, 0.0, tip) * tip - varianceByLogmoneyness(model, 0.0, ti) * ti) / (tip - ti)

            @inbounds for j = 2:M-1
                s2S = σi2 * Si[j]^2
                muS = μi * Si[j]
                s2S = conditionedVariance(varianceConditioner, s2S, muS, Si[j], Jhi[j-1], Jhi[j])
                rhsd[j] = ((muS * (Jhi[j-1] - Jhi[j]) + s2S) / (Jhi[j] * Jhi[j-1]) + ri)
                rhsdu[j] = -(s2S + muS * Jhi[j-1]) / (Jhi[j] * (Jhi[j] + Jhi[j-1]))
                rhsdl[j-1] = -(s2S - muS * Jhi[j]) / (Jhi[j-1] * (Jhi[j] + Jhi[j-1]))
            end
            #linear or Ke-rt same thing
            if s == 0.0
                s, a, b, w0, w1 = initRKLCoeffs(dt, rhsd, epsilonRKL=0.0, rklStages=0)
            end
            #  v0Matrix[1:end, 1:end] = vMatrix
            advectionCoeffDown = μi * Si[1] / Jhi[1]
            sinkCoeffDown = ri
            advectionCoeffUp = -μi * Si[end] / Jhi[end]
            sinkCoeffUp = ri
            advance(definition, payoff, tip - dt)
            #for each stage, iterate of payoff variables.
            mu1b = b[1] * w1 * dt
            for (iv, v) in (enumerate(eachcol(vMatrix))) #we don't go reverse here as we want to use initial vanilla price in KO for explicit step.
                bcDown, bcUp = boundaryConditions(definition, tip - mu1b, iv)
                rhsi = I - mu1b * rhs #TODO keep a fixed list of lhs,rhs.
                applyBoundaryCondition(Down(), iv, bcDown, Si, vMatrix, rhsi, mu1b, advectionCoeffDown, sinkCoeffDown) #if Dirichlet on grid, just do it, otherwise use Ghost point. rhs vs lhs?
                applyBoundaryCondition(Up(), iv, bcUp, Si, vMatrix, rhsi, mu1b, advectionCoeffUp, sinkCoeffUp)
                Y1 = @view v1Matrix[:, iv]
                mul!(Y1, rhsi, v)
                #    println(iv, " explicitStep", mu1b, " ", v, "\n ", Y1)
            end
            MY0Matrix = (v1Matrix - vMatrix) / mu1b
            #TODO  enforceLowerBound(Yjm, lowerBound, M)
            v0Matrix .= vMatrix
            for j = 2:s
                tj = tip - dt * (j^2 + j - 2) / (s^2 + s - 2)
                muu = (2 * j - 1) * b[j] / (j * b[j-1])
                muj = muu * w0
                mujb = muu * w1 * dt
                gammajb = -a[j-1] * mujb
                nuj = -1.0 * b[2] / (2.0 * b[1]) #b0 = b[1]
                if j > 2
                    nuj = -(j - 1) * b[j] / (j * b[j-2])
                end
                for (iv, v) in (enumerate(eachcol(vMatrix))) #we don't go reverse here as we want to use initial 
                    MY0 = @view MY0Matrix[:, iv]
                    Y0 = @view v0Matrix[:, iv]
                    Y1 = @view v1Matrix[:, iv]
                    Y2 = @view v2Matrix[:, iv]
                    bcDown, bcUp = boundaryConditions(definition, tj, iv)
                    rhsi = I - mujb * rhs
                    applyBoundaryCondition(Down(), iv, bcDown, Si, v1Matrix, rhsi, mujb, advectionCoeffDown, sinkCoeffDown) #if Dirichlet on grid, just do it, otherwise use Ghost point. rhs vs lhs?
                    applyBoundaryCondition(Up(), iv, bcUp, Si, v1Matrix, rhsi, mujb, advectionCoeffUp, sinkCoeffUp)
                    #Y2 += rhsi*Y1 - Y1
                    mul!(Y2, rhsi, Y1)
                    @. Y2 += (muj - 1) * Y1 + nuj * Y0 + (1 - nuj - muj) * v + gammajb * MY0 # + mujb*MYjm
                    #   explicitStep(mujb, rhs, Y1, Y2, Y2)
                    #TODO enforceLowerBound(Yj, lowerBound, M)
                end
                v0Matrix, v1Matrix = v1Matrix, v0Matrix
                v1Matrix, v2Matrix = v2Matrix, v1Matrix
            end
            vMatrix .= v1Matrix
            for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
                evaluate(definition, payoff, Si, iv)
                if isLBActive
                    lowerBound!(payoff, vLowerBound)
                end
            end
        end
        tip = ti
        if ti > 0.9
            if useTrace
                push!(trace.times,ti)
                push!(trace.values,vMatrix[:,1])
            end
        end
      
        if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
            #jump and interpolate        
            for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
                # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
                pp = QuadraticLagrangePP(Si, copy(v))
                if dividends[currentDivIndex].dividend.isProportional
                    @. v = pp(Si * (1 - dividends[currentDivIndex].dividend.amount))
                else
                    if dividendPolicy == Shift
                        @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
                    elseif dividendPolicy == Survivor
                        @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
                    else #liquidator
                        v1 = @view v1Matrix[:, iv]
                        @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                        evaluateSorted!(pp, v, v1)
                        # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
                    end
                end
            end
            currentDivIndex -= 1

        end
    end
    #PPInterpolation.computePP(pp,Si, @view(vMatrix[:,end]), PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())
    #return pp
    return QuadraticLagrangePP(Si, vMatrix[:, 1])
end

function computeRKLStages(dtexplicit, dt, ep)
    s = 1
    betaFunc = function (s::Int)
        w0 = 1 + ep / s^2
        _, tw0p, tw0p2 = legPoly(s, w0)
        beta = (w0 + 1) * tw0p2 / tw0p
        return beta - 2 * dt / (dtexplicit)
    end
    while s < 10000 && betaFunc(s) < 0
        s += 1
    end
    #s += Int(ceil(s/10))
    return s
end
function legPoly(s::Int, w0::Real)
    tjm = 1.0
    tj = w0
    if s == 1
        return tj, 1.0, 0.0
    end
    dtjm = 0.0
    dtj = 1.0
    d2tjm = 0.0
    d2tj = 0.0

    for j = 2:s
        onej = 1.0 / j
        tjp = (2 - onej) * w0 * tj - (1 - onej) * tjm
        dtjp = (2 - onej) * (tj + w0 * dtj) - (1 - onej) * dtjm
        d2tjp = (2 - onej) * (dtj * 2 + w0 * d2tj) - (1 - onej) * d2tjm
        tjm = tj
        dtjm = dtj
        d2tjm = d2tj
        tj = tjp
        dtj = dtjp
        d2tj = d2tjp
    end
    return tj, dtj, d2tj
end




function priceExplicitEuler(definition::StructureDefinition,
    spot::T,
    model,
    dividends::AbstractArray{CapitalizedDividend{T}};
    M=400, N=100, ndev=4, Smax=zero(T), Smin=zero(T), dividendPolicy::DividendPolicy=Liquidator, grid::Grid=UniformGrid(false), varianceConditioner::PecletConditioner=NoConditioner(), calibration=NoCalibration(), useSqrt=false,
    useTrace=false,trace=Trace(Vector{T}(),Vector{Vector{T}}())) where {T}
    obsTimes = observationTimes(definition)
    τ = last(obsTimes)
    t = collect(range(τ, stop=zero(T), length=N))
    if useSqrt
        du = sqrt(τ) / (N - 1)
        for i = 1:N-1
            ui = du * (i - 1)
            t[i] = τ - ui^2
        end
    end

    dividends = filter(x -> x.dividend.exDate <= τ, dividends)
    sort!(dividends, by=x -> x.dividend.exDate)
    divDates = [x.dividend.exDate for x in dividends]
    t = vcat(t, divDates, obsTimes)
    sort!(t, order=Base.Order.Reverse)
    specialPoints = sort(nonSmoothPoints(definition))
    if isempty(specialPoints)
        append!(specialPoints, spot)
    end
    xi = (range(zero(T), stop=one(T), length=M))
    rawForward = forward(model, spot, τ)
    Ui = if Smax == zero(T) || isnan(Smax)
        v0 = varianceByLogmoneyness(model, 0.0, τ) * τ
        rawForward * exp(ndev * sqrt(v0) - 0.5 * v0)
    else
        Smax
    end
    Li = if Smin < zero(T) || isnan(Smin)
        rawForward^2 / Ui
    else
        Smin
    end
    if !isempty(specialPoints)
        Ui = max(Ui, maximum(specialPoints) * 1.01)
        Li = min(Li, minimum(specialPoints) * 0.99)
    end
    isMiddle = zeros(Bool, length(specialPoints))
    Si = makeArray(grid, xi, Li, Ui, specialPoints, isMiddle)
    # println("S ", Si)
    #     println("t ",t)
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
    v1Matrix = copy(vMatrix)
    # println("initial payoff ",vMatrix[:,1], " ",vMatrix[:,2])
    Jhi = @. (Si[2:end] - Si[1:end-1])
    rhsd = zeros(T, length(Si))
    rhsdl = zeros(T, length(Si) - 1)
    rhsdu = zeros(T, length(Si) - 1)
    rhs = Tridiagonal(rhsdl, rhsd, rhsdu)

    currentDivIndex = length(dividends)
    if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
        #jump and interpolate        
        for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
            # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
            pp = QuadraticLagrangePP(Si, copy(v))
            if dividends[currentDivIndex].dividend.isProportional
                @. v = pp(Si * (1 - dividends[currentDivIndex].dividend.amount))
            else
                if dividendPolicy == Shift
                    @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
                elseif dividendPolicy == Survivor
                    @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
                else #liquidator
                    v1 = @view v1Matrix[:, iv]
                    @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                    evaluateSorted!(pp, v, v1)
                    # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
                end
            end
        end
        currentDivIndex -= 1
    end

    s = 0.0
    local a, b, w0, w1
    α = 1.5
    for i = 2:length(t)
        ti = t[i]
        dt = tip - ti
        if dt > 1e-8
            dfi = discountFactor(model, ti)
            dfip = discountFactor(model, tip)
            ri = calibrateRate(calibration, 1.0, dt, dfi, dfip)
            driftDfi = 1 / forward(model, 1.0, ti)
            driftDfip = 1 / forward(model, 1.0, tip)
            μi = calibrateDrift(calibration, 1.0, dt, dfi, dfip, driftDfi, driftDfip, ri)
            σi2 = (varianceByLogmoneyness(model, 0.0, tip) * tip - varianceByLogmoneyness(model, 0.0, ti) * ti) / (tip - ti)

            @inbounds for j = 2:M-1
                s2S = σi2 * Si[j]^2
                muS = μi * Si[j]
                s2S = conditionedVariance(varianceConditioner, s2S, muS, Si[j], Jhi[j-1], Jhi[j])
                rhsd[j] = ((muS * (Jhi[j-1] - Jhi[j]) + s2S) / (Jhi[j] * Jhi[j-1]) + ri)
                rhsdu[j] = -(s2S + muS * Jhi[j-1]) / (Jhi[j] * (Jhi[j] + Jhi[j-1]))
                rhsdl[j-1] = -(s2S - muS * Jhi[j]) / (Jhi[j-1] * (Jhi[j] + Jhi[j-1]))
            end
            #linear or Ke-rt same thing
            advectionCoeffDown = μi * Si[1] / Jhi[1]
            sinkCoeffDown = ri
            advectionCoeffUp = -μi * Si[end] / Jhi[end]
            sinkCoeffUp = ri
            maxnorm = maximum(abs.(rhs.d[2:end-1]) + abs.(rhs.dl[1:end-1] + abs.(rhs.du[2:end])))
            maxnorm = maximum((maxnorm, abs(rhs.d[1]) + abs(rhs.du[1]), abs(rhs.d[end]) + abs(rhs.dl[end-1])))
            # maxnorm =  61865/2
            dtexplicit = 1.0 / maxnorm
            if dtexplicit < dt 
            # print(ti, " ", dtexplicit, " ")
            end
            advance(definition, payoff, tip - dt)
            #for each stage, iterate of payoff variables.
            # println(" EIGENV0 ", minimum(eigen(rhs).values), " ", maximum(eigen(rhs).values))
            # tempMatrix = copy(vMatrix)
            for (iv, v) in (enumerate(eachcol(vMatrix))) #we don't go reverse here as we want to use initial vanilla price in KO for explicit step.
                bcDown, bcUp = boundaryConditions(definition, tip - dt, iv)
                rhsi = I - dt * rhs
                applyBoundaryCondition(Down(), iv, bcDown, Si, vMatrix, rhsi, dt, advectionCoeffDown, sinkCoeffDown) #if Dirichlet on grid, just do it, otherwise use Ghost point. rhs vs lhs?
                applyBoundaryCondition(Up(), iv, bcUp, Si, vMatrix, rhsi, dt, advectionCoeffUp, sinkCoeffUp)
                # if ti >= τ - 2dt
                #     println(iv, "rhsdl",rhsi.dl)
                #     println(iv, "rhsd",rhsi.d)
                #     println(iv, "rhsdu",rhsi.du)
                #     println(iv, " EIGENV ", (eigen(-I + rhsi).values), " ", maximum(eigen(-I + rhsi).values))
                # end
                Y1 = @view v1Matrix[:, iv]
                mul!(Y1, rhsi, v)
                v[1:end] = Y1
            end
            for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
                evaluate(definition, payoff, Si, iv)
                if isLBActive
                    lowerBound!(payoff, vLowerBound)
                end
            end
            tip = ti
            if ti > 0.9
                if useTrace
                    push!(trace.times,ti)
                    push!(trace.values,vMatrix[:,1])
                end
            end
            if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
                #jump and interpolate        
                for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
                    # PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())       
                    pp = QuadraticLagrangePP(Si, copy(v))
                    if dividends[currentDivIndex].dividend.isProportional
                        @. v = pp(Si * (1 - dividends[currentDivIndex].dividend.amount))
                    else
                        if dividendPolicy == Shift
                            @. v = pp(Si - dividends[currentDivIndex].dividend.amount)
                        elseif dividendPolicy == Survivor
                            @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
                        else #liquidator
                            v1 = @view v1Matrix[:, iv]
                            @. v1 = max(Si - dividends[currentDivIndex].dividend.amount, zero(T))
                            evaluateSorted!(pp, v, v1)
                            # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
                        end
                    end
                end
                currentDivIndex -= 1

            end
        end
    end
    #PPInterpolation.computePP(pp,Si, @view(vMatrix[:,end]), PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())
    #return pp
    return QuadraticLagrangePP(Si, vMatrix[:, 1])
end
