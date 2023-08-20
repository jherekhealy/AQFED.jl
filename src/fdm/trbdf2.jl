export DividendPolicy, priceTRBDF2, NoCalibration, ForwardCalibration
using AQFED.TermStructure
using LinearAlgebra
using PPInterpolation
@enum DividendPolicy begin
    Liquidator
    Survivor
    Shift
end

abstract type BoundaryConditionWay end
struct Down <: BoundaryConditionWay end
struct Up <: BoundaryConditionWay end

# @enum TimeSteppingName begin
#     TRBDF2
# end
abstract type TimeStepping end
struct TRBDF2{T,TS} <: TimeStepping
    vLowerBound::Vector{T}
    solver::TS
    lhs::Tridiagonal{T,Vector{T}}
    tri::Tridiagonal{T,Vector{T}}
    v0Matrix::Matrix{T}
    v1Matrix::Matrix{T}
end

struct CrankNicolson{T,TS} <: TimeStepping
    vLowerBound::Vector{T}
    solver::TS
    lhs::Tridiagonal{T,Vector{T}}
    tri::Tridiagonal{T,Vector{T}}
    v1Matrix::Matrix{T}
 end
struct BDF1{T,TS} <: TimeStepping
    vLowerBound::Vector{T}
    solver::TS
    lhs::Tridiagonal{T,Vector{T}}
    tri::Tridiagonal{T,Vector{T}}
    v1Matrix::Matrix{T}
end
struct Rannacher{T,TS} <: TimeStepping
    cn::CrankNicolson{T,TS}
    bdf1::BDF1{T,TS}
    bdf1Steps::Int
end


function makeIsMiddle(definition, specialPoints)
    return ones(Bool, length(specialPoints))
    # return zeros(Bool, length(specialPoints))
end
function makeIsMiddle(definition::Union{ContinuousKO,ContinuousKI}, specialPoints)
    isMiddle = Vector{Bool}()
    for point = specialPoints
        if typeof(definition.vanilla) <: Union{ContinuousKO,ContinuousKI}
            push!(isMiddle, point != definition.level && point != definition.vanilla.level)
        else
            push!(isMiddle, point != definition.level)
        end
    end
    return isMiddle
end

function makeIsMiddle(definition::Union{ContinuousDKO}, specialPoints)
    isMiddle = Vector{Bool}()
    for point = specialPoints
        push!(isMiddle, point != definition.levelUp && point != definition.levelDown)

    end
    return isMiddle
end

#TODO use term structure of rates and vols from a model.
#TODO use upstream/downstream deriv/expo fitting if convect dominates.
function priceTRBDF2(definition::StructureDefinition,
    spot::T,
    rawForward::T, #The raw forward to τ (without cash dividends)
    variance::T, #variance to maturity
    discountDf::T, #discount factor to payment date
    dividends::AbstractArray{CapitalizedDividend{T}};
    timeSteppingName="TRBDF2",
    solverName="TDMA", M=400, N=100, ndev=4, Smax=zero(T), Smin=zero(T), dividendPolicy::DividendPolicy=Liquidator, calibration=NoCalibration(), varianceConditioner::PecletConditioner=NoConditioner()) where {T}
    obsTimes = observationTimes(definition)
    τ = last(obsTimes)
    varianceSurface = FlatSurface(sqrt(variance / τ))
    discountCurve = ConstantRateCurve(-log(discountDf) / τ)
    driftCurve = ConstantRateCurve(log(rawForward / spot) / τ)
    model = TSBlackModel(varianceSurface, discountCurve, driftCurve)
    return priceTRBDF2(definition, spot, model, dividends, timeSteppingName=timeSteppingName, solverName=solverName, M=M, N=N, ndev=ndev, Smax=Smax, Smin=Smin, dividendPolicy=dividendPolicy, calibration=calibration, varianceConditioner=varianceConditioner)
end

struct Trace{T}
    times::Vector{T}
    values::Vector{Vector{T}}
end

function valuesToMatrix(vecvec::Vector{Vector{T}}) where {T}
    dim1 = length(vecvec)
    dim2 = length(vecvec[1])
    my_array = zeros(T, dim1, dim2)
    for i in 1:dim1
        for j in 1:dim2
            my_array[i, j] = vecvec[i][j]
        end
    end
    return my_array
end

function dtFactor(timeStepping::TRBDF2, dt::T, tIndex::Int) where {T}
    2 * one(T) - sqrt(2 * one(T))
end

function dtFactor(timeStepping::CrankNicolson, dt::T, tIndex::Int) where {T}
    one(T)
end

function dtFactor(timeStepping::Rannacher, dt::T, tIndex::Int) where {T}
    one(T)

end

function dtFactor(timeStepping::BDF1, dt::T, tIndex::Int) where {T}
    2 * one(T)
end

function makeSolver(solverName, vLowerBound::AbstractArray{T}, lhs, isLBActive) where {T}
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
    return LowerBoundSolver(solverLB, isLBActive, lhs, vLowerBound)
end

function TRBDF2(definition, payoff, solverName, Si::AbstractArray{T}) where {T}
    lhsdl = zeros(T, length(Si) - 1)
    lhsd = zeros(T, length(Si))
    lhsdu = zeros(T, length(Si) - 1)
    lhs = Tridiagonal(lhsdl, lhsd, lhsdu)
    tri = Tridiagonal(zeros(T, length(Si) - 1), zeros(T, length(Si)), zeros(T, length(Si) - 1))
    vMatrix = currentValue(payoff)
    v0Matrix = similar(vMatrix)
    v1Matrix = similar(vMatrix)
    vLowerBound = zeros(T, length(Si))
    isLBActive = isLowerBoundActive(definition, payoff)
    if isLBActive
        lowerBound!(payoff, vLowerBound)
    else
        ##FIXME how does the solver knows it is active or not?
    end
    solver = makeSolver(solverName, vLowerBound, lhs, isLBActive)
    return TRBDF2(vLowerBound, solver, lhs, tri, v0Matrix, v1Matrix)
end

function CrankNicolson(definition, payoff, solverName, Si::AbstractArray{T}) where {T}
    lhsdl = zeros(T, length(Si) - 1)
    lhsd = zeros(T, length(Si))
    lhsdu = zeros(T, length(Si) - 1)
    lhs = Tridiagonal(lhsdl, lhsd, lhsdu)
    tri = Tridiagonal(zeros(T, length(Si) - 1), zeros(T, length(Si)), zeros(T, length(Si) - 1))
    vMatrix = currentValue(payoff)
    v1Matrix = similar(vMatrix)
    vLowerBound = zeros(T, length(Si))
    isLBActive = isLowerBoundActive(definition, payoff)
    if isLBActive
        lowerBound!(payoff, vLowerBound)
    else
        ##FIXME how does the solver knows it is active or not?
    end
    solver = makeSolver(solverName, vLowerBound, lhs, isLBActive)
    return CrankNicolson(vLowerBound, solver, lhs, tri, v1Matrix)
end


function advance(timeStepping::TRBDF2, definition, payoff, tip, dt, tIndex, Si::AbstractArray{T}, rhs, vMatrix, advectionCoeffDown, sinkCoeffDown, advectionCoeffUp, sinkCoeffUp) where {T}
    beta = dtFactor(timeStepping, dt, tIndex)
    tri = timeStepping.tri
    v1Matrix = timeStepping.v1Matrix
    v0Matrix = timeStepping.v0Matrix
    rhsd = rhs.d
    rhsdl = rhs.dl
    rhsdu = rhs.du
    v0Matrix .= vMatrix
    advance(definition, payoff, tip - dt * beta)
    for (iv, v) in (enumerate(eachcol(vMatrix))) #we don't go reverse here as we want to use initial vanilla price in KO for explicit step.
        bcDown, bcUp = boundaryConditions(definition, tip - dt * beta, iv)
        tri.d .= rhsd #in fact only 1 or 2 lines need to be copied
        tri.dl .= rhsdl
        tri.du .= rhsdu
        applyBoundaryCondition(Down(), iv, bcDown, Si, vMatrix, tri, dt * beta / 2, advectionCoeffDown, sinkCoeffDown) #if Dirichlet on grid, just do it, otherwise use Ghost point. rhs vs lhs?
        applyBoundaryCondition(Up(), iv, bcUp, Si, vMatrix, tri, dt * beta / 2, advectionCoeffUp, sinkCoeffUp)
        #warning we have updated rhs for this iv, but we may want to keep initial rhsmatrix for iv-1.
        mul!(v, tri, vMatrix[:, iv])
        # println(iv, " ",bcDown, " ",bcUp," expl ",v)
        #  evaluate(definition,payoff, Si, iv)  #not necessary to update knockin values from vanilla.
    end
    lhsd = timeStepping.lhs.d
    lhsdl = timeStepping.lhs.dl
    lhsdu = timeStepping.lhs.du

    @. lhsd = one(T) - (rhsd - one(T))
    @. lhsdu = -rhsdu
    @. lhsdl = -rhsdl
    # lhsf = lu!(lhs)
    # lhsf = factorize(lhs)
    # ldiv!(v, lhsf , v1)
    advance(definition, payoff, tip - dt * beta)
    for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
        bcDown, bcUp = boundaryConditions(definition, tip - dt * beta, iv) #Dirichlet(level,value) or Linear()
        tri.d .= lhsd #in fact only 1 or 2 lines need to be copied
        tri.dl .= lhsdl
        tri.du .= lhsdu
        applyBoundaryConditionLeft(Down(), iv, bcDown, Si, vMatrix, tri, dt * beta / 2, advectionCoeffDown, sinkCoeffDown)
        applyBoundaryConditionLeft(Up(), iv, bcUp, Si, vMatrix, tri, dt * beta / 2, advectionCoeffUp, sinkCoeffUp)
        isLBActive = isLowerBoundActive(definition, payoff)
        setLowerBoundActive(timeStepping.solver, isLBActive)
        v1 = @view v1Matrix[:, iv]
        decompose(timeStepping.solver, tri)
        solve!(timeStepping.solver, v1, v)
        v[1:end] .= v1
        # println(iv," impl ",v)
        # evaluate(definition,payoff, Si, iv)  #not necessary to update knockin values from vanilla.
        if isLBActive
            lowerBound!(payoff, timeStepping.vLowerBound)
        end
    end
    solver = timeStepping.solver
    advance(definition, payoff, tip - dt)
    for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
        v1 = @view(v1Matrix[:, iv]) #copy(v)
        @. v = (v1 - (1 - beta)^2 * @view v0Matrix[:, iv]) / (beta * (2 - beta))
        # ldiv!(v , lhsf ,v1)
        bcDown, bcUp = boundaryConditions(definition, tip - dt * beta, iv) #Dirichlet(level,value) or Linear()
        tri.d .= lhsd #in fact only 1 or 2 lines need to be copied
        tri.dl .= lhsdl
        tri.du .= lhsdu
        applyBoundaryConditionLeft(Down(), iv, bcDown, Si, vMatrix, tri, dt * beta / 2, advectionCoeffDown, sinkCoeffDown)
        applyBoundaryConditionLeft(Up(), iv, bcUp, Si, vMatrix, tri, dt * beta / 2, advectionCoeffUp, sinkCoeffUp)
        isLBActive = isLowerBoundActive(definition, payoff)
        setLowerBoundActive(timeStepping.solver, isLBActive)
        decompose(timeStepping.solver, tri)
        solve!(timeStepping.solver, v1, v)
        v[1:end] = v1
        evaluate(definition, payoff, Si, iv)
        if isLBActive
            lowerBound!(payoff, timeStepping.vLowerBound)
        end
    end

end


function advance(timeStepping::CrankNicolson, definition, payoff, tip, dt, tIndex, Si::AbstractArray{T}, rhs, vMatrix, advectionCoeffDown, sinkCoeffDown, advectionCoeffUp, sinkCoeffUp) where {T}
    beta = dtFactor(timeStepping, dt, tIndex)
    tri = timeStepping.tri
    v1Matrix = timeStepping.v1Matrix
    rhsd = rhs.d
    rhsdl = rhs.dl
    rhsdu = rhs.du
    advance(definition, payoff, tip - dt * beta)
    for (iv, v) in (enumerate(eachcol(vMatrix))) #we don't go reverse here as we want to use initial vanilla price in KO for explicit step.
        bcDown, bcUp = boundaryConditions(definition, tip - dt * beta, iv)
        tri.d .= rhsd #in fact only 1 or 2 lines need to be copied
        tri.dl .= rhsdl
        tri.du .= rhsdu
        applyBoundaryCondition(Down(), iv, bcDown, Si, vMatrix, tri, dt * beta / 2, advectionCoeffDown, sinkCoeffDown) #if Dirichlet on grid, just do it, otherwise use Ghost point. rhs vs lhs?
        applyBoundaryCondition(Up(), iv, bcUp, Si, vMatrix, tri, dt * beta / 2, advectionCoeffUp, sinkCoeffUp)
        #warning we have updated rhs for this iv, but we may want to keep initial rhsmatrix for iv-1.
        mul!(v, tri, vMatrix[:, iv])
        # println(iv, " ",bcDown, " ",bcUp," expl ",v)
        #  evaluate(definition,payoff, Si, iv)  #not necessary to update knockin values from vanilla.
    end
    lhsd = timeStepping.lhs.d
    lhsdl = timeStepping.lhs.dl
    lhsdu = timeStepping.lhs.du

    @. lhsd = one(T) - (rhsd - one(T))
    @. lhsdu = -rhsdu
    @. lhsdl = -rhsdl
    # lhsf = lu!(lhs)
    # lhsf = factorize(lhs)
    # ldiv!(v, lhsf , v1)
    advance(definition, payoff, tip - dt)
    for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
        bcDown, bcUp = boundaryConditions(definition, tip - dt * beta, iv) #Dirichlet(level,value) or Linear()
        tri.d .= lhsd #in fact only 1 or 2 lines need to be copied
        tri.dl .= lhsdl
        tri.du .= lhsdu
        applyBoundaryConditionLeft(Down(), iv, bcDown, Si, vMatrix, tri, dt * beta / 2, advectionCoeffDown, sinkCoeffDown)
        applyBoundaryConditionLeft(Up(), iv, bcUp, Si, vMatrix, tri, dt * beta / 2, advectionCoeffUp, sinkCoeffUp)
        isLBActive = isLowerBoundActive(definition, payoff)
        setLowerBoundActive(timeStepping.solver, isLBActive)
        v1 = @view v1Matrix[:, iv]
        decompose(timeStepping.solver, tri)
        solve!(timeStepping.solver, v1, v)
        v[1:end] .= v1
        # println(iv," impl ",v)
        # evaluate(definition,payoff, Si, iv)  #not necessary to update knockin values from vanilla.
        if isLBActive
            lowerBound!(payoff, timeStepping.vLowerBound)
        end
    end
    for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
        evaluate(definition, payoff, Si, iv)
        isLBActive = isLowerBoundActive(definition, payoff)
        if isLBActive
            lowerBound!(payoff, timeStepping.vLowerBound)
        end
    end

end

function advance(timeStepping::Rannacher, definition, payoff, tip, dt, tIndex, Si::AbstractArray{T}, rhs, vMatrix, advectionCoeffDown, sinkCoeffDown, advectionCoeffUp, sinkCoeffUp) where {T}
    if tIndex <= timeStepping.bdf1Steps
        println("bdf1 ran ",tIndex)
        advance(timeStepping.bdf1,definition, payoff, tip, dt/2, tIndex, Si, rhs, vMatrix, advectionCoeffDown, sinkCoeffDown, advectionCoeffUp, sinkCoeffUp)
        advance(timeStepping.bdf1,definition, payoff, tip, dt/2, tIndex, Si, rhs, vMatrix, advectionCoeffDown, sinkCoeffDown, advectionCoeffUp, sinkCoeffUp)
    else
        advance(timeStepping.cn,definition, payoff, tip, dt, tIndex, Si, rhs, vMatrix, advectionCoeffDown, sinkCoeffDown, advectionCoeffUp, sinkCoeffUp)
    end

end
function BDF1(definition, payoff, solverName, Si::AbstractArray{T}) where {T}
    lhsdl = zeros(T, length(Si) - 1)
    lhsd = zeros(T, length(Si))
    lhsdu = zeros(T, length(Si) - 1)
    lhs = Tridiagonal(lhsdl, lhsd, lhsdu)
    tri = Tridiagonal(zeros(T, length(Si) - 1), zeros(T, length(Si)), zeros(T, length(Si) - 1))
    vMatrix = currentValue(payoff)
    v1Matrix = similar(vMatrix)
    vLowerBound = zeros(T, length(Si))
    isLBActive = isLowerBoundActive(definition, payoff)
    if isLBActive
        lowerBound!(payoff, vLowerBound)
    else
        ##FIXME how does the solver knows it is active or not?
    end
    solver = makeSolver(solverName, vLowerBound, lhs, isLBActive)
    return BDF1(vLowerBound, solver, lhs, tri, v1Matrix)
end

function advance(timeStepping::BDF1, definition, payoff, tip, dt, tIndex, Si::AbstractArray{T}, rhs, vMatrix, advectionCoeffDown, sinkCoeffDown, advectionCoeffUp, sinkCoeffUp) where {T}
    beta = dtFactor(timeStepping, dt, tIndex)
    tri = timeStepping.tri
    v1Matrix = timeStepping.v1Matrix
    rhsd = rhs.d
    rhsdl = rhs.dl
    rhsdu = rhs.du
    advance(definition, payoff, tip - dt)
    lhsd = timeStepping.lhs.d
    lhsdl = timeStepping.lhs.dl
    lhsdu = timeStepping.lhs.du

    @. lhsd = one(T) - (rhsd - one(T))
    @. lhsdu = -rhsdu
    @. lhsdl = -rhsdl
    # lhsf = lu!(lhs)
    # lhsf = factorize(lhs)
    # ldiv!(v, lhsf , v1)
    for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
        bcDown, bcUp = boundaryConditions(definition, tip - dt * beta, iv) #Dirichlet(level,value) or Linear()
        tri.d .= lhsd #in fact only 1 or 2 lines need to be copied
        tri.dl .= lhsdl
        tri.du .= lhsdu
        applyBoundaryConditionLeft(Down(), iv, bcDown, Si, vMatrix, tri, dt * beta / 2, advectionCoeffDown, sinkCoeffDown)
        applyBoundaryConditionLeft(Up(), iv, bcUp, Si, vMatrix, tri, dt * beta / 2, advectionCoeffUp, sinkCoeffUp)
        isLBActive = isLowerBoundActive(definition, payoff)
        setLowerBoundActive(timeStepping.solver, isLBActive)
        v1 = @view v1Matrix[:, iv]
        decompose(timeStepping.solver, tri)
        solve!(timeStepping.solver, v1, v)
        v[1:end] .= v1
        # println(iv," impl ",v)
        # evaluate(definition,payoff, Si, iv)  #not necessary to update knockin values from vanilla.
        if isLBActive
            lowerBound!(payoff, timeStepping.vLowerBound)
        end
    end
    for (iv, v) in Iterators.reverse(enumerate(eachcol(vMatrix)))
        evaluate(definition, payoff, Si, iv)
        isLBActive = isLowerBoundActive(definition, payoff)
        if isLBActive
            lowerBound!(payoff, timeStepping.vLowerBound)
        end
    end

end

# function priceTRBDF2(definition::StructureDefinition,
#     spot::T,
#     driftCurve::Curve, #The raw forward to τ (without cash dividends)
#     varianceSurface::VarianceSurface, #variance to maturity
#     discountCurve::Curve, #discount factor to payment date
#     dividends::AbstractArray{CapitalizedDividend{T}};
#     solverName="LUUL", M=400, N=100, ndev=4, Smax=zero(T), Smin=zero(T), dividendPolicy::DividendPolicy=Liquidator, grid::Grid=UniformGrid(false), varianceConditioner::PecletConditioner=NoConditioner(), calibration=NoCalibration(), useCN=false) where {T}
#     model   = TSBlackModel(varianceSurface, discountCurve, driftCurve)
#     return priceTRBDF2(definition,spot,model,dividends,
#      solverName=solverName, M=M, N=N,ndev=ndev, Smax=Smax, Smin=Smin, dividendsPolicy=dividendPolicy, grid=grid, varianceConditioner=varianceConditioner,calibration=calibration,useCN=useCN)
# end
function priceTRBDF2(definition::StructureDefinition,
    spot::T,
    model,
    dividends::AbstractArray{CapitalizedDividend{T}};
    timeSteppingName="TRBDF2",
    solverName="LUUL", M=400, N=100, ndev=4, Smax=zero(T), Smin=zero(T), dividendPolicy::DividendPolicy=Liquidator, grid::Grid=UniformGrid(false), varianceConditioner::PecletConditioner=NoConditioner(), calibration=NoCalibration(),
    useSqrt=false, useTrace=false, trace=Trace(Vector{T}(), Vector{Vector{T}}())) where {T}
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
        append!(isMiddle, false)
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
    isMiddle = makeIsMiddle(definition, specialPoints)
    if !isempty(specialPoints)
        Ui = max(Ui, if isMiddle[end]
            specialPoints[end] * 1.01
        else
            specialPoints[end]
        end)
        Li = min(Li, if isMiddle[1]
            specialPoints[1] * 0.99
        else
            specialPoints[1]
        end)
    end
    Si = makeArray(grid, xi, Li, Ui, specialPoints, isMiddle)
    #  println("S ",Si)   
    #     println("t ",t)
    tip = t[1]
    payoff = makeFDMStructure(definition, Si)
    advance(definition, payoff, tip)
    evaluate(definition, payoff, Si)
    vMatrix = currentValue(payoff)
    timeStepping =
        if timeSteppingName == "TRBDF2"
            TRBDF2(definition, payoff, solverName, Si)
        elseif timeSteppingName == "CN"
            CrankNicolson(definition, payoff, solverName, Si)
        elseif timeSteppingName == "BDF1"
            BDF1(definition, payoff, solverName, Si)
        elseif timeSteppingName == "Rannacher"
            Rannacher(CrankNicolson(definition,payoff,solverName, Si),BDF1(definition,payoff,solverName, Si),1)
        elseif timeSteppingName == "Rannacher4"
            Rannacher(CrankNicolson(definition,payoff,solverName, Si),BDF1(definition,payoff,solverName, Si),2)
        else
            TRBDF2(definition, payoff, solverName, Si)
        end
    # println("initial payoff ",vMatrix[:,1], " ",vMatrix[:,2])
    Jhi = @. (Si[2:end] - Si[1:end-1])
    rhsd = zeros(T, length(Si))
    rhsdu = zeros(T, length(Si) - 1)
    rhsdl = zeros(T, length(Si) - 1)
    rhs = Tridiagonal(rhsdl, rhsd, rhsdu)
    v1Matrix = similar(vMatrix) #for dividends
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
    tIndex = 1
    for i = 2:length(t)
        ti = t[i]
        dt = tip - ti
        if dt > 1e-8
            beta = dtFactor(timeStepping, dt, tIndex)
            dfi = discountFactor(model, ti)
            dfip = discountFactor(model, tip)
            ri = calibrateRate(calibration, beta, dt, dfi, dfip)
            driftDfi = 1 / forward(model, 1.0, ti)
            driftDfip = 1 / forward(model, 1.0, tip)
            μi = calibrateDrift(calibration, beta, dt, dfi, dfip, driftDfi, driftDfip, ri)
            σi2 = (varianceByLogmoneyness(model, 0.0, tip) * tip - varianceByLogmoneyness(model, 0.0, ti) * ti) / (tip - ti)
            @inbounds for j = 2:M-1
                s2S = σi2 * Si[j]^2
                muS = μi * Si[j]
                s2S = conditionedVariance(varianceConditioner, s2S, muS, Si[j], Jhi[j-1], Jhi[j])
                rhsd[j] = one(T) - dt * beta / 2 * ((muS * (Jhi[j-1] - Jhi[j]) + s2S) / (Jhi[j] * Jhi[j-1]) + ri)
                rhsdu[j] = dt * beta / 2 * (s2S + muS * Jhi[j-1]) / (Jhi[j] * (Jhi[j] + Jhi[j-1]))
                rhsdl[j-1] = dt * beta / 2 * (s2S - muS * Jhi[j]) / (Jhi[j-1] * (Jhi[j] + Jhi[j-1]))
            end
            #linear or Ke-rt same thing
            advectionCoeffDown = μi * Si[1] / Jhi[1]
            sinkCoeffDown = ri
            advectionCoeffUp = -μi * Si[end] / Jhi[end]
            sinkCoeffUp = ri
            advance(timeStepping, definition, payoff, tip, dt, tIndex, Si, rhs, vMatrix, advectionCoeffDown, sinkCoeffDown, advectionCoeffUp, sinkCoeffUp)
            tIndex+=1
        end
        tip = ti
        if ti > 0.9
            if useTrace
                push!(trace.times, ti)
                push!(trace.values, vMatrix[:, 1])
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

function applyBoundaryConditionLeft(way::TW, colIndex::Int, bcDown::T, S::AbstractArray{TS}, v0Matrix, rhs, dt, advectionCoeff, sinkCoeff) where {TW<:BoundaryConditionWay,T<:BoundaryCondition,TS}
    applyBoundaryCondition(way, colIndex, bcDown, S, v0Matrix, rhs, dt, advectionCoeff, sinkCoeff) #for dirichlet, left = right 
end

function applyBoundaryConditionLeft(::Down, colIndex::Int, bcDown::LinearBoundaryCondition, S::AbstractArray{TS}, v0Matrix, rhs, dt, advectionCoeff, sinkCoeff) where {TS}
    rhs.d[1] = one(TS) + dt * (sinkCoeff + advectionCoeff)
    rhs.du[1] = -dt * advectionCoeff
end
function applyBoundaryCondition(::Down, colIndex::Int, bcDown::LinearBoundaryCondition, S::AbstractArray{TS}, v0Matrix, rhs, dt, advectionCoeff, sinkCoeff) where {TS}
    rhs.d[1] = one(TS) - dt * (sinkCoeff + advectionCoeff)
    rhs.du[1] = dt * advectionCoeff
end
function applyBoundaryConditionLeft(::Up, colIndex::Int, bcDown::LinearBoundaryCondition, S::AbstractArray{TS}, v0Matrix, rhs, dt, advectionCoeff, sinkCoeff) where {TS}
    M = length(rhs.d)
    rhs.d[M] = one(TS) + dt * (sinkCoeff + advectionCoeff)
    rhs.dl[M-1] = -dt * advectionCoeff
end

function applyBoundaryCondition(::Up, colIndex::Int, bcDown::LinearBoundaryCondition, S::AbstractArray{TS}, v0Matrix, rhs, dt, advectionCoeff, sinkCoeff) where {TS}
    M = length(rhs.d)
    rhs.d[M] = one(TS) - dt * (sinkCoeff + advectionCoeff)
    rhs.dl[M-1] = dt * advectionCoeff
end

function applyBoundaryConditionOnGrid(::Union{Up,Down}, rowIndex::Int, colIndex::Int, bcDown::DirichletBoundaryCondition, v0Matrix::AbstractMatrix{TS}, lhs) where {TS}
    #TODO for loop from 1 to rowIndex
    lhs.d[rowIndex] = one(TS)
    if rowIndex < length(lhs.d)
        lhs.du[rowIndex] = zero(TS)
    end
    if rowIndex > 1
        lhs.dl[rowIndex-1] = zero(TS)
    end
    v0Matrix[rowIndex, colIndex] = bcDown.value
end


function interpolateBoundaryValue(v0Matrix, Sip, iU, Ui, bc::DirichletBoundaryCondition; interpolation="Quadratic")
    return bc.value
end

function interpolateBoundaryValue(v0Matrix, Sip, iU, Ui, bc::DirichletPayoffBoundaryCondition; interpolation="Quadratic")
    colIndex = bc.columnIndex
    if interpolation == "Quadratic"
        return if iU == length(Sip)
            v0Matrix[iU, colIndex] * ((Ui - Sip[iU-1]) * (Ui - Sip[iU-2]) / ((Sip[iU] - Sip[iU-1]) * (Sip[iU] - Sip[iU-2]))) + (Ui - Sip[iU-1]) * (Ui - Sip[iU]) / ((Sip[iU-2] - Sip[iU-1]) * (Sip[iU-2] - Sip[iU])) * v0Matrix[iU-2, colIndex] + (Ui - Sip[iU]) * (Ui - Sip[iU-2]) / ((Sip[iU-1] - Sip[iU]) * (Sip[iU-1] - Sip[iU-2])) * v0Matrix[iU-1, colIndex]
        elseif iU == 1
            v0Matrix[iU, colIndex] * ((Ui - Sip[iU+1]) * (Ui - Sip[iU+2]) / ((Sip[iU] - Sip[iU+1]) * (Sip[iU] - Sip[iU+2]))) + (Ui - Sip[iU+1]) * (Ui - Sip[iU]) / ((Sip[iU+2] - Sip[iU+1]) * (Sip[iU+2] - Sip[iU])) * v0Matrix[iU+2, colIndex] + (Ui - Sip[iU]) * (Ui - Sip[iU+2]) / ((Sip[iU+1] - Sip[iU]) * (Sip[iU+1] - Sip[iU+2])) * v0Matrix[iU+1, colIndex]
        else
            v0Matrix[iU, colIndex] * ((Ui - Sip[iU-1]) * (Ui - Sip[iU+1]) / ((Sip[iU] - Sip[iU-1]) * (Sip[iU] - Sip[iU+1]))) + (Ui - Sip[iU-1]) * (Ui - Sip[iU]) / ((Sip[iU+1] - Sip[iU-1]) * (Sip[iU+1] - Sip[iU])) * v0Matrix[iU+1, colIndex] + (Ui - Sip[iU]) * (Ui - Sip[iU+1]) / ((Sip[iU-1] - Sip[iU]) * (Sip[iU-1] - Sip[iU+1])) * v0Matrix[iU-1, colIndex]
        end
    else
        return (v0Matrix[iU, colIndex] * (Ui - Sip[iU-1]) + v0Matrix[iU-1, colIndex] * (Sip[iU] - Ui)) / (Sip[iU] - Sip[iU-1])
    end
end


function getBoundaryValue(bcDown::DirichletPayoffBoundaryCondition, v0Matrix, i)
    v0Matrix[i, bcDown.columnIndex]
end
function getBoundaryValue(bcDown::DirichletBoundaryCondition, v0Matrix, i)
    bcDown.value
end

function applyBoundaryCondition(way::Down, colIndex::Int, bcDown::Union{DirichletBoundaryCondition,DirichletPayoffBoundaryCondition}, Si::AbstractArray{TS}, v0Matrix, lhs, dt, advectionCoeff, sinkCoeff) where {TS}
    rowIndex = searchsortedfirst(Si, bcDown.level)
    for i = 1:rowIndex-2
        lhs.d[i] = one(TS)
        lhs.du[i] = zero(TS)
        if i > 1
        lhs.dl[i-1] = zero(TS)
        end
        v0Matrix[i, colIndex] = getBoundaryValue(bcDown, v0Matrix, i)
    end
    if abs(bcDown.level - Si[rowIndex]) <= sqrt(eps(TS))
        applyBoundaryConditionOnGrid(way, rowIndex, colIndex, bcDown, v0Matrix, lhs)
    else
        Li = bcDown.level
        Sip = Si
        iL = rowIndex # S[iL-1] < level < S[iL]
        value = interpolateBoundaryValue(v0Matrix, Sip, iL - 1, Li, bcDown)
        divisor = ((Li - Sip[iL]) * (Li - Sip[iL+1]) / ((Sip[iL-1] - Sip[iL]) * (Sip[iL-1] - Sip[iL+1])))
        v0Matrix[iL-1, colIndex] = (value - (Li - Sip[iL-1]) * (Li - Sip[iL]) / ((Sip[iL+1] - Sip[iL-1]) * (Sip[iL+1] - Sip[iL])) * v0Matrix[iL+1, colIndex] - (Li - Sip[iL-1]) * (Li - Sip[iL+1]) / ((Sip[iL] - Sip[iL-1]) * (Sip[iL] - Sip[iL+1])) * v0Matrix[iL, colIndex]) / divisor
        lhs.d[iL-1] = one(TS)
        lhs.du[iL-1] = zero(TS)
        if iL > 2
            lhs.dl[iL-2] = zero(TS)
        end
    end
end

function applyBoundaryConditionLeft(way::Down, colIndex::Int, bcDown::Union{DirichletBoundaryCondition,DirichletPayoffBoundaryCondition}, Si::AbstractArray{TS}, v0Matrix, lhs, dt, advectionCoeff, sinkCoeff) where {TS}
    rowIndex = searchsortedfirst(Si, bcDown.level)
    for i = 1:rowIndex-2
        lhs.d[i] = one(TS)
        lhs.du[i] = zero(TS)
        if i > 1
            lhs.dl[i-1] = zero(TS)
        end
        v0Matrix[i, colIndex] = getBoundaryValue(bcDown, v0Matrix, i)
    end
    if abs(bcDown.level - Si[rowIndex]) <= sqrt(eps(TS))
        applyBoundaryConditionOnGrid(way, rowIndex, colIndex, bcDown, v0Matrix, lhs)
    else
        Li = bcDown.level
        Sip = Si
        iL = rowIndex # S[iL-1] < level < S[iL]
        v0Matrix[iL-1, colIndex] = interpolateBoundaryValue(v0Matrix, Sip, iL - 1, Li, bcDown)
        lhs.d[iL-1] = (Li - Sip[iL]) * (Li - Sip[iL+1]) / ((Sip[iL-1] - Sip[iL]) * (Sip[iL-1] - Sip[iL+1]))
        lhs.du[iL-1] = (Li - Sip[iL-1]) * (Li - Sip[iL+1]) / ((Sip[iL] - Sip[iL-1]) * (Sip[iL] - Sip[iL+1]))
        if iL > 2
            lhs.dl[iL-2] = zero(TS)
        end
        lhsduu = (Li - Sip[iL-1]) * (Li - Sip[iL]) / ((Sip[iL+1] - Sip[iL-1]) * (Sip[iL+1] - Sip[iL]))
        lhs.d[iL-1] -= lhs.dl[iL-1] * lhsduu / lhs.du[iL]
        lhs.du[iL-1] -= lhs.d[iL] * lhsduu / lhs.du[iL]
        v0Matrix[iL-1, colIndex] -= v0Matrix[iL, colIndex] * lhsduu / lhs.du[iL]
    end
end

function applyBoundaryConditionOnGrid(::Union{Up,Down}, rowIndex::Int, colIndex::Int, bcDown::DirichletPayoffBoundaryCondition, v0Matrix::AbstractMatrix{TS}, lhs) where {TS}
    #TODO for loop from 1 to rowIndex
    lhs.d[rowIndex] = one(TS)
    if rowIndex < length(lhs.d)
        lhs.du[rowIndex] = zero(TS)
    end
    if rowIndex > 1
        lhs.dl[rowIndex-1] = zero(TS)
    end
    v0Matrix[rowIndex, colIndex] = v0Matrix[rowIndex, bcDown.columnIndex]
end

function applyBoundaryCondition(way::Up, colIndex::Int, bcDown::Union{DirichletBoundaryCondition,DirichletPayoffBoundaryCondition}, Si::AbstractArray{TS}, v0Matrix, rhs, dt, advectionCoeff, sinkCoeff; interpolation="Linear") where {TS}
    rowIndex = searchsortedfirst(Si, bcDown.level)
    for i = rowIndex+1:length(Si)
        rhs.d[i] = one(TS)
        if i < length(Si)
            rhs.du[i] = zero(TS)
        end
        rhs.dl[i-1] = zero(TS)
        v0Matrix[i, colIndex] = getBoundaryValue(bcDown, v0Matrix, i)
    end
    if abs(bcDown.level - Si[rowIndex]) <= sqrt(eps(TS))
        applyBoundaryConditionOnGrid(way, rowIndex, colIndex, bcDown, v0Matrix, rhs)
        #  println(" OnGrid ",v0Matrix[rowIndex, colIndex], " ", v0Matrix[rowIndex, colIndex]," ",v0Matrix[rowIndex-1, colIndex], " ", v0Matrix[rowIndex-2, colIndex])
    else
        #interpolate 
        Ui = bcDown.level
        Sip = Si
        iU = rowIndex # S[iU-1] < level < S[iU]
        value = interpolateBoundaryValue(v0Matrix, Sip, iU, Ui, bcDown; interpolation=interpolation)
        if interpolation == "Quadratic"
            divisor = ((Ui - Sip[iU-1]) * (Ui - Sip[iU-2]) / ((Sip[iU] - Sip[iU-1]) * (Sip[iU] - Sip[iU-2])))
            v0Matrix[iU, colIndex] = (value - (Ui - Sip[iU-1]) * (Ui - Sip[iU]) / ((Sip[iU-2] - Sip[iU-1]) * (Sip[iU-2] - Sip[iU])) * v0Matrix[iU-2, colIndex] - (Ui - Sip[iU]) * (Ui - Sip[iU-2]) / ((Sip[iU-1] - Sip[iU]) * (Sip[iU-1] - Sip[iU-2])) * v0Matrix[iU-1, colIndex]) / divisor
            rhs.d[iU] = one(TS)
            rhs.dl[iU-1] = zero(TS)
            if iU <= length(rhs.du)
                rhs.du[iU] = zero(TS)
            end
        else
            # v0Matrix[iU, colIndex] = ((Sip[iU] - Sip[iU-1]) * value - (Sip[iU] - Ui) * v0Matrix[iU-1, colIndex]) / (Ui - Sip[iU-1])
            rhs.d[iU] = one(TS)
            rhs.dl[iU-1] = zero(TS)
            v0Matrix[iU, colIndex] = value
            #  rhs.d[iU] = (Sip[iU]- Sip[iU-1])/ (Ui - Sip[iU-1])
            #  rhs.dl[iU-1] = -(Sip[iU]-Ui)/ (Ui - Sip[iU-1]) 
            rhs.d[iU-1] = rhs.d[iU-1] - rhs.du[iU-1] * ((Sip[iU] - Ui) / (Ui - Sip[iU-1]))
            rhs.du[iU-1] = rhs.du[iU-1] * ((Sip[iU] - Sip[iU-1]) / (Ui - Sip[iU-1]))

            if iU <= length(rhs.du)
                rhs.du[iU] = zero(TS)
            end
        end
        # println(" Quadratic ", value, " ", v0Matrix[iU, colIndex], " ", v0Matrix[iU-1, colIndex], " ", v0Matrix[iU-2, colIndex])
        # println("interpolated value ", value, " ", v0Matrix[iU, colIndex], " ", Ui, " ", v0Matrix[iU-1, colIndex], " ", v0Matrix[iU-2, colIndex])

    end
end


function applyBoundaryConditionLeft(way::Up, colIndex::Int, bcDown::Union{DirichletBoundaryCondition,DirichletPayoffBoundaryCondition}, Si::AbstractArray{TS}, v0Matrix, lhs, dt, advectionCoeff, sinkCoeff; interpolation="Quadratic") where {TS}
    rowIndex = searchsortedfirst(Si, bcDown.level)
    for i = rowIndex+1:length(Si)
        lhs.d[i] = one(TS)
        if i < length(Si)
            lhs.du[i] = zero(TS)
        end
        lhs.dl[i-1] = zero(TS)
        v0Matrix[i, colIndex] = getBoundaryValue(bcDown, v0Matrix, i)
    end
    if abs(bcDown.level - Si[rowIndex]) <= sqrt(eps(TS))
        applyBoundaryConditionOnGrid(way, rowIndex, colIndex, bcDown, v0Matrix, lhs)
    else
        Ui = bcDown.level
        Sip = Si
        iU = rowIndex # S[iU-1] < level < S[iU]
        # println("found boundary ",Si[iU-1]," ",bcDown.level," ",Si[iU])
        value = interpolateBoundaryValue(v0Matrix, Sip, iU, Ui, bcDown, interpolation=interpolation)
        v0Matrix[iU, colIndex] = value
        if interpolation == "Quadratic"
            lhs.d[iU] = (Ui - Sip[iU-1]) * (Ui - Sip[iU-2]) / ((Sip[iU] - Sip[iU-1]) * (Sip[iU] - Sip[iU-2]))
            lhs.dl[iU-1] = (Ui - Sip[iU]) * (Ui - Sip[iU-2]) / ((Sip[iU-1] - Sip[iU]) * (Sip[iU-1] - Sip[iU-2]))
            if iU <= length(lhs.du)
                lhs.du[iU] = zero(TS)
            end
            lhsdll = (Ui - Sip[iU-1]) * (Ui - Sip[iU]) / ((Sip[iU-2] - Sip[iU-1]) * (Sip[iU-2] - Sip[iU]))
            lhs.d[iU] -= lhs.du[iU-1] * lhsdll / lhs.dl[iU-2]
            lhs.dl[iU-1] -= lhs.d[iU-1] * lhsdll / lhs.dl[iU-2]
            v0Matrix[iU, colIndex] -= v0Matrix[iU-1, colIndex] * lhsdll / lhs.dl[iU-2]
        else
            lhs.d[iU] = (Ui - Sip[iU-1]) / (Sip[iU] - Sip[iU-1])   #lhd * v[iU] + lhsdl v[iU-1] = value
            lhs.dl[iU-1] = (Sip[iU] - Ui) / (Sip[iU] - Sip[iU-1])
            if iU <= length(lhs.du)
                lhs.du[iU] = zero(TS)
            end
        end
    end
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
