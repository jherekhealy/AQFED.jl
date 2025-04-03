using Roots
import AQFED.Math: normcdf, norminv, inv, ClosedTransformation, FitResult, MQMinTransformation, Quadrature, GaussKronrod
using ForwardDiff

export SmileConvention, BrokerConvention, CallDelta, PutDelta, ATMStraddle, ATMPremiumStraddle, CallPremiumDelta, PutPremiumDelta, CallForwardDelta, PutForwardDelta, CallPremiumForwardDelta, PutPremiumForwardDelta, strikeForDelta, logmoneynessForDelta, deltaForLogmoneyness, deltaForStrike, makeStrikeConvention



#conversion from quotes to call deltas
abstract type FXConvention end
struct SmileConvention <: FXConvention
end
function convertQuotesToDeltaVols(convention::SmileConvention, volAtm, bf25, rr25, bf10, rr10)
    vol10Call = volAtm + bf10 + rr10 / 2
    vol10Put = volAtm + bf10 - rr10 / 2
    vol25Call = volAtm + bf25 + rr25 / 2
    vol25Put = volAtm + bf25 - rr25 / 2
    return [vol10Put, vol25Put, volAtm, vol25Call, vol10Call]
end

abstract type StrikeConvention end
struct CallDelta <: StrikeConvention
end
struct PutDelta <: StrikeConvention
end
struct ATMStraddle <: StrikeConvention
end
struct ATMPremiumStraddle <: StrikeConvention
end
struct CallPremiumDelta <: StrikeConvention
end
struct PutPremiumDelta <: StrikeConvention
end
struct CallForwardDelta <: StrikeConvention
end
struct PutForwardDelta <: StrikeConvention
end
struct CallPremiumForwardDelta <: StrikeConvention
end
struct PutPremiumForwardDelta <: StrikeConvention
end

struct ConstantVarianceFunction{T} <: Function
    vol::T
end

(f::ConstantVarianceFunction)(x) = vol^2

function logmoneynessForDelta(convention::ATMStraddle, vol, tte, forward, df)
    return vol^2 * tte / 2
end
function logmoneynessForDelta(convention::ATMPremiumStraddle, vol, tte, forward, df)
    return -vol^2 * tte / 2
end

function logmoneynessForDelta(convention::ATMStraddle, delta, vol, tte, forward, df)
    return vol^2 * tte / 2
end
function logmoneynessForDelta(convention::ATMPremiumStraddle, delta, vol, tte, forward, df)
    return -vol^2 * tte / 2
end

function strikeForDelta(convention::T, vol, tte, forward, df) where {T <: StrikeConvention}
    return exp(logmoneynessForDelta(convention,vol,tte,forward,df))*forward
end

function strikeForDelta(convention::T, delta, vol, tte, forward, df) where {T <: StrikeConvention}
    return exp(logmoneynessForDelta(convention,delta, vol,tte,forward,df))*forward
end

function logmoneynessForDelta(convention::CallDelta, delta, vol, tte, forward, df)
    return -vol * sqrt(tte) * norminv(abs(delta) / df) + vol^2 * tte / 2
end

function logmoneynessForDelta(convention::PutDelta, delta, vol, tte, forward, df)
    return +vol * sqrt(tte) * norminv(abs(delta) / df) + vol^2 * tte / 2
end

function logmoneynessForDelta(convention::CallForwardDelta, delta, vol, tte, forward, df)
    return -vol * sqrt(tte) * norminv(abs(delta)) + vol^2 * tte / 2
end

function logmoneynessForDelta(convention::PutForwardDelta, delta, vol, tte, forward, df)
    #-N(-d1) = delta => d1 = -ninv(-delta)
    return +vol * sqrt(tte) * norminv(abs(delta)) + vol^2 * tte / 2
end

function logmoneynessForDelta(convention::CallPremiumDelta, delta, vol, tte, forward, df)
    return logmoneynessForDelta(CallPremiumForwardDelta(), delta/df, vol, tte, forward, df)
end

function logmoneynessForDelta(convention::PutPremiumDelta, delta, vol, tte, forward, df)
    return logmoneynessForForwardDeltaWithPremium(delta/df,vol,tte,forward)
end

function logmoneynessForDeltaNewton(convention::PutPremiumDelta, delta, vol ,tte, forward, df)
    objPut = function (y)
        delta - deltaForLogmoneyness(convention, y, vol,tte,forward,df)
    end
    objPutDer = x -> ForwardDiff.derivative(objPut, x)
    k = logmoneynessForDelta(PutDelta(), delta, vol, tte, forward, df)
    return Roots.find_zero((objPut, objPutDer), k, Roots.Newton())
end 

function logmoneynessForDelta(convention::CallPremiumForwardDelta, delta, vol, tte, forward, df)
    return logmoneynessForForwardDeltaWithPremium(delta,vol,tte,forward)
    #return logmoneynessForDeltaReiswich(convention, delta,vol,tte,forward,df)
end

function logmoneynessForDeltaReiswich(convention::CallPremiumForwardDelta, delta, vol, tte, forward, df)
  #  vol = max(eps(Float64),vol)
     ymax = logmoneynessForDelta(CallForwardDelta(),delta,vol,tte,forward,df) #Reiswich
    stddev = sqrt(vol*tte)
    objMax = function(y)
        d2 = -y/stddev - stddev/2
        value = stddev*normcdf(d2)-normpdf(d2)
        #println("ymin obj=",value)
        value
    end    
    bracket = (-stddev*5, ymax)
    # println("start ",vol)
    ymin =     Roots.find_zero(objMax,bracket,Roots.A42())+1e-10

    objCall = function (y)
        deltam = deltaForLogmoneyness(convention, y, vol,tte,forward,df)
        #println(y, " LFDC ",vol," ",deltam," ",delta-deltam)
        deltam - delta
    end
    if objCall(ymin) < 0
        return ymin
    end
   # objCallDer = x -> ForwardDiff.derivative(objCall, x)
   # k = logmoneynessForDelta(CallForwardDelta(), delta, vol, tte, forward, df)
    return Roots.find_zero(objCall, (ymin, ymax), Roots.A42())
end
function logmoneynessForDelta(convention::PutPremiumForwardDelta, delta, vol, tte, forward, df)
    return logmoneynessForForwardDeltaWithPremium(delta,vol,tte,forward)
end

function logmoneynessForDeltaNewton(convention::PutPremiumForwardDelta, delta, vol, tte, forward, df)
    objPut = function (y)
        delta - deltaForLogmoneyness(convention, y, vol,tte,forward,df)
    end
    objPutDer = x -> ForwardDiff.derivative(objPut, x)
    k = logmoneynessForDelta(PutForwardDelta(), delta, vol, tte, forward, df)
    return Roots.find_zero((objPut, objPutDer), k, Roots.Newton())
end



function deltaForStrike(convention::T, strike, vol, tte, forward, df) where {T <:StrikeConvention}
    return deltaForLogmoneyness(convention, log(strike/forward),vol,tte,forward,df)
end


function deltaForLogmoneyness(convention::ATMStraddle, y, vol, tte, forward, df)
    return vol^2 * tte / 2 #y = vol^2 * tte => 
end
function deltaForLogmoneyness(convention::ATMPremiumStraddle, y, vol, tte, forward, df)
    return -vol^2 * tte / 2
end

function deltaForLogmoneyness(convention::CallDelta, y, vol, tte, forward, df)
    return normcdf((-y + vol^2 * tte / 2) / (vol * sqrt(tte)))*df
end

function deltaForLogmoneyness(convention::CallForwardDelta, y, vol, tte, forward, df)
    return normcdf((-y + vol^2 * tte / 2) / (vol * sqrt(tte)))
end

function deltaForLogmoneyness(convention::CallPremiumDelta, y, vol, tte, forward, df)
    return exp(y) * normcdf((-y - vol^2 * tte / 2) / (vol * sqrt(tte)))*df
end
function deltaForLogmoneyness(convention::CallPremiumForwardDelta, y, vol, tte, forward, df)
    return exp(y) * normcdf((-y - vol^2 * tte / 2) / (vol * sqrt(tte)))
end

function deltaForLogmoneyness(convention::PutDelta, y, vol, tte, forward, df)
    return -normcdf((y - vol^2 * tte / 2) / (vol * sqrt(tte)))*df
end

function deltaForLogmoneyness(convention::PutForwardDelta, y, vol, tte, forward, df)
    return -normcdf((y - vol^2 * tte / 2) / (vol * sqrt(tte)))
end

function deltaForLogmoneyness(convention::PutPremiumDelta, y, vol, tte, forward, df)
    return -exp(y) * normcdf((y + vol^2 * tte / 2) / (vol * sqrt(tte)))*df
end
function deltaForLogmoneyness(convention::PutPremiumForwardDelta, y, vol, tte, forward, df)
    return -exp(y) * normcdf((y + vol^2 * tte / 2) / (vol * sqrt(tte)))
end



struct BrokerConvention{T} <: FXConvention
    tte::T
    forward::T
    dfForeign::T
    withPremium::Bool
    isForward::Bool
end

makeStrikeConvention(isCall::Bool, withPremium::Bool, isForward::Bool) =
    if isCall
        if withPremium
            if isForward
                CallPremiumForwardDelta()
            else
                CallPremiumDelta()
            end
        else
            if isForward
                CallForwardDelta()
            else
                CallDelta()
            end
        end
    else
        if withPremium
            if isForward
                PutPremiumForwardDelta()
            else
                PutPremiumDelta()
            end
        else
            if isForward
                PutForwardDelta()
            else
                PutDelta()
            end
        end
    end




abstract type AxisTransformation end
struct LogmoneynessAxisTransformation <: AxisTransformation
end
convertToLogmoneyness(::LogmoneynessAxisTransformation, y, vol) = y
solveLogmoneyness(::LogmoneynessAxisTransformation, y, vol) = y

struct StrikeAxisTransformation{T} <: AxisTransformation
    forward::T
end
convertToLogmoneyness(trans::StrikeAxisTransformation, strike, vol) = log(strike/trans.forward)
solveLogmoneyness(trans::StrikeAxisTransformation, strike, vol) = log(strike/trans.forward)

struct DeltaAxisTransformation <: AxisTransformation
    convention::FXConvention
end

function makeStrikeConventionFromDelta(convention::FXConvention,y)
    return if y == 0
        if convention.withPremium
            ATMPremiumStraddle()
        else
            ATMStraddle()
        end
    else
        makeStrikeConvention(y >= 0, convention.withPremium, convention.isForward)
    end
end

function convertToLogmoneyness(trans::DeltaAxisTransformation, y, vol) 
    convention = trans.convention
    strikeConv = makeStrikeConventionFromDelta(convention, y)
    logmoneynessForDelta(strikeConv,y,vol, convention.tte, convention.forward, convention.dfForeign) 
end

function solveLogmoneyness(trans::DeltaAxisTransformation,y, varianceByLogmoneynessFunction::F) where {F <: Function}
    convention = trans.convention
    strikeConv = makeStrikeConventionFromDelta(convention, y)
    solveLogmoneynessForDelta(strikeConv,y,varianceByLogmoneynessFunction, convention.tte, convention.forward, convention.dfForeign)   
end

function makeStrikeConvention(convention::BrokerConvention; deltas=[-0.10, -0.25, 0.5, 0.25, 0.10])    
    withPremium = convention.withPremium
    isForward = convention.isForward
    atmIndex = div(length(deltas) + 1, 2)
    [
        if index < atmIndex
            makeStrikeConvention(false, withPremium, isForward)
        elseif index > atmIndex
            makeStrikeConvention(true, withPremium, isForward)
        else
            if withPremium
                ATMPremiumStraddle()
            else()
                ATMStraddle()
            end
        end
        for (index, delta) = enumerate(deltas)]
end

function logmoneynessForDelta(deltaVols::Vector{Float64}, convention::BrokerConvention; deltas=[-0.10, -0.25, 0.5, 0.25, 0.10])
    convs = makeStrikeConvention(convention; deltas=deltas)
    forward = convention.forward
    tte = convention.tte
    dfForeign = convention.dfForeign
    map( (conv, delta, vol) -> logmoneynessForDelta(conv, delta, vol, tte, forward, dfForeign), convs, deltas, deltaVols)    
end

function strikeForDelta(deltaVols::Vector{Float64}, convention::BrokerConvention; deltas=[-0.10, -0.25, 0.5, 0.25, 0.10])
    ys = logmoneynessForDelta(deltaVols, convention, deltas=deltas)
    @. exp(ys)*convention.forward
end


function solveLogmoneynessForDelta(convention::T, delta,  varianceByLogmoneynessFunction::F ,tte,forward,df) where {T<:StrikeConvention, F <: Function}
    yMin = -10sqrt(varianceByLogmoneynessFunction(0.0)*sqrt(tte))
    yMax = -yMin
    #yTransformation = TanhTransformation(yMin,yMax)
    obj = function (y)
        vol = sqrt(varianceByLogmoneynessFunction(y))
        # ym = logmoneynessForDelta(convention, delta, vol, tte, forward, df)
        # println(vol," ",ym, " ",delta, " ",y)
        # ym-y
        # # y = yTransformation(iy)
        deltaM = max(-1.0,min(1.0,deltaForLogmoneyness(convention, y, vol ,tte,forward,df)))
        # println("solveP ",delta-deltaM, " ",y, " ",vol)
        delta - deltaM   
    end
    # objDer = x -> ForwardDiff.derivative(obj, x)
    # return Roots.find_zero((obj, objDer), 0.0, Roots.Newton())
 #   return Roots.find_zero(obj,inv(yTransformation,0.0), Roots.Secant(),atol=1e-8)
 return Roots.find_zero(obj,(yMin,yMax), Roots.A42(),atol=1e-8)
end

function solveLogmoneynessForDelta(convention::Union{CallPremiumDelta,CallPremiumForwardDelta}, delta,  varianceByLogmoneynessFunction::F ,tte,forward,df) where {F <: Function}
#     stddev0 = sqrt(varianceByLogmoneynessFunction(0.0)*tte)
#     objMax = function(y)
#         stddev = sqrt(max(1e-16,varianceByLogmoneynessFunction(y))*tte)
#         d2 = -y/stddev - stddev/2
#         value = stddev*normcdf(d2)-normpdf(d2)
#         println("ymin obj=",value, " ",stddev," ",y)
#         value
#     end    
#     bracket = (-stddev0*5, stddev0*5)
#     # println("start ",vol)
#     ymin =     Roots.find_zero(objMax,bracket,Roots.A42())+1e-10

#     objCall = function (y)
#         vol = sqrt(varianceByLogmoneynessFunction(y))
#         deltam = deltaForLogmoneyness(convention, y, vol,tte,forward,df)
#         #println(y, " LFDC ",vol," ",deltam," ",delta-deltam)
#         deltam - delta
#     end
#     if objCall(ymin) <= 0
#         return ymin
#     end
#     if objCall(stddev0*10) >= 0
#         return stddev0*10
#     end
#    # objCallDer = x -> ForwardDiff.derivative(objCall, x)
#    # k = logmoneynessForDelta(CallForwardDelta(), delta, vol, tte, forward, df)
#     return try
#         Roots.find_zero(objCall, (ymin, stddev0*10), Roots.A42())
#         catch exception
#             println(objCall(ymin), " ",objCall(stddev0*10))
#             throw(exception)
#         end


   
    # ymax = solveLogmoneynessForDelta(CallForwardDelta(),delta, varianceByLogmoneynessFunction, tte, forward,df)
    # obj = function (y)
    #     volm = sqrt(varianceByLogmoneynessFunction(y))
    #     deltam = deltaForLogmoneyness(convention, y, volm,tte,forward,df)
    #     # #println("solveLC ",y, " ",volm," ",deltam, " ",delta)
    #     deltam - delta
    #  end
    # #return Roots.find_zero(obj,0.0, Roots.Secant(),atol=1e-8)
    # return Roots.find_zero(obj,(0.0,ymax), Roots.A42(),atol=1e-8)
 
    obj = function (y)
        volm = sqrt(varianceByLogmoneynessFunction(y))
        # deltam = deltaForLogmoneyness(convention, y, volm,tte,forward,df)
        # #println("solveLC ",y, " ",volm," ",deltam, " ",delta)
        # deltam - delta
        ym = logmoneynessForDelta(convention, delta, volm, tte, forward, df)
    #    println(volm," ",ym-y, " ",ym, " ",y)
        ym - y
     end
    # # objDer = x -> ForwardDiff.derivative(obj, x)
    # # return Roots.find_zero((obj, objDer), 0.0, Roots.Newton())
    #  atmVolTte = sqrt(varianceByLogmoneynessFunction(0.0)*tte)
    #   return Roots.find_zero(obj,(-8atmVolTte,8atmVolTte), Roots.A42(),atol=1e-7)
    #println("ymax ",ymin," ", varianceByLogmoneynessFunction(ymin), " ",obj(-10.0)," ",obj(-0.5)," ",obj(0.0)," ",obj(0.01)," ",obj(1.0))
    return Roots.find_zero(obj,0.0, Roots.Secant(),atol=1e-8)
    # # atmVolTte = sqrt(varianceByLogmoneynessFunction(0.0)*tte)
    # # objAtm = obj(0.0)
    # # objYmax = obj(8atmVolTte)
    # # return if (sign(objAtm) != sign(objYmax))
    # #     Roots.find_zero(obj,(0.0,8atmVolTte), Roots.A42(),atol=1e-7)
    # # else
    # #     objYmin = obj(-8atmVolTte)
    # #     if (sign(objAtm) != sign(objYmin))
    # #         Roots.find_zero(obj,(-8atmVolTte,0.0), Roots.A42(),atol=1e-7)
    # #     else
    # ##          Roots.find_zero(obj,(-10atmVolTte,10atmVolTte), Roots.A42(),atol=1e-7)
    # #     end
    # # end 
    # #ym = logmoneynessForDelta(convention, delta, vol, tte, forward, df)
end

function solveLogmoneynessForDelta(convention::Union{ATMStraddle,ATMPremiumStraddle}, delta,  varianceByLogmoneynessFunction::F ,tte,forward,df) where {F <: Function}
    obj = function (y)
        y - logmoneynessForDelta(convention, sqrt(varianceByLogmoneynessFunction(y)) ,tte,forward,df)
    end
    #objDer = x -> ForwardDiff.derivative(obj, x)
#    return Roots.find_zero((obj, objDer), 0.0, Roots.Newton())
    return Roots.find_zero(obj,0.0, Roots.Secant())
end
