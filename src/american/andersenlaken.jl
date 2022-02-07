import AQFED.TermStructure: ConstantBlackModel
import AQFED.Black: blackScholesFormula
import Roots:find_zero, A42
export AndersenLakeNRepresentation, priceAmerican

#Andersen-Lake American option pricing under negative rates
struct AndersenLakeNRepresentation
    isCall::Bool
    model::ConstantBlackModel
    tauMax::Float64
    tauMaxOrig::Float64
    tauHat::Float64
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
    aUp = AndersenLakeRepresentation(model, tauMax, atol, nC, nIter, nTS1, nTS2, isCall=isCall, isLower=false)
    if (model.r < 0 && model.q < model.r ) 
        aDown = AndersenLakeRepresentation(model, tauMax, atol, nC, nIter, nTS1, nTS2, isCall=isCall, isLower=true)
        tauHat = aUp.tauHat
        tauStar = tauHat       
        #calculate intersection tauStar 
        logCapXD = log(aDown.capX)
        logCapX = log(aUp.capX)
        logBdown = logCapXD + sqrt(aDown.qvec[1])
        logBup = logCapX - sqrt(aUp.qvec[1])
        if logBdown > logBup 
            obj = function(τ)
                z = 2 * sqrt(( τ) / tauHat) - 1
                qck = chebQck(aUp.avec, z)
                lnBUp = logCapX - sqrt(qck)
                qck = chebQck(aDown.avec, z)
                lnBDown = logCapXD + sqrt(qck) 
                return lnBUp - lnBDown
            end
            tauStar = find_zero(obj, (0,tauHat), A42())
            println("tauStar ",tauStar)
        end
        return AndersenLakeNRepresentation(isCall, model, tauStar, tauMax, tauHat, nC, nTS1, nTS2,
        aUp.capX, aDown.capX, aUp.avec, aDown.avec, aUp.qvec, aDown.qvec, aUp.wvec, aUp.yvec)
    else 
        return AndersenLakeNRepresentation(isCall, model, tauMax, tauMax, tauMax, nC, nTS1, nTS2,
        aUp.capX, Float64[] , aUp.avec, Float64[], aUp.qvec, Float64[], aUp.wvec, aUp.yvec)
    end   
end