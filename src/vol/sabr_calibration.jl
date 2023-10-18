import AQFED.Math: inv as invTransform, TanhTransformation
import AQFED.TermStructure: SABRParams, SABRSection, Hagan2020,normalVarianceByMoneyness
#using Bachelier
using LeastSquaresOptim
import StatsBase: rmsd

#vol is black vol
function initialGuessBlackATM(f::Float64, term::Float64, β::Float64, vol::Float64, dvol::Float64, d2vol::Float64)
    oneBeta = 1 - β
    fonebeta = f^oneBeta
    alpha = vol * fonebeta
    temp = 2 * dvol + oneBeta * vol
    nuSq = 3 * vol * d2vol - 0.5 * oneBeta^2 * vol^2 + 1.5 * temp^2
    rho = 0.0
    nu = 0.0
    if nuSq <= 0
        if 2 * dvol + oneBeta * vol > 0
            rho = 1.0
        else
            rho = -1.0
        end
        nu = temp / rho
    else
        nu = sqrt(nuSq)
        rho = temp / nu
    end
    p0 = oneBeta^2 * term / (24 * fonebeta^2)
    p1 = rho * β * nu * term / (4 * fonebeta)
    p2 = 1 + (2 - 3 * rho^2) / 24 * nu^2 * term
    p3 = -vol * fonebeta
    r = if p0 != 0
        cubicRootsReal(p1/ p0, p2 / p0, p3 / p0)
     else
       quadRootsReal(p1,p2,p3)
      end
    smallestRealRoot = typemax(Float64)
    for ri = r
        if ri > 0 && ri < smallestRealRoot
            smallestRealRoot = ri
        end
    end
    α = if smallestRealRoot == typemax(Float64)
        1.0
    else
        smallestRealRoot
    end

    return SABRParams(α, β, rho, nu)
end

#vol is normal vol
function initialGuessNormalATM(f::Float64, term::Float64, β::Float64, vol::Float64, dvol::Float64, d2vol::Float64)
	fbeta = f^β
	alpha = vol / fbeta
    nuSq = (3*vol*d2vol-(β^2+β)/2*vol^2+3vol*(dvol-β/2*vol)+3/2*(2dvol-β*vol)^2)/f^2
	nu = sqrt(nuSq)
	rho = (2*dvol-β*vol)/(f*nu)

    p0 = (β^2-2β)*term/(24*f^2)*fbeta^2
	p1 = rho*β*nu*term/(4*f)*fbeta
	p2 = 1 + (2 - 3 * rho^2) / 24 * nuSq * term
    p3 = -vol /fbeta
	r = if p0 != 0
    	 cubicRootsReal(p1/ p0, p2 / p0, p3 / p0)
	else
		quadRootsReal(p1,p2,p3)
	end
    smallestRealRoot = typemax(Float64)
    for ri = r
        if ri > 0 && ri < smallestRealRoot
            smallestRealRoot = ri
        end
    end
    α = if smallestRealRoot == typemax(Float64)
        1.0
    else
        smallestRealRoot
    end

    return SABRParams(α, β, rho, nu)
end
function calibrateSABRSectionFromGuess(tte::T, forward::T, ys::AbstractArray{T}, blackVols::AbstractArray{T}, weights::AbstractArray{T}, guess::SABRParams) where {T}
    w = sqrt.(weights)
    w = w ./ sum(w)
    ρTrans = TanhTransformation(-1.0,1.0) 
    νTrans = MQMinTransformation(0.0,1.0)
    function obj!(fvec::Z, c::AbstractArray{W}) where {Z,W}
        p = SABRParams(c[1], guess.β, ρTrans(c[2]), νTrans(c[3]))
       s = SABRSection(Hagan2020(), p, tte, forward, 0.0)
        for (i, yi) = enumerate(ys)
            vol = sqrt(varianceByLogmoneyness(s, yi))
            fvec[i] = w[i] * (vol - blackVols[i])
        end
    end
    ρinv = inv(ρTrans, guess.ρ)
    νinv = inv(νTrans, guess.ν)
    fit = LeastSquaresOptim.optimize!(
        LeastSquaresProblem(x=[guess.α,ρinv,νinv], (f!)=obj!, autodiff=:central, #:forward is 4x faster than :central
            output_length=length(blackVols)),
        LevenbergMarquardt();
        iterations=1000
    )
    #println("sabr fit ",fit)
    # fvec = zeros(Float64, length(callPrices))
    # obj!(fvec, fit.minimizer)
    c = fit.minimizer
    p = SABRParams(c[1], guess.β, ρTrans(c[2]), νTrans(c[3]))
    s = SABRSection(Hagan2020(), p, tte, forward, 0.0)
    return s
end

function calibrateNormalSABRSectionFromGuess(tte::T, forward::T, strikes::AbstractArray{T}, normalVols::AbstractArray{T}, weights::AbstractArray{T}, guess::SABRParams) where {T}
    w = sqrt.(weights)
    w = w ./ sum(w)
	ρTrans = TanhTransformation(-1.0,1.0) 
    function obj!(fvec::Z, c::AbstractArray{W}) where {Z,W}
        p = SABRParams(c[1], guess.β, ρTrans(c[2]), c[3])
        s = SABRSection(Hagan2020(), p, tte, forward, 0.0)
        for (i, strikei) = enumerate(strikes)
            vol = sqrt(normalVarianceByMoneyness(s, strikei-forward))
            fvec[i] = w[i] * (vol - normalVols[i])
        end
    end
	ρinv = inv(ρTrans, guess.ρ)
    fit = LeastSquaresOptim.optimize!(
        LeastSquaresProblem(x=[guess.α,ρinv,guess.ν], (f!)=obj!, autodiff=:central, #:forward is 4x faster than :central
            output_length=length(normalVols)),
        LevenbergMarquardt();
        iterations=1000
    )
    c = fit.minimizer
    p = SABRParams(c[1], guess.β, ρTrans(c[2]), c[3])
    s = SABRSection(Hagan2020(), p, tte, forward, 0.0)
    return s
end

function computeRMSE(params::SABRSection, y::AbstractArray{T}, vols::AbstractArray{T}, weights::AbstractArray{T}) where {T}
    rmsd(weights .* sqrt.(varianceByLogmoneyness.(params, y)), weights .* vols)
end
