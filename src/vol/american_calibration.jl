#Calibration of the carry (discrete dividend amount or growth rate) and of the implied volatilities of listed American
#options from their market prices. The pricer is generic: any `p` with `priceAmerican(p, market, isCall, strike, τ, σ)` and
#`similar(p)` (for multithreading), e.g. AQFED.FDM.AmericanFDPricer or AQFED.American.JuZhongPricer.
import AQFED.American: priceAmerican
import AQFED.TermStructure: DiscreteDividendMarket, Dividend, forward, discountFactor, withDividendAmount, withGrowthRate
import AQFED.Black: impliedVolatility
import Roots: find_zero, A42
import GaussNewton

export AmericanOptionQuote, AmericanOptionSection, AmericanImpliedVolQuote, AmericanSectionCalibration
export forwardFromPutCallParity, calibrateCarryAndATMVols, americanImpliedVolatility, americanImpliedVolatilities, calibrateAmericanSection
export otmVolatility

struct AmericanOptionQuote{T}
    strike::T
    callBid::T
    callAsk::T
    callVolume::T
    putBid::T
    putAsk::T
    putVolume::T
end
callMid(q::AmericanOptionQuote) = (q.callBid + q.callAsk) / 2
putMid(q::AmericanOptionQuote) = (q.putBid + q.putAsk) / 2
#a strike is used when it has a price and some volume.
isQuoted(q::AmericanOptionQuote) = (callMid(q) != 0 || putMid(q) != 0) && (q.callVolume != 0 || q.putVolume != 0)

#listed options of a single maturity τ (year fraction), sorted by strike.
struct AmericanOptionSection{T}
    spot::T
    τ::T
    quotes::Vector{AmericanOptionQuote{T}}
    AmericanOptionSection(spot::T, τ::T, quotes::AbstractVector{AmericanOptionQuote{T}}) where {T} = new{T}(spot, τ, sort(collect(quotes), by = q -> q.strike))
end

#forward implied by put-call parity on mid prices, linearly interpolated between the two strikes bracketing the guess.
#For American options the parity is only approximate: this is a starting point for the carry calibration.
function forwardFromPutCallParity(section::AmericanOptionSection{T}, discountDf::T; forwardGuess::T = section.spot) where {T}
    quotes = filter(q -> q.callBid > 0 && q.putBid > 0, section.quotes)
    length(quotes) < 2 && throw(DomainError(length(quotes), "at least two strikes with call and put prices are required"))
    i = clamp(searchsortedfirst([q.strike for q in quotes], forwardGuess), 2, length(quotes))
    fParity(q) = q.strike + (callMid(q) - putMid(q)) / discountDf
    q0, q1 = quotes[i-1], quotes[i]
    return (fParity(q0) * (q1.strike - forwardGuess) + fParity(q1) * (forwardGuess - q0.strike)) / (q1.strike - q0.strike)
end

#applies f(pricer, i) to out[i] for i in 1:n, in chunks running in parallel, one pricer workspace per chunk.
function tmap!(f, out::AbstractVector, n::Int, pricers::AbstractVector)
    nChunks = min(length(pricers), n)
    if nChunks <= 1
        for i = 1:n
            out[i] = f(pricers[1], i)
        end
    else
        chunks = Iterators.partition(1:n, cld(n, nChunks))
        tasks = [Threads.@spawn(for i in chunk
            out[i] = f(pricers[c], i)
        end) for (c, chunk) in enumerate(chunks)]
        foreach(wait, tasks)
    end
    return out
end

makePricers(pricer, nThreads::Int) = vcat([pricer], [similar(pricer) for _ = 2:max(1, nThreads)])

#carry calibrated jointly with the ATM volatilities.
#  :dividend, the common amount of the dividends of ex-date before the maturity,
#  :growth, the growth rate μ of the market (e.g. to capture a borrow cost when there is no dividend before the maturity).
function carryVariable(market::DiscreteDividendMarket{T}, carry::Symbol, τ::T) where {T}
    if carry == :dividend
        any(d -> d.exDate <= τ, market.dividends) || throw(DomainError(carry, "no dividend before the maturity to calibrate"))
        return first(d for d in market.dividends if d.exDate <= τ).amount
    elseif carry == :growth
        return market.μ
    end
    throw(DomainError(carry, "carry must be :dividend or :growth"))
end
applyCarry(market::DiscreteDividendMarket{T}, carry::Symbol, c::T, τ::T) where {T} = carry == :dividend ? withDividendAmount(market, c, τ) : withGrowthRate(market, c)

#initial carry consistent with the forward f.
function initialCarry(market::DiscreteDividendMarket{T}, carry::Symbol, f::T, τ::T) where {T}
    if carry == :growth
        return log(f / market.spot) / τ - log(forward(withGrowthRate(market, zero(T)), τ) / market.spot) / τ
    end
    divs = filter(d -> d.exDate <= τ, market.dividends)
    if all(d -> d.isProportional, divs)
        return 1 - (f / market.spot * exp(-market.μ * τ))^(1 / length(divs))
    elseif all(d -> !d.isProportional, divs)
        return (market.spot - f * exp(-market.μ * τ)) / sum(d -> exp(-market.μ * d.payDate), divs)
    end
    return carryVariable(market, carry, τ)
end

#nATM strikes quoted for both calls and puts, closest to the forward.
function atmQuotes(section::AmericanOptionSection{T}, f::T, nATM::Int) where {T}
    quotes = filter(q -> callMid(q) > 0 && putMid(q) > 0, section.quotes)
    sort!(quotes, by = q -> abs(q.strike - f))
    return sort(quotes[1:min(nATM, length(quotes))], by = q -> q.strike)
end

#Calibrates the carry and the volatilities σᵢ of the ATM strikes such that the American call and put prices of a same strike
#share the same volatility: least squares on the relative errors price/marketMid - 1 of calls and puts.
#Returns (market, atmStrikes, atmVols, result) where `result` is the GaussNewton result, `nPrices` the number of prices computed.
function calibrateCarryAndATMVols(pricer, section::AmericanOptionSection{T}, market::DiscreteDividendMarket{T};
    carry::Symbol = :dividend, f::T = forward(market, section.τ), nATM::Int = 3, atmStrikes::AbstractVector{T} = T[],
    nThreads::Int = Threads.nthreads(), carryGuess::T = initialCarry(market, carry, f, section.τ), reltol::T = T(1e-8)) where {T}
    τ = section.τ
    df = discountFactor(market, τ)
    quotes = isempty(atmStrikes) ? atmQuotes(section, f, nATM) : [section.quotes[findfirst(q -> q.strike == k, section.quotes)] for k in atmStrikes]
    n = length(quotes)
    pricers = makePricers(pricer, nThreads)
    x0 = zeros(T, n + 1)
    x0[1] = carryGuess
    for (i, q) in enumerate(quotes) #European OTM vol as guess
        isCall = q.strike >= f
        x0[i+1] = impliedVolatility(isCall, isCall ? callMid(q) : putMid(q), f, q.strike, τ, df)
    end
    nPrices = Threads.Atomic{Int}(0)
    function residuals!(r, x)
        m = applyCarry(market, carry, x[1], τ)
        tmap!(r, 2n, pricers) do p, j
            i = j <= n ? j : j - n
            q = quotes[i]
            isCall = j <= n
            Threads.atomic_add!(nPrices, 1)
            priceAmerican(p, m, isCall, q.strike, τ, x[i+1]) / (isCall ? callMid(q) : putMid(q)) - 1
        end
    end
    #residuals of strike i depend only on the carry and σᵢ: bumping all the σ at once gives the diagonal part.
    r0 = zeros(T, 2n)
    rBump = zeros(T, 2n)
    xBump = similar(x0)
    function jacobian!(J, x)
        residuals!(r0, x)
        J .= zero(T)
        xBump .= x
        h = sqrt(eps(T))^(2 / 3) * max(abs(x[1]), T(1e-2))
        xBump[1] += h
        residuals!(rBump, xBump)
        @. J[:, 1] = (rBump - r0) / h
        xBump[1] = x[1]
        for i = 1:n
            xBump[i+1] = x[i+1] + sqrt(eps(T))^(2 / 3) * x[i+1]
        end
        residuals!(rBump, xBump)
        for i = 1:n
            hi = xBump[i+1] - x[i+1]
            J[i, i+1] = (rBump[i] - r0[i]) / hi
            J[n+i, i+1] = (rBump[n+i] - r0[n+i]) / hi
        end
        J
    end
    r = zeros(T, 2n)
    _, result = GaussNewton.optimize!(residuals!, jacobian!, x0, r, reltol = reltol, abstol = T(1e-14), stptol = T(1e-10))
    x = result.minimizer
    return (market = applyCarry(market, carry, x[1], τ), atmStrikes = [q.strike for q in quotes], atmVols = x[2:end], result = result, nPrices = nPrices[])
end

#volatility such that the American price matches `price`, by TOMS748 (Roots.A42). NaN when the price is not above the
#exercise value or when no volatility in [σMin,σMax] matches. The search starts from the bracket
#[guess/bracketFactor, guess*bracketFactor] and extends it if necessary.
function americanImpliedVolatility(pricer, market::DiscreteDividendMarket{T}, isCall::Bool, price::T, strike::T, τ::T;
    guess::T = T(0.3), xtol::T = T(1e-6), σMin::T = T(1e-3), σMax::T = T(5), bracketFactor::T = T(1.25)) where {T}
    intrinsic = max(isCall ? market.spot - strike : strike - market.spot, zero(T))
    if !(price > intrinsic) || !isfinite(price) || !isfinite(guess) || guess <= 0
        return T(NaN)
    end
    obj(σ) = priceAmerican(pricer, market, isCall, strike, τ, σ) - price
    a = max(guess / bracketFactor, σMin)
    fa = obj(a)
    while fa > 0 && a > σMin
        a = max(a / 2, σMin)
        fa = obj(a)
    end
    fa > 0 && return T(NaN)
    b = min(guess * bracketFactor, σMax)
    fb = obj(b)
    while fb < 0 && b < σMax
        a, fa = b, fb
        b = min(2b, σMax)
        fb = obj(b)
    end
    fb < 0 && return T(NaN)
    fa == 0 && return a
    fb == 0 && return b
    return find_zero(obj, (a, b), A42(), xatol = xtol)
end

struct AmericanImpliedVolQuote{T}
    strike::T
    callVolume::T
    putVolume::T
    callMid::T #implied volatilities, NaN when not available
    callBid::T
    callAsk::T
    putMid::T
    putBid::T
    putAsk::T
end
#out-of-the-money volatility (put below the forward), and the corresponding volume (e.g. as weight for a smile fit).
otmVolatility(v::AmericanImpliedVolQuote, f) = v.strike > f ? (v.callMid, v.callVolume) : (v.putMid, v.putVolume)

#implied volatilities of the mid, bid and ask prices of the calls and puts of the quoted strikes, computed in parallel.
function americanImpliedVolatilities(pricer, section::AmericanOptionSection{T}, market::DiscreteDividendMarket{T};
    nThreads::Int = Threads.nthreads(), xtol::T = T(1e-6), quoteFilter = isQuoted, bracketFactor::T = T(1.25)) where {T}
    τ = section.τ
    f = forward(market, τ)
    df = discountFactor(market, τ)
    quotes = filter(quoteFilter, section.quotes)
    n = length(quotes)
    #European implied volatility as guess, for each strike
    guesses = map(quotes) do q
        isCall = callMid(q) != 0
        vol = try
            impliedVolatility(isCall, isCall ? callMid(q) : putMid(q), f, q.strike, τ, df)
        catch
            T(NaN)
        end
        isfinite(vol) && vol > 0 ? vol : T(0.3)
    end
    prices(q) = (callMid(q), q.callBid, q.callAsk, putMid(q), q.putBid, q.putAsk)
    vols = zeros(T, 6n)
    tmap!(vols, 6n, makePricers(pricer, nThreads)) do p, j
        i, k = divrem(j - 1, 6) .+ 1
        q = quotes[i]
        americanImpliedVolatility(p, market, k <= 3, prices(q)[k], q.strike, τ, guess = guesses[i], xtol = xtol, bracketFactor = bracketFactor)
    end
    return [AmericanImpliedVolQuote(q.strike, q.callVolume, q.putVolume, vols[6i-5:6i]...) for (i, q) in enumerate(quotes)]
end

struct AmericanSectionCalibration{T,R}
    market::DiscreteDividendMarket{T} #calibrated market (carry)
    τ::T
    parityForward::T
    atmStrikes::Vector{T}
    atmVols::Vector{T}
    carryResult::R
    vols::Vector{AmericanImpliedVolQuote{T}}
end

#Full calibration of a section: forward from put-call parity, carry with the ATM volatilities, then American implied
#volatilities of all quoted strikes. `carry=:auto` calibrates the dividend amount if a dividend goes ex before the maturity,
#and the growth rate otherwise. `carryPricer` may be a faster approximation (e.g. JuZhongPricer() without dividends).
function calibrateAmericanSection(section::AmericanOptionSection{T}, market::DiscreteDividendMarket{T};
    pricer = nothing, carryPricer = pricer, carry::Symbol = :auto, nATM::Int = 3, atmStrikes::AbstractVector{T} = T[],
    nThreads::Int = Threads.nthreads(), xtol::T = T(1e-6)) where {T}
    isnothing(pricer) && throw(ArgumentError("a pricer is required, e.g. AQFED.FDM.AmericanFDPricer()"))
    τ = section.τ
    if carry == :auto
        carry = any(d -> d.exDate <= τ, market.dividends) ? :dividend : :growth
    end
    f = forwardFromPutCallParity(section, discountFactor(market, τ), forwardGuess = forward(market, τ))
    c = calibrateCarryAndATMVols(carryPricer, section, market, carry = carry, f = f, nATM = nATM, atmStrikes = atmStrikes, nThreads = nThreads)
    vols = americanImpliedVolatilities(pricer, section, c.market, nThreads = nThreads, xtol = xtol)
    return AmericanSectionCalibration(c.market, τ, f, c.atmStrikes, c.atmVols, c.result, vols)
end
