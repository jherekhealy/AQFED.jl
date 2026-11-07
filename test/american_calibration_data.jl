using Dates, CSV, DataFrames
import AQFED.TermStructure: Dividend
import AQFED.VolatilityModels: AmericanOptionQuote, AmericanOptionSection

#Option chain of a single maturity: one row per strike with the columns valuation_date, maturity_date, spot, strike,
#call_bid, call_ask, call_volume, put_bid, put_ask, put_volume (converted from a Nasdaq export).
#Returns the section (Act/365 maturity) and the valuation date.
function loadAmericanSection(filename::String)
    df = DataFrame(CSV.File(filename))
    valuationDate = df.valuation_date[1]
    τ = Dates.value(df.maturity_date[1] - valuationDate) / 365
    quotes = [AmericanOptionQuote(Float64.((r.strike, r.call_bid, r.call_ask, r.call_volume, r.put_bid, r.put_ask, r.put_volume))...) for r in eachrow(df)]
    return AmericanOptionSection(Float64(df.spot[1]), τ, quotes), valuationDate
end

#AAPL quarterly dividends of 0.52, modelled as proportional, ex-dates from Feb 4 2016 (16:40), paid 7 days later.
function aaplDividends(valuationDate::Date, spot::Float64; n::Int = 20, amount::Float64 = 0.52)
    valuation = DateTime(valuationDate)
    ex = DateTime(2016, 2, 4, 16, 40)
    while ex < valuation
        ex += Month(3)
    end
    yf(d) = Dates.value(d - valuation) / (86400000 * 365)
    divs = Dividend{Float64}[]
    for i = 1:n
        push!(divs, Dividend(amount / spot, yf(ex), yf(ex + Day(7)), true, false))
        ex += Month(3)
    end
    return divs
end
