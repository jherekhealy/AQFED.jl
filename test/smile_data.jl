# Market smiles shared by the tests and the book figures (book/ch07_*.jl), stored in test/data/smiles.
# Each file has the columns Strike, Vol, Weight, Forward, Expiry (one maturity per file).
using CSV

struct SmileData
    name::String
    tte::Float64
    forward::Float64
    strikes::Vector{Float64}
    vols::Vector{Float64}
    weights::Vector{Float64}
end

const SMILE_DIR = joinpath(@__DIR__, "data", "smiles")

function loadSmile(name::AbstractString)
    f = CSV.File(joinpath(SMILE_DIR, name * ".csv"))
    return SmileData(name, f.Expiry[1], f.Forward[1], collect(Float64, f.Strike), collect(Float64, f.Vol), collect(Float64, f.Weight))
end

logmoneyness(d::SmileData) = log.(d.strikes ./ d.forward)

#Kahale example: implied volatility surface on a grid of expiries and strikes
function loadKahaleSurface()
    f = CSV.File(joinpath(SMILE_DIR, "kahale.csv"))
    expiries = unique(collect(f.Expiry))
    strikes = unique(collect(f.Strike))
    vols = permutedims(reshape(collect(f.Vol), length(strikes), length(expiries)))
    return (spot=f.Spot[1], r=f.Rate[1], q=f.Dividend[1], expiries=expiries, strikes=strikes, vols=vols)
end

#implied volatility surface (Expiry, Forward, Strike, Vol) on a grid of expiries and strikes
function loadSurface(name::AbstractString)
    f = CSV.File(joinpath(SMILE_DIR, name * ".csv"))
    expiries = unique(collect(f.Expiry))
    strikes = unique(collect(f.Strike))
    forwards = [f.Forward[findfirst(==(t), f.Expiry)] for t in expiries]
    vols = permutedims(reshape(collect(f.Vol), length(strikes), length(expiries)))
    return (expiries=expiries, forwards=forwards, strikes=strikes, vols=vols)
end
