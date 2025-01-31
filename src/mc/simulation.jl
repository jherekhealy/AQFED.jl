using Statistics

function simulate(
    pathgen::PathGenerator{T},
    payoff::MCPayoff{T},
    start::Int,
    nSim::Int
) where {T}
specTimes = specificTimes(payoff)
tte = specTimes[end]
df = discountFactor(model, tte)
genTimes = pathgenTimes(pathgen, specTimes)
  
#for each simtime
#   calculate x(simtime)
#   if isObservationTime(simtime) #we know in advance all obs times
#      eventually transform x to exp(x)
#      advancePayoff(payoff, simtime, x) #record x eventually in "cashflow"
#      if hasPayment(payoff)
#        pv += df(paymentTime(payoff)) * evaluatePayoff(payoff, simtime) #df may be calculated of x, a discounter(x) but not a payoff, .
#                   no need for df(time) index since we do it only once for all paths. Could have a cache otherwise. what if multiple payments different pay date. could have a while (payment), advancePayment

pv =  Vector{T}(undef, nSim)
for (dim, t1) in enumerate(genTimes[2:end])
    evolvePathValues(pathgen, start, dim, t1)
    if isObservationTime(payoff, t1)
        x = getPathValues(pathgen, dim, t1) #x is 1D or not but PV is 1D!
        advancePayoff(payoff, t1, x)
        while hasNextPayment(payoff)
            @. pv +=  df(paymentTime(payoff)) * evaluatePayoff(payoff, t1) #df may be calculated of x, a discounter(x) but not a payoff, .
        end
    end
end
payoffMean = mean(pv)
return payoffMean, stdm(pv, payoffMean) / sqrt(length(pv))
end

