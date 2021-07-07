#use basket pricer to price single asset Asians
#supports a term-structure of volatilities and drifts
#past observations may be recorded at time 0 using forward=sum(w*obs) and rescaling the weight.
export priceAsianFixedStrike, priceAsianFloatingStrike

function priceAsianFixedStrike(
    p::DeelstraBasketPricer,
    isCall::Bool,
    strike::T,
    discountFactor::T,
    spot::T,
    forward::Vector{T}, #forward to each Asian observation t_i
    totalVariance::Vector{T}, #vol^2 * t_i
    weight::Vector{T},
)::T where {T}

    nAsset = length(totalVariance)
    spots = fill(spot, nAsset)
    correlation = zeros(T, (nAsset, nAsset))
    for (i, vi) in enumerate(totalVariance)
        for j = 1:(i-1)
            vj = totalVariance[j]
            if vi != 0 && vj != 0
                correlation[i, j] = min(vi, vj) / sqrt(vi * vj)
                # else is zero
            end
        end
        correlation[i, i] = one(T)
    end
    for i = 1:nAsset
        for j = i+1:nAsset
            correlation[i, j] = correlation[j, i]
        end
    end
    price = priceEuropean(p, isCall, strike, discountFactor, spots, forward, totalVariance, weight, correlation)
    return price
end

function priceAsianFloatingStrike(
    p::DeelstraBasketPricer,
    isCall::Bool,
    strikePercent::T,
    discountFactor::T,
    spot::T,
    forward::Vector{T}, #forward to each Asian observation t_i
    totalVariance::Vector{T}, #vol^2 * t_i
    weight::Vector{T},
)::T where {T}
    #change of numeraire leads to equivalence with fixed strike put
    #floating S,k ,r,mu = k* fixed S/k,S,mu,r
    nAsset = len(totalVariance)
    dfAsset = discountFactor * forward[nAsset]
    forwardAsset = zeros(T, nAsset)
    for (i, fi) in enumerate(forward)
        forwardAsset[i] = fi / forward[nAsset]
    end
    #r,q = q,r => r,q-r = q-r,r
    tvarAsset = zeros(T, nAsset)
    correlation = zeros(T, (nAsset, nAsset))
    spots = ones(T, nAsset)
    stte = totalVariance[nAsset]
    for (i, vi) in enumerate(totalVariance)
        siti = stte - vi
        tvarAsset[i] = siti
        for j = 1:(i-1)
            sjtj = stte - totalVariance[j]
            if siti == zero(T) || sjtj == zero(T)
                correlation[i, j] = zero(T)
            else
                correlation[i, j] = min(siti, sjtj) / sqrt(siti * sjtj) #covar(xi,xj)/voli*volj = E(xi*xj - E(xi)*E(xj))/voli*volj =? min(ti-tj)/sqrt(ti)*sqrt(tj)
            end
        end
        correlation[i, i] = one(T)
    end
    for i = 1:nAsset
        for j = i+1:nAsset
            correlation[i, j] = correlation[j, i]
        end
    end

    priceFixed =
        strikePercent *
        priceEuropean(p, !isCall, spot / strikePercent, dfAsset, spots, forwardAsset, tvarAsset, weight, correlation)
    return priceFixed
end
