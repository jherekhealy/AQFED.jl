#use basket pricer to price single asset Asians
#supports a term-structure of volatilities and drifts
#past observations may be recorded at time 0 using forward=sum(w*obs) and rescaling the weight.
export priceAsianFixedStrike, priceAsianFloatingStrike

function priceAsianFixedStrike(
    p::DeelstraBasketPricer,
    isCall::Bool,
    strike::AbstractFloat,
    discountFactor::AbstractFloat,
    spot::AbstractFloat,
    forward::AbstractArray{<:Number}, #forward to each Asian observation t_i
    totalVariance::AbstractArray{<:Number}, #vol^2 * t_i
    weight::AbstractArray{<:Number},
)::Number where {TV <: Number}

    nAsset = length(totalVariance)
    spots = fill(spot, nAsset)
    correlation = zeros(eltype(totalVariance), (nAsset, nAsset))
    for (i, vi) in enumerate(totalVariance)
        for j = 1:(i-1)
            vj = totalVariance[j]
            if vi != 0 && vj != 0
                correlation[i, j] = min(vi, vj) / sqrt(vi * vj)
                # else is zero
            end
        end
        correlation[i, i] = one(Float64)
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
    strikePercent::Number,
    discountFactor::Number,
    spot::Number,
    forward::Vector{<:Number}, #forward to each Asian observation t_i
    totalVariance::Vector{<:Number}, #vol^2 * t_i
    weight::Vector{<:Number},
)::Number
    #change of numeraire leads to equivalence with fixed strike put
    #floating S,k ,r,mu = k* fixed S/k,S,mu,r
    nAsset = length(totalVariance)
    dfAsset = discountFactor * forward[nAsset]
    forwardAsset = forward ./ forward[nAsset]
    #r,q = q,r => r,q-r = q-r,r
    tvarAsset = zeros(eltype(totalVariance), nAsset)
    correlation = zeros(eltype(totalVariance), (nAsset, nAsset))
    spots = ones(Float64, nAsset)
    stte = totalVariance[nAsset]
    for (i, vi) in enumerate(totalVariance)
        siti = stte - vi
        tvarAsset[i] = siti
        for j = 1:(i-1)
            sjtj = stte - totalVariance[j]
            if siti == zero(Float64) || sjtj == zero(Float64)
                correlation[i, j] = zero(Float64)
            else
                correlation[i, j] = min(siti, sjtj) / sqrt(siti * sjtj) #covar(xi,xj)/voli*volj = E(xi*xj - E(xi)*E(xj))/voli*volj =? min(ti-tj)/sqrt(ti)*sqrt(tj)
            end
        end
        correlation[i, i] = one(Float64)
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
