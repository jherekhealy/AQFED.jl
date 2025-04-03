#use basket pricer to price single asset Asians
#supports a term-structure of volatilities and drifts
#past observations may be recorded at time 0 using forward=sum(w*obs) and rescaling the weight.
using AQFED.Basket
export priceAsianFixedStrike, priceAsianFloatingStrike

function priceAsianFixedStrike(
    p::BasketPricer,
    isCall::Bool,
    strike::AbstractFloat,
    discountFactor::AbstractFloat,
    spot::AbstractFloat,
    forward::AbstractArray{TV}, #forward to each Asian observation t_i
    totalVariance::AbstractArray{TV}, #vol^2 * t_i
    weight::AbstractArray{TV},
)::Number where {TV<:Number}

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


function priceAsianBasketFixedStrike(
    p::BasketPricer,
    isCall::Bool,
    strike::AbstractFloat,
    discountFactor::AbstractFloat,
    spot::AbstractVector,
    forward::AbstractMatrix{TV}, #forward for each asset F_i to each Asian observation t_j
    totalVariance::AbstractMatrix{TV}, #vol^2 * t_i
    weight::AbstractMatrix{TV},
    correlation::AbstractMatrix{TV}, #S_k S_j
    t::AbstractVector{TV}
)::Number where {TV<:Number}

    nAsset = size(totalVariance, 1)
    n = size(totalVariance, 1) * size(totalVariance, 2)
    spotA = zeros(TV, n)
    forwardA = zeros(TV, n)
    totalVarianceA = zeros(TV, n)
    weightA = zeros(TV, n)
    correlationA = zeros(TV, (n, n))
    for is = eachindex(spot)
        for j = 1:size(totalVariance, 2)
            spotA[(is-1)*nAsset+j] = spot[is]
            forwardA[(is-1)*nAsset+j] = forward[is, j]
            weightA[(is-1)*nAsset+j] = weight[is, j]
            totalVarianceA[(is-1)*nAsset+j] = totalVariance[is, j]
        end

        for i = 1:size(totalVariance, 2)
            vi = totalVariance[is, i]
            for js = eachindex(spot)
                for j = 1:size(totalVariance, 2)
                    vj = totalVariance[js, j]
                    #   if vi != 0 && vj != 0
                    correlationA[(is-1)*nAsset+i, (js-1)*nAsset+j] = correlation[is, js] * min(t[i], t[j]) / sqrt(t[i] * t[j])
                    # else is zero
                    #  end
                end
            end
            #  correlationA[(is-1)*nAsset+i, (is-1)*nAsset+i] = one(Float64)
        end
    end
    # println(correlationA)
    # for i = 1:n
    #     for j = i+1:n
    #         correlationA[i, j] = correlationA[j, i]
    #     end
    # end
    price = priceEuropean(p, isCall, strike, discountFactor, spotA, forwardA, totalVarianceA, weightA, correlationA)
    return price
end

function priceAsianFloatingStrike(
    p::BasketPricer,
    isCall::Bool,
    strikePercent::Number,
    discountFactor::Number,
    spot::Number,
    forward::Vector{TV}, #forward to each Asian observation t_i
    totalVariance::Vector{TV}, #vol^2 * t_i
    weight::Vector{TV},
)::Number where {TV<:Number}
    if length(weight) == length(totalVariance) - 1
        weight = vcat(weight, zero(TV))
    else
        weight[end] = zero(TV)
    end

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
        priceEuropean(p, !isCall, 1.0 / strikePercent, dfAsset, spots, forwardAsset, tvarAsset, weight, correlation)
    return priceFixed
end
