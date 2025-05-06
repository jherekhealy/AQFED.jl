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

function priceAsianSpread(
    p::VorstGeometricExpansion,
    isCall::Bool,
    strikePct::T,
    discountFactor::AbstractFloat,
    spot::T, 
    forward::AbstractArray{TV}, #forward to each Asian observation t_i. 
    totalVariance::AbstractArray{TV}, #vol^2 * t_i
    weight::AbstractArray{TV},#for now assume first weights are < 0 and laast wieghts are > 0 for a spread. This is call like spread.
)::Number where {TV<:Number, T <: Number}

    # if strike != zero(TK)
    #     forward = vcat(strike,forward)
    #     spot = vcat(strike,spot)
    #     totalVariance = vcat(zero(TV),totalVariance)
    #     weight = vcat(-one(TV),weight)
    # end
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
    price = priceEuropeanSpread(p, isCall, strikePct, discountFactor, spots, forward, totalVariance, weight, correlation)
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


function priceAsianBasketSpread(
    p::BasketPricer,
    isCall::Bool,
    strike::AbstractFloat,
    discountFactor::AbstractFloat,
    spot::AbstractVector, #first asset must be the negatively weighted one. 
    forward::AbstractMatrix{TV}, #forward for each asset F_i to each Asian observation t_j
    totalVariance::AbstractMatrix{TV}, #vol^2 * t_i
    weight::AbstractMatrix{TV}, #negative weights for asset in the negative part of the spread, positive wieghts otherwise. In particular averaging is always fully positive or fully negative.
    correlation::AbstractMatrix{TV}, #S_k S_j
    t::AbstractVector{TV}
)::Number where {TV<:Number}
#if strike != 0, add an additional date at t=0, with forward=K, tvar=0, weight=-1/nAsset? or an additional asset with weight=1,0,0,0,0,0.
    nAsset = size(totalVariance, 1)
    if strike != 0
        totalVariance = hcat(zeros(TV,nAsset),totalVariance)
        weight = hcat(zeros(TV,nAsset),weight)
        forward = hcat(spot,forward)
        weight[1,1] = -one(TV)
        forward[1,1] = strike
        t = vcat(zero(TV), t)
    end
    nDates = size(totalVariance,2)
    #println(weight)
    n = size(totalVariance, 1) * size(totalVariance, 2)
    spotA = zeros(TV, n)
    forwardA = zeros(TV, n)
    totalVarianceA = zeros(TV, n)
    weightA = zeros(TV, n)
    correlationA = zeros(TV, (n, n))
    for is = eachindex(spot)
        for j = 1:size(totalVariance, 2)
            spotA[(is-1)*nDates+j] = spot[is]
            forwardA[(is-1)*nDates+j] = forward[is, j]
            weightA[(is-1)*nDates+j] = weight[is, j]
            totalVarianceA[(is-1)*nDates+j] = totalVariance[is, j]
        end

        for i = 1:size(totalVariance, 2)
            vi = totalVariance[is, i]
            for js = eachindex(spot)
                for j = 1:size(totalVariance, 2)
                    vj = totalVariance[js, j]
                    #   if vi != 0 && vj != 0
                    if t[i] == zero(TV) || t[j] == zero(TV)
                        correlationA[(is-1)*nDates+i, (js-1)*nDates+j] = zero(TV)
                    else
                    correlationA[(is-1)*nDates+i, (js-1)*nDates+j] = correlation[is, js] * min(t[i], t[j]) / sqrt(t[i] * t[j])
                    end
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
    price = priceEuropeanSpread(p, isCall, 1.0, discountFactor, spotA, forwardA, totalVarianceA, weightA, correlationA)
    return price
end

function priceAsianFloatingStrike(
    p::BasketPricer,
    isCall::Bool,
    strikePercent::Number,
    discountFactor::Number,
    spot::Number,
    forward::Vector{TV}, #forward to each Asian observation t_i, last one is maturity.
    totalVariance::Vector{TV}, #vol^2 * t_i
    weight::Vector{TV}, #typically, weights[1:n-1] = 1/(n-1); weights[n] = 0
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

