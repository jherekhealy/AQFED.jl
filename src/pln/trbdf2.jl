export DividendPolicy, priceEuropeanTRBDF2

using LinearAlgebra
using Dierckx
using PPInterpolation
@enum DividendPolicy begin
    Liquidator
    Survivor
    Shift
end
function priceEuropeanTRBDF2(isCall::Bool,
    strike::T,
    spot::T,
    rawForward::T, #The raw forward to τ (without cash dividends)
    variance::T, #variance to maturity
    τ::T,
    discountDf::T, #discount factor to payment date
    dividends::AbstractArray{CapitalizedDividend{T}}; M=400, N=100, ndev=4, dividendPolicy::DividendPolicy=Liquidator, grid="", alpha=0.01,useSpline=true) where {T}
    r = -log(discountDf) / τ
    q = r - log(rawForward / spot) / τ
    sigma = sqrt(variance / τ)
    xi = (range(zero(T), stop=one(T), length=M))
    Li = zero(T)
    Ui = rawForward * exp(ndev * sqrt(variance))
    Si = @. Li + xi * (Ui - Li)
    if grid == "Cubic"
        Si = makeCubicGrid(xi, Li, Ui, [Li, strike, Ui], alpha, shift=0.5)
    end
    t = collect(range(τ, stop=zero(T), length=N))
    dividends = filter(x -> x.dividend.exDate <= τ, dividends)
    sort!(dividends, by=x -> x.dividend.exDate)
    divDates = [x.dividend.exDate for x in dividends]
    t = vcat(t, divDates)
    sort!(t, order=Base.Order.Reverse)
    #    println("S ",Si)
    tip = t[1]
    v = @. max(Si - strike, zero(T))
    Jhi = @. (Si[2:end] - Si[1:end-1])
    rhsd = Array{T}(undef, length(Si))
    lhsd = ones(T, length(Si))
    rhsdl = Array{T}(undef, length(Si) - 1)
    lhsdl = Array{T}(undef, length(Si) - 1)
    rhsdu = Array{T}(undef, length(Si) - 1)
    lhsdu = Array{T}(undef, length(Si) - 1)
    lhs = Tridiagonal(lhsdl, lhsd, lhsdu)
    # lhsf = lu(lhs)
    rhs = Tridiagonal(rhsdl, rhsd, rhsdu)
    v0 = Array{T}(undef, length(Si))
    v1 = Array{T}(undef, length(Si))
    pp = PPInterpolation.PP(3, T, T, length(Si))
    currentDivIndex = length(dividends)
    if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
        #jump and interpolate        
        # pp = Spline1D(Si,v)
        PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())
       

        if dividendPolicy == Shift
            @. v = pp(Si - dividends[currentDivIndex].dividend.amount, zero(T))
        elseif dividendPolicy == Survivor
            @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
        else #liquidator
            ppIndex = 1
            for j=1:length(v)
                z = max(Si[j] - dividends[currentDivIndex].dividend.amount, zero(T))
                while (ppIndex < length(Si) && pp.x[ppIndex]< z) #Si[ppIndex]<=z<Si[ppIndex+1]  
                    ppIndex+=1
                end
                if (z != pp.x[ppIndex])
                    ppIndex-=1
                end
                v[j]= PPInterpolation.evaluate(pp, ppIndex, z)
            end
            # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
        end
        currentDivIndex -= 1
    end
    beta = 2 * one(T) - sqrt(2 * one(T))
    for i = 2:length(t)
        ti = t[i]
        dt = tip - ti
        if dt < 1e-8
            continue
        end
        v0[1:end] = v
        @inbounds for j = 2:M-1
            s2S = sigma^2 * Si[j]^2
            muS = (r - q) * Si[j]
            rhsd[j] = one(T) - dt * beta / 2 * ((muS * (Jhi[j-1] - Jhi[j]) + s2S) / (Jhi[j] * Jhi[j-1]) + r)
            rhsdu[j] = dt * beta / 2 * (s2S + muS * Jhi[j-1]) / (Jhi[j] * (Jhi[j] + Jhi[j-1]))
            rhsdl[j-1] = dt * beta / 2 * (s2S - muS * Jhi[j]) / (Jhi[j-1] * (Jhi[j] + Jhi[j-1]))
        end
        if false && isCall
            v[1] = zero(T)
            rhsd[1] = one(T)
            rhsdu[1] = zero(T)
        else
            #linear or Ke-rt same thing
            rhsd[1] = one(T) - dt * beta / 2 * (r + (r - q) * Si[1] / Jhi[1])
            rhsdu[1] = dt * beta / 2 * (r - q) * Si[1] / Jhi[1]
        end

        rhsd[M] = one(T) - dt * beta / 2 * (r - (r - q) * Si[end] / Jhi[end])
        rhsdl[M-1] = -dt * beta / 2 * (r - q) * Si[end] / Jhi[end]
        mul!(v1, rhs, v)
        @. lhsd = one(T) - (rhsd - one(T))
        @. lhsdu = -rhsdu
        @. lhsdl = -rhsdl
        # lhsf = lu!(lhs)
        # lhsf = factorize(lhs)
        # ldiv!(v, lhsf , v1)
        TDMA!(v, lhsdl, lhsd, lhsdu, v1)

        #BDF2 step
        @. v1 = (v - (1 - beta)^2 * v0) / (beta * (2 - beta))
        # ldiv!(v , lhsf ,v1)
        TDMA!(v, lhsdl, lhsd, lhsdu, v1)

        tip = ti
        if (currentDivIndex > 0 && tip == divDates[currentDivIndex])
            #jump and interpolate
            # pp = Spline1D(Si,v)
            PPInterpolation.computePP(pp, Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), VanAlbada())
            if dividendPolicy == Shift
                @. v = pp(Si - dividends[currentDivIndex].dividend.amount, zero(T))
            elseif dividendPolicy == Survivor
                @. v = pp(ifelse(Si - dividends[currentDivIndex].dividend.amount < zero(T), Si, Si - dividends[currentDivIndex].dividend.amount))
            else #liquidator
                #    @. v = pp(max(Si - dividends[currentDivIndex].dividend.amount, zero(T)))
                ppIndex = 1
            for j=1:length(v)
                z = max(Si[j] - dividends[currentDivIndex].dividend.amount, zero(T))
                while (ppIndex < length(Si) && (Si[ppIndex] < z)) #Si[ppIndex]<=z<Si[ppIndex+1]  
                    ppIndex+=1
                end
                if (z != Si[ppIndex])
                    ppIndex-=1
                end
                #  ppIndex = min(max(ppIndex,2),length(Si)-1)
                #  v1[j] = v[ppIndex]*(Si[ppIndex-1]-z)*(Si[ppIndex+1]-z)/((Si[ppIndex-1]-Si[ppIndex])*(Si[ppIndex+1]-Si[ppIndex]))+v[ppIndex-1]*(Si[ppIndex]-z)*(Si[ppIndex+1]-z)/((Si[ppIndex]-Si[ppIndex-1])*(Si[ppIndex+1]-Si[ppIndex-1]))+v[ppIndex+1]*(Si[ppIndex-1]-z)*(Si[ppIndex]-z)/((Si[ppIndex-1]-Si[ppIndex+1])*(Si[ppIndex]-Si[ppIndex+1]))
               v1[j]= PPInterpolation.evaluate(pp, ppIndex, z)
            end # println("jumped ",currentDivIndex, " of ",dividends[currentDivIndex].dividend.amount," tip ",tip)
            v[1:end]=v1
            end
            currentDivIndex -= 1
        end
    end
    # spl = Spline1D(Si, v)
    spl = makeCubicPP(Si, v, PPInterpolation.SECOND_DERIVATIVE, zero(T), PPInterpolation.SECOND_DERIVATIVE, zero(T), C2())

    return spl
end

using PolynomialRoots
function makeCubicGrid(xi::AbstractArray{T}, Smin::T, Smax::T, starPoints::AbstractArray{T}, alpha::T; shift=0.0) where {T}
    alphaScaled = alpha * (Smax - Smin)
    coeff = one(T) / 6
    starMid = zeros(T, length(starPoints) + 1)
    starMid[1] = Smin
    starMid[2:end-1] = (starPoints[1:end-1] + starPoints[2:end]) / 2
    starMid[end] = Smax
    c1 = zeros(T, length(starPoints))
    c2 = zeros(T, length(starPoints))
    for i = 1:length(starPoints)
        local r = filter(isreal, PolynomialRoots.roots([(starPoints[i] - starMid[i]) / alphaScaled, one(T), zero(T), coeff]))
        c1[i] = real(sort(r)[1])
        local r = filter(isreal, PolynomialRoots.roots([(starPoints[i] - starMid[i+1]) / alphaScaled, one(T), zero(T), coeff]))
        c2[i] = real(sort(r)[1])
    end
    dd = Array{T}(undef, length(starPoints) + 1)
    dl = Array{T}(undef, length(starPoints))
    dr = Array{T}(undef, length(starPoints))
    @. dl[1:end-1] = -alphaScaled * (3 * coeff * (c2[2:end] - c1[2:end]) * c1[2:end]^2 + c2[2:end] - c1[2:end])
    @. dr[2:end] = -alphaScaled * (3 * coeff * (c2[1:end-1] - c1[1:end-1]) * c2[1:end-1]^2 + c2[1:end-1] - c1[1:end-1])
    dd[2:end-1] = -dl[1:end-1] - dr[2:end]
    dd[1] = one(T)
    dd[end] = one(T)
    rhs = zeros(Float64, length(dd))
    rhs[end] = one(T)
    lhs = Tridiagonal(dl, dd, dr)
    local d = lhs \ rhs
    #  println("d ",d)
    @. c1 /= d[2:end] - d[1:end-1]
    @. c2 /= d[2:end] - d[1:end-1]
    #now transform
    dIndex = 2
    Sip = Array{Float64}(undef, length(xi))
    for i = 2:length(xi)-1
        ui = xi[i]
        while (dIndex <= length(d) && d[dIndex] < ui)
            dIndex += 1
        end
        dIndex = min(dIndex, length(d))
        t = c2[dIndex-1] * (ui - d[dIndex-1]) + c1[dIndex-1] * (d[dIndex] - ui)
        Sip[i] = starPoints[dIndex-1] + alphaScaled * t * (coeff * t^2 + 1)
    end
    Sip[1] = Smin
    if (shift != 0)
        Sip[1] -= (Sip[2] - Smin)
    end
    Sip[end] = Smax
    if (shift != 0)
        Sip[end] += (Smax - Sip[end-1])
    end
    return Sip
end

function TDMA!(x::AbstractArray{T}, a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, d::AbstractArray{T}) where {T}
    #  Thomas Algorithm
    #  Vectors a,c of length(b)-1    
    #  A*x = D
    #  [b[1]   c[1]               ] [ x[1] ]   [ D[1] ]
    #  [a[1]   b[2]   c[2]        ] [ x[2] ]   [ D[2] ]
    #  [    ...   ...   ...       ] [ ...  ] = [ ...  ]
    #  [      a[n-2] b[n-1] c[n-1]] [x[n-1]]   [D[n-1]]
    #  [              a[n-1]   b[n] ] [ x[n] ]   [ D[n] ]  
    n = length(d)
    b = copy(b)
    @inbounds for i = 2:n
        wi = a[i-1] / b[i-1]
        b[i] -= wi * c[i-1]
        d[i] -= wi * d[i-1]
    end
    x[n] = d[n] / b[n]
    @inbounds for k = n-1:-1:1
        x[k] = (d[k] - c[k] * x[k+1]) / b[k]
    end
    x
end

function TDMAfast!(x::AbstractArray{T}, a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, d::AbstractArray{T}) where {T}
    #  Thomas Algorithm
    #  Vectors a,c of length(b)-1    
    #  A*x = D
    #  [b[1]   c[1]               ] [ x[1] ]   [ D[1] ]
    #  [a[1]   b[2]   c[2]        ] [ x[2] ]   [ D[2] ]
    #  [    ...   ...   ...       ] [ ...  ] = [ ...  ]
    #  [      a[n-2] b[n-1] c[n-1]] [x[n-1]]   [D[n-1]]
    #  [              a[n-1]   b[n] ] [ x[n] ]   [ D[n] ]  
    n = length(d)
    c = copy(c)
    d = copy(d)
    b0 = b[1]
    bm = b[end]
    am = a[end]
    c[1] /= b0
    d[1] /= b0
    @inbounds for i = 2:n-1
        l = 1.0 / (b[i] - (a[i-1] * c[i-1]))
        c[i] *= l
        d[i] = (d[i] - (a[i-1] * d[i-1])) * l
    end
    i = n
    l = one(T) / (bm - (am * c[i-1]))
    #c[i] *= l
    d[i] = (d[i] - (am * d[i-1])) * l

    x[end] = d[end]
    @inbounds for i = n-1:-1:1
        x[i] = d[i] - (c[i] * x[i+1])
    end
end
