export toPositiveHypersphere!, fromPositiveHypersphere!
# sum yi = radius^2
#x in R^{n-1}, y in R^{n}+ on the sphere
function toPositiveHypersphere!(y::AbstractArray{TY}, radius::TR, x::AbstractArray{TX}) where {TY,TX,TR}
	sinProduct = radius
	for (i, xi) = enumerate(x) 
		y[i] = sinProduct * cos(xi)
		sinProduct *= sin(xi)
    end
	y[length(x)+1] = sinProduct
	sumY2 = zero(TX)
	for (i,yi)= enumerate(y) 
		y[i] = yi^2 
		sumY2 += y[i]
    end
	y[length(x)+1] += radius^2 - sumY2 #make sure it sums exactly
    return y
end

function fromPositiveHypersphere!(x::AbstractArray{TY}, radius::TR,y::AbstractArray{T}) where {TY,T,TR}
	#https://en.wikipedia.org/wiki/N-sphere
	n = length(y)
	hypothSq = y[n]
	for i = n-1:-1:1
		x[i] = π/2 - atan(sqrt(y[i]/hypothSq))
		hypothSq += y[i]
    end
    return x
end