using LinearAlgebra

export RegressionMethod, QRRegression, SVDRegression, CholeskyRegression
export RegressionFit, LinearRegressionFit, regress, predict

abstract type RegressionMethod end
abstract type RegressionFit end

struct LinearRegressionFit{T} <: RegressionFit
    beta::Vector{T}
end

@inline predict(fit::LinearRegressionFit, x::AbstractVector) = dot(fit.beta, x)

struct QRRegression <: RegressionMethod end

struct SVDRegression <: RegressionMethod
    tol::Float64
    thin::Bool
end
SVDRegression(; tol=1e-12, thin=true) = SVDRegression(tol, thin)

struct CholeskyRegression <: RegressionMethod
    lambda::Float64
    withConditioning::Bool
end
CholeskyRegression(; lambda=0.0, withConditioning=true) = CholeskyRegression(lambda, withConditioning)

function regress(::QRRegression, X::AbstractMatrix{T}, y::AbstractVector{T}) where {T}
    m, n = size(X)
    if m < n
        return LinearRegressionFit(zeros(T, n))
    end
    beta = X \ y
    return LinearRegressionFit(beta)
end

function regress(method::SVDRegression, X::AbstractMatrix{T}, y::AbstractVector{T}) where {T}
    m, n = size(X)
    if m == 0 || n == 0
        return LinearRegressionFit(zeros(T, n))
    end
    # Use thin SVD if method.thin is true (full=false in LinearAlgebra.svd)
    F = svd(X; full=!method.thin)
    S = F.S
    V = F.V
    
    # Compute pseudo-inverse of S
    S_inv = zeros(T, length(S))
    threshold = method.tol * m
    for i in 1:length(S)
        if abs(S[i]) > threshold
            S_inv[i] = one(T) / S[i]
        end
    end

    if method.thin
        # F.U is m x n
        Ut_y = F.U' * y
        temp = S_inv .* Ut_y
        beta = V * temp
    else
        # F.U is m x m, we only need the first n components
        Ut_y = (F.U[:, 1:n])' * y
        temp = S_inv .* Ut_y
        beta = V * temp
    end
    return LinearRegressionFit(beta)
end

function regress(method::CholeskyRegression, X::AbstractMatrix{T}, y::AbstractVector{T}) where {T}
    m, n = size(X)
    if m < n
        return LinearRegressionFit(zeros(T, n))
    end
    A = Matrix(X' * X)
    b = X' * y

    if method.withConditioning
        maxD = 1.0
        for i in 1:n
            d = abs(A[i, i])
            if d > maxD
                maxD = d
            end
        end
        eps_cond = eps(T) * n * maxD
        for i in 1:n
            A[i, i] += eps_cond
        end
    end

    if method.lambda > 0.0
        for i in 1:n
            A[i, i] += T(method.lambda)
        end
    end

    beta = try
        C = cholesky(Symmetric(A))
        C \ b
    catch
        A \ b
    end

    return LinearRegressionFit(beta)
end
