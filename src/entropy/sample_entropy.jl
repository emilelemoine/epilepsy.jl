"""
    sample_entropy(x, m, r, δ)

Compute the Sample Entropy[^1] of `x`.

Compute Sample Entropy with template size `m` and tolerance factor `r`.
If timestep `δ` is provided, downsample signal `x` with timestep `δ`.

# Examples
```julia
julia> sample_entropy([1, 2, 3, 1, 2, 3], m=2, r=0.2, δ=1)
0.6931471805599453
```
"""
function sample_entropy(x::AbstractArray, m=2, r=0.2, δ=1)
    #=
    x -> normalized signal (1d vector)
    m -> embedding dimension (must be > length of signal)
    r -> tolerance factor
    δ -> delay =#
    N = length(x)
    σ = std(x)
    tolerance = σ * r

    # Create template vectors (of length m + 1)
    matches = zeros(m + 1, N)
    for i in 1:m
        matches[i, 1:(N + 1 - i)] = x[i:end]
    end

    matches[m + 1, 1:(N + 1 - m - δ)] = x[m + δ:end]
    matches = matches[:, 1:N + 1 - m - δ]

    # Calculate pairwise distances for templates of length m
    dist_m = zeros(N - m, N - m)
    pairwise!(dist_m, Chebyshev(), matches[1:m,:], dims=2)
    # Extract upper triangle of distance matrix
    dist_m = dist_m[tril!(trues(size(dist_m)), -1)]
    # Count pairs of template that are within tolerance
    B = count(x -> x <= tolerance, dist_m)

    if B == 0
        return Inf, 0, 0
    end

    # Repeat for templates m+1
    dist_m1 = zeros(N - m, N - m)
    pairwise!(dist_m1, Chebyshev(), matches[1:m + 1,:], dims=2)
    dist_m1 = dist_m1[tril!(trues(size(dist_m1)), -1)]
    A = count(x -> x <= tolerance, dist_m1)

    -log(A / B), A, B

end

function multiscale_entropy(x::AbstractArray, τ=1, m=2, r=0.2, δ=1)

    if τ == 1
        sample_entropy(x, m, r, δ)
    end

    N = length(x)
    x = x[1:end - (N % τ)]
    N = length(x)
    # TODO ordre de dims à vérifier
    dims = (Int(N / τ), τ)
    x = reshape(x, dims)
    x_mse = mean(x, dims=1)

    sample_entropy(x_mse, m, r, δ)

end
