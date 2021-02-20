"""
    sample_entropy(x, m, r, τ)

Compute the Sample Entropy[^1] of `x`.

Compute Sample Entropy with template size `m` and tolerance factor `r`.
If timestep `τ` is provided, downsample signal `x` with timestep `τ`.

# Examples
```julia
julia> sample_entropy([1, 2, 3, 1, 2, 3], m=2, r=0.2, τ=1)
0.6931471805599453
```
"""
function sample_entropy(x::AbstractArray, m=2, r=0.2, τ=1)
    #=
    x -> normalized signal (1d vector)
    m -> embedding dimension (must be > length of signal)
    r -> tolerance factor
    τ -> delay =#
    # TODO For multiscale entropy: wavelet to isolate bands
    N = length(x)
    σ = std(x)
    tolerance = σ * r

    # Create template vectors (of length m + 1)
    matches = zeros(m + 1, N)
    for i in 1:m
        matches[i, 1:(N + 1 - i)] = x[i:end]
    end

    matches[m + 1, 1:(N + 1 - m - τ)] = x[m + τ:end]
    matches = matches[:, 1:N + 1 - m - τ]

    # Calculate pairwise distances for templates of length m
    dist_m = zeros(N - m, N - m)
    pairwise!(dist_m, Chebyshev(), matches[1:m,:], dims=2)
    # Extract upper triangle of distance matrix
    dist_m = dist_m[tril!(trues(size(dist_m)), -1)]
    # Count pairs of template that are within tolerance
    B = count(x -> x <= tolerance, dist_m)

    if B == 0
        return Inf
    end

    # Repeat for templates m+1
    dist_m1 = zeros(N - m, N - m)
    pairwise!(dist_m1, Chebyshev(), matches[1:m + 1,:], dims=2)
    dist_m1 = dist_m1[tril!(trues(size(dist_m1)), -1)]
    A = count(x -> x <= tolerance, dist_m1)

    -log(A / B)
    # Return object with B, A and CI around sampen

end
