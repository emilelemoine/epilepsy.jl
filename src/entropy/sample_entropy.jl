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
function sample_entropy(x::AbstractArray, m, r)
    #=
    x -> normalized signal (1d vector)
    m -> embedding dimension (must be > length of signal)
    r -> tolerance factor
    τ -> timestep =#
    # x = x[1:τ:end]
    # TODO
    # resample with moving average (length of samples = τ)
    N = length(x)
    σ = std(x)
    tolerance = σ * r

    # Create template vectors (of length m + 1)
    matches = zeros(m + 1, N)
    for i in 1:m + 1
        matches[i, 1:(N + 1 - i)] = x[i:end]
    end

    # Calculate pairwise distances for templates of length m
    dist_m = zeros(N, N)
    dist_m = pairwise!(dist_m, Chebyshev(), matches[1:m,:], dims=2)
    # Extract upper triangle of distance matrix
    dist_m = dist_m[tril!(trues(size(dist_m)), -1)]
    # Count pairs of template that are within tolerance
    B = count(x -> x <= tolerance, dist_m)

    if B == 0
        return Inf
    end

    # Repeat for templates m+1
    dist_m1 = zeros(N, N)
    dist_m1 = pairwise!(dist_m1, Chebyshev(), matches[1:m + 1,:], dims=2)
    dist_m1 = dist_m1[tril!(trues(size(dist_m1)), -1)]
    A = count(x -> x <= tolerance, dist_m1)

    sampen = -log(A / B)

end
