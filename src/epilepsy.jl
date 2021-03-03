module epilepsy

using Statistics
using LinearAlgebra
using Distances

export sample_entropy, multiscale_entropy, sampen_along_axis

# Write your package code here.
include("entropy/sample_entropy.jl")

end
