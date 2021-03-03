using epilepsy: sample_entropy, sampen_along_axis
using NPZ: npzread

eeg = npzread("test_data/eeg_processed.npy")
eeg_segment = eeg[1, 1:10000]

epoched_eeg = [[1, 2, 3, 4, 5, 4], [1, 1, 1, 1, 1, 1], [1, 2, 1, 2, 1, 2]]
sine = sin.(collect(1:0.1:1000))

m = 2
r = 0.2
δ = 1
τ = 3


n_seg = size(x, 1)
sampen = Array{Float64,2}(undef, n_seg, 3)

for i in eachindex(x)
    sampen[i, :] = sample_entropy(x)
end

for i in eachindex(epoched_eeg)
    x = epoched_eeg[i]
end


@test sample_entropy(ones(10), m, r) == 0.0
@test sample_entropy(sine, m, r) <= 0.5



for i in 0:10
    for length in [200, 2000, 20000]
        x = eeg[1, 1:length]
        println(sample_entropy(x, i, r))
        plot!(x)
    end
end
