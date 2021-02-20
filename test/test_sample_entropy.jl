using epilepsy: sample_entropy
using NPZ: npzread

eeg = npzread("test_data/eeg_processed.npy")
eeg_segment = eeg[1, 1:10000]

sine = sin.(collect(1:0.1:1000))

m = 3
r = 0.2

@test sample_entropy(ones(10), m, r) == 0.0
@test sample_entropy(sine, m, r) <= 0.5



for i in 0:10
    for length in [200, 2000, 20000]
        x = eeg[1, 1:length]
        println(sample_entropy(x, i, r))
        plot!(x)
    end
end
