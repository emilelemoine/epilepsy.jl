using epilepsy: sample_entropy, multiscale_entropy
using NPZ: npzread

eeg = npzread("test_data/eeg_processed.npy")
eeg_segment = eeg[1, 1:10000]

epoched_eeg = [[1, 2, 3, 4, 5, 4], [1, 1, 1, 1, 1, 1], [1, 2, 1, 2, 1, 2]]
sine = sin.(collect(1:0.1:1000))

m = 2
r = 0.2
δ = 1
τ = 3

@test sample_entropy(ones(10), m, r) == (0.0, 28, 28)
@test sample_entropy(sine, m, r)[1] <= 0.5
@test multiscale_entropy(sine, 1, m, r) == sample_entropy(sine, m, r)
