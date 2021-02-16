using epilepsy: sample_entropy
using NPZ: npzread

all_eeg = npzread("test_data/eeg_processed.npz")
eeg = all_eeg["arr_1"]
eeg_segment = eeg[1, 2000:10000]

sine = sin.(collect(1:0.1:1000))

m = 2
r = 0.2

@test sample_entropy(ones(10), m, r) == 0.0
@test sample_entropy(sine, m, r) <= 0.5
