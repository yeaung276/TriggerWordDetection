## configuration of project

# audio configuration
chunk_duration = 0.5                            # Each read length in seconds from mic.
fs = 44100                                      # sampling rate for mic
chunk_samples = int(fs * chunk_duration)        # Each read length in number of samples.

# model configuration
# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 10
feed_samples = int(fs * feed_duration)
Tx = 5511                                       # The number of time steps input to the model from the spectrogram
Ty = 1375                                       # The number of time steps output tor the detector from the model
n_freq = 101                                    # Number of frequencies input to the model at each time step of the spectrogram
tr_input_shape = (Tx, n_freq)

# spectrogram configuration
nfft = 200                                      # Length of each window segment
fs = 8000                                       # Sampling frequencies
noverlap = 120                                  # Overlap between windows

