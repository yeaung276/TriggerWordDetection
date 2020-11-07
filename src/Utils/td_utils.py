### This file is from the coursera course-Sequence Models(Trigger Word Detection System)
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    if type(wav_file) is str:
        rate, data = get_wav_info(wav_file)
    else:
        data = wav_file
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    # the spectrogram outputs (freqs, Tx)
    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio(path):
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir(path + "/positives"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav(path + "/positives/"+filename)
            activates.append(activate)
    for filename in os.listdir(path + "/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(path + "/backgrounds/"+filename)
            backgrounds.append(background)
    for filename in os.listdir(path + "/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav(path + "/negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass