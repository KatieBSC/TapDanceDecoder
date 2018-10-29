import librosa
import numpy as np


def add_noise(file_path):
    data = librosa.core.load(file_path)[0]
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise


def shift(file_path):
    data = librosa.core.load(file_path)[0]
    return np.roll(data, 300)


def stretch(file_path):
    data = librosa.core.load(file_path)[0]
    rate = 1.5
    data = librosa.effects.time_stretch(data, rate)
    return data
