import numpy as np
import librosa.display


# Feature parameters
n_mfcc = 20
n_fft = 2048  # default is 2048
frame_length = 250
hop_length = 125

# Feature generating functions
def get_features_bpm(path):
    samples, sample_rate = librosa.load(path)
    onset_env = librosa.onset.onset_strength(samples, sr=sample_rate)  # Assumes static tempo,for dynamic:aggregate=None
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)
    return tempo.item()


def get_features_stft(path):
    y, sr = librosa.load(path)
    d = np.abs(librosa.stft(y))
    return d


def get_features_mfcc(path):
    samples, sample_rate = librosa.load(path)
    mfccs = np.mean(librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)
    return mfccs


def get_features_ZCR(path):
    samples, sample_rate = librosa.load(path)
    return librosa.feature.zero_crossing_rate(samples, frame_length=frame_length, hop_length=hop_length)


def get_features_energy(path):
    samples, sample_rate = librosa.load(path)
    energy = np.array([sum(abs(samples[i:i+frame_length]**2)) for i in range(0, len(samples), hop_length)])
    return energy


def get_features_rmse(path):
    samples, sample_rate = librosa.load(path)
    rmse = librosa.feature.rmse(samples, frame_length=frame_length, hop_length=hop_length, center=True)
    return rmse[0]


# Feature utility functions
def build_list(step, folder, length):
    i = 1
    step_list = []
    while i <= length:
        name = step + "/" + str(folder) + "/" + str(i) + ".wav"
        step_list.append(name)
        i += 1
    return step_list


def get_label(path):
    if path.parts[-3] == 'Shuffle':
        return 1
    else:
        return 0
