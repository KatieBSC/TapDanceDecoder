import numpy as np
import librosa.display
import IPython
from pathlib import Path


# Feature generating functions

class Features:
    def __init__(self, samples, sample_rate, clip_length, n_mfcc, frame_length, hop_length):
        self.samples = samples
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length

    def get_features_mfcc(self):
        mfccs = np.mean(librosa.feature.mfcc(y=self.samples,
                                             sr=self.sample_rate,
                                             n_mfcc=self.n_mfcc).T, axis=0)
        return mfccs.tolist()

    def get_features_zcr(self):
        zcr = librosa.feature.zero_crossing_rate(self.samples,
                                                 frame_length=self.frame_length,
                                                 hop_length=self.hop_length)
        return zcr.T.flatten().tolist()

    def get_features_energy(self):
        energy = np.array([sum(abs(self.samples[i:i + self.frame_length] ** 2))
                           for i in range(0, len(self.samples), self.hop_length)])
        return energy.tolist()

    def get_features_rmse(self):
        rmse = librosa.feature.rmse(self.samples,
                                    frame_length=self.frame_length,
                                    hop_length=self.hop_length,
                                    center=True)
        return rmse[0].tolist()

    def get_features_bpm(self):
        onset_env = librosa.onset.onset_strength(self.samples,
                                                 sr=self.sample_rate)  # Assumes static tempo, dynamic:aggregate=None
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sample_rate)
        return tempo.tolist()

    def get_feature_array(self, feature_list):
        feature_dict = {}
        features = []
        for i in feature_list:
            if i == 'mfcc':
                feature_dict[i] = self.get_features_mfcc()
                features += (feature_dict.get('mfcc'))
            if i == 'zcr':
                feature_dict[i] = self.get_features_zcr()
                features += (feature_dict.get('zcr'))
            if i == 'energy':
                feature_dict[i] = self.get_features_energy()
                features += (feature_dict.get('energy'))
            if i == 'rmse':
                feature_dict[i] = self.get_features_rmse()
                features += (feature_dict.get('rmse'))
            if i == 'bpm':
                feature_dict[i] = self.get_features_bpm()
                features += (feature_dict.get('bpm'))
        return np.array(features)


# Feature utility functions

def get_label(path):
    path = Path(path)
    if path.parts[-3] == 'Shuffle':
        return 1
    else:
        return 0


def playback(path):
    y, sr = librosa.load(path)
    audio = IPython.display.Audio(data=y, rate=sr)
    return audio


def resize_signal(samples, sample_rate, length):
    if len(samples) < length:
        y = np.pad(samples, (0, length - len(samples)), 'constant')
    elif len(samples) > length:
        y = samples[:length]
    else:
        y = samples
    return y, sample_rate


def resample_signal(path, new_sr=22050):
    y, sr = librosa.load(path)
    new_y = librosa.resample(y, sr, new_sr)
    return new_y, new_sr


def remove_1_sec(y, sr):
    new_y = y[sr:]
    return new_y, sr
