import numpy as np
import librosa.display
from pydub import AudioSegment
from pathlib import Path


# Feature generating functions

class Features :
    def __init__(self, samples, sample_rate, clip_length, n_mfcc, frame_length, hop_length, feature_set):
        self.samples = samples
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.feature_set = feature_set


    def get_features_mfcc(self):
        mfccs = np.mean(librosa.feature.mfcc(y=self.samples,
                                             sr=self.sample_rate,
                                             n_mfcc=self.n_mfcc).T, axis=0)
        return mfccs[:,np.newaxis]


    def get_features_zcr(self):
        zcr = librosa.feature.zero_crossing_rate(self.samples,
                                                 frame_length=self.frame_length,
                                                 hop_length=self.hop_length)
        return zcr.T


    def get_features_energy(self):
        energy = np.array([sum(abs(self.samples[i:i + self.frame_length] ** 2))
                           for i in range(0, len(self.samples), self.hop_length)])
        return energy[:,np.newaxis]


    def get_features_rmse(self):
        rmse = librosa.feature.rmse(self.samples,
                                    frame_length=self.frame_length,
                                    hop_length=self.hop_length,
                                    center=True)
        return rmse[0][:,np.newaxis]


    def get_features_bpm(self):
        onset_env = librosa.onset.onset_strength(self.samples,
                                                 sr=self.sample_rate)  # Assumes static tempo, dynamic:aggregate=None
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sample_rate)
        return tempo[:,np.newaxis]

    def get_feature_array(self):
        feature_array = np.empty([1,1])
        for i in self.feature_set:
            #print(i)
            if i == 'mfcc':
                feature_array = np.concatenate((feature_array, self.get_features_mfcc()),
                                               axis=0)
            if i == 'zcr':
                feature_array = np.concatenate((feature_array, self.get_features_zcr()),
                                               axis=0)
            if i == 'energy':
                feature_array = np.concatenate((feature_array, self.get_features_energy()),
                                               axis=0)
            if i == 'rmse':
                feature_array = np.concatenate((feature_array, self.get_features_rmse()),
                                               axis=0)
            if i == 'bpm':
                feature_array = np.concatenate((feature_array, self.get_features_bpm()),
                                               axis=0)
        feature_array = feature_array.flatten()
        feature_array = np.delete(feature_array, 0)
        return feature_array


# Feature utility functions
def get_label(path):
    if path.parts[-3] == 'Shuffle':
        return 1
    else:
        return 0


def playback(path):
    song = AudioSegment.from_wav(path)
    return song


def resize_signal(path, length):
    samples, sample_rate = librosa.load(path)
    if len(samples) < length:
        y = np.pad(samples, (0, length - len(samples)), 'constant')
    elif len(samples) > length:
        y = samples[:length]
    else:
        y = samples
    return y, sample_rate
