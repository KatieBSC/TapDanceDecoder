import librosa
import features



# Parameters (set in training and validation)
clip_length = 20772
n_mfcc = 20
frame_length = 256
hop_length = 128

samples, sample_rate = librosa.load('../../../../Source/Shuffle/4/1.wav')
feature_set = ['mfcc']

new_features = features.Features(samples=samples,
                       sample_rate=sample_rate,
                       clip_length=clip_length,
                       n_mfcc=n_mfcc,
                       frame_length=frame_length,
                       hop_length=hop_length,
                       feature_set=feature_set)

print(new_features.get_feature_array().shape)

