import features
import torch
import warnings
import time

# Ignore scipy fftpack Future Warning raised by librosa
warnings.simplefilter(action='ignore', category=FutureWarning)

# Start the clock
start = time.time()

# Get test data
path = '../../../../Source/Shuffle/3/1.wav'

# Listen to test data
features.playback(path)

# Parameters (set in training and validation)
clip_length = 20772
n_mfcc = 20
frame_length = 256
hop_length = 128

# Reshape test data
samples, sample_rate = features.resize_signal(path=path, length=clip_length)

# Select features
feature_list = ['mfcc']

# Get feature input data
test_features = features.Features(samples=samples,
                                  sample_rate=sample_rate,
                                  clip_length=clip_length,
                                  n_mfcc=n_mfcc,
                                  frame_length=frame_length,
                                  hop_length=hop_length)

inputs = (test_features.get_feature_array(feature_list=feature_list))


# Select Model
model = '..train/trained_models/one_hidden_mfcc_128.pt'


# Load and Predict
dtype = torch.float
device = torch.device('cpu')

inputs = torch.tensor(inputs, device=device, dtype=dtype)

# Load model
model = torch.load(model)

outputs = model(inputs)

y_pred = (torch.argmax(outputs.data).numpy())

true = features.get_label(path)

print("What's on tap?")
print()
if y_pred == 1:
    print('Predicted: Shuffle')
elif y_pred == 0:
    print('Predicted: Ball change')
print()
if true == 1:
    print('It was a Shuffle.')
elif true == 0:
    print('It was a Ball change.')

# End the clock and print time
end = time.time()
print()
print(f'Time elapsed: {end - start}s.')
