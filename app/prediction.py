import features
import torch
import warnings
import time
import librosa

# Ignore scipy fftpack Future Warning raised by librosa
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_prediction(path):

    # Start the clock
    start = time.time()

    # Get test data
    # path = '../../../../Source/Shuffle/4/2.wav'
    # path = '../../../../Source/Clean_train_clips/Test_pad/Ball_change/5/5.wav'

    # Parameters (set in training and validation)
    clip_length = 20772
    n_mfcc = 20
    frame_length = 256
    hop_length = 128
    #top_db = 10

    # Reshape test data
    samples, sample_rate = features.resample_signal(path=path)
    samples, sample_rate = features.adjust_to_peak(y=samples, sr=sample_rate, height=0.25,
                                          start_pad=1600, clip_length=clip_length)
    #samples = librosa.effects.trim(samples, top_db=top_db)[0]
    #samples, sample_rate = features.resize_signal(samples, sample_rate, length=clip_length)

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
    model = 'app/models/one_hidden_mfcc_128.pt'

    # Load and Predict
    dtype = torch.float
    device = torch.device('cpu')

    inputs = torch.tensor(inputs, device=device, dtype=dtype)

    # Load model
    model = torch.load(model)

    outputs = model(inputs)

    y_pred = (torch.argmax(outputs.data).numpy())

    return_value = 'unknown'
    if y_pred == 1:
        return_value = 'Shuffle'
        print('Predicted: Shuffle')
    elif y_pred == 0:
        return_value = 'Ball change'
        print('Predicted: Ball change')

    # End the clock and print time
    end = time.time()
    total_time = round(end-start, 2)
    print()
    string = 'Prediction time: ' + str(total_time) + ' sec'
    return return_value, string
