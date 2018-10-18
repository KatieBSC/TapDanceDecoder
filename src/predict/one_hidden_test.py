import torch
import evaluation
import pandas as pd

#X_test = pd.read_csv('../../../../Source/Data/X_youtube.csv')
#y_test = pd.read_csv('../../../../Source/Data/y_youtube.csv')

#data = pd.read_csv('../../../../Source/Data/youtube_data.csv')
#data = data.sample(frac=1)

X_test = pd.read_csv('../../../../Source/Data/X_test_mfccplus.csv')
y_test = pd.read_csv('../../../../Source/Data/y_test_mfccplus.csv')


X_test = X_test.iloc[:, 1:21]
y_test = y_test['Labels']

dtype = torch.float
device = torch.device('cpu')

x_test = torch.tensor(X_test.values, device=device, dtype=dtype)
y_test = torch.tensor(y_test.values, device=device, dtype=torch.long).squeeze()

# Load model
model = torch.load('../train/trained_models/one_hidden_20f_sig_adam.pt')

outputs = model(x_test)
y_pred = torch.max(outputs.data, 1)[1]

# Calculate misclassification rate
misclassifiction = 1.0 * (y_test != y_pred).sum().item() / y_pred.size()[0]
print('Misclassification Rate: ', 100 * misclassifiction, '%')

predicted = y_pred.numpy()
true = y_test.numpy()

evaluation.get_errors(true, predicted)

# Export results
#evaluation.export(predicted, 'predictions/one_hidden_test_audio_20f_sig_adam.csv')
