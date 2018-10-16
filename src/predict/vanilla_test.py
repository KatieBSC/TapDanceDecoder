import torch
import evaluation
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, r2_score

X_test = pd.read_csv('../../../../Source/Data/X_test_audio.csv')
y_test = pd.read_csv('../../../../Source/Data/y_test_audio.csv')

X_test = X_test.iloc[:,1:]
y_test = y_test['Labels']

dtype = torch.float
device = torch.device('cpu')

x_test = torch.tensor(X_test.values, device=device, dtype=dtype)
y_test = torch.tensor(y_test.values, device=device, dtype=torch.long).squeeze()

# Load model
model = torch.load('../train/trained_models/vanilla_mfccplus_short.pt')

outputs = model(x_test)
y_pred = torch.max(outputs.data, 1)[1]

# Calculate misclassification rate
misclassifiction = 1.0 * (y_test != y_pred).sum().item() / y_pred.size()[0]
print('Misclassification Rate: ', 100 * misclassifiction, '%')

predicted = y_pred.numpy()
true = y_test.numpy()

print('Accuracy score: ', accuracy_score(true, predicted))
print('Log loss: ', log_loss(true, predicted))
print('R2: ', r2_score(true, predicted))
print('Calibration: ', predicted.mean() / true.mean())

# Export results
#evaluation.export(predictions, 'predictions/vanilla_test_mfccplus_short.csv')
