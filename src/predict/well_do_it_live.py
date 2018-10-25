import torch
import pandas as pd


input = pd.read_csv('../../../../Source/Data/X_validate_mfcc_zcr_energy_rmse_bpm.csv')
target = pd.read_csv('../../../../Source/Data/y_validate_mfcc_zcr_energy_rmse_bpm.csv')


input = input.iloc[20, 1:]
true = target.iloc[20,0]

dtype = torch.float
device = torch.device('cpu')
x_test = torch.tensor(input, device=device, dtype=dtype)

# Load model
model = torch.load('../train/trained_models/one_hidden_mfcc_zcr_energy_rmse_bpm.pt')

outputs = model(x_test)

y_pred = (torch.argmax(outputs.data).numpy())

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
