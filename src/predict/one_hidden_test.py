import torch
import numpy as np
import evaluation
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

#X_test = pd.read_csv('../../../../Source/Data/X_.csv')
#y_test = pd.read_csv('../../../../Source/Data/y_youtube.csv')

#data = pd.read_csv('../../../../Source/Data/youtube_data.csv')
#data = data.sample(frac=1)

X_test = pd.read_csv('../../../../Source/Data/X_validate_mfcc_zcr_energy_rmse_bpm.csv')
y_test = pd.read_csv('../../../../Source/Data/y_validate_mfcc_zcr_energy_rmse_bpm.csv')


X_test = X_test.iloc[:, 1:21]
#X_test = X_test.iloc[:, np.r_[1:21, 510]]
y_test = y_test['Labels']

dtype = torch.float
device = torch.device('cpu')

x_test = torch.tensor(X_test.values, device=device, dtype=dtype)
y_test = torch.tensor(y_test.values, device=device, dtype=torch.long).squeeze()

# Load model
model = torch.load('../train/trained_models/one_hidden_mfcc_128_2.pt')

#outputs = model(x_test)
#y_pred = torch.max(outputs.data, 1)[1]

#print((F.softmax(outputs, dim=1)))



N = x_test.shape[0]
D_in = x_test.shape[1]


x = x_test
y = y_test

# Hyper-parameters
learning_rate = 0.0005
batch_size = 4

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_hist = []

# Test
epochs = range(3000)
idx = 0
for t in epochs:
    for batch in range(0, int(N / batch_size)):
        # Calculate batch

        batch_x = x[batch * batch_size: (batch + 1) * batch_size, :]
        batch_y = y[batch * batch_size: (batch + 1) * batch_size]

        # Forward step
        outputs = model(batch_x)

        y_pred = torch.max(outputs.data, 1)[1]
        # Calculate errors
        loss = criterion(outputs, batch_y)

        # Backward step (gradients and weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Calculate the errors (Print errors every 50 iterations)
    if t % 50 == 0:
        loss_hist.append(loss.item())
        print(t, loss.item())



# Calculate misclassification rate
misclassifiction = 1.0 * (y_test != y_pred).sum().item() / y_pred.size()[0]
print('Misclassification Rate: ', 100 * misclassifiction, '%')

predicted = y_pred.numpy()
true = y_test.numpy()
evaluation.get_errors(true, predicted)


# Export results
#evaluation.export(loss_hist, 'predictions/test_errors_one_hidden_mfcc_128.csv')
#evaluation.export(predicted, 'predictions/one_hidden_test_mfcc_128_testset.csv')
#evaluation.export(true, 'predictions/true_one_hidden_test_mfcc_128_testset.csv')