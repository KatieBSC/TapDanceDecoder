import pandas as pd
import torch
import torch.nn as nn
import evaluation

dtype = torch.float
device = torch.device("cpu")


# Get data
X_train = pd.read_csv('../../../../Source/Data/X_train_audio_augmented_mfcc.csv')
y_train = pd.read_csv('../../../../Source/Data/y_train_audio_augmented_mfcc.csv')

inputs = X_train.iloc[:,1:].values
targets = y_train['Labels'].values

N = inputs.shape[0]
D_in = inputs.shape[1]
D_out = targets.max() + 1

x = torch.tensor(inputs, device=device, dtype=dtype)
y = torch.tensor(targets, device=device, dtype=torch.long).squeeze()

# Hyper-parameters
learning_rate = 0.005
batch_size = 64

# Neural Network
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, D_out)
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_hist = []

# Train
epochs = range(30000)
idx = 0
for t in epochs:
    for batch in range(0, int(N / batch_size)):
        # Calculate batch

        batch_x = x[batch * batch_size: (batch + 1) * batch_size, :]
        batch_y = y[batch * batch_size: (batch + 1) * batch_size]

        # Forward step
        outputs = model(batch_x)

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

# Save and export trained model and training errors
evaluation.export(loss_hist, 'train_errors/vanilla_mfcc.csv')
torch.save(model, 'trained_models/vanilla_mfcc.pt')
