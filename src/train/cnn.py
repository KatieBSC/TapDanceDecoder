import pandas as pd
import torch
import torch.nn as nn
import evaluation
import torch.nn.functional as F

dtype = torch.float
device = torch.device("cpu")

# Get data
X_train = pd.read_csv('../../../../Source/Data/X_train_signal.csv')
y_train = pd.read_csv('../../../../Source/Data/y_train_signal.csv')

train_data = X_train.iloc[:,1:].values
train_target = y_train['Labels']


N = train_data.shape[0]

x = torch.tensor(train_data, dtype=torch.float32)
y = torch.tensor(train_target, dtype=torch.long)


# Neural Network
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        # Input channels = 1, output channels = 6
        self.conv1 = torch.nn.Conv1d(1, 6, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # 384 input features, 32 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(6 * 20772, 32)  # Change for my inputs

        # 32 input features, 10 output features for our 10 defined classes, in my case 2 classes
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (1, 16, 16) to (6, 16, 16) NOT!
        x = F.relu(self.conv1(x))

        # Size changes from (6, 16, 16) to (6, 8, 8)
        x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (6, 8, 8) to (1, 384) NOT!
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 6 * 20772)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 384) to (1, 32) NOT!
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 32) to (1, 10)
        x = self.fc2(x)
        return (x)

model = Cnn()

learning_rate = 0.0001
batch_size = 32

# Regularisierung
weight_decay=0.001

# ADAM
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Loss
criterion = nn.CrossEntropyLoss()

loss_hist = []

# Train
epochs = range(100)
idx = 0
for t in epochs:
    for batch in range(0, int(N / batch_size)):
        # Berechne den Batch
        batch_x = x[batch * batch_size: (batch + 1) * batch_size, :]
        print(batch_size)

        batch_x = batch_x.reshape(batch_size, 1, 20772)
        print(batch_x.shape)

        batch_y = y[batch * batch_size: (batch + 1) * batch_size]
        print(batch_y.shape)


        # Berechne die Vorhersage (foward step)
        outputs = model.forward(batch_x)


        # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
        loss = criterion(outputs, batch_y)

        # Berechne die Gradienten und Aktualisiere die Gewichte (backward step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
    if t % 20 == 0:
        loss_hist.append(loss.item())
        print(t, loss.item())

# Save and export trained model and training errors
evaluation.export(loss_hist, 'train_errors/cnn_signal.csv')
torch.save(model, 'trained_models/cnn_signal.pt')