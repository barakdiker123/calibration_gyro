
import create_data
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import automate as am
from model_arch import VectorPredictor
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
# Load your dataframe (assuming you have 'est_x', 'est_y', 'est_z' as inputs
# and 'gt_x', 'gt_y', 'gt_z' as targets)
# df = pd.read_csv("your_data.csv")
# sensor_arr = [None for i in range(4)]
# sensor1, sensor2, sensor3, sensor4, train_indexes = am.singular_extract(
#    "Z_up_1302_1601.csv", [0, 0, 0])
# Extract inputs (estimated vectors) and targets (ground truth vectors)

# X = sensor1[['gyro_x', 'gyro_y', 'gyro_z']].values
# y = sensor1[['gt_x', 'gt_y', 'gt_z']].values
# y = pd.DataFrame([[0, 0, 60] for i in range(train_indexes)],
#                 columns=['gt_x', 'gt_y', 'gt_z']).values

sensor1_data, sensor2_data, sensor3_data, sensor4_data, gt = create_data.create_data()
X = sensor4_data[['gyro_x', 'gyro_y', 'gyro_z']].values
# y = sensor1[['gt_x', 'gt_y', 'gt_z']].values
y = gt.values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create PyTorch datasets and loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# Define a simple MLP model


# class VectorPredictor(nn.Module):
#    def __init__(self):
#        super(VectorPredictor, self).__init__()
#        self.model = nn.Sequential(
#            nn.Linear(3, 64),  # Input is 3D vector, output 64 features
#            nn.ReLU(),
#            nn.Linear(64, 3)  # Output is 3D vector
#        )
#        self.model1 = nn.Sequential(
#            nn.Linear(3, 3),  # Input is 3D vector, output 64 features
#            # nn.ReLU(),
#            # nn.Linear(64, 3)  # Output is 3D vector
#        )
#
#    def forward(self, x):
#        return self.model1(x)


# Initialize model, loss function, and optimizer
# model = VectorPredictor()
model = VectorPredictor().to(device)  # Move model to MPS
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 25
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 3 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss /
              len(train_loader):.6f}")

# Evaluate on test set
model.eval()
test_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_loader):.6f}")

torch.save(model.state_dict(), "model_sensor4.pth")
print("Saved PyTorch Model State to model.pth")


# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("model.pth", weights_only=True))
