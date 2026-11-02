import csv
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------
# 1. Load Behavior Cloning Dataset
# --------------------------------------------------

dataset_file = "data/processed/behavior_cloning_dataset.csv"

inputs = []
targets = []

with open(dataset_file, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        inputs.append([
            float(row["human_x"]),
            float(row["human_y"]),
            float(row["human_z"])
        ])

        targets.append([
            float(row["robot_x"]),
            float(row["robot_y"]),
            float(row["robot_z"])
        ])


# Convert data to PyTorch tensors
X = torch.tensor(inputs, dtype=torch.float32)
Y = torch.tensor(targets, dtype=torch.float32)

print("Dataset loaded!")
print("Input shape:", X.shape)
print("Target shape:", Y.shape)


# --------------------------------------------------
# 2. Define Behavior Cloning Neural Network
# --------------------------------------------------

class BehaviorCloningModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)


model = BehaviorCloningModel()

print("\nBehavior Cloning model created!")


# --------------------------------------------------
# 3. Loss Function and Optimizer
# --------------------------------------------------

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# 4. Train the Model
# --------------------------------------------------

epochs = 1000

print("\nTraining started...")

for epoch in range(epochs):

    # Forward pass
    predictions = model(X)

    # Calculate error
    loss = criterion(predictions, Y)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update model weights
    optimizer.step()

    # Display progress
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.8f}"
        )


# --------------------------------------------------
# 5. Save Trained Model
# --------------------------------------------------

model_path = "models/behavior_cloning_model.pth"

torch.save(model.state_dict(), model_path)

print("\nTraining completed!")
print("Model saved to:", model_path)
