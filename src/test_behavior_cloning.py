import torch
import torch.nn as nn


# ============================================================
# Behavior Cloning Model
# Must exactly match the training architecture
# ============================================================

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


# ============================================================
# Load trained model
# ============================================================

model_path = "models/behavior_cloning_model.pth"

model = BehaviorCloningModel()

model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu"
    )
)

model.eval()

print("Behavior Cloning model loaded successfully!")


# ============================================================
# Test input
# ============================================================

test_input = torch.tensor(
    [[0.0, 0.0, 0.0]],
    dtype=torch.float32
)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    prediction = model(test_input)


# ============================================================
# Display result
# ============================================================

print("\nTest input:")
print("Human XYZ:", test_input.numpy()[0])

print("\nPredicted robot position:")
print("Robot XYZ:", prediction.numpy()[0])
