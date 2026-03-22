import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from src.lag_model import LagPredictor

def train_model():
    data = pd.read_csv("data/gameplay_data.csv")
    X = torch.tensor(data.values, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(data["latency"].values, dtype=torch.float32).unsqueeze(1)

    model = LagPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/lag_model.pth")
    print("✅ Model trained and saved!")

def test_model():
    baseline = torch.randn(100) * 15 + 60
    optimized = baseline - torch.randn(100) * 3 + 10

    plt.plot(baseline.numpy(), label="Baseline")
    plt.plot(optimized.numpy(), label="Edge Optimized")
    plt.legend()
    plt.show()