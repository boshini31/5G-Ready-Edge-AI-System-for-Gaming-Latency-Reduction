import numpy as np
import pandas as pd

np.random.seed(42)
latency = np.random.normal(50, 10, 1000)
fps = np.random.normal(60, 5, 1000)

data = pd.DataFrame({"latency": latency, "fps": fps})
data.to_csv("data/gameplay_data.csv", index=False)

print("✅ Gameplay data generated at data/gameplay_data.csv")