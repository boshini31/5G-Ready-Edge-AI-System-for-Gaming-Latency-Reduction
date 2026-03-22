import requests
import numpy as np

metrics = np.random.normal(50, 10, (1, 10, 2)).tolist()
response = requests.post("http://localhost:5000/predict", json={"metrics": metrics})
print("Predicted latency:", response.json())