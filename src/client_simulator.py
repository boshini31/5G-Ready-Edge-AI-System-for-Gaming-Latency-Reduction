import requests
import numpy as np
import threading
import time
import matplotlib.pyplot as plt

# Number of simulated players
NUM_PLAYERS = 5

# Store results for visualization
predictions = []

def simulate_player(player_id):
    """Simulate one player sending metrics with jitter and spikes."""
    # Generate base latency values
    latency = np.random.normal(50, 10, 10)

    # Inject random spikes (20% chance)
    if np.random.rand() < 0.2:
        latency += 50

    # Add jitter (small random noise)
    latency += np.random.normal(0, 5, 10)

    # FPS values (just for demo)
    fps = np.random.normal(60, 5, 10)

    # Combine into metrics (shape: [1, seq_len, features])
    metrics = np.stack([latency, fps], axis=1).reshape(1, 10, 2).tolist()

    try:
        response = requests.post("http://localhost:5000/predict", json={"metrics": metrics})
        result = response.json()
        print(f"Player {player_id} predicted latency:", result)
        predictions.append((player_id, result["predicted_latency"]))
    except Exception as e:
        print(f"Player {player_id} error:", e)

def run_simulation():
    threads = []
    for i in range(NUM_PLAYERS):
        t = threading.Thread(target=simulate_player, args=(i,))
        threads.append(t)
        t.start()
        time.sleep(1)  # stagger requests slightly

    for t in threads:
        t.join()

    # Visualization: plot predicted latencies per player
    player_ids = [p[0] for p in predictions]
    latencies = [p[1] for p in predictions]

    plt.bar(player_ids, latencies)
    plt.xlabel("Player ID")
    plt.ylabel("Predicted Latency (ms)")
    plt.title("Predicted Latency per Player")
    plt.show()

if __name__ == "__main__":
    run_simulation()