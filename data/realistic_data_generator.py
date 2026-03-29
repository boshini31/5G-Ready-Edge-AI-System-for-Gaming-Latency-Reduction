"""
Realistic Gaming Network Dataset Generator
Generates synthetic but realistic network metrics for LSTM training.
Values mimic real multiplayer game traffic patterns.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ─── Config ───────────────────────────────────────────────────────────
NUM_SAMPLES = 10000
REGIONS = {
    "Mumbai":    {"base_latency": 55,  "jitter": 12},
    "Singapore": {"base_latency": 35,  "jitter": 8},
    "Tokyo":     {"base_latency": 28,  "jitter": 6},
    "Sydney":    {"base_latency": 75,  "jitter": 15},
    "Seoul":     {"base_latency": 25,  "jitter": 5},
    "Jakarta":   {"base_latency": 48,  "jitter": 11},
}

def generate_latency(base, jitter, n):
    """Generate realistic latency with time-of-day effect, jitter, and spikes."""
    t = np.linspace(0, 4 * np.pi, n)

    # Time-of-day effect (peak hours = higher latency)
    time_effect = 10 * np.sin(t) + 5

    # Base latency with jitter
    latency = base + time_effect + np.random.normal(0, jitter, n)

    # Random network spikes (5% chance)
    spikes = np.random.rand(n) < 0.05
    latency[spikes] += np.random.uniform(50, 150, spikes.sum())

    # Occasional congestion bursts (1% chance, sustained)
    for i in range(n):
        if np.random.rand() < 0.01:
            burst_len = np.random.randint(5, 20)
            end = min(i + burst_len, n)
            latency[i:end] += np.random.uniform(30, 80)

    return np.clip(latency, 5, 300)  # realistic bounds 5ms–300ms


def generate_bandwidth(latency, n):
    """Bandwidth inversely correlated with latency (higher latency = lower bandwidth)."""
    base_bw = 100 - (latency / 300) * 60   # 40–100 Mbps range
    noise = np.random.normal(0, 5, n)
    return np.clip(base_bw + noise, 5, 150)


def generate_packet_loss(latency, n):
    """Packet loss correlated with latency spikes."""
    loss = (latency / 300) * 5              # 0–5% range
    noise = np.random.normal(0, 0.3, n)
    loss += noise
    # Extra loss during spikes
    spike_mask = latency > 100
    loss[spike_mask] += np.random.uniform(1, 8, spike_mask.sum())
    return np.clip(loss, 0, 20)             # 0–20% range


def generate_fps(latency, n):
    """FPS inversely correlated with latency."""
    base_fps = 120 - (latency / 300) * 80  # 40–120 FPS range
    noise = np.random.normal(0, 5, n)
    return np.clip(base_fps + noise, 20, 144)


def generate_jitter(latency, n):
    """Jitter as variation in latency."""
    jitter = np.abs(np.diff(latency, prepend=latency[0]))
    noise = np.random.normal(0, 1, n)
    return np.clip(jitter + noise, 0, 50)


# ─── Generate data ────────────────────────────────────────────────────
print("🎮 Generating realistic gaming network dataset...")

all_data = []
per_region = NUM_SAMPLES // len(REGIONS)

for region, params in REGIONS.items():
    n = per_region
    latency     = generate_latency(params["base_latency"], params["jitter"], n)
    bandwidth   = generate_bandwidth(latency, n)
    packet_loss = generate_packet_loss(latency, n)
    fps         = generate_fps(latency, n)
    jitter_vals = generate_jitter(latency, n)

    df = pd.DataFrame({
        "timestamp":    pd.date_range("2024-01-01", periods=n, freq="1s"),
        "region":       region,
        "latency_ms":   np.round(latency, 2),       # TARGET ← real ms values
        "bandwidth_mbps": np.round(bandwidth, 2),
        "packet_loss_pct": np.round(packet_loss, 3),
        "fps":           np.round(fps, 1),
        "jitter_ms":    np.round(jitter_vals, 2),
        "player_count": np.random.randint(100, 5000, n),
    })
    all_data.append(df)
    print(f"  ✅ {region}: {n} samples | avg latency={latency.mean():.1f}ms | max={latency.max():.1f}ms")

final_df = pd.concat(all_data, ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# ─── Save ─────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
output_path = "data/realistic_gaming_network.csv"
final_df.to_csv(output_path, index=False)

print(f"\n✅ Dataset saved to {output_path}")
print(f"   Total rows    : {len(final_df):,}")
print(f"   Columns       : {list(final_df.columns)}")
print(f"   Latency range : {final_df['latency_ms'].min():.1f}ms – {final_df['latency_ms'].max():.1f}ms")
print(f"   Avg latency   : {final_df['latency_ms'].mean():.1f}ms")
print(f"\n📌 Column mapping for dashboard:")
print(f"   Latency (Target) → latency_ms")
print(f"   Bandwidth        → bandwidth_mbps")
print(f"   Packet Loss      → packet_loss_pct")
print(f"   Extra Feature    → jitter_ms")
print(f"\n🚀 Now upload data/realistic_gaming_network.csv to the dashboard!")