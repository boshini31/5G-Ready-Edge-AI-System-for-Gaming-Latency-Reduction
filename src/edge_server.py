from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.lag_model import LagPredictor

app = Flask(__name__)
CORS(app)

# ─── Global state ─────────────────────────────────────────────────────
model = None
model_input_size = 2
model_lock = threading.Lock()

uploaded_df = None

# Denormalization scale — saved during training, used during prediction
# Maps normalized 0–1 output back to real ms values
denorm_scale = {
    "min": 5.0,    # minimum latency in training data (ms)
    "max": 300.0,  # maximum latency in training data (ms)
}

train_progress = {
    "running": False,
    "epoch": 0,
    "total_epochs": 50,
    "loss": None,
    "best_loss": None,
    "done": False,
    "error": None,
    "accuracy": None,
}

# ─── Load initial model ───────────────────────────────────────────────
def load_model(input_size=2):
    global model, model_input_size
    m = LagPredictor(input_size=input_size)
    model_path = "models/lag_model.pth"
    if os.path.exists(model_path):
        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"✅ Model loaded (input_size={input_size})")
        except Exception as e:
            print(f"⚠️  Could not load weights ({e}), using fresh model")
    m.eval()
    model = m
    model_input_size = input_size

load_model(input_size=2)

# ─── /health ──────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "LagPredictor LSTM",
        "input_size": model_input_size,
        "denorm_scale": denorm_scale,
        "has_uploaded_data": uploaded_df is not None,
    })

# ─── /predict ─────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["metrics"]
        X = torch.tensor(data).float()

        # Ensure 3D: (batch, seq_len, features)
        if X.dim() == 2:
            X = X.unsqueeze(0)

        # Auto pad/trim features to match model input size
        current_size = model_input_size
        actual_size = X.shape[2]
        if actual_size < current_size:
            pad = torch.zeros(X.shape[0], X.shape[1], current_size - actual_size)
            X = torch.cat([X, pad], dim=2)
        elif actual_size > current_size:
            X = X[:, :, :current_size]

        with model_lock:
            with torch.no_grad():
                raw_prediction = model(X).item()

        # ✅ Denormalize: scale from 0–1 back to real ms range
        mn = denorm_scale["min"]
        mx = denorm_scale["max"]
        predicted_latency = mn + (raw_prediction * (mx - mn))

        # Clamp to realistic bounds
        predicted_latency = max(1.0, min(predicted_latency, 500.0))

        return jsonify({
            "predicted_latency": round(predicted_latency, 2),
            "raw": round(raw_prediction, 6),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── /upload-data ─────────────────────────────────────────────────────
@app.route("/upload-data", methods=["POST"])
def upload_data():
    global uploaded_df
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        f = request.files["file"]
        if not f.filename.endswith(".csv"):
            return jsonify({"error": "Only CSV files supported"}), 400

        df = pd.read_csv(f)
        uploaded_df = df

        cols = []
        for col in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            cols.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "min": round(float(df[col].min()), 2) if is_numeric else None,
                "max": round(float(df[col].max()), 2) if is_numeric else None,
                "mean": round(float(df[col].mean()), 2) if is_numeric else None,
            })

        return jsonify({
            "success": True,
            "rows": len(df),
            "columns": cols,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── /retrain ─────────────────────────────────────────────────────────
def run_training(target_col, feature_cols, epochs, seq_len, lr):
    global model, model_input_size, train_progress, uploaded_df, denorm_scale

    try:
        train_progress.update({
            "running": True, "epoch": 0, "loss": None,
            "best_loss": None, "done": False, "error": None,
            "total_epochs": epochs, "accuracy": None,
        })

        df = uploaded_df.copy()
        all_cols = [target_col] + feature_cols
        df = df[all_cols].dropna()

        # ✅ Save real min/max of target BEFORE normalization for denormalization
        target_min = float(df[target_col].min())
        target_max = float(df[target_col].max())
        denorm_scale["min"] = target_min
        denorm_scale["max"] = target_max
        print(f"📊 Target range saved: {target_min:.2f}ms – {target_max:.2f}ms")

        # Normalize each column to 0–1
        for col in all_cols:
            mn, mx = df[col].min(), df[col].max()
            df[col] = (df[col] - mn) / (mx - mn + 1e-8)

        values = df[all_cols].values.astype(np.float32)
        X_list, y_list = [], []
        for i in range(len(values) - seq_len):
            X_list.append(values[i:i+seq_len])
            y_list.append(values[i+seq_len][0:1])

        X_tensor = torch.tensor(np.array(X_list))
        y_tensor = torch.tensor(np.array(y_list))

        input_size = len(all_cols)
        new_model = LagPredictor(input_size=input_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)

        best_loss = float("inf")
        for epoch in range(epochs):
            new_model.train()
            optimizer.zero_grad()
            out = new_model(X_tensor)
            loss = criterion(out, y_tensor)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            if loss_val < best_loss:
                best_loss = loss_val

            train_progress["epoch"] = int(epoch + 1)
            train_progress["loss"] = round(loss_val, 6)
            train_progress["best_loss"] = round(best_loss, 6)
            time.sleep(0.05)

        # Save model
        os.makedirs("models", exist_ok=True)
        torch.save(new_model.state_dict(), "models/lag_model.pth")

        # Save denorm scale alongside model
        with open("models/denorm_scale.json", "w") as f:
            json.dump(denorm_scale, f)
        print(f"✅ Denorm scale saved: {denorm_scale}")

        with model_lock:
            new_model.eval()
            model = new_model
            model_input_size = input_size

        # R² score
        with torch.no_grad():
            preds = new_model(X_tensor).squeeze().numpy()
            actual = y_tensor.squeeze().numpy()
            ss_res = np.sum((actual - preds) ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            r2 = max(0, 1 - ss_res / (ss_tot + 1e-8))

        train_progress["accuracy"] = round(float(r2 * 100), 2)
        train_progress["done"] = True
        train_progress["running"] = False
        print(f"✅ Training complete! R²={r2:.4f} | latency range: {target_min:.1f}–{target_max:.1f}ms")

    except Exception as e:
        train_progress["error"] = str(e)
        train_progress["running"] = False
        print(f"❌ Training error: {e}")


@app.route("/retrain", methods=["POST"])
def retrain():
    global uploaded_df
    if uploaded_df is None:
        return jsonify({"error": "No data uploaded yet"}), 400
    if train_progress["running"]:
        return jsonify({"error": "Training already running"}), 400

    body = request.json or {}
    target_col   = body.get("target_col")
    feature_cols = body.get("feature_cols", [])
    epochs       = int(body.get("epochs", 50))
    seq_len      = int(body.get("seq_len", 10))
    lr           = float(body.get("lr", 0.001))

    if not target_col:
        return jsonify({"error": "target_col required"}), 400

    t = threading.Thread(
        target=run_training,
        args=(target_col, feature_cols, epochs, seq_len, lr),
        daemon=True
    )
    t.start()
    return jsonify({"status": "started"})


@app.route("/train-progress", methods=["GET"])
def train_progress_poll():
    """Safe JSON serialization of training progress."""
    try:
        safe = {
            "running":      bool(train_progress["running"]),
            "epoch":        int(train_progress["epoch"]),
            "total_epochs": int(train_progress["total_epochs"]),
            "loss":         float(train_progress["loss"]) if train_progress["loss"] is not None else None,
            "best_loss":    float(train_progress["best_loss"]) if train_progress["best_loss"] is not None else None,
            "done":         bool(train_progress["done"]),
            "error":        train_progress["error"],
            "accuracy":     float(train_progress["accuracy"]) if train_progress["accuracy"] is not None else None,
        }
        return jsonify(safe)
    except Exception as e:
        return jsonify({"error": str(e), "done": False, "running": False}), 200



def run_server():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    # Load saved denorm scale if exists
    if os.path.exists("models/denorm_scale.json"):
        with open("models/denorm_scale.json") as f:
            denorm_scale.update(json.load(f))
        print(f"📊 Loaded denorm scale: {denorm_scale['min']}ms – {denorm_scale['max']}ms")

    print("🚀 Edge AI Server v3 on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)





