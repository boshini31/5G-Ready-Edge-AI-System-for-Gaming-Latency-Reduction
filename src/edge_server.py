from flask import Flask, request, jsonify
import torch
from src.lag_model import LagPredictor

app = Flask(__name__)

def run_server():
    model = LagPredictor()
    model.load_state_dict(torch.load("models/lag_model.pth"))
    model.eval()

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.json["metrics"]
        X = torch.tensor(data).float()
        
        prediction = model(X).item()
        return jsonify({"predicted_latency": prediction})

    app.run(host="0.0.0.0", port=5000)