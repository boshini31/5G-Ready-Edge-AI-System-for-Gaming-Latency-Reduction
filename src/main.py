from src.pipeline import train_model, test_model
from src.edge_server import run_server

def main():
    print("Step 1: Training model...")
    train_model()

    print("Step 2: Testing performance...")
    test_model()

    print("Step 3: Starting edge server...")
    run_server()

if __name__ == "__main__":
    main()