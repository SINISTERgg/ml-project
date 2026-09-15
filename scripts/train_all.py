import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import main as train


if __name__ == "__main__":
    train()
    print("\nTraining + explainability complete.")
    print("Start the API with:  python -m uvicorn app.main:app --reload")