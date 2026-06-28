import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_manager import prepare_dataset
from reconstructor import build_timelines
from feature_engine import extract_features
from anomaly_detector import train_models


def main():
    print("=" * 60)
    print("AIRCRAFT NETWORK FORENSIC TOOL - BUILD PHASE")
    print("=" * 60)
    
    # Step 1: Load real data, inject synthetic attacks
    df = prepare_dataset("data/states_2018-05-28-00.csv", save_hybrid=True)
    
    # Step 2: Reconstruct per-aircraft timelines
    timelines = build_timelines(df)
    
    # Step 3: Extract forensic feature vectors
    features, labels, icaos = extract_features(timelines)
    
    # Step 4: Train models and save frozen artifacts
    iso_model, lstm_model, scaler, X_test_scaled, y_test, test_idx = train_models(features, labels)
    
    print("\n" + "=" * 60)
    print("BUILD COMPLETE - All artifacts saved to output/")
    print("=" * 60)
    print("\nNext step: Run 'python evaluate.py' to test the frozen models.")


if __name__ == "__main__":
    main()
