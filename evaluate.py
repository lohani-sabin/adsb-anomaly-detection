import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
sys.path.insert(0, current_dir)

import pandas as pd
from reconstructor import build_timelines
from feature_engine import extract_features
from evaluator import run_evaluation


def main():
    print("=" * 60)
    print("AIRCRAFT NETWORK FORENSIC TOOL - EVALUATE PHASE")
    print("=" * 60)
    
    # Check build artifacts exist
    required = ["output/scaler.pkl", "output/iso_model.pkl"]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"\nERROR: Missing build artifacts: {missing}")
        print("Run 'python build.py' first to train models.")
        sys.exit(1)
    
    # Determine dataset path
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        print(f"\nUsing custom dataset: {dataset_path}")
    else:
        dataset_path = "output/hybrid_dataset.csv"
        print(f"\nUsing default test dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"\nERROR: Dataset not found: {dataset_path}")
        sys.exit(1)
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"\tLoaded {len(df)} state vectors for {df['icao24'].nunique()} aircraft")
    
    # Build timelines
    timelines = build_timelines(df)
    
    # Extract features
    features, labels, icaos = extract_features(timelines)
    
    # Run evaluation
    run_evaluation(features, labels, icaos)


if __name__ == "__main__":
    main()