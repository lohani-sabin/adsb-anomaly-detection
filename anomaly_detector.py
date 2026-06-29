import numpy as np
np.random.seed(42)
import pandas as pd
import joblib
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    tf.random.set_seed(42)
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not installed. LSTM Autoencoder will be skipped.")


def prepare_data(features_df, labels_series, icaos_series=None, test_size=0.2, random_state=42):
    # Split data into train and test sets by aircraft (ICAO24 index), ensuring no data leakage.
    # Stratifies on attack_type so every attack variant appears in both train and test.
    print("\n" + "=" * 50)
    print("PHASE 4: MODEL TRAINING")
    print("=" * 50)

    indices = np.arange(len(features_df))

    # Build a per-aircraft attack_type series for stratification.
    # Requires the hybrid_dataset.csv written by prepare_dataset().
    attack_type_strat = labels_series.copy()  # fallback: binary label
    if icaos_series is not None and os.path.exists("output/hybrid_dataset.csv"):
        hybrid = pd.read_csv("output/hybrid_dataset.csv", usecols=['icao24', 'attack_type'])
        attack_map = hybrid.groupby('icao24')['attack_type'].first()
        attack_type_strat = icaos_series.map(attack_map).fillna('none')

    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=attack_type_strat
    )

    X_train = features_df.iloc[train_idx]
    X_test  = features_df.iloc[test_idx]
    y_train = labels_series.iloc[train_idx]
    y_test  = labels_series.iloc[test_idx]

    # For training: ONLY non-attack data (unsupervised learning on clean data)
    X_train_normal = X_train[y_train == 0]

    print(f"Training set: {len(X_train)} aircraft ({len(X_train_normal)} normal)")
    print(f"Test set: {len(X_test)} aircraft ({sum(y_test == 1)} attacks)")

    # Save test ICAOs so evaluate.py uses the same held-out set
    if icaos_series is not None:
        test_icaos = icaos_series.iloc[test_idx].values
        os.makedirs("output", exist_ok=True)
        np.save("output/test_icaos.npy", test_icaos)
        print(f"Saved {len(test_icaos)} test ICAOs to output/test_icaos.npy")

    return X_train_normal, X_test, y_train, y_test, train_idx, test_idx


def fit_scaler(X_train):
    # Fit RobustScaler on training data and save it.
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    os.makedirs("output", exist_ok=True)
    joblib.dump(scaler, "output/scaler.pkl")
    print("\nSaved scaler to output/scaler.pkl")
    
    return scaler, X_train_scaled


def train_isolation_forest(X_train_scaled, contamination=0.015, n_estimators=200, random_state=42):
    # Train Isolation Forest for instant statistical outlier detection.
    print("\nTraining Isolation Forest...")
    
    iso_model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    iso_model.fit(X_train_scaled)
    
    # Save frozen model
    joblib.dump(iso_model, "output/iso_model.pkl")
    print("Saved Isolation Forest to output/iso_model.pkl")
    
    return iso_model


def train_lstm_autoencoder(X_train_scaled, seq_length=10, epochs=50, batch_size=32):
    """
    Train LSTM Autoencoder for sequential behavioral anomaly detection.
    
    Architecture: Input -> LSTM(64) -> LSTM(32) -> Repeat -> LSTM(32) -> LSTM(64) -> Output
    Bottleneck (32 units) forces learning compressed 'normal behavior' signature.
    """
    if not TF_AVAILABLE:
        print("\nSkipping LSTM Autoencoder (TensorFlow not available)")
        return None
    
    print("\nTraining LSTM Autoencoder...")
    
    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
    
    # If we don't have enough samples for sequences, duplicate data
    if len(X_train_scaled) < seq_length * 2:
        print(f"  Expanding training data from {len(X_train_scaled)} to {seq_length * 3} samples...")
        X_train_scaled = np.tile(X_train_scaled, (seq_length * 3 // len(X_train_scaled) + 1, 1))[:seq_length * 3]
    
    X_train_seq = create_sequences(X_train_scaled, seq_length)
    
    n_features = X_train_scaled.shape[1]
    
    # Build model
    inputs = Input(shape=(seq_length, n_features))
    
    # Encoder
    encoded = LSTM(64, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)
    
    # Bottleneck
    bottleneck = RepeatVector(seq_length)(encoded)
    
    # Decoder
    decoded = LSTM(32, activation='relu', return_sequences=True)(bottleneck)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    
    # Output
    outputs = TimeDistributed(Dense(n_features))(decoded)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    
    # Train with early stopping
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train_seq, X_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save frozen model
    model.save("output/lstm_model.keras")
    print("Saved LSTM Autoencoder to output/lstm_model.keras")
    
    # Save training history for reporting
    history_df = pd.DataFrame({
        'epoch': range(1, len(history.history['loss']) + 1),
        'loss': history.history['loss'],
        'val_loss': history.history.get('val_loss', [None] * len(history.history['loss']))
    })
    history_df.to_csv("output/lstm_training_history.csv", index=False)
    
    return model


def train_models(features_df, labels_series, icaos_series=None, test_size=0.2, random_state=42):
    # Main training pipeline: scale data, train both models and save artifacts
    # Split data
    X_train_normal, X_test, y_train, y_test, train_idx, test_idx = prepare_data(
        features_df, labels_series, icaos_series, test_size, random_state
    )
    
    # Scale data (fit on normal training data only)
    scaler, X_train_scaled = fit_scaler(X_train_normal)
    X_test_scaled = scaler.transform(X_test)
    
    # Save test indices for evaluation
    np.save("output/test_indices.npy", test_idx)
    
    # Train Isolation Forest
    iso_model = train_isolation_forest(X_train_scaled)
    
    # Train LSTM Autoencoder
    lstm_model = train_lstm_autoencoder(X_train_scaled)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print("Frozen artifacts saved to output/:")
    print("\t- scaler.pkl (normalization)")
    print("\t- iso_model.pkl (Isolation Forest)")
    print("\t- lstm_model.keras (LSTM Autoencoder)")
    print("\t- test_indices.npy (test set indices)")
    
    return iso_model, lstm_model, scaler, X_test_scaled, y_test, test_idx


if __name__ == "__main__":
    from data_manager import prepare_dataset
    from reconstructor import build_timelines
    from feature_engine import extract_features
    
    df = prepare_dataset("data/states_2018-05-28-00.csv", save_hybrid=False)
    timelines = build_timelines(df)
    features, labels, icaos = extract_features(timelines)
    
    train_models(features, labels)