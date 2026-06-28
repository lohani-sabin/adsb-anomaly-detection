import pandas as pd
import numpy as np


def extract_features(timelines):
    # Extract forensic feature vectors from reconstructed timelines.
    # Returns: features DataFrame, labels Series, icao list
    print("\n" + "=" * 50)
    print("PHASE 3: FEATURE ENGINEERING")
    print("=" * 50)
    
    feature_rows = []
    labels = []
    icaos = []
    
    for icao, df in timelines.items():
        if len(df) < 5:  # Need minimum messages for meaningful features
            continue
        
        features = {}
        
        # === 1. INTER-ARRIVAL TIMING DELTA ===
        # How consistent are message intervals?
        inter_arrivals = df['inter_arrival'].dropna()
        features['inter_arrival_mean'] = inter_arrivals.mean()
        features['inter_arrival_std'] = inter_arrivals.std() if len(inter_arrivals) > 1 else 0
        features['inter_arrival_max'] = inter_arrivals.max()
        
        # === 2. BAROMETRIC ALTITUDE RATE OF CHANGE ===
        # How smooth is altitude change?
        alt_changes = df['baroaltitude'].diff().dropna()
        time_deltas = df['time'].diff().dropna()
        altitude_rates = alt_changes / time_deltas
        
        features['alt_rate_mean'] = altitude_rates.mean()
        features['alt_rate_std'] = altitude_rates.std() if len(altitude_rates) > 1 else 0
        features['alt_rate_max'] = altitude_rates.abs().max()
        
        # === 3. SIGNAL STRENGTH CONSISTENCY (RSSI) ===
        # If RSSI exists (from our synthetic injection), measure consistency
        if 'rssi' in df.columns:
            features['rssi_mean'] = df['rssi'].mean()
            features['rssi_std'] = df['rssi'].std() if len(df) > 1 else 0
            features['rssi_range'] = df['rssi'].max() - df['rssi'].min()
        else:
            # Proxy: use lastcontact gap as signal quality indicator
            if 'lastcontact' in df.columns:
                contact_gaps = df['time'] - df['lastcontact']
                features['rssi_mean'] = -contact_gaps.mean()  # Negative = weaker signal
                features['rssi_std'] = contact_gaps.std() if len(df) > 1 else 0
                features['rssi_range'] = contact_gaps.max() - contact_gaps.min()
            else:
                features['rssi_mean'] = 0
                features['rssi_std'] = 0
                features['rssi_range'] = 0
        
        # === 4. GEOGRAPHIC DISPLACEMENT ERROR (Dead Reckoning) ===
        # Predict next position from current velocity/heading, compare to actual
        displacement_errors = []
        
        for i in range(len(df) - 1):
            curr = df.iloc[i]
            next_row = df.iloc[i + 1]
            dt = next_row['time'] - curr['time']
            
            if dt <= 0:
                continue
            
            # Dead reckoning: predict next position
            # distance = velocity * time
            distance_m = curr['velocity'] * dt
            
            # Convert to lat/lon displacement (approximate)
            heading_rad = np.radians(curr['heading'])
            dlat = (distance_m * np.cos(heading_rad)) / 111320  # meters to degrees
            dlon = (distance_m * np.sin(heading_rad)) / (111320 * np.cos(np.radians(curr['lat'])))
            
            pred_lat = curr['lat'] + dlat
            pred_lon = curr['lon'] + dlon
            
            # Error: difference between predicted and actual
            lat_error = abs(pred_lat - next_row['lat'])
            lon_error = abs(pred_lon - next_row['lon'])
            
            displacement_errors.append(np.sqrt(lat_error**2 + lon_error**2))
        
        if displacement_errors:
            features['drift_mean'] = np.mean(displacement_errors)
            features['drift_std'] = np.std(displacement_errors) if len(displacement_errors) > 1 else 0
            features['drift_max'] = max(displacement_errors)
        else:
            features['drift_mean'] = 0
            features['drift_std'] = 0
            features['drift_max'] = 0
        
        # === 5. VELOCITY CONSISTENCY ===
        features['velocity_mean'] = df['velocity'].mean()
        features['velocity_std'] = df['velocity'].std() if len(df) > 1 else 0
        features['velocity_change_max'] = df['velocity'].diff().abs().max()
        
        # === 6. HEADING STABILITY ===
        headings = df['heading'].values
        heading_diffs = []
        for i in range(1, len(headings)):
            diff = abs(headings[i] - headings[i-1])
            diff = min(diff, 360 - diff)  # Handle 0/360 wrap
            heading_diffs.append(diff)
        
        features['heading_change_mean'] = np.mean(heading_diffs) if heading_diffs else 0
        features['heading_change_max'] = max(heading_diffs) if heading_diffs else 0
        
        # === 7. ALTITUDE RANGE ===
        features['altitude_range'] = df['baroaltitude'].max() - df['baroaltitude'].min()
        features['avg_altitude'] = df['baroaltitude'].mean()
        
        # === 8. MESSAGE COUNT ===
        features['message_count'] = len(df)
        features['duration_sec'] = df['time'].max() - df['time'].min()
        
        # Store
        feature_rows.append(features)
        labels.append(df['is_attack'].iloc[0])
        icaos.append(icao)
    
    features_df = pd.DataFrame(feature_rows)
    labels_series = pd.Series(labels, name='is_attack')
    icaos_series = pd.Series(icaos, name='icao24')
    
    # Handle NaN/Inf
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"Extracted {len(features_df)} feature vectors with {len(features_df.columns)} features")
    print(f"\tNormal aircraft: {(labels_series == 0).sum()}")
    print(f"\tAttack aircraft: {(labels_series == 1).sum()}")
    
    return features_df, labels_series, icaos_series


if __name__ == "__main__":
    from data_manager import prepare_dataset
    from reconstructor import build_timelines
    
    df = prepare_dataset("data/states_2018-05-28-00.csv", save_hybrid=False)
    timelines = build_timelines(df)
    features, labels, icaos = extract_features(timelines)
    
    print("\nFeature columns:")
    print(list(features.columns))
    print("\nSample features (first aircraft):")
    print(features.iloc[0])
