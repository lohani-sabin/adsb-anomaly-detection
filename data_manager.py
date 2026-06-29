import pandas as pd
import numpy as np
import random
import os


def load_real_data(filepath):
    # Load real ADS-B state vector data from CSV
    df = pd.read_csv(filepath)

    # Ensure required columns exist
    required = ['time', 'icao24', 'lat', 'lon', 'baroaltitude', 'velocity', 'heading', 'vertrate']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Filter airborne aircraft only
    if 'onground' in df.columns:
        df = df[df['onground'] == False].copy()

    # Drop rows with missing required fields
    df = df.dropna(subset=['lat', 'lon', 'baroaltitude', 'velocity', 'heading'])

    # Sort by aircraft and time
    df = df.sort_values(['icao24', 'time']).reset_index(drop=True)
    print(f"Loaded {len(df)} state vectors for {df['icao24'].nunique()} unique aircraft")
    return df


def _add_rf_realism(ghost_df, receiver_lat=51.47, receiver_lon=-0.46):
    # Add realistic signal strength variation based on distance from receiver.
    # Simulate receiver at London Heathrow
    R = 6371000  # Earth radius in meters
    
    lat_rad = np.radians(ghost_df['lat'])
    lon_rad = np.radians(ghost_df['lon'])
    recv_lat_rad = np.radians(receiver_lat)
    recv_lon_rad = np.radians(receiver_lon)
    
    dlat = lat_rad - recv_lat_rad
    dlon = lon_rad - recv_lon_rad
    
    # Haversine formula to calculate distance between aircraft and receiver
    a = np.sin(dlat/2)**2 + np.cos(recv_lat_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
    distance = 2 * R * np.arcsin(np.sqrt(a))
    
    # Signal strength stronger when closer, with noise
    base_rssi = -30 - (distance / 10000)  # dBm, rough approximation
    noise = np.random.normal(0, 2, len(ghost_df))
    ghost_df['rssi'] = base_rssi + noise
    
    return ghost_df


def inject_ghost_aircraft(df, num_ghosts=5):
    # Inject ghost aircraft by cloning real flight paths and changing ICAO addresses.
    # This replicates documented ghost aircraft attacks (DEFCON style).
    ghosts = []
    real_icaos = df['icao24'].unique()
    
    for i in range(num_ghosts):
        # Clone a real aircraft's entire trajectory
        template_icao = random.choice(real_icaos)
        template = df[df['icao24'] == template_icao].copy()
        
        if len(template) < 10:
            continue
        
        ghost = template.copy()
        ghost['icao24'] = f'FAKE{i+1:03d}'
        ghost['is_attack'] = 1
        ghost['attack_type'] = 'ghost_aircraft'
        
        # Add subtle RF variation (realistic signal behavior)
        ghost = _add_rf_realism(ghost)
        
        # Slight timestamp jitter (not perfect intervals)
        jitter = np.random.normal(0, 0.15, len(ghost))
        ghost['time'] = ghost['time'] + jitter
        
        ghosts.append(ghost)
    
    if ghosts:
        ghost_df = pd.concat(ghosts, ignore_index=True)
        print(f"Injected {len(ghost_df)} ghost aircraft state vectors ({num_ghosts} aircraft)")
        return pd.concat([df, ghost_df], ignore_index=True)
    
    return df


def inject_trajectory_spoof(df, num_targets=3):
    # Inject trajectory spoofing by gradually drifting real aircraft positions.
    # Replicates UT Austin GPS spoofing demonstration style attacks.
    real_icaos = df[df['is_attack'] == 0]['icao24'].unique()
    targets = random.sample(list(real_icaos), min(num_targets, len(real_icaos)))
    
    spoofed = []
    
    for target_icao in targets:
        target = df[df['icao24'] == target_icao].copy()
        
        if len(target) < 20:
            continue
        
        # Gradual drift
        drift_factor = np.linspace(0, 0.003, len(target))  # 0 to 0.003 degrees
        
        # Add slight drift to lat/lon
        target['lat'] = target['lat'] + drift_factor * np.random.choice([-1, 1])
        target['lon'] = target['lon'] + drift_factor * np.random.choice([-1, 1])
        
        # Keep velocity/heading consistent (the attacker tries to hide)
        target['is_attack'] = 1
        target['attack_type'] = 'trajectory_spoof'
        
        spoofed.append(target)
    
    if spoofed:
        spoof_df = pd.concat(spoofed, ignore_index=True)
        print(f"Injected {len(spoof_df)} trajectory spoof state vectors ({num_targets} aircraft)")
        return pd.concat([df, spoof_df], ignore_index=True)
    
    return df


def inject_message_deletion(df, num_targets=2, deletion_rate=0.4):
    # Inject Targeted Message Deletion (DoS Simulation).
    # Randomly removes messages from real aircraft to simulate jamming/DoS.
    real_icaos = df[df['is_attack'] == 0]['icao24'].unique()
    targets = random.sample(list(real_icaos), min(num_targets, len(real_icaos)))
    
    deleted = []
    
    for target_icao in targets:
        target = df[df['icao24'] == target_icao].copy()
        
        if len(target) < 20:
            continue
        
        # Randomly delete a percentage of messages
        keep_mask = np.random.random(len(target)) > deletion_rate
        keep_mask[0] = True   # keep first message
        keep_mask[-1] = True  # keep last message
        
        deleted_df = target[keep_mask].copy()
        deleted_df['is_attack'] = 1
        deleted_df['attack_type'] = 'message_deletion'
        
        deleted.append(deleted_df)
    
    if deleted:
        deleted_df = pd.concat(deleted, ignore_index=True)
        print(f"Injected {len(deleted_df)} message-deletion state vectors ({num_targets} aircraft, {deletion_rate*100:.0f}% deleted)")
        return pd.concat([df, deleted_df], ignore_index=True)
    
    return df


def prepare_dataset(filepath, save_hybrid=True):
    # Main function: load real data, inject attacks, save hybrid dataset
    print("=" * 50)
    print("PHASE 1: DATA GENERATION")
    print("=" * 50)
    
    # Load real baseline
    df = load_real_data(filepath)
    df['is_attack'] = 0
    df['attack_type'] = 'none'
    
    # Inject attacks
    df = inject_ghost_aircraft(df, num_ghosts=50)
    df = inject_trajectory_spoof(df, num_targets=30)
    df = inject_message_deletion(df, num_targets=20, deletion_rate=0.4)
    
    # Sort again after injection
    df = df.sort_values(['icao24', 'time']).reset_index(drop=True)
    
    print(f"\nFinal hybrid dataset: {len(df)} state vectors")
    print(f"  Real: {len(df[df['is_attack'] == 0])}")
    print(f"  Attacks: {len(df[df['is_attack'] == 1])}")
    print(f"  Unique aircraft: {df['icao24'].nunique()}")
    
    if save_hybrid:
        os.makedirs("output", exist_ok=True)
        df.to_csv("output/hybrid_dataset.csv", index=False)
        print("\nSaved hybrid dataset to output/hybrid_dataset.csv")
    
    return df


if __name__ == "__main__":
    prepare_dataset("data/states_2018-05-28-00.csv", save_hybrid=True)