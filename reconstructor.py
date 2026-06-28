import pandas as pd


def build_timelines(df):
    # Group ICAO24 and sort by time.
    # Returns a dictionary: {icao24: DataFrame of sorted state vectors}
    print("\n" + "=" * 50)
    print("PHASE 2: FORENSIC RECONSTRUCTION")
    print("=" * 50)
    
    timelines = {}
    
    for icao, group in df.groupby('icao24'):
        # Sort by time
        aircraft_df = group.sort_values('time').reset_index(drop=True)
        
        # Add sequence index
        aircraft_df['msg_seq'] = range(len(aircraft_df))
        
        # Add time since last message
        aircraft_df['inter_arrival'] = aircraft_df['time'].diff().fillna(0)
        
        # Mark if this is an attack aircraft
        aircraft_df['is_attack'] = aircraft_df['is_attack'].iloc[0]
        aircraft_df['attack_type'] = aircraft_df['attack_type'].iloc[0]
        
        timelines[icao] = aircraft_df
    
    print(f"Reconstructed {len(timelines)} aircraft timelines")
    
    # Print sample
    sample_icao = list(timelines.keys())[0]
    sample = timelines[sample_icao]
    print(f"\nSample timeline for {sample_icao}:")
    print(f"  Messages: {len(sample)}")
    print(f"  Duration: {sample['time'].max() - sample['time'].min():.0f} seconds")
    print(f"  Attack flag: {sample['is_attack'].iloc[0]}")
    
    return timelines


def get_timeline_summary(timelines):
    # Generate summary statistics for all timelines
    summaries = []
    
    for icao, df in timelines.items():
        summaries.append({
            'icao24': icao,
            'num_messages': len(df),
            'duration_sec': df['time'].max() - df['time'].min(),
            'start_lat': df['lat'].iloc[0],
            'start_lon': df['lon'].iloc[0],
            'end_lat': df['lat'].iloc[-1],
            'end_lon': df['lon'].iloc[-1],
            'max_altitude': df['baroaltitude'].max(),
            'min_altitude': df['baroaltitude'].min(),
            'avg_speed': df['velocity'].mean(),
            'max_speed': df['velocity'].max(),
            'is_attack': df['is_attack'].iloc[0],
            'attack_type': df['attack_type'].iloc[0]
        })
    
    return pd.DataFrame(summaries)


if __name__ == "__main__":
    from data_manager import prepare_dataset
    
    df = prepare_dataset("data/states_2018-05-28-00.csv", save_hybrid=False)
    timelines = build_timelines(df)
    
    summary = get_timeline_summary(timelines)
    print("\nTimeline Summary:")
    print(summary.head(10))