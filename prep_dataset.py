"""
Dataset Manifest Generation Script - STEP 1
Scans the GrowliFlowerL dataset and generates a manifest CSV mapping images 
to their Day After Planting (DAP) and 7-day average weather features.

Output: dataset_manifest.csv with columns:
[sequence_id, frame_index, elapsed_days, image_path, temperature, 
 relative_humidity, solar_radiation, precipitation]
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import re
from collections import defaultdict

# Configuration
DATASET_DIR = './dataset/'
IMAGE_BASE_DIR = os.path.join(DATASET_DIR, 'GrowliFlowerL', 'images')
COORDS_BASE_DIR = os.path.join(DATASET_DIR, 'GrowliFlowerR')
WEATHER_FILE = os.path.join(DATASET_DIR, 'growliflower_environmental_data.csv')
OUTPUT_FILE = 'dataset_manifest.csv'

# Sequence length for training (7 days of context)
SEQUENCE_LENGTH = 7


def load_weather_data(weather_file):
    """Load weather data from CSV."""
    df = pd.read_csv(weather_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


def parse_image_filename(filename):
    """Parse image filename to extract date and image ID. Format: patch_YYYY_MM_DD_XXXXX.jpg"""
    match = re.match(r'patch_(\d{4})_(\d{2})_(\d{2})_(\d+)\.jpg', filename)
    if match:
        year, month, day, img_id = match.groups()
        date_str = f'{year}-{month}-{day}'
        return date_str, img_id
    return None, None


def read_all_coords(coord_file):
    """Read all entries from a coordinate file. Format: image_name X_coord Y_coord DAP"""
    entries = []
    try:
        with open(coord_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    plant_name = parts[0]
                    try:
                        dap = int(parts[3])
                        entries.append({'plant_name': plant_name, 'dap': dap})
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading {coord_file}: {e}")
    return entries


def get_7day_avg_weather(weather_df, target_date, dap):
    """Get 7-day average weather leading up to target_date."""
    start_date = target_date - timedelta(days=SEQUENCE_LENGTH - 1)
    end_date = target_date
    
    mask = (weather_df.index >= start_date) & (weather_df.index <= end_date)
    window_data = weather_df[mask]
    
    if len(window_data) < SEQUENCE_LENGTH:
        return None
    
    return {
        'temperature': round(window_data['temperature'].mean(), 4),
        'relative_humidity': round(window_data['relative_humidity'].mean(), 4),
        'solar_radiation': round(window_data['solar_radiation'].mean(), 4),
        'precipitation': round(window_data['precipitation'].mean(), 4)
    }


def scan_images_and_build_manifest(image_dir, coords_dir, weather_df):
    """Scan all images and coordinate files to build manifest."""
    manifest_entries = []
    insufficient_weather = 0
    
    # First, collect all coordinate entries by date
    coord_entries_by_date = defaultdict(list)
    
    print("Loading coordinate files...")
    # Scan all coordinate files
    for field in ['Field1', 'Field2']:
        field_path = os.path.join(coords_dir, field)
        if not os.path.isdir(field_path):
            continue
        
        for date_folder in sorted(os.listdir(field_path)):
            date_path = os.path.join(field_path, date_folder)
            if not os.path.isdir(date_path):
                continue
            
            # Convert folder name to date string
            try:
                date_str = date_folder.replace('_', '-')
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                continue
            
            # Read all coordinate files for this date/field
            for coord_file in os.listdir(date_path):
                if 'pix_Coordinates' not in coord_file:
                    continue
                
                coord_filepath = os.path.join(date_path, coord_file)
                coord_entries = read_all_coords(coord_filepath)
                
                # Extract plot ID
                plot_match = re.search(r'Ref_Plot(\d+)', coord_file)
                plot_id = plot_match.group(1) if plot_match else 'unknown'
                sequence_id = f'{field}_Plot{plot_id}'
                
                for entry in coord_entries:
                    coord_entries_by_date[date_str].append({
                        'sequence_id': sequence_id,
                        'plant_name': entry['plant_name'],
                        'dap': entry['dap'],
                        'field': field
                    })
    
    print(f"  Loaded {sum(len(v) for v in coord_entries_by_date.values())} coordinate entries")
    
    # Group images by plant/sequence
    sequences = defaultdict(list)
    
    print("Scanning images...")
    
    # Walk through all image directories (Train, Val, Test)
    for split in ['Train', 'Val', 'Test']:
        split_dir = os.path.join(image_dir, split)
        if not os.path.isdir(split_dir):
            continue
        
        image_count = len([f for f in os.listdir(split_dir) if f.endswith('.jpg')])
        print(f"  Processing {split} set ({image_count} images)...")
        
        for filename in sorted(os.listdir(split_dir)):
            if not filename.endswith('.jpg'):
                continue
            
            # Parse image filename
            date_str, image_id = parse_image_filename(filename)
            if date_str is None:
                continue
            
            # Get all coordinate entries for this date
            entries_for_date = coord_entries_by_date.get(date_str, [])
            if not entries_for_date:
                continue
            
            # Convert date string to datetime
            try:
                target_date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                continue
            
            # For each coordinate entry, create a manifest entry
            # (All images for a date share the same coordinate context)
            for coord_entry in entries_for_date:
                dap = coord_entry['dap']
                sequence_id = coord_entry['sequence_id']
                
                # Get 7-day average weather
                weather_context = get_7day_avg_weather(weather_df, target_date, dap)
                if weather_context is None:
                    insufficient_weather += 1
                    continue
                
                # Create manifest entry
                entry = {
                    'sequence_id': sequence_id,
                    'frame_index': image_id,
                    'elapsed_days': dap,
                    'image_path': os.path.join(split, filename),
                    'temperature': weather_context['temperature'],
                    'relative_humidity': weather_context['relative_humidity'],
                    'solar_radiation': weather_context['solar_radiation'],
                    'precipitation': weather_context['precipitation'],
                    'split': split
                }
                
                sequences[sequence_id].append(entry)
                manifest_entries.append(entry)
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Manifest Generation Report")
    print("="*60)
    print(f"Total manifest entries: {len(manifest_entries)}")
    print(f"Total sequences (plants): {len(sequences)}")
    print(f"  - With train samples: {sum(1 for s in sequences.values() if any(e['split']=='Train' for e in s))}")
    print(f"  - With val samples: {sum(1 for s in sequences.values() if any(e['split']=='Val' for e in s))}")
    print(f"  - With test samples: {sum(1 for s in sequences.values() if any(e['split']=='Test' for e in s))}")
    
    print(f"\nEntries with insufficient weather data: {insufficient_weather}")
    
    # Calculate DAP range
    if manifest_entries:
        daps = [e['elapsed_days'] for e in manifest_entries]
        print(f"\nDAP range: {min(daps)} - {max(daps)} days")
        print(f"Average samples per sequence: {len(manifest_entries) / len(sequences):.1f}")
    
    return manifest_entries


def main():
    print("Step 1: Generating Dataset Manifest")
    print("="*60)
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Output file: {OUTPUT_FILE}\n")
    
    # Load weather data
    print("Loading weather data...")
    if not os.path.exists(WEATHER_FILE):
        print(f"ERROR: Weather file not found: {WEATHER_FILE}")
        return False
    
    weather_df = load_weather_data(WEATHER_FILE)
    print(f"  Loaded {len(weather_df)} days of weather data")
    print(f"  Date range: {weather_df.index.min().date()} to {weather_df.index.max().date()}")
    
    # Verify directories exist
    if not os.path.isdir(IMAGE_BASE_DIR):
        print(f"ERROR: Image directory not found: {IMAGE_BASE_DIR}")
        return False
    
    if not os.path.isdir(COORDS_BASE_DIR):
        print(f"ERROR: Coordinate directory not found: {COORDS_BASE_DIR}")
        return False
    
    # Scan and build manifest
    print()
    manifest_entries = scan_images_and_build_manifest(IMAGE_BASE_DIR, COORDS_BASE_DIR, weather_df)
    
    if not manifest_entries:
        print("ERROR: No valid entries found in dataset!")
        return False
    
    # Save manifest
    print()
    print("Saving manifest...")
    manifest_df = pd.DataFrame(manifest_entries)
    
    # Sort by sequence and frame index for better organization
    manifest_df['frame_index_int'] = manifest_df['frame_index'].astype(int)
    manifest_df = manifest_df.sort_values(['sequence_id', 'frame_index_int'])
    manifest_df = manifest_df.drop('frame_index_int', axis=1)
    
    # Save without split column
    output_columns = [
        'sequence_id', 'frame_index', 'elapsed_days', 'image_path',
        'temperature', 'relative_humidity', 'solar_radiation', 'precipitation'
    ]
    manifest_df[output_columns].to_csv(OUTPUT_FILE, index=False)
    
    print(f"✓ Manifest saved to {OUTPUT_FILE}")
    print(f"  Total rows: {len(manifest_df)}\n")
    print("Sample rows:")
    print(manifest_df[output_columns].head(10))
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
