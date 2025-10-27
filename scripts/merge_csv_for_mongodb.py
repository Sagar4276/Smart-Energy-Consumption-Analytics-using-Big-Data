#!/usr/bin/env python3
"""
Merge all CSV files into a comprehensive dataset for MongoDB import
Combines raw energy data, daily aggregations, features, forecasting results, and anomalies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
CSV_DIR = PROJECT_ROOT / "data" / "csv_exports"
OUTPUT_FILE = CSV_DIR / "complete_energy_dataset.csv"

def load_and_merge_csv_files():
    """Load all CSV files and merge them into a comprehensive dataset"""

    print("🔄 Loading and merging CSV files...")

    # Load all CSV files
    csv_files = {
        'raw_energy': CSV_DIR / 'raw_energy_data.csv',
        'daily': CSV_DIR / 'daily.csv',
        'features': CSV_DIR / 'energy_features.csv',
        'forecasting': CSV_DIR / 'forecasting_results.csv',
        'anomalies': CSV_DIR / 'anomalies.csv'
    }

    dataframes = {}

    # Load each CSV file
    for name, file_path in csv_files.items():
        try:
            print(f"📊 Loading {name} from {file_path.name}...")
            df = pd.read_csv(file_path, low_memory=False)
            dataframes[name] = df
            print(f"✅ Loaded {len(df):,} records from {name}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            continue

    # Start with the most comprehensive dataset (anomalies has all features)
    if 'anomalies' not in dataframes:
        print("❌ Anomalies data not available - cannot create comprehensive dataset")
        return None

    merged_df = dataframes['anomalies'].copy()
    print(f"📊 Starting with anomalies data: {len(merged_df):,} records")

    # Merge forecasting results (if available)
    if 'forecasting' in dataframes:
        try:
            # Ensure date columns are properly formatted
            merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
            forecasting_df = dataframes['forecasting'].copy()
            forecasting_df['date'] = pd.to_datetime(forecasting_df['date'], errors='coerce')

            # Merge on LCLid and date
            merged_df = merged_df.merge(forecasting_df, on=['LCLid', 'date'], how='left', suffixes=('', '_forecast'))
            print(f"✅ Merged forecasting data: {len(merged_df):,} total records")
        except Exception as e:
            print(f"⚠️ Could not merge forecasting data: {e}")

    # Merge daily data (if available and not already included)
    if 'daily' in dataframes:
        try:
            daily_df = dataframes['daily'].copy()
            daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')

            # Only merge columns that don't already exist
            existing_cols = set(merged_df.columns)
            daily_cols_to_merge = [col for col in daily_df.columns if col not in existing_cols or col in ['LCLid', 'date']]

            if len(daily_cols_to_merge) > 2:  # More than just LCLid and date
                daily_merge_df = daily_df[daily_cols_to_merge]
                merged_df = merged_df.merge(daily_merge_df, on=['LCLid', 'date'], how='left', suffixes=('', '_daily'))
                print(f"✅ Merged daily data: {len(merged_df):,} total records")
        except Exception as e:
            print(f"⚠️ Could not merge daily data: {e}")

    # Merge raw energy data (if available) - this is trickier due to DateTime format
    if 'raw_energy' in dataframes:
        try:
            raw_df = dataframes['raw_energy'].copy()
            raw_df['DateTime'] = pd.to_datetime(raw_df['DateTime'], errors='coerce')

            # Convert DateTime to date for merging
            raw_df['date'] = raw_df['DateTime'].dt.date
            raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce')

            # Aggregate raw data to daily level for merging
            raw_daily = raw_df.groupby(['LCLid', 'date']).agg({
                'KWH/hh (per half hour)': ['sum', 'mean', 'count']
            }).reset_index()

            # Flatten column names
            raw_daily.columns = ['LCLid', 'date', 'raw_total_kwh', 'raw_avg_kwh', 'raw_readings_count']

            # Only merge if these columns don't exist
            cols_to_merge = [col for col in raw_daily.columns if col not in merged_df.columns or col in ['LCLid', 'date']]

            if len(cols_to_merge) > 2:
                raw_merge_df = raw_daily[cols_to_merge]
                merged_df = merged_df.merge(raw_merge_df, on=['LCLid', 'date'], how='left', suffixes=('', '_raw'))
                print(f"✅ Merged raw energy data: {len(merged_df):,} total records")
        except Exception as e:
            print(f"⚠️ Could not merge raw energy data: {e}")

    # Clean up the merged dataframe
    print("🧹 Cleaning up merged dataframe...")

    # Remove duplicate columns (keep the first occurrence)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Fill missing values appropriately
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'is_anomaly':  # Don't fill anomaly flag
            merged_df[col] = merged_df[col].fillna(0)

    # Fill categorical columns
    categorical_cols = merged_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        merged_df[col] = merged_df[col].fillna('Unknown')

    # Sort by LCLid and date
    if 'date' in merged_df.columns:
        try:
            merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
            merged_df = merged_df.sort_values(['LCLid', 'date']).reset_index(drop=True)
        except:
            pass

    print(f"✅ Final merged dataset: {len(merged_df):,} records, {len(merged_df.columns)} columns")
    print(f"📊 Columns: {list(merged_df.columns)}")

    return merged_df

def save_merged_dataset(df):
    """Save the merged dataset to CSV"""

    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving merged dataset to {OUTPUT_FILE}...")

    # Save with compression to reduce file size
    df.to_csv(OUTPUT_FILE, index=False, compression='gzip')

    # Also save uncompressed version for MongoDB import
    uncompressed_file = OUTPUT_FILE.with_suffix('.csv')
    df.to_csv(uncompressed_file, index=False)

    print(f"✅ Saved compressed version: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"✅ Saved uncompressed version: {uncompressed_file} ({uncompressed_file.stat().st_size / 1024 / 1024:.1f} MB)")

    return uncompressed_file

def create_mongodb_import_instructions(csv_file):
    """Create MongoDB import instructions"""

    instructions_file = csv_file.parent / "mongodb_import_instructions.txt"

    instructions = f"""
# MongoDB Import Instructions for Complete Energy Dataset

## Dataset Overview
- File: {csv_file.name}
- Records: {pd.read_csv(csv_file, nrows=0).shape[1]:,} (estimated)
- Source: Merged from 5 CSV files (raw_energy_data, daily, energy_features, forecasting_results, anomalies)

## Import Command

### For local MongoDB:
mongoimport --db smart_energy --collection energy_data --file "{csv_file}" --type csv --headerline

### For MongoDB Atlas (Cloud):
mongoimport --uri "mongodb+srv://<username>:<password>@<cluster-url>/smart_energy?retryWrites=true&w=majority" --collection energy_data --file "{csv_file}" --type csv --headerline

## Alternative: Using mongo shell
mongo smart_energy
db.energy_data.drop()  // Clear existing data
// Then use mongoimport command above

## Data Structure
The dataset contains comprehensive energy analytics data with:
- Household information (LCLid)
- Time series data (date, DateTime)
- Energy consumption (daily_energy_kwh, raw_total_kwh, etc.)
- ML features (36+ engineered features)
- Forecasting results (predictions)
- Anomaly detection results (is_anomaly, z_score, etc.)
- Tariff information
- Temporal features (seasonal, calendar, lag features)

## Sample Query Examples

// Find all anomalies
db.energy_data.find({{"is_anomaly": 1}})

// Find data for specific household
db.energy_data.find({{"LCLid": "MAC000002"}})

// Find high consumption days
db.energy_data.find({{"daily_energy_kwh": {{"$gt": 20}}}})

// Aggregate average consumption by month
db.energy_data.aggregate([
  {{"$group": {{
    "_id": {{"$month": "$date"}},
    "avg_consumption": {{"$avg": "$daily_energy_kwh"}},
    "count": {{"$sum": 1}}
  }}}}
])

## Performance Tips
1. Create indexes on frequently queried fields:
   db.energy_data.createIndex({{"LCLid": 1}})
   db.energy_data.createIndex({{"date": 1}})
   db.energy_data.createIndex({{"is_anomaly": 1}})

2. For time series queries, consider creating a compound index:
   db.energy_data.createIndex({{"LCLid": 1, "date": 1}})

3. The dataset is optimized for analytics workloads with comprehensive feature engineering.
"""

    with open(instructions_file, 'w') as f:
        f.write(instructions)

    print(f"📝 Created MongoDB import instructions: {instructions_file}")

def main():
    """Main function to merge all CSV files"""

    print("🚀 Starting CSV file merging process...")
    print(f"📁 Working directory: {PROJECT_ROOT}")
    print(f"📂 CSV directory: {CSV_DIR}")

    # Check if CSV files exist
    csv_files = list(CSV_DIR.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in the csv_exports directory")
        return

    print(f"📊 Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")

    # Load and merge CSV files
    merged_df = load_and_merge_csv_files()

    if merged_df is None or merged_df.empty:
        print("❌ Failed to create merged dataset")
        return

    # Save merged dataset
    output_file = save_merged_dataset(merged_df)

    # Create MongoDB import instructions
    create_mongodb_import_instructions(output_file)

    print("\n" + "="*60)
    print("✅ SUCCESS: Complete energy dataset created!")
    print(f"📊 Final dataset: {len(merged_df):,} records, {len(merged_df.columns)} columns")
    print(f"💾 Saved to: {output_file}")
    print("🗄️ Ready for MongoDB import")
    print("="*60)

    # Show sample of the merged data
    print("\n📋 Sample of merged dataset:")
    print(merged_df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()