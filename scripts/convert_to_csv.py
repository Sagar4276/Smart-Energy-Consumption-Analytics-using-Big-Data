#!/usr/bin/env python3
"""
Convert Sample Parquet Files to CSV for MongoDB Import
Takes the second file (part-00001) from each processed data folder and samples to 10K rows max
"""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def convert_sample_parquet_to_csv():
    """Convert sample parquet files to CSV for MongoDB"""

    # Base directory
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed"
    csv_output_dir = base_dir / "data" / "csv_exports"

    # Create CSV output directory
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    # Define the folders and their corresponding second parquet files
    folders_and_files = {
        'raw_energy_data': 'part-00001-3a019976-4136-40a2-a798-1fefdba1cdd9-c000.snappy.parquet',
        'daily': 'part-00001-2f975aea-f9dc-42be-a9aa-2520acbc2e30-c000.snappy.parquet',
        'energy_features': 'part-00001-e2302739-0529-4ba7-b44a-5db06a26a9de-c000.snappy.parquet',
        'forecasting_results': 'part-00001-e7fa7e0e-807d-4baa-8961-7d3790bf0982-c000.snappy.parquet',
        'anomalies': 'part-00001-848befb5-4366-409d-8d7f-d647a970f334-c000.snappy.parquet'
    }

    print("� Converting Sample Parquet Files to CSV for MongoDB")
    print("=" * 60)
    print(f"� Processing directory: {processed_dir}")
    print(f"📁 Output directory: {csv_output_dir}")
    print("📊 Taking second file (part-00001) from each folder, max 10K rows")
    print()

    successful_conversions = 0
    total_start_time = datetime.now()

    for folder_name, parquet_file in folders_and_files.items():
        try:
            print(f"� Processing: {folder_name}")
            print("-" * 40)

            # Full path to parquet file
            parquet_path = processed_dir / folder_name / parquet_file
            csv_path = csv_output_dir / f"{folder_name}.csv"

            if not parquet_path.exists():
                print(f"⚠️ File not found: {parquet_path}")
                continue

            start_time = datetime.now()

            # Read parquet file
            print(f"� Reading {parquet_file}...")
            df = pd.read_parquet(parquet_path)

            original_rows = len(df)
            print(f"📊 Loaded {original_rows:,} rows, {len(df.columns)} columns")

            # Sample a smaller subset for MongoDB (max 10,000 rows)
            if len(df) > 10000:
                df = df.head(10000)
                print(f"📏 Sampled to 10,000 rows (from {original_rows:,} total)")
            else:
                print(f"📏 Using all {len(df)} rows (less than 10K)")

            # Save as CSV
            print(f"💾 Writing to CSV: {csv_path.name}")
            df.to_csv(csv_path, index=False)

            # Get file size
            file_size_kb = csv_path.stat().st_size / 1024
            duration = datetime.now() - start_time

            print(f"✅ Successfully converted!")
            print(f"   📁 Output: {csv_path.name}")
            print(f"   📊 Size: {file_size_kb:.1f} KB")
            print(f"   ⏱️ Duration: {duration.total_seconds():.2f} seconds")
            successful_conversions += 1

        except Exception as e:
            print(f"❌ Error converting {folder_name}: {e}")

        print()

    # Summary
    total_duration = datetime.now() - total_start_time
    print("=" * 60)
    print("📊 CONVERSION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful conversions: {successful_conversions}/{len(folders_and_files)}")
    print(f"⏱️ Total time: {total_duration.total_seconds():.2f} seconds")
    print(f"📁 CSV files saved to: {csv_output_dir}")

    if successful_conversions == len(folders_and_files):
        print("🎉 All conversions completed successfully!")
        print("\n💡 Next Steps for MongoDB:")
        print("   1. Open MongoDB Atlas/Compass")
        print("   2. Create database: 'smart_energy_analytics'")
        print("   3. Import each CSV file as a collection")
        print("   4. Use collection names: raw_energy_data, daily_energy_data, etc.")
    else:
        print(f"⚠️ {len(folders_and_files) - successful_conversions} conversions failed")

    # List created CSV files
    csv_files = list(csv_output_dir.glob("*.csv"))
    if csv_files:
        print("\n📋 Created CSV Files for MongoDB:")
        for csv_file in sorted(csv_files):
            size_kb = csv_file.stat().st_size / 1024
            print(f"   • {csv_file.name}: {size_kb:.1f} KB")

def main():
    """Main function"""
    convert_sample_parquet_to_csv()

if __name__ == "__main__":
    main()