#!/usr/bin/env python3
"""
Data Ingestion Script
Loads raw CSV files into HDFS using PySpark
Prioritizes full data, falls back to partitioned, and handles tariff.xlsx
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp
import os
import pandas as pd  # For tariff.xlsx if needed
from datetime import datetime

def create_spark_session():
    """Create Spark session optimized for 8GB systems"""
    os.environ['HADOOP_HOME'] = ''
    os.environ['HADOOP_USER_NAME'] = 'localuser'
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'

    return SparkSession.builder \
        .appName("EnergyDataIngestion") \
        .master("local[2]") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse") \
        .config("spark.hadoop.security.authentication", "simple") \
        .config("spark.hadoop.security.authorization", "false") \
        .config("spark.hadoop.security.groups.cache.secs", "0") \
        .config("spark.hadoop.security.groups.negative-cache.secs", "0") \
        .config("spark.ui.enabled", "false") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "20") \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()

def ingest_data(spark, data_dir):
    """Ingest CSV files from local directory, prioritizing full data"""
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"DATA INGESTION STARTED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    try:
        df = None
        data_source = None

        # 1. Check for full LCL data (CC_LCL-FullData.csv)
        full_data_path = os.path.join(data_dir, "LCL-FullData", "CC_LCL-FullData.csv")
        print(f"[1/3] Checking for full dataset...")
        print(f"      Path: {full_data_path}")
        
        if os.path.exists(full_data_path):
            print(f"      ✓ Found full LCL data!")
            print(f"      Loading full dataset...")
            df = spark.read.csv(full_data_path, header=True, inferSchema=True)
            data_source = "Full Dataset (CC_LCL-FullData.csv)"
            print(f"      ✓ Full dataset loaded successfully\n")
        else:
            print(f"      ✗ Full LCL data not found.\n")
            print(f"[2/3] Checking for partitioned data...")

            # 2. Fall back to partitioned data
            partitioned_dir = os.path.join(data_dir, "Partitioned LCL Data", "Small LCL Data")
            print(f"      Path: {partitioned_dir}")
            
            if os.path.exists(partitioned_dir):
                print(f"      ✓ Found partitioned data directory!")
                
                # Get all CSV files and sort them for consistent processing
                csv_files = sorted([f for f in os.listdir(partitioned_dir) 
                                  if f.endswith('.csv') and f.startswith('LCL-June2015v2_')])
                
                print(f"      ✓ Found {len(csv_files)} CSV files")
                
                if csv_files:
                    print(f"      Files range: {csv_files[0]} to {csv_files[-1]}")
                    print(f"      Processing all {len(csv_files)} files...")
                    
                    # Process all files
                    dfs = []
                    failed_files = []
                    
                    for idx, csv_file in enumerate(csv_files, 1):
                        file_path = os.path.join(partitioned_dir, csv_file)
                        try:
                            df_part = spark.read.csv(file_path, header=True, inferSchema=True)
                            dfs.append(df_part)
                            if idx % 20 == 0 or idx == len(csv_files):
                                print(f"      Progress: {idx}/{len(csv_files)} files processed")
                        except Exception as e:
                            print(f"      ✗ Error reading {csv_file}: {e}")
                            failed_files.append(csv_file)
                            continue

                    if dfs:
                        print(f"      Combining {len(dfs)} DataFrames...")
                        df = dfs[0]
                        for df_part in dfs[1:]:
                            df = df.union(df_part)
                        print(f"      ✓ Combined all partitioned files successfully")
                        data_source = f"Partitioned Data ({len(dfs)} files)"
                        
                        if failed_files:
                            print(f"      ⚠ Warning: {len(failed_files)} files failed to load:")
                            for failed in failed_files[:5]:  # Show first 5
                                print(f"        - {failed}")
                            if len(failed_files) > 5:
                                print(f"        ... and {len(failed_files) - 5} more")
                    else:
                        print(f"      ✗ No valid partitioned files could be loaded.")
                else:
                    print(f"      ✗ No CSV files found in partitioned directory.")
            else:
                print(f"      ✗ Partitioned data directory not found.\n")

        if df is None:
            print(f"\n{'='*60}")
            print("ERROR: No data loaded. Check your data folder structure.")
            print(f"Expected locations:")
            print(f"  1. {os.path.join(data_dir, 'LCL-FullData', 'CC_LCL-FullData.csv')}")
            print(f"  2. {os.path.join(data_dir, 'Partitioned LCL Data', 'Small LCL Data', '*.csv')}")
            print(f"{'='*60}\n")
            return

        print(f"\n[3/3] Processing loaded data...")
        print(f"      Data source: {data_source}")

        # 3. Handle tariff.xlsx (load separately if needed)
        print(f"\n[Optional] Checking for tariff data...")
        tariff_path = os.path.join(data_dir, "Tariffs.xlsx")
        if os.path.exists(tariff_path):
            print(f"      ✓ Found tariff data: {tariff_path}")
            try:
                tariff_df = pd.read_excel(tariff_path)
                print(f"      ✓ Tariff data loaded: {tariff_df.shape[0]} rows, {tariff_df.shape[1]} columns")
                print(f"      Note: Tariff data loaded separately (not merged into energy data)")
                # If you need to join tariff with energy data, add logic here (e.g., by LCLid)
            except Exception as e:
                print(f"      ✗ Error loading tariff data: {e}")
        else:
            print(f"      ℹ Tariff.xlsx not found (optional)")

        # Show initial record count
        initial_count = df.count()
        print(f"\n      Initial record count: {initial_count:,}")

        # Convert DateTime column
        print(f"      Converting DateTime column...")
        df = df.withColumn("DateTime", to_timestamp(col("DateTime")))

        # Show schema and sample
        print(f"\n{'='*60}")
        print("DATA SCHEMA:")
        print(f"{'='*60}")
        df.printSchema()
        
        print(f"\n{'='*60}")
        print("SAMPLE DATA (First 5 rows):")
        print(f"{'='*60}")
        df.show(5, truncate=False)

        # Write to local Parquet format for better performance
        # Use Windows-compatible path
        output_path = os.path.join(project_root, "data", "processed", "raw_energy_data")
        output_path = os.path.abspath(output_path)  # Ensure absolute path
        
        print(f"\n{'='*60}")
        print("SAVING PROCESSED DATA:")
        print(f"{'='*60}")
        print(f"      Output path: {output_path}")
        print(f"      Format: Parquet")
        print(f"      Mode: Overwrite")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.write.mode("overwrite").parquet(output_path)

        final_count = df.count()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n{'='*60}")
        print("INGESTION SUMMARY:")
        print(f"{'='*60}")
        print(f"  Data Source:     {data_source}")
        print(f"  Total Records:   {final_count:,}")
        print(f"  Output Location: {output_path}")
        print(f"  Start Time:      {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End Time:        {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration:        {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"  Status:          ✓ SUCCESS")
        print(f"{'='*60}\n")

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n{'='*60}")
        print("ERROR DURING INGESTION:")
        print(f"{'='*60}")
        print(f"  Error: {e}")
        print(f"  Duration before error: {duration:.2f} seconds")
        print(f"  Status: ✗ FAILED")
        print(f"{'='*60}\n")
        raise

if __name__ == "__main__":
    # Get the absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")

    print(f"\n{'='*60}")
    print("ENVIRONMENT SETUP:")
    print(f"{'='*60}")
    print(f"  Script location:  {script_dir}")
    print(f"  Project root:     {project_root}")
    print(f"  Data directory:   {data_dir}")
    print(f"  OS:               {os.name}")
    print(f"{'='*60}\n")

    spark = create_spark_session()

    ingest_data(spark, data_dir)

    spark.stop()