#!/usr/bin/env python3
"""
Feature Engineering Script
Creates time-series features for ML models
Includes tariff information and all household data
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, avg, stddev, max, min, sum as spark_sum
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from datetime import datetime
import os

def create_spark_session():
    """Create Spark session optimized for 8GB systems"""
    os.environ['HADOOP_HOME'] = ''
    os.environ['HADOOP_USER_NAME'] = 'localuser'
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'

    return SparkSession.builder \
        .appName("EnergyFeatureEngineering") \
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

def create_features(spark, input_path, output_path, batch_df=None):
    """Create features for forecasting and anomaly detection"""
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"FEATURE ENGINEERING STARTED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Read daily aggregated data (or use provided batch)
    if batch_df is not None:
        print(f"[1/7] Using provided batch data")
        df = batch_df
    else:
        print(f"[1/7] Reading daily aggregated data from: {input_path}")
        df = spark.read.parquet(input_path)
    print(f"      Input data schema:")
    df.printSchema()
    
    initial_count = df.count()
    initial_households = df.select("LCLid").distinct().count()
    print(f"      ✓ Initial records: {initial_count:,}")
    print(f"      ✓ Initial households: {initial_households:,}")
    
    # Identify tariff columns (columns that aren't standard energy/time columns)
    standard_columns = ['LCLid', 'date', 'year', 'month', 'day', 'weekday', 
                       'daily_energy_kwh', 'avg_hourly_energy', 'total_readings']
    tariff_columns = [c for c in df.columns if c not in standard_columns]
    
    if tariff_columns:
        print(f"      ✓ Found tariff columns: {tariff_columns}")
    else:
        print(f"      ℹ No tariff columns found (will proceed without)")
    
    print(f"\n      Sample input data:")
    df.show(5, truncate=False)

    # Sort by household and date
    print(f"\n[2/7] Sorting data by household and date...")
    df = df.orderBy("LCLid", "date")
    print(f"      ✓ Data sorted\n")

    # Define window for lag features
    print(f"[3/7] Creating lag features...")
    window_spec = Window.partitionBy("LCLid").orderBy("date")

    # Create lag features (previous days' energy)
    lag_periods = [1, 2, 3, 7, 14, 30]  # Extended to include 14 and 30 days
    for lag_days in lag_periods:
        df = df.withColumn(f"lag_{lag_days}_day", 
                          lag("daily_energy_kwh", lag_days).over(window_spec))
        print(f"      ✓ Created lag_{lag_days}_day feature")

    # Rolling averages
    print(f"\n[4/7] Creating rolling average features...")
    df = df.withColumn("rolling_avg_7d", 
                       avg("daily_energy_kwh").over(window_spec.rowsBetween(-7, -1)))
    print(f"      ✓ Created rolling_avg_7d")
    
    df = df.withColumn("rolling_avg_30d", 
                       avg("daily_energy_kwh").over(window_spec.rowsBetween(-30, -1)))
    print(f"      ✓ Created rolling_avg_30d")

    # Rolling standard deviation
    print(f"\n[5/7] Creating rolling statistics features...")
    df = df.withColumn("rolling_std_7d", 
                       stddev("daily_energy_kwh").over(window_spec.rowsBetween(-7, -1)))
    print(f"      ✓ Created rolling_std_7d")
    
    df = df.withColumn("rolling_std_30d", 
                       stddev("daily_energy_kwh").over(window_spec.rowsBetween(-30, -1)))
    print(f"      ✓ Created rolling_std_30d")
    
    # Rolling min/max for anomaly detection
    df = df.withColumn("rolling_min_7d", 
                       min("daily_energy_kwh").over(window_spec.rowsBetween(-7, -1)))
    df = df.withColumn("rolling_max_7d", 
                       max("daily_energy_kwh").over(window_spec.rowsBetween(-7, -1)))
    print(f"      ✓ Created rolling_min_7d and rolling_max_7d")
    
    # Rolling sum for weekly total
    df = df.withColumn("rolling_sum_7d", 
                       spark_sum("daily_energy_kwh").over(window_spec.rowsBetween(-7, -1)))
    print(f"      ✓ Created rolling_sum_7d")

    # Day of week features
    print(f"\n[6/7] Creating categorical features...")
    df = df.withColumn("is_weekend", 
                       F.when(col("weekday").isin(["Sat", "Sun"]), 1).otherwise(0))
    print(f"      ✓ Created is_weekend")

    # Seasonal features
    df = df.withColumn("is_summer", 
                       F.when(col("month").isin([6, 7, 8]), 1).otherwise(0))
    df = df.withColumn("is_winter", 
                       F.when(col("month").isin([12, 1, 2]), 1).otherwise(0))
    df = df.withColumn("is_spring", 
                       F.when(col("month").isin([3, 4, 5]), 1).otherwise(0))
    df = df.withColumn("is_fall", 
                       F.when(col("month").isin([9, 10, 11]), 1).otherwise(0))
    print(f"      ✓ Created seasonal features (summer, winter, spring, fall)")

    # Month as categorical feature (for cyclical patterns)
    df = df.withColumn("month_sin", F.sin(2 * 3.14159 * col("month") / 12))
    df = df.withColumn("month_cos", F.cos(2 * 3.14159 * col("month") / 12))
    print(f"      ✓ Created cyclical month features (month_sin, month_cos)")
    
    # Day of month as cyclical feature
    df = df.withColumn("day_sin", F.sin(2 * 3.14159 * col("day") / 31))
    df = df.withColumn("day_cos", F.cos(2 * 3.14159 * col("day") / 31))
    print(f"      ✓ Created cyclical day features (day_sin, day_cos)")

    # Tariff-based features if tariff columns exist
    if tariff_columns:
        print(f"\n      Creating tariff-based features...")
        # Only use actual tariff label column, not datetime
        tariff_col = None
        for tc in tariff_columns:
            if 'tariff' in tc.lower() and 'datetime' not in tc.lower():
                tariff_col = tc
                break

        if tariff_col:
            unique_tariffs = [row[tariff_col] for row in df.select(tariff_col).distinct().collect() if row[tariff_col] is not None]
            print(f"      Found tariff types in '{tariff_col}': {unique_tariffs}")

            for tariff_type in unique_tariffs:
                col_name = f"tariff_{tariff_type}".replace(' ', '_').lower()
                df = df.withColumn(col_name, F.when(col(tariff_col) == tariff_type, 1).otherwise(0))
                print(f"      ✓ Created {col_name}")
        else:
            print(f"      ℹ No valid tariff label column found — skipping one-hot encoding.")


    # Calculate energy change rate (day-over-day change)
    df = df.withColumn("energy_change", 
                       col("daily_energy_kwh") - col("lag_1_day"))
    df = df.withColumn("energy_change_pct", F.try_divide(col("energy_change"), col("lag_1_day")) * 100)

    print(f"      ✓ Created energy_change and energy_change_pct")

    # Deviation from rolling average (for anomaly detection)
    df = df.withColumn("deviation_from_avg_7d", 
                       col("daily_energy_kwh") - col("rolling_avg_7d"))
    df = df.withColumn("deviation_from_avg_30d", 
                       col("daily_energy_kwh") - col("rolling_avg_30d"))
    print(f"      ✓ Created deviation features")

    # Z-score for anomaly detection
    df = df.withColumn("z_score_7d", 
                       (col("daily_energy_kwh") - col("rolling_avg_7d")) / 
                       F.when(col("rolling_std_7d") > 0, col("rolling_std_7d")).otherwise(1))
    print(f"      ✓ Created z_score_7d for anomaly detection")

    # Drop rows with NaN from lag features (first 30 days per household)
    print(f"\n[7/7] Cleaning features...")
    before_dropna = df.count()
    df = df.dropna()
    after_dropna = df.count()
    final_households = df.select("LCLid").distinct().count()
    
    print(f"      Records before dropna: {before_dropna:,}")
    print(f"      Records after dropna:  {after_dropna:,}")
    print(f"      Removed:               {before_dropna - after_dropna:,} (warm-up period)")
    print(f"      Final households:      {final_households:,}")

    print(f"\n{'='*60}")
    print("FEATURES CREATED:")
    print(f"{'='*60}")
    df.printSchema()
    
    print(f"\n{'='*60}")
    print("SAMPLE FEATURE DATA:")
    print(f"{'='*60}")
    df.show(5, truncate=False)

    # Get feature statistics
    feature_columns = [c for c in df.columns if c not in ['LCLid', 'date', 'year', 'month', 'day', 'weekday']]
    print(f"\n      Total features created: {len(feature_columns)}")
    print(f"      Feature columns: {feature_columns[:10]}...")  # Show first 10

    # Save features
    print(f"\n{'='*60}")
    print("SAVING FEATURES:")
    print(f"{'='*60}")
    print(f"      Output path: {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.write.mode("overwrite").parquet(output_path)
    print(f"      ✓ Features saved successfully")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING SUMMARY:")
    print(f"{'='*60}")
    print(f"  Input Records:       {initial_count:,}")
    print(f"  Output Records:      {after_dropna:,}")
    print(f"  Input Households:    {initial_households:,}")
    print(f"  Output Households:   {final_households:,}")
    print(f"  Total Features:      {len(feature_columns)}")
    print(f"  Tariff Data Included: {'Yes' if tariff_columns else 'No'}")
    print(f"  Output Location:     {output_path}")
    print(f"  Duration:            {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"  Status:              ✓ SUCCESS")
    print(f"{'='*60}\n")

    return df

if __name__ == "__main__":
    # Get the absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print(f"\n{'='*60}")
    print("ENVIRONMENT SETUP:")
    print(f"{'='*60}")
    print(f"  Script location:  {script_dir}")
    print(f"  Project root:     {project_root}")
    print(f"{'='*60}\n")

    spark = create_spark_session()

    input_path = os.path.join(project_root, "data", "processed", "daily")
    output_path = os.path.join(project_root, "data", "processed", "energy_features")

    print(f"Input path:  {input_path}")
    print(f"Output path: {output_path}")

    # ========================================
    # BATCH PROCESSING MODE (Memory-Efficient)
    # ========================================
    print(f"\n{'='*60}")
    print("BATCH PROCESSING MODE ENABLED")
    print(f"{'='*60}\n")
    
    # 1️⃣ Read full dataset once to get household list
    print("[Step 1] Loading dataset to identify households...")
    full_df = spark.read.parquet(input_path)
    all_households = [r["LCLid"] for r in full_df.select("LCLid").distinct().collect()]
    total_records = full_df.count()
    
    print(f"  ✓ Total households: {len(all_households):,}")
    print(f"  ✓ Total records: {total_records:,}")
    print(f"  ✓ Avg records per household: {total_records//len(all_households):,}")
    
    # 2️⃣ Batch parameters (tune based on memory)
    batch_size = 500  # Process 100 households at a time (safe for 8GB system)
    batches = [all_households[i:i+batch_size] for i in range(0, len(all_households), batch_size)]
    
    print(f"\n[Step 2] Batch configuration:")
    print(f"  • Batch size: {batch_size} households")
    print(f"  • Total batches: {len(batches)}")
    print(f"  • Est. records per batch: ~{batch_size * (total_records//len(all_households)):,}")
    
    # Create temporary batch output directory
    batch_output_base = os.path.join(project_root, "data", "processed", "energy_features_batches")
    os.makedirs(batch_output_base, exist_ok=True)
    
    # 3️⃣ Process in batches
    print(f"\n[Step 3] Processing batches...")
    successful_batches = []
    failed_batches = []
    
    for idx, batch in enumerate(batches, 1):
        try:
            print(f"\n{'='*60}")
            print(f"🚀 BATCH {idx}/{len(batches)}")
            print(f"{'='*60}")
            print(f"  Households: {batch[0]} ... {batch[-1]}")
            print(f"  Count: {len(batch)} households")
            
            # Filter data for this batch
            batch_df = full_df.filter(col("LCLid").isin(batch))
            batch_count = batch_df.count()
            print(f"  Records in batch: {batch_count:,}")
            
            # Process this batch
            batch_output_path = os.path.join(batch_output_base, f"batch_{idx:03d}")
            create_features(spark, input_path=None, output_path=batch_output_path, batch_df=batch_df)
            
            successful_batches.append(idx)
            print(f"\n  ✓ Batch {idx} completed successfully")
            
            # Force garbage collection between batches
            batch_df.unpersist()
            del batch_df
            
        except Exception as e:
            print(f"\n  ✗ Batch {idx} FAILED: {e}")
            failed_batches.append(idx)
            import traceback
            traceback.print_exc()
            continue
    
    # 4️⃣ Merge all successful batches into final output
    if successful_batches:
        print(f"\n{'='*60}")
        print("MERGING BATCHES INTO FINAL OUTPUT")
        print(f"{'='*60}")
        print(f"  Successful batches: {len(successful_batches)}/{len(batches)}")
        
        # Read all batch outputs
        batch_paths = [os.path.join(batch_output_base, f"batch_{idx:03d}") for idx in successful_batches]
        batch_paths_str = [p for p in batch_paths if os.path.exists(p)]
        
        print(f"  Reading {len(batch_paths_str)} batch outputs...")
        
        # Use wildcard pattern for efficiency
        merged_df = spark.read.parquet(os.path.join(batch_output_base, "batch_*"))
        
        print(f"  Merged records: {merged_df.count():,}")
        print(f"  Merged households: {merged_df.select('LCLid').distinct().count():,}")
        
        # Write final output
        print(f"  Writing final output to: {output_path}")
        merged_df.write.mode("overwrite").parquet(output_path)
        
        print(f"  ✓ Final output saved successfully")
        
        # Optional: Clean up batch files to save space
        print(f"\n  Cleaning up temporary batch files...")
        import shutil
        shutil.rmtree(batch_output_base)
        print(f"  ✓ Temporary files removed")
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY:")
    print(f"{'='*60}")
    print(f"  Total batches: {len(batches)}")
    print(f"  Successful: {len(successful_batches)}")
    print(f"  Failed: {len(failed_batches)}")
    if failed_batches:
        print(f"  Failed batch numbers: {failed_batches}")
    print(f"  Final output: {output_path}")
    print(f"{'='*60}\n")

    spark.stop()