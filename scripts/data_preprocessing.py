#!/usr/bin/env python3
"""
Data Preprocessing Script
Cleans data, handles missing values, aggregates to hourly/daily levels, merges tariffs, and verifies households
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, dayofmonth, month, year, weekofyear, date_format, sum as spark_sum, avg, count, first
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import pandas as pd  # For tariff.xlsx
import os
from datetime import datetime

def create_spark_session():
    """Create Spark session optimized for 8GB systems"""
    os.environ['HADOOP_HOME'] = ''
    os.environ['HADOOP_USER_NAME'] = 'localuser'
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'

    return SparkSession.builder \
        .appName("EnergyDataPreprocessing") \
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

def load_and_validate_tariff_data(spark, data_dir):
    """Load and validate tariff.xlsx data"""
    tariff_path = os.path.join(data_dir, "Tariffs.xlsx")
    
    print(f"\n{'='*60}")
    print("LOADING TARIFF DATA:")
    print(f"{'='*60}")
    print(f"  Path: {tariff_path}")
    
    if not os.path.exists(tariff_path):
        print(f"  ℹ Tariff.xlsx not found - skipping tariff merge")
        print(f"{'='*60}\n")
        return None
    
    try:
        # Load tariff data with pandas
        tariff_df_pd = pd.read_excel(tariff_path)
        print(f"  ✓ Tariff Excel loaded: {tariff_df_pd.shape[0]} rows, {tariff_df_pd.shape[1]} columns")
        
        # Show tariff columns
        print(f"  Tariff columns: {list(tariff_df_pd.columns)}")
        
        # Convert to Spark DataFrame
        tariff_spark_df = spark.createDataFrame(tariff_df_pd)
        
        # Show tariff schema
        print(f"\n  Tariff Schema:")
        tariff_spark_df.printSchema()
        
        # Show sample tariff data
        print(f"\n  Sample Tariff Data:")
        tariff_spark_df.show(5, truncate=False)
        
        # Check for LCLid column (common join key)
        tariff_columns = [c.lower() for c in tariff_spark_df.columns]
        if 'lclid' not in tariff_columns:
            print(f"  ⚠ Warning: 'LCLid' column not found in tariff data")
            print(f"  Available columns: {tariff_spark_df.columns}")
        
        print(f"{'='*60}\n")
        return tariff_spark_df
        
    except Exception as e:
        print(f"  ✗ Error loading tariff data: {e}")
        print(f"{'='*60}\n")
        return None

def preprocess_data(spark, input_path, output_path, data_dir):
    """Preprocess the raw energy data"""
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"DATA PREPROCESSING STARTED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Read raw data
    print(f"[1/8] Reading raw data from: {input_path}")
    df = spark.read.parquet(input_path)

    print(f"      Raw data schema:")
    df.printSchema()
    initial_count = df.count()
    print(f"      ✓ Raw data count: {initial_count:,}\n")

    # Check initial households
    initial_households = df.select("LCLid").distinct().count()
    print(f"[2/8] Initial unique households: {initial_households:,}\n")

    # Handle missing values
    print(f"[3/8] Cleaning data and handling missing values...")
    before_clean = df.count()
    df = df.dropna(subset=["LCLid", "DateTime", "KWH/hh (per half hour) "])
    after_clean = df.count()
    print(f"      Records before: {before_clean:,}")
    print(f"      Records after:  {after_clean:,}")
    print(f"      Removed:        {before_clean - after_clean:,}\n")

    # Rename columns for consistency
    print(f"[4/8] Renaming columns...")
    df = df.withColumnRenamed("KWH/hh (per half hour) ", "energy_kwh")
    
    # Check if tariff_type column exists
    if "stdorToU" in df.columns:
        df = df.withColumnRenamed("stdorToU", "tariff_type")
        print(f"      ✓ Renamed 'stdorToU' to 'tariff_type'")
    else:
        print(f"      ℹ 'stdorToU' column not found in data")
    
    print(f"      ✓ Renamed 'KWH/hh (per half hour)' to 'energy_kwh'\n")

    # Replace "Null" strings with None
    print(f"[5/8] Handling invalid values...")
    df = df.withColumn("energy_kwh", 
                       F.when(col("energy_kwh") == "Null", None)
                        .otherwise(col("energy_kwh")))

    # Convert energy to float
    df = df.withColumn("energy_kwh", col("energy_kwh").cast("float"))

    # Drop rows where energy_kwh is null after casting
    before_null_drop = df.count()
    df = df.dropna(subset=["energy_kwh"])
    after_null_drop = df.count()
    print(f"      Removed {before_null_drop - after_null_drop:,} rows with null/invalid energy values\n")

    # Extract time features
    print(f"[6/8] Extracting time features...")
    df = df.withColumn("hour", hour(col("DateTime"))) \
           .withColumn("day", dayofmonth(col("DateTime"))) \
           .withColumn("month", month(col("DateTime"))) \
           .withColumn("year", year(col("DateTime"))) \
           .withColumn("weekday", date_format(col("DateTime"), "E")) \
           .withColumn("date", date_format(col("DateTime"), "yyyy-MM-dd"))
    print(f"      ✓ Added columns: hour, day, month, year, weekday, date\n")

    # Load and merge tariff data
    print(f"[7/8] Merging tariff data...")
    tariff_spark_df = load_and_validate_tariff_data(spark, data_dir)
    
    if tariff_spark_df is not None:
        # Check for common join key
        energy_columns_lower = [c.lower() for c in df.columns]
        tariff_columns_lower = [c.lower() for c in tariff_spark_df.columns]
        
        # Find LCLid column (case-insensitive)
        join_key = None
        for col_name in df.columns:
            if col_name.lower() == 'lclid':
                join_key = col_name
                break
        
        if join_key:
            before_merge = df.count()
            households_before = df.select(join_key).distinct().count()
            
            # Get tariff column names (excluding join key)
            tariff_columns = [c for c in tariff_spark_df.columns if c.lower() != 'lclid']
            
            print(f"      Joining on column: '{join_key}'")
            print(f"      Join type: left (preserves all energy records)")
            print(f"      Tariff columns to merge: {tariff_columns}")
            
            df = df.join(tariff_spark_df, df["DateTime"] == tariff_spark_df["TariffDateTime"], "left")

            
            after_merge = df.count()
            households_after = df.select(join_key).distinct().count()
            
            # Check merge success
            households_with_tariff = df.filter(col(tariff_columns[0]).isNotNull()).select(join_key).distinct().count() if tariff_columns else 0
            
            print(f"\n      Merge Results:")
            print(f"      ✓ Records before merge:  {before_merge:,}")
            print(f"      ✓ Records after merge:   {after_merge:,}")
            print(f"      ✓ Households before:     {households_before:,}")
            print(f"      ✓ Households after:      {households_after:,}")
            print(f"      ✓ Households w/ tariff:  {households_with_tariff:,}")
            
            if households_with_tariff < households_after:
                print(f"      ⚠ Warning: {households_after - households_with_tariff:,} households missing tariff data\n")
            else:
                print(f"      ✓ All households have tariff data!\n")
        else:
            print(f"      ✗ Error: Could not find 'LCLid' column for join")
            print(f"      Energy columns: {df.columns}")
            print(f"      Tariff columns: {tariff_spark_df.columns}\n")
    
    # Verify all households are included
    final_households = df.select("LCLid").distinct().count()
    print(f"[8/8] Final household verification:")
    print(f"      Initial households:  {initial_households:,}")
    print(f"      Final households:    {final_households:,}")
    if initial_households == final_households:
        print(f"      ✓ All households preserved!\n")
    else:
        print(f"      ⚠ Warning: {initial_households - final_households:,} households lost during preprocessing\n")

    # Aggregate to hourly level
    print(f"\n{'='*60}")
    print("AGGREGATING TO HOURLY LEVEL:")
    print(f"{'='*60}")
    
    # Include tariff info in aggregation
    agg_columns = ["LCLid", "date", "hour", "year", "month", "day", "weekday"]
    
    # Add tariff columns if they exist
    if tariff_spark_df is not None and tariff_columns:
        for tc in tariff_columns:
            if tc in df.columns:
                agg_columns.append(tc)
    
    hourly_df = df.groupBy(*agg_columns) \
                  .agg(spark_sum("energy_kwh").alias("hourly_energy_kwh"),
                       avg("energy_kwh").alias("avg_half_hour_energy"),
                       count("*").alias("num_readings"))

    print(f"  Hourly aggregated data:")
    hourly_df.show(5, truncate=False)
    hourly_count = hourly_df.count()
    print(f"  ✓ Hourly records: {hourly_count:,}\n")

    # Aggregate to daily level
    print(f"{'='*60}")
    print("AGGREGATING TO DAILY LEVEL:")
    print(f"{'='*60}")
    
    daily_agg_columns = ["LCLid", "date", "year", "month", "day", "weekday"]
    
    # Add tariff columns if they exist (use first() to get one value per group)
    daily_agg_exprs = [
        spark_sum("hourly_energy_kwh").alias("daily_energy_kwh"),
        avg("hourly_energy_kwh").alias("avg_hourly_energy"),
        spark_sum("num_readings").alias("total_readings")
    ]
    
    if tariff_spark_df is not None and tariff_columns:
        for tc in tariff_columns:
            if tc in hourly_df.columns:
                daily_agg_exprs.append(first(tc).alias(tc))
    
    daily_df = hourly_df.groupBy(*daily_agg_columns) \
                        .agg(*daily_agg_exprs)

    print(f"  Daily aggregated data:")
    daily_df.show(5, truncate=False)
    daily_count = daily_df.count()
    print(f"  ✓ Daily records: {daily_count:,}\n")

    # Reduce partitions before writing
    print(f"{'='*60}")
    print("SAVING PROCESSED DATA:")
    print(f"{'='*60}")
    
    hourly_df = hourly_df.coalesce(4)
    daily_df = daily_df.coalesce(4)

    # Save processed data
    hourly_output = os.path.join(output_path, "hourly")
    daily_output = os.path.join(output_path, "daily")
    
    print(f"  Saving hourly data to: {hourly_output}")
    hourly_df.write.mode("overwrite").parquet(hourly_output)
    print(f"  ✓ Hourly data saved")
    
    print(f"  Saving daily data to: {daily_output}")
    daily_df.write.mode("overwrite").parquet(daily_output)
    print(f"  ✓ Daily data saved\n")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"{'='*60}")
    print("PREPROCESSING SUMMARY:")
    print(f"{'='*60}")
    print(f"  Initial Records:     {initial_count:,}")
    print(f"  Final Records:       {after_null_drop:,}")
    print(f"  Initial Households:  {initial_households:,}")
    print(f"  Final Households:    {final_households:,}")
    print(f"  Hourly Records:      {hourly_count:,}")
    print(f"  Daily Records:       {daily_count:,}")
    print(f"  Tariff Data Merged:  {'Yes' if tariff_spark_df is not None else 'No'}")
    print(f"  Duration:            {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"  Status:              ✓ SUCCESS")
    print(f"{'='*60}\n")

    return hourly_df, daily_df

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
    print(f"{'='*60}\n")

    spark = create_spark_session()

    input_path = os.path.join(project_root, "data", "processed", "raw_energy_data")
    output_path = os.path.join(project_root, "data", "processed")

    print(f"Input path:  {input_path}")
    print(f"Output path: {output_path}")

    preprocess_data(spark, input_path, output_path, data_dir)

    spark.stop()