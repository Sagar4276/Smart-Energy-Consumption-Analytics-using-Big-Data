#!/usr/bin/env python3
"""
Anomaly Detection Model
Uses unsupervised ML to detect abnormal energy consumption
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, abs, stddev, avg, udf
from pyspark.sql.types import DoubleType
import math

def create_spark_session():
    """Create Spark session optimized for 8GB systems"""
    import os
    os.environ['HADOOP_HOME'] = ''
    os.environ['HADOOP_USER_NAME'] = 'localuser'
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'

    return SparkSession.builder \
        .appName("EnergyAnomalyDetection") \
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

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return float(math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2))))

def detect_anomalies(spark, features_path, output_path):
    """Detect anomalies using K-Means clustering with distance-based scoring"""
    from datetime import datetime
    
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"ANOMALY DETECTION STARTED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Load data
    print("[1/6] Loading feature data...")
    df = spark.read.parquet(features_path)
    total_count = df.count()
    households = df.select("LCLid").distinct().count()
    print(f"      ✓ Loaded {total_count:,} records from {households:,} households")
    
    print("\n      Sample data:")
    df.show(5, truncate=False)

    # Select features for clustering (expanded)
    print("\n[2/6] Preparing clustering features...")
    feature_cols = ["daily_energy_kwh", "rolling_avg_7d", "rolling_std_7d", "lag_1_day", "lag_7_day"]
    print(f"      ✓ Using {len(feature_cols)} features: {feature_cols}")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    kmeans = KMeans(k=3, featuresCol="scaled_features", predictionCol="cluster", seed=42)

    pipeline = Pipeline(stages=[assembler, scaler, kmeans])

    print("\n[3/6] Training K-Means clustering model (k=3)...")
    model = pipeline.fit(df)
    
    centers = model.stages[-1].clusterCenters()
    print(f"      ✓ Model trained successfully")
    print(f"\n      Cluster centers:")
    for i, center in enumerate(centers):
        print(f"        Cluster {i}: {center}")

    print("\n[4/6] Computing anomaly scores...")
    predictions = model.transform(df)

    # UDF to calculate distance to cluster center
    distance_udf = udf(lambda features, cluster: euclidean_distance(features, centers[cluster]), DoubleType())
    predictions = predictions.withColumn("distance_to_center", distance_udf(col("scaled_features"), col("cluster")))

    # Flag anomalies based on distance threshold (e.g., top 5% as anomalies)
    print("      Calculating anomaly threshold (95th percentile)...")
    distance_threshold = predictions.approxQuantile("distance_to_center", [0.95], 0.01)[0]
    print(f"      ✓ Anomaly threshold: {distance_threshold:.4f}")
    
    anomalies_df = predictions.withColumn("is_anomaly", (col("distance_to_center") > distance_threshold).cast("int"))

    print("\n[5/6] Analyzing results...")
    anomaly_count = anomalies_df.filter(col("is_anomaly") == 1).count()
    print(f"      ✓ Anomalies detected: {anomaly_count:,} out of {total_count:,} ({anomaly_count/total_count*100:.2f}%)")
    
    print("\n      Sample anomalies:")
    anomalies_df.filter(col("is_anomaly") == 1) \
        .select("LCLid", "date", "daily_energy_kwh", "rolling_avg_7d", "distance_to_center", "is_anomaly") \
        .orderBy(col("distance_to_center").desc()) \
        .show(10, truncate=False)

    print("\n[6/6] Saving results (memory-safe mode)...")
    print(f"      Output: {output_path}")
    
    # Memory-safe write with repartitioning
    num_partitions = max(20, total_count // 50000)  # ~50k records per partition
    print(f"      Repartitioning into {num_partitions} chunks for safe write...")
    
    try:
        anomalies_df.repartition(num_partitions) \
            .write.mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(output_path)
        print(f"      ✓ Successfully saved to Parquet (snappy compression)")
    except Exception as e:
        print(f"      ⚠️  Parquet write failed: {e}")
        print(f"      Falling back to CSV format...")
        csv_path = output_path + "_csv"
        anomalies_df.repartition(num_partitions) \
            .write.mode("overwrite") \
            .option("header", True) \
            .csv(csv_path)
        print(f"      ✓ Successfully saved to CSV: {csv_path}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*60}")
    print("ANOMALY DETECTION SUMMARY:")
    print(f"{'='*60}")
    print(f"  Total Records:       {total_count:,}")
    print(f"  Households:          {households:,}")
    print(f"  Anomalies Found:     {anomaly_count:,} ({anomaly_count/total_count*100:.2f}%)")
    print(f"  Anomaly Threshold:   {distance_threshold:.4f}")
    print(f"  Clusters Used:       3")
    print(f"  Features Used:       {len(feature_cols)}")
    print(f"  Output Partitions:   {num_partitions}")
    print(f"  Output Path:         {output_path}")
    print(f"  Duration:            {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"  Status:              ✓ SUCCESS")
    print(f"{'='*60}\n")

    return anomalies_df

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    spark = create_spark_session()

    features_path = os.path.join(project_root, "data", "processed", "energy_features")
    output_path = os.path.join(project_root, "data", "processed", "anomalies")

    print(f"Features path: {features_path}")
    print(f"Output path: {output_path}")

    detect_anomalies(spark, features_path, output_path)
    spark.stop()