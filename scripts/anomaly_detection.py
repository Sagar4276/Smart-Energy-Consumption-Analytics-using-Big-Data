#!/usr/bin/env python3
"""
Anomaly Detection Model
Uses unsupervised ML to detect abnormal energy consumption
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, abs, stddev, avg, udf, dayofweek, mean
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
import math

def create_spark_session():
    """Create Spark session optimized for 8GB systems"""
    import os
    os.environ['HADOOP_HOME'] = ''
    os.environ['HADOOP_USER_NAME'] = 'localuser'
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'
    # Disable problematic security features on Windows
    os.environ['SPARK_SECURITY_MANAGER_ENABLED'] = 'false'

    return SparkSession.builder \
        .appName("EnergyAnomalyDetection") \
        .master("local[4]") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse") \
        .config("spark.hadoop.security.authentication", "simple") \
        .config("spark.hadoop.security.authorization", "false") \
        .config("spark.hadoop.security.groups.cache.secs", "0") \
        .config("spark.hadoop.security.groups.negative-cache.secs", "0") \
        .config("spark.hadoop.security.UserGroupInformation.getCurrentUser", "false") \
        .config("spark.ui.enabled", "false") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.shuffle.targetPostShuffleInputSize", "64MB") \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return float(math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2))))

def detect_anomalies(spark, features_path, predictions_path, output_path):
    """Detect anomalies using hybrid approach: K-Means clustering + prediction residuals + temporal context"""
    from datetime import datetime
    
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"HYBRID ANOMALY DETECTION STARTED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Load feature data
    print("[1/7] Loading feature data...")
    df = spark.read.parquet(features_path)
    total_count = df.count()
    households = df.select("LCLid").distinct().count()
    print(f"      ✓ Loaded {total_count:,} records from {households:,} households")
    
    # Cache feature data for performance
    df.cache()
    
    # Load predictions and compute residuals
    print("\n[2/7] Loading predictions and computing residuals...")
    predictions_df = spark.read.parquet(predictions_path)
    predictions_df = predictions_df.withColumn("residual", abs(col("daily_energy_kwh") - col("prediction")))
    print(f"      ✓ Computed residuals for {predictions_df.count():,} predictions")
    
    # Cache predictions for performance
    predictions_df.cache()
    
    # Join features with residuals
    print("\n[3/7] Joining features with prediction residuals...")
    df = df.join(predictions_df.select("LCLid", "date", "prediction", "residual"), 
                 ["LCLid", "date"], "left")
    
    # Cache joined data for performance
    df.cache()
    
    # Add temporal features
    print("\n[4/7] Adding temporal context features...")
    df = df.withColumn("day_of_week", dayofweek("date"))
    df = df.withColumn("is_weekend", (col("day_of_week") >= 6).cast("int"))
    
    print("\n      Sample data with new features:")
    df.select("LCLid", "date", "daily_energy_kwh", "residual", "day_of_week", "is_weekend").show(5, truncate=False)

    # Select features for clustering (hybrid: consumption + residuals + temporal)
    print("\n[5/7] Preparing hybrid clustering features...")
    feature_cols = [
        "daily_energy_kwh", "rolling_avg_7d", "rolling_std_7d", 
        "lag_1_day", "lag_7_day", "residual", 
        "day_of_week", "is_weekend"
    ]
    print(f"      ✓ Using {len(feature_cols)} hybrid features: {feature_cols}")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    kmeans = KMeans(k=5, featuresCol="scaled_features", predictionCol="cluster", seed=42)  # Increased k for finer clusters

    pipeline = Pipeline(stages=[assembler, scaler, kmeans])

    print("\n[6/7] Training K-Means clustering model (k=5)...")
    model = pipeline.fit(df)
    
    centers = model.stages[-1].clusterCenters()
    print(f"      ✓ Model trained successfully")
    print(f"\n      Cluster centers (first 3 dimensions):")
    for i, center in enumerate(centers):
        print(f"        Cluster {i}: {center[:3]}...")

    print("\n[7/7] Computing hybrid anomaly scores...")
    predictions = model.transform(df)

    # UDF to calculate distance to cluster center
    distance_udf = udf(lambda features, cluster: euclidean_distance(features, centers[cluster]), DoubleType())
    predictions = predictions.withColumn("distance_to_center", distance_udf(col("scaled_features"), col("cluster")))

    # Per-cluster z-score anomaly detection (more adaptive than global threshold)
    print("      Computing per-cluster z-scores...")
    cluster_window = Window.partitionBy("cluster")
    
    predictions = predictions.withColumn("cluster_mean_dist", mean("distance_to_center").over(cluster_window))
    predictions = predictions.withColumn("cluster_std_dist", stddev("distance_to_center").over(cluster_window))
    
    predictions = predictions.withColumn(
        "z_score", 
        (col("distance_to_center") - col("cluster_mean_dist")) / col("cluster_std_dist")
    )
    
    # Flag anomalies based on z-score > 3 (99.7% confidence interval)
    anomalies_df = predictions.withColumn("is_anomaly", (col("z_score") > 3).cast("int"))

    print("\n[8/8] Analyzing hybrid anomaly results...")
    anomaly_count = anomalies_df.filter(col("is_anomaly") == 1).count()
    print(f"      ✓ Anomalies detected: {anomaly_count:,} out of {total_count:,} ({anomaly_count/total_count*100:.2f}%)")
    
    print("\n      Top anomalies by z-score:")
    anomalies_df.filter(col("is_anomaly") == 1) \
        .select("LCLid", "date", "daily_energy_kwh", "residual", "day_of_week", "z_score", "cluster") \
        .orderBy(col("z_score").desc()) \
        .show(10, truncate=False)

    print("\n[9/9] Saving hybrid results...")
    print(f"      Output: {output_path}")
    
    # Memory-safe write with repartitioning
    num_partitions = max(20, total_count // 50000)
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
    print("HYBRID ANOMALY DETECTION SUMMARY:")
    print(f"{'='*60}")
    print(f"  Total Records:       {total_count:,}")
    print(f"  Households:          {households:,}")
    print(f"  Anomalies Found:     {anomaly_count:,} ({anomaly_count/total_count*100:.2f}%)")
    print(f"  Clusters Used:       5")
    print(f"  Features Used:       {len(feature_cols)} (hybrid)")
    print(f"  Anomaly Method:      Per-cluster z-score (>3)")
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
    predictions_path = os.path.join(project_root, "data", "processed", "forecasting_results")
    output_path = os.path.join(project_root, "data", "processed", "anomalies")

    print(f"Features path: {features_path}")
    print(f"Predictions path: {predictions_path}")
    print(f"Output path: {output_path}")

    detect_anomalies(spark, features_path, predictions_path, output_path)
    spark.stop()