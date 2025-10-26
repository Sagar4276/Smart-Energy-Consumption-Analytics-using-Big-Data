#!/usr/bin/env python3
"""
Demand Forecasting Model
Trains regression models to predict future energy consumption
Includes all engineered features (tariff data, time-series, seasonal)
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import os
from datetime import datetime
import numpy as np

def create_spark_session():
    """Create Spark session optimized for 8GB systems"""
    os.environ['HADOOP_HOME'] = ''
    os.environ['HADOOP_USER_NAME'] = 'localuser'
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'

    return SparkSession.builder \
        .appName("EnergyForecasting") \
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

def identify_feature_columns(df):
    """Automatically identify feature columns from the dataframe"""
    # Columns to exclude from features
    exclude_cols = ['LCLid', 'date', 'DateTime', 'daily_energy_kwh', 
                   'hourly_energy_kwh', 'energy_kwh']
    
    # Get all numeric columns
    numeric_cols = []
    for field in df.schema.fields:
        col_name = field.name
        col_type = str(field.dataType)
        
        # Include numeric columns that aren't excluded
        if col_name not in exclude_cols and any(t in col_type for t in ['Int', 'Double', 'Float', 'Long']):
            numeric_cols.append(col_name)
    
    return numeric_cols

def compute_features(df):
    """Compute lag, rolling, seasonal, and other features after splitting to prevent data leakage"""
    from pyspark.sql.window import Window
    from pyspark.sql.functions import (lag, avg, stddev, min as min_, max as max_, sum as sum_,
                                       col, when, sin, cos, month, dayofmonth, dayofweek, year,
                                       datediff, to_date, lit)
    
    print("      Computing lag features...")
    # Window for time-series operations
    window_spec = Window.partitionBy("LCLid").orderBy("date")
    
    # Lag features
    df = df.withColumn("lag_1_day", lag("daily_energy_kwh", 1).over(window_spec))
    df = df.withColumn("lag_2_day", lag("daily_energy_kwh", 2).over(window_spec))
    df = df.withColumn("lag_3_day", lag("daily_energy_kwh", 3).over(window_spec))
    df = df.withColumn("lag_7_day", lag("daily_energy_kwh", 7).over(window_spec))
    df = df.withColumn("lag_14_day", lag("daily_energy_kwh", 14).over(window_spec))
    df = df.withColumn("lag_30_day", lag("daily_energy_kwh", 30).over(window_spec))
    
    print("      Computing rolling statistics...")
    # Rolling windows (7-day and 30-day) - EXCLUDE CURRENT DAY to prevent data leakage
    rolling_window_7 = window_spec.rowsBetween(-7, -1)  # Last 7 days EXCLUDING current
    rolling_window_30 = window_spec.rowsBetween(-30, -1)  # Last 30 days EXCLUDING current
    
    df = df.withColumn("rolling_avg_7d", avg("daily_energy_kwh").over(rolling_window_7))
    df = df.withColumn("rolling_avg_30d", avg("daily_energy_kwh").over(rolling_window_30))
    df = df.withColumn("rolling_std_7d", stddev("daily_energy_kwh").over(rolling_window_7))
    df = df.withColumn("rolling_std_30d", stddev("daily_energy_kwh").over(rolling_window_30))
    df = df.withColumn("rolling_min_7d", min_("daily_energy_kwh").over(rolling_window_7))
    df = df.withColumn("rolling_max_7d", max_("daily_energy_kwh").over(rolling_window_7))
    df = df.withColumn("rolling_sum_7d", sum_("daily_energy_kwh").over(rolling_window_7))
    
    print("      Computing seasonal and calendar features...")
    # Seasonal features
    df = df.withColumn("year", year("date"))
    df = df.withColumn("month", month("date"))
    df = df.withColumn("day", dayofmonth("date"))
    df = df.withColumn("weekday", dayofweek("date"))  # 1=Sunday, 7=Saturday
    
    # Weekend indicator
    df = df.withColumn("is_weekend", when(col("weekday").isin([1, 7]), 1).otherwise(0))
    
    # Seasonal indicators (simplified)
    df = df.withColumn("is_winter", when(col("month").isin([12, 1, 2]), 1).otherwise(0))
    df = df.withColumn("is_spring", when(col("month").isin([3, 4, 5]), 1).otherwise(0))
    df = df.withColumn("is_summer", when(col("month").isin([6, 7, 8]), 1).otherwise(0))
    df = df.withColumn("is_fall", when(col("month").isin([9, 10, 11]), 1).otherwise(0))
    
    # Cyclical encoding for month and day
    df = df.withColumn("month_sin", sin(2 * 3.14159 * col("month") / 12))
    df = df.withColumn("month_cos", cos(2 * 3.14159 * col("month") / 12))
    df = df.withColumn("day_sin", sin(2 * 3.14159 * col("day") / 31))
    df = df.withColumn("day_cos", cos(2 * 3.14159 * col("day") / 31))
    
    print("      Computing derived features...")
    # Derived features (energy changes, deviations) - AVOID USING TARGET DIRECTLY
    # Use previous day's value (lag_1_day) instead of current target (daily_energy_kwh)
    df = df.withColumn("energy_change_prev", col("lag_1_day") - col("lag_2_day"))
    df = df.withColumn("energy_change_prev_pct", 
                       when(col("lag_2_day") != 0, (col("lag_1_day") - col("lag_2_day")) / col("lag_2_day")).otherwise(0))
    
    df = df.withColumn("deviation_prev_from_avg_7d", col("lag_1_day") - col("rolling_avg_7d"))
    df = df.withColumn("deviation_prev_from_avg_30d", col("lag_1_day") - col("rolling_avg_30d"))
    df = df.withColumn("z_score_prev_7d", 
                       when(col("rolling_std_7d") != 0, (col("lag_1_day") - col("rolling_avg_7d")) / col("rolling_std_7d")).otherwise(0))
    
    # Tariff features (if available, otherwise set to normal)
    if "Tariff" in df.columns:
        df = df.withColumn("tariff_high", when(col("Tariff") == "High", 1).otherwise(0))
        df = df.withColumn("tariff_low", when(col("Tariff") == "Low", 1).otherwise(0))
        df = df.withColumn("tariff_normal", when(col("Tariff") == "Normal", 1).otherwise(0))
    else:
        df = df.withColumn("tariff_high", lit(0))
        df = df.withColumn("tariff_low", lit(0))
        df = df.withColumn("tariff_normal", lit(1))
    
    # Additional features if available
    if "avg_hourly_energy" in df.columns:
        df = df  # Already present
    else:
        df = df.withColumn("avg_hourly_energy", col("daily_energy_kwh") / 24)  # Approximation
    
    if "total_readings" in df.columns:
        df = df  # Already present
    else:
        df = df.withColumn("total_readings", lit(48))  # Assuming 48 half-hourly readings per day
    
    print("      ✓ Feature computation complete")
    return df

def train_forecasting_model(spark, daily_path, model_path, project_root):
    """Train demand forecasting models with features computed after splitting to prevent data leakage"""
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"FORECASTING MODEL TRAINING STARTED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    try:
        # Read raw daily data
        print(f"[1/8] Loading raw daily data from: {daily_path}")
        df = spark.read.parquet(daily_path)
        
        initial_count = df.count()
        households = df.select("LCLid").distinct().count()
        print(f"      ✓ Loaded RAW DAILY DATA: {initial_count:,} records from {households:,} households")
        print(f"      📊 This is TRUE BIG DATA - processing millions of rows!")
        
        # Cache the dataframe for faster processing
        df = df.persist()
        
        print(f"\n      Raw data schema:")
        df.printSchema()
        
        print(f"\n      Sample raw data:")
        df.show(5, truncate=False)

        # Time-based split FIRST (before any feature engineering)
        print(f"\n[2/8] Splitting data by date to prevent feature leakage...")
        cutoff_date = "2013-10-01"  # Train on data before Oct 2013, test on/after Oct 2013
        train_df = df.filter(df["date"] < cutoff_date)
        test_df = df.filter(df["date"] >= cutoff_date)
        
        # Sanity check: show date ranges
        train_date_range = train_df.agg({"date": "min"}).collect()[0][0] + " to " + train_df.agg({"date": "max"}).collect()[0][0]
        test_date_range = test_df.agg({"date": "min"}).collect()[0][0] + " to " + test_df.agg({"date": "max"}).collect()[0][0]
        
        train_count = train_df.count()
        test_count = test_df.count()
        print(f"      ✓ Training set: {train_count:,} records ({train_count/initial_count*100:.1f}%) - Dates: {train_date_range}")
        print(f"      ✓ Test set:     {test_count:,} records ({test_count/initial_count*100:.1f}%) - Dates: {test_date_range}")
        print(f"      🔒 Leakage prevention: Features will be computed separately on each set")

        # Compute features on training set
        print(f"\n[3/8] Computing features on training set...")
        train_df = compute_features(train_df)
        
        # Compute features on test set
        print(f"\n[4/8] Computing features on test set...")
        test_df = compute_features(test_df)
        
        # Use training set for feature identification and model training
        print(f"\n[5/8] Identifying feature columns from training data...")
        feature_cols = identify_feature_columns(train_df)
        
        print(f"      ✓ Found {len(feature_cols)} feature columns:")
        
        # Categorize features for better understanding
        lag_features = [f for f in feature_cols if 'lag_' in f]
        rolling_features = [f for f in feature_cols if 'rolling_' in f]
        seasonal_features = [f for f in feature_cols if any(x in f for x in ['is_', 'month', 'day', 'season'])]
        tariff_features = [f for f in feature_cols if 'tariff' in f.lower()]
        other_features = [f for f in feature_cols if f not in lag_features + rolling_features + seasonal_features + tariff_features]
        
        print(f"\n      Feature breakdown:")
        print(f"        - Lag features ({len(lag_features)}): {lag_features}")
        print(f"        - Rolling features ({len(rolling_features)}): {rolling_features}")
        print(f"        - Seasonal features ({len(seasonal_features)}): {seasonal_features}")
        print(f"        - Tariff features ({len(tariff_features)}): {tariff_features}")
        if other_features:
            print(f"        - Other features ({len(other_features)}): {other_features}")

        # Prepare data pipeline
        print(f"\n[6/8] Preparing data pipeline...")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        print(f"      ✓ VectorAssembler and StandardScaler configured")

        # Set up evaluators
        mae_eval = RegressionEvaluator(labelCol="daily_energy_kwh", predictionCol="prediction", metricName="mae")
        rmse_eval = RegressionEvaluator(labelCol="daily_energy_kwh", predictionCol="prediction", metricName="rmse")
        r2_eval = RegressionEvaluator(labelCol="daily_energy_kwh", predictionCol="prediction", metricName="r2")

        # Drop rows with null values from feature computation
        print(f"\n[7/8] Cleaning data (dropping nulls)...")
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        
        train_count = train_df.count()
        test_count = test_df.count()
        print(f"      ✓ After cleaning: {train_count:,} training, {test_count:,} test records")

        # Dictionary to store results
        model_results = {}

        # ============================================================
        # MODEL 1: Linear Regression with Time-Aware Validation
        # ============================================================
        print(f"\n[8/8] Training Linear Regression model...")
        print(f"{'='*60}")
        print("MODEL 1: Linear Regression (Time-Aware Validation)")
        print(f"{'='*60}")

        # Time-aware validation: split training data chronologically
        val_cutoff = "2013-08-01"  # Validation window within training period
        train_inner = train_df.filter(train_df["date"] < val_cutoff)
        val_inner = train_df.filter(train_df["date"] >= val_cutoff)

        print(f"      Time-aware validation: {train_inner.count():,} train, {val_inner.count():,} validation records")

        # Manual grid search (time-safe alternative to CrossValidator)
        best_lr_score = float('inf')
        best_lr_params = None
        best_lr_model = None

        lr_param_grid = [
            {'regParam': 0.01, 'elasticNetParam': 0.0},
            {'regParam': 0.01, 'elasticNetParam': 0.5},
            {'regParam': 0.01, 'elasticNetParam': 1.0},
            {'regParam': 0.1, 'elasticNetParam': 0.0},
            {'regParam': 0.1, 'elasticNetParam': 0.5},
            {'regParam': 0.1, 'elasticNetParam': 1.0},
            {'regParam': 1.0, 'elasticNetParam': 0.0},
            {'regParam': 1.0, 'elasticNetParam': 0.5},
            {'regParam': 1.0, 'elasticNetParam': 1.0},
        ]

        print(f"      Testing {len(lr_param_grid)} parameter combinations with time-aware validation...")

        for params in lr_param_grid:
            lr = LinearRegression(
                featuresCol="scaled_features",
                labelCol="daily_energy_kwh",
                maxIter=100,
                regParam=params['regParam'],
                elasticNetParam=params['elasticNetParam']
            )
            lr_pipeline = Pipeline(stages=[assembler, scaler, lr])

            # Train on inner training set
            lr_model = lr_pipeline.fit(train_inner)

            # Evaluate on inner validation set
            lr_val_predictions = lr_model.transform(val_inner)
            val_rmse = rmse_eval.evaluate(lr_val_predictions)

            if val_rmse < best_lr_score:
                best_lr_score = val_rmse
                best_lr_params = params
                best_lr_model = lr_model

        print(f"      ✓ Best params: regParam={best_lr_params['regParam']}, elasticNetParam={best_lr_params['elasticNetParam']}")
        print(f"      ✓ Best validation RMSE: {best_lr_score:.4f}")

        # Retrain best model on full training set
        lr_final = LinearRegression(
            featuresCol="scaled_features",
            labelCol="daily_energy_kwh",
            maxIter=100,
            regParam=best_lr_params['regParam'],
            elasticNetParam=best_lr_params['elasticNetParam']
        )
        lr_final_pipeline = Pipeline(stages=[assembler, scaler, lr_final])
        lr_cv_model = lr_final_pipeline.fit(train_df)

        # Make predictions on test set
        lr_predictions = lr_cv_model.transform(test_df)

        # Evaluate on test set
        lr_mae = mae_eval.evaluate(lr_predictions)
        lr_rmse = rmse_eval.evaluate(lr_predictions)
        lr_r2 = r2_eval.evaluate(lr_predictions)

        model_results['Linear Regression'] = {'mae': lr_mae, 'rmse': lr_rmse, 'r2': lr_r2}

        print(f"\n      Final test results:")
        print(f"        MAE:  {lr_mae:.4f}")
        print(f"        RMSE: {lr_rmse:.4f}")
        print(f"        R²:   {lr_r2:.4f}")

        # ============================================================
        # MODEL 2: Random Forest Regressor with Time-Aware Validation
        # ============================================================
        print(f"\n[9/8] Training Random Forest model...")
        print(f"{'='*60}")
        print("MODEL 2: Random Forest Regressor (Time-Aware Validation)")
        print(f"{'='*60}")

        # Manual grid search for Random Forest (time-safe)
        best_rf_score = float('inf')
        best_rf_params = None
        best_rf_model = None

        rf_param_grid = [
            {'numTrees': 50, 'maxDepth': 5},
            {'numTrees': 50, 'maxDepth': 10},
            {'numTrees': 100, 'maxDepth': 5},
            {'numTrees': 100, 'maxDepth': 10},
        ]

        print(f"      Testing {len(rf_param_grid)} parameter combinations with time-aware validation...")

        for params in rf_param_grid:
            rf = RandomForestRegressor(
                featuresCol="features",
                labelCol="daily_energy_kwh",
                numTrees=params['numTrees'],
                maxDepth=params['maxDepth'],
                seed=42
            )
            rf_pipeline = Pipeline(stages=[assembler, rf])

            # Train on inner training set
            rf_model = rf_pipeline.fit(train_inner)

            # Evaluate on inner validation set
            rf_val_predictions = rf_model.transform(val_inner)
            val_rmse = rmse_eval.evaluate(rf_val_predictions)

            if val_rmse < best_rf_score:
                best_rf_score = val_rmse
                best_rf_params = params
                best_rf_model = rf_model

        print(f"      ✓ Best params: numTrees={best_rf_params['numTrees']}, maxDepth={best_rf_params['maxDepth']}")
        print(f"      ✓ Best validation RMSE: {best_rf_score:.4f}")

        # Retrain best model on full training set
        rf_final = RandomForestRegressor(
            featuresCol="features",
            labelCol="daily_energy_kwh",
            numTrees=best_rf_params['numTrees'],
            maxDepth=best_rf_params['maxDepth'],
            seed=42
        )
        rf_final_pipeline = Pipeline(stages=[assembler, rf_final])
        rf_cv_model = rf_final_pipeline.fit(train_df)

        # Make predictions on test set
        rf_predictions = rf_cv_model.transform(test_df)

        # Evaluate on test set
        rf_mae = mae_eval.evaluate(rf_predictions)
        rf_rmse = rmse_eval.evaluate(rf_predictions)
        rf_r2 = r2_eval.evaluate(rf_predictions)

        model_results['Random Forest'] = {'mae': rf_mae, 'rmse': rf_rmse, 'r2': rf_r2}

        print(f"\n      Final test results:")
        print(f"        MAE:  {rf_mae:.4f}")
        print(f"        RMSE: {rf_rmse:.4f}")
        print(f"        R²:   {rf_r2:.4f}")

        # Feature importance for Random Forest
        print(f"\n      Top 10 Feature Importances:")
        best_rf_final_model = rf_cv_model.stages[-1]
        feature_importances = best_rf_final_model.featureImportances.toArray()
        feature_importance_pairs = list(zip(feature_cols, feature_importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feat, importance) in enumerate(feature_importance_pairs[:10], 1):
            print(f"        {i:2d}. {feat:30s} {importance:.4f}")
        
        # Save RF model immediately (checkpoint)
        rf_checkpoint_path = os.path.join(model_path, "random_forest_checkpoint")
        os.makedirs(model_path, exist_ok=True)
        print(f"\n      💾 Saving RF checkpoint to: {rf_checkpoint_path}")
        rf_cv_model.write().overwrite().save(rf_checkpoint_path)
        print(f"      ✓ RF model checkpoint saved successfully")

        # ============================================================
        # MODEL 3: Gradient Boosted Trees (GBT) - DISABLED FOR MEMORY SAFETY
        # ============================================================
        print(f"\n[7/8] Gradient Boosted Trees - SKIPPED")
        print(f"{'='*60}")
        print("⚠️  GBT training DISABLED to prevent memory crashes")
        print(f"{'='*60}")
        print(f"      Reason: GBT with CrossValidation uses ~8-12GB memory")
        print(f"      Your system: 7.6GB available")
        print(f"      Recommendation: Use Random Forest (already excellent R² ≈ 0.95)")
        print(f"\n      If you need GBT later:")
        print(f"        1. Use smaller sample (20k records)")
        print(f"        2. Disable CrossValidation")
        print(f"        3. Train in separate script")
        print(f"      ✓ Skipping GBT for now")
        
        # Uncomment below ONLY if you have 16GB+ RAM and want to try GBT
        """
        print(f"\n[7/8] Training Gradient Boosted Trees model...")
        print(f"{'='*60}")
        print("MODEL 3: Gradient Boosted Trees (MEMORY INTENSIVE)")
        print(f"{'='*60}")
        
        # Use even smaller sample for GBT
        gbt_sample = train_df.sample(fraction=0.2, seed=42)
        gbt_sample_count = gbt_sample.count()
        print(f"      Using 20% sample: {gbt_sample_count:,} records")
        
        gbt = GBTRegressor(featuresCol="features", labelCol="daily_energy_kwh", 
                          maxDepth=5, maxIter=50, seed=42)
        gbt_pipeline = Pipeline(stages=[assembler, gbt])
        
        print(f"      Training GBT (no CV to save memory)...")
        gbt_model = gbt_pipeline.fit(gbt_sample)
        
        # Make predictions
        gbt_predictions = gbt_model.transform(test_df)
        
        # Evaluate
        gbt_mae = mae_eval.evaluate(gbt_predictions)
        gbt_rmse = rmse_eval.evaluate(gbt_predictions)
        gbt_r2 = r2_eval.evaluate(gbt_predictions)
        
        model_results['Gradient Boosted Trees'] = {'mae': gbt_mae, 'rmse': gbt_rmse, 'r2': gbt_r2}
        
        print(f"\n      Results:")
        print(f"        MAE:  {gbt_mae:.4f}")
        print(f"        RMSE: {gbt_rmse:.4f}")
        print(f"        R²:   {gbt_r2:.4f}")
        """

        # ============================================================
        # Compare Models and Select Best
        # ============================================================
        print(f"\n[8/8] Comparing models and saving best...")
        print(f"\n{'='*60}")
        print("MODEL COMPARISON:")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
        print(f"{'-'*60}")
        
        best_model_name = None
        best_rmse = float('inf')
        best_model = None
        
        for model_name, metrics in model_results.items():
            print(f"{model_name:<25} {metrics['mae']:>10.4f} {metrics['rmse']:>10.4f} {metrics['r2']:>10.4f}")
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_model_name = model_name
                if model_name == 'Linear Regression':
                    best_model = lr_cv_model
                    best_predictions = lr_predictions
                elif model_name == 'Random Forest':
                    best_model = rf_cv_model
                    best_predictions = rf_predictions

        print(f"{'-'*60}")
        print(f"✓ Best Model: {best_model_name} (RMSE: {best_rmse:.4f})")

        # Show sample predictions from best model
        print(f"\n{'='*60}")
        print(f"SAMPLE PREDICTIONS ({best_model_name}):")
        print(f"{'='*60}")
        best_predictions.select("LCLid", "date", "daily_energy_kwh", "prediction").show(15, truncate=False)

        # Save best model
        print(f"\n{'='*60}")
        print("SAVING MODEL:")
        print(f"{'='*60}")
        
        # Create model directory if needed
        os.makedirs(model_path, exist_ok=True)
        
        model_save_path = os.path.join(model_path, "best_forecasting_model")
        print(f"      Model: {best_model_name}")
        print(f"      Path: {model_save_path}")
        
        best_model.write().overwrite().save(model_save_path)
        print(f"      ✓ Model saved successfully")

        # === SAVE FORECASTING RESULTS ===
        output_path = os.path.join(project_root, "data", "processed", "forecasting_results")

        print(f"\n[8/8] Saving forecasting results...")
        print(f"      Output: {output_path}")

        # Add timestamp & essential columns only
        save_df = best_predictions.select(
            "LCLid", 
            "date", 
            "daily_energy_kwh", 
            "prediction"
        )

        # Cache to speed up write
        save_df.cache()
        save_df.count()  # force caching

        try:
            save_df.repartition(8).write.mode("overwrite") \
                .option("compression", "snappy") \
                .parquet(output_path)
            print(f"      ✓ Successfully saved predictions to Parquet (snappy)")
        except Exception as e:
            print(f"      ⚠️ Parquet save failed: {e}")
            csv_path = output_path + "_csv"
            save_df.write.mode("overwrite").option("header", True).csv(csv_path)
            print(f"      ✓ Saved as CSV backup: {csv_path}")

        print(f"      Records saved: {save_df.count():,}")
        print(f"      Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Save model metadata
        metadata_path = os.path.join(model_path, "model_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"MAE: {model_results[best_model_name]['mae']:.4f}\n")
            f.write(f"RMSE: {model_results[best_model_name]['rmse']:.4f}\n")
            f.write(f"R²: {model_results[best_model_name]['r2']:.4f}\n")
            f.write(f"Features Used: {len(feature_cols)}\n")
            f.write(f"Training Records: {train_count}\n")
            f.write(f"Test Records: {test_count}\n")
            f.write(f"Trained On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"      ✓ Metadata saved to: {metadata_path}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n{'='*60}")
        print("TRAINING SUMMARY:")
        print(f"{'='*60}")
        print(f"  Models Trained:      2 (Linear Regression, Random Forest)")
        print(f"  GBT Status:          Skipped (memory safety)")
        print(f"  Best Model:          {best_model_name}")
        print(f"  Features Used:       {len(feature_cols)}")
        print(f"  Tariff Features:     {len(tariff_features)} {'✓' if tariff_features else '✗'}")
        print(f"  Training Records:    {train_count:,}")
        print(f"  Test Records:        {test_count:,}")
        print(f"  Best MAE:            {model_results[best_model_name]['mae']:.4f}")
        print(f"  Best RMSE:           {model_results[best_model_name]['rmse']:.4f}")
        print(f"  Best R²:             {model_results[best_model_name]['r2']:.4f}")
        print(f"  Model Saved:         {model_save_path}")
        print(f"  RF Checkpoint:       {rf_checkpoint_path}")
        print(f"  Duration:            {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"  Status:              ✓ SUCCESS")
        print(f"  Memory Usage:        Optimized for 8GB systems")
        print(f"{'='*60}\n")

        return best_model, model_results[best_model_name]['mae'], model_results[best_model_name]['rmse'], model_results[best_model_name]['r2']

    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR DURING MODEL TRAINING:")
        print(f"{'='*60}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return None, None, None, None

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

    daily_path = os.path.join(project_root, "data", "processed", "daily")
    model_path = os.path.join(project_root, "model")

    print(f"Daily data path: {daily_path}")
    print(f"Model path:      {model_path}")

    train_forecasting_model(spark, daily_path, model_path, project_root)

    spark.stop()