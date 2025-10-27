# Hadoop MapReduce Concepts in Smart Energy Analytics

## 📊 Understanding Hadoop MapReduce in Our PySpark Implementation

### 🎯 **Project Context**
While our project uses **PySpark** for big data processing, all PySpark operations can be mapped to traditional **Hadoop MapReduce** concepts. This document explains how our energy analytics pipeline implements MapReduce patterns and demonstrates partitioning strategies.

---

## 🏗️ **Hadoop MapReduce Architecture Overview**

### **Traditional Hadoop MapReduce Flow**
```
Input Data → Split → Map Tasks → Shuffle & Sort → Reduce Tasks → Output
```

### **Our PySpark Equivalent**
```
Raw CSV Files → Data Ingestion (Map) → Preprocessing (MapReduce) → Feature Engineering (MapReduce) → ML Training (Reduce) → Output
```

---

## 📁 **Phase 1: Data Ingestion - Map-Only Operations**

### **MapReduce Concept: Data Loading & Initial Processing**
```python
# Our PySpark Implementation (data_ingestion.py)
def ingest_data(spark, data_dir):
    # MAP PHASE: Read multiple CSV files in parallel
    df = spark.read.csv(full_data_path, header=True, inferSchema=True)

    # MAP PHASE: Schema validation and type conversion
    df = df.withColumn("DateTime", to_timestamp(col("DateTime")))

    # OUTPUT: Partitioned Parquet files
    df.write.mode("overwrite").parquet(output_path)
```

### **Hadoop MapReduce Equivalent**
```java
// Mapper Class
public class EnergyDataMapper extends Mapper<LongWritable, Text, Text, Text> {
    public void map(LongWritable key, Text value, Context context) {
        // Parse CSV line
        String[] fields = value.toString().split(",");
        String householdId = fields[0];
        String dateTime = fields[1];
        String energyValue = fields[2];

        // Emit key-value pairs
        context.write(new Text(householdId), new Text(dateTime + "," + energyValue));
    }
}

// No Reducer needed for this phase (Map-only job)
```

### **Partitioning Strategy**
```
Input: 167M+ records across multiple CSV files
Map Tasks: One per input file (parallel processing)
Shuffle: Hash partitioning by LCLid (household ID)
Output: 64 Parquet partitions (part-00000 to part-00063)
```

---

## 🔧 **Phase 2: Data Preprocessing - MapReduce Operations**

### **MapReduce Concept: Data Cleaning & Aggregation**
```python
# Our PySpark Implementation (data_preprocessing.py)
def preprocess_data(spark, input_path, output_path, data_dir):

    # MAP PHASE: Clean and transform each record
    df = df.dropna(subset=["LCLid", "DateTime", "energy_kwh"])
    df = df.withColumn("hour", hour(col("DateTime"))) \
           .withColumn("date", date_format(col("DateTime"), "yyyy-MM-dd"))

    # SHUFFLE & SORT: Group by household and date
    # REDUCE PHASE: Aggregate to daily level
    daily_df = df.groupBy("LCLid", "date", "year", "month", "day", "weekday") \
                 .agg(sum("energy_kwh").alias("daily_energy_kwh"),
                      avg("energy_kwh").alias("avg_hourly_energy"),
                      count("*").alias("total_readings"))
```

### **Hadoop MapReduce Equivalent**
```java
// Mapper: Extract features and emit intermediate key-value pairs
public class PreprocessingMapper extends Mapper<LongWritable, Text, Text, Text> {
    public void map(LongWritable key, Text value, Context context) {
        String[] fields = value.toString().split(",");
        String householdId = fields[0];
        String dateTime = fields[1];
        double energy = Double.parseDouble(fields[2]);

        // Extract time features
        String date = extractDate(dateTime);
        String compositeKey = householdId + "_" + date;

        // Emit: (household_date, energy_value)
        context.write(new Text(compositeKey), new Text(String.valueOf(energy)));
    }
}

// Reducer: Aggregate daily energy consumption
public class PreprocessingReducer extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text key, Iterable<Text> values, Context context) {
        double sum = 0.0;
        int count = 0;
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;

        for (Text value : values) {
            double energy = Double.parseDouble(value.toString());
            sum += energy;
            count++;
            max = Math.max(max, energy);
            min = Math.min(min, energy);
        }

        double avg = sum / count;
        String result = sum + "," + avg + "," + count + "," + max + "," + min;
        context.write(key, new Text(result));
    }
}
```

### **Partitioning Strategy**
```
Input Partitions: 64 Parquet files (from ingestion)
Map Tasks: 64 parallel mappers (one per partition)
Shuffle Key: Composite key (LCLid + date)
Reduce Tasks: 20 reducers (configured spark.sql.shuffle.partitions)
Output Partitions: 4 final Parquet files (coalesced for efficiency)
```

---

## ⚙️ **Phase 3: Feature Engineering - Complex MapReduce**

### **MapReduce Concept: Feature Creation Pipeline**
```python
# Our PySpark Implementation (feature_engineering.py)
def create_features(spark, input_path, output_path):

    # MAP PHASE: Load and prepare data
    df = spark.read.parquet(input_path)

    # WINDOW FUNCTIONS (MapReduce pattern)
    window_spec = Window.partitionBy("LCLid").orderBy("date")

    # REDUCE PHASE: Create lag features
    df = df.withColumn("lag_1d", lag("daily_energy_kwh", 1).over(window_spec))
    df = df.withColumn("lag_7d", lag("daily_energy_kwh", 7).over(window_spec))

    # REDUCE PHASE: Rolling statistics
    df = df.withColumn("rolling_avg_7d", avg("daily_energy_kwh")
                      .over(window_spec.rowsBetween(-6, 0)))
    df = df.withColumn("rolling_std_7d", stddev("daily_energy_kwh")
                      .over(window_spec.rowsBetween(-6, 0)))
```

### **Hadoop MapReduce Equivalent**
```java
// Mapper: Emit time series data with timestamps
public class FeatureMapper extends Mapper<LongWritable, Text, Text, Text> {
    public void map(LongWritable key, Text value, Context context) {
        String[] fields = value.toString().split(",");
        String householdId = fields[0];
        String date = fields[1];
        double energy = Double.parseDouble(fields[2]);

        // Emit with timestamp for sorting
        long timestamp = convertToTimestamp(date);
        String keyWithTime = householdId + "_" + timestamp;
        context.write(new Text(keyWithTime), new Text(date + "," + energy));
    }
}

// Reducer: Create lag and rolling features
public class FeatureReducer extends Reducer<Text, Text, Text, Text> {
    private List<Double> energyHistory = new ArrayList<>();

    public void reduce(Text key, Iterable<Text> values, Context context) {
        String[] keyParts = key.toString().split("_");
        String householdId = keyParts[0];

        // Collect all values for this household (sorted by time)
        energyHistory.clear();
        for (Text value : values) {
            String[] fields = value.toString().split(",");
            double energy = Double.parseDouble(fields[1]);
            energyHistory.add(energy);
        }

        // Create features for each time point
        for (int i = 0; i < energyHistory.size(); i++) {
            double currentEnergy = energyHistory.get(i);
            StringBuilder features = new StringBuilder();

            // Lag features
            if (i >= 1) features.append(energyHistory.get(i-1)).append(",");
            if (i >= 7) features.append(energyHistory.get(i-7)).append(",");

            // Rolling statistics (last 7 days)
            int start = Math.max(0, i-6);
            List<Double> window = energyHistory.subList(start, i+1);
            double avg = window.stream().mapToDouble(d -> d).average().orElse(0);
            double std = calculateStdDev(window);

            features.append(avg).append(",").append(std);

            context.write(new Text(householdId + "_" + i),
                         new Text(currentEnergy + "," + features.toString()));
        }
    }
}
```

---

## 🎯 **Partitioning Strategy Deep Dive**

### **1. Input Partitioning**
```python
# How we handle input partitioning
spark.conf.set("spark.sql.files.maxPartitionBytes", "128MB")  # Default
spark.conf.set("spark.sql.shuffle.partitions", "20")  # Shuffle partitions

# Reading creates partitions automatically
df = spark.read.parquet(input_path)  # Creates 64 partitions from 64 files
print(f"Input partitions: {df.rdd.getNumPartitions()}")  # Output: 64
```

### **2. Shuffle Partitioning**
```python
# GroupBy operations trigger shuffle
daily_df = df.groupBy("LCLid", "date") \
             .agg(sum("energy_kwh").alias("daily_energy_kwh"))

# Spark automatically partitions by hash of groupBy keys
# spark.sql.shuffle.partitions = 20 (configured)
print(f"Shuffle partitions: {daily_df.rdd.getNumPartitions()}")  # Output: 20
```

### **3. Output Partitioning**
```python
# Coalesce to reduce output partitions for efficiency
daily_df = daily_df.coalesce(4)  # Reduce to 4 output files
daily_df.write.mode("overwrite").parquet(output_path)

# Result: 4 Parquet files (part-00000.parquet to part-00003.parquet)
```

### **4. Window Function Partitioning**
```python
# Window functions partition by household for time series features
window_spec = Window.partitionBy("LCLid").orderBy("date")

df = df.withColumn("lag_1d", lag("daily_energy_kwh", 1).over(window_spec))
# Each partition contains all records for one household
# Enables efficient time series feature creation
```

---

## 📈 **Performance & Scalability Metrics**

### **MapReduce Job Performance**
```
Phase 1 (Ingestion):     9.55 minutes, 167M records, 64 partitions
Phase 2 (Preprocessing): 41.72 minutes, 167M → 1.93M records, 20 shuffle partitions
Phase 3 (Features):      26.62 minutes, 36 features created, window operations
Phase 4 (ML Training):   35.98 minutes, 99.87% accuracy achieved
Phase 5 (Anomaly):       0.88 seconds, 1,279 anomalies detected

Total Processing Time: 114.05 minutes
Peak Memory Usage: 4GB (on 8GB system)
Data Reduction: 167M → 318K records (99.8% reduction)
```

### **Hadoop Cluster Equivalent**
```
If running on Hadoop cluster:
- 64 map tasks (one per input partition)
- 20 reduce tasks (shuffle partitions)
- HDFS block size: 128MB
- Replication factor: 3
- JobTracker coordinates task distribution
```

---

## 🔄 **MapReduce Patterns Used in Our Project**

### **1. Data Aggregation Pattern**
```python
# PySpark (MapReduce equivalent)
result = df.groupBy("LCLid", "date") \
          .agg(sum("energy_kwh"), avg("energy_kwh"), count("*"))
```

### **2. Join Pattern**
```python
# PySpark (Reduce-side join)
result = energy_df.join(tariff_df, "LCLid", "left")
```

### **3. Sorting & Windowing Pattern**
```python
# PySpark (Secondary sort pattern)
window_spec = Window.partitionBy("LCLid").orderBy("date")
result = df.withColumn("rank", rank().over(window_spec))
```

### **4. Filtering & Projection Pattern**
```python
# PySpark (Map-only pattern)
result = df.filter(col("energy_kwh") > 0) \
           .select("LCLid", "date", "energy_kwh")
```

---

## 🎯 **Key Takeaways for Project Guide**

### **MapReduce Concepts Demonstrated**
1. **Parallel Processing**: 64-way parallelism across data partitions
2. **Shuffle & Sort**: Automatic data redistribution for grouping operations
3. **Fault Tolerance**: Spark handles task failures automatically
4. **Scalability**: Linear scaling with data size and cluster size

### **Partitioning Strategy**
1. **Input Partitioning**: Based on HDFS block size (128MB)
2. **Shuffle Partitioning**: Hash-based distribution for load balancing
3. **Output Partitioning**: Coalesced for optimal file sizes
4. **Window Partitioning**: Logical partitioning for time series operations

### **Why PySpark vs Traditional Hadoop**
- **Performance**: 10x faster than MapReduce for iterative algorithms
- **Ease of Use**: DataFrame API vs raw MapReduce code
- **Optimization**: Catalyst optimizer automatically optimizes execution plans
- **Integration**: Better integration with ML libraries (MLlib)

### **Production Readiness**
- **Fault Tolerance**: Automatic task retry and speculative execution
- **Monitoring**: Web UI for job progress and performance metrics
- **Resource Management**: Dynamic allocation based on workload
- **Data Locality**: HDFS-aware scheduling for optimal performance

---

*This document demonstrates how our PySpark implementation follows Hadoop MapReduce principles while providing modern big data processing capabilities for the Smart Energy Analytics project.*