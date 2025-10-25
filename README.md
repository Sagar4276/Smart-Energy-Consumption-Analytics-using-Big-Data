# Smart Energy Consumption Analytics using Big Data

## Overview
This project demonstrates end-to-end big data analytics for smart energy consumption, addressing real-world sustainability challenges. It leverages genuine big data tools like Hadoop and Apache Spark to process large-scale IoT-generated time-series data from smart meters. The goal is to build a scalable pipeline for analyzing energy usage patterns, enabling demand forecasting and anomaly detection to optimize energy consumption, reduce costs, and support environmental goals.

## ⭐ NEW: Docker Support - Zero Local Installation Required!
Run the entire project with Spark and Hadoop in Docker containers - no manual installation needed!
See **[Quick Start with Docker](#-quick-start-with-docker-recommended)** below.

## Tech Stack
- **Data Storage**: HDFS (Hadoop Distributed File System)
- **Processing**: Apache Spark (PySpark)
- **Analytics/ML**: Spark SQL + MLlib
- **Language**: Python
- **Visualization**: Streamlit / Matplotlib
- **Environment**: Docker (Recommended) / Local Cluster

## Dataset
UK Domestic Energy Smart Meter Dataset from London Datastore
- Size: ~167 million rows
- Coverage: 5,567 London households from November 2011 to February 2014

## Architecture
1. Data Ingestion: Download and ingest raw CSVs into HDFS
2. Data Processing: Preprocessing and feature engineering with PySpark
3. Analytics & ML: Demand forecasting and anomaly detection
4. Visualization: Interactive dashboards with Streamlit

## 🐳 Quick Start with Docker (Recommended)

### Prerequisites
- Docker Desktop installed and running ([Download](https://www.docker.com/products/docker-desktop))
- At least 8GB RAM allocated to Docker
- At least 10GB free disk space

### Step 1: Setup Docker Environment
**Windows PowerShell:**
```powershell
.\docker-setup.ps1
```

**Linux/Mac:**
```bash
chmod +x docker-setup.sh run_docker_pipeline.sh
./docker-setup.sh
```

### Step 2: Run Analytics Pipeline
**Windows PowerShell:**
```powershell
.\run_docker_pipeline.ps1
```

**Linux/Mac:**
```bash
./run_docker_pipeline.sh
```

### Step 3: Access Services
- **📊 Streamlit Dashboard**: http://localhost:8501
- **⚡ Spark Master UI**: http://localhost:8080
- **⚙️ Spark Worker UI**: http://localhost:8081
- **📓 Jupyter Notebook**: http://localhost:8888

### Docker Commands
```bash
# Start containers
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose stop

# Shutdown and clean
docker-compose down -v
```

📖 **See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for detailed Docker guide**

---

## 💻 Local Setup (Alternative to Docker)

### Prerequisites
1. **Python 3.7+**: Download from https://python.org
2. **Java 8+**: Download from https://adoptium.net/
3. **PowerShell Execution**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Setup Steps

1. **Setup Environment**
   ```powershell
   .\setup.ps1
   ```
   This installs PySpark and verifies prerequisites.

2. **Dataset Status: ✅ Already Available!**
   - Dataset is already present in `data/Partitioned LCL Data/Small LCL Data/`
   - Contains 168 CSV files with ~167 million records
   - No download needed - proceed to next step

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Data Pipeline**
   Execute the complete pipeline:
   ```powershell
   .\run_pipeline.ps1
   ```
   This automatically runs all scripts in order and processes your real dataset.

5. **Launch Visualization**
   ```bash
   streamlit run visualization/app.py
   ```

## Project Structure
- `data/`: Raw and processed datasets
- `scripts/`: PySpark scripts for ETL and ML
  - `data_ingestion.py`: Load CSV files into HDFS
  - `data_preprocessing.py`: Clean and aggregate data
  - `feature_engineering.py`: Create time-series features
  - `forecasting_model.py`: Train demand forecasting model
  - `anomaly_detection.py`: Implement anomaly detection
- `models/`: Trained models and outputs
- `visualization/`: Streamlit app and plots
  - `app.py`: Interactive dashboard
- `notebooks/`: Jupyter notebooks for exploration
  - `energy_analytics.ipynb`: Complete analysis workflow
- `docs/`: Documentation and reports

## Expected Outcomes
- Functional pipeline processing 167M rows
- Forecasting MAE: <0.15 kWh
- Anomaly detection: 95% true positives
- Interactive dashboards for insights