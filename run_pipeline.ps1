# Run Energy Analytics Pipeline
# This script executes the complete data pipeline locally

Write-Host "Starting Smart Energy Analytics Pipeline..."
Write-Host "Running in local Spark mode"
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Using Python: $pythonVersion"
} catch {
    Write-Host "Python not found. Please install Python 3.7+"
    exit 1
}

# Execute pipeline steps
$scripts = @(
    "data_ingestion.py",
    "data_preprocessing.py", 
    "feature_engineering.py",
    "forecasting_model.py",
    "anomaly_detection.py"
)

foreach ($script in $scripts) {
    Write-Host "Running $script..."
    $scriptPath = ".\scripts\$script"
    
    if (Test-Path $scriptPath) {
        $result = python $scriptPath 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error in $script : $result"
            exit 1
        }
        Write-Host "$script completed successfully"
        Write-Host ""
    } else {
        Write-Host "Script not found: $scriptPath"
        exit 1
    }
}

Write-Host "Pipeline execution complete!"
Write-Host ""
Write-Host "Results saved to data/processed/ and models/ folders"
Write-Host ""
Write-Host "To view results, run: streamlit run visualization/app.py"
Write-Host "Or explore the notebook: jupyter notebook notebooks/energy_analytics.ipynb"