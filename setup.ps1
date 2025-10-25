# Setup script for Smart Energy Consumption Analytics
# This script sets up the big data environment

Write-Host "Setting up Big Data environment..."
Write-Host "Note: Using local Spark mode for compatibility"
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion"
} catch {
    Write-Host "Python not found. Please install Python 3.7+ from https://python.org"
    exit 1
}

# Check if Java is installed (required for Spark)
try {
    $javaVersion = java -version 2>&1
    Write-Host "Java found: $javaVersion"
} catch {
    Write-Host "Java not found. Please install Java 8+ from https://adoptium.net/"
    exit 1
}

# Install PySpark if not already installed
Write-Host "Checking PySpark installation..."
try {
    python -c "import pyspark; print('PySpark version:', pyspark.__version__)" 2>$null
    Write-Host "PySpark is already installed"
} catch {
    Write-Host "Installing PySpark..."
    pip install pyspark==3.5.0
}

Write-Host ""
Write-Host "Environment setup complete!"
Write-Host "The pipeline will run in local Spark mode"
Write-Host "This provides the same big data processing capabilities"
Write-Host "but uses your local machine instead of Docker containers"
Write-Host ""
Write-Host "You can now run: .\run_pipeline.ps1"