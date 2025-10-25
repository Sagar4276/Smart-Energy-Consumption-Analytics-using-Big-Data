#!/usr/bin/env pwsh
# PowerShell script to run the complete analytics pipeline in Docker

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Energy Analytics Pipeline Runner" -ForegroundColor Cyan
Write-Host "  Running in Docker Containers" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "[Pre-check] Verifying Docker..." -ForegroundColor Yellow
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) 
{
    Write-Host "ERROR: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}
Write-Host "✓ Docker is running" -ForegroundColor Green
Write-Host ""

# Check if containers are running
Write-Host "[Pre-check] Verifying containers..." -ForegroundColor Yellow
$analyticsRunning = docker ps --filter "name=energy-analytics" --format "{{.Names}}" 2>$null
if (-not $analyticsRunning) {
    Write-Host "ERROR: energy-analytics container is not running." -ForegroundColor Red
    Write-Host "Please run: docker-compose up -d" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ All containers are running" -ForegroundColor Green
Write-Host ""

# Function to run a script in the analytics container
function Run-PipelineStep {
    param(
        [string]$StepName,
        [string]$ScriptPath,
        [int]$StepNumber,
        [int]$TotalSteps
    )
    
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Step [$StepNumber/$TotalSteps]: $StepName" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Script: $ScriptPath" -ForegroundColor Gray
    Write-Host ""
    
    $startTime = Get-Date
    
    # Run the script in the container
    docker exec energy-analytics python3 $ScriptPath
    
    if ($LASTEXITCODE -eq 0) 
    {
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        Write-Host ""
        Write-Host "✓ $StepName completed successfully in $([math]::Round($duration, 2)) seconds" -ForegroundColor Green
        Write-Host ""
        return $true
    } 
    else {
        Write-Host ""
        Write-Host "✗ $StepName failed!" -ForegroundColor Red
        Write-Host ""
        return $false
    }
}

# Pipeline execution
$totalSteps = 5
$pipelineStartTime = Get-Date

Write-Host "Starting Energy Analytics Pipeline..." -ForegroundColor Cyan
Write-Host "Total Steps: $totalSteps" -ForegroundColor Gray
Write-Host ""
Start-Sleep -Seconds 2

# Step 1: Data Ingestion
$success = Run-PipelineStep -StepName "Data Ingestion" `
                            -ScriptPath "/app/scripts/data_ingestion.py" `
                            -StepNumber 1 `
                            -TotalSteps $totalSteps
if (-not $success) { exit 1 }

# Step 2: Data Preprocessing
$success = Run-PipelineStep -StepName "Data Preprocessing" `
                            -ScriptPath "/app/scripts/data_preprocessing.py" `
                            -StepNumber 2 `
                            -TotalSteps $totalSteps
if (-not $success) { exit 1 }

# Step 3: Feature Engineering
$success = Run-PipelineStep -StepName "Feature Engineering" `
                            -ScriptPath "/app/scripts/feature_engineering.py" `
                            -StepNumber 3 `
                            -TotalSteps $totalSteps
if (-not $success) { exit 1 }

# Step 4: Forecasting Model
$success = Run-PipelineStep -StepName "Forecasting Model Training" `
                            -ScriptPath "/app/scripts/forecasting_model.py" `
                            -StepNumber 4 `
                            -TotalSteps $totalSteps
if (-not $success) { exit 1 }

# Step 5: Anomaly Detection
$success = Run-PipelineStep -StepName "Anomaly Detection" `
                            -ScriptPath "/app/scripts/anomaly_detection.py" `
                            -StepNumber 5 `
                            -TotalSteps $totalSteps
if (-not $success) { exit 1 }

# Pipeline completion summary
$pipelineEndTime = Get-Date
$totalDuration = ($pipelineEndTime - $pipelineStartTime).TotalSeconds

Write-Host "========================================" -ForegroundColor Green
Write-Host "  PIPELINE COMPLETED SUCCESSFULLY! ✓" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  • Total Steps Completed: $totalSteps/$totalSteps" -ForegroundColor White
Write-Host "  • Total Duration: $([math]::Round($totalDuration, 2)) seconds ($([math]::Round($totalDuration/60, 2)) minutes)" -ForegroundColor White
Write-Host "  • Start Time: $($pipelineStartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White
Write-Host "  • End Time: $($pipelineEndTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. View Streamlit Dashboard: http://localhost:8501" -ForegroundColor Yellow
Write-Host "  2. View Spark Master UI: http://localhost:8080" -ForegroundColor Yellow
Write-Host "  3. View Jupyter Notebook: http://localhost:8888" -ForegroundColor Yellow
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Cyan
Write-Host "  docker logs energy-analytics" -ForegroundColor Gray
Write-Host "  docker logs energy-dashboard" -ForegroundColor Gray
Write-Host ""
