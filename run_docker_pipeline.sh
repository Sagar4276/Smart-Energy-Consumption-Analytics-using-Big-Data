#!/bin/bash
# Bash script to run the complete analytics pipeline in Docker (Linux/Mac)

echo "========================================"
echo "  Energy Analytics Pipeline Runner"
echo "  Running in Docker Containers"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Docker is running
echo -e "${YELLOW}[Pre-check] Verifying Docker...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker is not running. Please start Docker.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"
echo ""

# Check if containers are running
echo -e "${YELLOW}[Pre-check] Verifying containers...${NC}"
if ! docker ps --filter "name=energy-analytics" --format "{{.Names}}" | grep -q "energy-analytics"; then
    echo -e "${RED}ERROR: energy-analytics container is not running.${NC}"
    echo -e "${YELLOW}Please run: docker-compose up -d${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All containers are running${NC}"
echo ""

# Function to run a script in the analytics container
run_pipeline_step() {
    local step_name=$1
    local script_path=$2
    local step_number=$3
    local total_steps=$4
    
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Step [$step_number/$total_steps]: $step_name${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo "Script: $script_path"
    echo ""
    
    start_time=$(date +%s)
    
    # Run the script in the container
    docker exec energy-analytics python3 "$script_path"
    
    if [ $? -eq 0 ]; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo ""
        echo -e "${GREEN}✓ $step_name completed successfully in $duration seconds${NC}"
        echo ""
        return 0
    else
        echo ""
        echo -e "${RED}✗ $step_name failed!${NC}"
        echo ""
        return 1
    fi
}

# Pipeline execution
total_steps=5
pipeline_start_time=$(date +%s)

echo -e "${CYAN}Starting Energy Analytics Pipeline...${NC}"
echo "Total Steps: $total_steps"
echo ""
sleep 2

# Step 1: Data Ingestion
run_pipeline_step "Data Ingestion" "/app/scripts/data_ingestion.py" 1 $total_steps || exit 1

# Step 2: Data Preprocessing
run_pipeline_step "Data Preprocessing" "/app/scripts/data_preprocessing.py" 2 $total_steps || exit 1

# Step 3: Feature Engineering
run_pipeline_step "Feature Engineering" "/app/scripts/feature_engineering.py" 3 $total_steps || exit 1

# Step 4: Forecasting Model
run_pipeline_step "Forecasting Model Training" "/app/scripts/forecasting_model.py" 4 $total_steps || exit 1

# Step 5: Anomaly Detection
run_pipeline_step "Anomaly Detection" "/app/scripts/anomaly_detection.py" 5 $total_steps || exit 1

# Pipeline completion summary
pipeline_end_time=$(date +%s)
total_duration=$((pipeline_end_time - pipeline_start_time))
duration_minutes=$((total_duration / 60))

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  PIPELINE COMPLETED SUCCESSFULLY! ✓${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}Summary:${NC}"
echo "  • Total Steps Completed: $total_steps/$total_steps"
echo "  • Total Duration: $total_duration seconds ($duration_minutes minutes)"
echo "  • Start Time: $(date -d @$pipeline_start_time '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r $pipeline_start_time '+%Y-%m-%d %H:%M:%S')"
echo "  • End Time: $(date -d @$pipeline_end_time '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r $pipeline_end_time '+%Y-%m-%d %H:%M:%S')"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo -e "  ${YELLOW}1. View Streamlit Dashboard: http://localhost:8501${NC}"
echo -e "  ${YELLOW}2. View Spark Master UI: http://localhost:8080${NC}"
echo -e "  ${YELLOW}3. View Jupyter Notebook: http://localhost:8888${NC}"
echo ""
echo -e "${CYAN}To view logs:${NC}"
echo "  docker logs energy-analytics"
echo "  docker logs energy-dashboard"
echo ""
