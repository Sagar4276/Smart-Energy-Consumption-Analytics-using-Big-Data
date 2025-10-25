#!/bin/bash
# Bash script to set up and start the Docker environment (Linux/Mac)

echo "========================================"
echo "  Energy Analytics Docker Setup"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Check if Docker is installed
echo -e "${YELLOW}Step 1: Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed.${NC}"
    echo -e "${YELLOW}Please install Docker from: https://www.docker.com/products/docker-desktop${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is installed${NC}"
docker --version
echo ""

# Check if Docker is running
echo -e "${YELLOW}Step 2: Checking if Docker is running...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker is not running.${NC}"
    echo -e "${YELLOW}Please start Docker and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"
echo ""

# Check if docker-compose is available
echo -e "${YELLOW}Step 3: Checking Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}ERROR: Docker Compose is not available.${NC}"
    echo -e "${YELLOW}Please install Docker Compose.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose is available${NC}"
docker-compose --version
echo ""

# Check if data directory exists
echo -e "${YELLOW}Step 4: Checking data directory...${NC}"
if [ ! -d "data" ]; then
    echo -e "${RED}ERROR: data directory not found.${NC}"
    echo -e "${YELLOW}Please ensure the data folder exists with your dataset.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Data directory found${NC}"
echo ""

# Create necessary directories
echo -e "${YELLOW}Step 5: Creating necessary directories...${NC}"
directories=("models" "notebooks" "data/processed")
for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GRAY}  Created: $dir${NC}"
    else
        echo -e "${GRAY}  Exists: $dir${NC}"
    fi
done
echo -e "${GREEN}✓ All directories ready${NC}"
echo ""

# Stop and remove existing containers
echo -e "${YELLOW}Step 6: Cleaning up existing containers...${NC}"
docker-compose down -v > /dev/null 2>&1
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

# Build and start containers
echo -e "${YELLOW}Step 7: Building and starting containers...${NC}"
echo -e "${GRAY}This may take several minutes on first run...${NC}"
echo ""
docker-compose up -d --build
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to start containers.${NC}"
    echo -e "${YELLOW}Check the error messages above for details.${NC}"
    exit 1
fi
echo ""
echo -e "${GREEN}✓ Containers built and started${NC}"
echo ""

# Wait for containers to be healthy
echo -e "${YELLOW}Step 8: Waiting for containers to be ready...${NC}"
sleep 10
docker-compose ps
echo ""

# Check container status
echo -e "${YELLOW}Step 9: Verifying container health...${NC}"
containers=("energy-spark-master" "energy-spark-worker" "energy-analytics" "energy-dashboard" "energy-jupyter")
all_healthy=true
for container in "${containers[@]}"; do
    status=$(docker ps --filter "name=$container" --format "{{.Status}}")
    if [[ $status == *"Up"* ]]; then
        echo -e "${GREEN}  ✓ $container is running${NC}"
    else
        echo -e "${RED}  ✗ $container is not running properly${NC}"
        all_healthy=false
    fi
done
echo ""

if [ "$all_healthy" = false ]; then
    echo -e "${YELLOW}WARNING: Some containers are not running properly.${NC}"
    echo -e "${GRAY}Check logs with: docker-compose logs${NC}"
    echo ""
fi

# Display success message and next steps
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SETUP COMPLETED SUCCESSFULLY! ✓${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}Your containers are now running:${NC}"
echo ""
echo "  🖥️  Spark Master UI:    http://localhost:8080"
echo "  ⚙️  Spark Worker UI:    http://localhost:8081"
echo "  📊 Streamlit Dashboard: http://localhost:8501"
echo "  📓 Jupyter Notebook:    http://localhost:8888"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo ""
echo -e "${YELLOW}  1. Run the analytics pipeline:${NC}"
echo "     ./run_docker_pipeline.sh"
echo ""
echo -e "${YELLOW}  2. Or run individual steps manually:${NC}"
echo "     docker exec energy-analytics python3 /app/scripts/data_ingestion.py"
echo "     docker exec energy-analytics python3 /app/scripts/data_preprocessing.py"
echo "     docker exec energy-analytics python3 /app/scripts/feature_engineering.py"
echo "     docker exec energy-analytics python3 /app/scripts/forecasting_model.py"
echo "     docker exec energy-analytics python3 /app/scripts/anomaly_detection.py"
echo ""
echo -e "${CYAN}Useful Commands:${NC}"
echo -e "${GRAY}  • View all containers:     docker-compose ps${NC}"
echo -e "${GRAY}  • View logs:               docker-compose logs -f${NC}"
echo -e "${GRAY}  • Stop containers:         docker-compose stop${NC}"
echo -e "${GRAY}  • Start containers:        docker-compose start${NC}"
echo -e "${GRAY}  • Restart containers:      docker-compose restart${NC}"
echo -e "${GRAY}  • Shutdown and clean:      docker-compose down -v${NC}"
echo -e "${GRAY}  • Access container shell:  docker exec -it energy-analytics bash${NC}"
echo ""
