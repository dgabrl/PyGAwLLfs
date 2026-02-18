#!/bin/bash

# =================================================================
# GAwLL - Execution Orchestrator
# Description: Automates environment setup and triggers the 
#              Genetic Algorithm with Linkage Learning pipeline.
# =================================================================

# Color definitions for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}>>> Starting PyGAwLL2 Environment Setup...${NC}"

# 1. Virtual Environment Management
VENV_NAME="venv_gawll"

if [ ! -d "$VENV_NAME" ]; then
    echo -e "${GREEN}Creating new Python Virtual Environment...${NC}"
    python3 -m venv "$VENV_NAME"
else
    echo -e "${BLUE}Existing Virtual Environment detected.${NC}"
fi

# 2. Environment Activation and Dependency Install
source "$VENV_NAME"/bin/activate
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}Installing/Updating project dependencies...${NC}"
    pip install -r requirements.txt
else
    echo -e "${RED}CRITICAL ERROR: requirements.txt not found!${NC}"
    exit 1
fi

# 3. Interactive Experiment Configuration
echo -e "\n${BLUE}--- Experiment Configuration ---${NC}"
read -p "Enter Dataset names (space separated, e.g., boson covidx): " DATASETS
read -p "Enter Model names (e.g., dt knn): " MODELS
read -p "Enter Number of Runs (e.g., 10): " N_RUNS

# 4. Execution
echo -e "${GREEN}>>> Launching GAwLL Pipeline...${NC}"
python3 src/main.py --datasets $DATASETS --models $MODELS --n_runs $N_RUNS --seed 42 2>> log_errors.log

# 6. Post-Execution Cleanup
echo -e "${BLUE}>>> Process completed. Results are available in /results.${NC}"
deactivate