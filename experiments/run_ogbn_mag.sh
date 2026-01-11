#!/bin/bash

# Experiment script for running S-GraphLLM on OGBN-MAG dataset

set -e

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/output"
LOG_DIR="${PROJECT_DIR}/logs"

# Create directories
mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

echo "=========================================="
echo "S-GraphLLM Experiment: OGBN-MAG Dataset"
echo "=========================================="
echo "Project directory: ${PROJECT_DIR}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Step 1: Download OGBN-MAG dataset
echo "Step 1: Preparing OGBN-MAG dataset..."
python3 << 'EOF'
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch_geometric
    from torch_geometric.datasets import OGB
    
    print("torch_geometric is available")
    print("Note: Full OGBN-MAG dataset download requires significant disk space (>100GB)")
    print("For this experiment, we'll use a smaller sample")
except ImportError:
    print("torch_geometric not available, using sample graph instead")
EOF

# Step 2: Run S-GraphLLM
echo ""
echo "Step 2: Running S-GraphLLM with sample graph..."
cd "${PROJECT_DIR}"
python3 -m src.main \
    --config configs/model_config.yaml \
    --query "What are the main research topics and their relationships in this academic network?" \
    --output "${OUTPUT_DIR}/result_sample.json" \
    --log-level INFO

# Step 3: Run tests
echo ""
echo "Step 3: Running tests..."
python3 -m pytest tests/ -v --tb=short

# Step 4: Generate report
echo ""
echo "Step 4: Generating experiment report..."
python3 << 'EOF'
import json
import os
from pathlib import Path

output_dir = Path("output")
result_file = output_dir / "result_sample.json"

if result_file.exists():
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    print("\n========== EXPERIMENT RESULTS ==========")
    print(f"Number of reasoning traces: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"\nTrace {i+1}:")
        print(f"  Query: {result.get('query', 'N/A')}")
        print(f"  Selected partitions: {result.get('selected_partitions', [])}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Reasoning steps: {len(result.get('reasoning_steps', []))}")
else:
    print("No results file found")
EOF

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
