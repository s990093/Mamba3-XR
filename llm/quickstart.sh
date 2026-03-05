#!/bin/bash
# Quick start script for Mamba-3 LLM experiments

echo "=========================================="
echo "Mamba-3 LLM - Quick Start"
echo "=========================================="
echo ""

# Activate virtual environment
source ../.venv/bin/activate

# Check if in correct directory
if [ ! -f "mamba3_lm.py" ]; then
    echo "❌ Error: Please run this script from the llm/ directory"
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Parse command
case "$1" in
    test)
        echo "🧪 Testing Mamba-3 LLM implementation..."
        python mamba3_lm.py
        ;;
    
    shakespeare)
        echo "🎭 Training on Shakespeare dataset..."
        python train_shakespeare.py
        ;;
    
    *)
        echo "Usage: ./quickstart.sh [command]"
        echo ""
        echo "Commands:"
        echo "  test        - Test the model implementation"
        echo "  shakespeare - Train on Shakespeare dataset"
        echo ""
        echo "Examples:"
        echo "  ./quickstart.sh test"
        echo "  ./quickstart.sh shakespeare"
        ;;
esac
