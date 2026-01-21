#!/bin/bash

# GBERT-Lite - Automated Setup and Run Script
# This script automates the entire setup and execution process

set -e  # Exit on error

echo "=========================================="
echo "GBERT-Lite Setup and Execution Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
check_python() {
    print_info "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_info "Found Python $PYTHON_VERSION"
        
        # Check if version is 3.10+
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
            print_info "Python version is compatible ✓"
        else
            print_error "Python 3.10+ is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.10+"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_info "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_info "Virtual environment created ✓"
    else
        print_info "Virtual environment already exists ✓"
    fi
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_info "Virtual environment activated ✓"
    else
        print_error "Virtual environment not found!"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_info "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_info "Dependencies installed ✓"
}

# Create directory structure
create_directories() {
    print_info "Creating directory structure..."
    mkdir -p data checkpoints results plots logs cache
    print_info "Directories created ✓"
}

# Check for dataset
check_dataset() {
    print_info "Checking for dataset..."
    if [ -f "data/Fake.csv" ] && [ -f "data/True.csv" ]; then
        print_info "Dataset found ✓"
        return 0
    else
        print_warning "Dataset not found!"
        echo ""
        echo "Please download the dataset:"
        echo "1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        echo "2. Download Fake.csv and True.csv"
        echo "3. Place them in the ./data/ directory"
        echo ""
        echo "Alternatively, run: python scripts/download_data.py"
        echo ""
        read -p "Do you want to continue without the dataset? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        return 1
    fi
}

# Run tests
run_tests() {
    print_info "Running tests..."
    pytest tests/ -v
    print_info "Tests passed ✓"
}

# Train model
train_model() {
    MODEL=$1
    print_info "Training $MODEL model..."
    python main.py --mode train --model "$MODEL"
    print_info "Training complete ✓"
}

# Evaluate model
evaluate_model() {
    MODEL=$1
    CHECKPOINT=$2
    print_info "Evaluating $MODEL model..."
    python main.py --mode evaluate --model "$MODEL" --checkpoint "$CHECKPOINT"
    print_info "Evaluation complete ✓"
}

# Run benchmark
run_benchmark() {
    print_info "Running comprehensive benchmark..."
    print_warning "This will take several hours!"
    python main.py --mode benchmark
    print_info "Benchmark complete ✓"
}

# Main menu
show_menu() {
    echo ""
    echo "=========================================="
    echo "GBERT-Lite - Main Menu"
    echo "=========================================="
    echo "1. Setup environment (first time setup)"
    echo "2. Train GBERT-Lite"
    echo "3. Train baseline model"
    echo "4. Evaluate model"
    echo "5. Run full benchmark"
    echo "6. Run tests"
    echo "7. Download dataset"
    echo "8. Exit"
    echo "=========================================="
    read -p "Enter your choice [1-8]: " choice
    
    case $choice in
        1)
            check_python
            create_venv
            activate_venv
            install_dependencies
            create_directories
            check_dataset
            print_info "Setup complete! ✓"
            show_menu
            ;;
        2)
            activate_venv
            check_dataset
            train_model "gbert_lite"
            show_menu
            ;;
        3)
            echo ""
            echo "Available baseline models:"
            echo "1. BERT"
            echo "2. RoBERTa"
            echo "3. DeBERTa"
            echo "4. GBERT"
            echo "5. BERT+GNN"
            read -p "Select model [1-5]: " model_choice
            
            case $model_choice in
                1) MODEL="bert";;
                2) MODEL="roberta";;
                3) MODEL="deberta";;
                4) MODEL="gbert";;
                5) MODEL="bert_gnn";;
                *) print_error "Invalid choice"; show_menu;;
            esac
            
            activate_venv
            check_dataset
            train_model "$MODEL"
            show_menu
            ;;
        4)
            read -p "Enter model name (gbert_lite/bert/roberta/etc.): " model_name
            read -p "Enter checkpoint path: " checkpoint_path
            activate_venv
            evaluate_model "$model_name" "$checkpoint_path"
            show_menu
            ;;
        5)
            activate_venv
            check_dataset
            run_benchmark
            show_menu
            ;;
        6)
            activate_venv
            run_tests
            show_menu
            ;;
        7)
            activate_venv
            python scripts/download_data.py --dataset fake_real_news
            show_menu
            ;;
        8)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            show_menu
            ;;
    esac
}

# Quick start mode
quick_start() {
    print_info "Starting GBERT-Lite Quick Start..."
    
    check_python
    create_venv
    activate_venv
    install_dependencies
    create_directories
    
    if check_dataset; then
        print_info "Running quick training..."
        train_model "gbert_lite"
        print_info "Quick start complete! ✓"
    else
        print_warning "Cannot proceed without dataset"
    fi
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments, show menu
    show_menu
else
    case $1 in
        --quick-start)
            quick_start
            ;;
        --setup)
            check_python
            create_venv
            activate_venv
            install_dependencies
            create_directories
            ;;
        --train)
            activate_venv
            train_model "${2:-gbert_lite}"
            ;;
        --benchmark)
            activate_venv
            run_benchmark
            ;;
        --test)
            activate_venv
            run_tests
            ;;
        --help)
            echo "Usage: ./run.sh [OPTION]"
            echo ""
            echo "Options:"
            echo "  (no args)        Show interactive menu"
            echo "  --quick-start    Quick setup and training"
            echo "  --setup          Setup environment only"
            echo "  --train [MODEL]  Train specific model"
            echo "  --benchmark      Run full benchmark"
            echo "  --test           Run tests"
            echo "  --help           Show this help message"
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
fi