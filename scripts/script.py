"""
Script to download datasets
"""

import os
import sys
from pathlib import Path
import argparse


def download_fake_real_news():
    """Download Fake and Real News Dataset from Kaggle"""
    
    print("Downloading Fake and Real News Dataset from Kaggle...")
    print("Note: You need to have Kaggle API configured")
    print("See: https://github.com/Kaggle/kaggle-api#api-credentials")
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("\nKaggle API not installed!")
        print("Install it using: pip install kaggle")
        sys.exit(1)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download dataset
    try:
        os.system("kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p ./data")
        
        # Unzip
        import zipfile
        zip_path = data_dir / "fake-and-real-news-dataset.zip"
        
        if zip_path.exists():
            print("\nExtracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove zip file
            zip_path.unlink()
            
            print("\n✓ Dataset downloaded successfully!")
            print(f"Files saved in: {data_dir.absolute()}")
        else:
            print("\n✗ Download failed!")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("2. Download the dataset")
        print("3. Extract Fake.csv and True.csv to ./data/")


def download_fakenewsnet():
    """Download FakeNewsNet dataset"""
    
    print("Downloading FakeNewsNet dataset...")
    print("Note: This dataset requires manual download")
    
    print("\nInstructions:")
    print("1. Visit: https://github.com/KaiDMML/FakeNewsNet")
    print("2. Follow their instructions to download the dataset")
    print("3. Place the downloaded data in: ./data/fakenewsnet/")
    print("\nFor this implementation, we'll use the Fake and Real News Dataset by default.")


def setup_directories():
    """Create necessary directories"""
    
    directories = [
        'data',
        'checkpoints',
        'results',
        'plots',
        'logs',
        'cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for GBERT-Lite")
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['fake_real_news', 'fakenewsnet', 'all'],
        default='fake_real_news',
        help='Dataset to download'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("GBERT-Lite Dataset Download Script")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Download datasets
    if args.dataset in ['fake_real_news', 'all']:
        download_fake_real_news()
    
    if args.dataset in ['fakenewsnet', 'all']:
        download_fakenewsnet()
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)