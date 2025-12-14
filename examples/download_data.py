"""Script to download News Category Dataset from Kaggle."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def download_kaggle_dataset():
    """Download News Category Dataset from Kaggle using Kaggle API."""
    
    print("=" * 60)
    print("News Category Dataset Downloader")
    print("=" * 60)
    print()
    
    # Check if kaggle credentials exist
    kaggle_config = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_config):
        print("ERROR: Kaggle API credentials not found!")
        print()
        print("To download the dataset, you need to:")
        print("1. Sign in to Kaggle (https://www.kaggle.com)")
        print("2. Go to Account settings")
        print("3. Click 'Create New API Token'")
        print("4. Place the downloaded kaggle.json file in ~/.kaggle/")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        print()
        return False
    
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading dataset from Kaggle...")
    print("Dataset: rmisra/news-category-dataset")
    print()
    
    try:
        import kaggle
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'rmisra/news-category-dataset',
            path=data_dir,
            unzip=True
        )
        
        print()
        print("=" * 60)
        print("Dataset downloaded successfully!")
        print(f"Location: {data_dir}")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {str(e)}")
        print()
        print("Alternative method:")
        print("1. Visit: https://www.kaggle.com/datasets/rmisra/news-category-dataset")
        print("2. Download the dataset manually")
        print(f"3. Extract to: {data_dir}")
        return False


if __name__ == "__main__":
    success = download_kaggle_dataset()
    sys.exit(0 if success else 1)
