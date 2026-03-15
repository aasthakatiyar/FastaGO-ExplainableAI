"""
Download pretrained DeepGOPlus dataset and model.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, output_path: Path) -> bool:
    """
    Download file with progress bar.
    
    Args:
        url (str): URL to download from
        output_path (Path): Output file path
        
    Returns:
        bool: True if successful
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        print(f"✓ Downloaded to: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {str(e)}")
        return False


def download_deepgoplus_data():
    """
    Download DeepGOPlus pretrained dataset.
    Downloads from: http://deepgoplus.bio2vec.net/data/data.tar.gz
    """
    data_url = "http://deepgoplus.bio2vec.net/data/data.tar.gz"
    data_path = Path("data/raw/data.tar.gz")
    
    print("Starting DeepGOPlus dataset download...")
    print(f"URL: {data_url}")
    
    if data_path.exists():
        print("✓ Dataset already exists. Skipping download.")
        return True
    
    success = download_file(data_url, data_path)
    return success


def download_pretrained_model():
    """
    Download pretrained DeepGOPlus model.
    
    Note: The model file is large (2+ GB) and requires manual download from:
    http://deepgoplus.bio2vec.net/
    
    Or use a compatible pre-trained model from the official repository.
    """
    model_path = Path("models/model.h5")
    
    print("\nPreTrained Model Download")
    print("-" * 60)
    
    if model_path.exists():
        print("✓ Model already exists. Ready to use!")
        return True
    
    print("✗ Model file not found: models/model.h5")
    print("\nTo obtain the pre-trained model:")
    print("1. Visit: http://deepgoplus.bio2vec.net/")
    print("2. Download the pre-trained model (model.h5)")
    print("3. Place it in the 'models/' directory")
    print("\nAlternatively, you can:")
    print("- Train your own model using the downloaded dataset")
    print("- Use a different pre-trained model compatible with DeepGOPlus")
    print("\nThe system can still work for testing without the model,")
    print("but predictions will not be available.")
    
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("DeepGOPlus Dataset Downloader")
    print("=" * 60)
    print()
    
    data_success = download_deepgoplus_data()
    model_success = download_pretrained_model()
    
    print("\n" + "=" * 60)
    if data_success:
        print("✓ Dataset download completed successfully!")
        if model_success:
            print("✓ Model is available. System ready for predictions!")
        else:
            print("\n⚠ Model file not available.")
            print("See instructions above to obtain the pre-trained model.")
            print("System will work for testing, but predictions require the model.")
    else:
        print("✗ Dataset download failed. Please check your internet connection")
        print("and the URLs, then try again.")
    print("=" * 60)
