"""
Extract DeepGOPlus dataset from downloaded archive.
"""

import tarfile
import zipfile
import shutil
from pathlib import Path


def extract_tar_gz(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract tar.gz archive.
    
    Args:
        archive_path (Path): Path to tar.gz file
        extract_to (Path): Directory to extract to
        
    Returns:
        bool: True if successful
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=extract_to, filter='data')
        
        print(f"✓ Extracted {archive_path.name} to {extract_to}")
        return True
        
    except Exception as e:
        print(f"✗ Extraction failed: {str(e)}")
        return False


def reorganize_extracted_files() -> bool:
    """
    Reorganize extracted files from nested structure.
    The archive extracts to data/raw/data/, but we need files in data/raw/
    
    Returns:
        bool: True if successful
    """
    try:
        extract_dir = Path("data/raw/data")
        target_dir = Path("data/raw")
        
        if not extract_dir.exists():
            print("✗ Extracted directory not found. Extraction may have failed.")
            return False
        
        # Move files from data/raw/data/* to data/raw/*
        moved = 0
        for item in extract_dir.iterdir():
            if item.is_dir():
                # Move directories (like diamond_db)
                target = target_dir / item.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(item), str(target))
                print(f"  → Moved directory: {item.name}")
                moved += 1
            else:
                # Move files
                target = target_dir / item.name
                if target.exists():
                    target.unlink()
                shutil.move(str(item), str(target))
                if item.name not in ['data.tar.gz']:  # Don't log the archive itself
                    print(f"  → Moved file: {item.name}")
                moved += 1
        
        # Remove empty extracted directory
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        
        print(f"✓ Reorganized {moved} items")
        return True
        
    except Exception as e:
        print(f"✗ Reorganization failed: {str(e)}")
        return False


def verify_extracted_files() -> bool:
    """
    Verify that all required files are present.
    
    Returns:
        bool: True if all critical files exist
    """
    # Critical files required for predictions
    required_files = [
        Path("data/raw/train_data.pkl"),
        Path("data/raw/test_data.pkl"),
        Path("data/raw/terms.pkl"),
        Path("data/raw/go.obo"),
        Path("data/raw/model.h5"),
    ]
    
    # Optional files
    optional_files = [
        Path("data/raw/train_data.dmnd"),
        Path("data/raw/test_data.fa"),
        Path("data/raw/train_data.fa"),
    ]
    
    missing_critical = []
    
    for file in required_files:
        if not file.exists():
            missing_critical.append(str(file))
    
    if missing_critical:
        print("✗ Missing critical files:")
        for item in missing_critical:
            print(f"  - {item}")
        return False
    
    # Create diamond_db if it doesn't exist
    diamond_db = Path("data/raw/diamond_db")
    if not diamond_db.exists():
        diamond_db.mkdir(parents=True, exist_ok=True)
        print("  Created diamond_db directory")
    
    # Check optional files
    missing_optional = []
    for file in optional_files:
        if not file.exists():
            missing_optional.append(file.name)
    
    if missing_optional:
        print(f"⚠ Optional files not found: {', '.join(missing_optional)}")
    
    print("✓ All critical files present!")
    return True


def copy_model_to_models() -> bool:
    """
    Copy the model.h5 from data/raw to models directory.
    
    Returns:
        bool: True if successful
    """
    src_model = Path("data/raw/model.h5")
    dst_model = Path("models/model.h5")
    
    if not src_model.exists():
        print(f"! Model not found in data/raw: {src_model}")
        return False
    
    try:
        dst_model.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if destination already exists
        if dst_model.exists():
            print(f"  Model already exists in {dst_model}")
            return True
        
        shutil.copy2(src_model, dst_model)
        file_size_mb = src_model.stat().st_size / 1024 / 1024
        print(f"✓ Copied model.h5 to: {dst_model}")
        print(f"  File size: {file_size_mb:.1f} MB")
        return True
        
    except Exception as e:
        print(f"⚠ Failed to copy model: {str(e)}")
        return False


def extract_deepgoplus_data():
    """
    Extract DeepGOPlus dataset from archive and reorganize files.
    """
    archive_path = Path("data/raw/data.tar.gz")
    extract_to = Path("data/raw")
    
    if not archive_path.exists():
        print(f"✗ Archive not found: {archive_path}")
        print("Please run download_data.py first")
        return False
    
    print("Extracting DeepGOPlus dataset...")
    success = extract_tar_gz(archive_path, extract_to)
    
    if success:
        print("\nReorganizing extracted files...")
        success = reorganize_extracted_files()
    
    if success:
        print("\nVerifying extracted files...")
        success = verify_extracted_files()
        
        if success:
            print("\nCopying model to models directory...")
            copy_model_to_models()
    
    return success


if __name__ == "__main__":
    print("=" * 60)
    print("DeepGOPlus Data Extractor")
    print("=" * 60)
    print()
    
    success = extract_deepgoplus_data()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Extraction completed successfully!")
        print("✓ All data files verified!")
        print("✓ Model copied to models/ directory!")
        print("\nNext step: Run predictions or web app")
    else:
        print("✗ Extraction failed.")
        print("Please check that data/raw/data.tar.gz exists and is valid.")
    print("=" * 60)
