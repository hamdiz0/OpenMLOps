#!/usr/bin/env python3
"""
Data Initialization Script for CIFAR-10 Dataset

This script downloads the CIFAR-10 dataset and prepares it for DVC versioning.
It saves the data as numpy arrays and pushes them to the MinIO remote storage.

Note: DVC is used WITHOUT Git integration (--no-scm mode).
"""

import os
import sys
import subprocess
import numpy as np
from pathlib import Path


def download_cifar10():
    """Download CIFAR-10 dataset using TensorFlow/Keras."""
    print("=" * 60)
    print("CIFAR-10 Data Initialization")
    print("=" * 60)

    # Import TensorFlow here to avoid import errors if not installed
    try:
        from tensorflow.keras.datasets import cifar10
    except ImportError:
        print("Error: TensorFlow not installed. Please install requirements first.")
        sys.exit(1)

    data_dir = Path("/app/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Downloading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(f"  - Training samples: {x_train.shape[0]}")
    print(f"  - Test samples: {x_test.shape[0]}")
    print(f"  - Image shape: {x_train.shape[1:]}")

    print("\n[2/4] Saving data to numpy files...")
    np.save(data_dir / "x_train.npy", x_train)
    np.save(data_dir / "y_train.npy", y_train)
    np.save(data_dir / "x_test.npy", x_test)
    np.save(data_dir / "y_test.npy", y_test)

    # Save class names for reference
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    np.save(data_dir / "class_names.npy", np.array(class_names))

    print("  - Saved: x_train.npy, y_train.npy")
    print("  - Saved: x_test.npy, y_test.npy")
    print("  - Saved: class_names.npy")

    return data_dir


def setup_dvc(data_dir: Path):
    """Add data to DVC tracking and push to remote (without Git)."""
    print("\n[3/4] Adding data to DVC tracking...")

    os.chdir("/app")

    # Initialize DVC without Git integration if not already done
    if not Path("/app/.dvc").exists():
        print("  - Initializing DVC (no-scm mode)...")
        subprocess.run(["dvc", "init", "--no-scm"], check=True)

    # Add data directory to DVC
    try:
        result = subprocess.run(["dvc", "add", "data"], capture_output=True, text=True)
        if result.returncode == 0:
            print("  - Data added to DVC tracking")
        else:
            print(f"  - DVC add output: {result.stdout}")
            if result.stderr:
                print(f"  - DVC add stderr: {result.stderr}")
    except Exception as e:
        print(f"  - Warning: {e}")

    print("\n[4/4] Pushing data to MinIO remote...")
    try:
        result = subprocess.run(["dvc", "push"], capture_output=True, text=True)
        if result.returncode == 0:
            print("  - Data pushed to MinIO successfully!")
        else:
            print(f"  - DVC push output: {result.stdout}")
            if result.stderr:
                print(f"  - DVC push stderr: {result.stderr}")
    except Exception as e:
        print(f"  - Warning during push: {e}")


def verify_data():
    """Verify the data was saved correctly."""
    print("\n" + "=" * 60)
    print("Data Verification")
    print("=" * 60)

    data_dir = Path("/app/data")

    files = [
        "x_train.npy",
        "y_train.npy",
        "x_test.npy",
        "y_test.npy",
        "class_names.npy",
    ]

    for f in files:
        filepath = data_dir / f
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  [OK] {f}: {size_mb:.2f} MB")
        else:
            print(f"  [MISSING] {f}: NOT FOUND")

    # Load and verify shapes
    print("\nData shapes:")
    x_train = np.load(data_dir / "x_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    x_test = np.load(data_dir / "x_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    print(f"  - x_train: {x_train.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - x_test: {x_test.shape}")
    print(f"  - y_test: {y_test.shape}")

    # Check if data.dvc file was created
    dvc_file = Path("/app/data.dvc")
    if dvc_file.exists():
        print(f"\n  [OK] DVC tracking file created: data.dvc")
    else:
        print(f"\n  [INFO] No data.dvc file (data tracked locally)")


def main():
    """Main function to orchestrate data initialization."""
    try:
        data_dir = download_cifar10()
        setup_dvc(data_dir)
        verify_data()

        print("\n" + "=" * 60)
        print("Data initialization complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run training pipeline: python run_training.py")
        print("  2. Run monitoring pipeline: python run_monitoring.py")
        print()

    except Exception as e:
        print(f"\nError during initialization: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
