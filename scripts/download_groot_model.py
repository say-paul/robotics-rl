#!/usr/bin/env python3
"""
Download GR00T VLA models from HuggingFace or Isaac-GR00T repository
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def download_from_huggingface(version="n1.6", target_dir="models/groot"):
    """
    Download GR00T model from HuggingFace.

    Args:
        version: Model version (n1.5, n1.6, n1.7)
        target_dir: Target directory for downloaded models
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("Install with: pip install huggingface-hub")
        sys.exit(1)

    # Map versions to HuggingFace repo IDs
    repo_map = {
        "n1.5": "nvidia/GR00T-N1.5-3B",
        "n1.6": "nvidia/GR00T-N1.6-3B",
        "n1.7": "nvidia/GR00T-N1.7-3B",
    }

    if version not in repo_map:
        print(f"ERROR: Unknown version '{version}'")
        print(f"Available versions: {list(repo_map.keys())}")
        sys.exit(1)

    repo_id = repo_map[version]
    print(f"📥 Downloading GR00T {version.upper()} from HuggingFace: {repo_id}")

    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Download entire repository
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_path),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ Downloaded to: {local_dir}")

        # List downloaded files
        print("\n📁 Downloaded files:")
        for file in Path(local_dir).rglob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.1f} MB)")

        return local_dir

    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)


def download_from_isaac_groot(version="n1.6", target_dir="models/groot"):
    """
    Clone Isaac-GR00T repository and extract models.

    Args:
        version: Model version (n1.5, n1.6, n1.7)
        target_dir: Target directory for models
    """
    isaac_groot_url = "https://github.com/NVIDIA/Isaac-GR00T.git"
    temp_clone_dir = "/tmp/Isaac-GR00T"

    print(f"📥 Cloning Isaac-GR00T repository...")

    # Clone with Git LFS
    try:
        # Check if Git LFS is installed
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Git LFS not found. Installing...")
        print("Run: sudo apt install git-lfs && git lfs install")
        sys.exit(1)

    # Clone repo
    if Path(temp_clone_dir).exists():
        print(f"Removing existing clone at {temp_clone_dir}")
        import shutil
        shutil.rmtree(temp_clone_dir)

    # Map versions to branches/tags
    branch_map = {
        "n1.5": "n1.5-release",
        "n1.6": "n1.6-release",
        "n1.7": "main",  # Latest on main
    }

    branch = branch_map.get(version, "main")

    try:
        print(f"Cloning branch/tag: {branch}")
        subprocess.run(
            ["git", "clone", "--branch", branch, "--depth", "1", isaac_groot_url, temp_clone_dir],
            check=True
        )

        print("Pulling LFS files...")
        subprocess.run(["git", "lfs", "pull"], cwd=temp_clone_dir, check=True)

        # Copy models to target directory
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        # Find and copy ONNX models
        onnx_files = list(Path(temp_clone_dir).rglob("*.onnx"))
        if not onnx_files:
            print("⚠️  No ONNX models found in repository")
            print("   Models may need to be exported from PyTorch checkpoints")
            return None

        print(f"\n📁 Found {len(onnx_files)} ONNX model(s):")
        for onnx_file in onnx_files:
            dest = target_path / onnx_file.name
            import shutil
            shutil.copy2(onnx_file, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✅ Copied: {onnx_file.name} ({size_mb:.1f} MB)")

        return str(target_path)

    except subprocess.CalledProcessError as e:
        print(f"❌ Git clone failed: {e}")
        sys.exit(1)


def download_sonic_models(target_dir="models/sonic_wbc"):
    """Download GEAR-SONIC WBC models from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("Install with: pip install huggingface-hub")
        sys.exit(1)

    repo_id = "nvidia/GEAR-SONIC"
    print(f"Downloading GEAR-SONIC models from HuggingFace: {repo_id}")

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_path),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
        )
        print(f"Downloaded to: {local_dir}")

        onnx_files = list(Path(local_dir).rglob("*.onnx"))
        print(f"\nONNX models found: {len(onnx_files)}")
        for f in onnx_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.relative_to(local_dir)} ({size_mb:.1f} MB)")

        return local_dir

    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)


def verify_models(model_dir):
    """
    Verify downloaded models are valid ONNX files.
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        print(f"❌ Model directory does not exist: {model_dir}")
        return False

    onnx_files = list(model_path.rglob("*.onnx"))

    if not onnx_files:
        print(f"⚠️  No ONNX files found in {model_dir}")
        return False

    try:
        import onnx

        print("\n🔍 Verifying ONNX models...")
        for onnx_file in onnx_files:
            print(f"  Checking: {onnx_file.name}")
            model = onnx.load(str(onnx_file))
            onnx.checker.check_model(model)
            print(f"    ✅ Valid ONNX model")

            # Print input/output info
            print(f"    Inputs:")
            for input_tensor in model.graph.input:
                shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                print(f"      - {input_tensor.name}: {shape}")

            print(f"    Outputs:")
            for output_tensor in model.graph.output:
                shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
                print(f"      - {output_tensor.name}: {shape}")

        return True

    except ImportError:
        print("⚠️  onnx package not installed, skipping verification")
        print("   Install with: pip install onnx")
        return True
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download GR00T VLA models")
    parser.add_argument(
        "--version",
        type=str,
        default="n1.6",
        choices=["n1.5", "n1.6", "n1.7"],
        help="GR00T model version to download"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "github"],
        help="Download source"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="models/groot",
        help="Target directory for models"
    )
    parser.add_argument(
        "--sonic",
        action="store_true",
        help="Download GEAR-SONIC WBC models (encoder, decoder, planner)"
    )
    parser.add_argument(
        "--sonic-dir",
        type=str,
        default="models/sonic_wbc",
        help="Target directory for SONIC models (default: models/sonic_wbc)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded ONNX models"
    )

    args = parser.parse_args()

    # Load HF_TOKEN from .env if not already in environment
    if not os.environ.get("HF_TOKEN"):
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")
                    break

    if args.sonic:
        print("=" * 60)
        print("GEAR-SONIC Model Downloader")
        print("=" * 60)
        print(f"Target:  {args.sonic_dir}")
        print("=" * 60)
        print()

        model_dir = download_sonic_models(args.sonic_dir)
    else:
        print("=" * 60)
        print("GR00T VLA Model Downloader")
        print("=" * 60)
        print(f"Version: {args.version.upper()}")
        print(f"Source:  {args.source}")
        print(f"Target:  {args.target_dir}")
        print("=" * 60)
        print()

        if args.source == "huggingface":
            model_dir = download_from_huggingface(args.version, args.target_dir)
        else:
            model_dir = download_from_isaac_groot(args.version, args.target_dir)

    if args.verify and model_dir:
        if not verify_models(model_dir):
            print("\nModel verification failed")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nModels available at: {model_dir}")


if __name__ == "__main__":
    main()
