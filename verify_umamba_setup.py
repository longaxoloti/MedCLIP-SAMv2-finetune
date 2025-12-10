"""
Verify U-Mamba integration and setup for MedCLIP-SAMv2
Run this script before starting training to check all dependencies
"""

import sys
import os
from pathlib import Path
import importlib
import subprocess


class Colors:
    """Terminal colors"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_status(message, status="info"):
    """Print colored status message"""
    if status == "success":
        print(f"{Colors.GREEN}✓{Colors.ENDC} {message}")
    elif status == "error":
        print(f"{Colors.RED}✗{Colors.ENDC} {message}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠{Colors.ENDC} {message}")
    elif status == "info":
        print(f"{Colors.BLUE}ℹ{Colors.ENDC} {message}")
    else:
        print(message)


def print_header(message):
    """Print section header"""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{message}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_status(f"Python version: {version_str}", "info")
    
    if version.major == 3 and version.minor >= 8:
        print_status("Python version OK", "success")
        return True
    else:
        print_status("Python 3.8+ required", "error")
        return False


def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        
        # Get version
        version = None
        for attr in ['__version__', 'VERSION', 'version']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if callable(version):
                    version = version()
                break
        
        version_str = str(version) if version else "unknown"
        
        # Check minimum version
        if min_version and version:
            from packaging import version as pkg_version
            if pkg_version.parse(str(version)) < pkg_version.parse(min_version):
                print_status(f"{package_name} version {version_str} < {min_version}", "warning")
                return False
        
        print_status(f"{package_name} {version_str}", "success")
        return True
        
    except ImportError:
        print_status(f"{package_name} not installed", "error")
        return False


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            num_gpus = torch.cuda.device_count()
            
            print_status(f"CUDA {cuda_version} available", "success")
            print_status(f"Number of GPUs: {num_gpus}", "info")
            
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print_status(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)", "info")
            
            return True
        else:
            print_status("CUDA not available", "warning")
            return False
            
    except ImportError:
        print_status("PyTorch not installed", "error")
        return False


def check_mamba_installation():
    """Check Mamba-SSM installation"""
    print_header("Checking Mamba-SSM Installation")
    
    # Check mamba_ssm package
    mamba_ok = check_package("mamba-ssm", "mamba_ssm")
    
    # Check causal_conv1d
    causal_ok = check_package("causal-conv1d", "causal_conv1d")
    
    # Try to import Mamba
    if mamba_ok:
        try:
            from mamba_ssm import Mamba
            test_mamba = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
            print_status("Mamba module can be instantiated", "success")
        except Exception as e:
            print_status(f"Mamba instantiation failed: {e}", "error")
            return False
    
    return mamba_ok and causal_ok


def check_nnunet_installation():
    """Check nnUNet installation"""
    print_header("Checking nnUNet Installation")
    
    # Check nnunetv2 package
    nnunet_ok = check_package("nnunetv2")
    
    # Check environment variables
    env_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    env_ok = True
    
    for var in env_vars:
        if var in os.environ:
            path = os.environ[var]
            print_status(f"{var}: {path}", "success")
        else:
            print_status(f"{var} not set", "warning")
            env_ok = False
    
    return nnunet_ok and env_ok


def check_umamba_integration():
    """Check U-Mamba networks and trainers"""
    print_header("Checking U-Mamba Integration")
    
    base_dir = Path(__file__).parent
    weak_seg_dir = base_dir / "weak_segmentation"
    
    if not weak_seg_dir.exists():
        print_status("weak_segmentation directory not found", "error")
        return False
    
    # Check networks
    nets_dir = weak_seg_dir / "nnunetv2" / "nets"
    required_nets = [
        "UMambaBot_2d.py",
        "UMambaBot_3d.py",
        "UMambaEnc_2d.py",
        "UMambaEnc_3d.py"
    ]
    
    print_status("Checking U-Mamba networks:", "info")
    nets_ok = True
    for net_file in required_nets:
        net_path = nets_dir / net_file
        if net_path.exists():
            print_status(f"  {net_file}", "success")
        else:
            print_status(f"  {net_file} missing", "error")
            nets_ok = False
    
    # Check trainers
    trainer_dir = weak_seg_dir / "nnunetv2" / "training" / "nnUNetTrainer"
    required_trainers = [
        "nnUNetTrainerUMambaBot.py",
        "nnUNetTrainerUMambaEnc.py",
        "nnUNetTrainerUMambaEncNoAMP.py"
    ]
    
    print_status("\nChecking U-Mamba trainers:", "info")
    trainers_ok = True
    for trainer_file in required_trainers:
        trainer_path = trainer_dir / trainer_file
        if trainer_path.exists():
            print_status(f"  {trainer_file}", "success")
        else:
            print_status(f"  {trainer_file} missing", "error")
            trainers_ok = False
    
    # Try to import trainers
    if trainers_ok:
        sys.path.insert(0, str(weak_seg_dir))
        try:
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
            print_status("\nnnUNetTrainerUMambaBot can be imported", "success")
        except ImportError as e:
            print_status(f"\nCannot import trainer: {e}", "error")
            trainers_ok = False
    
    return nets_ok and trainers_ok


def check_datasets():
    """Check dataset availability"""
    print_header("Checking Datasets")
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    if not data_dir.exists():
        print_status("data directory not found", "error")
        return False
    
    datasets = ["brain_tumors", "breast_tumors", "lung_CT", "lung_Xray"]
    datasets_ok = True
    
    for dataset_name in datasets:
        dataset_dir = data_dir / dataset_name
        
        if not dataset_dir.exists():
            print_status(f"{dataset_name}: not found", "warning")
            continue
        
        # Check subdirectories
        required_dirs = ["train_images", "train_masks", "test_images", "test_masks"]
        all_exist = all((dataset_dir / d).exists() for d in required_dirs)
        
        if all_exist:
            # Count images
            num_train = len(list((dataset_dir / "train_images").glob("*.png")))
            num_test = len(list((dataset_dir / "test_images").glob("*.png")))
            print_status(f"{dataset_name}: {num_train} train, {num_test} test", "success")
        else:
            print_status(f"{dataset_name}: incomplete (missing subdirectories)", "warning")
            datasets_ok = False
    
    return datasets_ok


def check_sam_checkpoint():
    """Check SAM checkpoint"""
    print_header("Checking SAM Checkpoint")
    
    base_dir = Path(__file__).parent
    sam_checkpoint = base_dir / "segment-anything" / "sam_checkpoints" / "sam_vit_h_4b8939.pth"
    
    if sam_checkpoint.exists():
        size_mb = sam_checkpoint.stat().st_size / 1e6
        print_status(f"SAM checkpoint found ({size_mb:.1f} MB)", "success")
        return True
    else:
        print_status("SAM checkpoint not found", "warning")
        print_status("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "info")
        return False


def check_biomedclip_model():
    """Check BiomedCLIP fine-tuned model"""
    print_header("Checking BiomedCLIP Model")
    
    base_dir = Path(__file__).parent
    model_path = base_dir / "saliency_maps" / "model" / "pytorch_model.bin"
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1e6
        print_status(f"BiomedCLIP model found ({size_mb:.1f} MB)", "success")
        return True
    else:
        print_status("BiomedCLIP fine-tuned model not found", "warning")
        print_status("Will use default BiomedCLIP from HuggingFace", "info")
        return False


def run_quick_test():
    """Run a quick integration test"""
    print_header("Running Quick Integration Test")
    
    try:
        # Test imports
        print_status("Testing imports...", "info")
        
        import torch
        import numpy as np
        from mamba_ssm import Mamba
        
        # Create small Mamba layer
        print_status("Creating Mamba layer...", "info")
        mamba = Mamba(d_model=32, d_state=8, d_conv=4, expand=2)
        
        # Test forward pass
        print_status("Testing forward pass...", "info")
        x = torch.randn(1, 100, 32)  # (batch, seq_len, d_model)
        
        if torch.cuda.is_available():
            mamba = mamba.cuda()
            x = x.cuda()
        
        with torch.no_grad():
            y = mamba(x)
        
        assert y.shape == x.shape, "Output shape mismatch"
        
        print_status("Integration test passed!", "success")
        return True
        
    except Exception as e:
        print_status(f"Integration test failed: {e}", "error")
        return False


def print_installation_instructions():
    """Print installation instructions for missing components"""
    print_header("Installation Instructions")
    
    print("""
To install missing dependencies:

1. Mamba-SSM (requires CUDA):
   pip install mamba-ssm>=1.2.0 causal-conv1d>=1.2.0

2. nnUNet:
   cd weak_segmentation
   pip install -e .

3. Other dependencies:
   pip install -r requirements.txt

4. Set environment variables (Linux/Mac):
   export nnUNet_raw="$(pwd)/nnUNet_raw"
   export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
   export nnUNet_results="$(pwd)/nnUNet_results"

   Or Windows (PowerShell):
   $env:nnUNet_raw = "$PWD/nnUNet_raw"
   $env:nnUNet_preprocessed = "$PWD/nnUNet_preprocessed"
   $env:nnUNet_results = "$PWD/nnUNet_results"

5. Download SAM checkpoint:
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   mv sam_vit_h_4b8939.pth segment-anything/sam_checkpoints/
""")


def main():
    print_header("U-MAMBA INTEGRATION VERIFICATION")
    
    all_checks = []
    
    # Python version
    print_header("Checking Python")
    all_checks.append(("Python", check_python_version()))
    
    # Core packages
    print_header("Checking Core Packages")
    all_checks.append(("torch", check_package("torch")))
    all_checks.append(("numpy", check_package("numpy")))
    all_checks.append(("SimpleITK", check_package("SimpleITK", "SimpleITK")))
    all_checks.append(("tqdm", check_package("tqdm")))
    all_checks.append(("transformers", check_package("transformers")))
    
    # CUDA
    print_header("Checking CUDA")
    all_checks.append(("CUDA", check_cuda()))
    
    # Mamba
    all_checks.append(("Mamba", check_mamba_installation()))
    
    # nnUNet
    all_checks.append(("nnUNet", check_nnunet_installation()))
    
    # U-Mamba integration
    all_checks.append(("U-Mamba Integration", check_umamba_integration()))
    
    # Datasets
    all_checks.append(("Datasets", check_datasets()))
    
    # SAM checkpoint
    all_checks.append(("SAM Checkpoint", check_sam_checkpoint()))
    
    # BiomedCLIP model
    all_checks.append(("BiomedCLIP", check_biomedclip_model()))
    
    # Quick test
    all_checks.append(("Integration Test", run_quick_test()))
    
    # Summary
    print_header("Summary")
    
    passed = sum(1 for _, status in all_checks if status)
    total = len(all_checks)
    
    print(f"\nTests passed: {passed}/{total}\n")
    
    for name, status in all_checks:
        status_str = "✓ PASS" if status else "✗ FAIL"
        color = Colors.GREEN if status else Colors.RED
        print(f"{color}{status_str}{Colors.ENDC} - {name}")
    
    print()
    
    if passed == total:
        print_status("All checks passed! Ready to train U-Mamba.", "success")
        print_status("\nNext steps:", "info")
        print("  1. python prepare_and_train_umamba.py --dataset breast_tumors")
        print("  2. Or run full pipeline: bash zeroshot_umamba.sh data/breast_tumors")
    else:
        print_status("Some checks failed. Please install missing dependencies.", "warning")
        print_installation_instructions()
    
    print()


if __name__ == "__main__":
    main()
