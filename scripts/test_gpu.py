"""
GPU/CUDA Test Script

Tests CuPy and PyTorch installation and GPU functionality.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_cupy_setup():
    """Test CuPy installation and CUDA device."""
    print("=" * 60)
    print("Test: CuPy Setup & Device")
    print("=" * 60)

    try:
        import cupy as cp
        print(f"✓ CuPy version: {cp.__version__}")
        
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"✓ Number of CUDA devices (CuPy): {device_count}")

        if device_count > 0:
            device = cp.cuda.Device(0)
            try:
                device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
                print(f"✓ Device 0: {device_name}")
            except:
                print(f"✓ Device 0: (name unavailable)")
            
            mem_info = device.mem_info
            print(f"  - Total Memory: {mem_info[1] / 1e9:.2f} GB")
            print(f"  - Free Memory: {mem_info[0] / 1e9:.2f} GB")
            return True
        else:
            print("✗ No CUDA devices found via CuPy")
            return False
    except ImportError as e:
        print(f"✗ CuPy not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error accessing CUDA device via CuPy: {e}")
        return False


def test_pytorch_setup():
    """Test PyTorch installation and CUDA device."""
    print("\n" + "=" * 60)
    print("Test: PyTorch Setup & Device")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA Available (PyTorch): {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"✓ Number of CUDA devices (PyTorch): {device_count}")
            
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ Device 0: {device_name}")
            
            # Memory info (alloc/reserved)
            # Note: total_memory is available in device properties
            props = torch.cuda.get_device_properties(0)
            print(f"  - Total Memory: {props.total_memory / 1e9:.2f} GB")
            
            return True
        else:
            print("✗ No CUDA devices found via PyTorch")
            return False
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error accessing CUDA device via PyTorch: {e}")
        return False


def test_cupy_operations():
    """Test basic array operations with CuPy."""
    print("\n" + "=" * 60)
    print("Test: CuPy Operations")
    print("=" * 60)

    try:
        import cupy as cp
        
        size = 1000
        cpu_array = np.random.rand(size, size).astype(np.float32)
        gpu_array = cp.array(cpu_array)

        print(f"✓ Created {size}x{size} array on GPU")
        
        start = time.time()
        gpu_result = cp.dot(gpu_array, gpu_array)
        cp.cuda.Stream.null.synchronize()
        end = time.time()
        
        print(f"✓ Matrix multiplication completed in {(end-start)*1000:.2f} ms")
        return True
    except Exception as e:
        print(f"✗ Error in CuPy operations: {e}")
        return False


def test_pytorch_operations():
    """Test basic tensor operations with PyTorch."""
    print("\n" + "=" * 60)
    print("Test: PyTorch Operations")
    print("=" * 60)

    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠ Skipping PyTorch operations (CUDA not available)")
            return True

        size = 1000
        # PyTorch uses slightly different syntax
        gpu_tensor = torch.rand(size, size, device='cuda')
        
        print(f"✓ Created {size}x{size} tensor on GPU")
        
        start = time.time()
        result = torch.matmul(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"✓ Matrix multiplication completed in {(end-start)*1000:.2f} ms")
        return True
    except Exception as e:
        print(f"✗ Error in PyTorch operations: {e}")
        return False


def test_array_backend():
    """Test the custom array_backend utility."""
    print("\n" + "=" * 60)
    print("Test: Array Backend Utility")
    print("=" * 60)

    try:
        from utils.array_backend import xp, BACKEND_NAME

        print(f"✓ Array backend loaded: {BACKEND_NAME}")
        # Only simple test needed
        test_array = xp.array([1, 2, 3])
        print(f"✓ Test array created with type: {type(test_array)}")
        return True
    except Exception as e:
        print(f"✗ Error testing array backend: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GPU/CUDA Test Suite (CuPy & PyTorch)")
    print("=" * 60 + "\n")

    results = []

    results.append(("CuPy Setup", test_cupy_setup()))
    results.append(("PyTorch Setup", test_pytorch_setup()))
    results.append(("CuPy Operations", test_cupy_operations()))
    results.append(("PyTorch Operations", test_pytorch_operations()))
    results.append(("Array Backend", test_array_backend()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())