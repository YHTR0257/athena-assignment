"""
Array backend utility for automatic numpy/cupy selection.

This module automatically selects CuPy if available and GPU is accessible,
otherwise falls back to NumPy.

Usage:
    from utils.array_backend import xp
    
    # Use xp just like numpy
    arr = xp.array([1, 2, 3])
    result = xp.sum(arr)
"""

import logging

logger = logging.getLogger(__name__)

def _get_array_backend():
    """
    Detect and return the appropriate array backend (cupy or numpy).
    
    Returns:
        module: cupy if available and GPU is accessible, otherwise numpy
    """
    try:
        import cupy as cp
        # Check if GPU is actually accessible
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            logger.info(f"CuPy detected with {device_count} GPU(s). Using GPU acceleration.")
            return cp
        else:
            logger.warning("CuPy is installed but no GPU detected. Falling back to NumPy.")
            import numpy as np
            return np
    except ImportError:
        logger.info("CuPy not installed. Using NumPy.")
        import numpy as np
        return np
    except Exception as e:
        logger.warning(f"Error initializing CuPy: {e}. Falling back to NumPy.")
        import numpy as np
        return np


# Automatically select backend on module import
xp = _get_array_backend()

# Export backend name for debugging
BACKEND_NAME = xp.__name__