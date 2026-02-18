"""
Hardware and Parallelism Configuration Module.

This module defines the computational limits for the GAwLL algorithm,
ensuring efficient use of CPU resources and preventing memory exhaustion
during large-scale experiments.

Attributes:
    CPU_USAGE_FRACTION (float): Percentage of CPU cores to be utilized (0.0 to 1.0).
    TOTAL_CORES (int): Physical and logical cores available on the system.
    SAFE_CORES (int): Number of cores allocated for parallel evaluation.
    CACHE_LIMIT (float): Maximum number of entries in the performance cache
        to prevent memory fragmentation.
    MODEL_N_JOBS (int): Fixed to 1 to avoid nested parallelism overhead.
"""

import os

# Parallelism control
# We fix MODEL_N_JOBS to 1 because the GA already manages parallelism
# at the population level through the multiprocessing Pool.
MODEL_N_JOBS: int = 1

# CPU Allocation
# Using a fraction of cores prevents the system (and the server)
# from becoming unresponsive.
CPU_USAGE_FRACTION: float = 0.8
TOTAL_CORES: int = os.cpu_count() or 1
SAFE_CORES: int = max(1, int(TOTAL_CORES * CPU_USAGE_FRACTION))

# Memory and Performance
# This limit ensures the cache doesn't grow indefinitely,
# which is crucial for long-running GA experiments.
CACHE_LIMIT: int = int(10e4)

def get_cpu_report() -> str:
    """
    Generates a summary of the current hardware allocation.

    Returns:
        str: A formatted string with core counts and usage percentages.
    """
    return (
        f"Hardware Report: {TOTAL_CORES} cores detected. "
        f"Allocated {SAFE_CORES} cores ({CPU_USAGE_FRACTION*100}% usage)."
    )