# High-Performance Implied Volatility Calculator (C++/Python)

This project provides a high-performance C++ implementation for calculating American option prices using a binomial tree with the SKA discrete dividend adjustment method, and finding the corresponding implied volatility. The calculation is parallelized using a thread pool for significant speedups when calculating volatilities for multiple options. The C++ code is wrapped using pybind11 for easy use as a Python library.

## Features

*   C++ implementation of the SKA binomial tree for American options with discrete cash and proportional dividends.
*   Brent's method root finding for implied volatility.
*   Parallel calculation of implied volatilities using a thread pool (`BS::thread_pool`).
*   Python bindings using `pybind11`.
*   CMake build system integrated with Python's `setuptools`.
*   Cross-platform build support (Windows, macOS, Linux).

## Prerequisites

*   **Python:** 3.8 or higher (including development headers).
*   **C++ Compiler:** A modern C++ compiler supporting C++20 (e.g., GCC 10+, Clang 10+, MSVC 19.29+ / Visual Studio 2019 v16.10+).
*   **CMake:** Version 3.12 or higher.
*   **Git:** To clone the repository and potentially fetch `pybind11`.
*   **pip:** Python package installer.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd implied_vol_cpp
    ```

2.  **Install Python dependencies:**
    It's highly recommended to use a virtual environment:
    ```bash
    python -m venv venv
    # Activate environment:
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate

    pip install -r requirements.txt
    # This installs numpy, pybind11, setuptools, wheel, cmake
    ```

3.  **Build and install the C++ extension:**
    This command uses `setup.py`, which invokes CMake to build the C++ code and then installs the Python package.
    ```bash
    pip install .
    ```
    *Alternatively, for development (editable install):*
    ```bash
    pip install -e .
    ```
    *Or, build in-place without installing:*
    ```bash
    python setup.py build_ext --inplace
    ```

### Build Commands (Manual CMake - if not using `pip install .`)

These commands are generally handled by `pip install .` or `python setup.py build_ext`, but show the underlying CMake process.

**General Process:**

```
# Create a build directory
mkdir build
cd build

# Configure using CMake (from the build directory)
# Adjust path to source directory (..) if needed
# Windows (Visual Studio 2019 or later)
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release

# macOS/Linux (Makefiles)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build using CMake
cmake --build . --config Release --parallel <num_jobs> # e.g., --parallel 4
```
