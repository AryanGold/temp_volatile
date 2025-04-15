cmake_minimum_required(VERSION 3.12)
project(ImpliedVolCpp VERSION 1.0)

# --- Pybind11 Setup ---
# Finds pybind11 development files (headers, cmake configs)
# It will find pybind11 installed via pip in the current Python environment
find_package(pybind11 CONFIG REQUIRED)

# --- C++ Standard and Compiler Flags ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard C++

# Add optimization flags for release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -flto -ffast-math -funroll-loops -fomit-frame-pointer -DNDEBUG") # Common flags for GCC/Clang
# Add MSVC specific flags if needed
if(MSVC)
    # Add specific MSVC flags, e.g. /O2, /DNDEBUG is added by CMake
    # /MP enables parallel builds in MSVC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    # Todo: add -Ofast -march=native -flto -ffast-math -funroll-loops -fomit-frame-pointer
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2") 
endif()


# --- Define the Python Module ---
# pybind11_add_module(<module_name> [SHARED|MODULE] [NO_EXTRAS] sources...)
# <module_name> should match the first argument to PYBIND11_MODULE in bindings.cpp
# SHARED: creates a shared library (.so or .pyd)
pybind11_add_module(implied_vol_cpp_lib SHARED
    cpp_src/ska_pricer.cpp
    cpp_src/implied_vol.cpp
    cpp_src/bindings.cpp
    # brent.hpp and thread_pool.hpp are header-only, no need to list here
)

# --- Explicitly set output directory property for the target ---
# CMAKE_LIBRARY_OUTPUT_DIRECTORY should be passed from setup.py and contain the build/lib.* path
if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(implied_vol_cpp_lib PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" # Set both just in case
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" # And archive for good measure
    )
    message(STATUS "Setting target output properties (LIBRARY, RUNTIME, ARCHIVE) to: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
else()
    message(WARNING "CMAKE_LIBRARY_OUTPUT_DIRECTORY not set from setup.py, output location might be default.")
endif()
# --- End explicit setting ---

# Target specific properties (optional)
# target_include_directories(implied_vol_cpp_lib PUBLIC ${pybind11_INCLUDE_DIRS}) # Handled by pybind11_add_module
target_compile_options(implied_vol_cpp_lib PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/EHsc /MP> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pthread>)
# Add -pthread for non-MSVC if using std::thread directly (not needed for BS::thread_pool usually)


# Installation (optional, useful if building outside setup.py)
# install(TARGETS implied_vol_cpp_lib DESTINATION lib/python${PY_VERSION_MAJOR}.${PY_VERSION_MINOR}/site-packages)

# You might need to tell CMake where cpp_src is if running from build dir
# target_include_directories(implied_vol_cpp_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp_src) # Handled by pybind11_add_module


# Example of defining version info macro for bindings.cpp
# target_compile_definitions(implied_vol_cpp_lib PRIVATE VERSION_INFO=${PROJECT_VERSION})
