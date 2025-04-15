import os
import sys
import subprocess
import platform
from pathlib import Path

# --- Add pybind11 import ---
try:
    import pybind11
except ImportError:
    print("Error: pybind11 is required to build this extension.")
    print("Please install it using: pip install pybind11")
    sys.exit(1)
# --- End pybind11 import ---

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments (for MSVC)
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

# A CMakeExtension needs a sourcedir instead of a file list.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # --- Point CMAKE_LIBRARY_OUTPUT_DIRECTORY to the intermediate build lib dir ---
        # self.build_lib is set by distutils/setuptools to the build/lib.* path
        extdir = Path(self.build_lib).resolve()
        # Ensure the directory exists (CMake might need it)
        extdir.mkdir(parents=True, exist_ok=True)
        print(f"--- Setting CMAKE_LIBRARY_OUTPUT_DIRECTORY to: {extdir}")
        # --- End new calculation ---

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        is_mingw = platform.system() == "Windows" and ("MSYSTEM" in os.environ or "MINGW_PREFIX" in os.environ)

        print(f"--- Detected System: {platform.system()}")
        print(f"--- Detected MSYS/MinGW: {is_mingw}")
        print(f"--- Compiler Type reported by distutils: {self.compiler.compiler_type}")

        # --- Get pybind11 CMake directory ---
        pybind11_cmake_dir = pybind11.get_cmake_dir()
        print(f"--- Found pybind11 CMake dir: {pybind11_cmake_dir}")

        # --- Define vcpkg path (Optional - uncomment and adjust if using toolchain file) ---
        # vcpkg_root = "D:/utils/vcpkg/vcpkg" # Use forward slashes
        # vcpkg_toolchain = f"{vcpkg_root}/scripts/buildsystems/vcpkg.cmake"
        # print(f"--- Using vcpkg toolchain: {vcpkg_toolchain}")
        # --- End vcpkg path ---

        cmake_args = [
            # Use the new extdir calculation here
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
            # --- Add vcpkg toolchain file (Optional) ---
            # f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain}",
            # --- End vcpkg toolchain file ---
        ]
        build_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # --- Generator and Platform Specific Logic ---
        if cmake_generator:
             cmake_args += [f"-G{cmake_generator}"]
             print(f"--- Using CMake generator from environment: {cmake_generator} ---")
        elif is_mingw:
            ninja_path = None
            for path in os.environ['PATH'].split(os.pathsep):
                potential_ninja = Path(path) / 'ninja.exe'
                if potential_ninja.is_file():
                    ninja_path = str(potential_ninja)
                    break
            if ninja_path:
                 print("--- Selecting Ninja generator for MinGW ---")
                 cmake_args += ["-G", "Ninja"]
            else:
                 print("--- Selecting MinGW Makefiles generator ---")
                 cmake_args += ["-G", "MinGW Makefiles"]
        elif self.compiler.compiler_type == "msvc":
            print("--- Configuring for MSVC ---")
            if sys.maxsize > 2**32:
                cmake_args += ["-A", PLAT_TO_CMAKE.get(self.plat_name, "x64")]
            else:
                cmake_args += ["-A", PLAT_TO_CMAKE.get(self.plat_name, "Win32")]

        generator_uses_config_flag = "Makefiles" in cmake_generator or "Ninja" in cmake_generator or (is_mingw and not cmake_generator) or ("NMake" in cmake_generator)
        if generator_uses_config_flag:
             build_args += ["--config", cfg]


        # --- Parallel Build Logic ---
        msvc_build_jobs = 0 # Initialize here
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_jobs = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
            self.announce(f"Using up to {build_jobs} C++ build jobs (can be overridden with CMAKE_BUILD_PARALLEL_LEVEL)", level=1)
            if generator_uses_config_flag:
                if hasattr(self, "parallel") and self.parallel:
                     build_args += [f"-j{self.parallel}"]
                else:
                     build_args += [f"-j{build_jobs}"]
            elif self.compiler.compiler_type == "msvc" and not is_mingw: # MSVC parallel logic
                 msvc_build_jobs = build_jobs
                 if hasattr(self, "parallel") and self.parallel:
                     msvc_build_jobs = self.parallel
        else: # If CMAKE_BUILD_PARALLEL_LEVEL is set
            build_jobs_env = int(os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL"))
            if generator_uses_config_flag:
                 build_args += [f"-j{build_jobs_env}"]
            elif self.compiler.compiler_type == "msvc" and not is_mingw:
                 msvc_build_jobs = build_jobs_env


        # --- Run CMake ---
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print("-" * 10, "CMake Configure Args", "-" * 10)
        # Replace backslashes for better readability in printout, especially on Windows
        cmake_args_str = ' '.join(cmake_args).replace('\\', '/')
        # Perform replacement *before* the f-string
        source_dir_str = ext.sourcedir.replace('\\', '/')
        print(f"cmake {source_dir_str} {cmake_args_str}")
        print("-" * 70)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )

        print("-" * 10, "Running CMake build", "-" * 40)
        cmake_build_cmd = ["cmake", "--build", "."] + build_args
        if self.compiler.compiler_type == "msvc" and not is_mingw and msvc_build_jobs > 0:
             cmake_build_cmd += ["--", f"/m:{msvc_build_jobs}"]

        print("-" * 10, "CMake Build Command", "-" * 10)
        print(f"{' '.join(cmake_build_cmd)}")
        print("-" * 70)

        subprocess.run(cmake_build_cmd, cwd=build_temp, check=True)

        # Setuptools' build_ext --inplace mechanism will handle the copy
        # from the CMAKE_LIBRARY_OUTPUT_DIRECTORY (self.build_lib)


# Read requirements for install_requires
try:
    with open("requirements.txt", "r") as f:
        install_requires = f.read().splitlines()
except FileNotFoundError:
    print("Warning: requirements.txt not found. Installing without dependencies.")
    install_requires = ["numpy", "pybind11>=2.6"] # Ensure pybind11 is listed if reqs missing

# Read README for long description
try:
    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text()
except FileNotFoundError:
    long_description = "A Python C++ extension using pybind11 for fast, parallel implied volatility calculation."


setup(
    name="implied_vol_cpp_lib", # Package name
    version="1.0.3", # Increment version
    author="Aryan Gold", # Updated author
    author_email="your.email@example.com", # Update with your email
    description="Fast C++ American option pricing and parallel implied volatility",
    long_description=long_description,
    long_description_content_type='text/markdown', # Specify markdown for README
    ext_modules=[CMakeExtension("implied_vol_cpp_lib")], # Module name must match PYBIND11_MODULE and CMakeLists target
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++ :: 17", # Changed back to 17 as 20 caused issues
        "Development Status :: 4 - Beta",
        "Environment :: Win32 (MSWindows)",
        "Environment :: MacOS X",
        "Environment :: Console",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords='option pricing implied volatility finance american option binomial tree parallel cpp pybind11',
)