import numpy as np
import time
import sys
from pathlib import Path

# Add the build directory to sys.path if running directly after build
# This is usually where the .pyd/.so file ends up before installation
# Adjust the path based on your build output structure (e.g., build/lib.win-amd64-...)
# For editable installs (pip install -e .), this might not be needed.
try:
    # Assuming you built in-place or installed it
    import implied_vol_cpp_lib
except ImportError:
    print("Trying to add build directory to path...")
    # This is a potential path structure, adjust if needed!
    build_dir = list(Path(__file__).parent.parent.glob('build/lib*'))
    if build_dir:
        print(f"Adding {build_dir[0].resolve()} to sys.path")
        sys.path.append(str(build_dir[0].resolve()))
        try:
             import implied_vol_cpp_lib
        except ImportError as e:
             print(f"Failed to import after adding build path: {e}")
             sys.exit(1)
    else:
        print("Build directory not found. Please build the package first (e.g., `pip install .` or `python setup.py build_ext --inplace`)")
        sys.exit(1)


# --- Test Data Generation ---
n_options = 1000 # Increase for performance testing
target_prices = np.linspace(9.5, 10.5, n_options)
S_arr = np.full(n_options, 100.0)
r_arr = np.full(n_options, 0.05)
q_arr = np.full(n_options, 0.03) # Continuous dividend yield for mu=r-q
T_arr = np.full(n_options, 1.0)
K_arr = np.linspace(90.0, 110.0, n_options) # Vary strikes

# Generate alternating call/put types
option_types_arr = np.array(['c' if i % 2 == 0 else 'p' for i in range(n_options)])

# Use a consistent initial guess or vary it
# iv_initial_guess = np.full(n_options, 0.25)
iv_initial_guess = np.linspace(0.15, 0.35, n_options)

# Dividend schedule (common for all options):
div_times = np.array([0.25, 0.55, 0.85]) # Example dividend times
div_prop = np.array([0.01, 0.0, 0.015])  # Example proportional dividends (use 0 for cash only)
div_cash = np.array([0.5, 1.0, 0.75])   # Example cash dividends

# Binomial Tree Parameters
N = 100 # Number of steps in the binomial tree
vol_lower = 1e-4
vol_upper = 5.0
tol = 1e-5

# --- Convert to lists for C++ function (pybind11 handles conversion) ---
target_prices_list = target_prices.tolist()
S_arr_list = S_arr.tolist()
r_arr_list = r_arr.tolist()
q_arr_list = q_arr.tolist()
T_arr_list = T_arr.tolist()
K_arr_list = K_arr.tolist()
option_types_list = option_types_arr.tolist()
iv_initial_guess_list = iv_initial_guess.tolist()
div_times_list = div_times.tolist()
div_prop_list = div_prop.tolist()
div_cash_list = div_cash.tolist()

# --- Run the C++ Implementation ---
print(f"Calculating {n_options} implied volatilities using C++ library (N={N})...")
start_time = time.time()

ivs_cpp = implied_vol_cpp_lib.vectorized_implied_volatility_parallel(
    target_prices_list,
    S_arr_list,
    r_arr_list,
    q_arr_list,
    T_arr_list,
    K_arr_list,
    option_types_list,
    iv_initial_guess_list,
    div_times_list,
    div_prop_list,
    div_cash_list,
    N,
    vol_lower=vol_lower,
    vol_upper=vol_upper,
    tol=tol
)

end_time = time.time()
print(f"C++ calculation finished in {end_time - start_time:.4f} seconds.")

# Convert result back to NumPy array for easier handling
ivs_cpp_np = np.array(ivs_cpp)

print("\nSample Implied Volatilities (C++):")
print(ivs_cpp_np[:10]) # Print the first 10 results
print(f"\nNumber of NaNs (failed calculations): {np.isnan(ivs_cpp_np).sum()}")


# --- Optional: Compare with original Python (if available and feasible time-wise) ---
# Note: Running the original Python for 1000 options might be very slow.
# You might want to run it for a smaller n_options for verification.

# from Input import vectorized_implied_volatility as vectorized_implied_volatility_py
# from Input import implied_volatility # Need this too

# print("\nCalculating with original Python (this might be slow)...")
# start_time_py = time.time()
# ivs_py = vectorized_implied_volatility_py(
#     target_prices, S_arr, r_arr, q_arr, T_arr, K_arr, option_types_arr,
#     iv_initial_guess, div_times, div_prop, div_cash, N,
#     vol_lower=vol_lower, vol_upper=vol_upper, tol=tol
# )
# end_time_py = time.time()
# print(f"Python calculation finished in {end_time_py - start_time_py:.4f} seconds.")
# print("\nSample Implied Volatilities (Python):")
# print(ivs_py[:10])
# print(f"\nDifference (max abs): {np.nanmax(np.abs(ivs_cpp_np - ivs_py)) if len(ivs_py)>0 else 'N/A'}")

# --- Test Single Pricer Call (Optional) ---
print("\nTesting single C++ pricer call:")
test_sigma = 0.25
price = implied_vol_cpp_lib.ska_model_option(
    S_arr_list[0], r_arr_list[0], q_arr_list[0], test_sigma, T_arr_list[0], K_arr_list[0],
    option_types_list[0], div_times_list, div_cash_list, div_prop_list, N
)
print(f"Price for option 0 with sigma={test_sigma}: {price}")
