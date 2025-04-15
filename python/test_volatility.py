import numpy as np
import time
import sys
from pathlib import Path
import math # For isnan

# Add the build directory to sys.path if running directly after build
try:
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


# =============================================================================
# --- Main Test Data Generation ---
# =============================================================================
n_options = 10 # Number of options to test
print(f"--- Setting up {n_options} options for IV calculation ---")

# --- Define parameters consistently ---
S_val = 100.0
r_val = 0.05
q_val = 0.03
T_val = 1.0
N_val = 100 # Number of steps for the C++ pricer

S_arr = np.full(n_options, S_val)
r_arr = np.full(n_options, r_val)
q_arr = np.full(n_options, q_val)
T_arr = np.full(n_options, T_val)
# Strikes slightly around the money - more likely to have prices in a solvable range
K_arr = np.linspace(95.0, 105.0, n_options)
# Target prices - choose a range that might be plausible for these strikes/params
# (Note: Some combinations might still be unsolvable if target price is outside model bounds)
target_prices = np.linspace(5.0, 15.0, n_options)
# Alternating option types
option_types_arr = np.array(['c', 'p'] * (n_options // 2 + 1))[:n_options]
# Use a constant initial guess for simplicity
iv_initial_guess = np.full(n_options, 0.25)

# Dividend schedule (common for all options):
div_times = np.array([0.25, 0.55, 0.85])
div_prop = np.array([0.01, 0.0, 0.015])
div_cash = np.array([0.5, 1.0, 0.75])
# Convert dividends to lists for C++ call
div_times_list = div_times.tolist()
div_prop_list = div_prop.tolist()
div_cash_list = div_cash.tolist()

# Binomial Tree Parameters for IV solver
vol_lower = 1e-4
vol_upper = 5.0
tol = 1e-5 # Standard tolerance

# --- Convert inputs to lists for C++ function ---
target_prices_list = target_prices.tolist()
S_arr_list = S_arr.tolist()
r_arr_list = r_arr.tolist()
q_arr_list = q_arr.tolist()
T_arr_list = T_arr.tolist()
K_arr_list = K_arr.tolist()
option_types_list = option_types_arr.tolist()
iv_initial_guess_list = iv_initial_guess.tolist()


# =============================================================================
# --- Run the C++ Implied Volatility Calculation ---
# =============================================================================
print(f"\nCalculating {n_options} implied volatilities using C++ library (N={N_val})...")
# --- Ensure C++ debug prints are commented out for clean output ---

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
    N_val, # Use consistent N
    vol_lower=vol_lower,
    vol_upper=vol_upper,
    tol=tol
)

end_time = time.time()
print(f"C++ calculation finished in {end_time - start_time:.4f} seconds.")

# Convert result back to NumPy array for easier handling
ivs_cpp_np = np.array(ivs_cpp)

# =============================================================================
# --- Display Results ---
# =============================================================================
print("\nResults:")
# Header - Adjust spacing as needed
print(f"{'Option':<6} {'Type':<4} {'K':<7} {'Target P':<10} {'Calc IV':<12}")
print("-" * 42) # Adjust separator length
for i in range(n_options):
    # Format NaN nicely for output
    iv_str = f"{ivs_cpp_np[i]:<12.6f}" if not np.isnan(ivs_cpp_np[i]) else f"{'NaN':<12}"
    print(f"{i:<6} {option_types_arr[i]:<4} {K_arr[i]:<7.2f} {target_prices[i]:<10.4f} {iv_str}")

print(f"\nNumber of NaNs (failed calculations): {np.isnan(ivs_cpp_np).sum()}")

# Optional: You could add a check here to see if the number of NaNs is unexpected.
if np.isnan(ivs_cpp_np).any():
    print("\nNote: Some options resulted in NaN. This might be because the target price")
    print("      is outside the possible range for the model with vols between")
    print(f"      {vol_lower} and {vol_upper}, or other numerical issues occurred.")