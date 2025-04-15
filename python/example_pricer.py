import numpy as np
import time
import sys
from pathlib import Path
import math

# Add the build directory to sys.path if running directly after build
try:
    import implied_vol_cpp_lib
except ImportError:
    print("Trying to add build directory to path...")
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
        print("Build directory not found. Please build the package first.")
        sys.exit(1)

# =============================================================================
# --- Test Data Generation ---
# =============================================================================
n_options = 10000 # Use a larger number to see timing differences
print(f"--- Setting up {n_options} options for pricing ---")

# --- Define parameters ---
S_val = 100.0
r_val = 0.05
q_val = 0.03
T_val = 1.0
N_val = 100 # Number of steps for the C++ pricer

S_arr = np.full(n_options, S_val)
r_arr = np.full(n_options, r_val)
q_arr = np.full(n_options, q_val)
T_arr = np.full(n_options, T_val)
# Vary strikes and sigma
K_arr = np.linspace(90.0, 110.0, n_options)
sigma_arr = np.linspace(0.15, 0.45, n_options)
# Alternating option types
option_types_arr = np.array(['c', 'p'] * (n_options // 2 + 1))[:n_options]

# Dividend schedule (common for all options):
div_times = np.array([0.25, 0.55, 0.85])
div_prop = np.array([0.01, 0.0, 0.015])
div_cash = np.array([0.5, 1.0, 0.75])
# Convert dividends to lists for C++ call
div_times_list = div_times.tolist()
div_prop_list = div_prop.tolist()
div_cash_list = div_cash.tolist()

# --- Convert inputs to lists for C++ function ---
S_arr_list = S_arr.tolist()
r_arr_list = r_arr.tolist()
q_arr_list = q_arr.tolist()
sigma_arr_list = sigma_arr.tolist()
T_arr_list = T_arr.tolist()
K_arr_list = K_arr.tolist()
option_types_list = option_types_arr.tolist()

# =============================================================================
# --- Warmup ---
# =============================================================================

prices_sync = implied_vol_cpp_lib.ska_model_option_sync(
    S_arr_list,
    r_arr_list,
    q_arr_list,
    sigma_arr_list,
    T_arr_list,
    K_arr_list,
    option_types_list,
    div_times_list,
    div_prop_list,
    div_cash_list,
    N_val
)

prices_async = implied_vol_cpp_lib.ska_model_option_async(
    S_arr_list,
    r_arr_list,
    q_arr_list,
    sigma_arr_list,
    T_arr_list,
    K_arr_list,
    option_types_list,
    div_times_list,
    div_prop_list,
    div_cash_list, 
    N_val
)


# =============================================================================
# --- Run Synchronous Calculation ---
# =============================================================================
print(f"\nCalculating {n_options} option prices sequentially (sync)...")
start_time_sync = time.time()

prices_sync = implied_vol_cpp_lib.ska_model_option_sync(
    S_arr_list,
    r_arr_list,
    q_arr_list,
    sigma_arr_list,
    T_arr_list,
    K_arr_list,
    option_types_list,
    div_times_list,
    div_prop_list,
    div_cash_list,
    N_val
)

end_time_sync = time.time()
time_sync = end_time_sync - start_time_sync
print(f"Sync calculation finished in {time_sync:.4f} seconds.")

# =============================================================================
# --- Run Asynchronous (Parallel) Calculation ---
# =============================================================================
print(f"\nCalculating {n_options} option prices in parallel (async)...")
start_time_async = time.time()

prices_async = implied_vol_cpp_lib.ska_model_option_async(
    S_arr_list,
    r_arr_list,
    q_arr_list,
    sigma_arr_list,
    T_arr_list,
    K_arr_list,
    option_types_list,
    div_times_list,
    div_prop_list,
    div_cash_list, 
    N_val
)

end_time_async = time.time()
time_async = end_time_async - start_time_async
print(f"Async calculation finished in {time_async:.4f} seconds.")

# =============================================================================
# --- Verify Results and Show Sample ---
# =============================================================================

# Convert results to numpy arrays
prices_sync_np = np.array(prices_sync)
prices_async_np = np.array(prices_async)

# Check for differences
diff = np.abs(prices_sync_np - prices_async_np)
max_diff = np.nanmax(diff) # Use nanmax in case of NaNs
num_nans_sync = np.isnan(prices_sync_np).sum()
num_nans_async = np.isnan(prices_async_np).sum()

print("\nVerification:")
print(f"  Max absolute difference between sync/async results: {max_diff:.2e}")
print(f"  Number of NaNs (sync): {num_nans_sync}")
print(f"  Number of NaNs (async): {num_nans_async}")
if time_sync > 0:
    speedup = time_sync / time_async if time_async > 0 else float('inf')
    print(f"  Approximate speedup (sync_time / async_time): {speedup:.2f}x")


print("\nSample Results (first 10):")
# Header - Adjust spacing as needed
print(f"{'Option':<6} {'Type':<4} {'K':<7} {'Sigma':<7} {'Price (sync)':<14} {'Price (async)':<14}")
print("-" * 60) # Adjust separator length
for i in range(min(n_options, 10)):
    # Format NaN nicely for output
    p_sync_str = f"{prices_sync_np[i]:<14.6f}" if not np.isnan(prices_sync_np[i]) else f"{'NaN':<14}"
    p_async_str = f"{prices_async_np[i]:<14.6f}" if not np.isnan(prices_async_np[i]) else f"{'NaN':<14}"
    print(f"{i:<6} {option_types_arr[i]:<4} {K_arr[i]:<7.2f} {sigma_arr[i]:<7.4f} {p_sync_str} {p_async_str}")

if num_nans_sync > 0 or num_nans_async > 0:
     print("\nNote: NaN results indicate potential issues in the pricer for specific inputs.")