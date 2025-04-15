import numpy as np
import time
import sys
from pathlib import Path
import math # For isnan

# --- Add imports for original Python functions ---
try:
    # Assumes Input.py is in the same directory or accessible in PYTHONPATH
    from Input import compute_fp as compute_fp_py
    from Input import compute_DT as compute_DT_py
    from Input import ska_model_option as ska_model_option_py
    print("Successfully imported Python reference functions from Input.py")
    run_comparisons = True
except ImportError as e:
    print(f"Warning: Could not import Python reference functions from Input.py: {e}")
    print("Skipping comparison tests.")
    run_comparisons = False
# --- End Python function imports ---


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
        print("Build directory not found. Please build the package first (e.g., `pip install .` or `python setup.py build_ext --inplace`)")
        sys.exit(1)


# =============================================================================
# DEBUGGING COMPARISONS SECTION
# =============================================================================
print("\n" + "="*20 + " DEBUGGING COMPARISONS " + "="*20)

# --- Parameters for Comparison Tests ---
# Use the same parameters as the first option in the main test below
S0_test = 100.0
K_test = 90.0 # Adjusted K to match the first option K_arr[0]
r_test = 0.05
q_test = 0.03 # Continuous dividend yield for mu=r-q
T_test = 1.0
N_test = 100
option_type_test = 'c' # Matches first option type

# Dividend schedule (common for all options):
div_times_test = np.array([0.25, 0.55, 0.85])
div_prop_test = np.array([0.01, 0.0, 0.015])
div_cash_test = np.array([0.5, 1.0, 0.75])

# Convert to lists for C++ functions if needed by bindings (though direct numpy often works)
div_times_list_test = div_times_test.tolist()
div_prop_list_test = div_prop_test.tolist()
div_cash_list_test = div_cash_test.tolist()

mu_test = r_test - q_test

# --- Test compute_fp ---
print("\n--- Testing compute_fp ---")
test_times_fp = [0.1, 0.25, 0.5, 0.7, 1.0]
if run_comparisons:
    for t in test_times_fp:
        py_val = compute_fp_py(t, mu_test, div_times_test, div_prop_test)
        cpp_val = implied_vol_cpp_lib.compute_fp(t, mu_test, div_times_list_test, div_prop_list_test) # Assuming compute_fp is bound
        diff = abs(py_val - cpp_val)
        print(f"t={t:.2f}: Py={py_val:.15f}, C++={cpp_val:.15f}, Diff={diff:.2e}")
else:
    print("Skipping compute_fp comparison.")


# --- Test compute_DT ---
print("\n--- Testing compute_DT ---")
test_times_dt = [0.0, 0.1, 0.25, 0.5, 0.7, 0.99, 1.0]
if run_comparisons:
    for t in test_times_dt:
        py_val = compute_DT_py(t, T_test, mu_test, div_times_test, div_prop_test, div_cash_test)
        cpp_val = implied_vol_cpp_lib.compute_DT(t, T_test, mu_test, div_times_list_test, div_prop_list_test, div_cash_list_test) # Assuming compute_DT is bound
        diff = abs(py_val - cpp_val)
        print(f"t={t:.2f}: Py={py_val:.15f}, C++={cpp_val:.15f}, Diff={diff:.2e}")
else:
     print("Skipping compute_DT comparison.")


# --- Test ska_model_option ---
# Note: Binding the C++ ska_model_option_cpp is needed for direct comparison
print("\n--- Testing ska_model_option ---")
test_sigmas = [0.0001, 0.1, 0.25, 0.5, 5.0]
# Check if the C++ function is bound
try:
    ska_cpp_func = implied_vol_cpp_lib.ska_model_option
    cpp_func_available = True
except AttributeError:
    print("C++ ska_model_option function not bound in library. Skipping direct comparison.")
    cpp_func_available = False

if run_comparisons and cpp_func_available:
    for sigma_test in test_sigmas:
        print(f"Testing sigma = {sigma_test:.4f}")
        # Python call (ensure Input.py uses numpy arrays)
        py_price = ska_model_option_py(S0_test, r_test, q_test, sigma_test, T_test, K_test, option_type_test,
                                       div_times_test, div_cash_test, div_prop_test, smooth=False, N=N_test)

        # C++ call (ensure lists are passed if bindings expect them)
        cpp_price = ska_cpp_func(S0_test, r_test, q_test, sigma_test, T_test, K_test, option_type_test,
                                 div_times_list_test, div_cash_list_test, div_prop_list_test, N=N_test)

        # Handle potential NaN from C++ side for comparison
        if math.isnan(cpp_price):
             diff = float('inf') # Indicate difference if C++ is NaN
             cpp_price_str = "nan"
        else:
             diff = abs(py_price - cpp_price)
             cpp_price_str = f"{cpp_price:.15f}"

        print(f"  Price: Py={py_price:.15f}, C++={cpp_price_str}, Diff={diff:.2e}")

elif run_comparisons and not cpp_func_available:
     print("Skipping ska_model_option comparison as C++ function is not bound.")
else:
     print("Skipping ska_model_option comparison.")


print("\n" + "="*23 + " END DEBUGGING " + "="*23 + "\n")
# =============================================================================
# END DEBUGGING COMPARISONS SECTION
# =============================================================================


# --- Main Implied Volatility Calculation ---
print("--- Running Implied Volatility Calculation ---")
n_options = 1 # Keep as 1 for focused debugging
# target_prices = np.linspace(9.5, 10.5, n_options)
# Use a target price likely achievable by the model based on previous tests
target_prices = np.full(n_options, 14.5) # Target price for K=90 call

S_arr = np.full(n_options, S0_test)
r_arr = np.full(n_options, r_test)
q_arr = np.full(n_options, q_test)
T_arr = np.full(n_options, T_test)
# K_arr = np.linspace(90.0, 110.0, n_options)
K_arr = np.full(n_options, K_test) # Use K=90 for the single option

# option_types_arr = np.array(['c' if i % 2 == 0 else 'p' for i in range(n_options)])
option_types_arr = np.full(n_options, option_type_test) # Use 'c'

iv_initial_guess = np.full(n_options, 0.25) # Example initial guess

# Use the same dividend schedule as comparison tests
div_times = div_times_test
div_prop = div_prop_test
div_cash = div_cash_test

# Binomial Tree Parameters
N = N_test # Use same N
vol_lower = 1e-4
vol_upper = 5.0
tol = 1e-5 # Standard tolerance

# --- Convert to lists for C++ function ---
target_prices_list = target_prices.tolist()
S_arr_list = S_arr.tolist()
r_arr_list = r_arr.tolist()
q_arr_list = q_arr.tolist()
T_arr_list = T_arr.tolist()
K_arr_list = K_arr.tolist()
option_types_list = option_types_arr.tolist()
iv_initial_guess_list = iv_initial_guess.tolist()
# Use lists defined earlier for test dividends
# div_times_list = div_times.tolist()
# div_prop_list = div_prop.tolist()
# div_cash_list = div_cash.tolist()

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
    div_times_list_test, # Pass test lists
    div_prop_list_test,
    div_cash_list_test,
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
print(ivs_cpp_np)
print(f"\nNumber of NaNs (failed calculations): {np.isnan(ivs_cpp_np).sum()}")


# --- Optional: Test single C++ pricer call ---
# print("\nTesting single C++ pricer call:")
# try:
#     test_sigma = 0.25
#     price = implied_vol_cpp_lib.ska_model_option(
#         S_arr_list[0], r_arr_list[0], q_arr_list[0], test_sigma, T_arr_list[0], K_arr_list[0],
#         option_types_list[0], div_times_list_test, div_cash_list_test, div_prop_list_test, N
#     )
#     print(f"Price for option 0 with sigma={test_sigma}: {price}")
# except AttributeError:
#     print("Cannot test single pricer: ska_model_option not bound.")
# except Exception as e:
#      print(f"Error testing single pricer: {e}")
