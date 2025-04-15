#include "implied_vol.hpp"
#include "brent.hpp"        
#include "BS_thread_pool.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <future>        // For std::future
#include <thread>        // For std::thread::hardware_concurrency
#include <limits>
#include <algorithm>

// For debug
#include <iostream>
#include <iomanip> // For std::fixed, std::setprecision

// --- Helper Function Implementations ---

double compute_fp(double t, double mu,
                  const std::vector<double>& div_times,
                  const std::vector<double>& div_props) {
    double exponent = mu * t;
    double product = 1.0;
    size_t n_div = div_times.size();
    for (size_t j = 0; j < n_div; ++j) {
        // Original logic from pricer.cpp and Input.py
        if (div_times[j] <= t) {
             // Ensure prop dividend is not 1.0 or more, which makes product zero or negative
             if (div_props[j] >= 1.0) {
                 // Handle this case - maybe return 0 or NaN? Let's return 0 for FP.
                 // This implies the stock value becomes zero after such a dividend.
                 // Or return NaN if it indicates an error. Let's try NaN first.
                 // std::cerr << "Warning: Proportional dividend >= 1.0 encountered in compute_fp." << std::endl;
                 // return std::numeric_limits<double>::quiet_NaN();
                 // Let's stick to original logic for now:
                 product *= (1.0 - div_props[j]);
                 if (product <= 0) break; // Optimization if product becomes non-positive
             } else {
                product *= (1.0 - div_props[j]);
             }
        }
    }
    // Check if product became zero or negative due to div_prop >= 1.0
    if (product <= 0) {
        // Decide handling: return 0.0 or NaN? Let's return 0.0 for FP.
        return 0.0;
    }
    return std::exp(exponent) * product;
}


double compute_DT(double t, double T, double mu,
                  const std::vector<double>& div_times,
                  const std::vector<double>& div_props,
                  const std::vector<double>& div_cash) {

    if (T <= 0) return 0.0; // Avoid division by zero if T=0

    double fp_t = compute_fp(t, mu, div_times, div_props);
     if (std::isnan(fp_t) || fp_t == 0.0) { // Check fp_t validity
         // std::cerr << "Warning: compute_fp(" << t << ") returned invalid value in compute_DT." << std::endl;
         // Decide handling: return 0 or NaN? Let's return 0 for the shift.
         return 0.0; // If fp_t is invalid, assume shift is 0? Or propagate NaN? Propagating NaN is safer.
         // return std::numeric_limits<double>::quiet_NaN();
     }

    double sum1 = 0.0;
    double sum2 = 0.0;
    size_t n_div = div_times.size();

    if (n_div != div_props.size() || n_div != div_cash.size()) {
        throw std::runtime_error("Dividend vector sizes mismatch in compute_DT");
    }

    for (size_t i = 0; i < n_div; ++i) {
        double ti = div_times[i];
        double di = div_cash[i];

        // Skip non-positive dividend times? pricer.cpp didn't explicitly skip.
        // Let's stick to pricer.cpp logic for now.
        // if (ti <= 0.0) continue;

        double fp_ti = compute_fp(ti, mu, div_times, div_props);

        // Check fp_ti before division
        if (std::isnan(fp_ti) || fp_ti == 0.0) {
            // std::cerr << "Warning: compute_fp(" << ti << ") returned invalid value in compute_DT loop." << std::endl;
            // Skip this dividend's contribution if fp_ti is invalid? Or return NaN?
            // Skipping seems more robust if one dividend is problematic.
            continue; // Skip this dividend contribution
            // return std::numeric_limits<double>::quiet_NaN(); // Propagate NaN
        }

        double term = di / fp_ti; // Calculate term only once

        if (t < ti && ti <= T) { // Condition for sum1 (Matches pricer.cpp)
            sum1 += term;
        }
        if (ti <= T) { // Condition for sum2 (Matches pricer.cpp)
            // Avoid division by zero if T=0 (handled at function start)
            sum2 += (ti / T) * term;
        }
    }
    return fp_t * (sum1 - sum2);
}


// --- Core Option Pricer ---

// --- Helper function for Zero-Volatility Price ---
// (Moved logic from inside ska_model_option_cpp)
double calculate_zero_vol_price(
    double S0, double r, double q, double T, double K,
    const std::string& option_type_str,
    const std::vector<double>& div_times,
    const std::vector<double>& div_cash,
    const std::vector<double>& div_prop)
{
     if (T <= 0) { // Handle T=0 case within helper too
        double payoff = 0.0;
        if (option_type_str == "c") { payoff = std::max(S0 - K, 0.0); }
        else { payoff = std::max(K - S0, 0.0); }
        return payoff;
     }

     double mu_calc = r - q;
     double fp_maturity = compute_fp(T, mu_calc, div_times, div_prop);
     if (std::isnan(fp_maturity)) return std::numeric_limits<double>::quiet_NaN();

     double sum_div_value = 0.0;
     size_t n_div_fwd = div_times.size();
     for (size_t j = 0; j < n_div_fwd; ++j) {
         if (div_times[j] > 0 && div_times[j] <= T) {
             double fp_j = compute_fp(div_times[j], mu_calc, div_times, div_prop);
             if (fp_j != 0.0 && !std::isnan(fp_j)) {
                 sum_div_value += div_cash[j] / fp_j;
             } else {
                 return std::numeric_limits<double>::quiet_NaN(); // Propagate failure
             }
         }
     }

     double forward_price = fp_maturity * (S0 - sum_div_value);
     if (std::isnan(forward_price)) return std::numeric_limits<double>::quiet_NaN();


     double payoff_at_maturity = 0.0;
     if (option_type_str == "c") {
         payoff_at_maturity = std::max(forward_price - K, 0.0);
     } else { // 'p'
         payoff_at_maturity = std::max(K - forward_price, 0.0);
     }

     return payoff_at_maturity * std::exp(-r * T);
}


double ska_model_option_cpp(
    double S0, double r, double q, double sigma, double T, double K,
    const std::string& option_type_str, // Keep original name and signature
    const std::vector<double>& div_times,
    const std::vector<double>& div_cash,
    const std::vector<double>& div_props,
    int N) // N is passed explicitly, no default needed here
{
    // Basic checks
     if (T <= 0.0) {
        // Use helper for consistency
        return calculate_zero_vol_price(S0, r, q, T, K, option_type_str, div_times, div_cash, div_props);
     }
     if (N <= 0) {
        throw std::invalid_argument("Number of steps N must be positive.");
     }

    // --- Combine sigma <= 0 and near-zero check ---
    // Define a threshold below which we treat sigma as effectively zero
    const double EFFECTIVE_ZERO_SIGMA_TOL = 1e-9; // Adjust if needed
    if (sigma < EFFECTIVE_ZERO_SIGMA_TOL) {
         // Directly return the zero-volatility price calculation
         return calculate_zero_vol_price(S0, r, q, T, K, option_type_str, div_times, div_cash, div_props);
    }
    // --- End combined check ---


    // --- Start of logic for sigma > EFFECTIVE_ZERO_SIGMA_TOL ---
    double mu = r - q;
    double dt = T / static_cast<double>(N);
    double sigma_sqrt_dt = sigma * std::sqrt(dt); // Calculate once
    double u = std::exp(sigma_sqrt_dt);
    double d = 1.0 / u;
    double u_minus_d = u - d; // Calculate once

    // Check if u and d are too close (should be redundant now but keep as safety)
    // Use a slightly larger tolerance here just in case exp() causes issues
    if (std::abs(u_minus_d) < std::numeric_limits<double>::epsilon() * 100) {
        // Fallback to zero-vol price if somehow sigma wasn't caught but u/d are identical
        return calculate_zero_vol_price(S0, r, q, T, K, option_type_str, div_times, div_cash, div_props);
    }

    double p = (std::exp(mu * dt) - d) / u_minus_d;
    double discount = std::exp(-r * dt);

    // --- Reinstate p-bounds check ---
    // Since we now only enter this block if sigma is sufficiently > 0,
    // p should ideally be within [0, 1]. If not, it indicates a potential
    // issue with large N or extreme parameters, so returning NaN is appropriate.
    const double P_TOL = 1e-10;
    if (p < -P_TOL || p > 1.0 + P_TOL) {
         std::cerr << "ERROR: Probability p=" << p << " significantly outside [0, 1] in ska_model_option_cpp.\n"
                   << "  sigma=" << sigma << ", T=" << T << ", N=" << N << ", r=" << r << ", q=" << q << "\n"
                   << "  u=" << std::fixed << std::setprecision(15) << u
                   << ", d=" << d << ", exp(mu*dt)=" << std::exp(mu * dt) << std::endl;
         return std::numeric_limits<double>::quiet_NaN();
    }
    // Clamp p if it's slightly outside [0, 1] due to numerical noise
    // This clamping is important for stability if p is e.g. 1.00000000001 or -0.00000000001
    if (p < 0.0 || p > 1.0) {
        // std::cerr << "WARNING: Clamping p=" << p << " to [0, 1] range." << std::endl;
        p = std::max(0.0, std::min(1.0, p));
    }
    // --- End reinstate p-bounds check ---

    // --- NO explicit arbitrage check (d >= exp_mu_dt || exp_mu_dt >= u) here ---
    // --- NO explicit p-bounds check here ---

    // Compute D_t (dividend shifts) at each time node.
    std::vector<double> D_t(N + 1, 0.0);
    for (int i = 0; i <= N; ++i) {
        double t_i = static_cast<double>(i) * dt; // Calculate time directly
        D_t[i] = compute_DT(t_i, T, mu, div_times, div_props, div_cash);
         if (std::isnan(D_t[i])) {
             // std::cerr << "ERROR: compute_DT returned NaN for t=" << t_i << std::endl;
             return std::numeric_limits<double>::quiet_NaN(); // Propagate NaN
         }
    }

    // Allocate flattened 1D array for tilde_S (dimension: (N+1) x (N+1)).
    std::vector<double> tilde_S((N + 1) * (N + 1), 0.0);

    // Initialization: at time 0, index 0.
    tilde_S[0] = S0 - D_t[0]; // Correct 1D access
     if (std::isnan(tilde_S[0])) { // Correct 1D access
         // std::cerr << "ERROR: Initial tilde_S[0] is NaN (S0=" << S0 << ", D_t[0]=" << D_t[0] << ")" << std::endl;
         return std::numeric_limits<double>::quiet_NaN();
     }


    // Build the binomial tree (Forward pass).
    for (int i = 1; i <= N; ++i) {
        // Calculate row starts for flattened array access
        int row_start = i * (N + 1);
        int prev_row_start = (i - 1) * (N + 1);

        // Downward move for leftmost node.
         if (std::isnan(tilde_S[prev_row_start])) return std::numeric_limits<double>::quiet_NaN(); // Correct 1D access
        tilde_S[row_start] = tilde_S[prev_row_start] * d; // Correct 1D access

        // Upward moves: nodes 1 to i.
        for (int j = 1; j <= i; ++j) {
             if (std::isnan(tilde_S[prev_row_start + (j - 1)])) return std::numeric_limits<double>::quiet_NaN(); // Correct 1D access
            tilde_S[row_start + j] = tilde_S[prev_row_start + (j - 1)] * u; // Correct 1D access
        }

        // Dividend adjustment: for dividends in interval ((i-1)*dt, i*dt]
        double t_lower = (i - 1) * dt;
        double t_upper = i * dt;
        for (size_t k = 0; k < div_times.size(); ++k) {
            if (div_times[k] > t_lower && div_times[k] <= t_upper) {
                double prop_mult = (1.0 - div_props[k]);
                if (prop_mult < 0) { prop_mult = 0.0; } // Handle div_prop >= 1 case
                for (int j = 0; j <= i; ++j) {
                    tilde_S[row_start + j] *= prop_mult; // Correct 1D access
                }
            }
        }
         // Check for NaNs after adjustment
         for (int j = 0; j <= i; ++j) {
             if (std::isnan(tilde_S[row_start + j])) { // Correct 1D access
                 // std::cerr << "ERROR: NaN detected in tilde_S at index " << (row_start + j) << " after div adjustment." << std::endl;
                 return std::numeric_limits<double>::quiet_NaN();
             }
         }
    }

    // Backward induction.
    std::vector<double> V((N + 1) * (N + 1), 0.0); // Option values (flattened)
    double K_T = K - D_t[N]; // Adjusted strike at maturity
     if (std::isnan(K_T)) return std::numeric_limits<double>::quiet_NaN(); // NaN check

    int final_row_idx = N * (N + 1); // Index for the start of the final row
    for (int j = 0; j <= N; ++j) {
        double ST = tilde_S[final_row_idx + j]; // Correct 1D access
         if (std::isnan(ST)) return std::numeric_limits<double>::quiet_NaN(); // NaN check
        if (option_type_str == "c")
            V[final_row_idx + j] = std::max(ST - K_T, 0.0); // Correct 1D access
        else // "p"
            V[final_row_idx + j] = std::max(K_T - ST, 0.0); // Correct 1D access
    }

    // Loop backwards from N-1 down to 0
    double p_up = p; // Use p directly
    double p_down = 1.0 - p;
    double disc_p_up = discount * p_up;
    double disc_p_down = discount * p_down;

    for (int i = N - 1; i >= 0; --i) {
        int row_start = i * (N + 1);
        int next_row_start = (i + 1) * (N + 1);
        double K_t = K - D_t[i]; // Adjusted strike at time i
         if (std::isnan(K_t)) return std::numeric_limits<double>::quiet_NaN(); // NaN check

        for (int j = 0; j <= i; ++j) {
            // Calculate continuation value
             if (std::isnan(V[next_row_start + (j + 1)]) || std::isnan(V[next_row_start + j])) { // Correct 1D access
                 return std::numeric_limits<double>::quiet_NaN();
             }
            double continuation = disc_p_up * V[next_row_start + (j + 1)] + disc_p_down * V[next_row_start + j]; // Correct 1D access

            // Calculate exercise value
            double current_tilde_S = tilde_S[row_start + j]; // Correct 1D access
             if (std::isnan(current_tilde_S)) return std::numeric_limits<double>::quiet_NaN(); // NaN check
            double exercise = 0.0;
            if (option_type_str == "c")
                exercise = std::max(current_tilde_S - K_t, 0.0);
            else // "p"
                exercise = std::max(K_t - current_tilde_S, 0.0);

             if (std::isnan(continuation) || std::isnan(exercise)) { // Final check before max
                 return std::numeric_limits<double>::quiet_NaN();
             }
            V[row_start + j] = std::max(exercise, continuation); // Correct 1D access
        }
    }

    // Final result is at the root of the option value tree
     if (std::isnan(V[0])) { // Correct 1D access
         // std::cerr << "ERROR: Final option value V[0] is NaN." << std::endl;
         return std::numeric_limits<double>::quiet_NaN();
     }
    return V[0]; // Correct 1D access
}

// --- Implied Volatility Solver ---

double implied_volatility_cpp(
    double target_price, double S0, double r, double q, double T, double K,
    const std::string& option_type,
    const std::vector<double>& div_times,
    const std::vector<double>& div_prop,
    const std::vector<double>& div_cash,
    int N,
    double initial_guess, // Keep param for API
    double vol_lower, double vol_upper, double tol)
{
    (void)initial_guess; // Suppress unused warning

    auto objective_func = [&](double sigma) {
        // Note: ska_model_option_cpp now handles sigma<=0 internally
        double model_price = ska_model_option_cpp(S0, r, q, sigma, T, K, option_type,
                                                  div_times, div_cash, div_prop, N);
        // Optional Debug Print inside Objective
        if (sigma == vol_lower || sigma == vol_upper) {
            std::cerr << std::fixed << std::setprecision(8)
                      << "    sigma=" << sigma << ", model_price=" << model_price << std::endl;
        }
        if (std::isnan(model_price)) {
             return std::numeric_limits<double>::quiet_NaN(); // Propagate NaN
        }
        return model_price - target_price;
    };

    // Debug Print before calling objective
    std::cerr << "IV Calc: Target=" << target_price << ", S0=" << S0 << ", K=" << K
              << ", T=" << T << ", Type=" << option_type << ", N=" << N
              << ", Bounds=[" << vol_lower << ", " << vol_upper << "]" << std::endl;

    double f_lower = objective_func(vol_lower);
    double f_upper = objective_func(vol_upper);

    // Debug Print after calling objective
    std::cerr << std::fixed << std::setprecision(8)
              << "  Price(low_vol) = " << (f_lower + target_price) << ", f_lower = " << f_lower << std::endl;
    std::cerr << std::fixed << std::setprecision(8)
              << "  Price(high_vol)= " << (f_upper + target_price) << ", f_upper = " << f_upper << std::endl;


    if (std::isnan(f_lower) || std::isnan(f_upper)) {
        std::cerr << "  -> Pricing failed at bounds, returning NaN." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (f_lower * f_upper >= 0) {
        // std::cerr << "  -> Root not bracketed (f_lower * f_upper = " << f_lower * f_upper << "), checking tolerance..." << std::endl;
         if (target_price > 1e-9 && std::abs(f_lower) < tol * target_price) {
            //  std::cerr << "     -> Approx match at lower bound." << std::endl;
             return vol_lower;
         }
         if (target_price > 1e-9 && std::abs(f_upper) < tol * target_price) {
            //  std::cerr << "     -> Approx match at upper bound." << std::endl;
             return vol_upper;
         }
        std::cerr << "     -> No match within tolerance, returning NaN." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }

    try {
        // std::cerr << "  -> Root bracketed. Calling brentq..." << std::endl;
        double result = RootFinding::brentq(objective_func, vol_lower, vol_upper, tol, 100);
        std::cerr << "  -> brentq result: " << result << std::endl;
        // Add a final check if brentq itself returns NaN
        if (std::isnan(result)) {
             std::cerr << "  -> brentq returned NaN." << std::endl;
        }
        return result;
    } catch (const std::exception& e) {
        std::cerr << "  -> Exception during brentq: " << e.what() << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
}



// --- Vectorized & Parallel Implementation ---

std::vector<double> vectorized_implied_volatility_cpp(
    const std::vector<double>& target_prices,
    const std::vector<double>& S_arr,
    const std::vector<double>& r_arr,
    const std::vector<double>& q_arr,
    const std::vector<double>& T_arr,
    const std::vector<double>& K_arr,
    const std::vector<std::string>& option_types_arr,
    const std::vector<double>& iv_initial_guess, // Keep for potential future use
    const std::vector<double>& div_times, // Common
    const std::vector<double>& div_prop,  // Common
    const std::vector<double>& div_cash,  // Common
    int N,                              // Common
    double vol_lower,                   // Common
    double vol_upper,                   // Common
    double tol)                         // Common
{
    size_t n_options = target_prices.size();
    if (S_arr.size() != n_options /* ... other checks ... */) {
        throw std::runtime_error("Input array sizes do not match target_prices size.");
    }

    std::vector<double> ivs(n_options);

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) { num_threads = 2; }

    BS::thread_pool pool(static_cast<std::size_t>(num_threads));
    std::vector<std::future<double>> futures;
    futures.reserve(n_options);

    for (size_t i = 0; i < n_options; ++i) {
        // Use submit_task based on user feedback
        futures.emplace_back(pool.submit_task(
            [=, &div_times, &div_prop, &div_cash]() -> double { // Capture common dividends by reference
                (void)iv_initial_guess[i]; // Suppress unused warning
                return implied_volatility_cpp(
                    target_prices[i], S_arr[i], r_arr[i], q_arr[i], T_arr[i], K_arr[i],
                    option_types_arr[i], div_times, div_prop, div_cash, N,
                    iv_initial_guess[i],
                    vol_lower, vol_upper, tol);
            }
        ));
    }

    for (size_t i = 0; i < n_options; ++i) {
        try {
            ivs[i] = futures[i].get();
        } catch (const std::exception& e) {
            // std::cerr << "Error calculating IV for option " << i << ": " << e.what() << std::endl;
            ivs[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    return ivs;
}