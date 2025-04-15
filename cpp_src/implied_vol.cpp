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

// Uncomment for show details trace info for Solver
//#define EXTRA_DEBUGS

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
    const std::string& option_type_str,
    const std::vector<double>& div_times,
    const std::vector<double>& div_cash,
    const std::vector<double>& div_prop,
    int N)
{
    // Basic checks
    if (T <= 0.0) {
        return calculate_zero_vol_price(S0, r, q, T, K, option_type_str, div_times, div_cash, div_prop);
    }
    if (N <= 0) {
        throw std::invalid_argument("Number of steps N must be positive.");
    }

    // --- Combined sigma <= 0 and near-zero check ---
    const double EFFECTIVE_ZERO_SIGMA_TOL = 1e-9;
    if (sigma < EFFECTIVE_ZERO_SIGMA_TOL) {
        return calculate_zero_vol_price(S0, r, q, T, K, option_type_str, div_times, div_cash, div_prop);
    }
    // --- End combined check ---

    // --- Start of logic for sigma > EFFECTIVE_ZERO_SIGMA_TOL ---
    double mu = r - q;
    double dt = T / static_cast<double>(N);
    double sigma_sqrt_dt = sigma * std::sqrt(dt);
    double u = std::exp(sigma_sqrt_dt);
    double d = 1.0 / u;
    double u_minus_d = u - d;

    // Fallback check if u and d are too close
    if (std::abs(u_minus_d) < std::numeric_limits<double>::epsilon() * 100) {
        return calculate_zero_vol_price(S0, r, q, T, K, option_type_str, div_times, div_cash, div_prop);
    }

    double p = (std::exp(mu * dt) - d) / u_minus_d;
    double discount = std::exp(-r * dt);

    // --- Clamp p (important for stability) ---
    if (p < 0.0 || p > 1.0) {
        p = std::max(0.0, std::min(1.0, p));
    }
    // --- End clamp p ---

    double p_up = p;
    double p_down = 1.0 - p;
    double disc_p_up = discount * p_up;
    double disc_p_down = discount * p_down;

    // Compute D_t shifts
    std::vector<double> D_t(N + 1, 0.0);
    for (int i = 0; i <= N; ++i) {
        double t_i = static_cast<double>(i) * dt;
        D_t[i] = compute_DT(t_i, T, mu, div_times, div_prop, div_cash);
        if (std::isnan(D_t[i])) { return std::numeric_limits<double>::quiet_NaN(); }
    }

    // Use vector<vector> for the tree (original approach)
    std::vector<std::vector<double>> tilde_S(N + 1);
    for(int i = 0; i <= N; ++i) {
        tilde_S[i].resize(i + 1); // Each row i has i+1 nodes
    }

    // Initialization
    tilde_S[0][0] = S0 - D_t[0];
    if (std::isnan(tilde_S[0][0])) { return std::numeric_limits<double>::quiet_NaN(); }

    // Forward pass (using 2D indexing)
    for (int i = 1; i <= N; ++i) {
        // Downward move
        if (std::isnan(tilde_S[i-1][0])) return std::numeric_limits<double>::quiet_NaN();
        tilde_S[i][0] = tilde_S[i-1][0] * d;
        // Upward moves
        for (int j = 1; j <= i; ++j) {
            //if (std::isnan(tilde_S[i-1][j-1])) return std::numeric_limits<double>::quiet_NaN();
            tilde_S[i][j] = tilde_S[i-1][j-1] * u;
        }

        // Dividend adjustment
        double t_lower = (i - 1) * dt;
        double t_upper = i * dt;
        for (size_t k = 0; k < div_times.size(); ++k) {
            if (div_times[k] > t_lower && div_times[k] <= t_upper) {
                double prop_mult = (1.0 - div_prop[k]);
                if (prop_mult < 0) { prop_mult = 0.0; }
                for (int j = 0; j <= i; ++j) {
                    tilde_S[i][j] *= prop_mult;
                }
            }
        }
         // NaN check after adjustment
         for (int j = 0; j <= i; ++j) {
             if (std::isnan(tilde_S[i][j])) { return std::numeric_limits<double>::quiet_NaN(); }
         }
    } // End forward pass

    // Backward induction (using 2D indexing)
    std::vector<std::vector<double>> V(N + 1);
     for(int i = 0; i <= N; ++i) {
        V[i].resize(i + 1); // Each row i has i+1 nodes
    }

    double K_T = K - D_t[N];
    if (std::isnan(K_T)) return std::numeric_limits<double>::quiet_NaN();

    // Terminal values
    for (int j = 0; j <= N; ++j) {
        double ST = tilde_S[N][j];
        if (std::isnan(ST)) return std::numeric_limits<double>::quiet_NaN();
        if (option_type_str == "c")
            V[N][j] = std::max(ST - K_T, 0.0);
        else // "p"
            V[N][j] = std::max(K_T - ST, 0.0);
    }

    // Backward loop
    for (int i = N - 1; i >= 0; --i) {
        double K_t = K - D_t[i];
        if (std::isnan(K_t)) return std::numeric_limits<double>::quiet_NaN();
        for (int j = 0; j <= i; ++j) {
            double v_next_up = V[i+1][j+1];
            double v_next_down = V[i+1][j];
             if (std::isnan(v_next_up) || std::isnan(v_next_down)) { return std::numeric_limits<double>::quiet_NaN(); }
            double continuation = disc_p_up * v_next_up + disc_p_down * v_next_down;

            double current_tilde_S = tilde_S[i][j];
             if (std::isnan(current_tilde_S)) { return std::numeric_limits<double>::quiet_NaN(); }
            double exercise = 0.0;
            if (option_type_str == "c")
                exercise = std::max(current_tilde_S - K_t, 0.0);
            else // "p"
                exercise = std::max(K_t - current_tilde_S, 0.0);

             if (std::isnan(continuation) || std::isnan(exercise)) { return std::numeric_limits<double>::quiet_NaN(); }
            V[i][j] = std::max(exercise, continuation);
        }
    } // End backward loop

    if (std::isnan(V[0][0])) { return std::numeric_limits<double>::quiet_NaN(); }
    return V[0][0]; // Return value at root
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
        double model_price = ska_model_option_cpp(S0, r, q, sigma, T, K, option_type,
                                                  div_times, div_cash, div_prop, N);

        #ifdef EXTRA_DEBUGS
        if (sigma == vol_lower || sigma == vol_upper) { // Only print for bounds to reduce noise
            std::cerr << std::fixed << std::setprecision(8)
                      << "    [Objective] sigma=" << sigma << ", model_price=" << model_price << std::endl;
        }
        #endif

        if (std::isnan(model_price)) {
             return std::numeric_limits<double>::quiet_NaN(); // Propagate NaN
        }
        return model_price - target_price;
    };

    #ifdef EXTRA_DEBUGS
    std::cerr << "IV Calc: Target=" << target_price << ", S0=" << S0 << ", K=" << K
              << ", T=" << T << ", Type=" << option_type << ", N=" << N
              << ", Bounds=[" << vol_lower << ", " << vol_upper << "]" << std::endl;
    #endif

    double f_lower = objective_func(vol_lower);
    double f_upper = objective_func(vol_upper);

    #ifdef EXTRA_DEBUGS
    std::cerr << std::fixed << std::setprecision(8)
              << "  Price(low_vol) = " << (f_lower + target_price) << ", f_lower = " << f_lower << std::endl;
    std::cerr << std::fixed << std::setprecision(8)
              << "  Price(high_vol)= " << (f_upper + target_price) << ", f_upper = " << f_upper << std::endl;
    #endif


    if (std::isnan(f_lower) || std::isnan(f_upper)) {
        #ifdef EXTRA_DEBUGS
        std::cerr << "  -> Pricing failed at bounds (NaN detected), returning NaN." << std::endl;
        #endif
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (f_lower * f_upper >= 0) {
        #ifdef EXTRA_DEBUGS
        std::cerr << "  -> Root not bracketed (f_lower * f_upper = " << (f_lower * f_upper) << "), checking tolerance..." << std::endl;
        #endif
        if (target_price > 1e-9 && std::abs(f_lower) < tol * target_price) {
            #ifdef EXTRA_DEBUGS
            std::cerr << "     -> Approx match at lower bound." << std::endl;
            #endif
            return vol_lower;
        }
        if (target_price > 1e-9 && std::abs(f_upper) < tol * target_price) {
            #ifdef EXTRA_DEBUGS
            std::cerr << "     -> Approx match at upper bound." << std::endl;
            #endif
            return vol_upper;
        }
        #ifdef EXTRA_DEBUGS
        std::cerr << "     -> No match within tolerance, returning NaN." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
        #endif
    }

    // If we reach here, root is bracketed
    try {
        #ifdef EXTRA_DEBUGS
        std::cerr << "  -> Root bracketed. Calling brentq..." << std::endl;
        #endif
        // Make sure brent.hpp still has its internal debug prints enabled if needed
        double result = RootFinding::brentq(objective_func, vol_lower, vol_upper, tol, 100);

        if (std::isnan(result)) {
             // This message will appear *after* the specific warning from brent.hpp
             std::cerr << "  Note: brentq solver returned NaN for Target=" << target_price
                       << ", K=" << K << ", Type=" << option_type << std::endl;
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