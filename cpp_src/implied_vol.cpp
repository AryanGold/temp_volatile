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

// --- Helper Function Implementations ---

double compute_fp(double t, double mu,
                  const std::vector<double>& div_times,
                  const std::vector<double>& div_props) {
    double exponent = mu * t;
    double product = 1.0;
    size_t n_div = div_times.size();
    for (size_t j = 0; j < n_div; ++j) {
        if (div_times[j] > 0 && div_times[j] <= t) { // Ensure time > 0
            product *= (1.0 - div_props[j]);
        }
    }
    return std::exp(exponent) * product;
}


double compute_DT(double t, double T, double mu,
                  const std::vector<double>& div_times,
                  const std::vector<double>& div_props,
                  const std::vector<double>& div_cash) {
    double fp_t = compute_fp(t, mu, div_times, div_props);
    double sum1 = 0.0;
    double sum2 = 0.0;
    size_t n_div = div_times.size();

    if (n_div != div_props.size() || n_div != div_cash.size()) {
        throw std::runtime_error("Dividend vector sizes mismatch in compute_DT");
    }

    for (size_t i = 0; i < n_div; ++i) {
        double ti = div_times[i];
        double di = div_cash[i];
        // delta_i (div_props[i]) is used indirectly via compute_fp

        if (ti <= 0.0) continue; // Skip non-positive dividend times

        double fp_ti = compute_fp(ti, mu, div_times, div_props);
        if (fp_ti == 0.0) {
             // Avoid division by zero, return NaN or handle appropriately
             // This might happen if prop dividend is 1.0
             return std::numeric_limits<double>::quiet_NaN();
        }

        if (ti > t && ti <= T) {
            sum1 += di / fp_ti;
        }
        if (ti <= T) { // Python logic was `if ti <= T:` for sum2
            if (T == 0.0) { // Avoid division by zero if T=0
                 // Decide behavior: maybe sum2 should be 0 if T=0?
                 // Original python code would divide by zero.
                 // Let's assume T > 0 for this term. If T can be 0, adjust logic.
                 if(T > std::numeric_limits<double>::epsilon()) {
                    sum2 += (ti / T) * (di / fp_ti);
                 }
            } else {
                 sum2 += (ti / T) * (di / fp_ti);
            }
        }
    }
    return fp_t * (sum1 - sum2);
}


// --- Core Option Pricer ---

double ska_model_option_cpp(
    double S0, double r, double q, double sigma, double T, double K,
    const std::string& option_type_str,
    const std::vector<double>& div_times,
    const std::vector<double>& div_cash,
    const std::vector<double>& div_prop,
    int N)
{
    if (T <= 0) { // Handle zero or negative maturity
       double payoff = 0.0;
       if (option_type_str == "c") {
            payoff = std::max(S0 - K, 0.0);
       } else if (option_type_str == "p") {
            payoff = std::max(K - S0, 0.0);
       } else {
           throw std::invalid_argument("Invalid option type. Use 'c' or 'p'.");
       }
       // No discounting needed if T=0
       return payoff;
    }
     if (N <= 0) {
        throw std::invalid_argument("Number of steps N must be positive.");
    }
     if (sigma <= 0) {
        // Handle non-positive sigma - maybe return intrinsic value or BS price with tiny sigma?
        // For simplicity, let's treat it like T=0 case for now or throw
         double payoff = 0.0;
         if (option_type_str == "c") {
             payoff = std::max(S0 - K, 0.0);
         } else {
             payoff = std::max(K - S0, 0.0);
         }
         // Need to discount back if T > 0
         return payoff * std::exp(-r * T);
     }


    double mu = r - q;
    double dt = T / static_cast<double>(N);
    double sigma_sqrt_dt = sigma * std::sqrt(dt);
    double u = std::exp(sigma_sqrt_dt);
    double d = 1.0 / u;
    double exp_mu_dt = std::exp(mu * dt);

    // Check for arbitrage condition for p
    if (d >= exp_mu_dt || exp_mu_dt >= u) {
        // This indicates potential instability or arbitrage in parameters
        // Return NaN or throw an error
        // For example, sigma might be too small relative to mu*dt
        // Let's return NaN for now, as the formula for p breaks down.
        return std::numeric_limits<double>::quiet_NaN();
        // Or potentially adjust u/d slightly if very close, but NaN is safer signal
    }

    double p = (exp_mu_dt - d) / (u - d);
    double discount = std::exp(-r * dt);
    double p_disc = p * discount;
    double one_minus_p_disc = (1.0 - p) * discount;

    // Calculate D_t shifts at each time step node
    std::vector<double> D_t(N + 1);
    for (int i = 0; i <= N; ++i) {
        double current_t = static_cast<double>(i) * dt;
        D_t[i] = compute_DT(current_t, T, mu, div_times, div_prop, div_cash);
         if (std::isnan(D_t[i])) {
             // Propagate NaN if compute_DT failed
             return std::numeric_limits<double>::quiet_NaN();
         }
    }

    // Initialize tree for tilde_S = S - D_t
    // Use 1D vector for space efficiency in backward pass
    std::vector<double> tilde_S_layer(N + 1);
    tilde_S_layer[0] = S0 - D_t[0]; // Initial shifted stock price

    // Forward pass to get terminal tilde_S values
    // We only strictly need the final layer for the standard tree payoff
    // Let's compute it directly without storing the full tilde_S tree

    std::vector<double> V = tilde_S_layer; // Use V to store current layer values, starting with tilde_S[0,0]


    for (int i = 1; i <= N; ++i) {
        // Calculate next layer of tilde_S values
        // Need N+1 size for layer i
        std::vector<double> next_V(i + 1);
        next_V[0] = V[0] * d; // Down move from lowest node
        for (int j = 1; j <= i; ++j) {
             next_V[j] = V[j-1] * u; // Up move from node below
        }
        V = next_V; // Update V to the current layer's tilde_S values (before prop div)

        // Apply proportional dividend adjustment if any occur *within* this step
        double t_start = static_cast<double>(i - 1) * dt;
        double t_end = static_cast<double>(i) * dt;
        for (size_t k = 0; k < div_times.size(); ++k) {
            if (div_times[k] > t_start && div_times[k] <= t_end) {
                if (div_prop[k] > 0.0) { // Only apply if prop div > 0
                    for (int j = 0; j <= i; ++j) {
                         V[j] *= (1.0 - div_prop[k]);
                    }
                }
            }
        }
    }
    // At this point, V holds the tilde_S values at maturity (time N*dt = T)

    // Calculate terminal option values (V at maturity)
    double K_T = K - D_t[N]; // Adjusted strike at maturity
    for (int j = 0; j <= N; ++j) {
        if (option_type_str == "c") {
            V[j] = std::max(V[j] - K_T, 0.0);
        } else { // Assume "p"
            V[j] = std::max(K_T - V[j], 0.0);
        }
    }

    // Backward induction (Option valuation)
    // We need the tilde_S tree structure implicitly or explicitly here.
    // Let's re-calculate tilde_S values backwards or store them.
    // Storing is simpler to code but uses more memory.
    // Let's try recalculating needed tilde_S nodes on the fly during backward pass.
    // To do this efficiently, we need the proportional dividend multipliers applied correctly.

    // Re-thinking: The forward pass calculated terminal tilde_S values.
    // The backward pass computes option values. Let's reuse V for option values.

    // We need tilde_S values at each step (i, j) for the early exercise check.
    // Let's store the whole tilde_S tree first, then do backward pass.

    std::vector<std::vector<double>> tilde_S(N + 1, std::vector<double>(N + 1));
    tilde_S[0][0] = S0 - D_t[0];

    for (int i = 1; i <= N; ++i) {
        double t_start = static_cast<double>(i - 1) * dt;
        double t_end = static_cast<double>(i) * dt;
        double prop_div_mult = 1.0;
        for (size_t k = 0; k < div_times.size(); ++k) {
            if (div_times[k] > t_start && div_times[k] <= t_end) {
                prop_div_mult *= (1.0 - div_prop[k]);
            }
        }

        tilde_S[i][0] = tilde_S[i-1][0] * d * prop_div_mult;
        for (int j = 1; j <= i; ++j) {
            tilde_S[i][j] = tilde_S[i-1][j-1] * u * prop_div_mult;
        }
    }

    // V already holds terminal option values based on tilde_S[N]


    // Backward induction loop
    for (int i = N - 1; i >= 0; --i) {
        double K_t = K - D_t[i]; // Adjusted strike at time i*dt
        // V currently holds values for layer i+1. We compute layer i.
        std::vector<double> current_V(i+1); // Temporary storage for layer i values
        for (int j = 0; j <= i; ++j) {
            double continuation = p_disc * V[j+1] + one_minus_p_disc * V[j];
            double exercise = 0.0;
            if (option_type_str == "c") {
                exercise = std::max(tilde_S[i][j] - K_t, 0.0);
            } else { // Assume "p"
                exercise = std::max(K_t - tilde_S[i][j], 0.0);
            }
            current_V[j] = std::max(exercise, continuation);
        }
        // Resize V and copy results (or use pointer swap for efficiency if V was dynamically sized)
        // Since V was sized N+1, we can just overwrite the needed part if careful,
        // but creating a new vector `current_V` is safer.
        V.resize(i + 1); // Adjust V to the size needed for the next iteration
        V = current_V;   // Copy results back to V for the next step i-1
    }

    // Final result is at the root of the option value tree
    return V[0];
}


// --- Implied Volatility Solver ---

double implied_volatility_cpp(
    double target_price, double S0, double r, double q, double T, double K,
    const std::string& option_type,
    const std::vector<double>& div_times,
    const std::vector<double>& div_prop,
    const std::vector<double>& div_cash,
    int N,
    double initial_guess, // Not used currently
    double vol_lower, double vol_upper, double tol)
{
    (void)initial_guess;
    
    // Define the objective function for the root finder
    auto objective_func = [&](double sigma) {
        if (sigma <= 0) return std::numeric_limits<double>::max(); // Avoid non-positive sigma
        double model_price = ska_model_option_cpp(S0, r, q, sigma, T, K, option_type,
                                                  div_times, div_cash, div_prop, N);
        if (std::isnan(model_price)) {
            // Handle pricing failure (e.g., instability)
            // Return a large number or NaN to signal failure to Brentq
             return std::numeric_limits<double>::quiet_NaN(); // Or a large signed value if Brentq needs it
        }
        return model_price - target_price;
    };

    // Check if bounds produce results of opposite signs
    double f_lower = objective_func(vol_lower);
    double f_upper = objective_func(vol_upper);

    if (std::isnan(f_lower) || std::isnan(f_upper)) {
         // Pricing failed at bounds, cannot proceed reliably
         return std::numeric_limits<double>::quiet_NaN();
    }

    if (f_lower * f_upper >= 0) {
        // Target price might be outside the possible range for given bounds.
        // Could be arbitrage, or bounds are too narrow/wide.
        // Check if target is very close to bound prices (within tolerance)
         if (std::abs(f_lower) < tol * target_price) return vol_lower; // Approx match at lower bound
         if (std::abs(f_upper) < tol * target_price) return vol_upper; // Approx match at upper bound

        // Otherwise, root not bracketed or doesn't exist in range
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Use Brent's method from brent.hpp
    try {
        return RootFinding::brentq(objective_func, vol_lower, vol_upper, tol, 100);
    } catch (const std::exception& e) {
        // Handle exceptions from brentq if it throws (e.g., max iterations)
        // Our implementation returns NaN on failure.
        return std::numeric_limits<double>::quiet_NaN();
    }
     // Catch potential issues within the lambda itself?
     // Brentq should handle function evaluation errors if objective returns NaN.
     // The return from brentq will be NaN if it fails or objective returns NaN during search.
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
    const std::vector<double>& iv_initial_guess, // Keep for potential future use (e.g. Newton)
    const std::vector<double>& div_times, // Common
    const std::vector<double>& div_prop,  // Common
    const std::vector<double>& div_cash,  // Common
    int N,                              // Common
    double vol_lower,                   // Common
    double vol_upper,                   // Common
    double tol)                         // Common
{
    size_t n_options = target_prices.size();
    if (S_arr.size() != n_options || r_arr.size() != n_options || q_arr.size() != n_options ||
        T_arr.size() != n_options || K_arr.size() != n_options || option_types_arr.size() != n_options ||
        iv_initial_guess.size() != n_options)
    {
        throw std::runtime_error("Input array sizes do not match target_prices size.");
    }

    std::vector<double> ivs(n_options);

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 2; // Default to 2 if hardware_concurrency fails
    }
    // Consider using slightly fewer threads than cores if tasks are CPU-bound
    // num_threads = std::max(1u, num_threads - 1);

    // Explicitly construct the pool ---
    // Use size_t for constructor argument as per BS::thread_pool docs
    BS::thread_pool pool(static_cast<std::size_t>(num_threads));

    std::vector<std::future<double>> futures;
    futures.reserve(n_options);

    // Submit tasks to the thread pool
    for (size_t i = 0; i < n_options; ++i) {
        // --- Check if pool.submit exists ---
        // This is a compile-time check basically. If it compiles, submit exists.
        // The error message suggests it doesn't, which is strange.
        // --- End check ---

        futures.emplace_back(pool.submit( 
            // Lambda function capturing necessary variables
            [=, &div_times, &div_prop, &div_cash]() -> double {
                // --- Suppress unused initial_guess warning inside lambda's scope ---
                // If you decide to keep the parameter but not use it yet
                 (void)iv_initial_guess[i];
                // --- End suppression ---

                return implied_volatility_cpp(
                    target_prices[i], S_arr[i], r_arr[i], q_arr[i], T_arr[i], K_arr[i],
                    option_types_arr[i], div_times, div_prop, div_cash, N,
                    iv_initial_guess[i], // Still passing it
                    vol_lower, vol_upper, tol);
            }
        ));
    }

    // Retrieve results from futures
    for (size_t i = 0; i < n_options; ++i) {
        try {
            ivs[i] = futures[i].get(); // get() will wait for the task to complete
        } catch (const std::exception& e) {
            // Handle exceptions thrown by the task if any
            // For now, store NaN to indicate failure for this option
            ivs[i] = std::numeric_limits<double>::quiet_NaN();
            // Optionally log the error: std::cerr << "Error calculating IV for option " << i << ": " << e.what() << std::endl;
        }
    }

    return ivs;
}
