#include "ska_pricer.hpp"
#include "BS_thread_pool.hpp" // Include thread pool for async version

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <future>
#include <thread>
#include <numeric>   // Required for std::accumulate if used later
#include <algorithm> // Required for std::max

// --- Helper Function Implementations (Copied from pricer.cpp / previous working version) ---

// compute_fp: Matches pricer.cpp (no > 0 check)
double compute_fp(double t, double mu,
                  const std::vector<double>& div_times,
                  const std::vector<double>& div_props) {
    double exponent = mu * t;
    double product = 1.0;
    size_t n_div = div_times.size();
    for (size_t j = 0; j < n_div; ++j) {
        if (div_times[j] <= t) {
             if (div_props[j] >= 1.0) {
                 // Stock value becomes zero or negative
                 product = 0.0;
                 break;
             } else {
                product *= (1.0 - div_props[j]);
             }
        }
    }
    if (product <= 0) {
        return 0.0; // Return 0 if product became non-positive
    }
    return std::exp(exponent) * product;
}

// compute_DT: Matches pricer.cpp
double compute_DT(double t, double T, double mu,
                  const std::vector<double>& div_times,
                  const std::vector<double>& div_props,
                  const std::vector<double>& div_cash) {

    if (T <= 0) return 0.0; // Avoid division by zero if T=0

    double fp_t = compute_fp(t, mu, div_times, div_props);
     if (std::isnan(fp_t) || fp_t == 0.0) {
         // If fp_t is invalid, assume shift is 0 or propagate NaN?
         // Let's return 0 for the shift, assuming invalid FP means no value.
         return 0.0;
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

        double fp_ti = compute_fp(ti, mu, div_times, div_props);

        if (std::isnan(fp_ti) || fp_ti == 0.0) {
            // Skip this dividend's contribution if fp_ti is invalid
            continue;
        }

        double term = di / fp_ti;

        if (t < ti && ti <= T) { // Condition for sum1
            sum1 += term;
        }
        if (ti <= T) { // Condition for sum2
            sum2 += (ti / T) * term;
        }
    }
    return fp_t * (sum1 - sum2);
}


// --- Core Pricer Function Implementation (Copied from pricer.cpp) ---
// Using the flattened vector approach from pricer.cpp as requested,
// but NOTE: our previous debugging showed the vector<vector> version matched Python better.
// If this causes issues, revert to the vector<vector> version from previous steps.
double ska_model_option(double S0, double r, double q, double sigma, double T, double K,
                        const std::string& option_type, // Renamed from option_type_str
                        const std::vector<double>& div_times,
                        const std::vector<double>& div_cash,
                        const std::vector<double>& div_props,
                        int N) // Removed smooth parameter, N is required
{
    // Basic checks
     if (T <= 0.0) {
        double payoff = 0.0;
        if (option_type == "c") { payoff = std::max(S0 - K, 0.0); }
        else if (option_type == "p") { payoff = std::max(K - S0, 0.0); }
        else { throw std::invalid_argument("Invalid option type. Use 'c' or 'p'."); }
        return payoff;
     }
     if (N <= 0) {
        throw std::invalid_argument("Number of steps N must be positive.");
     }
     // Add sigma check - needed for stability
     if (sigma <= 0.0) {
         // Calculate zero-volatility price (discounted forward payoff)
         double mu_calc = r - q;
         double fp_maturity = compute_fp(T, mu_calc, div_times, div_props);
         if (std::isnan(fp_maturity)) return std::numeric_limits<double>::quiet_NaN();
         double sum_div_value = 0.0;
         size_t n_div_fwd = div_times.size();
         for (size_t j = 0; j < n_div_fwd; ++j) {
             if (div_times[j] > 0 && div_times[j] <= T) {
                 double fp_j = compute_fp(div_times[j], mu_calc, div_times, div_props);
                 if (fp_j != 0.0 && !std::isnan(fp_j)) { sum_div_value += div_cash[j] / fp_j; }
                 else { return std::numeric_limits<double>::quiet_NaN(); }
             }
         }
         double forward_price = fp_maturity * (S0 - sum_div_value);
         if (std::isnan(forward_price)) return std::numeric_limits<double>::quiet_NaN();
         double payoff_at_maturity = 0.0;
         if (option_type == "c") { payoff_at_maturity = std::max(forward_price - K, 0.0); }
         else { payoff_at_maturity = std::max(K - forward_price, 0.0); }
         return payoff_at_maturity * std::exp(-r * T);
     }

    // --- Start of logic adapted from pricer.cpp ---
    double mu = r - q;
    double dt = T / static_cast<double>(N);
    double sigma_sqrt_dt = sigma * std::sqrt(dt);
    double u = std::exp(sigma_sqrt_dt);
    double d = 1.0 / u;
    double u_minus_d = u - d;

    // Check if u and d are too close
    if (std::abs(u_minus_d) < std::numeric_limits<double>::epsilon() * 100) {
        // Fallback to zero-vol price
         double mu_calc = r - q;
         double fp_maturity = compute_fp(T, mu_calc, div_times, div_props);
         if (std::isnan(fp_maturity)) return std::numeric_limits<double>::quiet_NaN();
         double sum_div_value = 0.0;
         size_t n_div_fwd = div_times.size();
         for (size_t j = 0; j < n_div_fwd; ++j) { /* ... calculate sum_div_value ... */
             if (div_times[j] > 0 && div_times[j] <= T) {
                 double fp_j = compute_fp(div_times[j], mu_calc, div_times, div_props);
                 if (fp_j != 0.0 && !std::isnan(fp_j)) { sum_div_value += div_cash[j] / fp_j; }
                 else { return std::numeric_limits<double>::quiet_NaN(); }
             }
         }
         double forward_price = fp_maturity * (S0 - sum_div_value);
         if (std::isnan(forward_price)) return std::numeric_limits<double>::quiet_NaN();
         double payoff_at_maturity = 0.0;
         if (option_type == "c") { payoff_at_maturity = std::max(forward_price - K, 0.0); }
         else { payoff_at_maturity = std::max(K - forward_price, 0.0); }
         return payoff_at_maturity * std::exp(-r * T);
    }

    double p = (std::exp(mu * dt) - d) / u_minus_d;
    double discount = std::exp(-r * dt);

    // --- Clamp p (essential for stability) ---
    if (p < 0.0 || p > 1.0) {
        p = std::max(0.0, std::min(1.0, p));
    }
    // --- End clamp p ---

    // Compute D_t shifts
    std::vector<double> D_t(N + 1, 0.0);
    for (int i = 0; i <= N; ++i) {
        double t_i = static_cast<double>(i) * dt;
        D_t[i] = compute_DT(t_i, T, mu, div_times, div_props, div_cash);
        if (std::isnan(D_t[i])) { return std::numeric_limits<double>::quiet_NaN(); }
    }

    // Allocate flattened 1D array for tilde_S
    std::vector<double> tilde_S((N + 1) * (N + 1), 0.0);
    tilde_S[0] = S0 - D_t[0];
    if (std::isnan(tilde_S[0])) { return std::numeric_limits<double>::quiet_NaN(); }

    // Build the binomial tree (Forward pass).
    for (int i = 1; i <= N; ++i) {
        int row_start = i * (N + 1);
        int prev_row_start = (i - 1) * (N + 1);
        tilde_S[row_start] = tilde_S[prev_row_start] * d;
        for (int j = 1; j <= i; ++j)
            tilde_S[row_start + j] = tilde_S[prev_row_start + (j - 1)] * u;

        double t_lower = (i - 1) * dt;
        double t_upper = i * dt;
        for (size_t k = 0; k < div_times.size(); ++k) {
            if (div_times[k] > t_lower && div_times[k] <= t_upper) {
                double prop_mult = (1.0 - div_props[k]);
                if (prop_mult < 0) { prop_mult = 0.0; }
                for (int j = 0; j <= i; ++j) {
                    tilde_S[row_start + j] *= prop_mult;
                }
            }
        }
         // Optional: Add NaN check here if needed
    }

    // Backward induction.
    std::vector<double> V((N + 1) * (N + 1), 0.0);
    double K_T = K - D_t[N];
    if (std::isnan(K_T)) return std::numeric_limits<double>::quiet_NaN();

    int final_row_idx = N * (N + 1);
    for (int j = 0; j <= N; ++j) {
        double ST = tilde_S[final_row_idx + j];
        if (std::isnan(ST)) return std::numeric_limits<double>::quiet_NaN();
        if (option_type == "c")
            V[final_row_idx + j] = std::max(ST - K_T, 0.0);
        else
            V[final_row_idx + j] = std::max(K_T - ST, 0.0);
    }

    double p_up = p;
    double p_down = 1.0 - p;
    double disc_p_up = discount * p_up;
    double disc_p_down = discount * p_down;

    for (int i = N - 1; i >= 0; --i) {
        int row_start = i * (N + 1);
        int next_row_start = (i + 1) * (N + 1);
        double K_t = K - D_t[i];
        if (std::isnan(K_t)) return std::numeric_limits<double>::quiet_NaN();

        for (int j = 0; j <= i; ++j) {
            double v_next_up = V[next_row_start + (j + 1)];
            double v_next_down = V[next_row_start + j];
            if (std::isnan(v_next_up) || std::isnan(v_next_down)) { return std::numeric_limits<double>::quiet_NaN(); }
            double continuation = disc_p_up * v_next_up + disc_p_down * v_next_down;

            double current_tilde_S = tilde_S[row_start + j];
            if (std::isnan(current_tilde_S)) { return std::numeric_limits<double>::quiet_NaN(); }
            double exercise = 0.0;
            if (option_type == "c")
                exercise = std::max(current_tilde_S - K_t, 0.0);
            else
                exercise = std::max(K_t - current_tilde_S, 0.0);

            if (std::isnan(continuation) || std::isnan(exercise)) { return std::numeric_limits<double>::quiet_NaN(); }
            V[row_start + j] = std::max(exercise, continuation);
        }
    }

    if (std::isnan(V[0])) { return std::numeric_limits<double>::quiet_NaN(); }
    return V[0];
    // --- End of logic adapted from pricer.cpp ---
}


// --- Vectorized Function Implementations ---

// Synchronous (Sequential) Vectorized Pricer
std::vector<double> ska_model_option_vectorized_sync(
    const std::vector<double>& S0_arr,
    const std::vector<double>& r_arr,
    const std::vector<double>& q_arr,
    const std::vector<double>& sigma_arr,
    const std::vector<double>& T_arr,
    const std::vector<double>& K_arr,
    const std::vector<std::string>& option_type_arr,
    // Common parameters
    const std::vector<double>& div_times,
    const std::vector<double>& div_cash,
    const std::vector<double>& div_props,
    int N)
{
    size_t n_options = S0_arr.size();
    // Basic size validation
    if (r_arr.size() != n_options || q_arr.size() != n_options || sigma_arr.size() != n_options ||
        T_arr.size() != n_options || K_arr.size() != n_options || option_type_arr.size() != n_options)
    {
        throw std::runtime_error("Input array sizes do not match in ska_model_option_vectorized_sync.");
    }

    std::vector<double> results(n_options);
    for (size_t i = 0; i < n_options; ++i) {
        try {
            results[i] = ska_model_option(
                S0_arr[i], r_arr[i], q_arr[i], sigma_arr[i], T_arr[i], K_arr[i],
                option_type_arr[i],
                div_times, div_cash, div_props, N
            );
        } catch (const std::exception& e) {
            // Handle potential errors from the single pricer (e.g., invalid N)
            // Or just let it propagate depending on desired behavior
             results[i] = std::numeric_limits<double>::quiet_NaN(); // Store NaN on error
             // Optionally print error: std::cerr << "Error pricing option " << i << ": " << e.what() << std::endl;
        }
         // Check for NaN result even if no exception occurred
         if (std::isnan(results[i])) {
              // Optionally print warning: std::cerr << "Warning: NaN result for option " << i << std::endl;
         }
    }
    return results;
}

// Asynchronous (Parallel) Vectorized Pricer
std::vector<double> ska_model_option_vectorized_async(
    const std::vector<double>& S0_arr,
    const std::vector<double>& r_arr,
    const std::vector<double>& q_arr,
    const std::vector<double>& sigma_arr,
    const std::vector<double>& T_arr,
    const std::vector<double>& K_arr,
    const std::vector<std::string>& option_type_arr,
    // Common parameters
    const std::vector<double>& div_times,
    const std::vector<double>& div_cash,
    const std::vector<double>& div_props,
    int N)
{
    size_t n_options = S0_arr.size();
    // Basic size validation
    if (r_arr.size() != n_options || q_arr.size() != n_options || sigma_arr.size() != n_options ||
        T_arr.size() != n_options || K_arr.size() != n_options || option_type_arr.size() != n_options)
    {
        throw std::runtime_error("Input array sizes do not match in ska_model_option_vectorized_async.");
    }

    std::vector<double> results(n_options);
    std::vector<std::future<double>> futures;
    futures.reserve(n_options);

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency() - 1;
    if (num_threads < 2) { num_threads = 2; }
    // You might want to limit threads further, e.g., std::min(num_threads, 8u)
    BS::thread_pool pool(static_cast<std::size_t>(num_threads));

    // Submit tasks
    for (size_t i = 0; i < n_options; ++i) {
        futures.emplace_back(pool.submit_task(
            // Capture common vectors by reference, others by value
            [=, &div_times, &div_cash, &div_props]() -> double {
                // No try-catch here, let exception propagate to future.get()
                // Or add try-catch and return NaN if needed inside lambda
                return ska_model_option(
                    S0_arr[i], r_arr[i], q_arr[i], sigma_arr[i], T_arr[i], K_arr[i],
                    option_type_arr[i],
                    div_times, div_cash, div_props, N
                );
            }
        ));
    }

    // Retrieve results
    for (size_t i = 0; i < n_options; ++i) {
        try {
            results[i] = futures[i].get();
             // Check for NaN result even if no exception occurred
             if (std::isnan(results[i])) {
                  // Optionally print warning: std::cerr << "Warning: NaN result for option " << i << std::endl;
             }
        } catch (const std::exception& e) {
            // Handle exceptions thrown by the task (e.g., from ska_model_option)
            results[i] = std::numeric_limits<double>::quiet_NaN(); // Store NaN on error
            // Optionally print error: std::cerr << "Error pricing option " << i << ": " << e.what() << std::endl;
        }
    }

    return results;
}
