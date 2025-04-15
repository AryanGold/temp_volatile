#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <numeric> // For std::accumulate if needed
#include <algorithm> // For std::max

// Forward declarations 

// Helper function: compute_fp
double compute_fp_native(double t, double mu,
                  const std::vector<double>& div_times,
                  const std::vector<double>& div_props);

// Helper function: compute_DT
double compute_DT_native(double t, double T, double mu,
                  const std::vector<double>& div_times,
                  const std::vector<double>& div_props,
                  const std::vector<double>& div_cash);

// Core American Option Pricer using SKA method
double ska_model_option_native_cpp(
    double S0, double r, double q, double sigma, double T, double K,
    const std::string& option_type_str, // Use string "c" or "p"
    const std::vector<double>& div_times,
    const std::vector<double>& div_cash,
    const std::vector<double>& div_prop,
    int N);

// Implied Volatility Solver (using Brentq)
double implied_volatility_cpp(
    double target_price, double S0, double r, double q, double T, double K,
    const std::string& option_type, // "c" or "p"
    const std::vector<double>& div_times,
    const std::vector<double>& div_prop,
    const std::vector<double>& div_cash,
    int N,
    double initial_guess, // Not used by Brentq, but kept for interface consistency maybe?
    double vol_lower = 1e-4, // Adjusted lower bound
    double vol_upper = 5.0,
    double tol = 1e-5); // Adjusted tolerance


// Vectorized and Parallelized Implied Volatility Calculation
std::vector<double> vectorized_implied_volatility_cpp(
    const std::vector<double>& target_prices,
    const std::vector<double>& S_arr,
    const std::vector<double>& r_arr,
    const std::vector<double>& q_arr,
    const std::vector<double>& T_arr,
    const std::vector<double>& K_arr,
    const std::vector<std::string>& option_types_arr,
    const std::vector<double>& iv_initial_guess, // Not used by Brentq, but here for API consistency
    const std::vector<double>& div_times, // Common dividend schedule
    const std::vector<double>& div_prop,  // Common dividend schedule
    const std::vector<double>& div_cash,  // Common dividend schedule
    int N,
    double vol_lower = 1e-4,
    double vol_upper = 5.0,
    double tol = 1e-5);
