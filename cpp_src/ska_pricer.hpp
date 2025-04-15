#pragma once

#include <vector>
#include <string>

// --- Helper Function Declarations (needed by ska_model_option) ---

double compute_fp(double t, double mu,
                  const std::vector<double>& div_times,
                  const std::vector<double>& div_props);

double compute_DT(double t, double T, double mu,
                  const std::vector<double>& div_times,
                  const std::vector<double>& div_props,
                  const std::vector<double>& div_cash);

// --- Core Pricer Function Declaration ---

double ska_model_option(double S0, double r, double q, double sigma, double T, double K,
                        const std::string& option_type,
                        const std::vector<double>& div_times,
                        const std::vector<double>& div_cash,
                        const std::vector<double>& div_props,
                        int N); // Removed smooth and default N for library use

// --- Vectorized Function Declarations ---

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
    int N);

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
    int N);
