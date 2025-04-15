#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> // Include if using numpy directly later

// Include C++ declarations
#include "ska_pricer.hpp"
#include "implied_vol.hpp" 

namespace py = pybind11;

PYBIND11_MODULE(implied_vol_cpp_lib, m) {
    m.doc() = "High-performance C++ implementation of American option pricing (SKA) and implied volatility calculation";

    /////////////////////////////////////////////////////////////////
    // SKA stuff

    // Bind the single option pricer (useful for testing)
    m.def("ska_model_option", &ska_model_option,
          "Price American option using SKA binomial tree method (C++)",
          py::arg("S0"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("T"), py::arg("K"),
          py::arg("option_type"), // expects "c" or "p"
          py::arg("div_times"), py::arg("div_cash"), py::arg("div_prop"),
          py::arg("N")
    );

    // Bind the Synchronous Vectorized Pricer
    m.def("ska_model_option_sync", &ska_model_option_vectorized_sync,
          "Price multiple American options sequentially (C++)",
          py::arg("S0_arr"), py::arg("r_arr"), py::arg("q_arr"), py::arg("sigma_arr"),
          py::arg("T_arr"), py::arg("K_arr"), py::arg("option_type_arr"),
          py::arg("div_times"), py::arg("div_cash"), py::arg("div_prop"),
          py::arg("N")
    );

    // Bind the Asynchronous (Parallel) Vectorized Pricer
    m.def("ska_model_option_async", &ska_model_option_vectorized_async,
          "Price multiple American options in parallel (C++)",
          py::arg("S0_arr"), py::arg("r_arr"), py::arg("q_arr"), py::arg("sigma_arr"),
          py::arg("T_arr"), py::arg("K_arr"), py::arg("option_type_arr"),
          py::arg("div_times"), py::arg("div_cash"), py::arg("div_prop"),
          py::arg("N")
    );
    /////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////
    // Volatility stuff

    // Bind the vectorized implied volatility function (existing)
    m.def("vectorized_implied_volatility_parallel",
          &vectorized_implied_volatility_cpp,
          "Compute implied volatilities for multiple options in parallel",
          py::arg("target_prices"), py::arg("S_arr"), py::arg("r_arr"), py::arg("q_arr"),
          py::arg("T_arr"), py::arg("K_arr"), py::arg("option_types_arr"), py::arg("iv_initial_guess"),
          py::arg("div_times"), py::arg("div_prop"), py::arg("div_cash"),
          py::arg("N"), py::arg("vol_lower") = 1e-4, py::arg("vol_upper") = 5.0, py::arg("tol") = 1e-5
    );

    // Bind the single option pricer
    m.def("ska_model_option_native", &ska_model_option_native_cpp,
          "C++ implementation of SKA binomial tree pricer (Native version for Volatility)",
          py::arg("S0"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("T"), py::arg("K"),
          py::arg("option_type"), // expects "c" or "p"
          py::arg("div_times"), py::arg("div_cash"), py::arg("div_prop"),
          py::arg("N") // N is required, no default
    );

    // Bind the single implied volatility function (optional, non-parallel)
     m.def("implied_volatility_single", &implied_volatility_cpp,
           "Compute implied volatility for a single option (C++)",
           py::arg("target_price"), py::arg("S0"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("K"),
           py::arg("option_type"),
           py::arg("div_times"), py::arg("div_prop"), py::arg("div_cash"),
           py::arg("N"),
           py::arg("initial_guess") = 0.2,
           py::arg("vol_lower") = 1e-4,
           py::arg("vol_upper") = 5.0,
           py::arg("tol") = 1e-5
    );
     /////////////////////////////////////////////////////////////////


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}