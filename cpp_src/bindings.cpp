#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // Automatic conversions for std::vector, std::string etc.
#include <pybind11/numpy.h>     // Optional, but useful for direct NumPy interaction if needed later

#include "implied_vol.hpp"    // Include our C++ function declarations

namespace py = pybind11;

// Helper to maybe convert numpy arrays if needed, though stl.h often handles lists/vectors
// py::array_t<double> --> std::vector<double> is handled by pybind11/stl.h

PYBIND11_MODULE(implied_vol_cpp_lib, m) { // Module name matches setup.py
    m.doc() = "High-performance C++ implementation of American option pricing (SKA) and implied volatility calculation"; // Optional module docstring

    // Bind the vectorized implied volatility function
    m.def("vectorized_implied_volatility_parallel", // Function name exposed to Python
          &vectorized_implied_volatility_cpp,      // Pointer to the C++ function
          "Compute implied volatilities for multiple options in parallel", // Docstring
          // Argument specification using py::arg
          py::arg("target_prices"),
          py::arg("S_arr"),
          py::arg("r_arr"),
          py::arg("q_arr"),
          py::arg("T_arr"),
          py::arg("K_arr"),
          py::arg("option_types_arr"),
          py::arg("iv_initial_guess"),
          py::arg("div_times"),
          py::arg("div_prop"),
          py::arg("div_cash"),
          py::arg("N"),
          py::arg("vol_lower") = 1e-4, // Default values match C++
          py::arg("vol_upper") = 5.0,
          py::arg("tol") = 1e-5
    );

    // Optionally bind the single option pricer if needed for testing/direct use
    m.def("ska_model_option", &ska_model_option_cpp,
          "Price American option using SKA binomial tree method",
          py::arg("S0"), py::arg("r"), py::arg("q"), py::arg("sigma"), py::arg("T"), py::arg("K"),
          py::arg("option_type"), // expects "c" or "p"
          py::arg("div_times"), py::arg("div_cash"), py::arg("div_prop"),
          py::arg("N")
    );

    // Optionally bind the single implied volatility function (non-parallel)
     m.def("implied_volatility_single", &implied_volatility_cpp,
           "Compute implied volatility for a single option",
           py::arg("target_price"), py::arg("S0"), py::arg("r"), py::arg("q"), py::arg("T"), py::arg("K"),
           py::arg("option_type"),
           py::arg("div_times"), py::arg("div_prop"), py::arg("div_cash"),
           py::arg("N"),
           py::arg("initial_guess") = 0.2, // Keep arg even if unused by Brentq
           py::arg("vol_lower") = 1e-4,
           py::arg("vol_upper") = 5.0,
           py::arg("tol") = 1e-5
    );

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
