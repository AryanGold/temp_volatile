#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> // Include if using numpy directly later

#include "implied_vol.hpp" // Include C++ declarations

namespace py = pybind11;

PYBIND11_MODULE(implied_vol_cpp_lib, m) {
    m.doc() = "High-performance C++ implementation of American option pricing (SKA) and implied volatility calculation";

    // Bind the vectorized implied volatility function (existing)
    m.def("vectorized_implied_volatility_parallel",
          &vectorized_implied_volatility_cpp,
          "Compute implied volatilities for multiple options in parallel",
          py::arg("target_prices"), py::arg("S_arr"), py::arg("r_arr"), py::arg("q_arr"),
          py::arg("T_arr"), py::arg("K_arr"), py::arg("option_types_arr"), py::arg("iv_initial_guess"),
          py::arg("div_times"), py::arg("div_prop"), py::arg("div_cash"),
          py::arg("N"), py::arg("vol_lower") = 1e-4, py::arg("vol_upper") = 5.0, py::arg("tol") = 1e-5
    );

    // --- Add Bindings for Debugging Comparison ---

    // Bind compute_fp
    m.def("compute_fp", &compute_fp, "C++ implementation of compute_fp",
          py::arg("t"), py::arg("mu"), py::arg("div_times"), py::arg("div_props"));

    // Bind compute_DT
    m.def("compute_DT", &compute_DT, "C++ implementation of compute_DT",
          py::arg("t"), py::arg("T"), py::arg("mu"),
          py::arg("div_times"), py::arg("div_props"), py::arg("div_cash"));

    // Bind the single option pricer
    m.def("ska_model_option", &ska_model_option_cpp,
          "C++ implementation of SKA binomial tree pricer",
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
    // --- End Bindings for Debugging Comparison ---


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}