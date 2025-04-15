#ifndef BRENT_HPP
#define BRENT_HPP

#include <cmath>
#include <limits>
#include <functional>
#include <stdexcept>
#include <algorithm> // for std::max, std::min
#include <iostream>  // For debugging output
#include <iomanip>   // For std::setprecision

namespace RootFinding {

// Basic Brent's method implementation with debugging
template <typename Func>
double brentq(Func f, double xa, double xb, double tol = 1e-5, int max_iter = 100) {
    // --- Debug Setup ---
    const bool enable_brentq_debug = true; // Set to false to disable debug prints
    auto print_debug = [&](const std::string& msg) {
        if (enable_brentq_debug) {
            std::cerr << "  [Brentq Debug] " << msg << std::endl;
        }
    };
    auto print_state = [&](int iter, double a, double b, double c, double d, double e, double fa, double fb, double fc) {
         if (enable_brentq_debug) {
             std::cerr << std::fixed << std::setprecision(10)
                       << "  [Brentq State Iter " << std::setw(3) << iter << "] "
                       << "a=" << a << ", b=" << b << ", c=" << c
                       << ", d=" << d << ", e=" << e << "\n"
                       << "                 " // Alignment
                       << "fa=" << fa << ", fb=" << fb << ", fc=" << fc
                       << std::endl;
         }
    };
    // --- End Debug Setup ---

    print_debug("Starting Brentq...");
    print_debug("Initial interval: [" + std::to_string(xa) + ", " + std::to_string(xb) + "]");

    double a = xa;
    double b = xb;
    double fa = f(a);
    double fb = f(b);

    print_debug("f(a=" + std::to_string(a) + ") = " + std::to_string(fa));
    print_debug("f(b=" + std::to_string(b) + ") = " + std::to_string(fb));


    if (std::isnan(fa) || std::isnan(fb)) {
        print_debug("ERROR: f(a) or f(b) is NaN at start.");
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (fa * fb >= 0.0) {
        print_debug("ERROR: Root not bracketed at start (fa * fb >= 0).");
        // Check tolerance near bounds
        if (std::abs(fa) < tol) { print_debug("Close match at a."); return a; }
        if (std::abs(fb) < tol) { print_debug("Close match at b."); return b; }
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Ensure |f(a)| >= |f(b)|
    if (std::abs(fa) < std::abs(fb)) {
        print_debug("Swapping a and b initially.");
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;     // c starts as the point *not* containing the best root estimate 'b'
    double fc = fa;
    double d = b - a; // Stores current step size (or previous step in some variants)
    double e = d;     // Stores previous step size delta

    print_debug("Initialization complete.");

    for (int iter = 0; iter < max_iter; ++iter) {
        print_state(iter, a, b, c, d, e, fa, fb, fc);

        // Check for convergence: interval size or f(b) near zero
        double tol1 = 2.0 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * tol;
        double xm = 0.5 * (c - b); // Midpoint of current bracket [b, c] or [c, b]

        if (std::abs(xm) <= tol1 || fb == 0.0) {
            print_debug("Convergence criteria met: |xm|=" + std::to_string(std::abs(xm)) + " <= tol1=" + std::to_string(tol1) + " or fb=" + std::to_string(fb) + " == 0.");
            return b; // Found root
        }

        // Is interpolation possible?
        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            print_debug("Attempting interpolation (e=" + std::to_string(e) + ", fa=" + std::to_string(fa) + ", fb=" + std::to_string(fb) + ")");
            double S = fb / fa; // Ratio for interpolation formulas
            double P, Q;

            if (a == c) { // Linear interpolation (secant method)
                print_debug("  Using Linear Interpolation (Secant)");
                P = 2.0 * xm * S;
                Q = 1.0 - S;
            } else { // Inverse quadratic interpolation
                 print_debug("  Using Inverse Quadratic Interpolation");
                double R = fb / fc; // Need fc != 0
                // Check for fc == 0 before division
                if (std::abs(fc) < std::numeric_limits<double>::epsilon()) {
                     print_debug("    fc is near zero, falling back to linear interpolation.");
                     P = 2.0 * xm * S;
                     Q = 1.0 - S;
                } else {
                    Q = fa / fc; // Need fc != 0
                    P = S * (2.0 * xm * Q * (Q - R) - (b - a) * (R - 1.0));
                    Q = (Q - 1.0) * (R - 1.0) * (S - 1.0);
                }
            }

            // Adjust sign: P/Q should represent the step delta
            if (P > 0.0) Q = -Q;
            P = std::abs(P);

            // Check if interpolation step is acceptable
            double min1 = 3.0 * xm * Q - std::abs(tol1 * Q);
            double min2 = std::abs(e * Q); // Compare with previous step size 'e'

            print_debug("  Interpolation check: P=" + std::to_string(P) + ", Q=" + std::to_string(Q) + ", min1=" + std::to_string(min1) + ", min2=" + std::to_string(min2));

            if (2.0 * P < std::min(min1, min2)) {
                // Accept interpolation step
                print_debug("  Interpolation step accepted.");
                e = d; // Store previous step delta
                d = P / Q; // Calculate the step to take
            } else {
                // Interpolation failed, use bisection
                print_debug("  Interpolation step rejected, using bisection.");
                d = xm;
                e = d;
            }
        } else { // Bisection must be used
            print_debug("Bounds force bisection (e=" + std::to_string(e) + " or |fa|<|fb|)");
            d = xm;
            e = d;
        }

        // Update 'a' and 'fa' for the next iteration
        a = b;
        fa = fb;

        // Perform the step: update 'b'
        double b_old = b; // Store for debugging
        if (std::abs(d) > tol1) {
            b += d;
            print_debug("Step taken: b = b_old + d = " + std::to_string(b_old) + " + " + std::to_string(d) + " = " + std::to_string(b));
        } else { // Step delta 'd' is too small, take minimal step based on tolerance
            b += (xm > 0.0 ? tol1 : -tol1); // Step by tol1 in the direction of the midpoint
            print_debug("Step taken: b = b_old + sign(xm)*tol1 = " + std::to_string(b_old) + " + " + std::to_string(xm > 0.0 ? tol1 : -tol1) + " = " + std::to_string(b));
        }

        // Evaluate function at the new 'b'
        fb = f(b);
        print_debug("New evaluation: f(b=" + std::to_string(b) + ") = " + std::to_string(fb));

        if (std::isnan(fb)) {
            print_debug("ERROR: f(b) returned NaN during iteration.");
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Update 'c' and 'fc' based on the sign of fb
        // Standard Brent: if f(b) and f(c) have different signs, the root is between b and c.
        // Otherwise, the root must be between a and b (since f(a) and f(b) must have different signs
        // if f(b) and f(c) have the same sign, because f(a) and f(c) initially had the same sign).
        // So, if f(b)*f(c) >= 0, set c=a, fc=fa.
        if (fb * fc >= 0.0) {
             print_debug("Updating c=a because fb*fc >= 0.");
             c = a;
             fc = fa;
        }
        // If fb*fc < 0, c and fc remain unchanged (bracket is [b, c])

        // Ensure |f(a)| >= |f(b)| for the next iteration's interpolation logic
        if (std::abs(fa) < std::abs(fb)) {
            print_debug("Swapping a and b post-iteration.");
            std::swap(a, b);
            std::swap(fa, fb);
        }
        // Update d and e for the next iteration (d was the step just taken, e was the one before that)
        // Note: d and e were already updated based on interpolation/bisection decision
    }

    // Max iterations reached
    print_debug("ERROR: Maximum iterations (" + std::to_string(max_iter) + ") reached without convergence.");
    return std::numeric_limits<double>::quiet_NaN();
}

} // namespace RootFinding

#endif // BRENT_HPP