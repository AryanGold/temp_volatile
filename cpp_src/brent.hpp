#pragma once

#include <cmath>
#include <limits>
#include <functional>
#include <stdexcept>
#include <algorithm> // for std::max

namespace RootFinding {

// Basic Brent's method implementation
// Based on description from Wikipedia and Numerical Recipes
template <typename Func>
double brentq(Func f, double xa, double xb, double tol = 1e-5, int max_iter = 100) {
    double a = xa;
    double b = xb;
    double fa = f(a);
    double fb = f(b);
    // --- Suppress unused variable warnings ---
    // double fs = std::numeric_limits<double>::quiet_NaN(); // Comment out if truly unused
    (void)fa; // fa might become unused depending on path, suppress warning
    (void)fb; // fb might become unused depending on path, suppress warning

    if (fa * fb >= 0.0) {
        // Return NaN or throw...
         // Re-evaluate if fa/fb are truly needed here or just for the check
        double f_a_eval = f(a); // Re-evaluate if needed
        double f_b_eval = f(b);
        if (f_a_eval * f_b_eval >= 0.0) {
             return std::numeric_limits<double>::quiet_NaN();
        }
         // Update fa/fb if re-evaluated
         fa = f_a_eval;
         fb = f_b_eval;
    }

    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;
    double fc = fa;
    // bool mflag = true; // Comment out if unused
    // double s = b; // Comment out if unused
    double d = b - a;
    double e = d;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (std::abs(fc) < std::abs(fb)) {
            a = b; fa = fb;
            b = c; fb = fc;
            c = a; fc = fa;
        }

        double tol1 = 2.0 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * tol;
        double xm = 0.5 * (c - b);

        if (std::abs(xm) <= tol1 || fb == 0.0) {
            return b; // Found root
        }

        // Attempt interpolation
        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            double S = fb / fa;
            double P, Q;
            if (a == c) { // Linear interpolation (secant)
                P = 2.0 * xm * S;
                Q = 1.0 - S;
            } else { // Inverse quadratic interpolation
                double R = fb / fc;
                Q = fa / fc;
                P = S * (2.0 * xm * Q * (Q - R) - (b - a) * (R - 1.0));
                Q = (Q - 1.0) * (R - 1.0) * (S - 1.0);
            }

            if (P > 0.0) Q = -Q; // Ensure Q is positive
            P = std::abs(P);

            // Check if interpolation is acceptable
            double min1 = 3.0 * xm * Q - std::abs(tol1 * Q);
            double min2 = std::abs(e * Q);
            if (2.0 * P < std::min(min1, min2)) {
                e = d; // Store previous step
                d = P / Q;
            } else { // Interpolation failed, use bisection
                d = xm;
                e = d;
            }
        } else { // Bisection must be used
            d = xm;
            e = d;
        }

        a = b; // Move last best guess to a
        fa = fb;

        if (std::abs(d) > tol1) { // Perform the step
            b += d;
        } else { // Step too small, use tolerance
            b += (xm > 0.0 ? tol1 : -tol1);
        }

        fb = f(b);
        // Check for bracketing if function evaluation is costly
         if (fb * fc >= 0.0) { // New 'b' did not preserve bracket with 'c'
             c = a; fc = fa;    // Reset 'c' to previous 'a'
             d = b - a; e = d;  // Update step sizes
         } else {
             // Bracket is preserved or fb is zero
             // Update c, fc based on mflag only if necessary (standard Brent)
             if ( (fa * fb) < 0 ) { // bracket between a and b
                 c = a; fc = fa;
             } else {
                 // bracket between c and b (should already be handled by fc*fb < 0 test)
             }
         }


    }

    // Max iterations reached
    // Return NaN
    return std::numeric_limits<double>::quiet_NaN();
}

} // namespace RootFinding
