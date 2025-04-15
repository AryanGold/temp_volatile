
import numpy as np
from scipy.optimize import brentq,newton




def compute_fp(t: float,
               mu: float,
               div_times: np.ndarray,
               div_props: np.ndarray) -> float:
    exponent = mu * t
    product = 1.0
    n_div = div_times.size
    for j in range(n_div):
        if div_times[j] <= t:
            product *= (1.0 - div_props[j])
    return np.exp(exponent) * product


def discounted_dividends_at_t(t, div_times_i, div_cash_i, mu, div_props_i):
    sum_div = 0.0
    n_div = div_times_i.size
    for j in range(n_div):
        if div_times_i[j] <= t:
            # compute forward-price factor at the dividend time
            exponent_j = mu * div_times_i[j]
            product_j = 1.0
            for k in range(n_div):
                if div_times_i[k] <= div_times_i[j]:
                    product_j *= (1.0 - div_props_i[k])
            fp_j = np.exp(exponent_j) * product_j
            sum_div += div_cash_i[j] / fp_j
    return sum_div


def compute_forward(spot, maturity, mu, div_times_i, div_cash_i, div_props_i):
    """
    JIT-friendly version of compute_forward() for a single snapshot.
    """
    fp_maturity = compute_fp(maturity, mu, div_times_i, div_props_i)
    div_value = discounted_dividends_at_t(maturity, div_times_i, div_cash_i, mu, div_props_i)
    return fp_maturity * (spot - div_value)

def compute_forward_at_t(s_t,t,T,mu,div_times,div_cash,div_props):
    residual_fp = np.exp(mu * (T - t))
    
    
    cum_prod = 1.0
    pv_cash_divs = 0.0
    
    # Loop over each dividend event.
    for t_i, d_i, delta_i in zip(div_times, div_cash, div_props):
        # Only include dividends with t < t_i <= T.
        if t < t_i <= T:
            cum_prod *= (1 - delta_i)
            fp_ti = np.exp(mu * (t_i - t))
            pv_cash_divs += d_i / fp_ti
            
    return residual_fp * cum_prod * (s_t - pv_cash_divs)
        
    
    
    


'''def compute_DT(t: float, T: float, mu: float,
                   div_times: np.ndarray,
                   div_props: np.ndarray,div_cash: np.ndarray) -> float:
    
    fp_t = compute_fp(t, mu, div_times, div_props)
    sum1 = 0.0
    sum2 = 0.0
    for ti, di, delta_i in zip(div_times,div_cash,div_props):
        fp_ti = compute_fp(ti, mu, div_times, div_props)
        if (ti > t) and (ti <= T):
            sum1 += di/ fp_ti
        if (ti > 0.0) and (ti <= T):
            sum2 += (ti / T) * (di / fp_ti)
    return fp_t * (sum1 - sum2)
'''

def compute_DT(t: float, T: float, mu: float,
               div_times: np.ndarray,
               div_props: np.ndarray, div_cash: np.ndarray) -> float:
    fp_t = compute_fp(t, mu, div_times, div_props)
    sum1 = 0.0
    sum2 = 0.0
    for ti, di, delta_i in zip(div_times, div_cash, div_props):
        fp_ti = compute_fp(ti, mu, div_times, div_props)
        if (ti > t) and (ti <= T):
            sum1 += di / fp_ti
        if ti <= T:
            sum2 += (ti / T) * (di / fp_ti)
    return fp_t * (sum1 - sum2)



# Black-Scholes formula for European options
def black_scholes_call(F, K, T, sigma, r):
    if T <= 0:
        return max(F - K, 0)
    d1 = (np.log(F / K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def black_scholes_put(F, K, T, sigma, r):
    if T <= 0:
        return max(K - F, 0)
    d1 = (np.log(F / K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))








def ska_model_option(S0, r, q, sigma, T, K, option_type, div_times=[0],div_cash=[0],div_prop=[0],smooth=False, N=100):
    mu = r - q
    

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(mu * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    times = np.linspace(0, T, N+1)
    D_t = [compute_DT(t, T,mu,div_times,div_prop,div_cash) for t in times]
    #print(D_t)

    tilde_S = np.zeros((N+1, N+1))

    # Initialize: pure stock at t=0 is S0 minus the shift at time 0.
    tilde_S[0, 0] = S0 - D_t[0]


    for i in range(1, N+1):

        tilde_S[i, 0] = ((tilde_S[i-1, 0]* d))


        for j in range(1, i+1):
            tilde_S[i, j] = ((tilde_S[i-1, j-1]*u))# - D_t[i-1]) * u) + D_t[i]

        for ti, delta_i in zip(div_times,div_prop):
            if (i-1) * dt < ti <= i * dt:
                for j in range(i+1):
                    tilde_S[i, j] *= (1 - delta_i)

    # Backward induction for the option price:
    V = np.zeros((N+1, N+1))

    # At maturity: adjust strike with the final shift.
    K_T = K - D_t[N]
    for j in range(N+1):

        if option_type == 'c':
            V[N, j] = max((tilde_S[N, j] ) - K_T, 0)
        else:
            V[N, j] = max(K_T - (tilde_S[N, j]), 0)

    
   # if smooth == True:
   #     #(s_t,t,T,mu,div_times,div_cash,div_props)
   # 
   #     i_ = N - 1

   #     dt_bs = D_t[-1]
   #     K_t_bs = K_T - dt_bs
   #     time_in = dt*i_

    #    for j in range(i_ + 1):
    #        adj_f = compute_forward_at_t(tilde_S[i_,j],time_in,T,mu,div_times,div_cash,div_prop) - dt_bs
    #        if option_type == 'c':
    #            continuation_value = black_scholes_call(adj_f, K_t_bs, T-time_in, sigma, r)
    #            hold_value = max(tilde_S[i_,j] - K_t_bs,0)
    #        else:
    #            continuation_value = black_scholes_put(adj_f, K_t_bs, T-time_in, sigma, r)
    #            hold_value = max(K_t_bs- tilde_S[i_,j],0)
    #        V[i_,j] = max(continuation_value,hold_value)
                

#    else:
    i_ = N #reindent when you figure it out
        
        
    
    

    for i in range(i_-1, -1, -1):
        K_t = K - D_t[i]
        for j in range(i+1):
            continuation = discount * (p * V[i+1, j+1] + (1 - p) * V[i+1, j])
            if option_type == 'c':
                exercise = max(tilde_S[i, j]  - K_t, 0)
            else:
                exercise = max(K_t - tilde_S[i, j], 0)

            V[i, j] = max(exercise, continuation)

    return V[0, 0]
    
        





    
# Example usage
S0 = 100.0 #array
K = 100.0 #array
r = 0.05 #array
T = 1.00 # array
q = 0.00 #array
sigma = 0.2 #array
option_type = 'p' #array
N =100 #same

div_times = np.array([0.01]) #same 
div_prop = np.array([0.015]) #same
div_cash = np.array([0.3]) #same
smooth = False #same
richardson = False #same




price_am =ska_model_option(S0, r, q, sigma, T, K, option_type, div_times,div_cash,div_prop,smooth, N)


#needed for vectorized_implied_volatility
def implied_vol_newton(target_price, S0, r, q, T, K, option_type, div_times, div_prop, div_cash, N,
                       initial_guess=0.2, tol=1e-4, maxiter=100):
    
    def objective(sigma):
        price = ska_model_option(S0, r, q, sigma, T, K, option_type, div_times, div_cash, div_prop,smooth=False, N=N)
        return price - target_price

    try:
        iv = newton(objective, initial_guess, tol=tol, maxiter=maxiter)
        return iv
    except Exception as e:
        print("Newton IV solver error:", e)
        return None

#needed for vectorized_implied_volatility
def implied_volatility(target_price, S0, r, q, T, K, option_type, div_times, div_prop, div_cash, N,
                       initial_guess=0.2, vol_lower=1e-2, vol_upper=5.0, tol=1e-4):
    iv_newton = implied_vol_newton(target_price, S0, r, q, T, K, option_type, div_times, div_prop, div_cash, N,
                                   initial_guess=initial_guess, tol=tol)
    if iv_newton is not None and vol_lower < iv_newton < vol_upper:
        return iv_newton
    else:
        try:
            iv = brentq(lambda sigma: ska_model_option(S0, r, q, sigma, T, K, option_type, div_times, div_cash, div_prop, smooth=False,N=N) - target_price,
                        vol_lower, vol_upper, xtol=tol)
            return iv
        except Exception as e:
            print("Bisection IV solver error:", e)
            return None

#convert in c++ and make a wrapper for this function to call from python
def vectorized_implied_volatility(target_prices, S_arr, r_arr, q_arr, T_arr, K_arr, option_types_arr, iv_initial_guess,
                                  div_times, div_prop, div_cash, N, vol_lower=1e-2, vol_upper=5.0, tol=1e-4):
    """
    Compute implied volatilities for a vector of options.
    
    Parameters:
        target_prices : 1D array of market option prices.
        S_arr         : 1D array of current stock prices.
        r_arr         : 1D array of risk-free rates.
        q_arr         : 1D array of dividend yields.
        T_arr         : 1D array of maturities.
        K_arr         : 1D array of strike prices.
        option_types_arr : 1D array of option type strings (e.g., 'call' or 'put' or 'c'/'p').
        iv_initial_guess : 1D array (or a scalar) of initial guesses for volatility.
        
        The dividend schedule inputs below are 1D arrays (common to all options):
        div_times     : 1D array of dividend times.
        div_prop      : 1D array of proportional dividend factors.
        div_cash      : 1D array of cash dividends.
        
        N             : Scalar, number of steps in the pricing tree.
        vol_lower     : Scalar, lower bound for the volatility search.
        vol_upper     : Scalar, upper bound for the volatility search.
        tol           : Scalar, tolerance for the solver.
    
    Returns:
        A NumPy array of implied volatilities (same length as target_prices).
        
    Notes:
        All inputs S_arr, r_arr, q_arr, T_arr, K_arr, option_types_arr, and iv_initial_guess
        must be 1D arrays of the same length. If iv_initial_guess is provided as a scalar, it
        is broadcast to an array.
    """
    # Convert target_prices to array and check lengths.
    target_prices = np.atleast_1d(target_prices)
    n_options = target_prices.size
    
    # Ensure S_arr, r_arr, q_arr, T_arr, K_arr, and option_types_arr are numpy arrays of the same length.
    S_arr = np.atleast_1d(S_arr)
    r_arr = np.atleast_1d(r_arr)
    q_arr = np.atleast_1d(q_arr)
    T_arr = np.atleast_1d(T_arr)
    K_arr = np.atleast_1d(K_arr)
    option_types_arr = np.atleast_1d(option_types_arr)    
    if np.isscalar(iv_initial_guess):
        iv_initial_guess = np.full(n_options, iv_initial_guess)
    else:
        iv_initial_guess = np.atleast_1d(iv_initial_guess)
        if iv_initial_guess.size != n_options:
            raise ValueError("iv_initial_guess must be a scalar or an array with the same length as target_prices.")
    
    # The dividend schedule (div_times, div_prop, div_cash) remains 1D and is used for all options.
    div_times = np.atleast_1d(div_times)
    div_prop = np.atleast_1d(div_prop)
    div_cash = np.atleast_1d(div_cash)
    
    ivs = np.empty(n_options)
    for i in range(n_options):
        iv = implied_volatility(target_prices[i],
                                S_arr[i],
                                r_arr[i],
                                q_arr[i],
                                T_arr[i],
                                K_arr[i],
                                option_types_arr[i],
                                div_times,
                                div_prop,
                                div_cash,
                                N,
                                initial_guess=iv_initial_guess[i],
                                vol_lower=vol_lower,
                                vol_upper=vol_upper,
                                tol=tol)
        ivs[i] = iv if iv is not None else np.nan
        
    return ivs

n_options = 5
target_prices = np.linspace(9.5, 10.5, n_options)
S_arr = np.full(n_options, 100.0)
r_arr = np.full(n_options, 0.05)
q_arr = np.full(n_options, 0.03)
T_arr = np.full(n_options, 1.0)
K_arr = np.full(n_options, 100.0)
option_types_arr = np.full(n_options, 'c')  # Using 'c' for call options, for instance.
iv_initial_guess = np.linspace(0.2, 0.25, n_options) 

# Dividend schedule (common for all options):
div_times = np.array([0.2, 0.4, 0.6])
div_prop = np.array([0.01, 0.04, 0.02])
div_cash = np.array([0.8, 2, 3])

N = 300
vol_lower = 1e-2
vol_upper = 5.0
tol = 1e-4

ivs = vectorized_implied_volatility(target_prices, S_arr, r_arr, q_arr, T_arr, K_arr,option_types_arr, iv_initial_guess,div_times, div_prop, div_cash, N,vol_lower=vol_lower, vol_upper=vol_upper, tol=tol)
print("Implied volatilities:", ivs)
