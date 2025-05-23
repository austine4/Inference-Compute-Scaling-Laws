import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from scipy.special import lambertw, comb, betainc
from scipy.optimize import minimize_scalar
from scipy.stats import norm, binom
import pandas as pd
from scipy.interpolate import griddata, LinearNDInterpolator
import warnings
import os
from enum import Enum

warnings.filterwarnings('ignore')


class InferencePolicy(Enum):
    """Enumeration of different inference policies."""
    COT = "cot"
    BEST_OF_N = "best_of_n"
    MARKOV_TREE_SEARCH = "markov_tree_search"
    CONSENSUS = "consensus"

class ModelFramework:
    """
    A framework for evaluating models with configurable parameters and policies.
    """
    
    # Default constants
    DEFAULT_CONSTANTS = {
        'L': 20, # Total number of skill levels
        'L_min': 5,  # Minimum skill level task
        'L_max': 10, # Maximum skill level task
        'm_min': 2, # Minimum number of skills
        'm_max': 30, # Maximum number of skills
        'S_l': 4e4, # Number of skills per level
        'd_t': 6, # Degrees of text pieces (number of skills required to understand a text) is binomially distributed with a fixed mean degree, d_t
        'zeta': 2.5e3, # Parameters per concept
        'tau': 1e4, # Tokens per training text piece
        'omega': 25, # Output tokens per inference step
        'kappa': 20, # D_tr / N for chinchilla optimal
        'beta': 2, # Factor to scale skills required to relevant set
        'rho': 1, # Search efficiency between relevant and required sets
        'delta': 1e-10, # Small value
        'X_MIN': 0.1,
        'NUM_PTS': 40 # 
    }
    
    def __init__(self, 
                 constants=None, 
                 policy_type=InferencePolicy.COT,
                 policy_params=None):
        """
        Initialize the model with configurable constants and policy.
        
        Args:
            constants (dict, optional): Dictionary of constants to override defaults.
            policy_type (InferencePolicy, optional): Type of inference policy to use.
            policy_params (dict, optional): Parameters for the inference policy.
        """
        # Set up constants
        self.constants = self.DEFAULT_CONSTANTS.copy()
        if constants:
            self.constants.update(constants)
        
        # Extract constants for easier access
        for key, value in self.constants.items():
            setattr(self, key, value)
            
        # Set policy type and parameters
        self.policy_type = policy_type
        self.policy_params = policy_params or {}
        
        # Cache for recursive values
        self._p_values = None
        self._gamma_values = None
    
    # =========================================================================
    # MODEL-SPECIFIC FUNCTIONS (CAN BE OVERRIDDEN)
    # =========================================================================
    
    def eta_l(self):
        """
        Compute eta_l based on skill level l.
        
        Args:
            l (int): Skill level
            
        Returns:
            float: eta_l value
        """
        # return np.exp(5 * l / self.L)
        return 5.0
    
    def sigma_l(self, l):
        """
        Compute sigma_l based on skill level l.
        
        Args:
            l (int): Skill level
            
        Returns:
            float: sigma_l value
        """
        return np.log2(l) if l > 1 else 0

    def xi_l(self, l):
        """
        Compute xi_l based on skill level l.
        
        Args:
            l (int): Skill level
            
        Returns:
            float: xi_l value
        """
        # return np.exp(-l/20)
        # return np.exp(-l/self.L)
        return np.exp(-10*l/self.L)
        # return np.exp(-l**2/80)
        # return np.exp(-l**3/1e3)
        # return np.exp(-l**3/2e3) # works for aime Claude
        # return np.exp(-l**4/4e4)
        # return np.exp(-np.exp(l/1e1))
    
    
    # Task distribution function over l,m
    def phi(self, l,m):

        # Let phi_l be a mixture of binomial distribution
        if l < self.L_min or l > self.L_max:
            return 0
        phi_l = 1/(self.L_max-self.L_min+1)

        """
        # Parameters for the three components:
        mu1, mu2, mu3 = 30, 60, 90      # centers for the peaks
        sigma1, sigma2, sigma3 = 1.0, 1.0, 1.0  # adjust these to control the width of each peak
        # Mixture weights (you can adjust these so that the peaks are similar in height)
        w1, w2, w3 = 0.33, 0.33, 0.34  # must sum to 1
        l_values = np.arange(1, L+1)
        # Define the unnormalized PMF using Gaussian-like bumps
        phi_l_unnormalized = (w1 * np.exp(-0.5 * ((l_values - mu1)/sigma1)**2) +
                        w2 * np.exp(-0.5 * ((l_values - mu2)/sigma2)**2) +
                        w3 * np.exp(-0.5 * ((l_values - mu3)/sigma3)**2))
        phi_l = (phi_l_unnormalized / phi_l_unnormalized.sum())[l-1]
        """

        # Let phi_m be a mixture of binomial distribution
        if m < self.m_min or m > self.m_max:
            return 0
        phi_m = 1/(self.m_max-self.m_min+1)
        """
        # Parameters for the three components:
        mu1, mu2 = 30, 70      # centers for the peaks
        sigma1, sigma2 = 1.0, 1.0  # adjust these to control the width of each peak
        # Mixture weights (you can adjust these so that the peaks are similar in height)
        w1, w2 = 0.5, 0.5  # must sum to 1
        m_values = np.arange(2, m_max+1)
        # Define the unnormalized PMF using Gaussian-like bumps
        phi_m_unnormalized = (w1 * np.exp(-0.5 * ((m_values - mu1)/sigma1)**2) +
                        w2 * np.exp(-0.5 * ((m_values - mu2)/sigma2)**2))
        phi_m = (phi_m_unnormalized / phi_m_unnormalized.sum())[m-2]
        """
        return phi_l * phi_m
    
    
    def phi_non(self, l, m, exp=1):
        """
        Compute the joint probability phi(l, m) = p(l) * p(m) with:
        
        p(l)  ∝ (l - L_min + 1)^alpha,
        p(m)  ∝ ((m - m_min + 1) * (m_max - m + 1))^beta
        
        Parameters:
            l (int): Task skill level.
            m (int): Number of skills.
            alpha (float): Exponent controlling the shape of the p(l) distribution.
            beta (float): Exponent controlling the peak sharpness of the p(m) distribution.
            
        Returns:
            float: The joint probability phi(l, m), or 0 if l or m is outside the allowed range.
        """
        # Check if l and m are within bounds.
        if l < self.L_min or l > self.L_max or m < self.m_min or m > self.m_max:
            return 0

        # Compute p(l) with a monomial weighting.
        weight_l = (l - self.L_min + 1)**exp
        norm_l = sum((j - self.L_min + 1)**exp for j in range(self.L_min, self.L_max + 1))
        p_l = weight_l / norm_l

        # Compute p(m) with a center-peaking weighting.
        weight_m = ((m - self.m_min + 1) * (self.m_max - m + 1))**exp
        norm_m = sum(((j - self.m_min + 1) * (self.m_max - j + 1))**exp for j in range(self.m_min, self.m_max + 1))
        p_m = weight_m / norm_m

        return p_l * p_m
    
    
    def q(self, p):
        """
        Compute probability of successfully completing a task based on connection probability.
        
        Args:
            p (float): Connection probability
            
        Returns:
            float: Success probability
        """
        return p
    
    # =========================================================================
    # CORE MATHEMATICAL FUNCTIONS
    # =========================================================================
    
    def D_KL(self, p, q):
        """
        Compute KL divergence between two Bernoulli distributions with probabilities p and q.
        Handles very small values with high precision.
    
        Args:
            p (float): Probability of success for first Bernoulli distribution
            q (float): Probability of success for second Bernoulli distribution
    
        Returns:
            float: KL divergence value (in nats)
        """
        # Convert to high precision float
        p = np.float128(p)
        q = np.float128(q)
    
        # Validate inputs
        if not (0 <= p <= 1 and 0 <= q <= 1):
            raise ValueError("Probabilities must be between 0 and 1")
    
        
        # Compute KL divergence for Bernoulli
        # KL = p * log(p/q) + (1-p) * log((1-p)/(1-q))
        term1 = p * (np.log(p) - np.log(q))
        term2 = (1 - p) * (np.log(1 - p) - np.log(1 - q))
    
        kl = term1 + term2
    
        # Check for invalid values
        if np.isnan(kl) or np.isinf(kl):
            return np.inf
    
        return float(kl)
    
    def compute_T(self, R):
        """
        Compute T based on R.
        
        Args:
            R (float): R parameter
            
        Returns:
            float: T value
        """
        return ((self.kappa * self.zeta) / self.tau) * R
    
    def compute_p_l(self, R, P_bar, eta, xi, gamma_prev, sigma):
        """
        Compute p_l based on the piecewise function.
        
        Args:
            R (float): R parameter
            P_bar (float): P_bar value
            eta (float): eta value
            xi (float): xi value
            gamma_prev (float): Previous gamma value
            sigma (float): Sigma value
            
        Returns:
            float: p_l value
        """        
        # Calculate d_delta
        delta_val = 1e-5 #1/((Sl**2)*L)      
        mu_val = xi*R/self.S_l
        d_delta = 0.99*mu_val*(1-np.sqrt(2*np.log(self.S_l/delta_val)/(mu_val)))
        d_delta = d_delta*(d_delta>0)

        
        if eta>d_delta:
            term1_pl = 0.0
        else:
            if (eta/d_delta < P_bar):
                term1_pl = 1-np.exp(-d_delta*self.D_KL(eta/d_delta, P_bar))
            else:
                term1_pl = (1/np.sqrt(2*d_delta))*np.exp(-d_delta*self.D_KL(eta/d_delta, P_bar))
        
        term2_pl = gamma_prev**(2*sigma) #  factor of 2 - one for s_1 and another for s_2    
        pl = term1_pl*term2_pl

        return pl
    

    def compute_gamma_l(self, p_l):
        """
        Compute gamma_l using the Lambert W function.
        
        Args:
            p_l (float): p_l value
            
        Returns:
            float: gamma_l value
        """
        z = -p_l * self.S_l * np.exp(-p_l * self.S_l)
        w_val = np.real(lambertw(z, k=0))
        gamma_l = 1 + (1 / (p_l * self.S_l)) * w_val
        # If gamma_l is nan, set it to 0
        if np.isnan(gamma_l):
            gamma_l = 0
        return gamma_l
    
    def compute_recursive_values(self, R, l_cap=None):
        """
        Compute p_l and gamma_l recursively for all levels up to l.
        
        Args:
            R (float): R parameter
            l_cap (int, optional): Maximum level. Defaults to self.L.
            
        Returns:
            tuple: (p_values, gamma_values)
        """
        if l_cap is None:
            l_cap = self.L
            
        # Initialize with base conditions
        gamma_values = [1.0]  # gamma_0 = 1
        p_values = [0.0]      # placeholder for p_0 (not used)
            
        # Calculate T based on R
        T = self.compute_T(R)        

        # calculate P_bar
        R_opt, _, _, _, Pb, _, _ = self.num_skills_learnt_biterr_infcmp(R, self.d_t, T*R, eps=0.5, optimize_flag=False)
        P_bar = 1-Pb
                
        # calculate eta                              
        eta = self.eta_l()
        
        # Recursively compute p_l and gamma_l for each level
        for l in range(1, l_cap + 1):            
            gamma_prev = gamma_values[l-1]
            sigma = self.sigma_l(l)        
            xi = self.xi_l(l)
            
            p_l = self.compute_p_l(R, P_bar, eta, xi, gamma_prev, sigma)
            p_values.append(p_l)
            
            if p_l == 0:
                gamma_l = 0.0
            else:
                gamma_l = self.compute_gamma_l(p_l)
            gamma_values.append(gamma_l)

        return p_values, gamma_values

    def num_skills_learnt_biterr_infcmp(self, s, dt, FLOPS, eps, optimize_flag=True):
        t = FLOPS/s  
        sbar = s*(1-eps)/eps
        n = sbar+s # = s/eps
        nv, nc = t, n # these are not number of variable nodes and check nodes. These are n's of binomial distribution
        ds = dt*t/s # dtbar*t/n = (dt/eps)*(t/(s/eps)) = dt*t/s
        dtbar = ds*n/t # = dt/eps
        pv, pc = ds/nv, dtbar/nc

        pv = min(pv, 1.0)
        pc = min(pc, 1.0)
        
        eps_BP = self.get_epsilon_BP_infcmp(nv, nc, pv, pc)
        eps_min = np.float128(1e-16)
        x_BP, Pb, gamma_BP = 0.0, 0.0, 0.0 # initialization
        if eps_BP == 1:
            s_learnt = s
            Q_val = 0.0
            alpha = 1e-3 # some small value so that P_B -> 0
        elif eps_BP == eps_min or eps_BP < eps:
            s_learnt = 0.0
            Q_val = 1.0
            alpha = 1e-3 # some small value so that P_B -> 1
            gamma_BP = 1.0
            x_BP = 1.0
        else:    
            x_BP = self.get_x_BP(eps_BP, nv, nc, pv, pc)    
            alpha = self.get_alpha(nv, nc, pv, pc, x_BP, eps_BP)      
            
            Q_val = 1-norm.cdf(float(np.sqrt(n)*(eps_BP - eps)/alpha), loc=0, scale=1)

            gamma_BP = eps_BP*self.L_func_binomial(1-self.rho_func_binomial(1-x_BP, pc, nc), pv, nv)
            Q_val = gamma_BP*Q_val/eps

            s_learnt = s*(1-Q_val)
            #print(f"s={s}, s_learnt={s_learnt}, eps_BP = {eps_BP}, Q={Q}")

        if optimize_flag:
            return -s_learnt
        else:
            return s_learnt, eps_BP, x_BP, Pb, Q_val, gamma_BP, alpha
    
    '''
    compute alpha for a given degree distribution
    '''
    def get_alpha(self, nv, nc, pv, pc, x_BP, eps_BP):
        if x_BP == self.X_MIN:
            alpha_val = 1e-5 # some small value so that P_B -> 0
            alpha_BEC_n_neps = alpha_val
        else:
            x_BP_bar = 1-x_BP
            y_BP = 1-self.rho_func_binomial(x_BP_bar, pc, nc)
            term1_numer1_2 = self.rho_func_binomial(x_BP_bar, pc, nc)**2-self.rho_func_binomial(x_BP_bar**2, pc, nc)
            term1_numer3 = self.rho_prime_func_binomial(x_BP_bar, pc, nc)*(1-2*x_BP*self.rho_func_binomial(x_BP_bar, pc, nc))
            term1_numer4 = (x_BP_bar**2)*self.rho_prime_func_binomial(x_BP_bar**2, pc, nc)
            term1_denom = self.L_prime_func_binomial(1, pv, nv)*(self.lambda_func_binomial(y_BP, pv, nv)**2)*(self.rho_prime_func_binomial(x_BP_bar, pc, nc)**2)

            term2_numer1 = (eps_BP**2)*(self.lambda_func_binomial(y_BP, pv, nv)**2)
            term2_numer2 = (eps_BP**2)*self.lambda_func_binomial(y_BP**2, pv, nv)
            term2_numer3 = (y_BP**2)*(eps_BP**2)*self.lambda_prime_func_binomial(y_BP**2, pv, nv)
            term2_denom = self.L_prime_func_binomial(1, pv, nv)*(self.lambda_func_binomial(y_BP, pv, nv)**2)

            alpha_eps = ((term1_numer1_2+term1_numer3+term1_numer4)/term1_denom + (term2_numer1+term2_numer2+term2_numer3)/term2_denom)**(1/2)

            # added on 04/21/2025
            # subtract variance due to channel noise, retain only variance due to decoding
            alpha_BEC_n_neps = np.sqrt(alpha_eps**2 - eps_BP*(1-eps_BP))

            if alpha_BEC_n_neps>2 or alpha_BEC_n_neps<0:
                print("Invalid alpha value")
                
            return alpha_BEC_n_neps
    
    '''
    Fixed point characterization of threshold
    '''    
    def lambda_func_binomial(self, x, p, n):
        return (p*x+(1-p))**(n-1)

    def lambda_prime_func_binomial(self, x, p, n):  
        return (n-1)*p*((p*x+(1-p))**(n-2))

    def L_func_binomial(self, x, p, n):  
        denom = (1-(1-p)**n)
        numer = (p*x+(1-p))**(n)-(1-p)**n
        return numer/denom

    def L_prime_func_binomial(self, x, p, n): 
        denom = (1-(1-p)**n)
        numer = n*p*(p*x+(1-p))**(n-1)
        return numer/denom

    def rho_func_binomial(self, x, p, n):
        return (p*x+(1-p))**(n-1)

    def rho_prime_func_binomial(self, x, p, n):  
        return (n-1)*p*((p*x+(1-p))**(n-2))

    def func_fx_pt(self, x, eps, nv, nc, pv, pc, neg=False):
        if neg:
            return -(x-eps*self.lambda_func_binomial(1-self.rho_func_binomial(1-x, pc, nc), pv, nv))
        else:
            return x-eps*self.lambda_func_binomial(1-self.rho_func_binomial(1-x, pc, nc), pv, nv)
    
    def get_epsilon_BP_infcmp(self, nv, nc, pv, pc):
        eps_low = np.float128(1e-16)
        eps_high = np.float128(1-eps_low)
        eps_tolerance = np.float128(1e-16)
        num_iters = 100
        iter = 0
        eps_BP = eps_high
        # eps_vec = np.zeros(num_iters)
        while iter < num_iters and np.abs(eps_low-eps_high)>eps_tolerance:    

            eps_tmp = (eps_low+eps_high)/2
            if self.is_solution_in_0_1(eps_tmp, nv, nc, pv, pc) == 0:
                eps_low = eps_tmp
            else:
                eps_high = eps_tmp
            iter += 1  

        if iter>0.9*num_iters:
            print("max iters reached")

        eps_BP = eps_tmp
        while self.is_solution_in_0_1(eps_BP, nv, nc, pv, pc) == 0 and eps_BP+eps_tolerance<1.0:
            eps_BP = eps_BP+eps_tolerance
        
        # >>>> for debug
        if self.is_solution_in_0_1(eps_BP, nv, nc, pv, pc) == 0:
            eps_BP = 1.0 #np.float128(1e-16)
            print(f"eps_BP={eps_BP}")    

        return eps_BP
    '''
    Checking if x - \lambda(1-\rho(1-x)) intersects x-axis or not for any x \in (0, 1] (Note the exclusion of 0)
    If it intersects, then the root corresponds to eps_tmp > \epsilon^{BP}
    '''
    def is_solution_in_0_1(self, eps_tmp, nv, nc, pv, pc):
        x_low = self.X_MIN #1e-16
        x_high = 1

        # min
        args = [eps_tmp, nv, nc, pv, pc, False] 
        min_x = self.my_minimizer(self.func_fx_pt, args, x_low, x_high, num_pts=self.NUM_PTS)
        # max
        args = [eps_tmp, nv, nc, pv, pc, True]
        max_x = self.my_minimizer(self.func_fx_pt, args, x_low, x_high, num_pts=self.NUM_PTS)

        if np.sign(self.func_fx_pt(min_x, eps_tmp, nv, nc, pv, pc)) == np.sign(self.func_fx_pt(max_x, eps_tmp, nv, nc, pv, pc)):
            return 0
        else:
            return 1
    
    def get_x_BP(self, eps_BP, nv, nc, pv, pc):
        x_low = self.X_MIN #1e-14 #1e-16
        x_high = 1
        # eps_BP = eps_BP+1e-8        
        if self.is_solution_in_0_1(eps_BP, nv, nc, pv, pc) == 0:
            x_BP = self.X_MIN
        else:
            args = [eps_BP, nv, nc, pv, pc, False]            
            x_BP = self.my_minimizer(self.func_fx_pt, args, x_low, x_high, num_pts=self.NUM_PTS)    

        return x_BP

    '''
    Optimize: grid search followed by scipy.optimize
    '''
    def my_minimizer(self, my_fun, args, x_low, x_high, num_pts=40):
        tol_val = 1e-18
        x_vec_tmp = np.linspace(x_low, x_high, num_pts)
        f_vec_tmp = np.zeros(x_vec_tmp.shape)        
        for ind_x, x in enumerate(x_vec_tmp):            
            f_min_tmp = my_fun(x, *args)
            f_vec_tmp[ind_x] = f_min_tmp
        x_init = x_vec_tmp[np.argmin(f_vec_tmp)] 

        x_lb = x_vec_tmp[max(np.argmin(f_vec_tmp)-1, 0)]
        x_ub =x_vec_tmp[min(np.argmin(f_vec_tmp)+1, len(f_vec_tmp)-1)]

        x_init = np.float128(x_init)

        minimizer_kwargs = {'method':'Bounded', 'bounds':(x_lb, x_ub), 'args':tuple(args), 'tol':tol_val}
        x_opt = minimize_scalar(my_fun, **minimizer_kwargs).x

        return x_opt
    
    def compute_m_l_prime(self, l, l_prime, m):
        """
        Compute m_l' based on the given formula.
        
        Args:
            l (int): Skill level
            l_prime (int): Target skill level
            m (int): Original m value
            
        Returns:
            float: m_l' value
        """
        if l_prime > l:
            # Calculate the product term in the denominator
            prod = 1
            for k in range(l+1, l_prime+1):
                prod *= self.sigma_l(k)
            result = max(np.ceil(m / prod), 2)
            # result = max(np.round(m / prod), 2)
        elif l_prime == l:
            result = m
        else:  # l_prime < l
            # Calculate the product term
            prod = 1
            for k in range(l_prime+1, l+1):
                prod *= self.sigma_l(k)
            result = np.ceil(m * prod)
            # result = np.round(m * prod)
        
        return result
    
    def compute_M_l_prime(self, m_l_prime):
        """
        Compute M_l' based on m_l'.
        
        Args:
            m_l_prime (float): m_l' value
            
        Returns:
            float: M_l' value
        """
        return np.ceil(m_l_prime + self.beta * m_l_prime)
        # return np.round(m_l_prime + self.beta * m_l_prime)
    
    def compute_training_cost(self, R):
        """
        Compute the training cost C_tr.
        
        Args:
            R (float): R parameter
            
        Returns:
            float: Training cost
        """
        return 6 * self.kappa * self.zeta**2 * R**2

    # =========================================================================
    # POLICY-SPECIFIC FUNCTIONS
    # =========================================================================

    def evaluate_allocation(self, C_tr, C_inf, l, m, p_values=None, gamma_values=None):
        """
        Evaluate allocation for fixed l and m based on the current policy.
        
        Args:
            C_tr (float): Training cost
            C_inf (float): Inference cost
            l (int): Skill level
            m (int): Difficulty parameter
            p_values (list, optional): Pre-computed p values
            gamma_values (list, optional): Pre-computed gamma values
            
        Returns:
            tuple: (accuracy, inference_cost)
        """
        # Ensure p_values and gamma_values are available
        if p_values is None or gamma_values is None:
            p_values, gamma_values = self.train(C_tr)
            
        if self.policy_type == InferencePolicy.COT:
            return self.evaluate_allocation_COT(C_tr, C_inf, l, m, p_values, gamma_values)
        elif self.policy_type == InferencePolicy.BEST_OF_N:
            return self.evaluate_allocation_best_of_N(C_tr, C_inf, l, m, p_values, gamma_values)
        elif self.policy_type == InferencePolicy.MARKOV_TREE_SEARCH:
            return self.evaluate_allocation_markov_tree_search(C_tr, C_inf, l, m, p_values, gamma_values)
        elif self.policy_type == InferencePolicy.CONSENSUS:
            return self.evaluate_allocation_consensus(C_tr, C_inf, l, m, p_values, gamma_values)
        else:
            raise ValueError(f"Unsupported policy type: {self.policy_type}")

    def compute_expected_steps_cot(self, gamma_l_star, m_l_star, r_l_star, K_max):
        """
        Compute the expected number of steps needed for Chain of Thought.
        Based on the formula:
        E[K|m,r_CoT,K_max]= sum(k=m to K_max) k * (gamma^m * I_r(m,k-m+1) - gamma^m * I_r(m,k-m))
                            + K_max * (1 - gamma^m * I_r(m,K_max-m+1))
        
        Args:
            gamma_l_star (float): gamma value
            m_l_star (float): m value
            r_l_star (float): r value
            K_max (float): Maximum number of steps
            
        Returns:
            float: Expected number of steps
        """
        # Initialize
        expected_steps = 0
        m = int(np.ceil(m_l_star))  # Ensure m is an integer
        # m = int(np.round(m_l_star))  # Ensure m is an integer
        r = r_l_star
        gamma = gamma_l_star
        
        """
        # Calculate using the summation formula
        for k in range(m, int(np.ceil(K_max)) + 1):
            # Calculate beta incomplete for k-m+1 and k-m
            beta_k_plus_1 = betainc(m, k-m+1, r)
            beta_k = betainc(m, k-m, r) if k > m else 0
            
            # Add to expected steps
            expected_steps += k * gamma**m * (beta_k_plus_1 - beta_k)
        
        # Add the final term for K_max
        expected_steps += K_max * (1 - gamma**m * betainc(m, K_max-m+1, r))
        """

        # Do to massive K_max, we approximate:
        expected_steps = min(K_max, m_l_star/r_l_star)
        return expected_steps

    def compute_accuracy_COT(self, gamma_l_star, m_l_star, r_l_star, K_max):
        """
        Compute the accuracy for Chain of Thought.
        
        Args:
            gamma_l_star (float): gamma_l* value
            m_l_star (float): m_l* value
            r_l_star (float): r_l* value
            K_max (float): N* value
            
        Returns:
            float: Accuracy
        """
        return (gamma_l_star ** m_l_star) * betainc(m_l_star, K_max - m_l_star + 1, r_l_star)
    
    # =========================================================================
    # EVALUATION FUNCTIONS
    # =========================================================================
    
    def train(self, C_tr):
        """
        Train the model and evaluate accuracy for a given allocation of C_tr.
        
        Args:
            C_tr (float): Training cost
            
        Returns:
            tuple: (p_values, gamma_values)
        """
        # Calculate R from C_tr
        R = np.sqrt(C_tr / (6 * self.kappa * self.zeta**2))
        
        # Compute recursive values
        p_values, gamma_values = self.compute_recursive_values(R, l_cap=self.L)
        
        # Cache the values
        self._p_values = p_values
        self._gamma_values = gamma_values

        return p_values, gamma_values
    
    def evaluate_allocation_COT(self, C_tr, C_inf, l, m, p_values=None, gamma_values=None):
        """
        Evaluate allocation for Chain of Thought with fixed l and m.
        
        Args:
            C_tr (float): Training cost
            C_inf (float): Inference cost
            l (int): Skill level
            m (int): Difficulty parameter
            p_values (list, optional): Pre-computed p values
            gamma_values (list, optional): Pre-computed gamma values
            
        Returns:
            tuple: (accuracy, actual_inference_cost)
        """
        # Ensure p_values and gamma_values are available
        if p_values is None or gamma_values is None:
            p_values, gamma_values = self.train(C_tr)
        
        # Calculate R from C_tr
        R = np.sqrt(C_tr / (6 * self.kappa * self.zeta**2))

        # Compute steps from C_inf
        K_max = C_inf / (2 * self.zeta * R * self.omega)

        # Find best l_star for this allocation
        best_accuracy = 0
        cost = 2 * self.zeta * R * self.omega * K_max
        
        # Find if any p_values is 1 and if so store index
        p_is_1_index = np.where(np.array(p_values) == 1)[0]
        # l_min = 1 if len(p_is_1_index) == 0 else p_is_1_index[-1]
        l_min = 1

        # Try different l_star values
        for l_prime in range(l_min, self.L_max + 1):
            p_l_prime = p_values[l_prime]
            gamma_l_prime = gamma_values[l_prime]
            
            # Skip if gamma is 0
            if gamma_l_prime == 0:
                break
                
            gamma_l_prime = gamma_values[l_prime]
            m_l_prime = self.compute_m_l_prime(l, l_prime, m)
            M_l_prime = self.compute_M_l_prime(m_l_prime)
            p_eff = p_l_prime * (self.rho / M_l_prime + 1-self.rho)
            r_l_prime = p_eff * self.q(p_l_prime)
            
            # Calculate accuracy
            accuracy = self.compute_accuracy_COT(gamma_l_prime, m_l_prime, r_l_prime, K_max)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Calculate expected steps and actual inference cost
                expected_steps = self.compute_expected_steps_cot(gamma_l_prime, m_l_prime, r_l_prime, K_max)
                cost = 2 * self.zeta * R * self.omega * expected_steps

        return best_accuracy, cost
    
    def evaluate_allocation_all(self, C_tr, C_inf):
        """
        Evaluate accuracy and cost for all possible values of l and m based on the current policy.
        
        Args:
            C_tr (float): Training cost
            C_inf (float): Inference cost
            
        Returns:
            tuple: (expected_accuracy, expected_cost)
        """
        if self.policy_type == InferencePolicy.COT:
            return self.evaluate_allocation_all_COT(C_tr, C_inf)
        elif self.policy_type == InferencePolicy.BEST_OF_N:
            return self.evaluate_allocation_all_best_of_N(C_tr, C_inf)
        elif self.policy_type == InferencePolicy.MARKOV_TREE_SEARCH:
            return self.evaluate_allocation_all_markov_tree_search(C_tr, C_inf)
        elif self.policy_type == InferencePolicy.CONSENSUS:
            return self.evaluate_allocation_all_consensus(C_tr, C_inf)
        else:
            raise ValueError(f"Unsupported policy type: {self.policy_type}")
    
    def evaluate_allocation_all_COT(self, C_tr, C_inf):
        """
        Evaluate accuracy and cost for all possible values of l and m for COT.
        
        Args:
            C_tr (float): Training cost
            C_inf (float): Inference cost
            
        Returns:
            tuple: (expected_accuracy, expected_cost)
        """
        # Initialize accuracy and cost
        expected_accuracy = 0
        expected_cost = 0
        
        # Compute p_values and gamma_values
        p_values, gamma_values = self.train(C_tr)
        # plt.figure(100)
        # plt.subplot(211)
        # plt.plot(p_values)
        # plt.subplot(212)
        # plt.plot(gamma_values)
        # plt.show(block=False)
        # brkpnt1 = 1

        # Iterate over all possible values of l and m
        for l in range(self.L_min, self.L_max+1):
            for m in range(self.m_min, self.m_max+1):
                # Get probability for this l,m pair
                prob = self.phi(l, m)
                
                # Get accuracy and cost for this l,m pair
                accuracy, cost = self.evaluate_allocation_COT(C_tr, C_inf, l, m, p_values, gamma_values)
                
                # Add weighted accuracy and cost to totals
                expected_accuracy += prob * accuracy
                expected_cost += prob * cost
        
        # print Ctr expected accuracy in a single line
        # print Ctr in xey format for example 1e1-, 2.5e15 etc

        # print(f"Expected accuracy: {expected_accuracy:.4f}, Ctr: {C_tr:0.2e}")

        return expected_accuracy, expected_cost

    def evaluate_allocation_all_best_of_N(self, C_tr, C_inf):
        """
        Evaluate accuracy and cost for all possible values of l and m with Best of N.
        
        Args:
            C_tr (float): Training cost
            C_inf (float): Inference cost
            
        Returns:
            tuple: (expected_accuracy, expected_cost)
        """
        # Initialize accuracy and cost
        expected_accuracy = 0
        expected_cost = 0
        
        # Compute p_values and gamma_values
        p_values, gamma_values = self.train(C_tr)

        # Iterate over all possible values of l and m
        for l in range(self.L_min, self.L_max+1):
            for m in range(self.m_min, self.m_max+1):
                # Get probability for this l,m pair
                prob = self.phi(l, m)
                
                # Get accuracy and cost for this l,m pair
                accuracy, cost = self.evaluate_allocation_best_of_N(C_tr, C_inf, l, m, p_values, gamma_values)
                
                # Add weighted accuracy and cost to totals
                expected_accuracy += prob * accuracy
                expected_cost += prob * cost
        
        return expected_accuracy, expected_cost

    def evaluate_allocation_all_markov_tree_search(self, C_tr, C_inf):
        """
        Evaluate accuracy and cost for all possible values of l and m with Markov Tree Search.
        
        Args:
            C_tr (float): Training cost
            C_inf (float): Inference cost
            
        Returns:
            tuple: (expected_accuracy, expected_cost)
        """
        # Initialize accuracy and cost
        expected_accuracy = 0
        expected_cost = 0
        
        # Compute p_values and gamma_values
        p_values, gamma_values = self.train(C_tr)

        # Iterate over all possible values of l and m
        for l in range(self.L_min, self.L_max+1):
            for m in range(self.m_min, self.m_max+1):
                # Get probability for this l,m pair
                prob = self.phi(l, m)
                
                # Get accuracy and cost for this l,m pair
                accuracy, cost = self.evaluate_allocation_markov_tree_search(C_tr, C_inf, l, m, p_values, gamma_values)
                
                # Add weighted accuracy and cost to totals
                expected_accuracy += prob * accuracy
                expected_cost += prob * cost
        
        return expected_accuracy, expected_cost

    def evaluate_allocation_all_consensus(self, C_tr, C_inf):
        """
        Evaluate accuracy and cost for all possible values of l and m with Consensus.
        
        Args:
            C_tr (float): Training cost
            C_inf (float): Inference cost
            
        Returns:
            tuple: (expected_accuracy, expected_cost)
        """
        # Initialize accuracy and cost
        expected_accuracy = 0
        expected_cost = 0
        
        # Compute p_values and gamma_values
        p_values, gamma_values = self.train(C_tr)

        # Iterate over all possible values of l and m
        for l in range(self.L_min, self.L_max+1):
            for m in range(self.m_min, self.m_max+1):
                # Get probability for this l,m pair
                prob = self.phi(l, m)
                
                # Get accuracy and cost for this l,m pair
                accuracy, cost = self.evaluate_allocation_consensus(C_tr, C_inf, l, m, p_values, gamma_values)
                
                # Add weighted accuracy and cost to totals
                expected_accuracy += prob * accuracy
                expected_cost += prob * cost
        
        return expected_accuracy, expected_cost
    
    # Minimize C_tot
    def find_optimal_allocation(self, total_budget, n_steps=10):
        """
        Find the optimal allocation of resources between training and inference.
        
        Args:
            total_budget (float): Total computational budget
            n_steps (int, optional): Number of steps to evaluate. Defaults to 10.
            
        Returns:
            tuple: (optimal_C_tr, optimal_C_inf, best_accuracy)
        """
        best_accuracy = 0
        optimal_C_tr = 0
        optimal_C_inf = total_budget
        
        # Try different allocations
        for i in range(n_steps + 1):
            C_tr = (i / n_steps) * total_budget
            C_inf = total_budget - C_tr
            
            # Skip if either cost is too small
            if C_tr < 1e-6 or C_inf < 1e-6:
                continue
                
            # Train the model
            self.train(C_tr)
            
            # Evaluate accuracy
            accuracy = self.evaluate_allocation_all(C_tr, C_inf)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                optimal_C_tr = C_tr
                optimal_C_inf = C_inf
        
        return optimal_C_tr, optimal_C_inf, best_accuracy
    
    # =========================================================================
    # DATA COLLECTION AND VISUALIZATION FUNCTIONS
    # =========================================================================
    
    def evaluate_grid(self, C_tr_values, C_inf_values, output_file=None, replace=False):
        """
        Evaluate accuracy and cost for all combinations of C_tr and C_inf and store in a DataFrame.
        
        Args:
            C_tr_values (list): List of training costs to evaluate
            C_inf_values (list): List of inference costs to evaluate
            output_file (str, optional): CSV file to save results
            replace (bool, optional): Whether to replace existing results
            
        Returns:
            pandas.DataFrame: DataFrame with results
        """
        # Import tqdm for progress tracking
        try:
            from tqdm import tqdm
        except ImportError:
            # If tqdm is not available, create a simple replacement
            def tqdm(iterable, **kwargs):
                return iterable
            print("Note: Install tqdm package for progress bars")
        
        # Initialize columns including the new cost metrics
        columns = ['C_tr', 'C_inf', 'C_inf_allocated', 'Accuracy', 'Policy', 'Params', 'Tokens']
        
        # Initialize or load accuracy DataFrame
        if output_file and os.path.exists(output_file) and not replace:
            try:
                accuracy_df = pd.read_csv(output_file)
                # Ensure all required columns exist
                for col in columns:
                    if col not in accuracy_df.columns:
                        accuracy_df[col] = None  # Add the column with None values
            except Exception as e:
                print(f"Error reading existing file: {e}")
                accuracy_df = pd.DataFrame(columns=columns)
        else:
            accuracy_df = pd.DataFrame(columns=columns)
        
        # Policy identifier
        policy_id = self.policy_type.value
        
        # Track statistics for reporting
        total_combinations = len(C_tr_values) * len(C_inf_values)
        processed = 0
        skipped = 0
        
        # Evaluate for all combinations with progress bar
        for C_tr in tqdm(C_tr_values, desc="Training costs", ncols=100):
            # Compute once for this C_tr
            self.train(C_tr)
            
            # Calculate R (params) based on C_tr
            params = np.sqrt(C_tr / (6 * self.kappa * self.zeta**2))
            
            # Use nested progress bar for inference costs
            for C_inf in tqdm(C_inf_values, desc=f"  Inference costs (C_tr={C_tr:.2e})", leave=False, ncols=100):
                # Check if this combination already exists and we're not replacing
                if not replace:
                    matching_rows = ((accuracy_df['C_tr'] == C_tr) & 
                                    (accuracy_df['C_inf_allocated'] == C_inf) & 
                                    (accuracy_df['Policy'] == policy_id))
                    if matching_rows.any():
                        skipped += 1
                        continue
                
                # Evaluate accuracy and cost
                accuracy, actual_cost = self.evaluate_allocation_all(C_tr, C_inf)
                processed += 1
                
                # Calculate the actual number of steps used
                # actual_cost = steps * 2 * self.zeta * params * self.omega
                # So: steps = actual_cost / (2 * self.zeta * params * self.omega)
                steps_used = actual_cost / (2 * self.zeta * params * self.omega)
                
                # Calculate tokens correctly from steps (each step uses omega tokens)
                tokens = steps_used * self.omega
                
                # For debugging - show both calculations
                allocated_tokens = C_inf / (2 * self.zeta * params * self.omega)
                print(f"Allocated tokens: {allocated_tokens}, Used tokens: {tokens}")
                
                # Remove existing row if replacing
                if replace:
                    accuracy_df = accuracy_df.loc[~((accuracy_df['C_tr'] == C_tr) & 
                                                (accuracy_df['C_inf_allocated'] == C_inf) & 
                                                (accuracy_df['Policy'] == policy_id))]
                
                # Add result to DataFrame
                new_row = pd.DataFrame({
                    'C_tr': [C_tr], 
                    'C_inf': [actual_cost],  # This is the expected inference cost calculated from the evaluation
                    'C_inf_allocated': [C_inf],  # This is the budget allocated to inference
                    'Accuracy': [accuracy],
                    'Policy': [policy_id],
                    'Params': [params],
                    'Tokens': [tokens]  # Correctly calculated tokens based on steps
                })
                accuracy_df = pd.concat([accuracy_df, new_row], ignore_index=True)
                
                # Save after each evaluation
                if output_file:
                    accuracy_df.to_csv(output_file, index=False)
        
        print(f"Completed: {processed} combinations processed, {skipped} combinations skipped.")
        return accuracy_df    
        

    def evaluate_grid_parallel(self, C_tr_values, C_inf_values, output_file=None, replace=False, n_jobs=-1):
        """Parallelized version of evaluate_grid"""
        # Import necessary libraries
        try:
            from joblib import Parallel, delayed
            from tqdm.auto import tqdm
        except ImportError:
            print("Install joblib and tqdm for parallel processing and progress bars")
            return self.evaluate_grid(C_tr_values, C_inf_values, output_file, replace)
        
        # Define a worker function that processes one (C_tr, C_inf) pair
        def process_pair(C_tr, C_inf):
            # Train for this C_tr
            self.train(C_tr)
            params = np.sqrt(C_tr / (6 * self.kappa * self.zeta**2))
            
            # Evaluate
            accuracy, actual_cost = self.evaluate_allocation_all(C_tr, C_inf)
            
            # Calculate the actual number of steps used
            steps_used = actual_cost / (2 * self.zeta * params * self.omega)
            
            # Calculate tokens correctly from steps
            tokens = steps_used * self.omega
            
            return {
                'C_tr': C_tr,
                'C_inf': actual_cost,
                'C_inf_allocated': C_inf,
                'Accuracy': accuracy,
                'Policy': self.policy_type.value,
                'Params': params,
                'Tokens': tokens  # Correctly calculated tokens
            }
        
        # Generate all pairs to evaluate
        pairs = [(C_tr, C_inf) for C_tr in C_tr_values for C_inf in C_inf_values]
        
        # Process in parallel with progress bar
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_pair)(C_tr, C_inf) 
            for C_tr, C_inf in tqdm(pairs, desc="Evaluating pairs")
        )
        
        # Convert results to DataFrame
        new_df = pd.DataFrame(results)
        
        # Handle existing file if needed
        if output_file and os.path.exists(output_file) and not replace:
            try:
                existing_df = pd.read_csv(output_file)
                df = pd.concat([existing_df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=['C_tr', 'C_inf_allocated', 'Policy'])
            except Exception as e:
                print(f"Error reading existing file: {e}")
                df = new_df
        else:
            df = new_df
        
        # Save results
        if output_file:
            df.to_csv(output_file, index=False)
            
        return df
    
    def plot_accuracy_vs_C_inf(self, accuracy_df=None, C_tr_values=None, figsize=(10, 6), save_path=None):
        """
        Plot accuracy as a function of inference cost for different training costs.
        
        Args:
            accuracy_df (pandas.DataFrame, optional): DataFrame with results
            C_tr_values (list, optional): Specific training costs to plot
            figsize (tuple, optional): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Check if we have data
        if accuracy_df is None:
            raise ValueError("No data provided. Please provide accuracy_df.")
        
        # Filter by policy
        policy_id = self.policy_type.value
        df = accuracy_df[accuracy_df['Policy'] == policy_id].copy()
        
        # No data for this policy
        if len(df) == 0:
            raise ValueError(f"No data found for policy {policy_id}")
        
        # Filter specific C_tr values if provided
        if C_tr_values:
            df = df[df['C_tr'].isin(C_tr_values)].copy()
        
        # Sort the dataframe
        df = df.sort_values(["C_tr", "C_inf"])
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each C_tr as a separate curve
        for C_tr, group in df.groupby("C_tr"):
            ax.plot(group["C_inf"], group["Accuracy"], marker='o', label=f'{C_tr:.2e}')
        
        # Add legend
        ax.legend(title="Training Compute")
        
        # Set log scale
        ax.set_xscale('log')
        
        # Labels
        ax.set_xlabel("Inference Compute")
        ax.set_ylabel("Expected Accuracy")
        ax.set_title(f"Expected Accuracy vs Inference Compute ({policy_id})")
        
        # Get data range
        min_x = df["C_inf"].min()
        max_x = df["C_inf"].max()
        
        # Expand the x limits slightly to ensure visibility
        log_min = np.floor(np.log10(min_x))
        log_max = np.ceil(np.log10(max_x))
        ax.set_xlim(10**log_min, 10**log_max)
        
        # Force grid to be behind the data
        ax.set_axisbelow(True)
        
        # COMPLETELY DIFFERENT APPROACH TO GRID LINES:
        # 1. First clear any automatic grid
        ax.grid(False)
        
        # 2. Manually draw major grid lines at powers of 10
        major_ticks = [10**i for i in range(int(log_min), int(log_max)+1)]
        for tick in major_ticks:
            ax.axvline(x=tick, color='black', linestyle='-', linewidth=0.7, alpha=0.4, zorder=0)
        
        # 3. Manually draw minor grid lines
        for decade in range(int(log_min), int(log_max)+1):
            for i in [2, 3, 4, 5, 6, 7, 8, 9]:
                ax.axvline(x=i*10**decade, color='gray', linestyle=':', linewidth=0.7, alpha=0.6, zorder=0)
        
        # Configure major ticks (powers of 10)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
        
        # Configure minor ticks in between powers of 10
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2,3,4,5,6,7,8,9]))
        
        # Optionally remove labels from minor ticks
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        return fig
    
    def plot_accuracy_vs_C_tr(self, accuracy_df=None, C_inf_values=None, figsize=(10, 6), save_path=None):
        """
        Plot accuracy as a function of training cost for different inference costs.
        
        Args:
            accuracy_df (pandas.DataFrame, optional): DataFrame with results
            C_inf_values (list, optional): Specific inference costs to plot
            figsize (tuple, optional): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Check if we have data
        if accuracy_df is None:
            raise ValueError("No data provided. Please provide accuracy_df.")
        
        # Filter by policy
        policy_id = self.policy_type.value
        df = accuracy_df[accuracy_df['Policy'] == policy_id].copy()
        
        # No data for this policy
        if len(df) == 0:
            raise ValueError(f"No data found for policy {policy_id}")
        
        # Filter specific C_inf values if provided
        if C_inf_values:
            df = df[df['C_inf'].isin(C_inf_values)].copy()
        
        # Sort the dataframe by C_inf
        df = df.sort_values(["C_inf", "C_tr"])
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each C_inf as a separate curve
        for C_inf, group in df.groupby("C_inf"):
            ax.plot(group["C_tr"], group["Accuracy"], marker='o', label=f'{C_inf:.2e}')
        
        # Add legend
        ax.legend(title="Inference Compute")
        
        # Set log scale
        ax.set_xscale('log')
        
        # Labels
        ax.set_xlabel("Training Compute")
        ax.set_ylabel("Expected Accuracy")
        ax.set_title(f"Expected Accuracy vs Training Compute ({policy_id})")
        
        # Get data range
        min_x = df["C_tr"].min()
        max_x = df["C_tr"].max()
        
        # Expand the x limits slightly to ensure visibility
        log_min = np.floor(np.log10(min_x))
        log_max = np.ceil(np.log10(max_x))
        ax.set_xlim(10**log_min, 10**log_max)
        
        # Force grid to be behind the data
        ax.set_axisbelow(True)
        
        # Enable major and minor grid
        ax.grid(True, which='major', color='black', linestyle='-', linewidth=0.7, alpha=0.4)
        ax.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
        
        # Configure major ticks (powers of 10)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
        
        # Configure minor ticks in between powers of 10 (2,3,4,5,6,7,8,9)
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2,3,4,5,6,7,8,9], numticks=12))
        
        # Optionally remove labels from minor ticks
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        return fig
    
    def plot_accuracy_vs_tokens(self, accuracy_df=None, C_tr_values=None, figsize=(10, 6), save_path=None):
        """
        Plot accuracy as a function of inference tokens for different training costs.
        
        Args:
            accuracy_df (pandas.DataFrame, optional): DataFrame with results
            C_tr_values (list, optional): Specific training costs to plot
            figsize (tuple, optional): Figure size
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Check if we have data
        if accuracy_df is None:
            raise ValueError("No data provided. Please provide accuracy_df.")
        
        # Filter by policy
        policy_id = self.policy_type.value
        df = accuracy_df[accuracy_df['Policy'] == policy_id].copy()
        
        # No data for this policy
        if len(df) == 0:
            raise ValueError(f"No data found for policy {policy_id}")
        
        # Filter specific C_tr values if provided
        if C_tr_values:
            df = df[df['C_tr'].isin(C_tr_values)].copy()
        
        # Sort the dataframe
        df = df.sort_values(["C_tr", "Tokens"])
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each C_tr as a separate curve
        for C_tr, group in df.groupby("C_tr"):
            ax.plot(group["Tokens"], group["Accuracy"], label=f'{C_tr:.2e}')
        

        # TESTING
        # Load 'Claude_3_7_Sonnet_Estimated_Aime_df.csv'
        cl = pd.read_csv('Claude_3_7_Sonnet_Estimated_Aime_df.csv')
        plt.plot(cl['Tokens'], cl['Accuracy'], marker='o', label='Estimated AIME')

        # Add legend
        ax.legend(title="Training Compute")
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        
        # Labels
        ax.set_xlabel("Inference Tokens")
        ax.set_ylabel("Expected Accuracy")
        ax.set_title(f"Expected Accuracy vs Inference Tokens ({policy_id})")
        
        # Get data range
        min_x = df["Tokens"].min()
        max_x = df["Tokens"].max()
        
        # Expand the x limits slightly to ensure visibility
        log_min = np.floor(np.log10(min_x))
        log_max = np.ceil(np.log10(max_x))
        ax.set_xlim(10**log_min, 10**log_max)
        
        # Force grid to be behind the data
        ax.set_axisbelow(True)
        
        # Draw major grid lines at powers of 10
        major_ticks = [10**i for i in range(int(log_min), int(log_max)+1)]
        for tick in major_ticks:
            ax.axvline(x=tick, color='black', linestyle='-', linewidth=0.7, alpha=0.4, zorder=0)
        
        # Draw minor grid lines
        for decade in range(int(log_min), int(log_max)+1):
            for i in [2, 3, 4, 5, 6, 7, 8, 9]:
                ax.axvline(x=i*10**decade, color='gray', linestyle=':', linewidth=0.7, alpha=0.6, zorder=0)
        
        # Configure major ticks (powers of 10)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
        
        # Configure minor ticks in between powers of 10
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2,3,4,5,6,7,8,9]))
        
        # Optionally remove labels from minor ticks
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        return fig
    
    def plot_contour(self, accuracy_df=None, inference_counts=None, figsize=(12, 10), 
                     save_path=None, xlim=None, ylim=None):
        """
        Create a contour plot of accuracy as a function of training and inference costs.
        Optionally overlay curves showing optimal allocation for different numbers of tasks.
        
        Args:
            accuracy_df (pandas.DataFrame, optional): DataFrame with results
            inference_counts (list, optional): List of inference counts to plot optimal curves for
            figsize (tuple, optional): Figure size
            save_path (str, optional): Path to save the figure
            xlim (tuple, optional): x-axis limits (min, max)
            ylim (tuple, optional): y-axis limits (min, max)
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Check if we have data
        if accuracy_df is None:
            raise ValueError("No data provided. Please provide accuracy_df.")
        
        # Convert inference_counts to numpy array to avoid issues with log10
        if inference_counts is not None:
            inference_counts = np.array(inference_counts, dtype=float)

        # Filter by policy
        policy_id = self.policy_type.value
        df = accuracy_df[accuracy_df['Policy'] == policy_id].copy()
        
        # No data for this policy
        if len(df) == 0:
            raise ValueError(f"No data found for policy {policy_id}")
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data from dataframe
        C_tr_values = df['C_tr'].values
        C_inf_values = df['C_inf'].values
        accuracy_values = df['Accuracy'].values
        
        # Create a regular grid for interpolation in log space
        log_tr_min, log_tr_max = np.log10(C_tr_values.min()), np.log10(C_tr_values.max())
        log_inf_min, log_inf_max = np.log10(C_inf_values.min()), np.log10(C_inf_values.max())
        
        grid_size = 100
        log_tr_grid = np.linspace(log_tr_min, log_tr_max, grid_size)
        log_inf_grid = np.linspace(log_inf_min, log_inf_max, grid_size)
        log_tr_mesh, log_inf_mesh = np.meshgrid(log_tr_grid, log_inf_grid)
        
        # Points need to be in log space for better interpolation
        log_points = np.column_stack((np.log10(list(C_tr_values)), np.log10(list(C_inf_values))))
        
        # Interpolate accuracy values onto the log grid
        grid_accuracy = griddata(log_points, accuracy_values, (log_tr_mesh, log_inf_mesh), method='linear')
        # Replace NaN values with zeros to avoid plotting issues
        grid_accuracy = np.nan_to_num(grid_accuracy, nan=0.0)
        grid_accuracy[grid_accuracy < 0] = 0.0
        
        # Convert mesh back to linear scale for plotting
        tr_mesh = 10**log_tr_mesh
        inf_mesh = 10**log_inf_mesh
        
        # Create contour levels excluding values above 0.99
        min_acc = max(0, accuracy_values.min())
        max_acc = min(0.99, accuracy_values.max())  # Cap at 0.99
        # max_acc = min(1.0, accuracy_values.max())  # Cap at 0.99
        levels = np.linspace(min_acc, max_acc, 10)
        
        # Define colormap for consistency
        cmap = plt.cm.viridis
        
        # Create colored contour lines with increased transparency
        contour_lines = ax.contour(tr_mesh, inf_mesh, grid_accuracy, levels=levels, 
                                  cmap=cmap, linewidths=1.0, alpha=0.7)  # More transparent
        
        # Add scatter points with increased transparency
        scatter = ax.scatter(C_tr_values, C_inf_values, c=accuracy_values, cmap=cmap,
                            edgecolor='k', s=15, alpha=0.7, zorder=5)  # Smaller and more transparent
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Expected Accuracy')
        cbar.ax.set_yticks(levels)
        cbar.ax.set_yticklabels([f'{level:.2f}' for level in levels])
        
        # Create an interpolator for accuracy as a function of C_tr and C_inf
        accuracy_interp = LinearNDInterpolator(log_points, accuracy_values)
        
        # Define the range of inference counts if provided
        if len(inference_counts)>0:
            # Create a red colormap with progressively darker shades
            # Use a reversed "Reds" colormap so larger values of I get darker shades
            red_cmap = cm.get_cmap('Reds')
            optimal_curve_colors = red_cmap(np.linspace(0.5, 1, len(inference_counts)))
            
            # For each inference count, find the optimal compute allocation
            for i, I in enumerate(inference_counts):
                # Create a range of total compute values to explore
                C_tot_values = np.logspace(log_tr_min, log_tr_max + np.log10(I*10**log_inf_max), 50)
                
                optimal_tr = []
                optimal_inf = []
                optimal_acc = []
                
                for C_tot in C_tot_values:
                    best_acc = 0
                    best_tr = 0
                    best_inf = 0
                    
                    # Sample different allocations of compute to find optimal
                    for ratio in np.linspace(0.01, 0.99, 50):
                        C_tr_test = C_tot * ratio
                        C_inf_test = (C_tot - C_tr_test) / I
                        
                        # Skip if C_inf is too small (below minimum in dataset)
                        if C_inf_test < 10**log_inf_min:
                            continue
                            
                        # Skip if C_tr is too small (below minimum in dataset)
                        if C_tr_test < 10**log_tr_min:
                            continue
                            
                        # Skip if values exceed our data range
                        if np.log10(C_tr_test) > log_tr_max or np.log10(C_inf_test) > log_inf_max:
                            continue
                        
                        # Get predicted accuracy
                        acc = accuracy_interp(np.log10(C_tr_test), np.log10(C_inf_test))
                        
                        # Update if this is better
                        if not np.isnan(acc) and acc > best_acc:
                            best_acc = acc
                            best_tr = C_tr_test
                            best_inf = C_inf_test
                
                # Only add if accuracy is below 0.99 and we found a valid solution
                if best_acc > 0 and best_acc <= 1.0:
                    optimal_tr.append(best_tr)
                    optimal_inf.append(best_inf)
                    optimal_acc.append(best_acc)
                
                # Plot the optimal curve with thicker lines to stand out
                if optimal_tr:  # Only if we have valid points
                    ax.plot(optimal_tr, optimal_inf, '-', color=optimal_curve_colors[i], 
                            linewidth=2.5, label=f'{I:.0e}', zorder=10)
        
        # Set log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Set axis limits if provided
        if xlim:
            ax.set_xlim(*xlim)
        else:
            # Default limits with some margin
            ax.set_xlim(10**log_tr_min * 0.5, 10**log_tr_max * 2)
            
        if ylim:
            ax.set_ylim(*ylim)
        else:
            # Default limits with some margin
            ax.set_ylim(10**log_inf_min * 0.5, 10**log_inf_max * 2)
        
        # Set axis labels and title
        ax.set_xlabel('Training Compute')
        ax.set_ylabel('Inference Compute per Task')
        ax.set_title(f'Optimal Compute Allocation for Different Numbers of Tasks ({policy_id})')
        
        # Add legend for inference counts if we plotted them
        if len(inference_counts)>0 and any(optimal_tr for optimal_tr in optimal_tr):
            ax.legend(title='Number of Tasks', loc='best')
        
        # Manually draw grid lines with increased transparency
        ax.grid(False)
        
        # X-axis major grid lines
        for decade in range(int(log_tr_min), int(log_tr_max)+1):
            ax.axvline(x=10**decade, color='black', linestyle='-', linewidth=0.7, alpha=0.3, zorder=0)
        
        # Y-axis major grid lines
        for decade in range(int(log_inf_min), int(log_inf_max)+1):
            ax.axhline(y=10**decade, color='black', linestyle='-', linewidth=0.7, alpha=0.3, zorder=0)
        
        # X-axis minor grid lines
        for decade in range(int(log_tr_min), int(log_tr_max)+1):
            for i in [2, 3, 4, 5, 6, 7, 8, 9]:
                ax.axvline(x=i*10**decade, color='gray', linestyle=':', linewidth=0.7, alpha=0.3, zorder=0)
        
        # Y-axis minor grid lines
        for decade in range(int(log_inf_min), int(log_inf_max)+1):
            for i in [2, 3, 4, 5, 6, 7, 8, 9]:
                ax.axhline(y=i*10**decade, color='gray', linestyle=':', linewidth=0.7, alpha=0.3, zorder=0)
        
        # Configure axis ticks
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2,3,4,5,6,7,8,9]))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2,3,4,5,6,7,8,9]))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        
        return fig
    
    def plot_tokens_contour(self, accuracy_df=None, inference_counts=None, figsize=(12, 10), 
                      save_path=None, xlim=None, ylim=None):
        """
        Create a contour plot of accuracy as a function of training cost and inference tokens.
        
        Args:
            accuracy_df (pandas.DataFrame, optional): DataFrame with results
            inference_counts (list, optional): List of inference counts to plot optimal curves for
            figsize (tuple, optional): Figure size
            save_path (str, optional): Path to save the figure
            xlim (tuple, optional): x-axis limits (min, max)
            ylim (tuple, optional): y-axis limits (min, max)
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Check if we have data
        if accuracy_df is None:
            raise ValueError("No data provided. Please provide accuracy_df.")
        
        # Filter by policy
        policy_id = self.policy_type.value
        df = accuracy_df[accuracy_df['Policy'] == policy_id].copy()
        
        # No data for this policy
        if len(df) == 0:
            raise ValueError(f"No data found for policy {policy_id}")
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data from dataframe
        C_tr_values = df['C_tr'].values
        tokens_values = df['Tokens'].values
        accuracy_values = df['Accuracy'].values
        
        # Create a regular grid for interpolation in log space
        log_tr_min, log_tr_max = np.log10(C_tr_values.min()), np.log10(C_tr_values.max())
        log_tokens_min, log_tokens_max = np.log10(tokens_values.min()), np.log10(tokens_values.max())
        
        grid_size = 100
        log_tr_grid = np.linspace(log_tr_min, log_tr_max, grid_size)
        log_tokens_grid = np.linspace(log_tokens_min, log_tokens_max, grid_size)
        log_tr_mesh, log_tokens_mesh = np.meshgrid(log_tr_grid, log_tokens_grid)
        
        # Points need to be in log space for better interpolation
        log_points = np.column_stack((np.log10(C_tr_values), np.log10(tokens_values)))
        
        # Interpolate accuracy values onto the log grid
        grid_accuracy = griddata(log_points, accuracy_values, (log_tr_mesh, log_tokens_mesh), method='linear')
        
        # Convert mesh back to linear scale for plotting
        tr_mesh = 10**log_tr_mesh
        tokens_mesh = 10**log_tokens_mesh
        
        # Create contour levels excluding values above 0.99
        min_acc = max(0, np.nanmin(accuracy_values))
        max_acc = min(0.99, np.nanmax(accuracy_values))  # Cap at 0.99
        levels = np.linspace(min_acc, max_acc, 10)
        
        # Define colormap for consistency
        cmap = plt.cm.viridis
        
        # Create colored contour lines with increased transparency
        contour_lines = ax.contour(tr_mesh, tokens_mesh, grid_accuracy, levels=levels, 
                                cmap=cmap, linewidths=1.0, alpha=0.7)
        
        # Add scatter points with increased transparency
        scatter = ax.scatter(C_tr_values, tokens_values, c=accuracy_values, cmap=cmap,
                            edgecolor='k', s=15, alpha=0.7, zorder=5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Expected Accuracy')
        cbar.ax.set_yticks(levels)
        cbar.ax.set_yticklabels([f'{level:.2f}' for level in levels])
        
        # Set log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Set axis limits if provided
        if xlim:
            ax.set_xlim(*xlim)
        else:
            # Default limits with some margin
            ax.set_xlim(10**log_tr_min * 0.5, 10**log_tr_max * 2)
            
        if ylim:
            ax.set_ylim(*ylim)
        else:
            # Default limits with some margin
            ax.set_ylim(10**log_tokens_min * 0.5, 10**log_tokens_max * 2)
        
        # Set axis labels and title
        ax.set_xlabel('Training Compute')
        ax.set_ylabel('Inference Tokens per Task')
        ax.set_title(f'Accuracy as a Function of Training Compute and Inference Tokens ({policy_id})')
        
        # Manually draw grid lines with increased transparency
        ax.grid(False)
        
        # X-axis major grid lines
        for decade in range(int(log_tr_min), int(log_tr_max)+1):
            ax.axvline(x=10**decade, color='black', linestyle='-', linewidth=0.7, alpha=0.3, zorder=0)
        
        # Y-axis major grid lines
        for decade in range(int(log_tokens_min), int(log_tokens_max)+1):
            ax.axhline(y=10**decade, color='black', linestyle='-', linewidth=0.7, alpha=0.3, zorder=0)
        
        # X-axis minor grid lines
        for decade in range(int(log_tr_min), int(log_tr_max)+1):
            for i in [2, 3, 4, 5, 6, 7, 8, 9]:
                ax.axvline(x=i*10**decade, color='gray', linestyle=':', linewidth=0.7, alpha=0.3, zorder=0)
        
        # Y-axis minor grid lines
        for decade in range(int(log_tokens_min), int(log_tokens_max)+1):
            for i in [2, 3, 4, 5, 6, 7, 8, 9]:
                ax.axhline(y=i*10**decade, color='gray', linestyle=':', linewidth=0.7, alpha=0.3, zorder=0)
        
        # Configure axis ticks
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2,3,4,5,6,7,8,9]))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2,3,4,5,6,7,8,9]))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        return fig


class BestOfNModel(ModelFramework):
    """
    Extension for Best of N policy.
    """
    
    def __init__(self, n_samples=3, **kwargs):
        """
        Initialize with custom parameters for Best of N.
        
        Args:
            n_samples (int): Number of samples to take
            **kwargs: Passed to parent class
        """
        kwargs['policy_type'] = InferencePolicy.BEST_OF_N
        kwargs['policy_params'] = kwargs.get('policy_params', {})
        kwargs['policy_params']['n_samples'] = n_samples
        super().__init__(**kwargs)
    
    def evaluate_allocation_best_of_N(self, C_tr, C_inf, l, m, p_values=None, gamma_values=None):
        """
        Evaluate allocation with Best of N for fixed l and m.
        
        Args:
            C_tr (float): Training cost
            C_inf (float): Inference cost
            l (int): Skill level
            m (int): Difficulty parameter
            p_values (list, optional): Pre-computed p values
            gamma_values (list, optional): Pre-computed gamma values
            
        Returns:
            tuple: (accuracy, actual_inference_cost)
        """
        # Ensure p_values and gamma_values are available
        if p_values is None or gamma_values is None:
            p_values, gamma_values = self.train(C_tr)
        
        # Get n_samples from policy_params
        n_samples = self.policy_params.get('n_samples', 3)
        
        # Calculate R from C_tr
        R = np.sqrt(C_tr / (6 * self.kappa * self.zeta**2))

        # Compute steps per sample from C_inf
        K_max_per_sample = C_inf / (n_samples * 2 * self.zeta * R * self.omega)
        
        # Find best l_star for this allocation
        best_accuracy = 0
        cost = n_samples * 2 * self.zeta * R * self.omega * K_max_per_sample
        
        # Find if any p_values is 1 and if so store index
        p_is_1_index = np.where(np.array(p_values) == 1)[0]
        l_min = 1 if len(p_is_1_index) == 0 else p_is_1_index[-1]

        # Try different l_star values
        for l_prime in range(l_min, self.L_max + 1):
            p_l_prime = p_values[l_prime]
            gamma_l_prime = gamma_values[l_prime]
            
            # Skip if gamma is 0
            if gamma_l_prime == 0:
                break
                
            gamma_l_prime = gamma_values[l_prime]
            m_l_prime = super().compute_m_l_prime(l, l_prime, m)
            M_l_prime = super().compute_M_l_prime(m_l_prime)
            p_eff = p_l_prime * (self.rho / M_l_prime + 1-self.rho)
            r_l_prime = p_eff * self.q(p_l_prime)
            
            # Calculate base accuracy for a single sample
            base_accuracy = super().compute_accuracy_COT(gamma_l_prime, m_l_prime, r_l_prime, K_max_per_sample)
            
            # Calculate Best-of-N accuracy
            accuracy = 1 - (1 - base_accuracy) ** n_samples
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Calculate expected steps per sample and total actual inference cost
                expected_steps_per_sample = super().compute_expected_steps_cot(gamma_l_prime, m_l_prime, r_l_prime, K_max_per_sample)
                cost = n_samples * 2 * self.zeta * R * self.omega * expected_steps_per_sample
        
        best_accuracy = min(best_accuracy, 1.0)

        return best_accuracy, cost


class MarkovTreeSearchModel(ModelFramework):
    """
    Extension for Markov Tree Search policy.
    """
    
    def __init__(self, branching_factor=3, **kwargs):
        """
        Initialize with custom parameters for Markov Tree Search.
        
        Args:
            branching_factor (int): Branching factor of the search tree
            **kwargs: Passed to parent class
        """
        kwargs['policy_type'] = InferencePolicy.MARKOV_TREE_SEARCH
        kwargs['policy_params'] = kwargs.get('policy_params', {})
        kwargs['policy_params']['branching_factor'] = branching_factor
        super().__init__(**kwargs)
    
    def evaluate_allocation_markov_tree_search(self, C_tr, C_inf, l, m, p_values=None, gamma_values=None):
        """
        Evaluate allocation with Markov Tree Search for fixed l and m.
        
        Args:
            C_tr (float): Training cost
            C_inf (float): Inference cost
            l (int): Skill level
            m (int): Difficulty parameter
            p_values (list, optional): Pre-computed p values
            gamma_values (list, optional): Pre-computed gamma values
            
        Returns:
            tuple: (accuracy, actual_inference_cost)
        """
        # Ensure p_values and gamma_values are available
        if p_values is None or gamma_values is None:
            p_values, gamma_values = self.train(C_tr)
        
        # Get parameters from policy_params
        branching_factor = self.policy_params.get('branching_factor', 3)
        
        # Calculate R from C_tr
        R = np.sqrt(C_tr / (6 * self.kappa * self.zeta**2))

        # Compute effective steps from C_inf (accounting for branching factor)
        K_max_effective = C_inf / (branching_factor * 2 * self.zeta * R * self.omega)
        
        # Find best l_star for this allocation
        best_accuracy = 0
        cost = branching_factor * 2 * self.zeta * R * self.omega * K_max_effective
        
        # Find if any p_values is 1 and if so store index
        p_is_1_index = np.where(np.array(p_values) == 1)[0]
        l_min = 1 if len(p_is_1_index) == 0 else p_is_1_index[-1]

        # Try different l_star values
        for l_prime in range(l_min, self.L_max + 1):
            p_l_prime = p_values[l_prime]
            gamma_l_prime = gamma_values[l_prime]
            
            # Skip if gamma is 0
            if gamma_l_prime == 0:
                break
                
            gamma_l_prime = gamma_values[l_prime]
            m_l_prime = super().compute_m_l_prime(l, l_prime, m)
            M_l_prime = super().compute_M_l_prime(m_l_prime)
            p_eff = p_l_prime * (self.rho / M_l_prime + 1-self.rho)
            r_l_prime = p_eff * self.q(p_l_prime)
            
            # Improved r for MCTS
            r_mcts = 1 - (1 - r_l_prime) ** branching_factor
            
            # Calculate accuracy using MCTS approach
            accuracy = super().compute_accuracy_COT(gamma_l_prime, m_l_prime, r_mcts, K_max_effective)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Calculate expected steps and actual inference cost
                expected_steps = super().compute_expected_steps_cot(gamma_l_prime, m_l_prime, r_mcts, K_max_effective)
                cost = branching_factor * 2 * self.zeta * R * self.omega * expected_steps
        
        return best_accuracy, cost

class ConsensusModel(ModelFramework):
    """
    Extension for Consensus Voting policy.
    """
    
    def __init__(self, consensus_count, alphabet_size=2, **kwargs):
        """
        Initialize with custom parameters for Consensus Voting.
        
        Args:
            consensus_count (int): Number of trials for consensus
            **kwargs: Passed to parent class
        """
        # Create a new enum value for Consensus
        # (assuming InferencePolicy is an Enum that can be extended)
        if not hasattr(InferencePolicy, 'CONSENSUS'):
            InferencePolicy.CONSENSUS = "consensus"
            
        kwargs['policy_type'] = InferencePolicy.CONSENSUS
        kwargs['policy_params'] = kwargs.get('policy_params', {})
        kwargs['policy_params']['consensus_count'] = consensus_count
        kwargs['policy_params']['alphabet_size'] = alphabet_size
        super().__init__(**kwargs)


    def consensus_accuracy_multialphabet(self, N_con, p, A): # IN PROGRESS NOT CORRECT
        """
        Calculate consensus accuracy with alphabet size A
        
        Parameters:
        N_con: Number of trials
        p: Base accuracy (probability of correct answer)
        A: Alphabet size (number of possible answers)
        
        Returns:
        Probability of correct consensus
        """
        
        # For alphabet size 2, use the efficient betainc calculation
        if A == 2:
            maj_threshold = np.ceil((N_con + 1) / 2)
            return betainc(maj_threshold, N_con-maj_threshold+1, p)
        
        # For alphabet size > 2, calculate the full probability
        total_prob = 0
        
        # Probability for each incorrect answer
        p_incorrect = (1-p)/(A-1)
        
        # For each possible number of correct answers
        for k in range(1, N_con + 1):  # k=0 won't win
            # Probability of getting exactly k correct answers
            p_k_correct = binom.pmf(k, N_con, p)
            
            # For alphabet size > 2, we need to calculate the probability 
            # that the maximum count of any incorrect answer is < k
            
            # Probability any single incorrect answer appears fewer than k times
            p_below_k = binom.cdf(k-1, N_con-k, p_incorrect/(1-p))
            
            # All A-1 incorrect answers must be below k
            p_all_below = p_below_k ** (A-1)
            
            total_prob += p_k_correct * p_all_below
        
        return total_prob
    
    def evaluate_allocation_consensus(self, C_tr, C_inf, l, m, p_values=None, gamma_values=None):
        """
        Evaluate allocation with Consensus for fixed l and m.
        
        Args:
            C_tr (float): Training cost
            C_inf (float): Inference cost
            l (int): Skill level
            m (int): Difficulty parameter
            p_values (list, optional): Pre-computed p values
            gamma_values (list, optional): Pre-computed gamma values
            
        Returns:
            tuple: (accuracy, actual_inference_cost)
        """
        # Ensure p_values and gamma_values are available
        if p_values is None or gamma_values is None:
            p_values, gamma_values = self.train(C_tr)
        
        # Get consensus_count from policy_params
        N_con = self.policy_params.get('consensus_count')
        alphabet_size = self.policy_params.get('alphabet_size',2)
        
        # Calculate R from C_tr
        R = np.sqrt(C_tr / (6 * self.kappa * self.zeta**2))

        # Compute steps per trial from C_inf
        K_max_per_trial = C_inf / (N_con * 2 * self.zeta * R * self.omega)
        
        # Find best l_star for this allocation
        best_base_accuracy = 0
        best_expected_steps = 0
        accuracy=0
        cost = N_con * 2 * self.zeta * R * self.omega * K_max_per_trial
        
        # Find if any p_values is 1 and if so store index
        p_is_1_index = np.where(np.array(p_values) == 1)[0]
        l_min = 1 if len(p_is_1_index) == 0 else p_is_1_index[-1]

        # Try different l_star values
        for l_prime in range(l_min, self.L_max + 1):
            p_l_prime = p_values[l_prime]
            gamma_l_prime = gamma_values[l_prime]
            
            # Skip if gamma is 0
            if gamma_l_prime == 0:
                break
                
            gamma_l_prime = gamma_values[l_prime]
            m_l_prime = super().compute_m_l_prime(l, l_prime, m)
            M_l_prime = super().compute_M_l_prime(m_l_prime)
            p_eff = p_l_prime * (self.rho / M_l_prime + 1-self.rho)
            r_l_prime = p_eff * self.q(p_l_prime)
            
            # Base policy success probability
            base_accuracy = super().compute_accuracy_COT(gamma_l_prime, m_l_prime, r_l_prime, K_max_per_trial)
            
            if base_accuracy > best_base_accuracy:
                best_base_accuracy = base_accuracy
                # Calculate expected steps per trial and total actual inference cost
                expected_steps_per_trial = super().compute_expected_steps_cot(gamma_l_prime, m_l_prime, r_l_prime, K_max_per_trial)
                best_expected_steps = expected_steps_per_trial

        # Consensus accuracy calculation
        if best_base_accuracy > 0:
            # Calculate accuracy using binomial distribution

            # For consensus voting, we need to find the majority threshold
            # maj_threshold is the minimum number of successes needed for majority
            
            maj_threshold = np.ceil((N_con+1) / alphabet_size)
            accuracy = betainc(maj_threshold, N_con-maj_threshold+1, best_base_accuracy)
            """
            if alphabet_size == 2:
                # Binary case - use efficient betainc function
                maj_threshold = np.ceil((N_con + 1) / 2)
                accuracy = betainc(maj_threshold, N_con-maj_threshold+1, best_base_accuracy)
            else:
                # Multi-alphabet case
                accuracy = self.consensus_accuracy_multialphabet(N_con, best_base_accuracy, alphabet_size)
            """

            cost = N_con * 2 * self.zeta * R * self.omega * best_expected_steps
        
        return accuracy, cost

class CustomDistributionModel(ModelFramework):
    """
    Extension with a custom distribution function.
    """
    
    def __init__(self, distribution_func=None, **kwargs):
        """
        Initialize with a custom distribution function.
        
        Args:
            distribution_func (callable, optional): Custom phi distribution function
            **kwargs: Passed to parent class
        """
        super().__init__(**kwargs)
        self._distribution_func = distribution_func
    
    def phi(self, l, m):
        """
        Use custom distribution function if provided.
        
        Args:
            l (int): Skill level
            m (int): Difficulty parameter
            
        Returns:
            float: Probability
        """
        if self._distribution_func:
            return self._distribution_func(l, m)
        else:
            return super().phi(l, m)


class CustomEtaSigmaModel(ModelFramework):
    """
    Extension with custom eta_l and sigma_l functions.
    """
    
    def __init__(self, eta_func=None, sigma_func=None, **kwargs):
        """
        Initialize with custom eta_l and sigma_l functions.
        
        Args:
            eta_func (callable, optional): Custom eta_l function
            sigma_func (callable, optional): Custom sigma_l function
            **kwargs: Passed to parent class
        """
        super().__init__(**kwargs)
        self._eta_func = eta_func
        self._sigma_func = sigma_func
    
    def eta_l(self, l):
        """
        Use custom eta_l function if provided.
        
        Args:
            l (int): Skill level
            
        Returns:
            float: eta_l value
        """
        if self._eta_func:
            return self._eta_func(l)
        else:
            return super().eta_l(l)
    
    def sigma_l(self, l):
        """
        Use custom sigma_l function if provided.
        
        Args:
            l (int): Skill level
            
        Returns:
            float: sigma_l value
        """
        if self._sigma_func:
            return self._sigma_func(l)
        else:
            return super().sigma_l(l)


def run_example(constants, base_C_inf_values, base_C_tr_values, con_count=64):
    """
    Example demonstrating how to use the framework with various model types.
    Tests each inference policy with appropriate budget scaling.
    """
    # Define constants
    """
    constants = {
        'L_min': 20,  # Minimum skill level
        'L_max': 80, # Maximum skill level
        'm_min': 10, # Minimum number of skills
        'm_max': 20, # Maximum number of skills
        'S_l': 1.1e4, # Number of skills per level
        'd_t': 6, # Degrees of text pieces (number of skills required to understand a text) is binomially distributed with a fixed mean degree, d_t
        'zeta': 2.5e3, # Parameters per concept
        'tau': 2e4, # Tokens per training text piece
        'omega': 25, # Output tokens per skill step
        'kappa': 20, # D_tr / N for chinchilla optimal
        'beta': 50, # Factor to scale skills required to relevant set
        'rho': .7 # Search efficiency between relevant and required sets (1 moves curve up and spreads out; 0 moves curve down and narrows)
    }
    """
    
    # Define base ranges for grid evaluation
    
    # Initialize results dataframe
    all_results_df = pd.DataFrame()
    
    # 1. TEST BASELINE COT MODEL
    print("Evaluating baseline COT model...")
    cot_model = ModelFramework(constants=constants, policy_type=InferencePolicy.COT)
    
    # Evaluate with baseline budget
    # cot_df = cot_model.evaluate_grid(
    #    base_C_tr_values, base_C_inf_values, 
    #    output_file="accuracy_results_cot.csv",
    #    replace=True
    # )

    cot_df = cot_model.evaluate_grid_parallel(
        base_C_tr_values, base_C_inf_values, 
        output_file="accuracy_results_cot.csv",
        replace=True,
        n_jobs=4
    )
    
    # Add to combined results
    all_results_df = pd.concat([all_results_df, cot_df], ignore_index=True)
    # Plot results
    try:
        plot_model_results(cot_model, cot_df, "cot")
    except Exception as e:
        print(f"Error plotting for COT model: {e}")
    """
    # Plot results
    try:
        plot_model_results(cot_model, cot_df, "cot")
    except Exception as e:
        print(f"Error plotting for COT model: {e}")

    cot_model.plot_accuracy_vs_tokens(
        accuracy_df=cot_df,
        save_path=f"accuracy_vs_tokens_{cot_model}.png"
    )
    
    # 2. TEST MCTS MODEL WITH 3X BUDGET
    print("Evaluating MCTS model with 3x inference budget...")
    branching_factor = 3
    mcts_model = MarkovTreeSearchModel(
        branching_factor=branching_factor, 
        constants=constants
    )
    
    # Scale inference budget by branching factor
    mcts_C_inf_values = base_C_inf_values * branching_factor
    
    # Evaluate with scaled budget
    mcts_df = mcts_model.evaluate_grid_parallel(
        base_C_tr_values, mcts_C_inf_values, 
        output_file="accuracy_results_mcts.csv",
        replace=True
    )
    
    # Add to combined results
    all_results_df = pd.concat([all_results_df, mcts_df], ignore_index=True)
    
    # Plot results
    try:
        plot_model_results(mcts_model, mcts_df, "mcts")
    except Exception as e:
        print(f"Error plotting for MCTS model: {e}")
    
    # 3. TEST BEST-OF-N MODEL WITH Nx BUDGET
    print("Evaluating Best-of-N model with scaled inference budget...")
    n_samples = 16
    bon_model = BestOfNModel(n_samples=n_samples, constants=constants)
    
    # Scale inference budget by n_samples
    bon_C_inf_values = base_C_inf_values * n_samples
    
    # Evaluate with scaled budget
    bon_df = bon_model.evaluate_grid_parallel(
        base_C_tr_values, bon_C_inf_values, 
        output_file="accuracy_results_best_of_n.csv",
        replace=True
    )
    
    # Add to combined results
    all_results_df = pd.concat([all_results_df, bon_df], ignore_index=True)
    
    # Plot results
    try:
        plot_model_results(bon_model, bon_df, "best_of_n")
    except Exception as e:
        print(f"Error plotting for Best-of-N model: {e}")
    
    # 4. TEST CONSENSUS MODEL WITH Nx BUDGET
    if hasattr(InferencePolicy, 'CONSENSUS'):
        print("Evaluating Consensus model with scaled inference budget...")
        consensus_count = con_count
        consensus_model = ConsensusModel(consensus_count=consensus_count, alphabet_size=100, constants=constants)
        
        # Scale inference budget by consensus_count
        consensus_C_inf_values = np.array(base_C_inf_values) * consensus_count

        
        # Evaluate with scaled budget
        consensus_df = consensus_model.evaluate_grid_parallel(
            base_C_tr_values, consensus_C_inf_values, 
            output_file="accuracy_results_consensus.csv",
            replace=True
        )
        # Add to combined results
        all_results_df = pd.concat([all_results_df, consensus_df], ignore_index=True)
        
        # Plot results
        try:
            plot_model_results(consensus_model, consensus_df, "consensus")
        except Exception as e:
            print(f"Error plotting for Consensus model: {e}")
    """
    # Save combined results
    #all_results_df.to_csv("all_models_results.csv", index=False)
    
    # Create comparison plots
    #try:
    #    plot_model_comparisons(all_results_df)
    #except Exception as e:
    #    print(f"Error creating comparison plots: {e}")
    
    #print("Evaluation complete. Results saved.")
    
    return all_results_df

def plot_accuracy_vs_train_cost(constants, token_budget, base_C_tr_values):
    cot_model = ModelFramework(constants=constants, policy_type=InferencePolicy.COT)




def plot_model_results(model, df, model_name):
    """Helper function to generate all plots for a single model"""
    # Accuracy vs. Inference Cost
    fig1 = model.plot_accuracy_vs_C_inf(
        accuracy_df=df,
        save_path=f"accuracy_vs_inf_{model_name}.png"
    )
    
    # # Accuracy vs. Training Cost
    fig2 = model.plot_accuracy_vs_C_tr(
        accuracy_df=df,
        save_path=f"accuracy_vs_tr_{model_name}.png"
    )
    
    # Contour plot
    # inference_counts = [1, 1e6, 1e9, 1e12]
    # inference_counts = [1e10, 1e14, 1e18, 1e22] 
    inference_counts = np.logspace(10, 30, 20)   
    fig3 = model.plot_contour(
        accuracy_df=df,
        inference_counts=inference_counts,
        save_path=f"contour_plot_{model_name}.png"
    )
    
    # Tokens vs. Accuracy
    fig4 = model.plot_accuracy_vs_tokens(
        accuracy_df=df,
        save_path=f"accuracy_vs_tokens_{model_name}.png"
    )
    
    # Tokens vs. Training Cost
    fig5 = model.plot_tokens_contour(
        accuracy_df=df,
        save_path=f"tokens_contour_{model_name}.png", xlim=(1e25, 1e26), ylim=(1e3, 1e5)
    )
    
    plt.close('all')  # Close all figures to free memory

def plot_model_comparisons(all_results_df):
    """Create comparison plots across different models"""
    # Select a few representative C_tr values
    C_tr_values = sorted(all_results_df['C_tr'].unique())
    selected_C_tr = [C_tr_values[1], C_tr_values[4], C_tr_values[-1]]  # Low, mid, high
    
    for C_tr in selected_C_tr:
        # Filter data for this C_tr
        df_subset = all_results_df[all_results_df['C_tr'] == C_tr].copy()
        
        # Accuracy vs. scaled inference (divide by multiplier)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for policy in df_subset['Policy'].unique():
            policy_data = df_subset[df_subset['Policy'] == policy]
            # Sort by increasing C_inf
            policy_data = policy_data.sort_values('C_inf')
            
            # Determine multiplier based on policy (COT=1, MCTS=3, Best-of-N=5, etc.)
            multiplier = 1
            if policy == "markov_tree_search":
                multiplier = 3
            elif policy == "best_of_n":
                multiplier = 5
            elif policy == "consensus":
                multiplier = 5
                
            # Plot normalized C_inf vs. accuracy
            ax.plot(policy_data['C_inf'] / multiplier, policy_data['Accuracy'], 
                   marker='o', label=policy)
        
        ax.set_xscale('log')
        ax.set_xlabel('Normalized Inference Compute (per single inference)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy vs. Normalized Inference Compute (C_tr={C_tr:.2e})')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'comparison_accuracy_vs_inf_C_tr_{C_tr:.2e}.png', dpi=300)
        plt.close()
        
        # Accuracy vs. scaled tokens
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for policy in df_subset['Policy'].unique():
            policy_data = df_subset[df_subset['Policy'] == policy]
            # Sort by increasing Tokens
            policy_data = policy_data.sort_values('Tokens')
            
            # Determine multiplier based on policy
            multiplier = 1
            if policy == "markov_tree_search":
                multiplier = 3
            elif policy == "best_of_n":
                multiplier = 5
            elif policy == "consensus":
                multiplier = 5
                
            # Plot normalized Tokens vs. accuracy
            ax.plot(policy_data['Tokens'] / multiplier, policy_data['Accuracy'], 
                   marker='o', label=policy)
        
        ax.set_xscale('log')
        ax.set_xlabel('Normalized Inference Tokens (per single inference)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy vs. Normalized Inference Tokens (C_tr={C_tr:.2e})')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'comparison_accuracy_vs_tokens_C_tr_{C_tr:.2e}.png', dpi=300)
        plt.close()

if __name__ == "__main__":
    # run_example()
    constants = {
        'L': 20, # Total number of skill levels
        'L_min': 5,  # Minimum skill level task
        'L_max': 10, # Maximum skill level task
        'm_min': 2, # Minimum number of skills
        'm_max': 30, # Maximum number of skills
        'S_l': 1e3, # Number of skills per level
        'd_t': 6, # Degrees of text pieces (number of skills required to understand a text) is binomially distributed with a fixed mean degree, d_t
        'zeta': 2.5e3, # Parameters per concept
        'tau': 1e4, # Tokens per training text piece
        'omega': 25, # Output tokens per inference step
        'kappa': 20, # D_tr / N for chinchilla optimal
        'beta': 2, # Factor to scale skills required to relevant set
        'rho': 1, # Search efficiency between relevant and required sets
        'delta': 1e-5, # Small value
        'X_MIN': 0.1,
        'NUM_PTS': 40 # 
    }
    
    # Base values for compute
    base_C_tr_values = np.logspace(10, 30, 10)
    base_C_inf_values = np.logspace(10, 30, 10)    

    run_example(constants, base_C_inf_values, base_C_tr_values, con_count=64)

    


    