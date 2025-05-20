# main script for plotting sum of p_l over all skill levels
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import *
import scipy as scp
from compute_optimal_utils_v6 import *
import sympy as smp
from scipy.special import lambertw

FONT_SIZE = 14
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["font.size"] = str(FONT_SIZE)
plt.rcParams['figure.figsize'] = 12, 10
NUM_PTS = 100

if __name__ == "__main__":
    # Set parameters
    dt = 6
    eps = 0.5
    
    # Create a grid for R and T values
    min_R, max_R = 1e2, 1e12
    min_T, max_T = 1e2, 1e12
    
    R_vec = np.logspace(np.log10(min_R), np.log10(max_R), NUM_PTS)
    T_vec = np.logspace(np.log10(min_T), np.log10(max_T), NUM_PTS)
    
    # Create a 2D grid for R and T
    R_grid, T_grid = np.meshgrid(R_vec, T_vec)
    
    # Initialize matrix to store sum of pl values
    sum_pl_matrix = np.zeros((NUM_PTS, NUM_PTS))
    
    # Setup GCC ratio calculation using Lambert W function
    S, l, T, d, a, G, p = smp.symbols("S l T d a G p")
    k = smp.simplify(S*p)  # average degree of G_2^{(l)}
    expr = G - (1-smp.exp(-a*G))
    roots = smp.solve(expr, G)
    G_sol = roots[0].subs(a, k)
    gcc_ratio_func = smp.lambdify([p, S], G_sol)
    
    # Parameters for pl calculation
    L = 100
    L_vec = np.arange(1, L+1, 1)
    Sl = 1e3  # Size of skill space
    Sl_vec = Sl * np.ones(L)
    
    # Create skill levels with different parameters
    eta_l = 5
    sigma_l_vec = np.round(np.log2(L_vec))
    xi_vec = np.exp(-L_vec/20)        
    
    # generate a range of kappas
    kappa_vec = np.logspace(np.log10(10), np.log10(40), 100)

    # C_tr_vec = [1e18, 1e20, 1e22]
    C_tr_vec = [1e20, 1e22, 1e24]
    zeta_val = 1e4
    tau_val = 4e4
    # varsigma, tau = 2e5, 8e5

    fmwk = ModelFramework()

    for ind, Ctr in enumerate(C_tr_vec):
        print(f"Calculating pl values for Ctr = {Ctr}...")
        N = np.sqrt(Ctr/(6*kappa_vec))
        R = N/zeta_val
        D = kappa_vec*N
        T = D/tau_val
        FLOPS_vec = R*T
        
        sum_pl_vec = np.zeros(len(R))
        for ind2, FLOPS in enumerate(FLOPS_vec):            
            # Get model parameters
            R_opt, _, _, _, Pb, _, _ = fmwk.num_skills_learnt_biterr_infcmp(R[ind2], dt, FLOPS, eps, optimize_flag=False)
            Pbar = 1-Pb
            
            # Initialize prerequisites factor
            prereq_factor = 1.0
            sum_pl = 0.0
                        
            # Calculate sum of pl over all skill levels
            for ind_l, l in enumerate(L_vec):
                # Calculate Rc2 (skill pairs)
                mu_val = xi_vec[ind_l]*R[ind2]/Sl
                delta_val = 1e-10
                d_delta = 0.99*mu_val*(1-np.sqrt(2*np.log(Sl/delta_val)/(mu_val)))
                d_delta = max(d_delta, 0)
                
                # Calculate pl using Chernoff bound
                if eta_l > d_delta:
                    term1_pl = 0.0
                else:
                    if eta_l/d_delta < Pbar:
                        term1_pl = (1 - np.exp(-d_delta*fmwk.D_KL(eta_l/d_delta, Pbar)))
                    else:
                        term1_pl = (1/np.sqrt(2*d_delta))*np.exp(-d_delta*fmwk.D_KL(min(eta_l/d_delta, 1.0), Pbar))
                
                term2_pl = prereq_factor**(2*sigma_l_vec[ind_l])
                pl = term1_pl * term2_pl
                
                # Weight by skill distribution and add to sum
                sum_pl += pl
                
                # Update prerequisite factor for next level
                gcc_val = gcc_ratio_func(pl, Sl)
                prereq_factor = gcc_val if not np.isnan(gcc_val) else 0.0
            
            # Store sum of pl values
            sum_pl_vec[ind2] = sum_pl
        
        # plot sum_pl_vec vs kappa
        plt.plot(kappa_vec, sum_pl_vec, label=f'FLOPS = {C_tr_vec[ind]:.0e}')        
            
    plt.xscale('log')
    plt.grid()
    plt.legend()
    # Save figure
    plt.savefig('sum_pl_vs_kappa.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    brkpnt1 = 1