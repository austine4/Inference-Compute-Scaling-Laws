from compute_optimal_utils_v6 import *

FONT_SIZE = 14
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["font.size"] = str(FONT_SIZE)

# run_example()
constants = {
    'L': 100,        # Maximum total skill level
    'L_min': 60,    # Minimum task skill level
    'L_max': 70,    # Maximum task skill level
    'm_min': 2,     # Minimum number of skills
    'm_max': 15,    # Maximum number of skills
    'S_l': 1e4,   # Number of skills per level
    'd_t': 6,       # Degrees of text pieces
    'zeta': 2.5e3,  # Parameters per concept
    'tau': 1e4,     # Tokens per training text piece
    'omega': 25,    # Output tokens per skill
    'kappa': 20,    # D_tr / N for chinchilla optimal
    'beta': 5,     # Factor to scale skills required to relevant set
    'rho': 0.5     # Search efficiency between relevant and required sets
}


# Base values for compute
# base_C_tr_values = np.logspace(24, 27, 20)
# base_C_inf_values = np.logspace(10, 30, 60) 

# base_C_tr_values = np.logspace(24, 27, 4)
# base_C_inf_values = np.logspace(10, 30, 10) 


base_C_tr_values = np.logspace(np.log10(1e25), np.log10(1e27), 50)
base_C_inf_values = np.logspace(np.log10(1e13), np.log10(5e17), 60)

run_example(constants, base_C_inf_values, base_C_tr_values, con_count=64)

