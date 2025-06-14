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

contour_type = "tokens_ctr"

if contour_type == "cinf_ctr":

    print("Evaluating baseline COT model...")
    cot_model = ModelFramework(constants=constants, policy_type=InferencePolicy.COT)

    # load data
    cot_df = pd.read_csv('accuracy_results_cot_ref.csv')

    cot_model.plot_contour(accuracy_df=cot_df, inference_counts=[1e7, 1e9, 1e12], figsize=(12, 10), save_path="AIME_accuracy_vs_cinf_ctr_claude.png", xlim=[], ylim=None)

elif contour_type == "tokens_ctr":
    print("Evaluating baseline AIME model...")
    cot_model = ModelFramework(constants=constants, policy_type=InferencePolicy.COT)

    # load data
    cot_df = pd.read_csv('accuracy_results_cot_ref.csv')    


    cot_model.plot_tokens_contour(accuracy_df=cot_df, inference_counts=None, figsize=(12, 10), save_path="AIME_accuracy_vs_tokens_ctr_claude.png", xlim=(1e24, 1e27), ylim=(1e0, 1e8))


brkpnt1 = 1

