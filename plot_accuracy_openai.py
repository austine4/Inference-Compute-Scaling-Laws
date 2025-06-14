import compute_optimal_utils_v6 as utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea
from matplotlib.ticker import FixedLocator, FixedFormatter
import os

FONT_SIZE = 14
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["font.size"] = str(FONT_SIZE)
plt.rcParams['figure.figsize'] = 12, 10
NUM_PTS = 100

# OpenAI

# Constants # Larger range of skill levels
# works with xi_l = np.exp(-10*l/L)
# change: sigma_l = 0.5*log(l), m_min = 2, S_l = 1e4
c = {
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

# Constants # Larger range of skill levels
# works with xi_l = np.exp(-10*l/L)
# change: sigma_l = 0.5*log(l), m_min = 2, S_l = 1e4

# c = {
#     'L': 100,        # Maximum total skill level
#     'L_min': 60,    # Minimum task skill level
#     'L_max': 70,    # Maximum task skill level
#     'm_min': 2,     # Minimum number of skills
#     'm_max': 15,    # Maximum number of skills
#     'S_l': 5e3,   # Number of skills per level
#     'd_t': 6,       # Degrees of text pieces
#     'zeta': 2.5e3,  # Parameters per concept
#     'tau': 1e4,     # Tokens per training text piece
#     'omega': 25,    # Output tokens per skill
#     'kappa': 22,    # D_tr / N for chinchilla optimal
#     'beta': 0,     # Factor to scale skills required to relevant set
#     'rho': 0     # Search efficiency between relevant and required sets
# }

# # Constants # Works awesome. Good job Austin!
# # works with xi_l = np.exp(-10*l/L)
# c = {
#     'L': 30,        # Maximum total skill level
#     'L_min': 15,    # Minimum task skill level
#     'L_max': 20,    # Maximum task skill level
#     'm_min': 5,     # Minimum number of skills
#     'm_max': 15,    # Maximum number of skills
#     'S_l': 2.5e4,   # Number of skills per level
#     'd_t': 6,       # Degrees of text pieces
#     'zeta': 2.5e3,  # Parameters per concept
#     'tau': 1e4,     # Tokens per training text piece
#     'omega': 25,    # Output tokens per skill
#     'kappa': 20,    # D_tr / N for chinchilla optimal
#     'beta': 5,     # Factor to scale skills required to relevant set
#     'rho': 0.5     # Search efficiency between relevant and required sets
# }

# # Constants
# c = {
#     'L': 30,        # Maximum total skill level
#     'L_min': 20,    # Minimum task skill level
#     'L_max': 22,    # Maximum task skill level
#     'm_min': 2,     # Minimum number of skills
#     'm_max': 18,    # Maximum number of skills
#     'S_l': 2.5e4,   # Number of skills per level
#     'd_t': 6,       # Degrees of text pieces
#     'zeta': 2.5e3,  # Parameters per concept
#     'tau': 1e4,     # Tokens per training text piece
#     'omega': 25,    # Output tokens per skill
#     'kappa': 20,    # D_tr / N for chinchilla optimal
#     'beta': 8,     # Factor to scale skills required to relevant set
#     'rho': 1     # Search efficiency between relevant and required sets
# }
 
# TRAINING
# C_inf_values = np.array([1e17])
# target_tokens = 64e3
# target_tokens = 100e3
target_tokens = 64e3
# C_inf_values = np.array([6e16])
# C_tr_values = np.logspace(np.log10(1.2e25),np.log10(3.8e25),15)

# C_tr_values = np.logspace(np.log10(1.7e25),np.log10(3.8e25),2)
# C_inf_values = np.logspace(16, 17.5, 20)

# C_tr_values = np.logspace(np.log10(1.7e25),np.log10(3.8e25), 10)
# C_inf_values = np.logspace(16, 18, 20)

# # Run overnight
C_tr_values = np.logspace(np.log10(1.7e25),np.log10(3.8e25), 50)
C_inf_values = np.logspace(16, 17.5, 50)

# C_tr_values = np.logspace(np.log10(1.7e25),np.log10(3.8e25), 3)
# C_inf_values = np.logspace(16, 17.5, 20)

# C_tr_values = np.array([3.8e25])
# delete accuracy_results_cot.csv 
# the following line only if file exists
if os.path.exists('accuracy_results_cot_claude_train.csv'):
    os.remove('accuracy_results_cot_claude_train.csv')

utils.run_example(c, base_C_inf_values= C_inf_values, base_C_tr_values=C_tr_values)
# rename accuracy_results_cot as accuracy_results_cot_claude
os.rename('accuracy_results_cot.csv', 'accuracy_results_cot_claude_train.csv')
cot_df = pd.read_csv('accuracy_results_cot_claude_train.csv')
print(cot_df)

 
# (Assume cot_df is your DataFrame already loaded with the data)
# Select a specific 'C_inf_allocated' value to plot:
c_inf = cot_df['C_inf_allocated'][0]#.unique()[42]
plotty = cot_df[cot_df['C_inf_allocated'] == c_inf]
 
fig1, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12, 6))

plt.subplot(121)

accs = []
max_err = 0
# Plot the data:
for C_tr in np.unique(cot_df['C_tr']):
    c_tr_sub_df = cot_df[cot_df['C_tr']==C_tr]
    # C_tr dataframe
    # token_list = np.array(c_tr_sub_df['Tokens'])
    N = np.sqrt(C_tr/(6*c['kappa']))
    token_list = np.array(c_tr_sub_df['C_inf_allocated'])/(2*N)
    # Find the index of the closest value to target_tokens
    idx = (np.abs(token_list - target_tokens)).argmin()
    # print the difference between tokens
    if np.abs(token_list[idx] - target_tokens)>max_err:
        max_err = np.abs(token_list[idx] - target_tokens)
        print( token_list[idx], target_tokens, max_err)
    accs.append(c_tr_sub_df[ np.array(c_tr_sub_df['C_inf_allocated'])/(2*N) == token_list[idx]]['Accuracy'])
plt.plot(C_tr_values, 100*np.array(accs), color='k', label=fr'Token Budget = {target_tokens:.1e}')
plt.legend()
plt.ylim((0,100))
plt.show(block=False)
print(f"Max error in token budget: {max_err:.2f}")
 
# Set the x-axis to log scale and overall limits:
ax1.set_xscale('log')
L_target = 1.6e25#1e25  # This corresponds to our "nice" value 1
U_target = 3.8e25  # This corresponds to our "nice" value 100
# ax1.set_xlim(L_target, U_target)
 
# Compute A and B for our transformation:
A = np.log10(L_target)
B = np.log10(U_target)
 
# Define the "nice" tick values spanning two decades:
# These represent the ticks: 1,2,3,4,5,6,7,8,9,10,20,30,...,90,100.
base_ticks = np.array([ 2, 3, 4, 5, 6, 7, 8, 9,
                       10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
 
# Map these "nice" values to actual x-axis positions using:
# x = 10^(A + (log10(T)/2)*(B - A))
new_ticks = 10 ** (A + (np.log10(base_ticks) / 2) * (B - A))
 
# Create a list of labels where only the first, tenth, and last ticks are labeled.
# (base_ticks[0] corresponds to T=1, base_ticks[9] to T=10, and base_ticks[18] to T=100)
labels = []
for i, tick in enumerate(new_ticks):
    if i in [8, 17]:
        labels.append(f"{tick:.2e}")
    else:
        labels.append("")
 
# Set the major ticks to these positions and apply our custom formatter:
ax1.xaxis.set_major_locator(FixedLocator(new_ticks))
ax1.xaxis.set_major_formatter(FixedFormatter(labels))
 
# Optionally, remove minor ticks and adjust tick mark length:
ax1.minorticks_off()
ax1.tick_params(axis='x', which='major', length=6)
 
ax1.set_ylim(0,100)
# plt.xlim(1.1e25,3.8e25)
 
# Label the axes and add a title:
ax1.set_xlabel('train-time compute (log scale)', fontsize=14)
ax1.set_ylabel('pass@1 accuracy', fontsize=14)
ax1.set_title('AIME accuracy during training', fontsize=16) 
# plt.show(block=False)
 

## INFERENCE
# C_inf_values = np.logspace(np.log10(1e13),np.log10(1e17),10)
# C_tr_values = np.array([4.56e24])

# C_inf_values = np.logspace(np.log10(1e15),np.log10(1e17),2)
# C_tr_values = np.array([3.8e25])

## Run overnight
C_inf_values = np.logspace(np.log10(1e15),np.log10(1e17),50)
C_tr_values = np.array([3.8e25])

if os.path.exists('accuracy_results_cot_claude_inf.csv'):
    os.remove('accuracy_results_cot_claude_inf.csv')
utils.run_example(c, base_C_inf_values= C_inf_values, base_C_tr_values=C_tr_values)
os.rename('accuracy_results_cot.csv', 'accuracy_results_cot_claude_inf.csv')
cot_df = pd.read_csv('accuracy_results_cot_claude_inf.csv')
 
# (Assume cot_df is your DataFrame already loaded with the data)
# Select a specific 'C_inf_allocated' value to plot:
c_tr = cot_df['C_tr'][0]
plotty = cot_df[cot_df['C_tr'] == c_tr]
 
# fig, ax = plt.subplots(figsize=(10, 8))

# Plot the data:
ax2.plot(plotty['C_inf'], 100*plotty['Accuracy'], color='k', label=f'Model (Training FLOPs = {c_tr:.1e})')
ax2.legend()

# Set the x-axis to log scale and overall limits:
ax2.set_xscale('log')
L_target = 1.2e15  # This corresponds to our "nice" value 1
U_target = 8e16  # This corresponds to our "nice" value 100
# ax2.set_xlim(L_target, U_target)
 
# Compute A and B for our transformation:
A = np.log10(L_target)
B = np.log10(U_target)
 
# Define the "nice" tick values spanning two decades:
# These represent the ticks: 1,2,3,4,5,6,7,8,9,10,20,30,...,90,100.
base_ticks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,
                       10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
 
# Map these "nice" values to actual x-axis positions using:
# x = 10^(A + (log10(T)/2)*(B - A))
new_ticks = 10 ** (A + (np.log10(base_ticks) / 2) * (B - A))
 
# Create a list of labels where only the first, tenth, and last ticks are labeled.
# (base_ticks[0] corresponds to T=1, base_ticks[9] to T=10, and base_ticks[18] to T=100)
labels = []
for i, tick in enumerate(new_ticks):
    if i in [1, 9, 18]:
        labels.append(f"{tick:.2e}")
    else:
        labels.append("")
 
# Set the major ticks to these positions and apply our custom formatter:
ax2.xaxis.set_major_locator(FixedLocator(new_ticks))
ax2.xaxis.set_major_formatter(FixedFormatter(labels))
 
# Optionally, remove minor ticks and adjust tick mark length:
ax2.minorticks_off()
ax2.tick_params(axis='x', which='major', length=6)
 
ax2.set_ylim(0,100)
# ax2.set_xlim(7.5e14, 7.5e16)
 
# Label the axes and add a title:
ax2.set_xlabel('test-time compute (log scale)', fontsize=14)
ax2.set_ylabel('pass@1 accuracy', fontsize=14)
ax2.set_title('AIME accuracy at test time', fontsize=16)

# save the figure
# save figure fig not the last one

fig1.savefig('AIME_accuracy_vs_train_inf_openai.png', dpi=300, bbox_inches='tight')
 
plt.show(block=False)

brkpnt1 = 1
