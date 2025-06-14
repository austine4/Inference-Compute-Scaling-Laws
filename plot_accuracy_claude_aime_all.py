import compute_optimal_utils_v6 as utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea
from matplotlib.ticker import FixedLocator, FixedFormatter
 
FONT_SIZE = 16
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["font.size"] = str(FONT_SIZE)
plt.rcParams['figure.figsize'] = 10, 8
NUM_PTS = 100

# Anthropic

# Constants # Need to adjust slope
# works with xi_l = np.exp(-(2*l/self.L+5))
# c = {
#     'L':200,        # Maximum total skill level
#     'L_min': 105,    # Minimum task skill level
#     'L_max': 115,    # Maximum task skill level
#     'm_min': 2,     # Minimum number of skills
#     'm_max': 10,    # Maximum number of skills
#     'S_l': 1e4,   # Number of skills per level
#     'd_t': 6,       # Degrees of text pieces
#     'zeta': 2.5e3,  # Parameters per concept
#     'tau': 1e4,     # Tokens per training text piece
#     'omega': 25,    # Output tokens per skill
#     'kappa': 20,    # D_tr / N for chinchilla optimal
#     'beta': 5,     # Factor to scale skills required to relevant set
#     'rho': 0.5     # Search efficiency between relevant and required sets
# }

# Constants # Works awesome. Larger range of skill levels
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

C_inf_values = np.logspace(14,20,20)
C_tr_values = np.array([3.35e25])
utils.run_example(c, base_C_inf_values= C_inf_values, base_C_tr_values=C_tr_values)
 
# Load data
df_3_25 = pd.read_csv('accuracy_results_cot.csv')
claude = pd.read_csv('Claude_3_7_Sonnet_Estimated_Aime_df.csv')
 
# Calculate R using the formula
C_tr = 3.35e25
R = np.sqrt(C_tr / (6 * c['kappa'] * c['zeta']**2))
 
# Calculate K_max and allocated tokens correctly
df_3_25['K_max'] = df_3_25['C_inf_allocated'] / (2 * c['zeta'] * R * c['omega'])
df_3_25['allocated_tokens'] = df_3_25['K_max'] * c['omega']
 
# Sort the data by allocated_tokens for finding the specific budget points
df_sorted = df_3_25.sort_values('allocated_tokens')
 
# Define the exact token budgets we want to mark
token_budgets = [2000, 4000, 8000, 16000, 32000, 64000]
 
# Find the closest points in the data to our desired token budgets
budget_points = []
for budget in token_budgets:
    # Find index of the closest value
    idx = (np.abs(df_sorted['allocated_tokens'] - budget)).argmin()
    budget_points.append(df_sorted.iloc[idx])
   
# Print the exact allocated token values we found
for i, budget in enumerate(token_budgets):
    print(f"Closest point to {budget} tokens: {budget_points[i]['allocated_tokens']:.2f} tokens")

# plt.close("all")
# # Create log scale figure
# fig1, ax1 = plt.subplots(figsize=(12, 9))
fig1, (ax1, ax2) = plt.subplots(1,2,figsize=(22, 10), gridspec_kw={'width_ratios': [1.5, 1]})
# different figure size for two subplots


# Plot the actual tokens used (blue line)
ax1.plot(df_3_25['Tokens'], df_3_25['Accuracy'], linestyle='-', linewidth=2, color='b',
         label='Model (3.35e25 Training FLOPs)', zorder =10)
 

# Plot Claude's results
ax1.scatter(claude['Tokens'], claude['Accuracy'], marker='o', color='orange', 
           label='Claude 3.7 Sonnet', zorder =10)
 
# Set log scale for x-axis
ax1.set_xscale('log')
 
# Set limits for x-axis
ax1.set_xlim(5e2, 1.5e5)
ax1.set_ylim(0.08, 0.75)
# ax1.set_xlim(1e0, 6e4)
 
# Set axis labels and title
ax1.set_xlabel('Tokens (Log Scale)', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.legend(loc='best', fontsize=14)
ax1.set_title('AIME Accuracy vs. Tokens with Budget Markers', fontsize=16)
# plt.grid(True, alpha=0.3)

# Function to format numbers with commas
def format_with_commas(number):
    return f"{number:,}"
 
# Add budget markers - blue labels on the left side
for i, point in enumerate(budget_points):
    # Add the exact budget label (e.g., "2k budget")
    budget_in_k = token_budgets[i] // 1000
    # Format the tokens with commas
    formatted_tokens = format_with_commas(int(point['Tokens']))
    budget_label = f"{budget_in_k}k budget\n{formatted_tokens} tokens"
    
    # Special positioning for 4k and 16k budgets
    if budget_in_k in [4, 16]:
        x_offset = -80  # Move further left
    else:
        x_offset = -50  # Standard offset
   
    # Place the label with offset to the left
    offsetbox = TextArea(budget_label, textprops=dict(color='b', size=9, weight='bold'))
   
    # The arrow points from the box to the exact point on the curve
    # Make arrowhead solid and smaller
    ab = AnnotationBbox(offsetbox,
                        (point['Tokens'], point['Accuracy']),  # Point to the actual token point on curve
                        xybox=(x_offset, 0),  # Adjusted offset
                        box_alignment=(1, 0.5),
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle='->', color='b', 
                                        connectionstyle="angle3",
                                        shrinkA=0, shrinkB=3,  # Smaller arrowhead
                                        mutation_scale=8))     # Smaller arrowhead
    ax1.add_artist(ab)
 
# Identify Claude 64k point for special handling
claude_64k = None
for i, row in claude.iterrows():
    if 'Budget' in row and not pd.isna(row['Budget']) and row['Budget'] == 64000:
        claude_64k = row
 
# Add the 64k Claude point at the start (to make sure it's on bottom)
if claude_64k is not None:
    # Format with commas
    formatted_tokens = format_with_commas(int(claude_64k['Tokens']))
    budget_label = f"64k budget\n{formatted_tokens} tokens"
           
    # Add label with special positioning
    offsetbox = TextArea(budget_label, textprops=dict(color='orange', size=9, weight='bold'))
    ab = AnnotationBbox(offsetbox,
                        (claude_64k['Tokens'], claude_64k['Accuracy']),
                        xybox=(150, 0),  # Offset to the right
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle='->', color='orange', 
                                      connectionstyle="angle3",
                                      shrinkA=0, shrinkB=3,    # Smaller arrowhead
                                      mutation_scale=8))      # Smaller arrowhead
    ax1.add_artist(ab)
       
# Add budget labels to Claude points - with tighter boxes (except for 64k)
for i, row in claude.iterrows():
    if 'Budget' in row and not pd.isna(row['Budget']) and row['Budget'] != 64000:
        # Format the budget label
        if row['Budget'] >= 1000:
            budget_in_k = int(row['Budget']/1000)
            # Format the tokens with commas
            formatted_tokens = format_with_commas(int(row['Tokens']))
            budget_label = f"{budget_in_k}k budget\n{formatted_tokens} tokens"
        else:
            formatted_tokens = format_with_commas(int(row['Tokens']))
            budget_label = f"{int(row['Budget'])} budget\n{formatted_tokens} tokens"
           
        # Add label to Claude point with offset to the right and tighter box
        offsetbox = TextArea(budget_label, textprops=dict(color='orange', size=9, weight='bold'))
        ab = AnnotationBbox(offsetbox,
                            (row['Tokens'], row['Accuracy']),
                            xybox=(100, 0),  # Offset to the right
                            boxcoords="offset points",
                            arrowprops=dict(arrowstyle='->', color='orange', 
                                          connectionstyle="angle3",
                                          shrinkA=0, shrinkB=3,  # Smaller arrowhead
                                          mutation_scale=8))    # Smaller arrowhead
        ax1.add_artist(ab)
 
# Plot the actual tokens used (blue line)
ax2.plot(df_3_25['Tokens'], df_3_25['Accuracy'], linestyle='-', color='b',
         label='Model (3.35e25 Training FLOPs)', zorder=10)
 
# Plot Claude's results
ax2.scatter(claude['Tokens'], claude['Accuracy'], marker='o', color='orange',
           label='Claude 3.7 Sonnet', zorder=10)
 
# Set log scale for x-axis
ax2.set_xscale('log')
ax2.set_xlabel('Tokens (Log Scale)', fontsize=14)
ax2.set_xlim(1e2, 1e7)

# save figure
fig1.savefig('AIME_Accuracy_vs_Tokens_with_Budget_Markers.png', dpi=300, bbox_inches='tight')
 
plt.tight_layout()
plt.show(block=False)

brkpnt1 = 1