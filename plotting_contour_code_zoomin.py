import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import LinearNDInterpolator, griddata
import pandas as pd
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
 
# Set font
plt.rcParams.update({
    'font.family': 'STIXGeneral'
})
 
s_test=20
alpha_test=0.3
 
def sci(x):
    """Format scientific notation without leading zeros in exponent."""
    # Convert to float if it's a string
    if isinstance(x, str):
        x = float(x)
   
    # Format with scientific notation
    s = "{:e}".format(x)
   
    # Split into coefficient and exponent
    if 'e' in s:
        coef, exp = s.split('e')
       
        # Clean up coefficient
        if coef.endswith('.0'):
            coef = coef[:-2]
        elif coef.endswith('.'):
            coef = coef[:-1]
        else:
            coef = coef.rstrip('0').rstrip('.')
           
        # Clean up exponent (remove leading zero)
        if exp.startswith('+'):
            exp = exp[1:]  # Remove +
        if exp.startswith('-0'):
            exp = '-' + exp[2:]  # Keep the minus but remove leading zero
        elif exp.startswith('0'):
            exp = exp[1:]  # Just remove leading zero
           
        return f"{coef}e{exp}"
    else:
        return s.rstrip('0').rstrip('.')
 
def plot_contour_fixed(ax=None, accuracy_df=None, inference_counts=None, figsize=(12, 10),
                     save_path=None, xlim=None, ylim=None, is_colorbar=True, global_norm=None):
    """
    Fixed version of the contour plot function with consistent color normalization.
    """
    # Check if we have data
    df = accuracy_df.copy()
   
    # Extract data from dataframe
    C_tr_values = df['C_tr'].values
    C_inf_values = df['C_inf'].values
    accuracy_values = df['Accuracy'].values
   
    # Filter out zero accuracies for better interpolation
    valid_mask = accuracy_values > 0
    C_tr_values = C_tr_values[valid_mask]
    C_inf_values = C_inf_values[valid_mask]
    accuracy_values = accuracy_values[valid_mask]
   
    print(f"After filtering zeros: {len(C_tr_values)} points")
   
    # Create a regular grid for interpolation in log space
    log_tr_min, log_tr_max = np.log10(C_tr_values.min()), np.log10(C_tr_values.max())
    log_inf_min, log_inf_max = np.log10(C_inf_values.min()), np.log10(C_inf_values.max())
   
    print(f"Log ranges - C_tr: [{log_tr_min:.1f}, {log_tr_max:.1f}], C_inf: [{log_inf_min:.1f}, {log_inf_max:.1f}]")
   
    grid_size = 100
    log_tr_grid = np.linspace(log_tr_min, log_tr_max, grid_size)
    log_inf_grid = np.linspace(log_inf_min, log_inf_max, grid_size)
    log_tr_mesh, log_inf_mesh = np.meshgrid(log_tr_grid, log_inf_grid)
   
    # Points need to be in log space for better interpolation
    log_points = np.column_stack((np.log10(C_tr_values), np.log10(C_inf_values)))
   
    # Interpolate accuracy values onto the log grid
    grid_accuracy = griddata(log_points, accuracy_values, (log_tr_mesh, log_inf_mesh), method='linear')
   
    # Convert mesh back to linear scale for plotting
    tr_mesh = 10**log_tr_mesh
    inf_mesh = 10**log_inf_mesh
   
    # Use global normalization if provided, otherwise create based on data
    if global_norm is None:
        min_acc = max(0.02, accuracy_values.min())
        max_acc = min(0.98, accuracy_values.max())
        global_norm = plt.Normalize(vmin=min_acc, vmax=max_acc)
   
    # Create contour levels
    min_acc = max(0.02, accuracy_values.min())
    max_acc = min(0.98, accuracy_values.max())
    levels = np.linspace(min_acc, max_acc, 10)
   
    # Define colormap for consistency
    cmap = plt.cm.inferno
   
    # Create colored contour lines using global normalization
    contour_lines = ax.contour(tr_mesh, inf_mesh, grid_accuracy, levels=levels,
                              cmap=cmap, norm=global_norm, linewidths=1.0, alpha=1)
   
    # Add scatter points using global normalization
    scatter = ax.scatter(C_tr_values, C_inf_values, c=accuracy_values, cmap=cmap, norm=global_norm,
                         s=s_test, alpha=alpha_test, zorder=5)
   
    # Add colorbar using global normalization
    if is_colorbar:
        s_dummy = ax.scatter(C_tr_values, C_inf_values, c=accuracy_values, cmap=cmap, norm=global_norm, s=0, alpha=1, zorder=5)
        cbar = plt.colorbar(s_dummy, ax=ax)
        cbar.set_label('Accuracy', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
   
    # Create a more robust interpolator
    def get_accuracy_robust(log_tr, log_inf):
        # Try linear interpolation first
        acc = griddata(log_points, accuracy_values, (log_tr, log_inf), method='linear')
        if np.isnan(acc):
            # Fallback to nearest neighbor
            acc = griddata(log_points, accuracy_values, (log_tr, log_inf), method='nearest')
        return acc if not np.isnan(acc) else 0
   
    # Store curves for legend
    plotted_curves = []
   
    # Define the range of inference counts if provided
    if inference_counts:
        print(f"\nProcessing inference counts: {inference_counts}")
        optimal_curve_colors = ['#2FB8C6', '#338A94', '#015D67']
       
        for i, I in enumerate(inference_counts):
            print(f"\n--- Processing I = {I:.0e} ---")
           
            # More reasonable total compute range
            # Base it on actual data ranges
            C_tr_data_min = 10**log_tr_min
            C_tr_data_max = 10**log_tr_max
            C_inf_data_min = 10**log_inf_min
            C_inf_data_max = 10**log_inf_max
           
            # Total compute should be reasonable: C_tr + I * C_inf
            C_tot_min = C_tr_data_min + I * C_inf_data_min
            C_tot_max = C_tr_data_max + I * C_inf_data_max
           
            print(f"Initial C_tot range: {C_tot_min:.2e} to {C_tot_max:.2e}")
           
            # First pass: sample broadly to find where we get reasonable accuracies (5-95%)
            C_tot_broad = np.logspace(np.log10(C_tot_min), np.log10(C_tot_max), 50)
            good_C_tot_range = []
           
            for C_tot in C_tot_broad:
                max_acc_for_this_C_tot = 0
                # Quick sample to see what accuracy range we get
                for ratio in np.linspace(0.1, 0.9, 10):
                    C_tr_test = C_tot * ratio
                    C_inf_test = (C_tot - C_tr_test) / I
                   
                    if (C_tr_test < C_tr_data_min or C_tr_test > C_tr_data_max or
                        C_inf_test < C_inf_data_min or C_inf_test > C_inf_data_max):
                        continue
                    
                    log_tr_test = np.log10(C_tr_test)
                    log_inf_test = np.log10(C_inf_test)
                    acc = get_accuracy_robust(log_tr_test, log_inf_test)
                    max_acc_for_this_C_tot = max(max_acc_for_this_C_tot, acc)
               
                # Keep C_tot values that can achieve 5-95% accuracy
                if 0.05 <= max_acc_for_this_C_tot <= 0.95:
                    good_C_tot_range.append(C_tot)
            
            if len(good_C_tot_range) == 0:
                print(f"No good C_tot values found for I = {I:.0e} (no 5-95% accuracy range)")
                continue
               
            # Focus sampling on the good range
            C_tot_min_good = min(good_C_tot_range)
            C_tot_max_good = max(good_C_tot_range)
            print(f"Good C_tot range: {C_tot_min_good:.2e} to {C_tot_max_good:.2e}")
           
            # Dense sampling in the good range
            C_tot_values = np.logspace(np.log10(C_tot_min_good), np.log10(C_tot_max_good), 5)
           
            optimal_tr = []
            optimal_inf = []
            optimal_acc = []
           
            for j, C_tot in enumerate(C_tot_values):
                best_acc = 0
                best_tr = 0
                best_inf = 0
                valid_count = 0
               
                # Sample different allocations - use more fine-grained sampling
                for ratio in np.linspace(0.05, 0.95, 30):  # Increased resolution
                    C_tr_test = C_tot * ratio
                    C_inf_test = (C_tot - C_tr_test) / I
                   
                    # Check bounds more carefully
                    if (C_tr_test < C_tr_data_min or C_tr_test > C_tr_data_max or
                        C_inf_test < C_inf_data_min or C_inf_test > C_inf_data_max):
                        continue
                   
                    valid_count += 1
                   
                    # Get predicted accuracy
                    log_tr_test = np.log10(C_tr_test)
                    log_inf_test = np.log10(C_inf_test)
                    acc = get_accuracy_robust(log_tr_test, log_inf_test)
                   
                    # Update if this is better
                    if acc > best_acc:
                        best_acc = acc
                        best_tr = C_tr_test
                        best_inf = C_inf_test
               
                # Only keep points with reasonable accuracy (5% to 95%)
                if 0.05 <= best_acc <= 0.95 and valid_count > 0:
                    optimal_tr.append(best_tr)
                    optimal_inf.append(best_inf)
                    optimal_acc.append(best_acc)
           
            print(f"Found {len(optimal_tr)} optimal points for I = {I:.0e}")
            if len(optimal_tr) > 0:
                print(f"Accuracy range: {min(optimal_acc):.3f} to {max(optimal_acc):.3f}")
               
                # Verify all accuracies are in the good range
                if min(optimal_acc) < 0.05 or max(optimal_acc) > 0.95:
                    print(f"WARNING: Some accuracies outside 5-95% range, skipping curve")
                    continue
            else:
                print(f"No points found in 5-95% accuracy range, skipping curve")
                continue
           
            # Plot the optimal curve if we have enough points
            if len(optimal_tr) >= 3:  # Need at least 3 points for meaningful fitting
                print(f"Plotting curve for I = {sci(I)} with {len(optimal_tr)} points")
               
                optimal_tr = np.array(optimal_tr)
                optimal_inf = np.array(optimal_inf)
               
                # Sort by training compute
                sort_indices = np.argsort(optimal_tr)
                optimal_tr_sorted = optimal_tr[sort_indices]
                optimal_inf_sorted = optimal_inf[sort_indices]
               
                # Plot the points first
                line_plot = ax.plot(optimal_tr_sorted, optimal_inf_sorted, '-',
                                   color=optimal_curve_colors[i], linewidth=2.0,
                                   alpha=0.8, zorder=10, markersize=6)
               
                # Fit linear line in log space
                log_tr = np.log10(optimal_tr_sorted)
                log_inf = np.log10(optimal_inf_sorted)
               
                try:
                    coeffs = np.polyfit(log_tr, log_inf, 1)
                    slope, intercept = coeffs[0], coeffs[1]
                   
                    # Calculate R-squared
                    log_inf_pred = np.polyval(coeffs, log_tr)
                    ss_res = np.sum((log_inf - log_inf_pred) ** 2)
                    ss_tot = np.sum((log_inf - np.mean(log_inf)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                   
                    print(f"Fit: slope={slope:.2f}, intercept={intercept:.2f}, R²={r_squared:.3f}")
                   
                    # Only plot fitted line if fit is reasonable
                    if r_squared > 0.5:  # Only show fit if R² > 0.5
                        # Create fitted line
                        log_tr_line = np.linspace(log_tr.min() - 0.1, log_tr.max() + 0.1, 100)
                        log_inf_line = np.polyval(coeffs, log_tr_line)
                        tr_line = 10**log_tr_line
                        inf_line = 10**log_inf_line
                       
                        # Plot the fitted line
                        ax.plot(tr_line, inf_line, '--', color=optimal_curve_colors[i],
                               linewidth=2.0, alpha=0.8, zorder=9)
                       
                        # Create legend with fit
                        a = 10**intercept
                        b = slope
                        legend_label = f'I={sci(I)}: C_inf = {a:.1e} C_tr^{b:.2f} (R²={r_squared:.2f})'
                    else:
                        legend_label = f'I={sci(I)}: {len(optimal_tr)} points (poor fit)'
                       
                except Exception as e:
                    print(f"Fitting failed: {e}")
                    legend_label = f'{sci(I)}: {len(optimal_tr)} points'
               
                # Add to legend
                plotted_curves.append((line_plot[0], legend_label))
            else:
                print(f"Not enough points for I = {I:.0e} (only {len(optimal_tr)} points, need ≥3)")
   
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
   
    # Set axis limits
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
   
    # Set axis labels and title
    ax.set_xlabel('Training Compute', fontsize=16)
    ax.set_ylabel('Inference Compute per Task', fontsize=16)
    ax.set_title(f'Chinchilla Pretrained Total-Compute-Optimal Allocations (CoT)', fontsize=18)
   
    # Add legend if we have curves
    if plotted_curves:
        handles, labels = zip(*plotted_curves)
        # Bold FONT
        ax.legend(handles, labels, loc='upper right', prop={'weight': 'bold', 'size': 14})
        print(f"Legend added with {len(plotted_curves)} curves")
   
    # Add grid
    ax.grid(True, alpha=0.3)
 
def create_combined_plots():
    """Create combined subplot with larger plot on left, smaller plot with colorbar on right"""
   
    # Load data
    cot_df = pd.read_csv('accuracy_results_cot_fit2.csv')
   
    # Create a global normalization from 0 to 1 for consistent color mapping
    global_norm = plt.Normalize(vmin=0.0, vmax=1.0)
   
    print(f"Global color normalization range: 0.0 to 1.0")
   
    # Create figure with subplots - larger left plot, smaller right plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(48, 20), gridspec_kw={'width_ratios': [2, 2]})
   
    # ==================== LEFT PLOT: Larger contour plot ====================
    plot_contour_fixed(
        ax=ax1,
        accuracy_df=cot_df,
        inference_counts=[1e9, 1e11, 1e13],
        xlim=[1e25, 1e27],
        ylim=[3e13, 3e17],
        is_colorbar=False,  # No colorbar on left plot
        global_norm=global_norm
    )
   
    # Increase tick label sizes for left plot
    ax1.tick_params(axis='both', which='major', labelsize=14)
   
    # ==================== RIGHT PLOT: Smaller focused plot (15-17 range) with colorbar ====================
    y1 = 15
    y2 = 17
   
    # Filter data for the smaller plot
    cot_df_small = cot_df[(cot_df['C_inf'] < 10**y2) & (cot_df['C_inf'] > 10**y1)]
    cot_df_small = cot_df_small[(cot_df_small['C_tr'] < 1e26) & (cot_df_small['C_tr'] > 1e25)]
   
    # Extract arrays for fitting
    C_tr_vals = cot_df_small['C_tr'].values
    C_inf_vals = cot_df_small['C_inf'].values
    acc_vals = cot_df_small['Accuracy'].values
   
    # Fit the plane in log-space
    log_Ctr = np.log10(C_tr_vals)
    log_Cinf = np.log10(C_inf_vals)
    X = np.column_stack([-log_Ctr, acc_vals])
    model = LinearRegression().fit(X, log_Cinf)
    m, a, b = model.coef_[0], model.coef_[1], model.intercept_
    r2_score = model.score(X, log_Cinf)
   
    # Build log-log grid and interpolate accuracy
    log_tr_min, log_tr_max = log_Ctr.min(), log_Ctr.max()
    log_inf_min, log_inf_max = log_Cinf.min(), log_Cinf.max()
    grid_size = 100
    log_tr = np.linspace(log_tr_min, log_tr_max, grid_size)
    log_inf = np.linspace(log_inf_min, log_inf_max, grid_size)
    Ltr, Linf = np.meshgrid(log_tr, log_inf)
    pts = np.column_stack([log_Ctr, log_Cinf])
    grid_acc = griddata(pts, acc_vals, (Ltr, Linf), method='linear')
   
    # Contour levels from .10 to .90 for better visibility in this region
    levels = np.linspace(0.10, 0.90, 9)
   
    # Plot contours on right subplot using GLOBAL normalization
    cmap = plt.cm.inferno
   
    cs = ax2.contour(
        10**Ltr, 10**Linf, grid_acc,
        levels=levels, cmap=cmap, norm=global_norm,
        linewidths=1.0, alpha=1
    )
   
    # Better contour labeling - all horizontal text
    labels = ax2.clabel(cs, inline=True, fmt='%1.2f', fontsize=12, colors='black')
    for label in labels:
        label.set_rotation(0)
   
    # Add scatter points for the colorbar reference
    scatter = ax2.scatter(C_tr_vals, C_inf_vals, c=acc_vals, cmap=cmap, norm=global_norm,
                         s=s_test, alpha=alpha_test, zorder=5)
   
 
    dummy = ax2.scatter(C_tr_vals, C_inf_vals, c=acc_vals, cmap=cmap, norm=global_norm,
                         edgecolor='k', s=0, alpha=1, zorder=5)
   
    # Add colorbar to the right plot
    cbar = plt.colorbar(dummy, ax=ax2)
    cbar.set_label('Accuracy', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
   
    # Set log scales for right plot
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(1e25, 1e26)
    ax2.set_ylim(10**y1, 10**y2)
   
    # Grid lines for right plot
    ax2.grid(False)
    for d in range(int(np.floor(log_tr_min)), int(np.ceil(log_tr_max))+1):
        ax2.axvline(10**d, color='black', linestyle='-', lw=0.7, alpha=0.3)
        for i in range(2,10):
            ax2.axvline(i*10**d, color='gray', linestyle=':', lw=0.7, alpha=0.3)
   
    for d in range(int(np.floor(log_inf_min)), int(np.ceil(log_inf_max))+1):
        ax2.axhline(10**d, color='black', linestyle='-', lw=0.7, alpha=0.3)
        for i in range(2,10):
           ax2.axhline(i*10**d, color='gray', linestyle=':', lw=0.7, alpha=0.3)
   
    ax2.xaxis.set_major_locator(ticker.LogLocator(10))
    ax2.yaxis.set_major_locator(ticker.LogLocator(10))
    ax2.xaxis.set_minor_locator(ticker.LogLocator(10, subs=range(2,10)))
    ax2.yaxis.set_minor_locator(ticker.LogLocator(10, subs=range(2,10)))
    ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
   
    # Increase tick label sizes
    ax2.tick_params(axis='both', which='major', labelsize=14)
   
    # Overlay the fitted straight lines with matching colors (using global normalization)
    ctr_line = np.logspace(log_tr_min, log_tr_max, 200)
    for acc0 in levels:
        color = cmap(global_norm(acc0))
        log_inf_pred = -m * np.log10(ctr_line) + a*acc0 + b
        ax2.plot(
            ctr_line, 10**log_inf_pred,
            '--', lw=2, color=color
        )
   
    # Annotate the fitted-plane equation with R²
    eq = (f"$\\log_{{10}}(C_{{inf}}) = -{m:.2f}\\log_{{10}}(C_{{tr}})"
          f" + {a:.2f}\\times\\mathrm{{Accuracy}} + {b:.2f}$ "
          f"$(R^2 = {r2_score:.2f})$")
  
    eq = (
    r"$\mathbf{log_{10}}\mathbf{(}\mathbf{C}_{\mathbf{inf}}\mathbf{)} = "
    r"\mathbf{-%0.2f}\mathbf{log_{10}}\mathbf{(}\mathbf{C}_{\mathbf{tr}}\mathbf{)}"
    r" + \mathbf{%0.2f} \times \mathbf{Accuracy} + \mathbf{%0.2f}$ "
    r"$\mathbf{(R^2 = %0.2f)}$" % (m, a, b, r2_score)
    )
 
    ax2.text(
        0.98, 0.02, eq,
        transform=ax2.transAxes, ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
        fontsize=14
    )
   
    # Labels for right plot
    ax2.set_xlabel('Training Compute', fontsize=16)
    ax2.set_ylabel('Inference Compute per Task', fontsize=16)
    ax2.set_title('Example Tradeoff in Training and Inference', fontsize=18)
   
    plt.tight_layout()
    plt.show()
   
    # Print model coefficients
    print(f"\nModel coefficients for right plot:")
    print(f"m (training compute coefficient): {m:.4f}")
    print(f"a (accuracy coefficient): {a:.4f}")
    print(f"b (intercept): {b:.4f}")
    print(f"R² score: {r2_score:.4f}")
 
# Run the combined plot
create_combined_plots()

brkpnt = 1

brkpnt = 1

brkpnt = 1
 