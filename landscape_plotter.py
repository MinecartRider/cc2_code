import numpy as np
import matplotlib.pyplot as plt
from challenge_utils import evaluate_parameters
from constants import REPORTER_PARAMETERS, GROUP_INDICES

def plot_cry1_landscape(t_step=0.01, radius=15):
    """
    Plots a contour map of the objective function for the two CRY1 parameters.
    radius determines how many t_steps in each direction to explore.
    """
    p_rep = REPORTER_PARAMETERS.copy()
    
    # Extract the two indices for CRY1
    cry1_indices = GROUP_INDICES["CRY1"]
    if len(cry1_indices) != 2:
        raise ValueError(f"Expected 2 parameters for CRY1, found {len(cry1_indices)}.")
        
    idx_x, idx_y = cry1_indices[0], cry1_indices[1]
    base_x, base_y = p_rep[idx_x], p_rep[idx_y]
    
    # Generate the grid arrays
    x_vals = np.arange(base_x - radius * t_step, base_x + (radius + 1) * t_step, t_step)
    y_vals = np.arange(base_y - radius * t_step, base_y + (radius + 1) * t_step, t_step)
    
    # Enforce strictly positive biological constraints
    x_vals = x_vals[x_vals > 1e-6]
    y_vals = y_vals[y_vals > 1e-6]
    
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)
    
    total_points = X.size
    print(f"Evaluating {total_points} grid points. This will take a few minutes...")
    
    # Evaluate the objective at every (x, y) combination
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p_test = p_rep.copy()
            p_test[idx_x] = X[i, j]
            p_test[idx_y] = Y[i, j]
            
            # If the solver fails/returns inf, we catch it
            result = evaluate_parameters(p_test)
            Z[i, j] = result.get("objective", np.inf)
            
        print(f"Completed row {i + 1}/{X.shape[0]}")

    # --- FIND THE MINIMUM ---
    # np.nanargmin safely ignores any inf/NaN values where the solver failed
    min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    best_x = X[min_idx]
    best_y = Y[min_idx]
    min_score = Z[min_idx]
    
    print("\n--- GRID SEARCH RESULTS ---")
    print(f"Minimum Objective J: {min_score:.6f}")
    print(f"Best Coordinates: Parameter {idx_x} = {best_x:.6f}, Parameter {idx_y} = {best_y:.6f}")
    # Mask out infinity/NaNs so they don't break the color scale
    Z_masked = np.ma.masked_invalid(Z)
    Z_masked = np.ma.masked_where(Z_masked == np.inf, Z_masked)

    # Generate the contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z_masked, levels=30, cmap='viridis_r')
    plt.colorbar(contour, label="Objective J (Lower is better)")
    
    # Plot the starting reporter baseline as a red dot
    plt.plot(base_x, base_y, 'ro', markersize=8, label="Reporter Baseline")
    
    plt.title("CRY1 Objective Landscape")
    plt.xlabel(f"CRY1 Parameter {idx_x}")
    plt.ylabel(f"CRY1 Parameter {idx_y}")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("cry1_landscape.png", dpi=150)
    print("\nPlot saved as cry1_landscape.png")
    
if __name__ == "__main__":
    # Adjust t_step for resolution, and radius for how far to look.
    # A radius of 15 creates a 31x31 grid (961 evaluations).
    plot_cry1_landscape(t_step=0.01, radius=15)