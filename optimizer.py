import numpy as np
from challenge_utils import evaluate_parameters
from constants import REPORTER_PARAMETERS, GROUP_INDICES

def gene_indices(genes):
    if isinstance(genes, str):
        genes = [genes]
    return np.concatenate([GROUP_INDICES[gene] for gene in genes])

def update_pars(pold, pnew, genes):
    updated = np.asarray(pold, dtype=float).copy()
    updated[gene_indices(genes)] = pnew
    return updated

def gradient_calculator(func, q, h=1e-4):
    grad = np.zeros_like(q)
    for i in range(len(q)):
        q_plus, q_minus = q.copy(), q.copy()
        
        # Ensure the step size is never larger than 10% of the parameter itself
        h_safe = min(h, q[i] * 0.1)
        
        q_plus[i] += h_safe
        q_minus[i] -= h_safe
        
        # Calculate the centered difference using the safe step size
        grad[i] = (func(q_plus) - func(q_minus)) / (2 * h_safe)
        
    return grad

if __name__ == "__main__":
    # 1. Define your saved array from the 50th step
    saved_p = np.array([
    0.2428824295, 0.32515446780000001, 0.0012427507, 0.41151656096432876, 0.0014498909,
    0.073112585800000005, 0.066247264400000005, 0.27499067960000001, 0.60923825460000003, 0.5490839153,
    2.0944834430000001, 0.081444524599999998, 0.2357602144, 0.49029141079999999, 1.1094139264,
    1.0923791030000001, 0.0942352568, 26.974767933999999, 2.7313294030000002, 3.2836338450109746,
    4.5264633761999997, 0.00027825560000000003, 0.015193632021321924, 0.052771229056914802, 1.7016120651,
    2.1727019427999998, 1.3968953974, 0.3257597398609326, 1.3419368066999999, 1.9077837282000001,
    0.24170709949999999, 1.2207190914977808, 1.8308520332, 1.8899129916999999, 2.0327902579999999,
    1.5493019781999999, 0.90000000000000002, 1.9032897012000001, 0.55957819929999997, 0.35183144290000001,
    5.2736612934, 2.8719117418, 0.096236099697741023, 0.057131757900000003, 0.2118550237725722,
])
    
    # Define the genes you were optimizing
    GENES = ["BMAL"] 
    
    # 2. Extract the starting 'q' values from your saved array, NOT the reporter
    q = saved_p[gene_indices(GENES)].copy()
    
    # 3. Update the objective function to patch 'q' into your saved baseline
    def objective_for_q(q_trial):
        p_trial = update_pars(saved_p, q_trial, GENES)
        return evaluate_parameters(p_trial)["objective"]
        
    steps = 100  # Run for another 50 steps
    learning_rate = 0.0001 
    
    print(f"Resuming from objective: {objective_for_q(q):.4f}")
    
    # ... (keep the rest of your gradient descent loop exactly the same)
    for iteration in range(steps):
        grad = gradient_calculator(objective_for_q, q)
        
        # Safety catch 1: If gradient blows up, stop
        if not np.all(np.isfinite(grad)):
            print(f"Iteration {iteration+1:2} | Gradient is NaN/Inf. Stopping early.")
            break
            
        q_next = np.maximum(q - learning_rate * grad, 1e-6)
        
        # Safety catch 2: If the new parameters break the simulation, stop
        current_obj = objective_for_q(q_next)
        if not np.isfinite(current_obj):
            print(f"Iteration {iteration+1:2} | Step pushed the model into an unstable state. Stopping early.")
            break
            
        # If safe, accept the step
        q = q_next
        print(f"Iteration {iteration+1:2} | Objective: {current_obj:.12f}")
        
   # --- Generate and print the final vector ---
    p_edit = update_pars(saved_p, q, GENES)
    
    print("\n--- FINAL CANDIDATE VECTOR ---")
    print("Copy the array below into your p_edit variable in challenge_submission.py:\n")
    
    print("np.array([")
    for old_val, new_val in zip(saved_p, p_edit):
        # If the value changed, print it with full precision and a comment highlighting it
        if old_val != new_val:
            print(f"    {new_val!r},  # <--- SHIFTED (originally {old_val!r})")
        else:
            print(f"    {new_val!r},")
    print("])")