"""Clock recovery: edit a parameter vector and compare it with the target."""
import numpy as np
from challenge_utils import (
    SCORE_IMPROVEMENT_THRESHOLD,
    evaluate_parameters,  # (parameters) -> objective and components
    plot_model,           # (parameters, filename="challenge_clock.png") -> saves a plot
    print_objective,      # (result, parameters, title) -> prints a formatted report
)
from constants import REPORTER_PARAMETERS, GROUP_INDICES

# Functions imported from challenge_utils:
#
# evaluate_parameters(parameters)
#     Simulate the clock for a 45-entry parameter vector and return a dict
#     with the objective J and its components, using the supplied period estimator.
#
# plot_model(parameters, filename="challenge_clock.png")
#     Plot the PER, CRY1, CRY2, ROR and REV trajectories and save the
#     figure to filename.
#
# print_objective(result, parameters=REPORTER_PARAMETERS, title="Objective report")
#     Print a formatted report. result is the output of evaluate_parameters,
#     parameters is the vector that produced it, and title is the heading.


def gene_indices(genes):
    """Return full-vector indices belonging to the named genes, in gene order."""
    if isinstance(genes, str):
        genes = [genes]
    return np.concatenate([GROUP_INDICES[gene] for gene in genes])


def update_pars(pold, pnew, genes):
    """Copy selected gene entries from reduced pnew into full vector pold."""
    updated = np.asarray(pold, dtype=float).copy()
    indices = gene_indices(genes)
    if np.asarray(pnew).shape != (len(indices),):
        raise ValueError(f"pnew must contain {len(indices)} values for {genes}.")
    updated[indices] = pnew
    return updated

if __name__ == "__main__":
    # Keep the reporter reference unchanged.
    p_rep = REPORTER_PARAMETERS.copy()

    # Simulate, measure the period, and evaluate the scalar objective.
    result = evaluate_parameters(p_rep)
    print_objective(result, p_rep, "DO NOTHING: REPORTER BASELINE")

    # The following is a bad example for an edit. It includes a small noise edit
    # on two forbidden entries.
    rng = np.random.default_rng(1)
    p_noise = p_rep.copy()
    p_noise[3:5] *= 1 + rng.normal(0, 0.001, 2)  # vdp and kdp are forbidden.
    noise_result = evaluate_parameters(p_noise)
    print_objective(noise_result, p_noise, "RANDOM NOISE: FORBIDDEN PARAMETERS")

    # The following is a legal example, but without much improvement.
    # The available genes are: PER, CRY1, CRY2, ROR, REV, BMAL.
    # Let's work with a short vector that contains only the selected genes.
    GENES = ["CRY1","ROR","BMAL"]
    indices = gene_indices(GENES)
    pnew = p_rep[indices].copy()
    pnew *= 1.1                         # Increase each selected parameter by 10%.
    p_edit = update_pars(p_rep, pnew, GENES)  # Keep every other entry unchanged
    
    # display result and plot solution
    result = evaluate_parameters(p_edit)  
    print_objective(result, p_edit, "SINGLE EDIT: SELECTED GENES +10%")
    plot_model(p_edit)

    # This is the part where you would create your own p_edit
    # An outline of how to use gradient descent on a selected gene set is given below
    # The gradient_calculator is the function you would need to write
    # using the ideas from Q3.
    # GENES = ["CRY1"]
    # q = p_rep[gene_indices(GENES)].copy()  # Optimize only these gene values
    # def objective_for_q(q):                 # Every trial obeys the constraint
    #     p_trial = update_pars(p_rep, q, GENES)
    #     return evaluate_parameters(p_trial)["objective"]
    # for iteration in range(steps):
    #     gradient = gradient_calculator(objective_for_q, q)  # You implement this
    #     q = q - step_size * gradient
    # p_edit = update_pars(p_rep, q, GENES)

    result = evaluate_parameters(p_edit)  # Score the final vector after your algorithm.
    # Check the final score against the assignment threshold (higher is better). 
    score_improvement = result["score_improvement_percent"]  # Percent improvement in the full score J.
    if score_improvement is None:
        print("The final vector could not be scored: " + result.get("failure", "simulation failed"))
    elif result.get("valid", False) and score_improvement >= SCORE_IMPROVEMENT_THRESHOLD:
        print(f"You are above the assignment threshold ({score_improvement:.1f}% improvement).")
    else:
        print(f"You are below the assignment threshold ({score_improvement:.1f}% improvement).")
