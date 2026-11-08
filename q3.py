"""Goldbeter period sensitivity exercises."""

import numpy as np
from scipy.integrate import solve_ivp

from q2 import PARAMS, X0, goldbeter_rhs

END_TIME = 120.0


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def simulate(vd):
    """Integrate the model with the supplied PER degradation rate."""
    p = dict(PARAMS)
    p['vd'] = vd

    def nuclear_per_peak(t, x):
        return goldbeter_rhs(t, x, p)[4]
    nuclear_per_peak.direction = -1
    nuclear_per_peak.terminal = False
    return solve_ivp(lambda t, x: goldbeter_rhs(t, x, p), (0, END_TIME), X0,
                     method="Radau", max_step=0.5, rtol=1e-9, atol=1e-11,
                     events=nuclear_per_peak, dense_output=True)


def _period_from_solution(sol):
    intervals = np.diff(sol.t_events[0])
    if len(intervals) < 3:
        return np.nan
    intervals = intervals[-3:]
    if np.std(intervals) / np.mean(intervals) > 0.01:
        return np.nan
    return np.mean(intervals)


# Question 3(a): estimate the period at the supplied degradation rate.
def estimate_period(vd):
    """Estimate the stable late-cycle period of nuclear PER."""
    # YOUR CODE HERE
    raise NotImplementedError()


# Question 3(b): compute the centered finite-difference sensitivity.
def finite_difference(h):
    """Return the centered estimate of dT/dvd using physical step h."""
    # YOUR CODE HERE
    raise NotImplementedError()


# Question 3(c): apply Richardson extrapolation to the centered difference.
def richardson(h):
    """Return the fourth-order estimate built from two centered differences."""
    # YOUR CODE HERE
    raise NotImplementedError()


def plot_exercises():
    import matplotlib.pyplot as plt

    vd = PARAMS['vd']
    changes = (-0.01, 0.0, 0.01)

    # Question 3(a): compare nuclear-PER cycles and their measured periods.
    fig, ax = plt.subplots()
    times = np.linspace(END_TIME / 2, END_TIME, 2001)
    for change in changes:
        current_vd = vd + change
        sol = simulate(current_vd)
        period = estimate_period(current_vd)
        ax.plot(times, sol.sol(times)[4],
                label=f"$v_d={current_vd:.2f}$, T={period:.3f} h")
    ax.set(xlabel="Time (h)", ylabel="Nuclear PER $p_N$",
           title="Period change under PER degradation perturbations")
    ax.legend()
    save_figure(fig, "q3a_periods.png")

    # Question 3(b): estimate the dimensionless sensitivity for each step.
    steps = np.array([0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
    d2 = {h: finite_difference(float(h)) for h in steps}
    sensitivity = vd / estimate_period(vd) * np.array([d2[h] for h in steps])
    fig, ax = plt.subplots()
    ax.semilogx(steps, sensitivity, "o-", label="Centered difference")
    ax.set(xlabel="Finite-difference step $h$", ylabel="$S_{T,v_d}$",
           title="Estimated dimensionless period sensitivity")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    save_figure(fig, "q3b_sensitivity.png")

    # Question 3(c): add Richardson estimates to the same sensitivity plot.
    richardson_sensitivity = vd / estimate_period(vd) * np.array(
        [richardson(float(h)) for h in steps])
    fig, ax = plt.subplots()
    ax.semilogx(steps, sensitivity, "o-", label="Centered difference")
    ax.semilogx(steps, richardson_sensitivity, "s--", label="Richardson")
    ax.set(xlabel="Finite-difference step $h$", ylabel="$S_{T,v_d}$",
           title="Convergence of the period sensitivity coefficient")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    save_figure(fig, "q3c_convergence.png")


if __name__ == "__main__":
    plot_exercises()
