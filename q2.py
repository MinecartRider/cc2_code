"""Goldbeter circadian-clock model exercises."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.signal import find_peaks

# State order: [M, P0, P1, P2, PN].
PARAMS = dict(vs=0.76, vm=0.65, vd=0.95, ks=0.38, kt1=1.9, kt2=1.3,
              v1=3.2, v2=1.58, v3=5.0, v4=2.5,
              k1=2.0, k2=2.0, k3=2.0, k4=2.0, ki=1.0, km1=0.5,
              kd=0.2, n=4.0)
X0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
EPSILONS = (1.0, 0.1, 0.01, 0.001)
RTOL = 1e-7
ATOL = 1e-9
T_END = 100.0


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


# Question 2(a): implement the unscaled Goldbeter model.
def goldbeter_rhs(t, x, p=PARAMS):
    """Evaluate the Goldbeter model for ``x = [M, P0, P1, P2, PN]``."""
    # YOUR CODE HERE
    J1 = p["v1"] * x[1] / (p["k1"] + x[1])
    J2 = p["v2"] * x[2] / (p["k2"] + x[2])
    J3 = p["v3"] * x[2] / (p["k3"] + x[2])
    J4 = p["v4"] * x[3] / (p["k4"] + x[3])
    
    M = (p["vs"] / (1 + np.power(x[4]/ p["ki"], p["n"]))) - ((p["vm"] * x[0]) / (p["km1"] + x[0]))
    P0 = p["ks"] * x[0] - J1 + J2
    P1 = J1 - J2 - J3 + J4
    P2 = J3 - J4 - (p["kt1"] * x[3]) + (p["kt2"] * x[4]) - (p["vd"] * x[3] / (p["kd"] + x[3]))
    PN = (p["kt1"] * x[3]) - (p["kt2"] * x[4])
    return [M, P0, P1, P2, PN]
    raise NotImplementedError()


# Question 2(a): add timescale separation around the supplied model.
def goldbeter_rhs_epsilon(t, x, p=PARAMS, epsilon=1.0):
    """Scale ``v1``--``v4`` by ``1/epsilon`` and call ``goldbeter_rhs``."""
    scaled_p = p.copy()
    scaled_p['v1'] = p['v1'] / epsilon
    scaled_p['v2'] = p['v2'] / epsilon
    scaled_p['v3'] = p['v3'] / epsilon
    scaled_p['v4'] = p['v4'] / epsilon
    return goldbeter_rhs(t, x, scaled_p)
    
    # YOUR CODE HERE
    raise NotImplementedError()


def simulate(epsilon, method="RK45"):
    """Integrate one 100-hour run, retaining dense output for plotting."""
    return solve_ivp(lambda t, x: goldbeter_rhs_epsilon(t, x, PARAMS, epsilon),
                     (0, T_END), X0, method=method, rtol=RTOL, atol=ATOL,
                     dense_output=True)


# Question 2(a): estimate the oscillation period from nuclear-PER peaks.
def estimate_period(solution):
    """Estimate the period from late peaks of nuclear PER, ignoring transients."""
    times = np.linspace(0, T_END, 20001)
    nuclear_per = solution.sol(times)[4]
    peaks, _ = find_peaks(nuclear_per)
    peaks = peaks[2:]  # ignore the first two transient peaks
    if len(peaks) < 3:
        return np.nan
    return np.mean(np.diff(times[peaks]))


# Question 2(b): count steps used by the full-model solver.
def rk45_step_count(epsilon):
    """Run RK45 and return its number of time steps."""
    # YOUR CODE HERE
    return len(simulate(epsilon, "RK45").t)-1
    raise NotImplementedError()


# Question 2(b): count steps used by the Radau solver.
def radau_step_count(epsilon):
    """Run Radau and return its number of time steps."""
    # YOUR CODE HERE
    return len(simulate(epsilon, "Radau").t)-1
    raise NotImplementedError()


# Question 2(c): impose rapid equilibrium on the phosphorylation subsystem.
def fast_equilibrium(C, p=PARAMS):
    """Find the equilibrium states from the equal-flux conditions and total C."""
    if C <= 0:
        return np.zeros(3)

    upper = C
    if p['v2'] > p['v1']:
        upper = min(upper, p['v1']*p['k2']/(p['v2']-p['v1']))
    if p['v3'] > p['v4']:
        upper = min(upper, p['v4']*p['k3']/(p['v3']-p['v4']))
    upper *= 1 - 1e-12

    def states(P1):
        r12 = p['v2']*P1/(p['k2']+P1)
        r34 = p['v3']*P1/(p['k3']+P1)
        P0 = p['k1']*r12/(p['v1']-r12)
        P2 = p['k4']*r34/(p['v4']-r34)
        return np.array([P0, P1, P2])

    P1 = brentq(lambda y: states(y).sum()-C, 0, upper, xtol=1e-13)
    return states(P1)


def reduced_rhs(t, y, p=PARAMS):
    """Reduced model for y = [M, C, PN], with phosphorylation at equilibrium."""
    M, C, PN = y
    P2 = fast_equilibrium(C, p)[2]
    dM = p['vs']/(1+(PN/p['ki'])**p['n']) - (p['vm']*M)/(p['km1']+M)
    dC = p['ks']*M - (p['vd']*P2)/(p['kd']+P2) - p['kt1']*P2 + p['kt2']*PN
    dPN = p['kt1']*P2 - p['kt2']*PN
    return np.array([dM, dC, dPN])


def simulate_reduced(method="RK45"):
    """Integrate the rapid-equilibrium model from the full model's initial state."""
    y0 = np.array([X0[0], X0[1]+X0[2]+X0[3], X0[4]])
    return solve_ivp(reduced_rhs, (0, T_END), y0, method=method,
                     rtol=RTOL, atol=ATOL, dense_output=True)


def reduced_error(epsilon, full_solution=None):
    """Return the largest relative slow-state error after the initial transient."""
    full = full_solution if full_solution is not None else simulate(epsilon, "RK45")
    reduced = simulate_reduced("RK45")
    times = np.linspace(1, T_END, 2001)
    full_y = full.sol(times)
    full_slow = np.vstack((full_y[0], full_y[1:4].sum(axis=0), full_y[4]))
    reduced_y = reduced.sol(times)
    scales = np.max(np.abs(full_slow), axis=1)
    return np.max(np.abs(full_slow-reduced_y)/scales[:, None])


def reduced_step_count():
    """Return the number of accepted RK45 steps for the reduced model."""
    return len(simulate_reduced("RK45").t)-1


def plot_exercises():
    import matplotlib.pyplot as plt

    # Question 2(a): plot nuclear PER for each epsilon.
    rk_runs = {epsilon: simulate(epsilon, "RK45") for epsilon in EPSILONS}
    times = np.linspace(0, T_END, 2001)
    fig, ax = plt.subplots()
    for epsilon, sol in rk_runs.items():
        ax.plot(times, sol.sol(times)[4], label=f"$\\epsilon={epsilon:g}$")
    ax.set(xlabel="Time (h)", ylabel="Nuclear PER $p_N$",
           title="Goldbeter oscillations as phosphorylation speeds up")
    ax.legend()
    save_figure(fig, "q2a_periods.png")

    # Question 2(b): compare full-model solver step counts.
    rk45_steps = [rk45_step_count(epsilon) for epsilon in EPSILONS]
    radau_steps = [radau_step_count(epsilon) for epsilon in EPSILONS]
    fig, ax = plt.subplots()
    ax.loglog(EPSILONS, rk45_steps, "o-", label="RK45")
    ax.loglog(EPSILONS, radau_steps, "s--", label="Radau")
    ax.set(xlabel="$\\epsilon$", ylabel="Time steps used by solver",
           title="Full-model solver steps as reaction rates increase")
    ax.legend()
    save_figure(fig, "q2b_step_counts.png")

    # Question 2(c): compare the reduced model with the full model.
    errors = [reduced_error(epsilon, rk_runs[epsilon]) for epsilon in EPSILONS]
    fig, ax = plt.subplots()
    ax.loglog(EPSILONS, errors, "o-")
    ax.set(xlabel="$\\epsilon$", ylabel="Maximum relative error in $M$, $C$, and $P_N$",
           title="Rapid-equilibrium approximation error")
    ax.grid(True, which="both", alpha=0.25)
    save_figure(fig, "q2c_error.png")

    # Show representative time-series snapshots of the slow variables.
    reduced = simulate_reduced("RK45")
    snapshot_times = np.linspace(1, T_END, 2001)
    reduced_slow = reduced.sol(snapshot_times)
    snapshot_epsilons = (1.0, 0.01, 0.001)
    fig, axes = plt.subplots(1, len(snapshot_epsilons), sharey=True,
                             figsize=(12, 3.5))
    colors = ("tab:blue", "tab:orange", "tab:green")
    labels = ("$M$", "$C$", "$P_N$")
    for ax, epsilon in zip(axes, snapshot_epsilons):
        full_y = rk_runs[epsilon].sol(snapshot_times)
        full_slow = np.vstack((full_y[0], full_y[1:4].sum(axis=0), full_y[4]))
        for j, (color, label) in enumerate(zip(colors, labels)):
            ax.plot(snapshot_times, full_slow[j], color=color, label=f"Full {label}")
            ax.plot(snapshot_times, reduced_slow[j], "--", color=color,
                    label=f"Reduced {label}")
        ax.set_title(f"$\\epsilon={epsilon:g}$")
        ax.set_xlabel("Time (h)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Concentration (model units)")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Full and rapid-equilibrium time-series snapshots")
    save_figure(fig, "q2c_time_series.png")

    reduced_steps = reduced_step_count()
    fig, ax = plt.subplots()
    ax.loglog(EPSILONS, rk45_steps, "o-", label="Full RK45")
    ax.loglog(EPSILONS, radau_steps, "s--", label="Full Radau")
    ax.plot(EPSILONS, [reduced_steps] * len(EPSILONS), "^:",
            color="tab:green", label="Reduced RK45")
    ax.set(xlabel="$\\epsilon$", ylabel="Time steps used by solver",
           title="Step counts with and without reduction")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    save_figure(fig, "q2c_step_counts.png")


if __name__ == "__main__":
    plot_exercises()
