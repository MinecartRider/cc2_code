"""Goodwin feedback model exercises."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

ALPHA = 10.0


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


# Question 1(a): implement the Goodwin model
def goodwin_rhs(t, x, alpha, H):
    """Return the Goodwin model derivative for an arbitrary loop length.

    The state ``x`` has one entry per species. Return a NumPy array with the
    same length, using the last species to regulate the first and each earlier
    species to produce the next one.
    """
    dx=[]
    dx.append((alpha) / (1 + (x[-1] ** H)) - x[0])
    for i in range(1,len(x)):
        dx.append(x[i-1]-x[i])
    return np.array(dx)

# Question 1(b): investigate the effective time-delay
def feedback_delay(N, alpha=ALPHA, H=8, threshold=0.1, end=300):
    """Return the first time the last species rises above ``threshold``.

    Initialize the first species to one and the others to zero. Have another
    look at Assignment 1's Coding Challenge for event handling with ``solve_ivp``.
    Use an event's ``direction`` and ``terminal`` attributes to select the crossing
    and stop there; event times are returned in ``sol.t_events``.
    """
    hit_thresh=lambda t, x:x[N-1]-threshold
    hit_thresh.terminal = True


    x0=[]
    x0.append(1)
    for i in range(1,N):
        x0.append(0)
    sol = solve_ivp(
    lambda t, x: goodwin_rhs(t,x,alpha,H),
    [0, end], x0, events=hit_thresh)
    return sol.t_events[0][0]



# Question 1(c): estimate the period and variability of late peak intervals.
def oscillation_stats(N, H, alpha=ALPHA, discard=3, end=300, samples=12001):
    """Return the mean and sample SD of late peak intervals, or NaNs.

    Start from ``np.ones(N)`` and sample at ``np.linspace(0, end, samples)``;
    use ``rtol=1e-8`` and ``atol=1e-10`` in solve_ivp.
    Discard the first ``discard`` peaks of ``x[0]``. Return the mean and sample
    SD if at least four peaks remain, the interval coefficient of variation is
    at most 1%, the last-cycle amplitude is at least ``0.01``, and its change
    from the preceding cycle is at most 0.1% of the larger amplitude.
    If any of these conditions are not fulfilled return ``(np.nan, np.nan)``.
    You might want to use these functions (given in no particular order):
    np.std, np.ptp, find_peaks, np.mean
    """
    x0=np.ones(N)
    times=np.linspace(0, end, samples)
    sol = solve_ivp(
        lambda t, x: goodwin_rhs(t, x, alpha, H),
        [0, end], x0, rtol=1e-8, atol=1e-10, t_eval=times
    )
    data=sol.y[0]
    peaks, properties = find_peaks(data)
    peaks2=peaks[discard:]
    delta_t=[]
    for i in range(1,len(peaks2)):
        delta_t.append(peaks2[i]-peaks2[i-1])

    mean=np.mean(delta_t)
    std=np.std(delta_t)
    amp_n=np.ptp(data[peaks2[-2]:peaks2[-1]])
    amp_n_1=np.ptp(data[peaks2[-3]:peaks2[-2]])
    delta_amp=abs(amp_n-amp_n_1)/max(amp_n,amp_n_1)
    var=std/mean
    print(var)
    print(delta_amp)
    print(amp_n)

    if(len(peaks2)>3 and var<=0.01 and amp_n>=0.01 and delta_amp<=0.001):
        return (mean, std)

    else:
        return(np.nan,np.nan)




def plot_exercises():
    import matplotlib.pyplot as plt

    # Q1(a): compare trajectories from two initial states.
    times = np.linspace(0, 80, 1601)
    fig, ax = plt.subplots()
    for x0 in (np.ones(3), np.array([0.2, 1.3, 2.0])):
        sol = solve_ivp(lambda t, x: goodwin_rhs(t, x, ALPHA, 2),
                        (0, 80), x0, t_eval=times)
        ax.plot(sol.t, sol.y[0], label=f"$x(0)={np.array2string(x0)}$")
    ax.set(xlabel="Time", ylabel="$x_1(t)$", title="Different initial states")
    ax.legend(fontsize=8)
    save_figure(fig, "q1a_equilibrium.png")

    # Q1(b): the first-passage time grows with the loop length.
    lengths = np.arange(2, 9)
    delays = [feedback_delay(n) for n in lengths]
    fig, ax = plt.subplots()
    ax.plot(lengths, delays, "o-")
    ax.set(xlabel="Loop length $N$", ylabel="First time $x_N>0.1$",
           title="Effective feedback delay")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "q1b_delay.png")

    # Q1(c): color points by the period; crosses mark cases without a sustained cycle.
    lengths = np.arange(3, 8)
    hills = np.arange(2, 13)
    cycle_points, no_cycle = [], []
    for N in lengths:
        for H in hills:
            period, _ = oscillation_stats(N, H)
            (cycle_points if np.isfinite(period) else no_cycle).append((N, H, period))
    fig, ax = plt.subplots()
    if no_cycle:
        ax.scatter([v[0] for v in no_cycle], [v[1] for v in no_cycle],
                   marker="x", color="0.55", label="No sustained cycle")
    if cycle_points:
        points = ax.scatter([v[0] for v in cycle_points], [v[1] for v in cycle_points],
                            c=[v[2] for v in cycle_points], cmap="viridis",
                            label="Sustained cycle")
        fig.colorbar(points, ax=ax, label="Estimated period")
    ax.set(xlabel="Loop length $N$", ylabel="Hill coefficient $H$",
           title="Goodwin oscillation map", xticks=lengths, yticks=hills)
    ax.legend()
    save_figure(fig, "q1c_phase_map.png")


if __name__ == "__main__":
    plot_exercises()
