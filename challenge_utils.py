import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from constants import (
    WT_PERIOD, WT_PHASES, PARAMETER_NAMES, REPORTER_PARAMETERS,
    GROUP_INDICES, PHASE_SPECIES,
)

# This is the initial value
X0 = np.array([2.94399774, 1.45982459, 1.37963843, 1.53271061, 0.64180771,
               2.51814106, 5.27221041, 5.14682548, 1.29418516, 0.89956422,
               0.53545950, 0.36875376, 0.09342431, 0.05479389])

# These are the ODEs for the Brown-Doyle model
def _rhs(_t, x, p):
    p0, c1, c2, ror, rev, P, C1, C2, ROR, REV, b, B, C1N, C2N = x
    (vtp, ktp, kb, vdp, kdp, vtc1, vtc2, ktc, vdc1, vdc2, kdc,
     vtror, ktror, vdror, kdror, vtrev, ktrev, vdrev, kdrev, klp,
     vdP, kdP, vaCP, vdCP, vdC1, kdC, vdC2, klror, vdROR, kdROR,
     klrev, vdREV, kdREV, vxROR, vxREV, kxREV, kxROR, vdb, kdb,
     klb, vdB, kdB, vdC1N, kdCP, vdC2N) = p
    txne = lambda vmax, km, d1, d2: vmax / (km + kb / B + d1 + d2)
    mmdeg = lambda s, vmax, km: -vmax * s / (km + s)
    bind = lambda s1, s2, z: -vaCP * s1 * s2 + vdCP * z
    return np.array([
        txne(vtp, ktp, C1N, C2N) + mmdeg(p0, vdp, kdp),
        txne(vtc1, ktc, C1N, C2N) + mmdeg(c1, vdc1, kdc),
        txne(vtc2, ktc, C1N, C2N) + mmdeg(c2, vdc2, kdc),
        txne(vtror, ktror, C1N, C2N) + mmdeg(ror, vdror, kdror),
        txne(vtrev, ktrev, C1N, C2N) + mmdeg(rev, vdrev, kdrev),
        klp*p0 + mmdeg(P, vdP, kdP) + bind(P, C1, C1N) + bind(P, C2, C2N),
        c1 + mmdeg(C1, vdC1, kdC) + bind(P, C1, C1N),
        c2 + mmdeg(C2, vdC2, kdC) + bind(P, C2, C2N),
        klror*ror + mmdeg(ROR, vdROR, kdROR), klrev*rev + mmdeg(REV, vdREV, kdREV),
        (vxROR*ROR + vxREV) / (1 + kxREV*REV + kxROR*ROR) + mmdeg(b, vdb, kdb),
        klb*b + mmdeg(B, vdB, kdB),
        -vdC1N*C1N/(kdCP+C1N+C2N) - bind(C1, P, C1N),
        -vdC2N*C2N/(kdCP+C1N+C2N) - bind(C2, P, C2N),
    ])

EDITABLE_INDICES = np.concatenate(list(GROUP_INDICES.values()))
FORBIDDEN_INDICES = np.setdiff1d(np.arange(45), EDITABLE_INDICES)

GENE_COSTS = dict(PER=.405, CRY1=.061, CRY2=.156, ROR=.168, REV=.172, BMAL=.127)
# Precomputed reporter error for the supplied model, measurements and solver settings.
REPORTER_RECOVERY_ERROR = 0.0052930402208124965
SCORE_IMPROVEMENT_THRESHOLD = 65
OBJECTIVE_THRESHOLD = .35
END_TIME, SAMPLE_START, SAMPLE_POINTS = 360., 200., 6001
RTOL, ATOL, MAX_STEP = 1e-7, 1e-9, 1.5


def validate_parameters(parameters):
    """Return a copy; reject wrong shapes, nonfinite values, or nonpositive rates."""
    p = np.asarray(parameters, dtype=float)
    if p.shape != (45,) or not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("Supply exactly 45 positive, finite parameters.")
    return p.copy()


def simulate(parameters=REPORTER_PARAMETERS):
    """Return a dense SciPy solution; time is in hours."""
    parameters = validate_parameters(parameters)
    sol = solve_ivp(lambda t, x: _rhs(t, x, parameters), (0., END_TIME), X0.copy(),
                    method="LSODA", rtol=RTOL, atol=ATOL,
                    max_step=MAX_STEP, dense_output=True)
    if not sol.success or sol.sol is None:
        raise RuntimeError("Clock integration failed: " + sol.message)
    return sol


def plot_model(parameters=REPORTER_PARAMETERS, filename="challenge_clock.png"):
    """Plot the candidate clock. Target reference measurements are not trajectories."""
    import matplotlib.pyplot as plt
    candidate = simulate(validate_parameters(parameters))
    times = np.linspace(SAMPLE_START, END_TIME, 2001)
    y = candidate.sol(times)
    fig, axes = plt.subplots(len(PHASE_SPECIES), 1, figsize=(10, 10), sharex=True)
    for ax, (name, index) in zip(axes, PHASE_SPECIES):
        ax.plot(times, y[index], label="Candidate", color="tab:blue")
        ax.set_ylabel(name)
    axes[0].legend()
    axes[-1].set_xlabel("Time (hours)")
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    return filename


def peak_data(solution):
    """Supplied: dominant-peak candidates with quadratic time interpolation."""
    times = np.linspace(SAMPLE_START, END_TIME, SAMPLE_POINTS)
    y = solution.sol(times)
    peaks = {}
    for name, index in PHASE_SPECIES:
        values = y[index]
        locations, _ = find_peaks(values, prominence=.01*np.ptp(values))
        curvature = values[locations-1] - 2*values[locations] + values[locations+1]
        shift = .5*(values[locations-1]-values[locations+1])/curvature
        peaks[name] = (times[locations]+shift*(times[1]-times[0]), values[locations])
    return peaks


def estimate_period(solution, peaks=None):
    """Return the settled PER period in hours, or raise ValueError if not settled.

    peak_data(solution)["PER"][0] supplies late peak times. Use the final
    five peaks (four intervals), and reject fewer peaks or an interval
    coefficient of variation above 0.01. Do not estimate from the transient.
    """
    peaks = peak_data(solution) if peaks is None else peaks
    intervals = np.diff(peaks["PER"][0][-5:])
    if (len(intervals) != 4 or not np.all(np.isfinite(intervals))
            or np.any(intervals <= 0) or np.std(intervals, ddof=1)/np.mean(intervals) > .01):
        raise ValueError("Too few stable PER peaks.")
    return float(np.mean(intervals))



def evaluate_parameters(parameters):
    """Return objective measurements and edit diagnostics for one vector."""
    return objective(parameters)


def measure_clock(parameters):
    """Supplied phase extraction: return period and five phases in RADIANS."""
    sol = simulate(parameters)
    peaks = peak_data(sol)
    period = float(estimate_period(sol, peaks))
    if not np.isfinite(period) or period <= 0:
        raise ValueError("The period estimator must return a positive finite period in hours.")
    if len(peaks["PER"][0]) < 5:
        raise ValueError("Too few late PER peaks for phase extraction.")
    if np.ptp(sol.sol(np.linspace(SAMPLE_START, END_TIME, 1001))[5]) < .01:
        raise ValueError("PER oscillation amplitude is too small.")
    per = peaks["PER"][0][-5:]
    phases = [0.]
    for name, _ in PHASE_SPECIES[1:]:
        ts, heights = peaks[name]
        samples = []
        for left, right in zip(per[:-1], per[1:]):
            mask = (ts > left) & (ts <= right)
            if not np.any(mask):
                raise ValueError(f"No settled {name} peak in a PER cycle.")
            samples.append(2*np.pi*(ts[mask][np.argmax(heights[mask])]-left)/(right-left))
        phases.append(float(np.angle(np.mean(np.exp(1j*np.array(samples))))))
    return period, np.array(phases)


def align_phases(reference_radians, candidate_radians):
    """Remove one common rotation; return wrapped residuals in radians.

    Both inputs must be finite, nonempty, equal-shaped 1-D radian vectors.
    Residuals lie in [-pi, pi]. A common shift alone therefore has zero error.
    """
    reference = np.asarray(reference_radians, float)
    candidate = np.asarray(candidate_radians, float)
    if (reference.ndim != 1 or reference.size == 0 or reference.shape != candidate.shape
            or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate))):
        raise ValueError("Supply two equal-length finite phase vectors in radians.")
    difference = candidate-reference
    rotation = float(np.angle(np.sum(np.exp(1j*difference))))
    residual = np.angle(np.exp(1j*(difference-rotation)))
    return residual


def aligned_phase_error(reference_radians, candidate_radians):
    """Supplied: aligned RMS phase mismatch in cycles, ready for the objective.

    Example: aligned_phase_error([0., 1., 2.], [.4, 1.4, 2.4]) is zero:
    adding a common phase shift does not change relative gene timing.
    """
    residual = align_phases(reference_radians, candidate_radians)
    return float(np.sqrt(np.mean(residual**2))/(2*np.pi))


def edit_summary(parameters):
    """Count edits relative to REPORTER, never relative to WT."""
    changed = np.asarray(parameters) != REPORTER_PARAMETERS
    genes = [gene for gene, ix in GROUP_INDICES.items() if np.any(changed[ix])]
    forbidden = [PARAMETER_NAMES[i] for i in FORBIDDEN_INDICES if changed[i]]
    return {
        "genes_edited": genes,
        "number_genes_edited": len(genes),
        "number_parameters_edited": int(np.sum(changed)),
        "forbidden_parameters": forbidden,
        "number_forbidden_parameters": len(forbidden),
    }


def objective_helper(parameters):
    """Return a dict with period_error, phase_error, and recovery_error.

    period_error = abs(T - T_WT) / T_WT  (dimensionless).
    phase_error = RMS(aligned residuals) / (2*pi)  (cycles, NOT radians).
    recovery_error E = (period_error**2 + phase_error**2) / 2.

    Phases are aligned with align_phases, which accepts reference/candidate
    radians and returns residual radians. Do not penalize the common
    rotation. The helper returns these measured quantities for scoring.
    """
    period, phases = measure_clock(parameters)
    # Both circular alignment and its RMS reduction are supplied for you.
    phase_error = aligned_phase_error(WT_PHASES, phases)
    period_error = abs(period-WT_PERIOD)/WT_PERIOD
    return dict(period=period, period_error=period_error, phase_error=phase_error,
                recovery_error=(period_error**2+phase_error**2)/2)

def objective(parameters):
    """Actual vector score, with a supplied group penalty.

    J = E/E(nothing) + sum of costs of edited genes.
    A valid do-nothing vector has J=1. Forbidden edits are flagged separately;
    the formula's finite score is still returned for inspection.
    """
    p = validate_parameters(parameters)
    edits = edit_summary(p)
    try:
        rec = objective_helper(p)
        baseline = REPORTER_RECOVERY_ERROR
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("The reporter baseline error must be finite and positive.")
        if any(not np.isfinite(rec[key]) or rec[key] < 0
               for key in ("period_error", "phase_error", "recovery_error")):
            raise ValueError("The objective helper returned invalid error components.")
        penalty = sum(GENE_COSTS[g] for g in edits["genes_edited"])
        diagnostic_score = rec["recovery_error"]/baseline + penalty
        return {
            **rec, **edits,
            "valid": not bool(edits["forbidden_parameters"]),
            "failure": "Forbidden parameters changed." if edits["forbidden_parameters"] else None,
            "objective": float(diagnostic_score),
            "recovery_improvement_percent": 100*(1-rec["recovery_error"]/baseline),
            "score_improvement_percent": 100*(1-diagnostic_score),
        }
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        return {**edits, "valid": False, "failure": str(exc),
                "objective": float("inf"), "score_improvement_percent": None}


def print_objective(result, parameters=REPORTER_PARAMETERS, title="Objective report"):
    """Print a clearly headed report with aligned measurements and edit checks."""
    edits = edit_summary(parameters)
    score = result.get("objective", float("inf"))
    improvement = result.get("score_improvement_percent")
    print("\n" + "="*64)
    print(f" {title:^62} ")
    print("="*64)
    print(f"  {'Objective J':<34}{score:>12.6f}")
    value = f"{improvement:.2f}%" if improvement is not None else "not scored"
    print(f"  {'Score improvement':<34}{value:>12}")
    recovery = result.get("recovery_improvement_percent")
    if recovery is not None:
        print(f"  {'Recovery improvement':<34}{recovery:>10.3f} %")
    for label, key, scale, unit in (("Period error", "period_error", 100, "%"),
                                    ("Aligned phase error", "phase_error", 360, "deg")):
        value = result.get(key)
        display = f"{scale*value:.3f} {unit}" if value is not None else "unavailable"
        print(f"  {label:<34}{display:>12}")
    print(f"  {'Edited genes':<34}{', '.join(edits['genes_edited']) or 'none':>12}")
    print(f"  {'Parameters edited':<34}{edits['number_parameters_edited']:>12}")
    forbidden = ', '.join(edits['forbidden_parameters']) or 'none'
    print(f"  {'Forbidden edits':<34}{forbidden:>12}")
    if edits["forbidden_parameters"]:
        print("  Status: INVALID: forbidden parameters changed")
    elif not np.isfinite(score):
        print("  Status: FAILED: " + result.get("failure", "could not measure the clock"))
    else:
        print("  Status: valid")
    print("="*64 + "\n")
