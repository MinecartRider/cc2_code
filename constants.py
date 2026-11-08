import numpy as np

# These are the period and phases for the wild type. You want to get as close
# to these as possible. Phases are in PHASE_SPECIES order, in radians.
WT_PERIOD = 24.000001503568697
WT_PHASES = np.array([
    0, -0.41423016152565123, -0.31144899475959048, -0.49609438418354862, -1.5544283823097969,
])
WT_PHASES.setflags(write=False)

# These are the names of all the parameters in the model.
PARAMETER_NAMES = (
    "vtp", "ktp", "kb", "vdp", "kdp", "vtc1", "vtc2", "ktc", "vdc1", "vdc2", "kdc",
    "vtror", "ktror", "vdror", "kdror", "vtrev", "ktrev", "vdrev", "kdrev", "klp",
    "vdP", "kdP", "vaCP", "vdCP", "vdC1", "kdC", "vdC2", "klror", "vdROR", "kdROR",
    "klrev", "vdREV", "kdREV", "vxROR", "vxREV", "kxREV", "kxROR", "vdb", "kdb",
    "klb", "vdB", "kdB", "vdC1N", "kdCP", "vdC2N",
)

# Full luminescent reporter: this is your no-edit starting point.
REPORTER_PARAMETERS = np.array([
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

# Biological groups used to count edited genes and apply the group penalty.
# Each gene is charged once for any exact change from the reporter.
# CRY1 includes shared promoter ktc; BMAL includes shared binding constant kb.
DNA_GROUPS = {
    "PER": ("vtp", "ktp"), "CRY1": ("vtc1", "ktc"), "CRY2": ("vtc2",),
    "ROR": ("vtror", "ktror"), "REV": ("vtrev", "ktrev"),
    "BMAL": ("vxROR", "vxREV", "kxREV", "kxROR", "kb"),
}

GROUP_INDICES = {
    gene: np.array([PARAMETER_NAMES.index(name) for name in names])
    for gene, names in DNA_GROUPS.items()
}

PHASE_SPECIES = (("PER", 5), ("CRY1", 6), ("CRY2", 7), ("ROR", 8), ("REV", 9))
