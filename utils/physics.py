def to_unit(value, unit, target_unit="fb"):
    unit = unit.lower()
    exponents = {
        "mb": -3,
        "ub": -6,
        "nb": -9,
        "pb": -12,
        "fb": -15,
        "ab": -18,
    }
    if unit not in exponents:
        raise ValueError(f"Unsupported unit: {unit}")
    else:
        exponent = exponents.get(unit) - exponents[target_unit]
        return value * 10**exponent


def parse_mg_output(mg_output):
    val, _, unc, unit = mg_output.split("Cross-section :   ")[1].split("\n")[0].split()
    val = to_unit(float(val), unit, target_unit="fb")
    unc = to_unit(float(unc), unit, target_unit="fb")
    return val, unc


def parse_pythia_output(pythia_output):
    xsec_stats = pythia_output.split("PYTHIA Event and Cross Section Statistics")[1]
    unit = xsec_stats.split("(estimated) (")[1].split(")")[0]
    val, unc = xsec_stats.split("sum")[1].splitlines()[0].split("|")[-2].split()

    val = to_unit(float(val), unit, target_unit="fb")
    unc = to_unit(float(unc), unit, target_unit="fb")

    filter_efficiency = float(
        pythia_output.split("ResonanceDecayFilterHook efficiency = ")[1].splitlines()[0]
    )
    return val, unc, filter_efficiency


def pythia_xsec_modulation(pythia_config):
    """
    Get cross-section modulation based on pythia config.
    Only use if MadGraph is used to generate the hard process.
    Checks that patterns start lines.
    """
    modulations = {
        ("25:onMode = off", "25:onIfMatch = 22 22"): 2.270e-3,  # H->yy BR
        # ("other_pattern1", "other_pattern2"): factor,
    }
    
    modulation = 1.0
    config_lines = pythia_config.split('\n')
    
    for patterns, factor in modulations.items():
        if all(any(line.startswith(pattern) for line in config_lines) for pattern in patterns):
            modulation *= factor

    return modulation