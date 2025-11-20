def to_femto(value, unit):
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
        exponent = exponents.get(unit) - exponents["fb"]
        return value * 10**exponent


def parse_mg_output(mg_output):
    val, _, unc, unit = mg_output.split("Cross-section :   ")[1].split("\n")[0].split()
    val = to_femto(float(val), unit)
    unc = to_femto(float(unc), unit)
    return val, unc


def parse_pythia_output(pythia_output):
    xsec_stats = pythia_output.split("PYTHIA Event and Cross Section Statistics")[1]
    unit = xsec_stats.split("(estimated) (")[1].split(")")[0]
    val, unc = xsec_stats.split("sum")[1].splitlines()[0].split("|")[-2].split()

    val = to_femto(float(val), unit)
    unc = to_femto(float(unc), unit)

    filter_efficiency = float(
        pythia_output.split("ResonanceDecayFilterHook efficiency = ")[1].splitlines()[0]
    )
    return val, unc, filter_efficiency
