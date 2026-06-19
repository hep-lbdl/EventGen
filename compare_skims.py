#!/usr/bin/env python3
"""Overlay kinematic distributions of two SkimEvents h5 files.

Defaults compare the inline-decay vs MadSpin thFCNC samples to validate that
MadSpin reproduces the inline kinematics. Each variable is area-normalized
(shape comparison) with a madspin/inline ratio panel; a weighted KS distance
is printed per variable.

Usage:
    python compare_skims.py                       # thFCNC inline vs madspin
    python compare_skims.py LABEL_A:PATH_A LABEL_B:PATH_B [-o out.pdf]
"""
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = (
    "/pscratch/sd/d/dnoll/projects/haxad/EventGenDelphes/output/version_dev/"
    "SkimEvents/{proc}/ecm_13000.00/ATLAS/all/skimmed.h5"
)

# Bookkeeping columns that are not per-event kinematics.
SKIP = {
    "event_number",
    "event_weight",
    "sumw_presel",
    "sumw_postsel",
    "mg_xsec [fb]",
    "mg_xsec_unc [fb]",
    "pythia_xsec [fb]",
    "pythia_xsec_unc [fb]",
    "pythia_filter_efficiency",
}


def weighted_hist(x, w, bins):
    """Density-normalized weighted counts and Poisson-ish errors per bin."""
    counts, _ = np.histogram(x, bins=bins, weights=w)
    sumw2, _ = np.histogram(x, bins=bins, weights=w**2)
    norm = counts.sum() * np.diff(bins)
    norm = np.where(norm == 0, 1.0, norm)
    return counts / norm, np.sqrt(sumw2) / norm


def ks_distance(xa, wa, xb, wb):
    """Weighted two-sample KS distance (max gap between weighted CDFs)."""
    xs = np.sort(np.unique(np.concatenate([xa, xb])))
    # Build weighted CDFs on the common support.
    def cdf(x, w):
        order = np.argsort(x)
        xc, wc = x[order], w[order]
        c = np.cumsum(wc)
        c = c / c[-1]
        return np.interp(xs, xc, c, left=0.0, right=1.0)

    return np.max(np.abs(cdf(xa, wa) - cdf(xb, wb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?", default="inline:" + BASE.format(proc="thFCNC_utHyy_tphi"))
    ap.add_argument("b", nargs="?", default="madspin:" + BASE.format(proc="thFCNC_utHyy_tphi_madspin"))
    ap.add_argument("-o", "--out", default="compare_skims.pdf")
    args = ap.parse_args()

    (la, pa), (lb, pb) = (s.split(":", 1) for s in (args.a, args.b))
    da = pd.read_hdf(pa, key="events")
    db = pd.read_hdf(pb, key="events")
    wa, wb = da["event_weight"].to_numpy(), db["event_weight"].to_numpy()

    print(f"{la:>10}: {len(da):>6} events, sumw={wa.sum():.4g}")
    print(f"{lb:>10}: {len(db):>6} events, sumw={wb.sum():.4g}")
    if "has_lepton" in da:
        fa = np.average(da["has_lepton"].to_numpy(), weights=wa)
        fb = np.average(db["has_lepton"].to_numpy(), weights=wb)
        print(f"leptonic fraction: {la}={fa:.3f}  {lb}={fb:.3f}")
    print(f"{'variable':>20}  KS")

    variables = [c for c in da.columns if c not in SKIP and c in db.columns]

    with PdfPages(args.out) as pdf:
        for var in variables:
            xa = pd.to_numeric(da[var], errors="coerce").to_numpy(dtype=float)
            xb = pd.to_numeric(db[var], errors="coerce").to_numpy(dtype=float)
            ma, mb = np.isfinite(xa), np.isfinite(xb)
            xa, wai = xa[ma], wa[ma]
            xb, wbi = xb[mb], wb[mb]
            if len(xa) == 0 or len(xb) == 0:
                continue

            combined = np.concatenate([xa, xb])
            uniq = np.unique(combined)
            if len(uniq) <= 12:  # integer / boolean-like
                edges = np.append(uniq, uniq[-1] + 1) - 0.5
                bins = edges
            else:
                lo, hi = np.percentile(combined, [0.5, 99.5])
                if lo == hi:
                    lo, hi = combined.min(), combined.max() + 1e-9
                bins = np.linspace(lo, hi, 41)
            centers = 0.5 * (bins[:-1] + bins[1:])

            ha, ea = weighted_hist(xa, wai, bins)
            hb, eb = weighted_hist(xb, wbi, bins)
            ks = ks_distance(xa, wai, xb, wbi)
            print(f"{var:>20}  {ks:.4f}")

            fig, (ax, axr) = plt.subplots(
                2, 1, sharex=True, figsize=(6, 5),
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            )
            ax.step(centers, ha, where="mid", label=f"{la} (n={len(xa)})", color="C0")
            ax.fill_between(centers, ha - ea, ha + ea, step="mid", color="C0", alpha=0.2)
            ax.step(centers, hb, where="mid", label=f"{lb} (n={len(xb)})", color="C1")
            ax.fill_between(centers, hb - eb, hb + eb, step="mid", color="C1", alpha=0.2)
            ax.set_ylabel("a.u. (norm.)")
            ax.legend(title=f"KS={ks:.3f}", fontsize=8)
            ax.set_title(var)

            ratio = np.divide(hb, ha, out=np.full_like(hb, np.nan), where=ha > 0)
            rerr = np.abs(ratio) * np.sqrt(
                np.divide(eb, hb, out=np.zeros_like(eb), where=hb > 0) ** 2
                + np.divide(ea, ha, out=np.zeros_like(ea), where=ha > 0) ** 2
            )
            axr.errorbar(centers, ratio, yerr=rerr, fmt="o", ms=3, color="k")
            axr.axhline(1.0, color="gray", lw=1, ls="--")
            axr.set_ylim(0.5, 1.5)
            axr.set_ylabel(f"{lb}/{la}", fontsize=8)
            axr.set_xlabel(var)

            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nWrote {args.out} ({len(variables)} variables)")


if __name__ == "__main__":
    main()
