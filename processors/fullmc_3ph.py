from .common import select_pair
from .fullmc import Processor as _FullMC


class Processor(_FullMC):
    """Same as fullmc, but the trigger / rel-pT / myy cuts use the
    select_pair'd diphoton (Higgs-mass-matched) instead of the leading two pT-ordered photons."""

    def get_cut_photons(self, inc_photons):
        return select_pair(inc_photons)
