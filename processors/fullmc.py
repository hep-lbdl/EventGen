import numpy as np
import awkward as ak
import dask_awkward as dak
import coffea.processor as processor
from coffea.nanoevents.methods import candidate

# TODO: Define somewhere else
Z_MASS_GEV = 91.1880
EL_MASS_GEV = 0.00051099895
MU_MASS_GEV = 0.1056583755


def select_pair(particles, condition=lambda c: abs((c["0"] + c["1"]).mass - 125)):
    """
    Select pair of particles for which the condition is minimal.
    Returns a collection of all particles with the pair as the first two.
    The ordering of the remaining particles is left intact.
    Currently only implemented for three particles.
    """
    c = ak.combinations(particles, 2)
    ordering = ak.argmin(condition(c), axis=-1)
    ordering = ak.fill_none(ordering, 0)
    npartitions = ordering.npartitions
    indices = dak.from_awkward(
        ak.concatenate([
            ak.Array(
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 2, 0],
            ])
            for i in range(npartitions)
        ]),
        npartitions=npartitions,
    )
    reorder = indices[ordering]
    return particles[reorder]


class Processor(processor.ProcessorABC):
    def postprocess(self, accumulator):
        pass

    def process(self, events):
        pad = lambda x, target: ak.pad_none(x, target=target, clip=True)

        # ===== inc_ photons (pT-ordered, pad 3) =====
        inc_photons = pad(events.Photon, 3)
        inc_photons = ak.zip(
            {
                "pt": inc_photons.pt,
                "eta": inc_photons.eta,
                "phi": inc_photons.phi,
                "mass": ak.zeros_like(inc_photons.pt),
                "charge": ak.zeros_like(inc_photons.pt),
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        diphoton_mass = (inc_photons[:, 0] + inc_photons[:, 1]).mass
        diphoton_pt = (inc_photons[:, 0] + inc_photons[:, 1]).pt
        diphoton_delta_r = inc_photons[:, 0].delta_r(inc_photons[:, 1])
        inc_photons_pt_rel = inc_photons.pt / diphoton_mass[:, None]
        inc_photon1_pt_rel, inc_photon2_pt_rel, inc_photon3_pt_rel = (inc_photons_pt_rel[:, i] for i in range(3))

        # ===== pair_ photons (select_pair reordered for Higgs matching) =====
        pair_photons = select_pair(inc_photons)
        pair_diphoton_mass = (pair_photons[:, 0] + pair_photons[:, 1]).mass
        pair_diphoton_pt = (pair_photons[:, 0] + pair_photons[:, 1]).pt
        pair_diphoton_delta_r = pair_photons[:, 0].delta_r(pair_photons[:, 1])

        """
        Cut-based related: Only save n_object and bool flag
        is_lb: n_e+n_mu >=1, nb>=1
        is_tlep: n_e + n_mu == 1, n_j==n_b==1
        is_thad: n_e + n_mu == 0, n_j==3 n_b==1 (BDT/top tagging missing now)
        is_Zveto: nl==2, same flavor, mll in Z window (10GeV)
        is_SS:  nl==2, same sign,
        myy_23 not possible now. skip.
        tau skip now (no good defination in Delphes)
        Overlap removal skip now. Use Delphes unique object.
        """
        # inclusive photon 22GeV 2p5 eta
        sel_ph = (events.Photon.pt > 22) & (events.Photon.eta < 2.5) & (events.Photon.eta > -2.5)
        sel_n_ph = ak.sum(sel_ph, axis=-1)

        # leptons
        sel_el = (events.Electron.pt > 10) & (events.Electron.eta < 2.5) & (events.Electron.eta > -2.5)  # fmt: skip
        sel_mu = (events.Muon.pt > 10) & (events.Muon.eta < 2.7) & (events.Muon.eta > -2.7)

        # analyze lepton flavor multiplicity
        sel_n_e = ak.sum(sel_el, axis=-1)
        sel_n_mu = ak.sum(sel_mu, axis=-1)
        sel_is_ee = sel_n_e == 2
        sel_is_mumu = sel_n_mu == 2
        sel_is_emu = (sel_n_e == 1) & (sel_n_mu == 1)

        sel2_el = pad(events.Electron[sel_el], 2)
        sel2_mu = pad(events.Muon[sel_mu], 2)

        sel2_el_4vec = ak.zip(
            {
                "pt": sel2_el.pt,
                "eta": sel2_el.eta,
                "phi": sel2_el.phi,
                "mass": ak.ones_like(sel2_el.pt) * EL_MASS_GEV,
                "charge": sel2_el.Charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        sel2_mu_4vec = ak.zip(
            {
                "pt": sel2_mu.pt,
                "eta": sel2_mu.eta,
                "phi": sel2_mu.phi,
                "mass": ak.ones_like(sel2_mu.pt) * MU_MASS_GEV,
                "charge": sel2_mu.Charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        # analyzing the charge and flavor
        sel2_el_ch = sel2_el_4vec.charge
        sel2_mu_ch = sel2_mu_4vec.charge
        sel_is_SS = (
            (sel_is_ee & (sel2_el_ch[:, 0] * sel2_el_ch[:, 1] > 0))
            | (sel_is_emu & (sel2_el_ch[:, 0] * sel2_mu_ch[:, 1] > 0))
            | (sel_is_mumu & (sel2_mu_ch[:, 0] * sel2_mu_ch[:, 1] > 0))
        )

        # Veto Z peak
        m_ee = (sel2_el_4vec[:, 0] + sel2_el_4vec[:, 1]).mass
        m_mm = (sel2_mu_4vec[:, 0] + sel2_mu_4vec[:, 1]).mass
        sel_is_Zveto = ~(
            (sel_is_ee & (m_ee > Z_MASS_GEV - 10) & (m_ee < Z_MASS_GEV + 10))
            | (sel_is_mumu & (m_mm > Z_MASS_GEV - 10) & (m_mm < Z_MASS_GEV + 10))
        )

        # ===== inc_ jets (pad 4) =====
        inc_jets = pad(events.Jet, 4)
        inc_jets = ak.zip(
            {
                "pt": inc_jets.pt,
                "eta": inc_jets.eta,
                "phi": inc_jets.phi,
                "mass": inc_jets.mass,
                "charge": ak.zeros_like(inc_jets.pt),
                "BTag": inc_jets.BTag,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        dijet_mass = (inc_jets[:, 0] + inc_jets[:, 1]).m
        dijet_delta_r = inc_jets[:, 0].delta_r(inc_jets[:, 1])
        ht_30 = ak.sum(events.Jet.pt[events.Jet.pt > 30], axis=-1)

        # count j and b here
        sel_jet = (events.Jet.pt > 25) & (events.Jet.eta < 4.4) & (events.Jet.eta > -4.4)
        sel_bjet = (
            (events.Jet.pt > 25)
            & (events.Jet.eta < 2.5)
            & (events.Jet.eta > -2.5)
            & (events.Jet.BTag == 1)
        )
        sel_n_j = ak.sum(sel_jet, axis=-1)
        sel_n_b = ak.sum(sel_bjet, axis=-1)
        sel_n_lep = sel_n_e + sel_n_mu
        sel_is_lb = (sel_n_lep >= 1) & (sel_n_b >= 1)
        sel_is_tlep = (sel_n_lep == 1) & (sel_n_j == 1) & (sel_n_b == 1)
        sel_is_thad = (sel_n_lep == 0) & (sel_n_j == 3) & (sel_n_b == 1)

        sel_fatjet = (
            (events.FatJet.pt > 200)
            & (events.FatJet.pt < 3000)
            & (events.FatJet.eta < 2.0)
            & (events.FatJet.eta > -2.0)
            & (events.FatJet.mass > 40)
            & (events.FatJet.mass < 600)
        )
        sel_n_fj = ak.sum(sel_fatjet, axis=-1)

        # ===== sel_ photon particles (pad 3) =====
        sel3_ph = pad(events.Photon[sel_ph], 3)
        sel3_ph_4vec = ak.zip(
            {
                "pt": sel3_ph.pt,
                "eta": sel3_ph.eta,
                "phi": sel3_ph.phi,
                "mass": ak.zeros_like(sel3_ph.pt),
                "charge": ak.zeros_like(sel3_ph.pt),
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        # ===== sel_ jet particles (pad 4) =====
        sel4_jet = pad(events.Jet[sel_jet], 4)
        sel4_jet_4vec = ak.zip(
            {
                "pt": sel4_jet.pt,
                "eta": sel4_jet.eta,
                "phi": sel4_jet.phi,
                "mass": sel4_jet.mass,
                "charge": ak.zeros_like(sel4_jet.pt),
                "BTag": sel4_jet.BTag,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        # ===== sel_ bjet particles (pad 4) =====
        sel4_bjet = pad(events.Jet[sel_bjet], 4)
        sel4_bjet_4vec = ak.zip(
            {
                "pt": sel4_bjet.pt,
                "eta": sel4_bjet.eta,
                "phi": sel4_bjet.phi,
                "mass": sel4_bjet.mass,
                "charge": ak.zeros_like(sel4_bjet.pt),
                "BTag": sel4_bjet.BTag,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        # ===== sel_ fatjet particles (pad 2) =====
        sel2_fj = pad(events.FatJet[sel_fatjet], 2)
        sel2_fj_tau = ak.pad_none(sel2_fj.Tau_5, target=4, clip=True, axis=-1)
        sel_fj_tau21 = sel2_fj_tau[:, :, 1] / sel2_fj_tau[:, :, 0]
        sel_fj_tau32 = sel2_fj_tau[:, :, 2] / sel2_fj_tau[:, :, 1]
        sel_fj_tau43 = sel2_fj_tau[:, :, 3] / sel2_fj_tau[:, :, 2]
        sel2_fj_4vec = ak.zip(
            {
                "pt": sel2_fj.pt,
                "eta": sel2_fj.eta,
                "phi": sel2_fj.phi,
                "mass": sel2_fj.mass,
                "charge": ak.zeros_like(sel2_fj.pt),
                "BTag": sel2_fj.BTag,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        # ===== inc_ fatjets (pad 2) =====
        inc_fatjets = pad(events.FatJet, 2)
        fatjets_tau = ak.pad_none(inc_fatjets.Tau_5, target=4, clip=True, axis=-1)
        inc_fatjets_tau21 = fatjets_tau[:, :, 1] / fatjets_tau[:, :, 0]
        inc_fatjets_tau32 = fatjets_tau[:, :, 2] / fatjets_tau[:, :, 1]
        inc_fatjets_tau43 = fatjets_tau[:, :, 3] / fatjets_tau[:, :, 2]
        inc_fatjets = ak.zip(
            {
                "pt": inc_fatjets.pt,
                "eta": inc_fatjets.eta,
                "phi": inc_fatjets.phi,
                "mass": inc_fatjets.mass,
                "charge": ak.zeros_like(inc_fatjets.pt),
                "BTag": inc_fatjets.BTag,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        # ===== inc_ muons (pad 2) =====
        inc_muons = pad(events.Muon, 2)
        inc_muons = ak.zip(
            {
                "pt": inc_muons.pt,
                "eta": inc_muons.eta,
                "phi": inc_muons.phi,
                "mass": inc_muons.mass,
                "charge": inc_muons.Charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        inc_n_muons = ak.num(events.Muon)

        # ===== inc_ electrons (pad 2) =====
        inc_electrons = pad(events.Electron, 2)
        inc_electrons = ak.zip(
            {
                "pt": inc_electrons.pt,
                "eta": inc_electrons.eta,
                "phi": inc_electrons.phi,
                "mass": inc_electrons.mass,
                "charge": inc_electrons.Charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        inc_n_electrons = ak.num(events.Electron)

        has_lepton = (inc_n_muons > 0) | (inc_n_electrons > 0)

        met_pt = events.MissingET.MET
        met_phi = events.MissingET.phi

        # misc event features
        n_photons = ak.num(events.Photon)
        n_jets = ak.num(events.Jet)
        n_fatjets = ak.num(events.FatJet)
        event_weight = events.Event.Weight
        event_number = events.Event.Number

        # event selection (pT-ordered photons for standard ATLAS cuts)
        """
        # Main Selection:
        ## Event
        HasPrimaryVertex: HGamEventInfoAuxDyn.numberOfPrimaryVertices > 0 -> dont
        TwoLossePhotons + e y ambiguity cut: HGamEventInfoAuxDyn.NLoosePhotons >= 2 -> implemented
        ## Trigger:
        Full implementation (EventInfoAuxDyn.passTrig_HLT_g35_medium_g25_medium_L12EM20VH || EventInfoAuxDyn.passTrig_HLT_g140_loose || EventInfoAuxDyn.passTrig_HLT_g35_loose_g25_loose || EventInfoAuxDyn.passTrig_HLT_g120_loose)
        Di lepton triggers: lead photon pT > 35 && sublead photon pT > 25 -> implemented
        Trigger Matching: HGamEventInfoAuxDyn.isPassedTriggerMatch -> dont
        ## Photons
        Tight Photon ID: HGamEventInfoAuxDyn.isPassedPID -> dont
        Photon Iso: HGamEventInfoAuxDyn.isPassedIsolation -> later?
        https://gitlab.cern.ch/atlas/athena/-/blob/main/PhysicsAnalysis/AnalysisCommon/IsolationSelection/Root/IsolationSelectionTool.cxx#L220-251
        https://gitlab.cern.ch/atlas/athena/-/blob/main/PhysicsAnalysis/AnalysisCommon/IsolationSelection/Root/IsolationConditionFormula.cxx#L44
        Relative Pt cut (The two leading, pre-selected photons pass the 0.4 / 0.3 relative pT cuts, relative to myy): HGamEventInfoAuxDyn.isPassedRelPtCuts -> implemented
        myy mass window cut: (HGamEventInfoAuxDyn.m_yy > 105000) && (HGamEventInfoAuxDyn.m_yy < 160000) -> implemented
        """
        good = n_photons >= 2
        # Trigger (pT-ordered leading photons)
        trigger = ((inc_photons.pt[:, 0] > 35) & (inc_photons.pt[:, 1] > 25)) | (inc_photons.pt[:, 0] > 140)  # fmt: skip # fine since the pT/myy cut 0.3 and myy cut with 105GeV ~ pT2>35GeV
        good = good & trigger
        # Rel pT cut (pT-ordered diphoton)
        rel_pt_cut = (inc_photon1_pt_rel > 0.4) & (inc_photon2_pt_rel > 0.3)
        good = good & rel_pt_cut
        # Myy mass window (pT-ordered diphoton)
        myy_cut = (diphoton_mass > 105) & (diphoton_mass < 160)
        good = good & myy_cut
        # Prevent None in mask
        good = ak.fill_none(good, False)

        scale = lambda x: x * 1_000
        return {
            "cutflow": {
                "n_total": ak.num(good, axis=0),
                "n_good": ak.sum(good),
                "sumw_presel": ak.sum(event_weight),
                "sumw_postsel": ak.sum(event_weight[good]),
            },
            "events": ak.zip(
                {
                    # === diphoton (select_pair) ===
                    "diphoton_mass": scale(diphoton_mass)[good],
                    "diphoton_pt": scale(diphoton_pt)[good],
                    "diphoton_delta_R": diphoton_delta_r[good],
                    # === event-level ===
                    "dijet_mass": scale(dijet_mass)[good],
                    "dijet_delta_R": dijet_delta_r[good],
                    "HT_30": scale(ht_30)[good],
                    "has_lepton": has_lepton[good],
                    "met_pt": scale(met_pt)[good],
                    "met_phi": met_phi[good],
                    "n_photon": n_photons[good],
                    "n_jet": n_jets[good],
                    "n_fatjet": n_fatjets[good],
                    "event_weight": event_weight[good],
                    "event_number": event_number[good],
                    # === sel_ (cut-based) ===
                    "sel_n_pho": sel_n_ph[good],
                    "sel_n_lep": sel_n_lep[good],
                    "sel_n_jet": sel_n_j[good],
                    "sel_n_bjt": sel_n_b[good],
                    "sel_is_Zveto": sel_is_Zveto[good],
                    "sel_is_SS": sel_is_SS[good],
                    "sel_is_lb": sel_is_lb[good],
                    "sel_is_tlep": sel_is_tlep[good],
                    "sel_is_thad": sel_is_thad[good],
                    "sel_n_fj": sel_n_fj[good],
                    # === sel_ photon particles (pad 3) ===
                    "sel_photon1_pt": scale(sel3_ph_4vec.pt[:, 0])[good],
                    "sel_photon1_eta": sel3_ph_4vec.eta[:, 0][good],
                    "sel_photon1_phi": sel3_ph_4vec.phi[:, 0][good],
                    "sel_photon1_m": scale(sel3_ph_4vec.mass[:, 0])[good],
                    "sel_photon2_pt": scale(sel3_ph_4vec.pt[:, 1])[good],
                    "sel_photon2_eta": sel3_ph_4vec.eta[:, 1][good],
                    "sel_photon2_phi": sel3_ph_4vec.phi[:, 1][good],
                    "sel_photon2_m": scale(sel3_ph_4vec.mass[:, 1])[good],
                    "sel_photon3_pt": scale(sel3_ph_4vec.pt[:, 2])[good],
                    "sel_photon3_eta": sel3_ph_4vec.eta[:, 2][good],
                    "sel_photon3_phi": sel3_ph_4vec.phi[:, 2][good],
                    "sel_photon3_m": scale(sel3_ph_4vec.mass[:, 2])[good],
                    # === sel_ muon particles (pad 2) ===
                    "sel_muon1_pt": scale(sel2_mu_4vec.pt[:, 0])[good],
                    "sel_muon1_eta": sel2_mu_4vec.eta[:, 0][good],
                    "sel_muon1_phi": sel2_mu_4vec.phi[:, 0][good],
                    "sel_muon1_m": scale(sel2_mu_4vec.mass[:, 0])[good],
                    "sel_muon2_pt": scale(sel2_mu_4vec.pt[:, 1])[good],
                    "sel_muon2_eta": sel2_mu_4vec.eta[:, 1][good],
                    "sel_muon2_phi": sel2_mu_4vec.phi[:, 1][good],
                    "sel_muon2_m": scale(sel2_mu_4vec.mass[:, 1])[good],
                    # === sel_ electron particles (pad 2) ===
                    "sel_electron1_pt": scale(sel2_el_4vec.pt[:, 0])[good],
                    "sel_electron1_eta": sel2_el_4vec.eta[:, 0][good],
                    "sel_electron1_phi": sel2_el_4vec.phi[:, 0][good],
                    "sel_electron1_m": scale(sel2_el_4vec.mass[:, 0])[good],
                    "sel_electron2_pt": scale(sel2_el_4vec.pt[:, 1])[good],
                    "sel_electron2_eta": sel2_el_4vec.eta[:, 1][good],
                    "sel_electron2_phi": sel2_el_4vec.phi[:, 1][good],
                    "sel_electron2_m": scale(sel2_el_4vec.mass[:, 1])[good],
                    # === sel_ jet particles (pad 4) ===
                    "sel_jet1_pt": scale(sel4_jet_4vec.pt[:, 0])[good],
                    "sel_jet1_eta": sel4_jet_4vec.eta[:, 0][good],
                    "sel_jet1_phi": sel4_jet_4vec.phi[:, 0][good],
                    "sel_jet1_m": scale(sel4_jet_4vec.mass[:, 0])[good],
                    "sel_jet1_btag": ak.fill_none(sel4_jet_4vec.BTag[:, 0][good], np.nan),
                    "sel_jet2_pt": scale(sel4_jet_4vec.pt[:, 1])[good],
                    "sel_jet2_eta": sel4_jet_4vec.eta[:, 1][good],
                    "sel_jet2_phi": sel4_jet_4vec.phi[:, 1][good],
                    "sel_jet2_m": scale(sel4_jet_4vec.mass[:, 1])[good],
                    "sel_jet2_btag": ak.fill_none(sel4_jet_4vec.BTag[:, 1][good], np.nan),
                    "sel_jet3_pt": scale(sel4_jet_4vec.pt[:, 2])[good],
                    "sel_jet3_eta": sel4_jet_4vec.eta[:, 2][good],
                    "sel_jet3_phi": sel4_jet_4vec.phi[:, 2][good],
                    "sel_jet3_m": scale(sel4_jet_4vec.mass[:, 2])[good],
                    "sel_jet3_btag": ak.fill_none(sel4_jet_4vec.BTag[:, 2][good], np.nan),
                    "sel_jet4_pt": scale(sel4_jet_4vec.pt[:, 3])[good],
                    "sel_jet4_eta": sel4_jet_4vec.eta[:, 3][good],
                    "sel_jet4_phi": sel4_jet_4vec.phi[:, 3][good],
                    "sel_jet4_m": scale(sel4_jet_4vec.mass[:, 3])[good],
                    "sel_jet4_btag": ak.fill_none(sel4_jet_4vec.BTag[:, 3][good], np.nan),
                    # === sel_ bjet particles (pad 4) ===
                    "sel_bjet1_pt": scale(sel4_bjet_4vec.pt[:, 0])[good],
                    "sel_bjet1_eta": sel4_bjet_4vec.eta[:, 0][good],
                    "sel_bjet1_phi": sel4_bjet_4vec.phi[:, 0][good],
                    "sel_bjet1_m": scale(sel4_bjet_4vec.mass[:, 0])[good],
                    "sel_bjet1_btag": ak.fill_none(sel4_bjet_4vec.BTag[:, 0][good], np.nan),
                    "sel_bjet2_pt": scale(sel4_bjet_4vec.pt[:, 1])[good],
                    "sel_bjet2_eta": sel4_bjet_4vec.eta[:, 1][good],
                    "sel_bjet2_phi": sel4_bjet_4vec.phi[:, 1][good],
                    "sel_bjet2_m": scale(sel4_bjet_4vec.mass[:, 1])[good],
                    "sel_bjet2_btag": ak.fill_none(sel4_bjet_4vec.BTag[:, 1][good], np.nan),
                    "sel_bjet3_pt": scale(sel4_bjet_4vec.pt[:, 2])[good],
                    "sel_bjet3_eta": sel4_bjet_4vec.eta[:, 2][good],
                    "sel_bjet3_phi": sel4_bjet_4vec.phi[:, 2][good],
                    "sel_bjet3_m": scale(sel4_bjet_4vec.mass[:, 2])[good],
                    "sel_bjet3_btag": ak.fill_none(sel4_bjet_4vec.BTag[:, 2][good], np.nan),
                    "sel_bjet4_pt": scale(sel4_bjet_4vec.pt[:, 3])[good],
                    "sel_bjet4_eta": sel4_bjet_4vec.eta[:, 3][good],
                    "sel_bjet4_phi": sel4_bjet_4vec.phi[:, 3][good],
                    "sel_bjet4_m": scale(sel4_bjet_4vec.mass[:, 3])[good],
                    "sel_bjet4_btag": ak.fill_none(sel4_bjet_4vec.BTag[:, 3][good], np.nan),
                    # === sel_ fatjet particles (pad 2) ===
                    "sel_fatjet1_pt": scale(sel2_fj_4vec.pt[:, 0])[good],
                    "sel_fatjet1_eta": sel2_fj_4vec.eta[:, 0][good],
                    "sel_fatjet1_phi": sel2_fj_4vec.phi[:, 0][good],
                    "sel_fatjet1_m": scale(sel2_fj_4vec.mass[:, 0])[good],
                    "sel_fatjet1_btag": ak.fill_none(sel2_fj_4vec.BTag[:, 0][good], np.nan),
                    "sel_fatjet1_tau21": sel_fj_tau21[:, 0][good],
                    "sel_fatjet1_tau32": sel_fj_tau32[:, 0][good],
                    "sel_fatjet1_tau43": sel_fj_tau43[:, 0][good],
                    "sel_fatjet2_pt": scale(sel2_fj_4vec.pt[:, 1])[good],
                    "sel_fatjet2_eta": sel2_fj_4vec.eta[:, 1][good],
                    "sel_fatjet2_phi": sel2_fj_4vec.phi[:, 1][good],
                    "sel_fatjet2_m": scale(sel2_fj_4vec.mass[:, 1])[good],
                    "sel_fatjet2_btag": ak.fill_none(sel2_fj_4vec.BTag[:, 1][good], np.nan),
                    "sel_fatjet2_tau21": sel_fj_tau21[:, 1][good],
                    "sel_fatjet2_tau32": sel_fj_tau32[:, 1][good],
                    "sel_fatjet2_tau43": sel_fj_tau43[:, 1][good],
                    # === pair_ photons (select_pair Higgs matching) ===
                    "pair_diphoton_mass": scale(pair_diphoton_mass)[good],
                    "pair_diphoton_pt": scale(pair_diphoton_pt)[good],
                    "pair_diphoton_delta_R": pair_diphoton_delta_r[good],
                    "pair_photon1_pt": scale(pair_photons.pt[:, 0])[good],
                    "pair_photon1_eta": pair_photons.eta[:, 0][good],
                    "pair_photon1_phi": pair_photons.phi[:, 0][good],
                    "pair_photon1_m": scale(pair_photons.mass[:, 0])[good],
                    "pair_photon2_pt": scale(pair_photons.pt[:, 1])[good],
                    "pair_photon2_eta": pair_photons.eta[:, 1][good],
                    "pair_photon2_phi": pair_photons.phi[:, 1][good],
                    "pair_photon2_m": scale(pair_photons.mass[:, 1])[good],
                    # === inc_ photons (pT-ordered, pad 3) ===
                    "inc_photon1_pt": scale(inc_photons.pt[:, 0])[good],
                    "inc_photon1_eta": inc_photons.eta[:, 0][good],
                    "inc_photon1_phi": inc_photons.phi[:, 0][good],
                    "inc_photon1_m": scale(inc_photons.mass[:, 0])[good],
                    "inc_photon1_pt_rel": inc_photon1_pt_rel[good],
                    "inc_photon2_pt": scale(inc_photons.pt[:, 1])[good],
                    "inc_photon2_eta": inc_photons.eta[:, 1][good],
                    "inc_photon2_phi": inc_photons.phi[:, 1][good],
                    "inc_photon2_m": scale(inc_photons.mass[:, 1])[good],
                    "inc_photon2_pt_rel": inc_photon2_pt_rel[good],
                    "inc_photon3_pt": scale(inc_photons.pt[:, 2])[good],
                    "inc_photon3_eta": inc_photons.eta[:, 2][good],
                    "inc_photon3_phi": inc_photons.phi[:, 2][good],
                    "inc_photon3_m": scale(inc_photons.mass[:, 2])[good],
                    "inc_photon3_pt_rel": inc_photon3_pt_rel[good],
                    # === inc_ muons (pad 2) ===
                    "inc_muon1_pt": scale(inc_muons.pt[:, 0])[good],
                    "inc_muon1_eta": inc_muons.eta[:, 0][good],
                    "inc_muon1_phi": inc_muons.phi[:, 0][good],
                    "inc_muon1_m": scale(inc_muons.mass[:, 0])[good],
                    "inc_muon2_pt": scale(inc_muons.pt[:, 1])[good],
                    "inc_muon2_eta": inc_muons.eta[:, 1][good],
                    "inc_muon2_phi": inc_muons.phi[:, 1][good],
                    "inc_muon2_m": scale(inc_muons.mass[:, 1])[good],
                    "inc_n_muon": inc_n_muons[good],
                    # === inc_ electrons (pad 2) ===
                    "inc_electron1_pt": scale(inc_electrons.pt[:, 0])[good],
                    "inc_electron1_eta": inc_electrons.eta[:, 0][good],
                    "inc_electron1_phi": inc_electrons.phi[:, 0][good],
                    "inc_electron1_m": scale(inc_electrons.mass[:, 0])[good],
                    "inc_electron2_pt": scale(inc_electrons.pt[:, 1])[good],
                    "inc_electron2_eta": inc_electrons.eta[:, 1][good],
                    "inc_electron2_phi": inc_electrons.phi[:, 1][good],
                    "inc_electron2_m": scale(inc_electrons.mass[:, 1])[good],
                    "inc_n_electron": inc_n_electrons[good],
                    # === inc_ jets (pad 4) ===
                    "inc_jet1_pt": scale(inc_jets.pt[:, 0])[good],
                    "inc_jet1_eta": inc_jets.eta[:, 0][good],
                    "inc_jet1_phi": inc_jets.phi[:, 0][good],
                    "inc_jet1_m": scale(inc_jets.mass[:, 0])[good],
                    "inc_jet1_btag": ak.fill_none(inc_jets.BTag[:, 0][good], np.nan),
                    "inc_jet2_pt": scale(inc_jets.pt[:, 1])[good],
                    "inc_jet2_eta": inc_jets.eta[:, 1][good],
                    "inc_jet2_phi": inc_jets.phi[:, 1][good],
                    "inc_jet2_m": scale(inc_jets.mass[:, 1])[good],
                    "inc_jet2_btag": ak.fill_none(inc_jets.BTag[:, 1][good], np.nan),
                    "inc_jet3_pt": scale(inc_jets.pt[:, 2])[good],
                    "inc_jet3_eta": inc_jets.eta[:, 2][good],
                    "inc_jet3_phi": inc_jets.phi[:, 2][good],
                    "inc_jet3_m": scale(inc_jets.mass[:, 2])[good],
                    "inc_jet3_btag": ak.fill_none(inc_jets.BTag[:, 2][good], np.nan),
                    "inc_jet4_pt": scale(inc_jets.pt[:, 3])[good],
                    "inc_jet4_eta": inc_jets.eta[:, 3][good],
                    "inc_jet4_phi": inc_jets.phi[:, 3][good],
                    "inc_jet4_m": scale(inc_jets.mass[:, 3])[good],
                    "inc_jet4_btag": ak.fill_none(inc_jets.BTag[:, 3][good], np.nan),
                    # === inc_ fatjets (pad 2) ===
                    "inc_fatjet1_pt": scale(inc_fatjets.pt[:, 0])[good],
                    "inc_fatjet1_eta": inc_fatjets.eta[:, 0][good],
                    "inc_fatjet1_phi": inc_fatjets.phi[:, 0][good],
                    "inc_fatjet1_m": scale(inc_fatjets.mass[:, 0])[good],
                    "inc_fatjet1_btag": ak.fill_none(inc_fatjets.BTag[:, 0][good], np.nan),
                    "inc_fatjet1_tau21": inc_fatjets_tau21[:, 0][good],
                    "inc_fatjet1_tau32": inc_fatjets_tau32[:, 0][good],
                    "inc_fatjet1_tau43": inc_fatjets_tau43[:, 0][good],
                    "inc_fatjet2_pt": scale(inc_fatjets.pt[:, 1])[good],
                    "inc_fatjet2_eta": inc_fatjets.eta[:, 1][good],
                    "inc_fatjet2_phi": inc_fatjets.phi[:, 1][good],
                    "inc_fatjet2_m": scale(inc_fatjets.mass[:, 1])[good],
                    "inc_fatjet2_btag": ak.fill_none(inc_fatjets.BTag[:, 1][good], np.nan),
                    "inc_fatjet2_tau21": inc_fatjets_tau21[:, 1][good],
                    "inc_fatjet2_tau32": inc_fatjets_tau32[:, 1][good],
                    "inc_fatjet2_tau43": inc_fatjets_tau43[:, 1][good],
                }
            ),
        }
