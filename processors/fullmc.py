import numpy as np
import awkward as ak
import coffea.processor as processor

from .common import (
    pad,
    select_pair,
    to_candidate,
    Z_MASS_GEV,
    EL_MASS_GEV,
    MU_MASS_GEV,
)


class Processor(processor.ProcessorABC):
    def postprocess(self, accumulator):
        pass

    def get_cut_photons(self, inc_photons):
        """Photon collection that drives the trigger / rel-pT / myy cuts.
        Override in subclasses to use the select_pair'd pair instead of pT-ordered."""
        return inc_photons

    def process(self, events):
        # ===== inc_ photons (pT-ordered, pad 3) =====
        inc_photons = to_candidate(pad(events.Photon, 3), mass=0)
        diphoton_mass = (inc_photons[:, 0] + inc_photons[:, 1]).mass
        diphoton_pt = (inc_photons[:, 0] + inc_photons[:, 1]).pt
        diphoton_delta_r = inc_photons[:, 0].delta_r(inc_photons[:, 1])
        inc_photons_pt_rel = inc_photons.pt / diphoton_mass[:, None]

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
        sel_ph = (events.Photon.pt > 22) & (abs(events.Photon.eta) < 2.5)
        sel_n_ph = ak.sum(sel_ph, axis=-1)

        # leptons
        sel_el = (events.Electron.pt > 10) & (abs(events.Electron.eta) < 2.5)
        sel_mu = (events.Muon.pt > 10) & (abs(events.Muon.eta) < 2.7)
        sel_n_e = ak.sum(sel_el, axis=-1)
        sel_n_mu = ak.sum(sel_mu, axis=-1)
        sel_is_ee = sel_n_e == 2
        sel_is_mumu = sel_n_mu == 2
        sel_is_emu = (sel_n_e == 1) & (sel_n_mu == 1)

        sel2_el_4vec = to_candidate(pad(events.Electron[sel_el], 2), mass=EL_MASS_GEV, charge=True)
        sel2_mu_4vec = to_candidate(pad(events.Muon[sel_mu], 2), mass=MU_MASS_GEV, charge=True)

        # charge / flavor analysis
        sel2_el_ch = sel2_el_4vec.charge
        sel2_mu_ch = sel2_mu_4vec.charge
        sel_is_SS = (
            (sel_is_ee & (sel2_el_ch[:, 0] * sel2_el_ch[:, 1] > 0))
            | (sel_is_emu & (sel2_el_ch[:, 0] * sel2_mu_ch[:, 1] > 0))
            | (sel_is_mumu & (sel2_mu_ch[:, 0] * sel2_mu_ch[:, 1] > 0))
        )

        # Z peak veto
        m_ee = (sel2_el_4vec[:, 0] + sel2_el_4vec[:, 1]).mass
        m_mm = (sel2_mu_4vec[:, 0] + sel2_mu_4vec[:, 1]).mass
        sel_is_Zveto = ~(
            (sel_is_ee & (m_ee > Z_MASS_GEV - 10) & (m_ee < Z_MASS_GEV + 10))
            | (sel_is_mumu & (m_mm > Z_MASS_GEV - 10) & (m_mm < Z_MASS_GEV + 10))
        )

        # ===== jets =====
        inc_jets = to_candidate(pad(events.Jet, 4), btag=True)
        dijet_mass = (inc_jets[:, 0] + inc_jets[:, 1]).m
        dijet_delta_r = inc_jets[:, 0].delta_r(inc_jets[:, 1])
        ht_30 = ak.sum(events.Jet.pt[events.Jet.pt > 30], axis=-1)

        sel_jet = (events.Jet.pt > 25) & (abs(events.Jet.eta) < 4.4)
        sel_bjet = (events.Jet.pt > 25) & (abs(events.Jet.eta) < 2.5) & (events.Jet.BTag == 1)
        sel_n_j = ak.sum(sel_jet, axis=-1)
        sel_n_b = ak.sum(sel_bjet, axis=-1)
        sel_n_lep = sel_n_e + sel_n_mu
        sel_is_lb = (sel_n_lep >= 1) & (sel_n_b >= 1)
        sel_is_tlep = (sel_n_lep == 1) & (sel_n_j == 1) & (sel_n_b == 1)
        sel_is_thad = (sel_n_lep == 0) & (sel_n_j == 3) & (sel_n_b == 1)

        sel4_jet_4vec = to_candidate(pad(events.Jet[sel_jet], 4), btag=True)
        sel4_bjet_4vec = to_candidate(pad(events.Jet[sel_bjet], 4), btag=True)

        # ===== sel_ photons =====
        sel3_ph_4vec = to_candidate(pad(events.Photon[sel_ph], 3), mass=0)

        # ===== fatjets =====
        sel_fatjet = (
            (events.FatJet.pt > 200)
            & (events.FatJet.pt < 3000)
            & (abs(events.FatJet.eta) < 2.0)
            & (events.FatJet.mass > 40)
            & (events.FatJet.mass < 600)
        )
        sel_n_fj = ak.sum(sel_fatjet, axis=-1)

        sel2_fatjets_pad = pad(events.FatJet[sel_fatjet], 2)
        sel2_fatjets = to_candidate(sel2_fatjets_pad, btag=True)
        sel2_fatjets_tau = ak.pad_none(sel2_fatjets_pad.Tau_5, target=4, clip=True, axis=-1)
        sel2_fatjets_tau21 = sel2_fatjets_tau[:, :, 1] / sel2_fatjets_tau[:, :, 0]
        sel2_fatjets_tau32 = sel2_fatjets_tau[:, :, 2] / sel2_fatjets_tau[:, :, 1]
        sel2_fatjets_tau43 = sel2_fatjets_tau[:, :, 3] / sel2_fatjets_tau[:, :, 2]

        inc_fatjets_pad = pad(events.FatJet, 2)
        inc_fatjets = to_candidate(inc_fatjets_pad, btag=True)
        inc_fatjets_tau = ak.pad_none(inc_fatjets_pad.Tau_5, target=4, clip=True, axis=-1)
        inc_fatjets_tau21 = inc_fatjets_tau[:, :, 1] / inc_fatjets_tau[:, :, 0]
        inc_fatjets_tau32 = inc_fatjets_tau[:, :, 2] / inc_fatjets_tau[:, :, 1]
        inc_fatjets_tau43 = inc_fatjets_tau[:, :, 3] / inc_fatjets_tau[:, :, 2]

        # ===== inc_ leptons =====
        inc_muons = to_candidate(pad(events.Muon, 2), charge=True)
        inc_n_muons = ak.num(events.Muon)

        inc_electrons = to_candidate(pad(events.Electron, 2), charge=True)
        inc_n_electrons = ak.num(events.Electron)
        has_lepton = (inc_n_muons > 0) | (inc_n_electrons > 0)

        met_pt = events.MissingET.MET
        met_phi = events.MissingET.phi

        n_photons = ak.num(events.Photon)
        n_jets = ak.num(events.Jet)
        n_fatjets = ak.num(events.FatJet)
        event_weight = events.Event.Weight
        event_number = events.Event.Number

        # ===== event selection =====
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
        cut_photons = self.get_cut_photons(inc_photons)
        cut_diphoton_mass = (cut_photons[:, 0] + cut_photons[:, 1]).mass
        cut_pt_rel1 = cut_photons.pt[:, 0] / cut_diphoton_mass
        cut_pt_rel2 = cut_photons.pt[:, 1] / cut_diphoton_mass

        good = n_photons >= 2
        # Trigger
        trigger = ((cut_photons.pt[:, 0] > 35) & (cut_photons.pt[:, 1] > 25)) | (cut_photons.pt[:, 0] > 140)  # fmt: skip
        good = good & trigger
        # Rel pT cut
        good = good & (cut_pt_rel1 > 0.4) & (cut_pt_rel2 > 0.3)
        # Myy mass window
        good = good & (cut_diphoton_mass > 105) & (cut_diphoton_mass < 160)
        # Prevent None in mask
        good = ak.fill_none(good, False)

        scale = lambda x: x * 1_000

        # ===== build output =====
        output = {
            # diphoton (pT-ordered)
            "diphoton_mass": scale(diphoton_mass)[good],
            "diphoton_pt": scale(diphoton_pt)[good],
            "diphoton_delta_R": diphoton_delta_r[good],
            # event-level
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
            # sel_ counts and flags
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
            # pair_ diphoton
            "pair_diphoton_mass": scale(pair_diphoton_mass)[good],
            "pair_diphoton_pt": scale(pair_diphoton_pt)[good],
            "pair_diphoton_delta_R": pair_diphoton_delta_r[good],
            # inc lepton counts
            "inc_n_muon": inc_n_muons[good],
            "inc_n_electron": inc_n_electrons[good],
        }

        # particle-level kinematics
        # (name, collection, n, has_btag)
        particle_blocks = [
            ("sel_photon", sel3_ph_4vec, 3, False),
            ("sel_muon", sel2_mu_4vec, 2, False),
            ("sel_electron", sel2_el_4vec, 2, False),
            ("sel_jet", sel4_jet_4vec, 4, True),
            ("sel_bjet", sel4_bjet_4vec, 4, True),
            ("pair_photon", pair_photons, 2, False),
            ("inc_photon", inc_photons, 3, False),
            ("inc_muon", inc_muons, 2, False),
            ("inc_electron", inc_electrons, 2, False),
            ("inc_jet", inc_jets, 4, True),
        ]
        for name, coll, n, has_btag in particle_blocks:
            for i in range(n):
                output[f"{name}{i+1}_pt"] = scale(coll.pt[:, i])[good]
                output[f"{name}{i+1}_eta"] = coll.eta[:, i][good]
                output[f"{name}{i+1}_phi"] = coll.phi[:, i][good]
                output[f"{name}{i+1}_m"] = scale(coll.mass[:, i])[good]
                if has_btag:
                    output[f"{name}{i+1}_btag"] = ak.fill_none(coll.BTag[:, i][good], np.nan)

        # fatjets carry tau ratios alongside kinematics + btag
        fatjet_blocks = [
            ("sel_fatjet", sel2_fatjets, 2, sel2_fatjets_tau21, sel2_fatjets_tau32, sel2_fatjets_tau43),  # fmt: skip
            ("inc_fatjet", inc_fatjets,  2, inc_fatjets_tau21,  inc_fatjets_tau32,  inc_fatjets_tau43),  # fmt: skip
        ]
        for name, coll, n, t21, t32, t43 in fatjet_blocks:
            for i in range(n):
                output[f"{name}{i+1}_pt"] = scale(coll.pt[:, i])[good]
                output[f"{name}{i+1}_eta"] = coll.eta[:, i][good]
                output[f"{name}{i+1}_phi"] = coll.phi[:, i][good]
                output[f"{name}{i+1}_m"] = scale(coll.mass[:, i])[good]
                output[f"{name}{i+1}_btag"] = ak.fill_none(coll.BTag[:, i][good], np.nan)
                output[f"{name}{i+1}_tau21"] = t21[:, i][good]
                output[f"{name}{i+1}_tau32"] = t32[:, i][good]
                output[f"{name}{i+1}_tau43"] = t43[:, i][good]

        # inc photon pt_rel (relative to inc diphoton mass)
        for i in range(3):
            output[f"inc_photon{i+1}_pt_rel"] = inc_photons_pt_rel[:, i][good]

        return {
            "cutflow": {
                "n_total": ak.num(good, axis=0),
                "n_good": ak.sum(good),
                "sumw_presel": ak.sum(event_weight),
                "sumw_postsel": ak.sum(event_weight[good]),
            },
            "events": ak.zip(output),
        }
