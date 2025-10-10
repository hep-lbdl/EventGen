import numpy as np
import awkward as ak
import coffea.processor as processor
from coffea.nanoevents.methods import candidate


class Processor(processor.ProcessorABC):
    def postprocess(self, accumulator):
        pass

    def process(self, events):
        pad = lambda x, target: ak.pad_none(x, target=target, clip=True)
        photons = pad(events.Photon, 2)
        hgamma = ak.zip(
            {
                "pt": photons.pt,
                "eta": photons.eta,
                "phi": photons.phi,
                "mass": ak.zeros_like(photons.pt),
                "charge": ak.zeros_like(photons.pt),
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        diphoton_mass = (hgamma[:, 0] + hgamma[:, 1]).mass
        diphoton_pt = (hgamma[:, 0] + hgamma[:, 1]).pt
        diphoton_delta_r = hgamma[:, 0].delta_r(hgamma[:, 1])
        gamma_pt_rel = photons.pt / diphoton_mass[:, None]
        photon1_pt_rel, photon2_pt_rel = gamma_pt_rel[:, 0], gamma_pt_rel[:, 1]

        jets = pad(events.Jet, 4)
        jets = ak.zip(
            {
                "pt": jets.pt,
                "eta": jets.eta,
                "phi": jets.phi,
                "mass": jets.mass,
                "charge": ak.zeros_like(jets.pt),
                "BTagPhys": jets.BTagPhys,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        dijet_mass = (jets[:, 0] + jets[:, 1]).m
        dijet_delta_r = jets[:, 0].delta_r(jets[:, 1])
        ht_30 = ak.sum(events.Jet.pt[events.Jet.pt > 30], axis=-1)

        fatjets = pad(events.FatJet, 2)
        fatjets_tau = ak.pad_none(fatjets.Tau_5, target=4, clip=True, axis=-1)
        fatjets_tau21 = fatjets_tau[:, :, 1] / fatjets_tau[:, :, 0]
        fatjets_tau32 = fatjets_tau[:, :, 2] / fatjets_tau[:, :, 1]
        fatjets_tau43 = fatjets_tau[:, :, 3] / fatjets_tau[:, :, 2]
        fatjets = ak.zip(
            {
                "pt": fatjets.pt,
                "eta": fatjets.eta,
                "phi": fatjets.phi,
                "mass": fatjets.mass,
                "charge": ak.zeros_like(fatjets.pt),
                "BTagPhys": fatjets.BTagPhys,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        muons = pad(events.Muon, 2)
        muons = ak.zip(
            {
                "pt": muons.pt,
                "eta": muons.eta,
                "phi": muons.phi,
                "mass": muons.mass,
                "charge": muons.Charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        n_muons = ak.num(muons)

        electrons = pad(events.Electron, 2)
        electrons = ak.zip(
            {
                "pt": electrons.pt,
                "eta": electrons.eta,
                "phi": electrons.phi,
                "mass": electrons.mass,
                "charge": electrons.Charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        n_electrons = ak.num(electrons)

        has_lepton = (n_muons > 0) | (n_electrons > 0)

        met_pt = events.MissingET.MET
        met_phi = events.MissingET.phi

        # misc event features
        n_photons = ak.num(events.Photon)
        n_jets = ak.num(events.Jet)
        n_fatjets = ak.num(events.FatJet)
        event_weight = events.Event.Weight
        event_number = events.Event.Number

        # event selection
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
        # Trigger
        trigger = ((photons[:, 0].pt > 35) & (photons[:, 1].pt > 25)) | (photons[:, 0].pt > 140)  # fmt: skip
        good = good & trigger
        # Rel pT cut
        rel_pt_cut = (photon1_pt_rel > 0.4) & (photon2_pt_rel > 0.3)
        good = good & rel_pt_cut
        # Myy mass window
        myy_cut = (diphoton_mass > 105) & (diphoton_mass < 160)
        good = good & myy_cut
        # Prevent None in mask
        good = ak.fill_none(good, False)
        scale = lambda x: x * 1_000

        output = dict()
        for particle, collection, n in [
            ("photon", hgamma, 2),
            ("muon", muons, 2),
            ("electron", electrons, 2),
            ("jet", jets, 4),
            ("fatjet", fatjets, 2),
        ]:
            for i in range(n):
                output[f"{particle}{i+1}_pt"] = scale(collection.pt[:, i])[good]
                output[f"{particle}{i+1}_eta"] = collection.eta[:, i][good]
                output[f"{particle}{i+1}_phi"] = collection.phi[:, i][good]
                output[f"{particle}{i+1}_m"] = scale(collection.mass[:, i])[good]
                if particle in ["jet", "fatjet"]:
                    output[f"{particle}{i+1}_btag"] = ak.fill_none(collection.BTagPhys[:, i][good], np.nan)
                if particle == "fatjet":
                    for i in range(n):
                        output[f"fatjet{i+1}_tau21"] = fatjets_tau21[:, i][good]
                        output[f"fatjet{i+1}_tau32"] = fatjets_tau32[:, i][good]
                        output[f"fatjet{i+1}_tau43"] = fatjets_tau43[:, i][good]
        output.update(
            {
                # photons
                "diphoton_mass": scale(diphoton_mass)[good],
                "diphoton_pt": scale(diphoton_pt)[good],
                "diphoton_delta_R": diphoton_delta_r[good],
                # photon pt rel
                "photon1_pt_rel": photon1_pt_rel[good],
                "photon2_pt_rel": photon2_pt_rel[good],
                # jets
                "dijet_mass": scale(dijet_mass)[good],
                "dijet_delta_R": dijet_delta_r[good],
                "HT_30": scale(ht_30)[good],
                # met
                "met_pt": scale(met_pt)[good],
                "met_phi": met_phi[good],
                # misc features
                "n_photon": n_photons[good],
                "n_muon": n_muons[good],
                "n_electron": n_electrons[good],
                "has_lepton": has_lepton[good],
                "n_jet": n_jets[good],
                "n_fatjet": n_fatjets[good],
                "event_weight": event_weight[good],
                "event_number": event_number[good],
            }
        )
        return {
            "cutflow": {
                "total": ak.num(good, axis=0),
                "good": ak.sum(good),
            },
            "events": ak.zip(output),
        }
