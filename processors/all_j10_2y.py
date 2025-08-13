import numpy as np
import awkward as ak
import coffea.processor as processor
from coffea.nanoevents.methods import candidate
# adding truth filter for Hyy
# logic: count photon from  daughters of PID=25 

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

        jets = pad(events.Jet, 2)
        jets = ak.zip(
            {
                "pt": jets.pt,
                "eta": jets.eta,
                "phi": jets.phi,
                "mass": jets.mass,
                "charge": ak.zeros_like(jets.pt),
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        dijet_mass = (jets[:, 0] + jets[:, 1]).m
        dijet_delta_r = jets[:, 0].delta_r(jets[:, 1])
        ht_30 = ak.sum(events.Jet.pt[events.Jet.pt > 30], axis=-1)

        has_lepton = (ak.num(events.Muon) > 0) | (ak.num(events.Electron) > 0)

        met_pt = events.MissingET.MET

        # misc event features
        n_photons = ak.num(events.Photon)
        n_jets = ak.num(events.Jet)
        event_weight = events.Event.Weight
        event_number = events.Event.Number

        # event selection
        cut_j10 = ak.fill_none(jets[:, 0].pt > 10, False)
        part_higgs=events.Particle[events.Particle.PID==25]
        n_y = ak.sum(events.Particle.PID[part_higgs.D1]==22,axis=-1) + ak.sum(events.Particle.PID[part_higgs.D2]==22,axis=-1) 
        cut_2y = ak.fill_none(n_y >= 2, False)
        good = cut_j10 & cut_2y

        return {
            "cutflow": {
                "total": ak.num(good, axis=0),
                "good": ak.sum(good),
            },
            "events": ak.zip(
                {
                    # photons
                    "diphoton_mass": diphoton_mass[good],
                    "diphoton_pt": diphoton_pt[good],
                    "diphoton_delta_R": diphoton_delta_r[good],
                    # photon 1
                    "photon1_pt": hgamma.pt[:, 0][good],
                    "photon1_eta": hgamma.eta[:, 0][good],
                    "photon1_phi": hgamma.phi[:, 0][good],
                    "photon1_m": hgamma.m[:, 0][good],
                    "photon1_pt_rel": photon1_pt_rel[good],
                    # photon 1
                    "photon2_pt": hgamma.pt[:, 1][good],
                    "photon2_eta": hgamma.eta[:, 1][good],
                    "photon2_phi": hgamma.phi[:, 1][good],
                    "photon2_m": hgamma.m[:, 1][good],
                    "photon2_pt_rel": photon2_pt_rel[good],
                    # jets
                    "dijet_mass": dijet_mass[good],
                    "dijet_delta_R": dijet_delta_r[good],
                    "HT_30": ht_30[good],
                    # jet 1
                    "jet1_pt": jets.pt[:, 0][good],
                    "jet1_eta": jets.eta[:, 0][good],
                    "jet1_phi": jets.phi[:, 0][good],
                    "jet1_m": jets.m[:, 0][good],
                    # jet 2
                    "jet2_pt": jets.pt[:, 1][good],
                    "jet2_eta": jets.eta[:, 1][good],
                    "jet2_phi": jets.phi[:, 1][good],
                    "jet2_m": jets.m[:, 1][good],
                    # leptons
                    "has_lepton": has_lepton[good],
                    # met
                    "met_pt": met_pt[good],
                    # misc features
                    "n_photon": n_photons[good],
                    "n_jet": n_jets[good],
                    "event_weight": event_weight[good],
                    "event_number": event_number[good],
                }
            ),
        }
