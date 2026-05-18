import numpy as np
import awkward as ak
import coffea.processor as processor
from coffea.nanoevents.methods import candidate


class Processor(processor.ProcessorABC):
    def postprocess(self, accumulator):
        pass

    def process(self, events):
        pad = lambda x, target: ak.pad_none(x, target=target, clip=True)
        jets = pad(events.Jet, 2)
        good = jets[:, 0].pt > 50
        good = ak.fill_none(good, False)
        event_weight = events.Event.Weight

        return {
            "cutflow": {
                "n_total": ak.num(good, axis=0),
                "n_good": ak.sum(good),
                "sumw_presel": ak.sum(event_weight),
                "sumw_postsel": ak.sum(event_weight[good]),
            },
            "events": ak.zip(
                {
                    "jet1_pt": jets[:, 0].pt[good],
                    "jet2_pt": jets[:, 1].pt[good],
                }
            ),
        }
