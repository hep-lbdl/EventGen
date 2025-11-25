import os
import importlib
import subprocess
import shutil

import luigi
import law
import numpy as np
import pandas as pd
from coffea.nanoevents import DelphesSchema
from coffea.dataset_tools import (
    apply_to_fileset,
    preprocess,
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import dask
from dask.distributed import Client
from dask import delayed

from utils.numpy import NumpyEncoder
from utils.infrastructure import ClusterMixin, silentremove
from utils.physics import parse_mg_output, parse_pythia_output


class BaseTask(law.Task):
    """
    Base task which provides some convenience methods
    """

    version = law.Parameter(default="dev")

    def store_parts(self):
        task_name = self.__class__.__name__
        return (
            os.getenv("GEN_OUT"),
            f"version_{self.version}",
            task_name,
        )

    def local_path(self, *path):
        sp = self.store_parts()
        sp += path
        return os.path.join(*(str(p) for p in sp))

    def local_target(self, *path, **kwargs):
        return law.LocalFileTarget(self.local_path(*path), **kwargs)

    def local_directory_target(self, *path, **kwargs):
        return law.LocalDirectoryTarget(self.local_path(*path), **kwargs)


class ProcessMixin:
    process = law.Parameter(default="test")
    ecm = luigi.FloatParameter(default=13000.0)

    def store_parts(self):
        sp = super().store_parts()
        return sp + (self.process, f"ecm_{self.ecm:.2f}")

    @property
    def process_config_dir(self):
        return f"{os.getenv('GEN_CODE')}/config/processes/{self.process}"

    @property
    def common_model_dir(self):
        """
        load common UFO models
        """
        return f"{os.getenv('GEN_CODE')}/config/models"

    @property
    def common_param_dir(self):
        """
        load common param_card
        """
        return f"{os.getenv('GEN_CODE')}/config/params"

    @property
    def madgraph_config_file(self):
        return f"{self.process_config_dir}/madgraph.dat"

    @property
    def pythia_config_file(self):
        return f"{self.process_config_dir}/pythia.cmnd"

    @property
    def has_madgraph_config(self):
        return os.path.isfile(self.madgraph_config_file)


class DetectorMixin:
    detector = law.Parameter(default="ATLAS")

    def store_parts(self):
        sp = super().store_parts()
        return sp + (self.detector,)

    @property
    def detector_config_file(self):
        return f"delphes_card_{self.detector}.tcl"

    @property
    def detector_config(self):
        filename = self.detector_config_file
        user_config = f"{os.getenv('GEN_CODE')}/config/cards/{filename}"
        default_config = f"{os.getenv('DELPHES_DIR')}/cards/{filename}"
        if os.path.exists(user_config):
            return user_config
        elif os.path.exists(default_config):
            return default_config
        else:
            raise FileNotFoundError(f"Detector configuration  not found: {filename}")


class ProcessorMixin:
    processor = law.Parameter(default="test")

    def store_parts(self):
        sp = super().store_parts()
        return sp + (self.processor,)

    @property
    def processor_module(self):
        return importlib.import_module(f"processors.{self.processor}")

    @property
    def processor_class(self):
        return getattr(self.processor_module, "Processor")


class NEventsMixin:
    n_events = luigi.IntParameter(default=1000)


class MadgraphConfig(ProcessMixin, law.ExternalTask):
    def output(self):
        return law.LocalFileTarget(self.madgraph_config_file)


class PythiaConfig(ProcessMixin, law.ExternalTask):
    def output(self):
        return law.LocalFileTarget(self.pythia_config_file)


class ChunkedEventsTask(NEventsMixin):
    n_max = luigi.IntParameter(default=1000000)

    @property
    def brakets(self):
        n_events = int(self.n_events)
        n_max = int(self.n_max)
        starts = range(0, n_events, n_max)
        stops = list(starts)[1:] + [n_events]
        brakets = zip(starts, stops)
        return list(brakets)

    @property
    def n_brakets(self):
        return len(self.brakets)

    @property
    def identifiers(self):
        return list(f"{i}_with_{int(self.n_max)}" for i in range(self.n_brakets))


class Madgraph(
    ChunkedEventsTask,
    ProcessMixin,
    ClusterMixin,
    BaseTask,
):
    # Base random seed
    seed = 42

    # SLURM Configuration
    # Walltime is dynamic, see below
    cores = 1
    qos = "shared"
    walltime = "09:59:00"

    def requires(self):
        return MadgraphConfig.req(self)

    @property
    def executable(self):
        return f"{os.getenv('MADGRAPH_DIR')}/bin/mg5_aMC"

    @property
    def memory(self):
        if self.process in ["nonres_yy_jjj"]:
            return "48GB"
        else:
            return "24GB"

    def output(self):
        return {
            identifier: {
                "config": self.local_target(f"{identifier}/config.dat"),
                "madgraph_dir": self.local_directory_target(f"{identifier}/mg"),
                "events": self.local_target(f"{identifier}/mg/Events/run_01/unweighted_events.lhe.gz"),  # fmt: skip
                "out": self.local_target(f"{identifier}/out.txt"),
            }
            for identifier in self.identifiers
        }

    @staticmethod
    def fun(info):
        exe, config, out = info

        # Process
        cmd = [exe, "-f", config]
        with open(out, "w") as out_file:
            result = subprocess.call(cmd, stdout=out_file, stderr=out_file)

        return result

    def run(self):
        madgraph_config_base = self.input().load(formatter="text")
        # Set up the tasks to compute
        cmds = []

        for i, identifier, (start, stop) in zip(
            range(len(self.identifiers)),
            self.identifiers,
            self.brakets,
        ):
            config_target = self.output()[identifier]["config"]
            madgraph_target = self.output()[identifier]["madgraph_dir"]
            events_target = self.output()[identifier]["events"]
            out_target = self.output()[identifier]["out"]

            # In case the task already successfully finished an identifier
            if events_target.exists():
                continue

            n_events = stop - start
            madgraph_config = str(madgraph_config_base)
            madgraph_config = madgraph_config.replace("SEED_PLACEHOLDER", str(self.seed + i))
            madgraph_config = madgraph_config.replace("NEVENTS_PLACEHOLDER", str(int(n_events)))
            madgraph_config = madgraph_config.replace("EBEAM_PLACEHOLDER", str(self.ecm / 2))
            madgraph_config = madgraph_config.replace("OUTPUT_PLACEHOLDER", madgraph_target.path)
            madgraph_config = madgraph_config.replace("MODEL_PLACEHOLDER", self.common_model_dir)
            madgraph_config = madgraph_config.replace("PARAM_PLACEHOLDER", self.common_param_dir)
            config_target.dump(madgraph_config, formatter="text")
            out_target.parent.touch()

            cmds.append(
                [
                    self.executable,
                    config_target.path,
                    out_target.path,
                ]
            )

        # Connect to the cluster
        cluster = self.start_cluster(len(cmds))
        client = Client(cluster)

        # Use client.map to parallelize the tasks
        futures = client.map(self.fun, cmds)

        # Gather the results
        client.gather(futures)

        # Scale down and close the cluster
        cluster.scale(0)
        client.close()
        cluster.close()


class DelphesPythia8(
    DetectorMixin,
    ChunkedEventsTask,
    ProcessMixin,
    ClusterMixin,
    BaseTask,
):
    # SLURM Configuration
    cores = 1
    memory = "1GB"
    walltime = "24:00:00"
    qos = "shared"

    def output(self):
        return {
            identifier: {
                "config": self.local_target(f"{identifier}/config.txt"),
                "events": self.local_target(f"{identifier}/events.root"),
                "out": self.local_target(f"{identifier}/out.txt"),
            }
            for identifier in self.identifiers
        }

    def requires(self):
        if self.has_madgraph_config:
            return {
                "madgraph": Madgraph.req(self),
                "pythia_config": PythiaConfig.req(self),
            }
        else:
            return {"pythia_config": PythiaConfig.req(self)}

    @property
    def executable(self):
        if shutil.which("DelphesPythia8Filtered"):
            return "DelphesPythia8Filtered"
        else:
            raise Exception("Did you activate the 'eventgen' conda env?")

    @staticmethod
    def fun(info):
        exe, detector, process, events, out = info

        # Write events to tmp file
        tmp_events = events + ".tmp"

        # If tmp file exists from a previous run, just remove it
        silentremove(tmp_events)

        # Process
        cmd = [exe, detector, process, tmp_events]
        with open(out, "w") as out_file:
            result = subprocess.call(cmd, stdout=out_file, stderr=out_file)

        # Move events from tmp file to final dir
        shutil.move(tmp_events, events)

        return result

    @law.decorator.safe_output
    def run(self):
        detector_config = self.detector_config
        pythia_config = self.input()["pythia_config"].load(formatter="text")

        # Set up the tasks to compute
        cmds = []
        for identifier, (start, stop) in zip(self.identifiers, self.brakets):
            config_target = self.output()[identifier]["config"]
            events_target = self.output()[identifier]["events"]
            out_target = self.output()[identifier]["out"]
            # In case the task already successfully finished an identifier
            if events_target.exists():
                continue

            config_target.parent.touch()
            events_target.parent.touch()
            out_target.parent.touch()

            n_events = stop - start
            pythia_config = pythia_config.replace("NEVENTS_PLACEHOLDER", str(int(n_events)))
            pythia_config = pythia_config.replace("ECM_PLACEHOLDER", str(self.ecm))

            if self.has_madgraph_config:
                madgraph_events = self.input()["madgraph"][identifier]["events"].path
                pythia_config = pythia_config.replace("INPUT_PLACEHOLDER", madgraph_events)

            config_target.dump(pythia_config, formatter="text")

            cmds.append(
                [
                    self.executable,
                    detector_config,
                    config_target.path,
                    events_target.path,
                    out_target.path,
                ]
            )

        # Connect to the cluster
        cluster = self.start_cluster(len(cmds))
        client = Client(cluster)

        # Use client.map to parallelize the tasks
        futures = client.map(self.fun, cmds)

        # Gather the results
        results = client.gather(futures)

        # Scale down and close the cluster
        cluster.scale(0)
        client.close()
        cluster.close()


class SkimEvents(
    ProcessorMixin,
    DetectorMixin,
    ChunkedEventsTask,
    NEventsMixin,
    ProcessMixin,
    ClusterMixin,
    BaseTask,
):
    # SLURM Configuration
    cores = 8
    walltime = "01:00:00"
    qos = "shared"
    arch = "cpu"

    step_size = luigi.IntParameter(default=0)

    @property
    def memory(self):
        if self.process in ["nonres_yy_jjj"]:
            return "60GB"
        else:
            return "28GB"

    def requires(self):
        return DelphesPythia8.req(self)

    def output(self):
        return {
            "cutflow": self.local_target("cutflow.json"),
            "events": self.local_target("skimmed.h5"),
        }

    @staticmethod
    def get_single_job(req_dict):
        return list(req_dict.values())[0]

    @law.decorator.safe_output
    def run(self):
        inputs = self.input()
        # Get FSet
        fset = {"all": {"files": {inp["events"].path: "Delphes" for inp in inputs.values()}}}

        # Start Preprocessing
        if self.step_size > 0:
            dataset_runnable, _ = preprocess(fset, step_size=self.step_size)
        else:
            dataset_runnable, _ = preprocess(fset)

        # Apply to Fileset
        to_compute = apply_to_fileset(
            self.processor_class(),
            dataset_runnable,
            schemaclass=DelphesSchema,
        )

        # Compute Payload
        cluster = self.start_cluster(1)
        with Client(cluster) as client:
            (output,) = dask.compute(to_compute)

        # Scale down and close the cluster
        cluster.scale(0)
        client.close()
        cluster.close()

        output = output["all"]

        # Write cutflow to json
        cutflow = output["cutflow"]
        self.output()["cutflow"].dump(cutflow, cls=NumpyEncoder)

        # Create dataframe from events
        events = output["events"]
        df = pd.DataFrame(events.to_numpy().data)

        # add efficiencies
        eff = cutflow["good"] / cutflow["total"]
        df["selection_efficiency"] = eff

        # Write cross-section info for MadGraph
        if self.has_madgraph_config:
            one_mg_job = self.get_single_job(self.requires().input()["madgraph"])
            mg_output = one_mg_job["out"].load()
            mg_xsec, mg_xsec_unc = parse_mg_output(mg_output)
        else:
            mg_xsec, mg_xsec_unc = np.nan, np.nan
        df["mg_xsec [fb]"], df["mg_xsec_unc [fb]"] = mg_xsec, mg_xsec_unc

        # Write cross-section and decay filter info for Pythia
        one_pythia_job = self.get_single_job(self.input())
        pythia_out = one_pythia_job["out"].load()
        pythia_xsec, pythia_xsec_unc, pythia_filter_efficiency = parse_pythia_output(pythia_out)
        df["pythia_xsec [fb]"], df["pythia_xsec_unc [fb]"] = pythia_xsec, pythia_xsec_unc
        df["pythia_filter_efficiency"] = pythia_filter_efficiency

        # Write events to hdf5
        df.to_hdf(self.output()["events"].path, key="events")


class PlotEvents(SkimEvents):
    def requires(self):
        return SkimEvents.req(self)

    def output(self):
        return self.local_directory_target("plots.pdf")

    def run(self):
        df = pd.read_hdf(self.input()["events"].path, key="events")
        # Save plots
        self.output().parent.touch()
        with PdfPages(self.output().path) as pdf:
            for column in df.columns:
                values = df[column]

                # Clean values
                values.replace([np.inf, -np.inf], np.nan, inplace=True)
                if not values.notna().any():
                    continue
                if values.dtype == bool:
                    values = values.astype(int)

                # Determine number of bins
                bins = min(len(values.unique() * 2), 50)

                # Plot histogram
                plt.hist(values, bins=bins, alpha=0.7)

                # Decorate plot
                title = f"Process {self.process} @ {self.detector} using {self.processor} proc."
                plt.title(title)
                plt.xlabel(column)
                plt.ylabel("Entries")
                plt.tight_layout()
                pdf.savefig()
                plt.close()


class PlotEventsWrapper(BaseTask):
    def requires(self):
        config = dict(
            detector="ATLAS_fatjet_skimAll",
            processor="yy",
            ecm=13000.0,
        )
        ret = {}
        ret.update(
            {"nonres_yy_jjj": PlotEvents.req(self, process="nonres_yy_jjj", n_events=1e8, **config)}
        )
        ret.update(
            {
                process: PlotEvents.req(self, process=process, n_events=4e6, **config)
                for process in [
                    "ggh_yy",
                    "ttH_yy",
                    "vbf_yy",
                    "vh_yy",
                    "WN_HyyN_150",
                    "WN_HyyN_200",
                    "WN_HyyN_300",
                    "WN_HyyN_600",
                    "XSH_500_100",
                    "XSH_750_100_ll",
                    "TT_tZNtHyyN",
                    "ZpHyyA_200",
                    "ZpHyyA_500",
                    "thFCNC_ctHyy_tcphi",
                    "thFCNC_utHyy_tphi",
                    "ttFCNC_tcHyy_tcphi",
                    "ttFCNC_tuHyy_tphi",
                    "Hl_Hyyl_150",
                    "Hl_Hyyl_300",
                    "Hl_Hyyl_450",
                    "WlZvHv_Hyyl_200",
                    "WlZvHv_Hyyl_400",
                    "WlZvHv_Hyyl_600",
                    "BB_bHNbHyyN_500_180_50",
                    "BB_bHNbHyyN_1000_205_60",
                    "BB_bHNbHyyN_1200_205_60",
                ]
            }
        )
        ret.update(
            {
                process: PlotEvents.req(self, process=process, n_events=2e7, **config)
                for process in [
                    "ZpHyyA_500",
                ]
            }
        )
        return ret

    def output(self):
        return self.local_directory_target("summary.json")

    def run(self):
        summary = {
            process: req.input()["cutflow"].load()["good"]
            for process, req in self.requires().items()
        }
        self.output().dump(summary)


class PlotEventsContrastiveWrapper(BaseTask, law.WrapperTask):
    def requires(self):
        config = dict(
            processor="contrastive",
            ecm=13000.0,
        )
        req = []
        req += [
            PlotEvents.req(
                self,
                process="nonres_yy_jjj",
                n_events=3e7,
                detector="ATLAS_fatjet",
                **config,
            )
        ]
        req += [
            PlotEvents.req(
                self,
                process=process,
                n_events=2e6,
                detector="ATLAS_fatjet",
                **config,
            )
            for process in [
                "ggh_yy",
                "ttH_yy",
                "vbf_yy",
                "vh_yy",
                "WN_HyyN_150",
                "WN_HyyN_200",
                "WN_HyyN_300",
                "WN_HyyN_600",
                "XSH_500_100",
                "XSH_750_100_ll",
                "TT_tZNtHyyN",
                "ZpHyyA_200",
                "ZpHyyA_500",
                "thFCNC_ctHyy_tcphi",
                "thFCNC_utHyy_tphi",
                "ttFCNC_tcHyy_tcphi",
                "ttFCNC_tuHyy_tphi",
            ]
        ]
        req += [
            PlotEvents.req(
                self,
                process=process,
                n_events=2e6,
                n_max=1e5,
                detector="ATLAS_fatjet_skimAll",
                **config,
            )
            for process in [
                "Hl_Hyyl_150",
                "Hl_Hyyl_300",
                "Hl_Hyyl_450",
                "WlZvHv_Hyyl_200",
                "WlZvHv_Hyyl_400",
                "WlZvHv_Hyyl_600",
                "BB_bHNbHyyN_500_180_50",
                "BB_bHNbHyyN_1000_205_60",
                "BB_bHNbHyyN_1200_205_60",
            ]
        ]
        return req
