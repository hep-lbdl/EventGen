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
from utils.physics import parse_mg_output, parse_pythia_output, pythia_xsec_modulation


# Heavy processes (multi-leg matching, large SUSY decays) need extra
# walltime/memory for both direct generation and gridpack warmup.
_MADGRAPH_LONG_PROCESSES = {
    "WlZvHv_Hyyl_600",
    "TT_tZNtHyyN",
    "WN_HyyN_150",
    "WN_HyyN_200",
    "WN_HyyN_300",
    "WN_HyyN_600",
    "BB_bHNbHyyN_1000_205_60",
    "BB_bHNbHyyN_1200_205_60",
}
_MADGRAPH_MEDIUM_PROCESSES = {"nonres_llyy_jj"}
_MADGRAPH_HIGH_MEM_PROCESSES = _MADGRAPH_LONG_PROCESSES | {"nonres_llyy_jj"}
_MADGRAPH_MEDIUM_MEM_PROCESSES = {"nonres_yy_jjj"}


def _madgraph_walltime(process):
    if process in _MADGRAPH_LONG_PROCESSES:
        return "47:59:00"
    if process in _MADGRAPH_MEDIUM_PROCESSES:
        return "23:59:00"
    return "09:59:00"


def _madgraph_memory(process):
    if process in _MADGRAPH_HIGH_MEM_PROCESSES:
        return "128GB"
    if process in _MADGRAPH_MEDIUM_MEM_PROCESSES:
        return "48GB"
    return "24GB"


def _render_madgraph_config(
    template,
    *,
    n_events,
    seed,
    ebeam,
    output_dir,
    common_model_dir,
    common_param_dir,
    gridpack=False,
):
    cfg = str(template)
    cfg = cfg.replace("SEED_PLACEHOLDER", str(int(seed)))
    cfg = cfg.replace("NEVENTS_PLACEHOLDER", str(int(n_events)))
    cfg = cfg.replace("EBEAM_PLACEHOLDER", str(ebeam))
    cfg = cfg.replace("OUTPUT_PLACEHOLDER", output_dir)
    cfg = cfg.replace("MODEL_PLACEHOLDER", common_model_dir)
    cfg = cfg.replace("PARAM_PLACEHOLDER", common_param_dir)
    if gridpack:
        cfg = cfg.rstrip() + "\nset gridpack True\n"
    return cfg


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


class MadgraphGridpack(ProcessMixin, ClusterMixin, BaseTask):
    """
    Build a Madgraph gridpack once per (process, ecm). The resulting tarball
    bakes the matrix-element compilation, integration grid, and param_card so
    that downstream per-chunk event generation skips those steps and only
    runs the (fast) gridrun phase.
    """

    seed = 42
    # Warmup nevents; MG drives grid precision via its own
    # accuracy/points/iterations defaults, so this value is largely cosmetic.
    n_warmup_events = 1000

    cores = 16
    qos = "shared"

    @property
    def walltime(self):
        return _madgraph_walltime(self.process)

    @property
    def memory(self):
        return _madgraph_memory(self.process)

    def requires(self):
        return MadgraphConfig.req(self)

    @property
    def executable(self):
        return f"{os.getenv('MADGRAPH_DIR')}/bin/mg5_aMC"

    def output(self):
        return {
            "config": self.local_target("config.dat"),
            "madgraph_dir": self.local_directory_target("mg"),
            "gridpack": self.local_target("mg/run_01_gridpack.tar.gz"),
            "out": self.local_target("out.txt"),
        }

    @staticmethod
    def fun(info):
        exe, config, out = info
        cmd = [exe, "-f", config]
        with open(out, "w") as out_file:
            return subprocess.call(cmd, stdout=out_file, stderr=out_file)

    def run(self):
        if self.output()["gridpack"].exists():
            return

        config_target = self.output()["config"]
        madgraph_target = self.output()["madgraph_dir"]
        out_target = self.output()["out"]

        # Stale dir would cause mg5_aMC's `output` to prompt y/n and desync.
        if os.path.exists(madgraph_target.path):
            shutil.rmtree(madgraph_target.path)

        rendered = _render_madgraph_config(
            self.input().load(formatter="text"),
            n_events=self.n_warmup_events,
            seed=self.seed,
            ebeam=self.ecm / 2,
            output_dir=madgraph_target.path,
            common_model_dir=self.common_model_dir,
            common_param_dir=self.common_param_dir,
            gridpack=True,
        )
        config_target.dump(rendered, formatter="text")
        out_target.parent.touch()

        cluster = self.start_cluster(1)
        with cluster, Client(cluster) as client:
            futures = client.map(
                self.fun,
                [[self.executable, config_target.path, out_target.path]],
            )
            client.gather(futures)


class Madgraph(
    ChunkedEventsTask,
    ProcessMixin,
    ClusterMixin,
    BaseTask,
):
    # SLURM Configuration
    walltime = "24:00:00"

    # Base random seed
    seed = 42
    use_gridpack = luigi.BoolParameter(default=False)

    def requires(self):
        reqs = {"config": MadgraphConfig.req(self)}
        if self.use_gridpack:
            reqs["gridpack"] = MadgraphGridpack.req(self)
        return reqs

    @property
    def executable(self):
        return f"{os.getenv('MADGRAPH_DIR')}/bin/mg5_aMC"

    @property
    def walltime(self):
        return _madgraph_walltime(self.process)

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

    @staticmethod
    def gridpack_fun(info):
        exe_dir, n_events, seed, gridpack_tar, events_path, out = info

        # Fresh extraction per chunk: workers run concurrently and would
        # otherwise collide on the GridRun_<seed>/ output dir.
        if os.path.exists(exe_dir):
            shutil.rmtree(exe_dir)
        os.makedirs(exe_dir)

        with open(out, "w") as out_file:
            rc = subprocess.call(
                ["tar", "xzf", gridpack_tar, "-C", exe_dir],
                stdout=out_file,
                stderr=out_file,
            )
            if rc != 0:
                return rc
            rc = subprocess.call(
                ["./run.sh", str(int(n_events)), str(int(seed))],
                cwd=exe_dir,
                stdout=out_file,
                stderr=out_file,
            )
            if rc != 0:
                return rc

        # run.sh drops events.lhe.gz at the extraction root; re-home it
        # under Events/run_01/ so the output path matches non-gridpack mode.
        final_dir = os.path.dirname(events_path)
        os.makedirs(final_dir, exist_ok=True)
        shutil.move(os.path.join(exe_dir, "events.lhe.gz"), events_path)
        return 0

    def run(self):
        if self.use_gridpack:
            self._run_gridpack()
        else:
            self._run_direct()

    def _run_direct(self):
        madgraph_config_base = self.input()["config"].load(formatter="text")
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

            # Remove a stale process dir from a previous failed attempt,
            # otherwise mg5_aMC's `output` prompts y/n and desyncs the script.
            if os.path.exists(madgraph_target.path):
                shutil.rmtree(madgraph_target.path)

            n_events = stop - start
            madgraph_config = _render_madgraph_config(
                madgraph_config_base,
                n_events=n_events,
                seed=self.seed + i,
                ebeam=self.ecm / 2,
                output_dir=madgraph_target.path,
                common_model_dir=self.common_model_dir,
                common_param_dir=self.common_param_dir,
            )
            config_target.dump(madgraph_config, formatter="text")
            out_target.parent.touch()

            cmds.append(
                [
                    self.executable,
                    config_target.path,
                    out_target.path,
                ]
            )

        # Connect to the cluster and run the tasks
        cluster = self.start_cluster(len(cmds))
        with cluster, Client(cluster) as client:
            futures = client.map(self.fun, cmds)
            client.gather(futures)

    def _run_gridpack(self):
        gridpack_tar = self.input()["gridpack"]["gridpack"].path
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

            if events_target.exists():
                continue

            n_events = stop - start
            seed = self.seed + i

            # Record the invocation so the chunk's `config` output exists
            # (law uses it for completeness) and so the run is reproducible.
            config_target.dump(
                f"# gridpack invocation\n"
                f"gridpack: {gridpack_tar}\n"
                f"nevents:  {int(n_events)}\n"
                f"seed:     {int(seed)}\n"
                f"command:  ./run.sh {int(n_events)} {int(seed)}\n",
                formatter="text",
            )
            out_target.parent.touch()

            cmds.append(
                [
                    madgraph_target.path,
                    n_events,
                    seed,
                    gridpack_tar,
                    events_target.path,
                    out_target.path,
                ]
            )

        cluster = self.start_cluster(len(cmds))
        with cluster, Client(cluster) as client:
            futures = client.map(self.gridpack_fun, cmds)
            client.gather(futures)


class DelphesPythia8(
    DetectorMixin,
    ChunkedEventsTask,
    ProcessMixin,
    ClusterMixin,
    BaseTask,
):
    # SLURM Configuration
    memory = "2GB"
    walltime = "24:00:00"

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
        detector_config_base = self.detector_config
        pythia_config_base = self.input()["pythia_config"].load(formatter="text")

        # Set up the tasks to compute
        cmds = []
        for identifier, (start, stop) in zip(self.identifiers, self.brakets):
            detector_config = str(detector_config_base)
            pythia_config = str(pythia_config_base)

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

        # Connect to the cluster and run the tasks
        cluster = self.start_cluster(len(cmds))
        with cluster, Client(cluster) as client:
            futures = client.map(self.fun, cmds)
            client.gather(futures)


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
        with cluster, Client(cluster) as client:
            (output,) = dask.compute(to_compute)

        output = output["all"]

        # Write cutflow to json
        cutflow = output["cutflow"]
        self.output()["cutflow"].dump(cutflow, cls=NumpyEncoder)

        # Create dataframe from events
        events = output["events"]
        df = pd.DataFrame(events.to_numpy().data)

        # add weight sum pre-selection
        df["sumw_presel"] = cutflow["sumw_presel"]
        df["sumw_postsel"] = cutflow["sumw_postsel"]

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
        if self.has_madgraph_config:
            # Apply modulation factors (pythia PDG filter not reflected in xsec)
            pythia_config = one_pythia_job["config"].load()
            modulation = pythia_xsec_modulation(pythia_config)
            pythia_xsec *= modulation
            pythia_xsec_unc *= modulation

        df["pythia_xsec [fb]"], df["pythia_xsec_unc [fb]"] = pythia_xsec, pythia_xsec_unc
        df["pythia_filter_efficiency"] = pythia_filter_efficiency

        # Write events to hdf5
        df.to_hdf(
            self.output()["events"].path,
            key="events",
            format="table",
        )


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
                finite = values.dropna()
                if finite.min() == finite.max():
                    bins = 1
                else:
                    bins = min(len(finite.unique()) * 2, 50)

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


class PlotEventsWrapper(ProcessorMixin, BaseTask):
    def requires(self):
        config = dict(
            detector="ATLAS_fatjet_skimAll",
            ecm=13000.0,
        )
        ret = {}
        ret.update(
            {
                process: PlotEvents.req(self, process=process, n_events=2e7, **config)
                for process in [
                    "nonres_yy_jjj",
                    "nonres_llyy_jj",
                    "nonres_ttyy",
                    "ZpHyyA_300",
                    "ZpHyyA_400",
                    "ZpHyyA_500",
                ]
            }
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
                    "HH",
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
        return ret

    def output(self):
        return self.local_directory_target("summary.json")

    def run(self):
        summary = {}
        for process, req in self.requires().items():
            events = pd.read_hdf(req.input()["events"].path, stop=1)
            event_summary = {
                identifier: float(events[identifier].iloc[0])
                for identifier in [
                    "mg_xsec [fb]",
                    "pythia_xsec [fb]",
                    "pythia_filter_efficiency",
                ]
            }
            cutflow = req.input()["cutflow"].load()
            summary[process] = {
                "good": cutflow["n_good"],
                "total": cutflow["n_total"],
            }
            summary[process].update(event_summary)
        self.output().dump(summary)
