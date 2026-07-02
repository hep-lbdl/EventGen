import io
import os
import importlib
import stat
import subprocess
import shutil
import tarfile
import tempfile
import time
import uuid

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
from dask.distributed import Client, wait
from dask import delayed
import awkward as ak

from utils.numpy import NumpyEncoder
from utils.infrastructure import ClusterMixin, silentremove
from utils.physics import parse_mg_output, parse_pythia_output, pythia_xsec_modulation

# Inline-decay SUSY cascade processes: need extra memory during event generation.
_MADGRAPH_EXTRA_PROCESSES = {
    "BB_bZNbHyyN_500_180_50",
    "BB_bZNbHyyN_1000_205_60",
    "BB_bZNbHyyN_1200_205_60",
    "CC_cZNcHyyN_500_180_50",
    "CC_cZNcHyyN_1000_205_60",
    "CC_cZNcHyyN_1200_205_60",
    "TT_tZNtHyyN_500_180_50",
    "TT_tZNtHyyN_1000_205_60",
    "TT_tZNtHyyN_1200_205_60",
}

# walltime/memory for both direct generation and gridpack warmup.
_GRIDPACK_EXTRA_PROCESSES = _MADGRAPH_EXTRA_PROCESSES | {
    "WlZvHv_Hyyl_600",
    "TT_tZNtHyyN",
    "WN_HyyN_150",
    "WN_HyyN_200",
    "WN_HyyN_300",
    "WN_HyyN_600",
    "BB_bHNbHyyN_1000_205_60",
    "BB_bHNbHyyN_1200_205_60",
    "nonres_llyy_jj",
    "nonres_yy_jjj",
    "nonres_yy_j_nlo",
    "nonres_llyy_j_nlo",
}

# NLO processes. Gridpack behavior is different to LO.
_NLO_PROCESSES = {
    "nonres_yy_j_nlo",
    "nonres_llyy_j_nlo",
}


def _madgraph_walltime(process):
    if process in _GRIDPACK_EXTRA_PROCESSES:
        return "23:59:00"
    else:
        return "09:59:00"


def _madgraph_memory(process):
    if process in _GRIDPACK_EXTRA_PROCESSES:
        return "128GB"
    else:
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
    nb_core=None,
):
    cfg = str(template)
    cfg = cfg.replace("SEED_PLACEHOLDER", str(int(seed)))
    cfg = cfg.replace("NEVENTS_PLACEHOLDER", str(int(n_events)))
    cfg = cfg.replace("EBEAM_PLACEHOLDER", str(ebeam))
    cfg = cfg.replace("OUTPUT_PLACEHOLDER", output_dir)
    cfg = cfg.replace("MODEL_PLACEHOLDER", common_model_dir)
    cfg = cfg.replace("PARAM_PLACEHOLDER", common_param_dir)
    if nb_core is not None and int(nb_core) > 1:
        # run_mode/nb_core are mg5 toplevel settings, so they must be set
        # before `launch` switches the prompt into the madevent context.
        cfg = cfg.replace(
            "\nlaunch\n",
            f"\nset run_mode 2\nset nb_core {int(nb_core)}\nlaunch\n",
            1,
        )
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


_NLO_GRIDPACK_RUN_SH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "utils", "nlo_gridpack_run.sh"
)


def _create_nlo_gridpack(process_dir, gridpack_path):
    """
    Pack the compiled NLO process directory into run_01_gridpack.tar.gz.

    The NLO best practice (see aMC@NLO docs) is:
      1. Run generate_events once to set the integration grids (warmup).
      2. Tar the process directory.
      3. On each cluster node: extract, update the seed via SubProcesses/randinit,
         run ./bin/generate_events --only-generation --nocompile.

    Each chunk extracts to its own TMPDIR, so runs are fully independent.
    Events/ from the warmup run is excluded to keep the tarball small.
    """
    with open(_NLO_GRIDPACK_RUN_SH) as f:
        run_sh = f.read()

    os.makedirs(os.path.dirname(gridpack_path), exist_ok=True)
    abs_process_dir = os.path.abspath(process_dir)
    abs_gridpack = os.path.abspath(gridpack_path)

    # Write to a temp file first so the walk doesn't find the partial tarball
    tmp_path = abs_gridpack + ".tmp"
    with tarfile.open(tmp_path, "w:gz") as tf:
        # Add run.sh at tarball root
        info = tarfile.TarInfo(name="run.sh")
        encoded = run_sh.encode()
        info.size = len(encoded)
        info.mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        tf.addfile(info, io.BytesIO(encoded))

        # Pack the process directory, excluding Events/ (warmup output not needed)
        for root, dirs, files in os.walk(abs_process_dir):
            rel_root = os.path.relpath(root, abs_process_dir)
            # Skip Events/ to keep the tarball small
            dirs[:] = [d for d in dirs if not (rel_root == "." and d == "Events")]
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.join(rel_root, fname)
                tf.add(fpath, arcname=arcname)

    os.replace(tmp_path, abs_gridpack)


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

    cores = 32
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
            "gridpack": self.local_target("mg/run_01_gridpack.tar.gz"),
            "out": self.local_target("out.txt"),
        }

    @property
    def is_nlo(self):
        return self.process in _NLO_PROCESSES

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
        out_target = self.output()["out"]
        madgraph_dir = self.local_path("mg")

        # Stale dir would cause mg5_aMC's `output` to prompt y/n and desync.
        if os.path.exists(madgraph_dir):
            shutil.rmtree(madgraph_dir)

        rendered = _render_madgraph_config(
            self.input().load(formatter="text"),
            n_events=self.n_warmup_events,
            seed=self.seed,
            ebeam=self.ecm / 2,
            output_dir=madgraph_dir,
            common_model_dir=self.common_model_dir,
            common_param_dir=self.common_param_dir,
            nb_core=self.cores,
        )
        if not self.is_nlo:
            # LO gridpacks are built directly by mg5_aMC via this directive.
            rendered = rendered.rstrip() + "\nset gridpack True\n"
        config_target.dump(rendered, formatter="text")
        out_target.parent.touch()

        cluster = self.start_cluster(1)
        with cluster, Client(cluster) as client:
            futures = client.map(
                self.fun,
                [[self.executable, config_target.path, out_target.path]],
            )
            wait(futures)

        if self.is_nlo:
            # NLO (aMC@NLO) doesn't support the LO gridpack tarball: generate
            # warmup events normally, then wrap the compiled process
            # directory with a synthetic run.sh gridpack after the fact.
            _create_nlo_gridpack(madgraph_dir, self.output()["gridpack"].path)


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

    def requires(self):
        return MadgraphGridpack.req(self)

    @property
    def walltime(self):
        return _madgraph_walltime(self.process)

    @property
    def cores(self):
        return 10 if self.process in _NLO_PROCESSES else 1

    @property
    def memory(self):
        if self.process in _NLO_PROCESSES:
            return "20GB"
        if self.process in _MADGRAPH_EXTRA_PROCESSES:
            return "4GB"
        else:
            return "2GB"

    def output(self):
        return {
            identifier: {
                "config": self.local_target(f"{identifier}/config.dat"),
                "events": self.local_target(f"{identifier}/events.lhe.gz"),
                "out": self.local_target(f"{identifier}/out.txt"),
            }
            for identifier in self.identifiers
        }

    @staticmethod
    def fun(info):
        (
            n_events,
            seed,
            gridpack_tar,
            events_path,
            out,
            config_path,
            madspin_exe,
            madspin_card_tmpl,
        ) = info

        # Stub written from the dask worker so the per-chunk creates don't
        # all serialize through the login-node MDS.
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            f.write(
                f"# gridpack invocation\n"
                f"gridpack: {gridpack_tar}\n"
                f"nevents:  {n_events}\n"
                f"seed:     {seed}\n"
                f"command:  ./run.sh {n_events} {seed}\n"
            )

        os.makedirs(os.path.dirname(out), exist_ok=True)

        # Extract + run on the compute node's local storage ($TMPDIR),
        # then move to output path. Prevents leaving 1000s of MG
        # files on pscratch.
        scratch_root = os.environ.get("TMPDIR") or "/tmp"
        with tempfile.TemporaryDirectory(prefix="mg5_", dir=scratch_root) as exe_dir:
            with open(out, "w") as out_file:
                rc = subprocess.call(
                    ["tar", "xzf", gridpack_tar, "-C", exe_dir],
                    stdout=out_file,
                    stderr=out_file,
                )
                if rc != 0:
                    return rc
                rc = subprocess.call(
                    ["./run.sh", str(n_events), str(seed)],
                    cwd=exe_dir,
                    stdout=out_file,
                    stderr=out_file,
                )
                if rc != 0:
                    return rc

                events_src = os.path.join(exe_dir, "events.lhe.gz")

                # Optional MadSpin decay: when the process ships a
                # madspin_card, decay the (production-only, MLM-merged) LHE
                # here, before showering. Done post-merge so the cascade
                # decay products never pollute the jet matching.
                if madspin_card_tmpl is not None:
                    if not os.path.isfile(events_src):
                        return 1
                    card = madspin_card_tmpl.replace("SEED_PLACEHOLDER", str(seed))
                    card = card.replace("EVENTS_PLACEHOLDER", events_src)
                    card_path = os.path.join(exe_dir, "madspin_card.dat")
                    with open(card_path, "w") as card_file:
                        card_file.write(card)
                    rc = subprocess.call(
                        [madspin_exe, card_path],
                        cwd=exe_dir,
                        stdout=out_file,
                        stderr=out_file,
                    )
                    if rc != 0:
                        return rc
                    # MadSpin writes <input>_decayed.lhe.gz next to the input.
                    events_src = os.path.join(exe_dir, "events_decayed.lhe.gz")

            # run.sh (or MadSpin) drops the LHE at the extraction root; move it
            # onto the shared FS at the task's flat events path.
            if not os.path.isfile(events_src):
                # run.sh/MadSpin exited 0 but produced no LHE
                # (e.g. refinement non-convergence). Treat as a failed chunk.
                return 1
            os.makedirs(os.path.dirname(events_path), exist_ok=True)
            shutil.move(events_src, events_path)
        # exe_dir is auto-cleaned by TemporaryDirectory (including on raises).
        return 0

    def run(self):
        gridpack_tar = self.input()["gridpack"].path
        cmds = []

        # Opt-in MadSpin: a process decays via MadSpin iff it ships a
        # madspin_card.dat; otherwise events are left as MG5 produced them.
        madspin_card_path = f"{self.process_config_dir}/madspin_card.dat"
        if os.path.isfile(madspin_card_path):
            with open(madspin_card_path) as f:
                madspin_card_tmpl = f.read()
            madspin_exe = f"{os.getenv('MADGRAPH_DIR')}/MadSpin/madspin"
        else:
            madspin_card_tmpl = None
            madspin_exe = None

        outputs = self.output()
        for i, identifier, (start, stop) in zip(
            range(len(self.identifiers)),
            self.identifiers,
            self.brakets,
        ):
            config_target = outputs[identifier]["config"]
            events_target = outputs[identifier]["events"]
            out_target = outputs[identifier]["out"]

            if events_target.exists():
                continue

            cmds.append(
                [
                    int(stop - start),
                    int(self.seed + i),
                    gridpack_tar,
                    events_target.path,
                    out_target.path,
                    config_target.path,
                    madspin_exe,
                    madspin_card_tmpl,
                ]
            )

        cluster = self.start_cluster(len(cmds))
        with cluster, Client(cluster) as client:
            futures = client.map(self.fun, cmds)
            wait(futures)


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

    # Base random seed
    seed = 42

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

        outputs = self.output()
        inputs = self.input()
        madgraph_inputs = inputs["madgraph"] if self.has_madgraph_config else None

        for i, identifier, (start, stop) in zip(
            range(len(self.identifiers)),
            self.identifiers,
            self.brakets,
        ):
            detector_config = str(detector_config_base)
            pythia_config = str(pythia_config_base)

            config_target = outputs[identifier]["config"]
            events_target = outputs[identifier]["events"]
            out_target = outputs[identifier]["out"]
            # In case the task already successfully finished an identifier
            if events_target.exists():
                continue

            config_target.parent.touch()
            events_target.parent.touch()
            out_target.parent.touch()

            n_events = stop - start
            pythia_config = pythia_config.replace("NEVENTS_PLACEHOLDER", str(int(n_events)))
            pythia_config = pythia_config.replace("ECM_PLACEHOLDER", str(self.ecm))
            pythia_config = pythia_config.replace("SEED_PLACEHOLDER", str(self.seed + i))

            if madgraph_inputs is not None:
                madgraph_events = madgraph_inputs[identifier]["events"].path
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
            wait(futures)


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
    cluster_mode = "local"
    cores = 1
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

    @staticmethod
    def get_complete_pythia_job(req_dict):
        for job in req_dict.values():
            if "PYTHIA Event and Cross Section Statistics" in job["out"].load():
                return job
        raise RuntimeError("No Pythia job has a complete cross-section output block")

    @law.decorator.safe_output
    def run(self):
        inputs = self.input()
        # Get FSet
        fset = {"all": {"files": {inp["events"].path: "Delphes" for inp in inputs.values()}}}

        # Start Preprocessing
        step_size = self.step_size if self.step_size > 0 else None
        dataset_runnable, _ = preprocess(fset, step_size=step_size)

        # Apply to Fileset
        to_compute = apply_to_fileset(
            self.processor_class(),
            dataset_runnable,
            schemaclass=DelphesSchema,
        )

        # Compute Payload
        print("Starting Dask cluster and computing...")
        cluster = self.start_cluster(128)
        with cluster, Client(cluster) as client:
            (output,) = dask.compute(to_compute)

        output = output["all"]

        # Write cutflow to json
        cutflow = output["cutflow"]
        self.output()["cutflow"].dump(cutflow, cls=NumpyEncoder)

        # Create dataframe from events
        print("Creating dataframe from events...")
        events = output["events"]
        df = ak.to_dataframe(events)
        # option<bool> awkward columns land as mixed object dtype in pandas; cast them.
        for col in df.select_dtypes("object").columns:
            if pd.api.types.infer_dtype(df[col], skipna=True) == "boolean":
                df[col] = df[col].fillna(False).astype(bool)

        # add weight sum pre-selection
        df["sumw_presel"] = cutflow["sumw_presel"]
        df["sumw_postsel"] = cutflow["sumw_postsel"]

        print("Parsing cross-section info...")
        # Get MG cross-section (computed at MadgraphGridpack build time);
        if self.has_madgraph_config:
            gridpack_task = self.requires().requires()["madgraph"].requires()
            mg_xsec, mg_xsec_unc = parse_mg_output(gridpack_task.output()["out"].load())
        else:
            mg_xsec, mg_xsec_unc = np.nan, np.nan
        df["mg_xsec [fb]"], df["mg_xsec_unc [fb]"] = mg_xsec, mg_xsec_unc

        # Write cross-section and decay filter info for Pythia
        one_pythia_job = self.get_complete_pythia_job(self.input())
        pythia_xsec, pythia_xsec_unc, pythia_filter_efficiency = parse_pythia_output(one_pythia_job["out"].load())
        if self.has_madgraph_config:
            modulation = pythia_xsec_modulation(one_pythia_job["config"].load())
            pythia_xsec *= modulation
            pythia_xsec_unc *= modulation

        df["pythia_xsec [fb]"], df["pythia_xsec_unc [fb]"] = pythia_xsec, pythia_xsec_unc
        df["pythia_filter_efficiency"] = pythia_filter_efficiency
        
        # Fix pandas dataframe memory layout
        df = df.copy()

        print("Writing events to hdf5...")
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
                    # Single-value column: give np.histogram a non-zero range.
                    v = float(finite.iloc[0])
                    bins = [v - 0.5, v + 0.5]
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
                process: PlotEvents.req(self, process=process, n_events=2e8, n_max=1e5, **config)
                for process in [
                    "nonres_yy_jjj",
                ]
            }
        )
        ret.update(
            {
                process: PlotEvents.req(self, process=process, n_events=2e7, **config)
                for process in [
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
                    "XHH_280",
                    "XHH_500",
                    "XHH_1000",
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
                    "BB_bZNbHyyN_500_180_50",
                    "BB_bZNbHyyN_1000_205_60",
                    "BB_bZNbHyyN_1200_205_60",
                    "CC_cZNcHyyN_500_180_50",
                    "CC_cZNcHyyN_1000_205_60",
                    "CC_cZNcHyyN_1200_205_60",
                    "TT_tZNtHyyN_500_180_50",
                    "TT_tZNtHyyN_1000_205_60",
                    "TT_tZNtHyyN_1200_205_60",
                    "HVT_VcXjjHyy_500_10",
                    "HVT_VcXjjHyy_500_300",
                    "HVT_VcXjjHyy_2000_300",
                    "HVT_VcXjjHyy_2000_1000",
                    "HVT_VcXjjHyy_2000_1700",
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


class RunRunze(PlotEventsWrapper):
    """
    Scoped-down PlotEventsWrapper: only nonres_yy_jjj and nonres_llyy_jj,
    using the new MLM matching setup.
    """

    version = law.Parameter(default="prod_12_mlm")

    def requires(self):
        config = dict(
            detector="ATLAS_fatjet_skimAll",
            ecm=13000.0,
            processor="fullmc",
        )
        return {
            "nonres_yy_jjj": PlotEvents.req(self, process="nonres_yy_jjj", n_events=1e8, n_max=1e5, **config),
            "nonres_llyy_jj": PlotEvents.req(self, process="nonres_llyy_jj", n_events=1e7, n_max=1e5, **config),
        }


