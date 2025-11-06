import os
import logging
import warnings
import errno

import law
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster


logger = logging.getLogger(__name__)

# Filter dask warnings about used port
warnings.filterwarnings(
    "ignore",
    message=r"Port \d+ is already in use",
    category=UserWarning,
    module="distributed.node",
)


configs = {
    "perlmutter_debug": dict(
        cores=1,
        memory="16GB",
        walltime="00:30:00",
        job_extra_directives=["--qos=debug", "-C cpu"],
    ),
    "perlmutter_small": dict(
        cores=1,
        memory="16GB",
        walltime="01:00:00",
        job_extra_directives=["--qos=shared", "-C cpu"],
    ),
    "perlmutter_medium": dict(
        cores=1,
        memory="1GB",
        walltime="24:00:00",
        job_extra_directives=["--qos=shared", "-C cpu"],
    ),
    "perlmutter_node": dict(
        cores=236,
        memory="400GB",
        walltime="1:00:00",
        job_extra_directives=["--qos=regular", "-C cpu"],
    ),
}


class ClusterMixin:
    cluster_mode = law.Parameter(default="local")

    cores = 1
    memory = "1GB"
    walltime = "01:00:00"
    qos = "shared"
    arch = "cpu"

    @property
    def log_dir(self):
        return [
            f'-o {os.getenv("GEN_SLURM")}/slurm-%j.out',
            f'-e {os.getenv("GEN_SLURM")}/slurm-%j.err',
        ]

    def start_cluster(self, n_nodes=1):
        # Set up the SLURM cluster
        # Always scale to the maximum number of nodes,
        # dask will use however much it needs, until
        # all jobs are successfully finished
        if self.cluster_mode == "local":
            cluster = LocalCluster()
        elif self.cluster_mode == "fullnode":
            # Run on allocated interactive node and we use all resources
            # use extra thread for IO or potential MT speedup, leave some mem for system and temp swap
            cluster = LocalCluster(32, threads_per_worker=8, memory_limit="12GiB")
        elif self.cluster_mode == "slurm":
            cluster = SLURMCluster(
                cores=self.cores,
                memory=self.memory,
                walltime=self.walltime,
                job_extra_directives=[f"--qos={self.qos}", f"-C {self.arch}"]
                + self.log_dir,
            )
            cluster.scale(n_nodes)
        else:
            raise ValueError(f"Unknown cluster mode {self.cluster}")
        logger.info(f"Dask dashboard at {cluster.dashboard_link}")
        return cluster


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred