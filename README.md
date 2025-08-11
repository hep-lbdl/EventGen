# EventGenDelphes
Minimal python-based workflow to run particle physics simulations on the Perlmutter cluster using Madgraph, Pythia, and Delphes.
It also includes software to skim and plot key features of the events.

## Setup
To setup your working area do:
```bash
source setup.sh
```

When not done yet, this command installs the necessary software environment, pythia, and delphes.
Also it activates the software environment and sets the necessary environment variables.

## Configs
The process configurations are done in the `config` directory.
To add a process, add a new directory with the process name.
In the new directory you might add a madgraph configuration `madgraph.dat` and a pythia configuration `pythia.cmnd`.

If a madgraph configuration exist, the workflow will produce an LHE file with madgraph and feed this file to pythia and delphes.
If only a pythia configuration exists, the workflow will generate the events with pythia only.

You should use the following placeholders, which will be automatically filled/overwritten:

### In `madgraph.dat`
Use:
```bash
output OUTPUT_PLACEHOLDER
launch
set nevents NEVENTS_PLACEHOLDER
```
to set the output directory and the number of events.

### In `pythia.cmnd`
Use:
```bash
Main:numberOfEvents = NEVENTS_PLACEHOLDER 

Beams:frameType = 4
Beams:LHEF = INPUT_PLACEHOLDER
```
to set the number of events and import the correct madgraph produced LHE files.

## Workflow
The workflow is run with `law`.

To index the current available tasks and make them and their parameters usable in the command line please do:
```bash
law index
```

Then you may run the workflow with executing e.g.:
```bash
law run PlotEvents --process test --processor test
```

You can look at all outputs of the workflow with
```bash
law run PlotEvents --process test --processor test --print-status 4
```

which produces the following output (paths are shortened):
```bash
print task status with max_depth 4 and target_depth 0

0 > PlotEvents(cluster_mode=local, version=dev, process=test, detector=CMS, processor=test, n_events=1000)
│     LocalDirectoryTarget(fs=local_fs, path=output/version_dev/PlotEvents/test/n_events_1000/CMS/test/plots.pdf)
│       existent
│
└──1 > SkimEvents(cluster_mode=local, version=dev, process=test, detector=CMS, processor=test, n_events=1000)
   │     cutflow: LocalFileTarget(fs=local_fs, path=output/version_dev/SkimEvents/test/n_events_1000/CMS/test/cutflow.json)
   │       existent
   │     events: LocalFileTarget(fs=local_fs, path=output/version_dev/SkimEvents/test/n_events_1000/CMS/test/skimmed.h5)
   │       existent
   │
   └──2 > DelphesPythia8(cluster_mode=local, version=dev, process=test, detector=CMS, n_events=1000, n_max=1000000)
      │     0:
      │       config: LocalFileTarget(fs=local_fs, path=output/version_dev/DelphesPythia8/test/n_events_1000/CMS/0/config.txt)
      │         existent
      │       events: LocalFileTarget(fs=local_fs, path=output/version_dev/DelphesPythia8/test/n_events_1000/CMS/0/events.root)
      │         existent
      │       out: LocalFileTarget(fs=local_fs, path=output/version_dev/DelphesPythia8/test/n_events_1000/CMS/0/out.txt)
      │         existent
      │
      ├──3 > Madgraph(version=dev, process=test, n_events=1000, n_max=1000000)
      │  │     0:
      │  │       config: LocalFileTarget(fs=local_fs, path=output/version_dev/Madgraph/test/n_events_1000/config_0.dat)
      │  │         existent
      │  │       madgraph_dir: LocalDirectoryTarget(fs=local_fs, path=output/version_dev/Madgraph/test/n_events_1000/out_0)
      │  │         existent
      │  │       events: LocalFileTarget(fs=local_fs, path=output/version_dev/Madgraph/test/n_events_1000/out_0/Events/run_01/unweighted_events.lhe.gz)
      │  │         existent
      │  │
      │  └──4 > MadgraphConfig(process=test, external)
      │           LocalFileTarget(fs=local_fs, path=config/test/madgraph.dat)
      │             existent
      │
      └──3 > PythiaConfig(process=test, external)
               LocalFileTarget(fs=local_fs, path=config/test/pythia.cmnd)
                 existent
```
We can see, that the workflow executed all necessary steps, such as using madgraph to produce `LHE` files, using pythia and delphes to produce `root` files, skimming the events, and finally plotting some key features.

## Misc
Madgraph creates a file named `py.py` in the directory in which the execution takes place. This file does not seem to cause any problems even when running many madgraph processes simultaneously. Please find a discussion about it [here](https://answers.launchpad.net/mg5amcnlo/+question/679610).

## Extra:
The current version of madgraph can't produce NLO processes in the environment which has root installed.
Therefore, we currently have two different software environments: `madgraph.yml` for Madgraph, and `eventgen.yml` for all the rest.