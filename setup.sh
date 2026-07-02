#!/usr/bin/env bash
action() {
    # Set version of used software
    # Launchpad removed the 3.5.11 tarball; use the CERN LCG/GENSER mirror instead
    local madgraph_download_dir="https://lcgpackages.web.cern.ch/tarFiles/sources/MCGeneratorsTarFiles"
    local madgraph_download_file="MG5_aMC_v3.5.11"

    # Set main directories
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # set PYTHONPATH
    export PYTHONPATH="${this_dir}:${PYTHONPATH}"

    CONFIG_FILE="${this_dir}/.config"

    # Function to read the output directory from the config file
    read_config() {
        if [[ -f $CONFIG_FILE ]]; then
            source $CONFIG_FILE
        else
            export GEN_OUT=""
        fi
    }

    # Function to write the output directory to the config file
    write_config() {
        echo "export GEN_OUT=\"$GEN_OUT\"" > $CONFIG_FILE
    }

    # Prompt user for input if GEN_OUT is not set
    prompt_user() {
        read -p "Enter the output directory: " user_input
        if [[ -n $user_input ]]; then
            export GEN_OUT=$user_input
            write_config
        fi
    }

    # Main script execution
    read_config

    if [[ -z $GEN_OUT ]]; then
        echo "No output directory configured."
        prompt_user
    fi

    # Use the GEN_OUT in your script
    echo "Using output directory: $GEN_OUT"

    # Set code and law area
    export GEN_CODE="${this_dir}"
    export GEN_SLURM="${GEN_OUT}/slurm"

    export LAW_HOME="${this_dir}/.law"
    export LAW_CONFIG_FILE="${this_dir}/law.cfg"

    export SOFTWARE_DIR="${this_dir}/software"
    mkdir -p $SOFTWARE_DIR

    export MADGRAPH_DIR="${SOFTWARE_DIR}/${madgraph_download_file//./_}"

    # If no conda available, activate it
    if ! command -v conda >/dev/null 2>&1; then
        module load python
    fi

    # Fixed absolute path (not --name) so group members can activate it
    # regardless of their own conda envs_dirs config.
    export EVENTGEN_ENV="/pscratch/sd/d/dnoll/tools/conda/eventgen"

    # If conda env does not exist yet, create it
    if [ ! -d "${EVENTGEN_ENV}" ]; then
        yes | conda create --prefix "${EVENTGEN_ENV}"
        yes | conda env update --prefix "${EVENTGEN_ENV}" --file eventgen.yml
        # Install temporary Delphes fix (H->yy filter) from:
        # https://github.com/qibin2020/delphes/commit/2104fd9
        cp /pscratch/sd/d/dnoll/projects/haxad/EventGenDelphes/bin/DelphesPythia8Filtered "${EVENTGEN_ENV}/bin/"
        chmod u+x "${EVENTGEN_ENV}/bin/DelphesPythia8Filtered"
        chgrp -R m3246 "${EVENTGEN_ENV}"
        chmod -R g+rX "${EVENTGEN_ENV}"
    fi
    chmod o+x "$(dirname "${EVENTGEN_ENV}")" "$(dirname "$(dirname "${EVENTGEN_ENV}")")"

    # Activate the conda environment by path
    conda activate "${EVENTGEN_ENV}"

    # law setup
    source "$( law completion )" ""

    # If Pythia not installed yet, do so
    if [ -d "$MADGRAPH_DIR" ]; then
        echo "Madgraph already installed"
    else
        echo "Installing Madgraph"
        cd $SOFTWARE_DIR
        wget ${madgraph_download_dir}/${madgraph_download_file}.tar.gz
        tar -xf "${madgraph_download_file}.tar.gz"
        rm "${madgraph_download_file}.tar.gz"
        
        cd $this_dir
    fi

    export PYTHIA_DIR="${CONDA_PREFIX}"
    export DELPHES_DIR="${CONDA_PREFIX}"
}
action
