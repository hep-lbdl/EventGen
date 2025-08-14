#!/usr/bin/env bash
action() {
    # Set version of used software
    local madgraph_download_dir="https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download"
    local madgraph_download_file="MG5_aMC_v3.5.9"

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

    # Setup software directories
    export SOFTWARE_DIR="${this_dir}/software"
    mkdir -p $SOFTWARE_DIR

    export MADGRAPH_DIR="${SOFTWARE_DIR}/${madgraph_download_file//./_}"

    # If no conda available, activate it
    if ! command -v conda >/dev/null 2>&1; then
        module load python
    fi

    # If conda env "madgraph" does not exist create it
    if ! conda env list | grep -q '^madgraph'; then
        mamba create --name madgraph
        mamba env update -n madgraph --file madgraph.yml -y
    fi

    # If conda env "eventgen" does not exist create it
    if ! conda env list | grep -q '^eventgen'; then
        mamba create --name eventgen
        mamba env update -n eventgen --file eventgen.yml -y
    fi

    # Activate conda environment eventgen
    conda activate madgraph

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
