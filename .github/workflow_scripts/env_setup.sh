function setup_build_env {
    python -m pip install --upgrade pip
    python -m pip install tox
    python -m pip install flake8
    python -m pip install bandit
    python -m pip install packaging
    python -m pip install ruff
}

function setup_pytorch_cuda_env {
    # Use wheel bundled CUDA instead of DLC CUDA with fallback to compatibility check bypass
    PYTORCH_CUDA_PATH=$(python -c '
try:
    import torch
    torch_cuda_path=torch._C._cuda_getLibPath()
    print(torch_cuda_path if torch_cuda_path else "")
except:
    print("")
')

    if [ -n "$PYTORCH_CUDA_PATH" ]; then
        echo "Using PyTorch bundled CUDA libraries from: $PYTORCH_CUDA_PATH"
        export LD_LIBRARY_PATH=$PYTORCH_CUDA_PATH:$LD_LIBRARY_PATH
    else
        echo "Warning: Could not get PyTorch bundled CUDA path. Falling back to PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1"
        export PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1
    fi
}

function setup_build_contrib_env {
    python -m pip install --upgrade pip
    python -m pip install -r $(dirname "$0")/../../docs/requirements_doc.txt
    export AG_DOCS=1
    export AUTOMM_TUTORIAL_MODE=1 # Disable progress bar in MultiModalPredictor
}

function setup_benchmark_env {
    git clone https://github.com/autogluon/autogluon-bench.git
    cd autogluon-bench
    pip install -e ".[tests]"
    cd ..
    pip install pyarrow
    git clone https://github.com/autogluon/autogluon-dashboard.git
    pip install -e ./autogluon-dashboard
    pip install yq
    pip install s3fs
}

function setup_hf_model_mirror {
    pip install PyYAML
    SUB_FOLDER="$1"
    python $(dirname "$0")/setup_hf_model_mirror.py --model_list_file $(dirname "$0")/../../multimodal/tests/hf_model_list.yaml --sub_folder $SUB_FOLDER
}

function install_local_packages {
    while(($#)) ; do
        python -m pip install --upgrade -e $1
        shift
    done
}

function install_tabular {
    python -m pip install --upgrade pygraphviz
    install_local_packages "tabular/$1"
}

function install_tabular_platforms {
    # pygraphviz will be installed with conda in platform tests
    install_local_packages "tabular/$1"
}

function install_multimodal {
    source $(dirname "$0")/setup_mmcv.sh

    # launch different process for each test to make sure memory is released
    python -m pip install --upgrade pytest-xdist
    install_local_packages "multimodal/$1"
    setup_mmcv
    # python -m pip install --upgrade "mmocr<1.0"  # not compatible with mmcv 2.0
}

function install_all {
    install_local_packages "common/[tests]" "core/[all]" "features/" "tabular/[all,tests]" "timeseries/[all,tests]" "eda/[tests]"
    install_multimodal "[tests]"
    install_local_packages "autogluon/"
}

function install_all_windows {
    install_local_packages "common/[tests]" "core/[all]" "features/" "tabular/[all,tests]" "timeseries/[all,tests]" "eda/[tests]"
    install_multimodal "[tests]"
    install_local_packages "autogluon/"
}

function install_all_no_tests {
    install_local_packages "common/" "core/[all]" "features/" "tabular/[all]" "timeseries/[all]" "eda/"
    install_multimodal
    install_local_packages "autogluon/"
}

function build_pkg {
    pip install --upgrade setuptools wheel
    while(($#)) ; do
        cd "$1"/
        python setup.py sdist bdist_wheel
        cd ..
        shift
    done
}

function build_all {
    build_pkg "common" "core" "features" "tabular" "multimodal" "timeseries" "autogluon" "eda"
}
