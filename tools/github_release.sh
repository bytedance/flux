#!/bin/bash
export http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export no_proxy="byted.org" 

OUTPUT_DIR="$(pwd)/github_release"
mkdir -p "$OUTPUT_DIR"
# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda command could not be found. Please install Anaconda or Miniconda."
    exit 1
fi

# Check if nvcc is installed
if ! command -v nvcc &> /dev/null
then
    echo "nvcc command could not be found. Please ensure CUDA is installed and nvcc is in your PATH."
    exit 1
fi


CONDA_PATH=$(which conda)
ANACONDA_HOME=$(dirname $(dirname $CONDA_PATH))
# 获取当前 nvcc 版本号
nvcc_version=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")

# 提取主版本号和次版本号
major_version=$(echo $nvcc_version | cut -d. -f1)
minor_version=$(echo $nvcc_version | cut -d. -f2)

# 版本号比较函数
version_greater_than() {
    if [ "$1" -gt "$3" ] || { [ "$1" -eq "$3" ] && [ "$2" -ge "$4" ]; }; then
        return 0
    else
        return 1
    fi
}


# Get CUDA version
cuda_version=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
cuda_version_number=${cuda_version//./}
cuda_version_number=${cuda_version_number:0:3} # Truncate to first three characters, e.g., 11.8 -> 118

# Define versions
pytorch_versions=("2.1.0" "2.2.0" "2.3.0" "2.4.0")
python_versions=("3.8" "3.9" "3.10" "3.11")

if version_greater_than $major_version $minor_version 12 0; then
    cuda_suffix="cu121"
else
    cuda_suffix="cu118"
fi
echo "cuda_suffix ${cuda_suffix}"

rm -rf dist
export FLUX_USE_LOCAL_VERSION=0
# Loop through each combination of PyTorch and Python versions
for pytorch_version in "${pytorch_versions[@]}"
do
    for python_version in "${python_versions[@]}"
    do
        env_name="torch${pytorch_version}_py${python_version}_cuda${cuda_version_number}"
        echo "Creating environment $env_name with PyTorch $pytorch_version, Python $python_version, and CUDA $cuda_version"
        # Check if the environment already exists and remove it if it does
        # if conda env list | grep -q $env_name; then
        #     echo "Environment $env_name already exists. Removing it."
        #     conda env remove -n $env_name -y
        # fi
        #Create the environment
        conda create -n $env_name python=$python_version -y

        # Activate the environment
        source $ANACONDA_HOME/etc/profile.d/conda.sh
        conda activate $env_name

        echo "Environment $env_name created and build.sh executed successfully."
        
        # # Install the specific version of PyTorch
        # conda install pytorch==$pytorch_version cudatoolkit=$cuda_version -c pytorch -y
        pip3 install cmake packaging
        pip3 install torch==${pytorch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/${cuda_suffix}
        # # Execute build.sh
        ./build.sh --clean-all
        ./build.sh --arch "80;90" --package
        # rename the whl
        for file in dist/*.whl; do
            if [[ -f $file ]]; then
                IFS='.' read -r -a version_parts <<< "$pytorch_version"
                short_version="${version_parts[0]}.${version_parts[1]}"
                torch_suffix="torch$short_version"
                BASE_NAME=$(basename "$file")
                NEW_NAME=$(echo "$BASE_NAME" | sed "s/\(byte_flux-[0-9]*\.[0-9]*\.[0-9]*\)\(-cp[0-9]*-cp[0-9]*-linux_x86_64\.whl\)/\1+$cuda_version_number$torch_suffix\2/")
                mv "$file" "dist/$NEW_NAME"
                echo "Renamed $BASE_NAME to $NEW_NAME"
            fi
        done
        mv dist "$OUTPUT_DIR/$env_name"
        # Deactivate the environment
        conda deactivate

    done
done

echo "All environments created and build.sh executed successfully."
