# Artifact Appendix (Required for all badges)

Paper title: **SentinelTouch: A Lightweight Privacy-Preserving Biometric-Fingerprinting Authentication and Identification System Based on Neural Networks and Homomorphic Encryption**

Requested Badge(s):
- [x] **Available**
- [x] **Functional**
- [x] **Reproduced**


## Description (Required for all badges)

SentinelTouch is an open-source implementation of the system described in the paper. It provides two primary pipelines for fingerprint authentication and identification under privacy constraints:

- Full-privacy pipeline: the image preprocessing and subsequent model inference are performed under fully homomorphically encrypted (HE) inputs, enabling one-to-many identification without revealing raw biometric data.
- Hybrid pipeline: images are preprocessed locally to produce compact embeddings; only those embeddings are encrypted and processed under HE, reducing computation and communication costs while preserving privacy for the identification step.

Key contributions implemented in this artifact:

- A preprocessing pipeline that reduces fingerprint image dimensionality while retaining discriminative features used by the neural models.
- An  HE-friendly neural architecture for fingerprint processing.
- Native C++ binaries that run HE-friendly inference (`hybrid_16`, `hybrid_32`, `lenet5_16`, `lenet5_32`).
- Python utilities for dataset preparation, embeddings extraction, and evaluation (under `python/`).

This repository enables readers to reproduce the experiments in the paper (accuracy, runtime, memory), explore trade-offs between embeddings sizes, and test hybrid vs full-privacy pipelines.

### Security / Privacy Issues and Ethical Concerns (Required for all badges)

SentinelTouch is intended to improve privacy by enabling operations on encrypted biometric data. Nevertheless, users should be aware of the following:

- The codebase contains utilities and datasets for biometric data — handle datasets according to local laws and institutional review board (IRB) policies.
- The artifact does not attempt to deanonymize or re-identify users beyond intended authentication/identification workflows; misuse for surveillance or other unethical purposes is prohibited.
- No known software vulnerabilities are intentionally provided. Users should follow usual security best practices: update dependencies, run in isolated environments, and restrict access to any produced models or keys.


## Hardware Requirements (Required for Functional and Reproduced badges)

The experiments reported in the paper were executed on a workstation with an Intel Core i7-14700K (20 cores) and 32 GB RAM running Ubuntu 24.04. Reproduction is possible on less powerful machines, though runtimes will increase.

Recommended minimal hardware for functional testing:

- CPU: 20 cores (x86_64)
- RAM: 8 GB (16 GB recommended for building OpenFHE)
- Disk: 10 GB free for a native build; 25 GB if using Docker images and datasets


### Software Requirements (Required for Functional and Reproduced badges)

The repository has native build steps that require a C++ toolchain and optional Docker-based runtime. Primary software used and tested:

- OS: Ubuntu 22.04 / 24.04 (Linux recommended; other POSIX systems may work)
- C++: clang or gcc, cmake, build-essential
- Python: 3.10+ and pip (used for dataset utilities and evaluation scripts)
- Libraries: OpenFHE (for homomorphic encryption)
- Container: Docker (optional, recommended for repeatability)

Install dependencies via the provided `Dockerfile` for a reproducible environment, or follow manual instructions below.


### Estimated Time and Storage Consumption (Required for Functional and Reproduced badges)

- Minimal end-to-end setup (using Docker): ~30–90 minutes machine speed.
- Native build (OpenFHE + project): ~1 hours depending on CPU cores.
- Storage: ~10 GB (native) to ~25 GB (Docker image + data)


## Environment (Required for all badges)

Clone the repository and follow the README or use the Docker image for the quickest reproducible environment. You can either build OpenFHE locally and compile the C++ binaries, or use the `Dockerfile` included to produce a ready-to-run image.

### Set up the environment (Required for Functional and Reproduced badges)

#### Option A — Native build (manual)

#### Build OpenFHE Project

After installing OpenFHE as in [OpenFHE](https://github.com/openfheorg/openfhe-development)

```
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe
sudo apt-get install build-essential #this already includes g++
sudo apt-get install cmake
sudo apt-get install clang
sudo apt-get install libomp5
sudo apt-get install libomp-dev
mkdir build
cd build
cmake ..
make
sudo make install
```


## Setup SentinelTouch using this command:

```
git clone http://gitlab.ascslab-members.org/ngesbrian/sentineltouch.git
cd sentineltouch
mkdir build
cd build
cmake ..
make
```

## Run the Builds of SentinelTouch

Run the build from the SentinelTouch for results
```
./hybrid_16
```

Run a Python evaluation script (inside `python/`):

```
python3 python/pca_1_to_N_authentication.py
```


#### Option B — Docker (recommended for reproducibility)


To build the docker image.

```bash
docker build -t sentineltouch .
docker run  sentineltouch # run the default entry which is hybrid_16
docker run sentineltouch hybrid_32  # run the hybrid 32 entry point
docker run sentineltouch lenet5_16 # run the full lenet 16 embeddings 32 entry point
docker run sentineltouch lenet5_32 # run the full-pipeline with 32 embedding space. 


docker run sentineltouch python pca_1_to_N_authentication.py # run the python module for training and validation of model
docker run sentineltouch python exportAllWeights.py # run the python module to export all weights
```

Note: the `docker-entrypoint.sh` supports aliases such as `hybrid_16`, `hybrid_32`, `lenet5_16`, `lenet5_32`, and `python`.


### Testing the Environment (Required for Functional and Reproduced badges)

Basic checks to confirm the environment is functional:

1. Build or pull the Docker image (or build natively).
2. From the repository root run the hybrid binary to verify it starts and prints expected status logs:

```bash
./build/hybrid_16 --help
# or inside docker
docker run sentineltouch hybrid_16
```

3. Run a new training using `python/pca_1_to_N_authentication.py`


### Main Results and Claims

The repository contains scripts and binaries required to reproduce the main results in the paper:

- Accuracy evaluation: use `python/pca_1_to_N_authentication.py` or the scripts in `python/` to train/evaluate on the provided validation splits.
- Runtime and memory: measured by running the native binaries `hybrid_16`, `hybrid_32`, `lenet5_16`, `lenet5_32`; these binaries include internal logging for runtime counters.

See `python/metadata/` and `results/` for example outputs and expected result formats used in the paper.


## Limitations (Required for Functional and Reproduced badges)

- Reproducing exact runtime numbers requires similar hardware and compiler / BLAS configurations; small differences are expected across machines.
- The proof-of-concept datasets and pre-processing are tailored for fingerprint images used in this paper; adapting to other biometric modalities will require dataset-specific tuning.

Though not directly evaluated, we provide `python/raspberry_pi_evaluation/` for `PCA+LeNet-5` and `ResNet-18` comparision on Raspberry PI.


## Notes on Reusability (Encouraged for all badges)

This section documents how others can reuse components of SentinelTouch in their own work.

- **Modular design:** The repository separates preprocessing, model training, and HE inference. Reuse the preprocessing pipeline (`python/utils` and `python/*_dataset.py`) to produce compact embeddings for other models.
- **Models and weights:** Trained model weights are stored in `trained_models/`. You can load these weights to run evaluation or adapt them to transfer-learning experiments.
- **HE integration:** The HE interface is implemented using OpenFHE. The project includes example parameter sets and wrapper code to adapt other models to HE by replacing the inference core with a compatible implementation.
- **Extending pipelines:** To add a new network or change embedding size:
  1. Add a new model in `python/` and implement a small wrapper to export weights in the same format as the existing loaders.
  2. Add a corresponding C++ inference model or adopt our provided model t ouse your weights.
- **Licensing and attribution:** The code is open-source (see repository `README.md` for exact license). When reusing components, attribute the original authors and follow the license terms.

---