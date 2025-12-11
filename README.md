# SentinelTouch

This work introduces SentinelTouch, an open-source framework for privacy-preserving fingerprint authentication and identification system that delivers both efficiency and accuracy in HE environments.
Our key insight is a twofold optimization: 

1) A preprocessing pipeline that reduces fingerprint image dimensions to as low while preserving most of its discriminative details 

2) The design of a lightweight, HE-friendly neural network that generalizes effectively on this compact data. 

We present two pipelines for usage: 

1) Full-privacy pipeline, where encrypted images are processed entirely under HE settings, achieving user identification in a one-to-many setting. 

2) A hybrid pipeline, where only encrypted embeddings are processed under HE settings, achieving one-to-many user identification.


## Build OpenFHE Project

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


## Docker Image. 

To build the docker image.

```
docker build -t sentineltouch .
docker run  sentineltouch # run the default entry which is hybrid_16
docker run ./hybrid_32 sentineltouch # run the hybrid 32 entry point
docker run ./lenet5_16 sentineltouch # run the full lenet 16 embeddings 32 entry point
docker run ./lenet5 sentineltouch # run the full-pipeline with 32 embedding space. 


docker run sentineltouch python pca_1_to_N_authentication.py # run the python module for training and validation of model
docker run sentineltouch python exportAllWeights.py # run the python module to export all weights


```
