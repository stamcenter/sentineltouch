# ============================
# STAGE 1: Build OpenFHE
# ============================
FROM ubuntu:22.04 AS openfhe-build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    clang \
    libomp5 \
    libomp-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/openfheorg/openfhe-development.git
WORKDIR /opt/openfhe-development

RUN mkdir build && cd build && \
    cmake .. \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install


# ============================
# STAGE 2: Build SentinelTouch
# ============================
FROM ubuntu:22.04 AS sentinel-build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    clang \
    libomp5 \
    libomp-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=openfhe-build /usr/local /usr/local

WORKDIR /opt

# Clone repo into sentineltouchartifact
RUN git clone http://gitlab.ascslab-members.org/ngesbrian/sentineltouchartifact.git

# Rename folder to "sentineltouch"
RUN mv sentineltouchartifact sentineltouch

WORKDIR /opt/sentineltouch

RUN rm -rf build && mkdir build && cd build && \
    cmake .. \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)


# ============================
# STAGE 3: Runtime Image (Combined C++ and Python)
# ============================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install all runtime dependencies (C++ and Python)
RUN apt-get update && apt-get install -y \
    libomp5 \
    python3 \
    python3-pip \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy OpenFHE libraries
COPY --from=openfhe-build /usr/local /usr/local

# Copy built C++ binaries and source
COPY --from=sentinel-build /opt/sentineltouch /opt/sentineltouch

# Copy all data directories (preserve full structure)
COPY embedding_csv /opt/sentineltouch/embedding_csv
COPY encrypted_db /opt/sentineltouch/encrypted_db
COPY encrypted_users /opt/sentineltouch/encrypted_users
COPY exported_images /opt/sentineltouch/exported_images
COPY images /opt/sentineltouch/images
COPY trained_models /opt/sentineltouch/trained_models
COPY python /opt/sentineltouch/python

# Install Python dependencies
COPY requirements.txt /opt/sentineltouch/
RUN pip3 install --no-cache-dir -r /opt/sentineltouch/requirements.txt

# Create entry point script
RUN mkdir -p /opt/sentineltouch/entrypoints
COPY docker-entrypoint.sh /opt/sentineltouch/entrypoints/docker-entrypoint.sh
RUN chmod +x /opt/sentineltouch/entrypoints/docker-entrypoint.sh

# Set working directory to project root
WORKDIR /opt/sentineltouch

# Use the entry point script
ENTRYPOINT ["/opt/sentineltouch/entrypoints/docker-entrypoint.sh"]