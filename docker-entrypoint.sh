#!/bin/bash

# Default to hybrid_16 if no arguments provided
if [ $# -eq 0 ]; then
    cd /opt/sentineltouch/build
    exec ./hybrid_16
fi

# Check if first argument is "python"
if [ "$1" == "python" ]; then
    shift  # Remove "python" from arguments
    cd /opt/sentineltouch/python
    exec python3 "$@"
fi

# Check if first argument is "hybrid_32"
if [ "$1" == "hybrid_32" ]; then
    shift  # Remove from arguments
    cd /opt/sentineltouch/build
    exec ./hybrid_32 "$@"
fi

# Check if first argument is "lenet5_16"
if [ "$1" == "lenet5_16" ]; then
    shift  # Remove from arguments
    cd /opt/sentineltouch/build
    exec ./lenet5_16 "$@"
fi

# Check if first argument is "lenet5_32"
if [ "$1" == "lenet5_32" ]; then
    shift  # Remove from arguments
    cd /opt/sentineltouch/build
    exec ./lenet5_32 "$@"
fi

# Check if first argument is "cpp" (generic C++ runner)
if [ "$1" == "cpp" ]; then
    shift  # Remove "cpp" from arguments
    cd /opt/sentineltouch/build
    exec ./hybrid_16 "$@"
fi

# Otherwise, treat as direct command
exec "$@"