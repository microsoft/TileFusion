#!/bin/bash

# pytest tests/python/test_scatter_nd.py 2>&1 | tee test.log
# pytest tests/python/test_flash_attn.py 2>&1 | tee -a test.log
# pytest tests/python/test_fused_two_gemms.py 2>&1 | tee -a test.log

CACHE_DIR="$HOME/.cache/tilefusion"

if [ -d "$CACHE_DIR" ]; then
    echo "Cache directory found at: $CACHE_DIR"

    echo "Current contents of cache directory:"
    ls -l "$CACHE_DIR/"

    echo "Deleting all files in cache directory..."
    rm -rf "$CACHE_DIR"/*

    if [ $? -eq 0 ]; then
        echo "Successfully cleared cache directory"
    else
        echo "Error: Failed to clear cache directory"
        exit 1
    fi
else
    echo "Cache directory does not exist at: $CACHE_DIR"
    exit 0
fi

python tests/python/test_fused_two_gemms.py 2>&1 | tee test.log
