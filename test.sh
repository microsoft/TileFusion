#!/bin/bash

pytest tests/python/test_scatter_nd.py 2>&1 | tee test.log
pytest tests/python/test_flash_attn.py 2>&1 | tee -a test.log
