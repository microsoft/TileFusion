#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# for file in find(); do
#     unit = $(basename $file .py)
#     make unit_test UNIT_TEST=$unit
# done

for file in $(find tests/python -name *.py); do
    unit=$(basename $file .py)
    make unit_test UNIT_TEST=$unit
done
