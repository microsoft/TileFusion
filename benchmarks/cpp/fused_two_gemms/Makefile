# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


BUILD_DIR := build

.PHONY: build clean run

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && make -j$(proc)

clean:
	@rm -rf $(BUILD_DIR)

run: build
	@$(BUILD_DIR)/bench_fused_two_gemms
