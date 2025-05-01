BUILD_DIR := build
EXECUTABLE := fused_two_gemms

.PHONY: build clean run

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && make -j$(proc)

run: build
	@echo "Running $(EXECUTABLE)..."
	@cd $(BUILD_DIR) && ./$(EXECUTABLE)

clean:
	@rm -rf $(BUILD_DIR)
