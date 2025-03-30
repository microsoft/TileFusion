EXAMPLE_DIR := examples
CPP_UT		?= test_gemm
WITH_TEST ?= ON

BUILD_DIR 	:= build
DYNAMIC_LIB	:= $(BUILD_DIR)/libtilefusion.so

.PHONY: build example unit_test clean

build:
	@mkdir -p build
	@cd build && cmake -DWITH_TESTING=$(WITH_TEST) .. && make -j$(proc)

$(DYNAMIC_LIB): build
unit_test_cpp: $(DYNAMIC_LIB)
	@cd $(BUILD_DIR) && ctest -R $(CPP_UT) -V

clean:
	@rm -rf build
