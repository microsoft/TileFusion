EXAMPLE_DIR := examples
TEST_DIR   	:= tests/python
UNIT_TEST 	?= test_scatter_nd
CPP_UT		?= test_gemm
CPP_UTS 	:= scripts/unittests/run_all_cpp_tests.sh

PY_EXAMPLE 	?= $(EXAMPLE_DIR)/python/scatter_nd.py
UNIT   		?= $(TEST_DIR)/$(UNIT_TEST).py

WITH_TEST ?= ON

BUILD_DIR 	:= build
DYNAMIC_LIB	:= $(BUILD_DIR)/libtilefusion.so

.PHONY: build example unit_test clean

build:
	@mkdir -p build 
	@cd build && cmake -DWITH_TESTING=$(WITH_TEST) .. && make -j$(proc)

$(DYNAMIC_LIB): build

py_example: $(DYNAMIC_LIB)
	@python3 $(PY_EXAMPLE)

unit_test: $(DYNAMIC_LIB)
	@python3 $(UNIT)

unit_test_cpp: $(DYNAMIC_LIB)
	@cd $(BUILD_DIR) && ctest -R $(CPP_UT) -V

unit_test_cpps: $(DYNAMIC_LIB)
	@sh $(CPP_UTS)

clean:
	@rm -rf build
