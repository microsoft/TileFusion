EXAMPLE_DIR := examples
CPP_UT		?= test_gemm
CPP_UTS 	:= scripts/unittests/run_all_cpp_tests.sh

# TODO: Update this command as the Python example is outdated and
# not in sync with the latest changes in the master branch.
# PY_EXAMPLE 	?= $(EXAMPLE_DIR)/python/scatter_nd.py

WITH_TEST ?= ON

BUILD_DIR 	:= build
DYNAMIC_LIB	:= $(BUILD_DIR)/libtilefusion.so

.PHONY: build example unit_test clean

build:
	@mkdir -p build
	@cd build && cmake -DWITH_TESTING=$(WITH_TEST) .. && make -j$(proc)

$(DYNAMIC_LIB): build

# TODO: Update this command as the Python example is outdated and
# not in sync with the latest changes in the master branch.
# py_example: $(DYNAMIC_LIB)
# 	@python3 $(PY_EXAMPLE)

unit_test_cpp: $(DYNAMIC_LIB)
	@cd $(BUILD_DIR) && ctest -R $(CPP_UT) -V

unit_test_cpps: $(DYNAMIC_LIB)
	@sh $(CPP_UTS)

clean:
	@rm -rf build
