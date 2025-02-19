#!/bin/bash

build_dir="_build"

if [ ! -d "$build_dir" ]; then
   mkdir $build_dir
fi

cd $build_dir

if [ -d "CMakefile" ]; then
   rm -rf CMakefile
fi

if [ -f "CMakeCache" ]; then
   rm CMakeCache
fi


cmake -DCMAKE_C_COMPILER=`which gcc` \
   -DCMAKE_CXX_COMPILER=`which g++` \
   ../../

# TEST="tests/cpp/test_swizzled_copy"
# TEST="tests/cpp/test_tiled_matrix_layout"

# if [ -f "$TEST" ]; then
#     rm $TEST
# fi

make -j24 2>&1 | tee ../make.log

# if [ -f "$TEST" ]; then
#     ./$TEST 2>&1 | tee ../test.log
# fi

cd ..


make unit_test_cpps 2>&1 | tee unittests.log

# python3 setup.py clean
# export CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# python3 setup.py develop 2>&1 | tee build/build.log
# python3 setup.py build bdist_wheel 2>&1 | tee build.log
