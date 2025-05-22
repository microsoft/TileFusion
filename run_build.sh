#!/bin/bash

proj_root_dir=$(pwd)
build_dir="$proj_root_dir/build/temp.linux-x86_64-cpython-312"

# lib_dir="$build_dir/src"
lib_dir="$proj_root_dir/build/lib.linux-x86_64-cpython-312/tilefusion"
lib_name="libtilefusion.so"

source_lib="$lib_dir/$lib_name"
target_lib="$proj_root_dir/tilefusion/$lib_name"

if [ -f "$source_lib" ]; then
    rm "$source_lib"
fi

if [ -f "$target_lib" ]; then
    rm "$target_lib"
fi

echo "build dir: $build_dir"
cd $build_dir

# if [ -f "CMakeCache.txt" ]; then
#     rm CMakeCache.txt
# fi

# if [ -d "CMakeFiles" ]; then
#     rm -rf CMakeFiles
# fi

# cmake -DCMAKE_BUILD_TYPE=Debug ../../ 2>&1 | tee ../../build.log

# TEST_NAME="test_single_wmma"
TEST_NAME="test_g2r_copy"
TEST_DIR="$build_dir/tests/cpp"
TEST_PATH="$TEST_DIR/${TEST_NAME}"

if [ -f "$TEST_PATH" ]; then
    rm "$TEST_PATH"
fi

make -j32 2>&1 | tee ../../build.log

if [ -f "$source_lib" ]; then
    echo "Copying $source_lib to $target_lib"
    cp "$source_lib" "$target_lib"
else
    echo "Failed to build TileFusion library"
    exit 1
fi

cd $proj_root_dir

if [ -f "$TEST_PATH" ]; then
    echo "Running test: $TEST_PATH"
    $TEST_PATH 2>&1 | tee test.log
else
    echo "Test file not found: $TEST_PATH"
    exit 1
fi


# python setup.py develop 2>&1 | tee build.log
# ./test.sh 2>&1 | tee test.log
