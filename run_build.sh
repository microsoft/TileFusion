#!/bin/bash

cur_dir=$(pwd)
build_dir="$cur_dir/build/temp.linux-x86_64-cpython-312"

lib_dir="$build_dir/src"
lib_name="libtilefusion.so"

source_lib="$lib_dir/$lib_name"
target_lib="tilefusion/$lib_name"

if [ -f "$source_lib" ]; then
    rm -rf "$source_lib"
fi

if [ -f "$target_lib" ]; then
    rm -rf "$target_lib"
fi

cd $build_dir

if [ -f "CMakeCache.txt" ]; then
    rm -rf CMakeCache.txt CMakeFiles
fi

if [ -d "CMakeFiles" ]; then
    rm -rf CMakeFiles
fi

cmake ../../

make -j32 2>&1 | tee ../../build.log

cd $cur_dir

if [ -f "$source_lib" ]; then
    cp "$source_lib" "$target_lib"
else
    echo "Failed to build TileFusion library"
    exit 1
fi

exit 0

python setup.py develop 2>&1 | tee build.log

# ./test.sh 2>&1 | tee test.log
