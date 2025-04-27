#!/bin/bash

proj_root_dir=$(pwd)
build_dir="$proj_root_dir/build/temp.linux-x86_64-cpython-312"

lib_dir="$build_dir/src"
lib_name="libtilefusion.so"

source_lib="$lib_dir/$lib_name"
target_lib="$proj_root_dir/tilefusion/$lib_name"

if [ -f "$source_lib" ]; then
    rm "$source_lib"
fi

if [ -f "$target_lib" ]; then
    rm "$target_lib"
fi

cd $build_dir

if [ -f "CMakeCache.txt" ]; then
    rm CMakeCache.txt
fi

if [ -d "CMakeFiles" ]; then
    rm -rf CMakeFiles
fi

cmake -DCMAKE_BUILD_TYPE=Debug ../../ 2>&1 | tee ../../build.log

make -j32 2>&1 | tee -a ../../build.log

if [ -f "$source_lib" ]; then
    cp "$source_lib" "$target_lib"
else
    echo "Failed to build TileFusion library"
    exit 1
fi

cd tests/cpp/

GLOG_logtostderr=1 GLOG_colorlogtostderr=1 GLOG_v=3 \
    ./test_jit 2>&1 | tee ../../test.log

cd $cur_dir

# exit 0

# python setup.py develop 2>&1 | tee build.log

# ./test.sh 2>&1 | tee test.log
