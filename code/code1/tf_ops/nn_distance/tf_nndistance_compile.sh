#/bin/bash

TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

echo $TF_INC
echo $TF_LIB

nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared -o tf_nndistance_so.so tf_nndistance.cc tf_nndistance_g.cu.o  -L/usr/local/cuda/lib64 -L $TF_LIB -ltensorflow_framework -I $TF_INC -fPIC -lcudart -D_GLIBCXX_USE_CXX11_ABI=0

