TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared -o tf_nndistance_so.so tf_nndistance.cc tf_nndistance_g.cu.o -L/usr/local/cuda/lib64 -I $TF_INC -fPIC -lcudart
nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o tf_nndistance_so.so tf_nndistance.cc tf_nndistance_g.cu.o  -L $CUDA_PATH/lib64 -L $TF_LIB -ltensorflow_framework -I $TF_INC -fPIC -lcudart

