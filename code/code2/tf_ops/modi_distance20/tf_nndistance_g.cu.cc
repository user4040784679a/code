#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <iostream>

__global__ void NmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,
                                int* flag1, int* flag2){
    const int batch=512;
    __shared__ float buf[batch*3];

    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int k2=0;k2<m;k2+=batch){
            int end_k=min(m,k2+batch)-k2;
            for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
                buf[j]=xyz2[(i*m+k2)*3+j];
            }
            __syncthreads();
            for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
                if (flag1[i*n+j] != 0) {
                    continue;
                }

                float x1=xyz[(i*n+j)*3+0];
                float y1=xyz[(i*n+j)*3+1];
                float z1=xyz[(i*n+j)*3+2];
                
                int best_i = 0;
                float best = 3.40282e+38;

                int end_ka=end_k-(end_k&3);

                if (end_ka==batch){
                    for (int k=0;k<batch;k+=4){
                        {
                            float x2=buf[k*3+0]-x1;
                            float y2=buf[k*3+1]-y1;
                            float z2=buf[k*3+2]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best && flag2[i*n + k+k2] == 0){
                                best=d;
                                best_i=k+k2;
                            }
                        }
                        {
                            float x2=buf[k*3+3]-x1;
                            float y2=buf[k*3+4]-y1;
                            float z2=buf[k*3+5]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best && flag2[i*n + k+k2+1] == 0){
                                best=d;
                                best_i=k+k2+1;
                            }
                        }
                        {
                            float x2=buf[k*3+6]-x1;
                            float y2=buf[k*3+7]-y1;
                            float z2=buf[k*3+8]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best && flag2[i*n + k+k2+2] == 0){
                                best=d;
                                best_i=k+k2+2;
                            }
                        }
                        {
                            float x2=buf[k*3+9]-x1;
                            float y2=buf[k*3+10]-y1;
                            float z2=buf[k*3+11]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best && flag2[i*n + k+k2+3] == 0){
                                best=d;
                                best_i=k+k2+3;
                            }
                        }
                    }
                }else{
                    for (int k=0;k<end_ka;k+=4){
                        {
                            float x2=buf[k*3+0]-x1;
                            float y2=buf[k*3+1]-y1;
                            float z2=buf[k*3+2]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best && flag2[i*n + k+k2] == 0){
                                best=d;
                                best_i=k+k2;
                            }
                        }
                        {
                            float x2=buf[k*3+3]-x1;
                            float y2=buf[k*3+4]-y1;
                            float z2=buf[k*3+5]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best && flag2[i*n + k+k2+1] == 0){
                                best=d;
                                best_i=k+k2+1;
                            }
                        }
                        {
                            float x2=buf[k*3+6]-x1;
                            float y2=buf[k*3+7]-y1;
                            float z2=buf[k*3+8]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best && flag2[i*n + k+k2+2] == 0){
                                best=d;
                                best_i=k+k2+2;
                            }
                        }
                        {
                            float x2=buf[k*3+9]-x1;
                            float y2=buf[k*3+10]-y1;
                            float z2=buf[k*3+11]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best && flag2[i*n + k+k2+3] == 0){
                                best=d;
                                best_i=k+k2+3;
                            }
                        }
                    }
                }

                for (int k=end_ka;k<end_k;k++){
                    float x2=buf[k*3+0]-x1;
                    float y2=buf[k*3+1]-y1;
                    float z2=buf[k*3+2]-z1;
                    float d=x2*x2+y2*y2+z2*z2;
                    if (d<best && flag2[i*n + k+k2] == 0){
                        best=d;
                        best_i=k+k2;
                    }
                }

                if (k2==0 || result[(i*n+j)]>best){
                    result[(i*n+j)]=best;
                    result_i[(i*n+j)]=best_i;
                }
            }
            __syncthreads();
        }
    }
}

__global__
void checkFlag(int b, int n, int* idx1, int* idx2, int* flag1, int* flag2, int offset) {
    int idx = offset + threadIdx.x;
    int bb =  blockIdx.x * (blockDim.x * 2);

    int result1_idx = idx1[bb + idx];
    int temp_idx = idx2[bb + result1_idx];

    if (temp_idx == idx && flag1[bb + idx] == 0) {
        ++flag1[bb + idx];
        ++flag2[bb + result1_idx];
    }
}

void NmDistanceKernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,
                float * result,int * result_i, float * result2,int * result2_i){
    int idx_size = sizeof(int) * b*n;
    int* flag1; 
    int* flag2;

    cudaMallocManaged(&flag1, idx_size);
    cudaMallocManaged(&flag2, idx_size);

    cudaMemset(&flag1, 0, idx_size);
    cudaMemset(&flag2, 0, idx_size);

    for (int i=0; i<20; i++) {		
        NmDistanceKernel<<<dim3(32,16,1),512>>>(b,n,xyz,m,xyz2,result,result_i, flag1, flag2);
        NmDistanceKernel<<<dim3(32,16,1),512>>>(b,m,xyz2,n,xyz,result2,result2_i, flag2, flag1);

        checkFlag<<<b, 1024>>>(b, n, result_i, result2_i, flag1, flag2, 0);
        checkFlag<<<b, 1024>>>(b, n, result_i, result2_i, flag1, flag2, 1024);
    }
    
    cudaFree(flag1);
    cudaFree(flag2);
}

__global__ void NmDistanceGradKernel(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
            float x1=xyz1[(i*n+j)*3+0];
            float y1=xyz1[(i*n+j)*3+1];
            float z1=xyz1[(i*n+j)*3+2];
            int j2=idx1[i*n+j];
            float x2=xyz2[(i*m+j2)*3+0];
            float y2=xyz2[(i*m+j2)*3+1];
            float z2=xyz2[(i*m+j2)*3+2];
            float g=grad_dist1[i*n+j]*2;
            atomicAdd(&(grad_xyz1[(i*n+j)*3+0]),g*(x1-x2));
            atomicAdd(&(grad_xyz1[(i*n+j)*3+1]),g*(y1-y2));
            atomicAdd(&(grad_xyz1[(i*n+j)*3+2]),g*(z1-z2));
            atomicAdd(&(grad_xyz2[(i*m+j2)*3+0]),-(g*(x1-x2)));
            atomicAdd(&(grad_xyz2[(i*m+j2)*3+1]),-(g*(y1-y2)));
            atomicAdd(&(grad_xyz2[(i*m+j2)*3+2]),-(g*(z1-z2)));
        }
    }
}
void NmDistanceGradKernelLauncher(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2){
    cudaMemset(grad_xyz1,0,b*n*3*4);
    cudaMemset(grad_xyz2,0,b*m*3*4);
    NmDistanceGradKernel<<<dim3(1,16,1),256>>>(b,n,xyz1,m,xyz2,grad_dist1,idx1,grad_xyz1,grad_xyz2);
    NmDistanceGradKernel<<<dim3(1,16,1),256>>>(b,m,xyz2,n,xyz1,grad_dist2,idx2,grad_xyz2,grad_xyz1);
}

#endif
