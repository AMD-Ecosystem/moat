#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
__global__ void fill(float* p, size_t n, float v){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) p[i]=v; }
__global__ void peek(const float* p, size_t n, float* out){ out[0]=p[0]; out[1]=p[n/2]; out[2]=p[n-1]; }
int main(int argc,char**argv){
  size_t bytes=atoll(argv[1]); int delay=atoi(argv[2]); size_t n=bytes/4;
  hipStream_t nb; hipStreamCreateWithFlags(&nb,hipStreamNonBlocking);
  float *src,*dst,*out; hipMalloc(&src,bytes); hipMalloc(&dst,bytes); hipMallocManaged(&out,16);
  int stale=0;
  for(int r=0;r<10;r++){
    fill<<<(n+255)/256,256,0,nb>>>(src,n,1.0f+r); fill<<<(n+255)/256,256,0,nb>>>(dst,n,0.f); hipDeviceSynchronize();
    auto t0=std::chrono::steady_clock::now();
    hipMemcpyDtoD((hipDeviceptr_t)dst,(hipDeviceptr_t)src,bytes);
    while(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()<delay);
    peek<<<1,1,0,nb>>>(dst,n,out); hipStreamSynchronize(nb);
    if(out[2]!=1.0f+r) stale++;
    hipDeviceSynchronize();
  }
  hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b); hipEventRecord(a,0); hipMemcpyDtoD((hipDeviceptr_t)dst,(hipDeviceptr_t)src,bytes); hipEventRecord(b,0); hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms,a,b);
  printf("bytes=%zu delay=%dus stale_last=%d/10 copy=%.3f ms\n",bytes,delay,stale,ms);
}
