#include<iostream>
#include<time.h>
#define SIZE 100000
using namespace std;

__global__ void addVect(int *vect1 ,int *vect2 , int *resultVect){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    //printf("Thread id == %d || Block Id == %d\n",threadIdx.x,blockDim.x);
    resultVect[i] = vect1[i] + vect2[i];
}

int main(){
    int *d_inVect1,*d_inVect2,*d_outResultVector;
    int vect1[SIZE],vect2[SIZE];
    int resultVect[SIZE];
    cudaEvent_t gpu_start,gpu_stop;
    float gpu_elapsed_time;
                                
    // Initializing both the vectors
    for(int i = 0 ; i < SIZE ; i++){
        vect1[i] = i;
        vect2[i] = i;
    }
    // Parallel code

    // Allocate memory on GPU for 3 vectors
    cudaMalloc((void**)&d_inVect1,SIZE*(sizeof(int)));
    cudaMalloc((void**)&d_inVect2,SIZE*(sizeof(int)));
    cudaMalloc((void**)&d_outResultVector,SIZE*(sizeof(int)));

    // CPY the vector contents
    cudaMemcpy(d_inVect1,vect1,SIZE*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_inVect2,vect2,SIZE*sizeof(int),cudaMemcpyHostToDevice);

    // Start record for gpu_start
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start,0);

    int blk = SIZE/1024;
    // Call the kernel
    addVect<<<blk+1,1024>>>(d_inVect1,d_inVect2,d_outResultVector);
    cudaDeviceSynchronize();
    cudaEventRecord(gpu_stop,0);
    // Copy gpu mem to cpu mem
    cudaMemcpy(resultVect,d_outResultVector,SIZE*sizeof(int),cudaMemcpyDeviceToHost);
        
    
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time,gpu_start,gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    cout<<"The time taken by GPU is :"<<gpu_elapsed_time<<endl;
    
    // verify that the GPU did the work we requested
    bool success = true;
    int total=0;
    cout<<"\nChecking "<<SIZE<<" values in the array.\n";
    for (int i=0; i<SIZE; i++) {
        if ((vect1[i] + vect2[i]) != resultVect[i]) {
            printf( "Error:  %d + %d != %d\n", vect1[i], vect2[i], resultVect[i] );
            success = false;
        }
        total += 1;
    }
    if (success)  cout<<"We did it "<<total<<"  values correct!\n";

    // Sequential code
    clock_t startTime = clock();
    int resultVect2[SIZE];
    for(int i = 0 ; i < SIZE ; i++){
        resultVect2[i] = vect1[i] * vect2[i];
    }
    clock_t endTime = clock();
    printf("\nTime for sequential: %.4f",((float)(endTime-startTime)/CLOCKS_PER_SEC)*1000);
    printf("\n Speedup= %.4f",(((float)(endTime-startTime)/CLOCKS_PER_SEC)*1000)/gpu_elapsed_time);
    return 0;
}
