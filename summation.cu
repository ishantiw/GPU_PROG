/*
Author: Ishan Tiwari & Shashi Mohan Reddy Ravula
Description: This is the main file where all the memory allocation for CPU and GPU is done. 
Also it invokes the kernel and do the reduction after getting the output from the kernel.
*/
#include "utils.h"
#include <stdlib.h>

struct results
{
	float sum;
};

#include "summation_kernel.cu"

// CPU implementation

//Sum in decreasing order of indices
float log2_series(int n)
{
	int i; float sum1 = 0;
	for (i = n - 1; i >= 0; i--)
	{
		if (i % 2 == 0)
		{
			sum1 += 1 / float(i + 1);
		}
		else
		{
			sum1 += -1 / float(i + 1);
		}
	}
	return sum1;
}

//Sum in increasing order of indices
float log2_series(int n)
{
	float sum = 0;

	for (int i = 0; i<n - 1; i++)
	{
		if (i % 2 == 0)
		{
			sum += 1 / float(i + 1);
		}
		else if (i % 2 == 1)
		{
			sum += -1 / float(i + 1);
		}
	}
	return sum;
}

int main(int argc, char ** argv)
{
	int data_size = 1024 * 1024 * 128;

	// Run CPU version
	double start_time = getclock();
	float log2 = log2_series(data_size);
	double end_time = getclock();

	printf("CPU result: %f\n", log2);
	printf(" log(2)=%f\n", log(2.0));
	printf(" time=%fs\n", end_time - start_time);

	// Parameter definition
	int threads_per_block = 4 * 32;
	int blocks_in_grid = 8;

	int num_threads = threads_per_block * blocks_in_grid;

	// Timer initialization
	cudaEvent_t start, stop;
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));


	// Allocating output data on CPU
	float final_cpu;
	float * partial_cpu = (float*)malloc(blocks_in_grid*sizeof(float));
	float * data_out_cpu = (float*)malloc(data_size*sizeof(float));


	// Allocating output data on GPU
	float * data_gpu, *partial_gpu, *block_partial, *final_gpu;
	cudaMalloc((void**)&final_gpu, sizeof(float));
	cudaMalloc((void**)&partial_gpu, blocks_in_grid*sizeof(float));
	cudaMalloc((void**)&data_gpu, num_threads*sizeof(float));

	// Start timer
	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	// Execute kernel
	//Strategy2:Version1: All the threads adding elements parted with a distance of nthreads number of elements.
	summation_kernel1 << <blocks_in_grid, threads_per_block >> >(data_size, data_gpu);

	//Strategy1:Version2 Every thread is adding nthreads number of elements.
	//summation_kernel1_version2<<<blocks_in_grid, threads_per_block>>>(data_size,data_gpu);

	//Getting the partial results by returning total sum from each block and also allocating size for the shared memory used for partial results
	summation_kernel2 << <blocks_in_grid, threads_per_block, threads_per_block * sizeof(float) >> >(data_size, data_gpu, partial_gpu);

	//This kernel will take the partial result from "summation_kernel2" in "partial_gpu" variable
	summation_kernel3 << <1, blocks_in_grid, sizeof(float) >> >(partial_gpu, final_gpu);

	// Stop timer
	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	// Get results back
	//cudaMemcpy(data_out_cpu,data_gpu,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(partial_cpu, partial_gpu, blocks_in_grid*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&final_cpu, final_gpu, sizeof(float), cudaMemcpyDeviceToHost);

	// Finish reduction
	// TODO
	float sum = 0.0;
	for (int i = 0; i<blocks_in_grid; i++)
	{
		printf("value of %f", partial_cpu[i]);
		sum += partial_cpu[i];
	}

	// Cleanup
	// TODO
	free(data_out_cpu);
	free(partial_cpu);
	cudaFree(data_gpu);
	cudaFree(partial_gpu);
	cudaFree(final_gpu);

	printf("GPU results:\n");
	printf(" Sum result by value from per block: %f\n", sum);
	printf("Final Sum: %f\n", final_cpu);

	float elapsedTime;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms

	double total_time = elapsedTime / 1000.;	// s
	double time_per_iter = total_time / (double)data_size;
	double bandwidth = sizeof(float) / time_per_iter; // B/s

	printf(" Total time: %g s,\n Per iteration: %g ns\n Throughput: %g GB/s\n",
		total_time,
		time_per_iter * 1.e9,
		bandwidth / 1.e9);

	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(stop));
	getchar();
	return 0;
}


